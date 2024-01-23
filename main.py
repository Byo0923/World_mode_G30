# coding: utf-8
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

torch.autograd.set_detect_anomaly(True)

from tqdm import tqdm
import time

import logging
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

# modelディレクトリの作成
os.makedirs("model", exist_ok=True)

# GPU利用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 現在の日時を取得し、フォルダ名用にフォーマット
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_directory = f"logs/{current_time}"

# ログを保存するディレクトリを作成
os.makedirs(log_directory, exist_ok=True)

# ログファイルの設定
log_file = os.path.join(log_directory, "logfile.log")

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler(log_file),  # ファイルハンドラー
        logging.StreamHandler()         # ストリーム（コンソール）ハンドラ
    ])

# ログを記録するテストメッセージ
# logging.info("プログラムが起動しました。")
# logging.warning("これは警告メッセージです。")
# logging.error("これはエラーメッセージです。")

# デバイスがCUDAでない場合は警告を出す
if device.type != 'cuda':
    logging.error('CUDA is not available. Using CPU instead.')
else :
    logging.warning('CUDA is available. Using GPU instead.')

writer: SummaryWriter = SummaryWriter(
    log_dir=log_directory
)  

# tensorboard --logdir=logd

batch_size = 4
logging.info("batch_size = " + str(batch_size) )
learning_epoch_num = 1000
logging.info("learning_epoch_num = " + str(learning_epoch_num) )

from ml.text.bert import BERT_Report

# 事前学習済みモデルのロード
from transformers import BertModel
bert = BertModel.from_pretrained('bert-base-uncased')
# tokenizer インスタンスの生成
# 対象モデルは'bert-base-uncased'
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# モデルインスタンスの生成
one_report_dim = 64

bert_model = BERT_Report(bert, one_report_dim, tokenizer).to(device)
param_bert_num = 0

# 特定の層を凍結する
for name, param in bert_model.bert.named_parameters():
    # ここで凍結したい層の名前を指定します。例えば、'encoder.layer.11'とすると最後の層のみをファインチューニングします。
    if ('encoder.layer.11' not in name) and ('pooler' not in name):    
        param.requires_grad = False
    # print(param_bert_num)
    # print(name)
    # print(param.requires_grad)
    param_bert_num += 1

# # BERTモデルのパラメータをフリーズ
# for name, param in bert_model.bert.named_parameters(): # 198
#     # Bertのパラメータ名を出力
#     print(param_bert_num)
#     print(name)
#     #param.requires_grad = False
#     param_bert_num += 1

bert_model_optimizer = torch.optim.AdamW(bert_model.parameters(), lr=0.0001)
# bert_model.eval()

from ml.price.transfomer import TransformerModelWithPositionalEncoding
# モデルのパラメータを設定
price_feature_dim = 11
hidden_size = 512
price_output_size = 256
nhead = 8
num_encoder_layers = 3
dim_feedforward = 2048

# モデルのインスタンス化
transformer_model_with_attention = TransformerModelWithPositionalEncoding(price_feature_dim, hidden_size, price_output_size, nhead, num_encoder_layers, dim_feedforward).to(device)
transformer_optimizer = torch.optim.AdamW(transformer_model_with_attention.parameters(), lr=0.0001)


from ml.text.text_concatenate import TEXT_Cconcatenate
all_reports_dim = one_report_dim * 5
all_reports_out_dim = 128
text_cconcatenate = TEXT_Cconcatenate(all_reports_dim, all_reports_out_dim).to(device)
text_cconcatenate_optimizer = torch.optim.AdamW(text_cconcatenate.parameters(), lr=0.0001)

# Generator
from ml.gan.generator import Generator
generator   = Generator().to(device)
optimizer_generator = torch.optim.AdamW(generator.parameters(), lr=0.0001)

# Discriminator
from ml.gan.discriminator import *
discriminator = NewDiscriminator().to(device)#Discriminator().to(device)
optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=0.0001)


# RMSE損失関数を定義
def rmse_loss(input, target):
    return torch.sqrt(torch.mean((input - target) ** 2))

# 二値交差エントロピー損失関数を定義(Discriminator用)
criterion = nn.BCELoss()

# データ読み込み
data = pd.read_pickle('data/sentiment_stock_data.pkl')
train_data = data[data.index < '2022-01-03']
valid_data = data[data.index >= '2022-01-03']

# FinancialDataset クラスのインポート
from data.data_loder import FinancialDataset
train_dataset = FinancialDataset(train_data, week_len=4, target_len=80)
valid_dataset = FinancialDataset(valid_data, week_len=4, target_len=80)
print("train_dataset", len(train_dataset))
print("valid_dataset", len(valid_dataset))

# # DataLoader を使用してデータセットをロード
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn)

# エポックを通しての損失を格納するリスト
losses_generator = []
losses_discriminator = []

max_grad_norm = 1.0  # 勾配の最大ノルム

for epoch in tqdm(range(learning_epoch_num)):
    # Generatorの学習
    for j, data in enumerate(train_dataloader):
        #print("data_shape", data[0].shape)
        week = data[0].shape[1]
        num = data[0].shape[2]
        b = data[0].shape[0]
        prices = data[0].reshape(b, week*num ,price_feature_dim).to(device)
        #print("prices" , prices.shape)
        latent_vector, attention_weights = transformer_model_with_attention(prices)
        #print("latent_vector" , latent_vector.shape)

        text = data[1].to(device)
        region_num = text.shape[2]
        #print("region" , region_num)
        all_reports = None
        for w in range(week):
            tex_latent_regions =[]
            for r in range(region_num):
                text_region = text[:,w, r ].reshape(b, 500)
                # text_region = text[0,w, r ].reshape(850)
                #print("text_region" , text_region.shape)
                tex_latent = bert_model(text_region)
                #print("tex_latent" , tex_latent.shape)
                tex_latent_regions.append(tex_latent)
            week_reports = torch.cat( [tex_latent_regions[0],tex_latent_regions[1],tex_latent_regions[2],tex_latent_regions[3],tex_latent_regions[4]]  , dim=1)
            #print("week_reports" , week_reports.shape)

            # Text concat
            week_report = text_cconcatenate(week_reports)
            #print("Week_report", week_report.shape)
            if all_reports == None:
                all_reports = week_report
            else:
                all_reports = torch.cat([all_reports, week_report], dim=1)#all_reports.append(week_reports)
        #print("all_reports", all_reports.shape)

        # all_reportsとPriceのLatentをCat
        all_reports_latent = torch.cat( [all_reports, latent_vector] , dim=1)
        #print("all_reports_latent", all_reports_latent.shape)

        # Generatorに入力
        predictions = generator(all_reports_latent)
        #print("predictions", predictions.shape)

        # 勾配をゼロにリセット
        bert_model_optimizer.zero_grad()
        transformer_optimizer.zero_grad()
        text_cconcatenate_optimizer.zero_grad()
        optimizer_generator.zero_grad()
        
        #print("target_shape", data[2].shape)
        # 教師データと推定データを生成
        target = data[2].to(device)  # 教師データ
        #print("target", target.shape)

        # 損失の計算
        loss = rmse_loss(predictions, target)

        # 誤差逆伝播
        loss.backward(retain_graph=False)

        # 勾配クリッピングの適用
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(transformer_model_with_attention.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(text_cconcatenate.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)

        # TensorをNumPyに変換
        loss_numpy = loss.cpu().detach().numpy()
        # 損失をリストに追加
        losses_generator.append(loss_numpy)

        # パラメータの更新
        bert_model_optimizer.step()
        transformer_optimizer.step()
        text_cconcatenate_optimizer.step()
        optimizer_generator.step()

        # Discriminatorの学習
        # 勾配をゼロにリセット  
        optimizer_discriminator.zero_grad()

        # 本物のデータを判定
        d_real = discriminator(torch.permute(target, (0,2,1)))
        #print("d_real", d_real.shape)

        # 偽物のデータを判定．
        d_fake = discriminator(torch.permute(predictions.detach(), (0,2,1)))
        
        # 損失の計算
        d_real_loss = criterion(d_real, torch.ones_like(d_real))
        d_fake_loss = criterion(d_fake, torch.zeros_like(d_fake))
        d_loss = d_real_loss + d_fake_loss

        # 誤差逆伝播
        d_loss.backward()

        # loss をリストに追加
        losses_discriminator.append(d_loss.cpu().detach().numpy())

        # 勾配クリッピングの適用
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)

        # パラメータの更新
        optimizer_discriminator.step()

    # エポックの平均損失の計算
    avg_generator_loss = np.mean(losses_generator)
    avg_discriminator_loss = np.mean(losses_discriminator)
    # TensorBoardに記録
    writer.add_scalar('Training loss(Generator)', avg_generator_loss, epoch)
    writer.add_scalar('Training loss(Discriminator)', avg_discriminator_loss, epoch)

    logging.info(f"Epoch {epoch+1}/{learning_epoch_num} : Generator loss = {avg_generator_loss}, Discriminator loss = {avg_discriminator_loss}")
    #logging.info('Training loss(Generator) : epoch = '  + str(epoch) +  '  loss = ' + str(avg_generator_loss) ) 

    # modelの保存. model/ に保存される
    if epoch % 10 == 0:
        torch.save(bert_model.state_dict(), f"model/bert_model_{epoch}.pth")
        torch.save(transformer_model_with_attention.state_dict(), f"model/transformer_model_with_attention_{epoch}.pth")
        torch.save(text_cconcatenate.state_dict(), f"model/text_cconcatenate_{epoch}.pth")
        torch.save(generator.state_dict(), f"model/generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"model/discriminator_{epoch}.pth")




    # # torch.Size([1, 20, 3]) の形状を持つランダムなテンソルを作成
    # random_tensor = torch.randn(1, 20, 3)

    # print("pred L100", predictions.shape)  # 潜在ベクトルの形状を確認
    # fake_flat = predictions.view(batch_size, 60, 1)
    # print("fake_flat L100", fake_flat.shape)  # 潜在ベクトルの形状を確認
    # latent_vector = latent_vector.view(batch_size, 128, 1)
    # discriminator_fake = torch.cat( [latent_vector, fake_flat], dim=1 )
    # print("discriminator_fake L100", discriminator_fake.shape)  # 潜在ベクトルの形状を確認

    # discriminator(discriminator_fake)

    # # # 出力の形状を確認
    # # print("潜在ベクトルの形状:", latent_vector.shape)
    # # print("アテンション重みの形状:", attention_weights.shape)