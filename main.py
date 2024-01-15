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

batch_size = 32
logging.info("batch_size = " + str(batch_size) )
learning_epoch_num = 100
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
bert_model_optimizer = torch.optim.AdamW(bert_model.parameters(), lr=0.0001)
# bert_model.eval()


from ml.price.transfomer import TransformerModelWithPositionalEncoding
# モデルのパラメータを設定
price_feature_dim = 11
hidden_size = 512
price_output_size = 128
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


from ml.gan.generator import Generator
generator   = Generator(batch_size).to(device)
optimizer_generator = torch.optim.AdamW(bert_model.parameters(), lr=0.0001)

from ml.gan.discriminator import Discriminator
discriminator   = Discriminator().to(device)
optimizer_discriminator = torch.optim.AdamW(bert_model.parameters(), lr=0.0001)


# RMSE損失関数を定義
def rmse_loss(input, target):
    return torch.sqrt(torch.mean((input - target) ** 2))

criterion = nn.BCELoss()


# データ読み込み
data = pd.read_pickle('sentiment_tech_data.pkl')
# print("L129 data" , data)

# FinancialDataset クラスのインポート
# from your_module import FinancialDataset

# データセットのインスタンスを作成
# ここでは `data` を事前に定義または読み込んでいると仮定
from data.data_loder import FinancialDataset
financial_dataset = FinancialDataset(data, week_len=4, target_len=80)

# # DataLoader を使用してデータセットをロード
financial_dataloader = DataLoader(financial_dataset, batch_size=batch_size, shuffle=True, collate_fn=FinancialDataset.collate_fn)
for data in financial_dataloader:
    week = data[0].shape[1]
    num = data[0].shape[2]
    prices = data[0].reshape(batch_size, week*num ,price_feature_dim).to(device)
    print("prices" , prices.shape)
    latent_vector, attention_weights = transformer_model_with_attention(prices)
    print("latent_vector" , latent_vector.shape)

    text = data[1].to(device)
    region_num = text.shape[2]
    print("region" , region_num)
    all_reports =[]
    for w in range(week):
        tex_latent_regions =[]
        for r in range(region_num):
            text_region = text[0,w, r ].squeeze(1)
            # text_region = text[0,w, r ].reshape(850)
            print("text_region" , text_region.shape)
            tex_latent = bert_model(text_region)
            print("tex_latent" , tex_latent.shape)
            tex_latent_regions.append(tex_latent)
        week_reports = torch.cat( [tex_latent_regions[0],tex_latent_regions[1],tex_latent_regions[2],tex_latent_regions[3],tex_latent_regions[4]]  , dim=0)
        print("week_reports" , week_reports.shape)
        all_reports.append(week_reports)

    # text_w2 = 
    # text_w3 = 
    # text_w4 = 
    # print("text" , text.shape)
    # tex_latent_w1 = bert_model(input)
    # all_reports = torch.cat( [predictions,predictions,predictions,predictions,predictions]  , dim=0)
    # print(all_reports.shape)  # 潜在ベクトルの形状を確認
    # print(data[0])
    # print(data[1])

for i in tqdm(range(learning_epoch_num)):
    
    data_length = len(financial_dataloader.dataset)
    for data in financial_dataloader:
        if  data > batch_size - 1 :
            print(i)
            print("-batch_latent_vector-" , batch_latent_vector.shape)
            batch_latent_vector = None

        latent_vector = latent_vector.reshape(1,1,price_output_size)
        if batch_latent_vector ==  None:
            batch_latent_vector = latent_vector
        else:
            batch_latent_vector = torch.cat( [batch_latent_vector, latent_vector] , dim=0)


    # 勾配をゼロにリセット
    bert_model_optimizer.zero_grad()
    transformer_optimizer.zero_grad()
    text_cconcatenate_optimizer.zero_grad()
    optimizer_generator.zero_grad()

text_2023_01_us = """
The US stock market sentiment in the first week of January 2023 showed a cautiously optimistic outlook, influenced by various factors including sector performances, rate hike expectations, and company-specific news.
1. **January Indicator Trifecta and Sector Performance**: The stock market experienced what is known as the "January Indicator Trifecta." This refers to a Santa Claus rally, positive first five days of January, and a positive January Barometer. The occurrence of all three indicators historically suggests a favorable market in the following 11 months. In terms of sector performance, Consumer Discretionary and Communication Services led the gains. The Nasdaq Composite showed a strong performance, especially in technology stocks, while small-cap stocks indicated by the S&P 600 Small Cap index also rose significantly.
2. **Federal Reserve and Rate Hike Odds**: The market was anticipating a 25 basis point rate increase at the February Federal Reserve meeting. This expectation was reflected in the pricing of fed fund futures. Treasury yields saw some weakness, with the 10-year Treasury yield dropping to 3.51%, which was below the October peak of 4.25%.
3. **Corporate Earnings and Stock Performance**: About one-third of S&P 500 companies reported a 5% decline in Q4 profits, compared to an expected 3.2% decline. Despite this, there were sectors like Energy, Industrials, and Consumer Discretionary that saw significant earnings growth. Notably, the worst-performing stocks of 2022 saw an average increase of 20.1% in early 2023, suggesting a short-term reversion of oversold stocks rather than a fundamental shift in market leadership.
4. **Influence of Company-Specific News**: Individual companies also influenced market sentiment. For instance, Tesla's shares went up after announcing price cuts in China, while Bed Bath & Beyond's shares declined significantly due to bankruptcy considerations. Costco's stock gained after reporting positive December sales data.
5. **Overall Market Dynamics**: The first week of January 2023 closed higher for US stocks, spurred by a favorable jobs report and corporate news. The CBOE Volatility Index (VIX), often regarded as a fear gauge, decreased by 11% in January, indicating a decrease in market volatility.
In summary, the first week of January 2023 in the US stock market was marked by a mix of optimism driven by sector performances and cautious sentiment due to economic indicators and corporate earnings. While there was a positive outlook based on the January indicators, the market remained sensitive to rate hikes and individual corporate performances.
References: 
- StockCharts.com【6†source】
- Nasdaq【7†source】
- Yahoo Finance【8†source】【9†source】
"""

encoding_text = bert_model.encoding(text_2023_01_us)
input = encoding_text.input_ids.to(device)

# print("main L63", predictions.shape)

learning_epoch_num = 10
# 合計値(total)を設定


# print("main L78", all_reports_latent.shape)  # 潜在ベクトルの形状を確認



for i in tqdm(range(learning_epoch_num)):
    # 勾配計算を行わないコンテキスト内でモデルを評価
    # with torch.no_grad():

    all_reports_latent = text_cconcatenate(all_reports)

    # ダミーの時系列データを生成（長さは15から20のランダム）
    sequence_length = random.randint(15, 20)
    dummy_time_series = torch.randn(sequence_length, 1, price_feature_dim).to(device)




    # Transformerモデルに入力
    dummy_time_series = dummy_time_series.permute(1, 0, 2)
    # print("dummy_time_series:", dummy_time_series.shape)

    # print("latent_vector:", latent_vector.shape)
    # print("all_reports_latent:", all_reports_latent)

    input_generator = torch.cat( [all_reports_latent, latent_vector] , dim=0)
    input_generator = input_generator.reshape(1,1,input_generator.shape[0])
    # print("main L90", input_generator.shape)  # 潜在ベクトルの形状を確認
    predictions = generator(input_generator)

    # ランダムな教師データと推定データを生成
    target = torch.randn(batch_size, 20, 3).to(device)  # 教師データ

    # 損失の計算
    loss = rmse_loss(predictions, target)

    # 誤差逆伝播
    loss.backward(retain_graph=True)

    # パラメータの更新
    bert_model_optimizer.step()
    transformer_optimizer.step()
    text_cconcatenate_optimizer.step()
    optimizer_generator.step()

    # 損失の出力
    print(f"RMSE Loss: {loss.item()}")

    # torch.Size([1, 20, 3]) の形状を持つランダムなテンソルを作成
    random_tensor = torch.randn(1, 20, 3)



    print("pred L100", predictions.shape)  # 潜在ベクトルの形状を確認
    fake_flat = predictions.view(batch_size, 60, 1)
    print("fake_flat L100", fake_flat.shape)  # 潜在ベクトルの形状を確認
    latent_vector = latent_vector.view(batch_size, 128, 1)
    discriminator_fake = torch.cat( [latent_vector, fake_flat], dim=1 )
    print("discriminator_fake L100", discriminator_fake.shape)  # 潜在ベクトルの形状を確認

    discriminator(discriminator_fake)

    # # 出力の形状を確認
    # print("潜在ベクトルの形状:", latent_vector.shape)
    # print("アテンション重みの形状:", attention_weights.shape)