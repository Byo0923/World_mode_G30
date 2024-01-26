import os
import logging
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from data.data_loder import FinancialDataset
import argparse
from util import rmse_loss, plot_target
import matplotlib.pyplot as plt
from models import StockPriceEstimator, Discriminator

# tensorboard --logdir=result

# 設定と定義
torch.autograd.set_detect_anomaly(True)
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

# 結果を保存するディレクトリを作成
result_dir = f"result/{current_time}"
os.makedirs(result_dir, exist_ok=True)

# logを保存するディレクトリを作成
log_dir = f"{result_dir}/log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "logfile.log")

# modelを保存するディレクトリを作成
model_dir = f"{result_dir}/model"
os.makedirs(model_dir, exist_ok=True)

# 検証結果を保存するディレクトリを作成
valid_dir = f"{result_dir}/valid"
os.makedirs(valid_dir, exist_ok=True)
train_dir = f"{result_dir}/train"
os.makedirs(train_dir, exist_ok=True)

# Figureを保存するディレクトリを作成
fig_dir = f"{result_dir}/fig"
os.makedirs(fig_dir, exist_ok=True)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ])

# TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# ハイパーパラメータ
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_epoch_num", type=int, default=1000)
parser.add_argument("--one_report_dim", type=int, default=64)
parser.add_argument("--price_feature_dim", type=int, default=11)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--price_output_size", type=int, default=256)
parser.add_argument("--nhead", type=int, default=8)
parser.add_argument("--num_encoder_layers", type=int, default=3)
parser.add_argument("--dim_feedforward", type=int, default=2048)
parser.add_argument("--max_grad_norm", type=int, default=1.0)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--use_discriminator", type=bool, default=False)
args = parser.parse_args()  

# パラメータのロギング
for key, value in vars(args).items():
    logging.info(f"{key} : {value}")
print(args.one_report_dim)
# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデル
generator = StockPriceEstimator(args).to(device)
discriminator = Discriminator().to(device)

# オプティマイザー
optimizer_g = optim.Adam(generator.parameters(), lr=args.learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=args.learning_rate)

# データの読み込み
data = pd.read_pickle('data/sentiment_stock_data.pkl')
train_data = data[data.index < '2022-01-03']
valid_data = data[data.index >= '2022-01-03']
# '2018-01-01' < data.index < '2019-12-31'のデータをvalid_train_dataとして使用
valid_train_data = data[(data.index >= '2018-01-01') & (data.index < '2019-12-31')]
train_dataset = FinancialDataset(train_data, week_len=4, target_len=80)
valid_dataset = FinancialDataset(valid_data, week_len=4, target_len=80)
valid_train_dataset = FinancialDataset(valid_train_data, week_len=4, target_len=80)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=valid_dataset.collate_fn)
valid_train_dataloader = DataLoader(valid_train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=valid_train_dataset.collate_fn)

# 訓練ループ
for epoch in tqdm(range(args.learning_epoch_num)):
    losses_train_g = []
    losses_train_d = []
    losses_valid_g = []
    losses_valid_d = []

    rmses_valid_g = []

    generator.train()

    # Discriminatorを使う場合
    if args.use_discriminator:
        # 損失関数の定義
        criterion = nn.BCELoss()
        error_func = rmse_loss()
        for j, data in enumerate(train_dataloader):
            price, text, target = data[0].to(device), data[1].to(device), data[2].to(device)

            ############### Generatorの学習 #######################
            # 勾配をゼロにリセット
            optimizer_g.zero_grad()
            # Generatorに入力
            pred = generator(price, text)
            # Discriminatorを欺く方向に学習
            d_fake = discriminator(torch.permute(pred, (0,2,1)))
            # 損失の計算
            loss_train_g = criterion(d_fake, torch.ones_like(d_fake))
            # 誤差逆伝播
            loss_train_g.backward(retain_graph=True)
            # 勾配クリッピングの適用
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_grad_norm)
            # 損失をリストに追加
            losses_train_g.append(loss_train_g.cpu().detach().numpy())
            # パラメータの更新
            optimizer_g.step()

            ############### Discriminatorの学習 #######################
            # Discriminatorの学習
            # 勾配をゼロにリセット  
            optimizer_d.zero_grad()
            # 本物のデータを判定
            d_real = discriminator(torch.permute(target, (0,2,1)))
            # 偽物のデータを判定．
            d_fake = discriminator(torch.permute(pred.detach(), (0,2,1)))
            # 損失の計算
            d_real_loss = criterion(d_real, torch.ones_like(d_real) * 0.9)
            d_fake_loss = criterion(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2
            # 誤差逆伝播/
            d_loss.backward()
            # loss をリストに追加
            losses_train_d.append(d_loss.cpu().detach().numpy())
            # 勾配クリッピングの適用
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
            # パラメータの更新
            optimizer_d.step()

        # モデルの評価
        with torch.no_grad():
            pred_list = []
            target_list = []
            for j, x in enumerate(valid_dataloader):
                price, text, target = x[0].to(device), x[1].to(device), x[2].to(device)
                # Generatorに入力
                pred = generator(price, text)
                # 予測値をリストに追加
                pred_list.append(pred.cpu().detach().numpy())
                # 正解値をリストに追加
                target_list.append(target.cpu().detach().numpy())
                # RMSEの計算
                rmse_valid_g = error_func(pred, target)
                # 損失をリストに追加
                rmses_valid_g.append(rmse_valid_g.cpu().detach().numpy())

        # エポックの終わりにログを記録
        avg_loss_train_g = np.mean(losses_train_g)
        avg_loss_valid_g = np.mean(losses_valid_g)
        avg_loss_train_d = np.mean(losses_train_d)
        avg_rmse_valid_g = np.mean(rmses_valid_g)
        
        writer.add_scalar('Training loss(Generator)', avg_loss_train_g, epoch)
        writer.add_scalar('Validation loss(Generator)', avg_loss_valid_g, epoch)
        writer.add_scalar('Training loss(Discriminator)', avg_loss_train_d, epoch)
        writer.add_scalar('Validation RMSE(Generator)', avg_rmse_valid_g, epoch)

        # 可視化
        pred, target = np.concatenate(pred_list, axis=0), np.concatenate(target_list, axis=0)
        # (b, f, dim) -> (b*f, dim)に変換
        pred, target = pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(pred[:, 0], label="pred")
        ax.fill_between(range(len(pred[:, 0])), pred[:, 1], pred[:, 2], color="blue", alpha=0.3)
        ax.plot(target[:, 0], label="target")
        ax.fill_between(range(len(target[:, 0])), target[:, 1], target[:, 2], color="blue", alpha=0.3)
        plt.legend()
        plt.title(f"Prediction and Target, RMSE: {avg_loss_valid_g}")

        writer.add_figure('Fig_valid', fig, epoch)

        logging.info(f"Epoch {epoch+1}/{args.learning_epoch_num} : Training loss = {avg_loss_train_g}, Validation loss = {avg_loss_valid_g}")

        # 10エポック毎にモデルの評価と保存
        if epoch % 10 == 0:
            # モデルの保存
            torch.save(generator.state_dict(), f"{model_dir}/generator_{epoch}.pth")
            # 予測値と正解値の可視化
            np.save(f"{valid_dir}/pred_{epoch}.npy", pred)
            np.save(f"{valid_dir}/target_{epoch}.npy", target)
            fig.savefig(f"{fig_dir}/valid_{epoch}.png")
            
            logging.info("Model and fig saved.")
    
    # Discriminatorを使わない場合
    else:
        # 損失関数の定義
        criterion = rmse_loss()
        # モデルの学習
        for j, data in enumerate(train_dataloader):
            price, text, target = data[0].to(device), data[1].to(device), data[2].to(device)

            ############### Generatorの教師あり学習 #######################
            # 勾配をゼロにリセット
            optimizer_g.zero_grad()
            # Generatorに入力
            pred = generator(price, text)
            # 損失の計算
            loss_g = criterion(pred, target)
            # 誤差逆伝播
            loss_g.backward(retain_graph=True)
            # 勾配クリッピングの適用
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_grad_norm)
            # 損失をリストに追加
            losses_train_g.append(loss_g.cpu().detach().numpy())
            # パラメータの更新
            optimizer_g.step()
        
        # モデルの評価
        generator.eval()
        with torch.no_grad():
            pred_list = []
            target_list = []
            for j, x in enumerate(valid_dataloader):
                price, text, target = x[0].to(device), x[1].to(device), x[2].to(device)
                # Generatorに入力
                pred = generator(price, text)
                # 予測値をリストに追加
                pred_list.append(pred.cpu().detach().numpy())
                # 正解値をリストに追加
                target_list.append(target.cpu().detach().numpy())
                # 損失の計算
                loss_valid_g = criterion(pred, target)
                # 損失をリストに追加
                losses_valid_g.append(loss_valid_g.cpu().detach().numpy())
            
        # エポックの終わりにログを記録
        avg_loss_train_g = np.mean(losses_train_g)
        avg_loss_valid_g = np.mean(losses_valid_g)
        writer.add_scalar('Training loss(Generator)', avg_loss_train_g, epoch)
        writer.add_scalar('Validation loss(Generator)', avg_loss_valid_g, epoch)

        # 可視化
        pred, target = np.concatenate(pred_list, axis=0), np.concatenate(target_list, axis=0)
        # (b, f, dim) -> (b*f, dim)に変換
        pred, target = pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(pred[:, 0], label="pred")
        ax.fill_between(range(len(pred[:, 0])), pred[:, 1], pred[:, 2], color="blue", alpha=0.3)
        ax.plot(target[:, 0], label="target")
        ax.fill_between(range(len(target[:, 0])), target[:, 1], target[:, 2], color="blue", alpha=0.3)
        plt.legend()
        plt.title(f"Prediction and Target, RMSE: {avg_loss_valid_g}")

        writer.add_figure('Fig_valid', fig, epoch)

        logging.info(f"Epoch {epoch+1}/{args.learning_epoch_num} : Training loss = {avg_loss_train_g}, Validation loss = {avg_loss_valid_g}")

        # 10エポック毎にモデルの評価と保存
        if epoch % 10 == 0:
            # モデルの保存
            torch.save(generator.state_dict(), f"{model_dir}/generator_{epoch}.pth")
            # 予測値と正解値の可視化
            np.save(f"{valid_dir}/pred_{epoch}.npy", pred)
            np.save(f"{valid_dir}/target_{epoch}.npy", target)
            fig.savefig(f"{fig_dir}/valid_{epoch}.png")
        
        # Trainデータでの評価
        generator.eval()
        with torch.no_grad():
            pred_list = []
            target_list = []
            rmses_train_g = []
            for j, x in enumerate(valid_train_dataloader):
                price, text, target = x[0].to(device), x[1].to(device), x[2].to(device)
                # Generatorに入力
                pred = generator(price, text)
                # 予測値をリストに追加
                pred_list.append(pred.cpu().detach().numpy())
                # 正解値をリストに追加
                target_list.append(target.cpu().detach().numpy())
                # 損失の計算
                rmse = criterion(pred, target)
                # 損失をリストに追加
                rmses_train_g.append(rmse.cpu().detach().numpy())
            
        # エポックの終わりにログを記録
        avg_rmse_train_g = np.mean(rmses_train_g)

        # 可視化
        pred, target = np.concatenate(pred_list, axis=0), np.concatenate(target_list, axis=0)
        # (b, f, dim) -> (b*f, dim)に変換
        pred, target = pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(pred[:, 0], label="pred")
        ax.fill_between(range(len(pred[:, 0])), pred[:, 1], pred[:, 2], color="blue", alpha=0.3)
        ax.plot(target[:, 0], label="target")
        ax.fill_between(range(len(target[:, 0])), target[:, 1], target[:, 2], color="blue", alpha=0.3)
        plt.legend()
        plt.title(f"Prediction and Target, RMSE: {avg_rmse_train_g}")

        writer.add_figure('Fig_train', fig, epoch)

        # 10エポック毎にモデルの評価と保存
        if epoch % 10 == 0:
            # 予測値と正解値の可視化
            np.save(f"{train_dir}/pred_{epoch}.npy", pred)
            np.save(f"{train_dir}/target_{epoch}.npy", target)
            fig.savefig(f"{fig_dir}/train_{epoch}.png")