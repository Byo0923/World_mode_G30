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
from util import visualize_result

# 設定と定義
torch.autograd.set_detect_anomaly(True)
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

base_dir = f"valid_result/{current_time}"
os.makedirs(base_dir, exist_ok=True)

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

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデル
generator = StockPriceEstimator(args).to(device)

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

if not args.use_discriminator:
    generator.load_state_dict(torch.load('result/2024-02-16-22-35/generator_40900.pth'))

    generator.eval()
    with torch.no_grad():
        pred_list = []
        target_list = []
        price_list = []
        for i, (price, text, target) in enumerate(valid_dataloader):
            price = price.to(device)
            text = text.to(device)
            target = target.to(device)

            price_list.append(price.cpu().detach().numpy())
            # Generatorに入力
            pred = generator(price, text)
            # 予測値をリストに追加
            pred_list.append(pred.cpu().detach().numpy())
            # 正解値をリストに追加
            target_list.append(target.cpu().detach().numpy())
        
        pred = np.concatenate(pred_list, axis=0)
        target = np.concatenate(target_list, axis=0)
        price = np.concatenate(price_list, axis=0)

        for num in range(len(pred)):
            fig = visualize_result(price[num], pred[num], target[num])
            os.makedirs(f"{base_dir}/fig/each_fig/", exist_ok=True)
            fig.savefig(f"{base_dir}/fig/each_fig/{num}.png")
            plt.close()

        # 全区間のプロット
        pred, target = pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
        fig_all, ax = plt.subplots(figsize=(20, 10))
        ax.plot(pred[:, 0], label="pred")
        ax.fill_between(range(len(pred[:, 0])), pred[:, 1], pred[:, 2], color="blue", alpha=0.3)
        ax.plot(target[:, 0], label="target")
        ax.fill_between(range(len(target[:, 0])), target[:, 1], target[:, 2], color="blue", alpha=0.3)
        plt.legend()

        fig_all.savefig(f"{base_dir}/fig/all_fig.png")
