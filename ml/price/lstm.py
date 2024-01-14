import torch
import torch.nn as nn
import random

# LSTMモデルの定義
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTMを通して、最後の隠れ状態を取得
        _, (hidden, _) = self.lstm(x)
        # 隠れ状態を全結合層に通す
        output = self.fc(hidden[-1])
        return output

# モデルのパラメータを設定
input_size = 1  # 時系列データの特徴量の数（この場合は1）
hidden_size = 50  # LSTMの隠れ状態のサイズ
output_size = 128  # 出力する潜在変数の次元数

# モデルのインスタンス化
model = LSTMModel(input_size, hidden_size, output_size)

# ダミーの時系列データを生成（長さは15から20のランダム）
sequence_length = random.randint(15, 20)

# 修正：入力データの次元を調整
dummy_time_series = torch.randn(1, sequence_length, input_size)  # [バッチサイズ, シーケンス長, 特徴量数]

# モデルを再度実行して潜在ベクトルを取得
latent_vector = model(dummy_time_series)
latent_vector.shape  # 潜在ベクトルの形状を確認
