import torch
import torch.nn as nn
import random

# Transformerエンコーダモデルの定義
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, src):
        src = self.linear_in(src)
        output = self.transformer_encoder(src)
        output = self.linear_out(output[-1])  # 最後のタイムステップの出力のみを使用
        return output

# モデルのパラメータを設定
input_size = 10  # 時系列データの特徴量の数
hidden_size = 512  # Transformer内部の隠れ状態のサイズ
output_size = 128  # 出力する潜在変数の次元数
nhead = 8  # マルチヘッドアテンションのヘッド数
num_encoder_layers = 3  # エンコーダレイヤーの数
dim_feedforward = 2048  # フィードフォワードネットワークの次元

# モデルのインスタンス化
transformer_model = TransformerModel(input_size, hidden_size, output_size, nhead, num_encoder_layers, dim_feedforward)

# ダミーの時系列データを生成（長さは15から20のランダム）
sequence_length = random.randint(15, 20)
dummy_time_series = torch.randn(sequence_length, 1, input_size)

# Transformerモデルに入力
# バッチサイズが1のため、[シーケンス長, バッチサイズ, 特徴量数]に変形
dummy_time_series = dummy_time_series.permute(1, 0, 2)
latent_vector = transformer_model(dummy_time_series)
latent_vector.shape  # 潜在ベクトルの形状を確認