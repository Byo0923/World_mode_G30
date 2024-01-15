import torch
import torch.nn as nn
import random
import numpy as np
import math

# 位置エンコーディングの関数を定義
def positional_encoding(seq_len, d_model):
    """ 位置エンコーディングを生成する関数 """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return torch.tensor(pe, dtype=torch.float)

# Transformerエンコーダモデルの定義（位置エンコーディングとアテンションメカニズム付き）
class TransformerModelWithPositionalEncoding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead, num_encoder_layers, dim_feedforward, max_seq_len=200):
        super(TransformerModelWithPositionalEncoding, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.pos_encoder = positional_encoding(max_seq_len, hidden_size).to("cuda")
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, src):
        # 入力データに位置エンコーディングを加算
        src = self.linear_in(src) + self.pos_encoder[:src.size(1), :]
        output = self.transformer_encoder(src)

        # アテンションの重みを計算
        attn_weights = torch.softmax(self.attention_weights(output).squeeze(-1), dim=1)

        # 重み付き平均ベクトルを計算
        weighted_output = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)
        output = self.linear_out(weighted_output).squeeze(0)
        return output, attn_weights

