import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 最初の次元がバッチなら，dim=1方向に結合して可視化
def plot_target(target):
    sma_line, upper_line, lower_line = target.T
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(sma_line)
    ax.fill_between(range(len(sma_line)), sma_line, upper_line, color="blue", alpha=0.3)
    ax.fill_between(range(len(sma_line)), sma_line, lower_line, color="red", alpha=0.3)
    ax.legend(["SMA", "Upper", "Lower"])

    return fig

class rmse_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return torch.sqrt(torch.mean((input - target) ** 2))

# 可視化用の関数
def visualize_result(input, pred, target):
    input = np.reshape(input, (-1, 11))
    input_size = input.shape[0] # 8, 4, 31, 11
    x_input = np.arange(input_size)
    x_pred = np.arange(input_size, input_size + pred.shape[0])
    input_sma = input[:, 6]
    input_upper = input[:, 9]
    input_lower = input[:, 10] 

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(x_input, input_sma, label="input")
    ax.fill_between(x_input, input_upper, input_lower, color="blue", alpha=0.3)
    
    ax.plot(x_pred, pred[:, 0], label="pred")
    ax.fill_between(x_pred, pred[:, 1], pred[:, 2], color="blue", alpha=0.3)

    ax.plot(x_pred, target[:, 0], label="target")
    ax.fill_between(x_pred, target[:, 1], target[:, 2])
    ax
    ax.legend()
    return fig