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