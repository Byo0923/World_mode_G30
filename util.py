import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 最初の次元がバッチなら，dim=1方向に結合して可視化
def plot_target(target):
    if target.shape[0] > 1:
        target = torch.cat([t for t in target], dim=0)
    sma_line, upper_line, lower_line = target.T
    plt.plot(sma_line)
    plt.fill_between(range(len(sma_line)), sma_line, upper_line, color="blue", alpha=0.3)
    plt.fill_between(range(len(sma_line)), sma_line, lower_line, color="red", alpha=0.3)
    plt.legend(["SMA", "Upper", "Lower"])
    plt.show()