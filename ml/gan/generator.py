import torch
import torch.nn as nn
import random

class Generator(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.linear_1 = nn.Linear(256, 256)
        self.linear_2 = nn.Linear(256, 60)
        self.dropout = nn.Dropout(0.2)
        self.batch_size = batch_size
    def forward(self, x):
        out_1 = self.linear_1(x)
        out_2 = self.linear_2(out_1)
        out_3 = out_2.view(self.batch_size, 20, 3)
        return out_3