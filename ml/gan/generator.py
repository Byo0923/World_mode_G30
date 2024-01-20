import torch
import torch.nn as nn
import random

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(768, 256)
        self.linear_2 = nn.Linear(256, 60)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        b = x.size(0)
        out_1 = self.linear_1(x)
        out_2 = self.linear_2(out_1)
        out_3 = out_2.view(b, 20, 3)
        return out_3