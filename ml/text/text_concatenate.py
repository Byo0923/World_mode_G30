import torch
import torch.nn as nn
import random

class TEXT_Cconcatenate(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.linear_3 = nn.Linear(256, output_size)

    def forward(self, x):
        out_1 = self.linear_1(x)
        out_2 = self.linear_2(out_1)
        out_3 = self.linear_3(out_2)
        return out_3