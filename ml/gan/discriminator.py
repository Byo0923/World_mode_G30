import torch
import torch.nn as nn
import random

class Discriminator(nn.Module):
    def __init__(self, sig=True):
        super().__init__()
        self.sig = sig
        self.conv1 = nn.Conv1d(188, 256, kernel_size=3, stride=1, padding='same')  # 入力チャンネル数を3に変更
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding='same')
        self.linear1 = nn.Linear(128, 220)
        self.batch1 = nn.BatchNorm1d(220)
        self.linear2 = nn.Linear(220, 220)
        self.batch2 = nn.BatchNorm1d(220)
        self.linear3 = nn.Linear(220, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        print(" conv3" , conv3.shape)
        flatten_x = conv3.reshape(conv3.shape[0], conv3.shape[1])
        print(" flatten_x" , flatten_x.shape)
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        if self.sig:
            return self.sigmoid(self.linear3(out_2))
        else:
            return self.linear3(out_2)
        
# target [batch, signal_length, signal_channel]=[4, 20, 3]が本物か偽物かを判定するDiscriminator
class NewDiscriminator(nn.Module):
    def __init__(self, sig=True):
        super().__init__()
        self.sig = sig
        self.conv1 = nn.Conv1d(3, 256, kernel_size=3, stride=1, padding='same')  # 入力チャンネル数を3に変更
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding='same')
        self.linear1 = nn.Linear(2560, 1024)
        self.batch1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512)
        self.batch2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        #print(" conv3" , conv3.shape)
        #dim = 1, 2を平滑化
        flatten_x = conv3.reshape(conv3.shape[0], conv3.shape[1]*conv3.shape[2])
        
        #flatten_x = conv3.reshape(conv3.shape[1], conv3.shape[1])
        #print(" flatten_x" , flatten_x.shape)
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        if self.sig:
            return self.sigmoid(self.linear3(out_2))
        else:
            return self.linear3(out_2)
    




