import torch
import torch.nn as nn
import random
        
# target [batch, signal_length, signal_channel]=[4, 20, 3]が本物か偽物かを判定するDiscriminator
class Discriminator(nn.Module):
    def __init__(self, sig=True):
        super().__init__()
        self.sig = sig
        self.conv1 = nn.Conv1d(3, 256, kernel_size=3, stride=1, padding='same')  # 入力チャンネル数を3に変更
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding='same')
        self.linear1 = nn.Linear(2560, 1024)
        #self.batch1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512)
        #self.batch2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 256, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(0.01),
        )
        self.fc = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        latent = self.conv_layers(x)
        latent = latent.reshape(latent.shape[0], latent.shape[1]*latent.shape[2])
        out = self.fc(latent)
        return out
        # if self.sig:
        #     return self.sigmoid(self.linear3(out_2))
        # else:
        #     return self.linear3(out_2)
    




