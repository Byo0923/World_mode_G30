import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

# モデルの定義
import torch.nn as nn

class BERT_Report(nn.Module):
    def __init__(self,
                 bert,
                 output_dim,
                 tokenizer):
        
        super().__init__()
        
        self.bert = bert
        self.linear_1 = nn.Linear(768, 256)
        self.linear_2 = nn.Linear(256, output_dim)
        self.tokenizer = tokenizer
        
    def forward(self, text):
        output_1 = self.bert(text)[1].squeeze(0)
        output_2 = self.linear_1(output_1)  # バッチ次元を除去)
        output_3 = self.linear_2(output_2)
        return output_3
    
    def encoding(self, text):
        encoding = self.tokenizer(
            text, 
            max_length =500, 
            padding ="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        return encoding

# # モデルインスタンスの生成
# # 出力は感情分析なので2
# OUTPUT_DIM = 2

# model = BERTSentiment(bert, OUTPUT_DIM).to(device)
# model.eval()

# input = encoding.input_ids.to(device)
# predictions = model(input)
# print(predictions)
