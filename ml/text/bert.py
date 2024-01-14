import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

from mmbt.data.helpers import get_data_loaders
from mmbt.models import get_model
from mmbt.utils.logger import create_logger
from mmbt.utils.utils import *


# 事前学習済みモデルのロード
from transformers import BertModel
bert = BertModel.from_pretrained('bert-base-uncased')

# モデルの定義
# 事前学習済みモデルの後段に線形関数を追加し、この出力で感情分析をする
import torch.nn as nn

class BERTSentiment(nn.Module):
    def __init__(self,
                 bert,
                 output_dim):
        
        super().__init__()
        
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.out = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        #text = [batch size, sent len]

        #embedded = [batch size, emb dim]
        embedded = self.bert(text)[1]
        print("embedded" , embedded.size() )

        #output = [batch size, out dim]
        output = self.out(embedded)
        
        return output
# モデルインスタンスの生成
# 出力は感情分析なので2
OUTPUT_DIM = 2

model = BERTSentiment(bert, OUTPUT_DIM).to(device)
model.eval()

input = encoding.input_ids.to(device)
predictions = model(input)
print(predictions)
