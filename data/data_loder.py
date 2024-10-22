import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import logging

class FinancialDataset(Dataset):
    def __init__(self, data, week_len=4, target_len=80):
        self.data = data
        self.week_len = week_len
        self.target_len = target_len
        self.scaling_param = 20000

        logging.info(f"Start preprocessing.")
        self.x_tech, self.x_text, self.t = self.preprocessing(data, week_len, target_len)
        logging.info(f"Finish preprocessing. Data size is {len(self.x_tech)}")

    def __len__(self):
        return len(self.x_tech)

    def __getitem__(self, idx):
        return self.x_tech[idx], self.x_text[idx], self.t[idx]

    def preprocessing(self, data, week_len, target_len):
        target_len = int(target_len / 4) # 20
        x_tech_list = []
        x_text_list = []
        t_list = []

        #max_len = 0

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for week in range(data["Week"].min(), data["Week"].max() + 1 - week_len):
            if not len(data[data["Week"] >= week + 4]) < target_len:
                target_data = data[data["Week"] >= week + 4].reset_index(drop=True)[:target_len]
            else:
                break

            target_data = target_data.reset_index(drop=True)
            #target_data = target_data[["open_mid", "high_mid", "low_mid", "close_mid", "20SMA", "20Upper", "20Lower"]].values
            target_data = target_data[["20SMA", "20Upper", "20Lower"]].values
            # scaling
            target_data = target_data / self.scaling_param
            t_list.append(target_data)

            x_text_region_list = []
            week_4_data_list = []
            for i in range(week, week + 4):
                week_data = data[data["Week"] == i]
                x_tech = week_data.reset_index(drop=True)
                x_tech = x_tech.drop(["Week", "Year", "US", "JP", "EU", "CH", "GE"], axis=1)
                x_tech = x_tech.values
                x_tech = torch.tensor(x_tech, dtype=torch.float32)
                # scaling
                x_tech = x_tech / self.scaling_param
                week_4_data_list.append(x_tech)

                # x_techのテンソルの最大値をtorch.maxで取得
                #max_len = max(max_len, torch.max(x_tech))
                

                # センチメントデータ
                ids_list = []
                for region in ["US", "JP", "EU", "CH", "GE"]:
                    # sentiment = week_data[week_data["Week"] == i][region][0]
                    sentiment = week_data[week_data["Week"] == i].iloc[0][region]

                    encoding = tokenizer(
                        sentiment, 
                        max_length=500, 
                        padding="max_length", 
                        truncation=True,
                        return_tensors="pt"
                    )
                    ids_list.append(encoding.input_ids)
                x_text_region = torch.stack(ids_list, dim=0)
                x_text_region_list.append(x_text_region)

            x_text_list.append(torch.stack(x_text_region_list, dim=0))
            x_tech_list.append(week_4_data_list)

        x_text = torch.stack(x_text_list, dim=0)
        t = torch.tensor(t_list, dtype=torch.float32)
        x_tech = x_tech_list

        #print(max_len)

        return x_tech, x_text, t

    # @staticmethod
    # def collate_fn(batch):
    #     print("batch_collate", batch[0][0][0].shape)
    #     # バッチ内の各データポイントに対してパディングを適用
    #     x_tech_batch = [pad_sequence(week_data, batch_first=True) for week_data in zip(*[item[0] for item in batch])]
    #     x_tech_batch = torch.cat(x_tech_batch, dim=0).view(len(batch), -1, *x_tech_batch[0].shape[1:])
        
    #     x_text_batch = torch.stack([item[1] for item in batch])
    #     t_batch = torch.stack([item[2] for item in batch])

    #     return x_tech_batch, x_text_batch, t_batch

    @staticmethod
    def collate_fn(batch):
        #print("batch_collate", batch[0][0][0].shape)
        # 固定長31で各週のデータをパディング
        padded_week_data = []
        for week_data in zip(*[item[0] for item in batch]):
            padded_data = [x[:31] if x.shape[0] > 31 else F.pad(x, (0, 0, 0, 31 - x.shape[0]), "constant", 0) for x in week_data]
            padded_week_data.append(torch.stack(padded_data, dim=0))

        # バッチ内でデータを組み合わせる
        x_tech_batch = torch.stack(padded_week_data, dim=1)

        x_text_batch = torch.stack([item[1] for item in batch])
        t_batch = torch.stack([item[2] for item in batch])

        return x_tech_batch, x_text_batch, t_batch

if __name__ == "__main__":
    # データローダーランダムシードの設定
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)

    # データ読み込み
    data = pd.read_pickle('sentiment_tech_data.pkl')
    financial_dataset = FinancialDataset(data, week_len=4, target_len=80)

    # # DataLoader を使用してデータセットをロード
    financial_dataloader = DataLoader(financial_dataset, batch_size=16, shuffle=True, collate_fn=FinancialDataset.collate_fn)

    for i, data in enumerate(financial_dataloader):
        print("Stock:", data[0].shape)
        print("Text:", data[1].shape)
        print("Target", data[2].shape)
        break
    




 