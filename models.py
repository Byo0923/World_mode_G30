import torch
import torch.nn as nn
from ml.text.bert import BERT_Report
from ml.price.transfomer import TransformerModelWithPositionalEncoding
from ml.text.text_concatenate import TEXT_Concatenate
from transformers import BertModel, BertTokenizer

# StockPriceEstimator
class StockPriceEstimator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        bert = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BERT_Report(bert, args.one_report_dim, tokenizer)
        self.transformer_model = TransformerModelWithPositionalEncoding(
            args.price_feature_dim, args.hidden_size, args.price_output_size, args.nhead, 
            args.num_encoder_layers, args.dim_feedforward)
        self.text_concatenate = TEXT_Concatenate(args.one_report_dim * 5, 128)
        self.generator = Generator()

        # 凍結する層の設定
        for name, param in self.bert_model.bert.named_parameters():
            if 'encoder.layer.11' not in name and 'pooler' not in name:
                param.requires_grad = False

    def forward(self, price, text):
        b, week, num = price.shape[0], price.shape[1], price.shape[2]
        region_num = text.shape[2]
        prices = price.reshape(b, week*num ,self.args.price_feature_dim)

        latent_vector, attention_weights = self.transformer_model(prices)
        
        all_reports = None
        for w in range(week):
            tex_latent_regions =[]
            for r in range(region_num):
                text_region = text[:,w, r ].reshape(b, 500)
                tex_latent = self.bert_model(text_region)
                tex_latent_regions.append(tex_latent)
            week_report = torch.cat( [tex_latent_regions[0],tex_latent_regions[1],tex_latent_regions[2],tex_latent_regions[3],tex_latent_regions[4]]  , dim=1)

            # Text concat
            week_report = self.text_concatenate(week_report)
            if all_reports == None:
                all_reports = week_report
            else:
                all_reports = torch.cat([all_reports, week_report], dim=1)#all_reports.append(week_reports)

        # all_reportsとPriceのLatentをCat
        all_reports_latent = torch.cat( [all_reports, latent_vector] , dim=1)

        out = self.generator(all_reports_latent).view(b, 20, 3)
        return out

class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.linear_1 = nn.Linear(768, 256)
        self.linear_2 = nn.Linear(256, 60)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out_1 = self.linear_1(x)
        out_2 = self.linear_2(self.relu(out_1))
        return out_2
    
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


    

