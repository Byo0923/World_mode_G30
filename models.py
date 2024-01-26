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
            #if 'encoder.layer.11' not in name and 'pooler' not in name:
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
        #all_reports = torch.zeros_like(all_reports).to("cuda")
        latent_vector = torch.zeros_like(latent_vector).to("cuda")
        all_reports_latent = torch.cat( [all_reports, latent_vector] , dim=1)

        out = self.generator(all_reports_latent).view(b, 20, 3)
        return out

class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.layers = nn.Sequential(  
            nn.Linear(768, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 60),
        )

    def forward(self, x):
        return self.layers(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__() 

        self.conv_layers = nn.Sequential(
            nn.Conv1d(20, 10, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv1d(10, 5, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv1d(5, 1, kernel_size=3, stride=1, padding='same'),
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
        self.fc2 = nn.Sequential(
            nn.Linear(3, 1),
        )

    def forward(self, x):
        x = x.reshape(-1, x.shape[2], x.shape[1])
        latent = self.conv_layers(x)
        latent = latent.reshape(-1, latent.shape[2])
        out = self.fc2(latent)
        return out

if __name__ == "__main__":
    import argparse
    # ハイパーパラメータ
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_epoch_num", type=int, default=1000)
    parser.add_argument("--one_report_dim", type=int, default=64)
    parser.add_argument("--price_feature_dim", type=int, default=11)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--price_output_size", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--max_grad_norm", type=int, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    discriminator = Discriminator()
    x = torch.randn(8, 3, 20)
    out = discriminator(x)
    print(out.shape)

    model = StockPriceEstimator(args).to("cuda")
    print(model)

