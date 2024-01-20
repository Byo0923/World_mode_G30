'''
20231124
made by Yuma Koyamatsu
'''

import pandas as pd
import os
import glob
import numpy as np
import datetime
import time
import sys
import zipfile
from natsort import natsorted

# ベースディレクトリ
base_dir = 'US500'

# zip_files
zip_files = glob.glob(base_dir + '/*.zip')

# unzip展開先フォルダ
unzip_dir = f'tmp/{base_dir}'
if not os.path.exists(unzip_dir):
    os.makedirs(unzip_dir)

# Zip展開
for zip_file in zip_files:
    with zipfile.ZipFile(zip_file) as existing_zip:
        existing_zip.extractall(unzip_dir)

# ディレクトリ内のディレクトリ名を取得
dirs = natsorted(os.listdir(unzip_dir))
df_list = []
for dir in dirs:
    csv_dir = os.path.join(unzip_dir, dir)
    files = natsorted(glob.glob(csv_dir + '/*.csv'))
    
    for file in files:
        df = pd.read_csv(file, encoding='shift_jis')
        df_list.append(df)

df = pd.concat(df_list, axis=0).reset_index(drop=True)

# カラム名を変更
df = df.rename(
    columns={
        '日時': 'date',
        '始値(BID)': 'open_bid',
        '高値(BID)': 'high_bid',
        '安値(BID)': 'low_bid',
        '終値(BID)': 'close_bid',
        '始値(ASK)': 'open_ask',
        '高値(ASK)': 'high_ask',
        '安値(ASK)': 'low_ask',
        '終値(ASK)': 'close_ask',
    }
)

# save
df.to_csv(f'{base_dir}.csv', index=False)
# pkl
df.to_pickle(f'{base_dir}.pkl')