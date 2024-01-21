import os
import pandas as pd

# 初期化
df = pd.DataFrame(columns=['Year', 'Week', 'US', 'JP', 'EU', 'CH', 'GE'])

# ファイルを読み込む
for filename in os.listdir('.'):
    if filename.endswith('.txt'):
        # ファイル名から情報を抽出
        name_parts = filename.split('_')
        year = name_parts[0]
        week = name_parts[1]
        category = name_parts[2].replace('.txt', '')

        # カテゴリを市場にマッピング
        category_mapping = {
            'US Stock Market': 'US',
            'Japanese Stock Market': 'JP',
            'European Stock Market': 'EU',
            'Chinese Stock Market': 'CH',
            'Global Economic Outlook': 'GE'
        }
        category = category_mapping.get(category, category)

        # ファイルを開く
        with open(filename, 'r', encoding='UTF-16') as file:
            content = file.read().strip()

        # データフレームに追加
        if len(df[(df['Year'] == year) & (df['Week'] == week)]) > 0:
            df.loc[(df['Year'] == year) & (df['Week'] == week), category] = content
        else:
            df.loc[len(df)] = {'Year': year, 'Week': week, category: content}

# YearとWeekの小さい順に行を並び替え
df['Year'] = df['Year'].astype(int)
df['Week'] = df['Week'].astype(int)
df = df.sort_values(['Year', 'Week'])

# 行のインデックスをリセット
df = df.reset_index(drop=True)

# pickleファイルとして保存
df.to_pickle('output.pkl')