import pandas as pd

df = pd.read_csv('../data/MELD/train_sent_emo.csv')
sub_df = df.sample(n=5, random_state=220511, axis=0, ignore_index=True)
sub_df.to_csv('../data/MELD/meld_tiny.csv', index=False)