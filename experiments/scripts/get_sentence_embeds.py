from sentence_transformers import SentenceTransformer, models
import pandas as pd
import numpy as np
from torch import cuda
import argparse
import pdb
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', help='path to data directory')
    parser.add_argument('--csv-path', help='csv file name')
    parser.add_argument('--lm', default='all-MiniLM-L6-v2', help='language model for sentence embeddings')
    parser.add_argument('--text-name', help='name of column containing sentences')
    parser.add_argument('--lm-name')
    parser.add_argument('--lm-library', default='sentence-transformers')

    args = parser.parse_args()

    return args

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

args = get_args()
if args.lm_name == None:
    args.lm_name = args.lm

if args.lm_library == 'sentence-transformers':
    model = SentenceTransformer(args.lm)
elif args.lm_library == 'transformers':
    word_model = models.Transformer(args.lm)
    pooling = models.Pooling(word_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_model, pooling])
    
model.to(device)

df = pd.read_csv('{}{}'.format(args.data_dir, args.csv_path))

sentence_embeds = model.encode(df[args.text_name].values,
                              show_progress_bar=True)

csv_filename = args.csv_path.split('.')[0]

np.save('{}{}_embeds_{}.npy'.format(args.data_dir, csv_filename, args.lm_name), sentence_embeds)