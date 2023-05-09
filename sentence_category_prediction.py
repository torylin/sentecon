import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from torch import cuda
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.cluster import KMeans
from empath import Empath
import pdb
import argparse
from tqdm import tqdm
import liwc
import os
import torch
import warnings
warnings.filterwarnings("ignore")

tqdm.pandas()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', default='../data/', help='path to data directory')
    parser.add_argument('-r', '--ref', default='MELD', help='path to reference corpus')
    parser.add_argument('--lm', default='all-MiniLM-L6-v2', help='language model for sentence embeddings')
    parser.add_argument('--target', help='name of target in task dataset')
    parser.add_argument('--text-name', help='name of column containing sentences')
    parser.add_argument('--lexicon', default='empath')
    parser.add_argument('--distance', default='cosine', help='distance metric used to compute embedding similarity')
    parser.add_argument('--dataset', help='name of task dataset')
    parser.add_argument('--lm-name')
    parser.add_argument('--num-centroids', type=int, default=1)
    parser.add_argument('--seed', type=int, default=230321)

    args = parser.parse_args()

    return args


def compute_representation(row, embeddings, centroids, num_centroids, rel_cats, rel_cats_header, centroid_names, distance):
    embedding = embeddings[row.name]
    similarities = dict.fromkeys(rel_cats, 0)

    # if distance == 'cosine':
    rel_centroids = centroids[sorted(list(set(rel_cats_header) & set(centroids.columns)))]
    centroids_array = rel_centroids.to_numpy()
    newsim = torch.nn.CosineSimilarity(dim=1)

    new_similarities = newsim(torch.tensor(embedding).to(device), torch.tensor(centroids_array.T).to(device))
    new_similarities = new_similarities.view(-1, num_centroids).max(dim=1).values
    
    cols = sorted(list(set(rel_cats) & set(centroid_names)))
    new_similarities = dict(zip(cols, new_similarities.tolist()))
    similarities.update(new_similarities)
    
    # else:
    #     for category in rel_cats:
    #         if category in centroids:
    #             if distance == 'L2':
    #                 dist = euclidean_distances(
    #                     embedding.reshape(1, -1), centroids[category].values.reshape(1, -1))[0][0]
    #                 similarities[category] = np.exp(-dist * 1 / len(embedding))
    #             elif distance == 'L1':
    #                 dist = manhattan_distances(
    #                     embedding.reshape(1, -1), centroids[category].values.reshape(1, -1))[0][0]
    #                 similarities[category] = np.exp(-dist * 1 / len(embedding))
    #         else:
    #             similarities[category] = 0
        
    return similarities

def compute_centroids(corpus_df, num_centroids, seed):
    centroids = dict()
    categories = set(corpus_df['category'].values)
    
    for category in categories:
        cat_embeds = np.vstack(corpus_df[corpus_df['category'] == category]['sentence_embedding'].values)
        if num_centroids == 1:
            centroids[category] = cat_embeds.mean(axis=0)
        else:
            cluster_model = KMeans(n_clusters=num_centroids, random_state=seed)
            try:
                cluster_model.fit(cat_embeds)
            except:
                pdb.set_trace()
            clusters = cluster_model.predict(cat_embeds)
            for cluster in set(clusters):
                centroids['{}_c{}'.format(category, cluster)] = cat_embeds[clusters == cluster].mean(axis=0)  

        centroids = pd.DataFrame.from_dict(centroids)

    return centroids

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

args = get_args()
if args.lm_name == None:
    args.lm_name = args.lm

lexicon = Empath()
if args.lexicon == 'empath':
    category_names = list(lexicon.cats.keys())
elif args.lexicon == 'LIWC':
    parse, category_names = liwc.load_token_parser('../liwc_dict/LIWC2015_English_Flat.dic')
    category_names = category_names[21:]

if args.num_centroids > 1:
    category_headers = []
    for cluster in range(args.num_centroids):
        category_headers += ['{}_c{}'.format(category, cluster) for category in category_names]
else:
    category_headers = category_names
        
corpus_df = pd.read_pickle('{}{}/{}_corpus_{}_{}.pkl'.format(
        args.data_dir, args.ref, args.ref, args.lexicon, args.lm_name))

corpus_df.dropna(inplace=True)
print(corpus_df.shape)

centroids = compute_centroids(corpus_df, args.num_centroids, args.seed)
centroids = centroids.reindex(sorted(centroids.columns), axis=1)
centroid_names = sorted(list(set([col.split('_')[0] for col in centroids.columns])))

if args.dataset == 'MELD':
    args.target = ['Sentiment', 'Emotion']

    train_sent = pd.read_csv('{}{}/dev.csv'.format(args.data_dir, args.dataset))
    train_embeds = np.load('{}{}/dev_embeds_{}.npy'.format(args.data_dir, args.dataset, args.lm_name))
else:
    train_sent = pd.read_csv('{}{}/train.csv'.format(args.data_dir, args.dataset))
    train_embeds = np.load('{}{}/train_embeds_{}.npy'.format(args.data_dir, args.dataset, args.lm_name))
    
test_sent = pd.read_csv('{}{}/test.csv'.format(args.data_dir, args.dataset))
test_embeds = np.load('{}{}/test_embeds_{}.npy'.format(args.data_dir, args.dataset, args.lm_name))

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = SentenceTransformer(args.lm)
# model = model.to(device)

# example_sent = ['I was happy at first, but the job offer slipped through my fingers']
# example_embed = model.encode(example_sent, show_progress_bar=True)
# example_dict = {'Utterance': example_sent}
# example_df = pd.DataFrame(example_dict)
# example_sentecon = compute_representation(example_df.iloc[0], example_embed, centroids, args.num_centroids, category_names, category_headers, centroid_names, args.distance)
# # norm = np.sum(list(example_sentecon.values()))
# # for k, v in example_sentecon.items():
# #     example_sentecon[k] /= norm

# example_sents = ['I was happy at first', 'but the job offer slipped through my fingers']
# example_embeds = model.encode(example_sents, show_progress_bar=True)
# example_dicts = {'Utterance': example_sents}
# example_dfs = pd.DataFrame(example_dicts)
# example_sentecon1 = compute_representation(example_dfs.iloc[0], example_embeds[0], centroids, args.num_centroids, category_names, category_headers, centroid_names, args.distance)
# example_sentecon2 = compute_representation(example_dfs.iloc[1], example_embeds[1], centroids, args.num_centroids, category_names, category_headers, centroid_names, args.distance)
# # norm1 = np.sum(list(example_sentecon1.values()))
# # norm2 = np.sum(list(example_sentecon2.values()))
# # for k, v in example_sentecon1.items():
# #     example_sentecon1[k] /= norm1
# #     example_sentecon2[k] /= norm2

# pdb.set_trace()

train_sent['representation'] = train_sent.progress_apply(
    compute_representation, 
    args=(train_embeds, centroids, args.num_centroids, category_names, category_headers, centroid_names, args.distance),
    axis=1)

rep_df = pd.DataFrame(train_sent['representation'].values.tolist())

rep_df[args.target] = train_sent[args.target]

if args.num_centroids == 1:
    rep_df.to_csv('{}{}/train_sentence_{}_centroid_{}_{}_ref{}.csv'.format(
        args.data_dir, args.dataset, args.lexicon, args.distance, args.lm_name, args.ref))
else:
        rep_df.to_csv('{}{}/train_sentence_{}_centroid_{}_{}_{}_ref{}.csv'.format(
        args.data_dir, args.dataset, args.lexicon, args.num_centroids, args.distance, args.lm_name, args.ref))

test_sent['representation'] = test_sent.progress_apply(
    compute_representation, 
    args=(test_embeds, centroids, args.num_centroids, category_names, category_headers, centroid_names, args.distance),
    axis=1)

rep_df_test = pd.DataFrame(test_sent['representation'].values.tolist())

rep_df_test[args.target] = test_sent[args.target]

if args.num_centroids == 1:
    rep_df_test.to_csv('{}{}/test_sentence_{}_centroid_{}_{}_ref{}.csv'.format(
        args.data_dir, args.dataset, args.lexicon, args.distance, args.lm_name, args.ref))
else:
    rep_df_test.to_csv('{}{}/test_sentence_{}_centroid_{}_{}_{}_ref{}.csv'.format(
        args.data_dir, args.dataset, args.lexicon, args.num_centroids, args.distance, args.lm_name, args.ref))