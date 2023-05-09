from transformers import AutoTokenizer, AutoModel
import torch
import argparse
import pdb
import numpy as np
import liwc
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ttest_rel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm', default='sentence-transformers/all-mpnet-base-v2',
                        help='language model for sentence embeddings')
    parser.add_argument('--keyword')

    args = parser.parse_args()
    return args

def compute_representation(embedding, centroids, rel_cats, distance):

    similarities = dict.fromkeys(rel_cats, 0)

    rel_centroids = centroids[rel_cats]
    cols = rel_centroids.columns
    centroids_array = rel_centroids.to_numpy()
    newsim = torch.nn.CosineSimilarity(dim=1)

    similarities = newsim(embedding, torch.tensor(centroids_array.T).to(device))
    similarities = dict(zip(cols, similarities.tolist()))

    return similarities


def compute_centroids(corpus_df):
    centroids = dict()
    categories = set(corpus_df['category'].values)

    for category in categories:
        centroids[category] = np.vstack(corpus_df[corpus_df['category'] == category]['sentence_embedding'].values).mean(
            axis=0)

    centroids = pd.DataFrame.from_dict(centroids)

    return centroids

args = get_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

tokenizer = AutoTokenizer.from_pretrained(args.lm)
model = AutoModel.from_pretrained(args.lm)
model.to(device)

parse, category_names = liwc.load_token_parser('../liwc_dict/LIWC2015_English_Flat.dic')
category_names = category_names[21:]

# INSERT WORDS HERE
if args.keyword == 'bright':
    sst = pd.read_csv('../data/SST/train.csv')
    scale_df = pd.read_csv('../data/SST/train_sentence_LIWC_centroid_cosine_all-mpnet-base-v2_refLIWCVocab.csv',
                       index_col=0)

    utterances = sst[sst['sentence'].str.contains(
        args.keyword, case=False)]['sentence'].str.lower().values.tolist() + [args.keyword]

    lively_words = ["shining", "vivid", "beaming"]
    smart_words = ["intelligent", "smart", "clever"]
    lively_sentence_idx = np.array([0, 4, 7, 8, 12])
    smart_sentence_idx = np.array([2, 3, 6, 9, 13])

elif args.keyword == 'hard':
    sst = pd.read_csv('../data/MELD/train_sent_emo.csv')
    scale_df = pd.read_csv('../data/MELD/train_sentence_LIWC_centroid_cosine_all-mpnet-base-v2_refLIWCVocab.csv',
                       index_col=0)
    utterances = sst[sst['Utterance'].str.contains(args.keyword, case=False)]['Utterance'].str.lower().values.tolist() + [args.keyword]

    lively_words = ['forceful', 'strong', 'aggressive']
    smart_words = ['difficult', 'tough', 'arduous']
    lively_sentence_idx = np.array([1, 22, 24, 26, 27, 32, 33, 38, 39, 40])
    smart_sentence_idx = np.array([0, 3, 4, 5, 9, 10, 11, 14, 17, 18, 19, 21, 23, 28, 29,
        30, 31, 37, 42, 45])

elif args.keyword == 'dull':
    sst = pd.read_csv('../data/SST/train.csv')
    scale_df = pd.read_csv('../data/SST/train_sentence_LIWC_centroid_cosine_all-mpnet-base-v2_refLIWCVocab.csv',
                       index_col=0)
    utterances = sst[sst['sentence'].str.contains(args.keyword, case=False)]['sentence'].str.lower().values.tolist() + [args.keyword]
    lively_words = ['boring', 'uninteresting', 'tiresome']
    smart_words = ['unintelligent', 'stupid', 'slow-witted']
    lively_sentence_idx = np.array([0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22,
                                    23, 24, 25, 29, 30, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46,
                                    47, 48, 49, 50, 52, 53, 54, 55])
    smart_sentence_idx = np.array([6, 7, 9, 13, 26, 28])

elif args.keyword == 'fair':
    sst = pd.read_csv('../data/MELD/train_sent_emo.csv')
    scale_df = pd.read_csv('../data/MELD/train_sentence_LIWC_centroid_cosine_all-mpnet-base-v2_refLIWCVocab.csv',
                       index_col=0)
    utterances = sst[sst['Utterance'].str.contains(
        args.keyword, case=False)]['Utterance'].str.lower().values.tolist() + [args.keyword]

    for i in range(len(utterances)):
        print('{} {}'.format(i, utterances[i]))

    lively_words = ["reasonable", "adequate", "sufficient"]
    smart_words = ["just", "equitable", "honorable"]
    lively_sentence_idx = np.array([5, 6, 11])
    smart_sentence_idx = np.array([0, 2, 4, 7, 8, 9, 12, 13])


elif args.keyword == 'dark':
    sst = pd.read_csv('../data/SST/train.csv')
    scale_df = pd.read_csv('../data/SST/train_sentence_LIWC_centroid_cosine_all-mpnet-base-v2_refLIWCVocab.csv',
                       index_col=0)
    utterances = sst[sst['sentence'].str.contains(
        args.keyword, case=False)]['sentence'].str.lower().values.tolist() + [args.keyword]
    # for i in range(len(utterances)):
    #     print('{} {}'.format(i, utterances[i]))
    # pdb.set_trace()
    lively_words = ["dim", "unlit", "black"]
    smart_words = ["grim", "macabre", "sinister"]
    lively_sentence_idx = np.array([1, 7, 14, 19, 23, 24, 25, 29, 35, 40, 41, 42, 55])
    smart_sentence_idx = np.array([0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 17, 18, 20, 21, 22,
        26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 
        56, 57, 58])

elif args.keyword == 'cool':
    sst = pd.read_csv('../data/SST/train.csv')
    scale_df = pd.read_csv('../data/SST/train_sentence_LIWC_centroid_cosine_all-mpnet-base-v2_refLIWCVocab.csv',
                       index_col=0)
    utterances = sst[sst['sentence'].str.contains(
        args.keyword, case=False)]['sentence'].str.lower().values.tolist() + [args.keyword]

    # for i in range(len(utterances)):
    #     print('{} {}'.format(i, utterances[i]))

    lively_words = ["calm", "unemotional", "dispassionate"]
    smart_words = ["good", "impressive", "excellent"]

    lively_sentence_idx = [0, 4, 5, 11, 20]
    smart_sentence_idx = [1, 3, 6, 9, 10, 13, 14, 15, 16, 18, 19, 21, 22]

    # pdb.set_trace()

scale_df = scale_df[category_names].sort_index(axis=1)

input_ids = tokenizer(utterances, truncation=True, padding=True, return_tensors="pt")
# tokens = tokenizer.batch_decode(input_ids)
input_ids.to(device)

tokens = []
for i in range(len(utterances)):
    utterance = utterances[i]
    tokenized = input_ids[i]
    utterance_tokens = []
    for j in range(len(tokenized.offsets)):
        token = utterance[tokenized.offsets[j][0]:tokenized.offsets[j][1]]
        utterance_tokens.append(token)
    tokens.append(np.array(utterance_tokens))

tokens = np.array(tokens)
keyword_idx = list(np.where(np.char.find(tokens, args.keyword) >= 0))
dups = [idx for idx, item in enumerate(keyword_idx[0]) if item in keyword_idx[0][:idx]]
keyword_idx[0] = np.delete(keyword_idx[0], dups)
keyword_idx[1] = np.delete(keyword_idx[1], dups)

out = model(**input_ids)
keyword_embeds = out.last_hidden_state[keyword_idx[0], keyword_idx[1], :].detach()

lively_word_input_ids = tokenizer(lively_words, truncation=True, padding=True, return_tensors="pt")
lively_word_input_ids.to(device)

out_lively_word = model(**lively_word_input_ids)
lively_word_embeds = out_lively_word.last_hidden_state[:,1,:].detach()

smart_word_input_ids = tokenizer(smart_words, truncation=True, padding=True, return_tensors="pt")
smart_word_input_ids.to(device)

out_smart_word = model(**smart_word_input_ids)
smart_word_embeds = out_smart_word.last_hidden_state[:,1,:].detach()

sim = torch.nn.CosineSimilarity(dim=0)

lively_sims = []
smart_sims = []
for i in range(keyword_embeds.shape[0]):
    sublist_lively = []
    sublist_smart = []
    for j in range(lively_word_embeds.shape[0]):
        # print(utterances[i], '|', lively_words[j])
        lively_sim = sim(keyword_embeds[i], lively_word_embeds[j]).item()
        sublist_lively.append(lively_sim)
        # print(lively_sim)
    # print('{} | lively: {}'.format(utterances[i], np.mean(sublist_lively)))
    lively_sims.append(np.mean(sublist_lively))

    # print("\nSmart sims")
    for j in range(smart_word_embeds.shape[0]):
        # print(utterances[i], '|', smart_words[j])
        smart_sim = sim(keyword_embeds[i], smart_word_embeds[j]).item()
        sublist_smart.append(smart_sim)
        # print(smart_sim)
    # print('{} | intelligent: {}'.format(utterances[i], np.mean(sublist_smart)))
    smart_sims.append(np.mean(sublist_smart))
    print('{} {} | shining - intelligent: {}'.format(i, utterances[i], np.mean(sublist_lively) - np.mean(sublist_smart)))

lively_sims = np.array(lively_sims)
smart_sims = np.array(smart_sims)
sims_diff = lively_sims[:-1] - smart_sims[:-1]

print("Sentences with meaning 1")
print("Similarity to meaning 1: {}".format(np.mean(lively_sims[lively_sentence_idx])))
print("Similarity to meaning 2: {}".format(np.mean(smart_sims[lively_sentence_idx])))
print("t-test p-value: {}".format(ttest_rel(lively_sims[lively_sentence_idx], smart_sims[lively_sentence_idx])))

print("\nSentences with meaning 2")
print("Similarity to meaning 1: {}".format(np.mean(lively_sims[smart_sentence_idx])))
print("Similarity to meaning 2: {}".format(np.mean(smart_sims[smart_sentence_idx])))
print("t-test p-value: {}".format(ttest_rel(lively_sims[smart_sentence_idx], smart_sims[smart_sentence_idx])))

# print("'Intelligent' sentences | shining - intelligent: {}".format(np.mean(sims_diff[smart_sentence_idx])))
# print("'Shining' sentences | shining - intelligent: {}".format(np.mean(sims_diff[lively_sentence_idx])))
# print("'Intelligent' sentences | average 'shining' sim: {}".format(np.mean(lively_sims[smart_sentence_idx])))
# print("'Shining' sentences | average 'shining' sim: {}".format(np.mean(lively_sims[lively_sentence_idx])))
# print("'Intelligent' sentences | average 'intelligent' sim: {}".format(np.mean(smart_sims[smart_sentence_idx])))
# print("'Shining' sentences | average 'intelligent' sim: {}".format(np.mean(smart_sims[lively_sentence_idx])))
# print("Bright 'shining' sim: {}".format(lively_sims[-1]))
# print("Bright 'intelligent' sim: {}".format(smart_sims[-1]))


corpus_df = pd.read_pickle('../data/LIWCVocab/LIWCVocab_corpus_LIWC_all-mpnet-base-v2.pkl')
corpus_df.dropna(inplace=True)

parse, category_names = liwc.load_token_parser('../liwc_dict/LIWC2015_English_Flat.dic')
category_names = category_names[21:]

centroids = compute_centroids(corpus_df)
centroids = centroids.reindex(sorted(centroids.columns), axis=1)

scale_df = scale_df.reindex(sorted(scale_df.columns), axis=1)
scale_df = scale_df[category_names]

scaler = StandardScaler()
scaler.fit(scale_df)

keyword_reps = []

for i in range(keyword_embeds.shape[0]):
    rep = compute_representation(keyword_embeds[i], centroids, category_names, 'cosine')
    keyword_reps.append(rep)

keyword_df = pd.DataFrame(keyword_reps)
keyword_df = keyword_df[category_names].sort_index(axis=1)
keyword_df = scaler.transform(keyword_df)
keyword_df = scale(keyword_df, axis=1, with_mean=True, with_std=True, copy=True)

lively_word_reps = []
for i in range(lively_word_embeds.shape[0]):
    rep = compute_representation(lively_word_embeds[i], centroids, category_names, 'cosine')
    lively_word_reps.append(rep)

lively_word_df = pd.DataFrame(lively_word_reps)
lively_word_df = lively_word_df[category_names].sort_index(axis=1)
lively_word_df = scaler.transform(lively_word_df)
lively_word_df = scale(lively_word_df, axis=1, with_mean=True, with_std=True, copy=True)

smart_word_reps = []
for i in range(smart_word_embeds.shape[0]):
    rep = compute_representation(smart_word_embeds[i], centroids, category_names, 'cosine')
    smart_word_reps.append(rep)

smart_word_df = pd.DataFrame(smart_word_reps)
smart_word_df = smart_word_df[category_names].sort_index(axis=1)
smart_word_df = scaler.transform(smart_word_df)
smart_word_df = scale(smart_word_df, axis=1, with_mean=True, with_std=True, copy=True)

# pca = PCA()
# all_embeds = np.vstack([keyword_df, lively_word_df, smart_word_df])
all_embeds = keyword_df
# all_embeds_pc = pca.fit_transform(all_embeds.detach().cpu())

tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=20230110)
# all_embeds_tsne = tsne.fit_transform(all_embeds_pc)
# all_embeds_tsne = tsne.fit_transform(all_embeds.detach().cpu())
all_embeds_tsne = tsne.fit_transform(all_embeds)

colors = np.array([None]*all_embeds_tsne.shape[0])
colors[lively_sentence_idx] = 'Meaning 1'
colors[smart_sentence_idx] = 'Meaning 2'
colors[keyword_embeds.shape[0]:keyword_embeds.shape[0]+lively_word_embeds.shape[0]] = 'Reference 1'
colors[keyword_embeds.shape[0]+lively_word_embeds.shape[0]:] = 'Reference 2'

tsne_df = pd.DataFrame()
tsne_df['color'] = colors
tsne_df['c1'] = all_embeds_tsne[:,0]
tsne_df['c2'] = all_embeds_tsne[:,1]

sns.set(font_scale=1.5)
ax = sns.scatterplot(x='c1', y='c2', hue=tsne_df.color.tolist(), data=tsne_df)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set(title="\'{}\'".format(args.keyword))
ax.get_figure().savefig('../plots/tsne_{}.png'.format(args.keyword), bbox_inches='tight')

lively_sims = []
smart_sims = []
print()
for i in range(keyword_df.shape[0]):
    sublist_lively = []
    sublist_smart = []
    for j in range(lively_word_df.shape[0]):
        lively_sim = sim(torch.tensor(keyword_df[i]), torch.tensor(lively_word_df[j])).item()
        sublist_lively.append(lively_sim)
        # print(lively_sim)
    # print('{} | lively: {}'.format(utterances[i], np.mean(sublist_lively)))
    lively_sims.append(np.mean(sublist_lively))

    # print("\nSmart sims")
    for j in range(smart_word_df.shape[0]):
        # print(utterances[i], '|', smart_words[j])
        smart_sim = sim(torch.tensor(keyword_df[i]), torch.tensor(smart_word_df[j])).item()
        sublist_smart.append(smart_sim)
        # print(smart_sim)
    # print('{} | intelligent: {}'.format(utterances[i], np.mean(sublist_smart)))
    smart_sims.append(np.mean(sublist_smart))
    print('{} {} | shining - intelligent: {}'.format(i, utterances[i], np.mean(sublist_lively) - np.mean(sublist_smart)))

lively_sims = np.array(lively_sims)
smart_sims = np.array(smart_sims)
# max_sim = np.max(lively_sims[:-1].tolist() + smart_sims[:-1].tolist())
# min_sim = np.min(lively_sims[:-1].tolist() + smart_sims[:-1].tolist())
# mean_sim = np.mean(lively_sims[:-1].tolist() + smart_sims[:-1].tolist())
# lively_sims_norm = (lively_sims - min_sim) / (max_sim - min_sim)
# smart_sims_norm = (smart_sims - min_sim) / (max_sim - min_sim)
# sims_diff_norm = lively_sims_norm - smart_sims_norm
sims_diff = lively_sims[:-1] - smart_sims[:-1]
# pdb.set_trace()
# sims_diff /= (sims_diff.max() - sims_diff.min())

print("Sentences with meaning 1")
print("Similarity to meaning 1: {}".format(np.mean(lively_sims[lively_sentence_idx])))
print("Similarity to meaning 2: {}".format(np.mean(smart_sims[lively_sentence_idx])))
print("t-test p-value: {}".format(ttest_rel(lively_sims[lively_sentence_idx], smart_sims[lively_sentence_idx])))

print("\nSentences with meaning 2")
print("Similarity to meaning 1: {}".format(np.mean(lively_sims[smart_sentence_idx])))
print("Similarity to meaning 2: {}".format(np.mean(smart_sims[smart_sentence_idx])))
print("t-test p-value: {}".format(ttest_rel(lively_sims[smart_sentence_idx], smart_sims[smart_sentence_idx])))

print('\nSame-meaning similarity: {}'.format(np.mean(np.concatenate([lively_sims[lively_sentence_idx],
    smart_sims[smart_sentence_idx]]))))
print('Opposite-meaning similarity: {}'.format(np.mean(np.concatenate([lively_sims[smart_sentence_idx],
    smart_sims[lively_sentence_idx]]))))
print('Individual similarity ratio: {}'.format(np.mean(np.concatenate([lively_sims[lively_sentence_idx],
    smart_sims[smart_sentence_idx]]))/np.mean(np.concatenate([lively_sims[smart_sentence_idx],
    smart_sims[lively_sentence_idx]]))))

same_sims = []
opp_sims = []
for i in range(len(lively_sims)):
    if i in lively_sentence_idx:
        same_sims.append(lively_sims[i])
        opp_sims.append(smart_sims[i])
    elif i in smart_sentence_idx:
        same_sims.append(smart_sims[i])
        opp_sims.append(lively_sims[i])

print('t-test p-value: {}'.format(ttest_rel(same_sims, opp_sims)))

# print("'Intelligent' sentences | shining - intelligent: {}".format(np.mean(sims_diff[smart_sentence_idx])))
# print("'Shining' sentences | shining - intelligent: {}".format(np.mean(sims_diff[lively_sentence_idx])))
# print("'Intelligent' sentences | average 'shining' sim: {}".format(np.mean(lively_sims[smart_sentence_idx])))
# print("'Shining' sentences | average 'shining' sim: {}".format(np.mean(lively_sims[lively_sentence_idx])))
# print("'Intelligent' sentences | average 'intelligent' sim: {}".format(np.mean(smart_sims[smart_sentence_idx])))
# print("'Shining' sentences | average 'intelligent' sim: {}".format(np.mean(smart_sims[lively_sentence_idx])))
# print("Bright 'shining' sim: {}".format(lively_sims[-1]))
# print("Bright 'intelligent' sim: {}".format(smart_sims[-1]))