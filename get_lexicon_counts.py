import os
import pickle
import liwc
from empath import Empath
import pandas as pd
import contractions
import numpy as np
import argparse
import pdb
from collections import Counter
from tqdm import tqdm
import nltk
nltk.download('punkt')
tqdm.pandas()
import gensim
import gensim.downloader as api

from sklearn.metrics.pairwise import cosine_similarity

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data/', help='path to data directory')
    parser.add_argument('--dataset', default='MELD')
    parser.add_argument('--text-name', default='Utterance')
    parser.add_argument('--target', default='Sentiment')
    parser.add_argument('--csv-path')
    parser.add_argument('--lexicon', default='Empath')
    parser.add_argument('--word2vec', action='store_true')

    args = parser.parse_args()

    return args

def normalize(word_vec):
    norm = np.linalg.norm(word_vec)
    if norm == 0: 
       return word_vec
    return word_vec / norm


def get_liwc_labels(utterance, rel_cats, binary=False):
    try:
        tokens = nltk.word_tokenize(contractions.fix(utterance).lower())
    except:
        pdb.set_trace()
    counts = dict(Counter(category for token in tokens for category in parse(token)))

    # print(counts)

    label_vec = np.zeros(len(rel_cats))
    bin_label_vec = np.zeros(len(rel_cats))

    for i in range(len(rel_cats)):
        if not binary:
            label_vec[i] += counts.get(rel_cats[i], 0)
        else:
            if counts.get(rel_cats[i], 0) > 0:
                bin_label_vec[i] = 1

    if not binary:
        return label_vec
    return bin_label_vec

def get_word2vec_labels(utterance, rel_cats, vocab, vocab_embeds, cats_to_words, model):
    try:
        tokens = nltk.word_tokenize(contractions.fix(utterance).lower())
    except:
        pdb.set_trace()

    counts = dict.fromkeys(rel_cats, 0)

    tokens_invocab = np.zeros(len(tokens), dtype=bool)

    label_vec = np.zeros(len(rel_cats))

    for i in range(len(tokens)):
        tokens_invocab[i] = model.__contains__(tokens[i])

    if np.sum(tokens_invocab) == 0:
        return label_vec
    
    token_embeds = np.zeros((len(tokens), 300))
    token_invocab_embeds = model[np.array(tokens)[tokens_invocab]]

    token_embeds[tokens_invocab] = token_invocab_embeds

    similarities = cosine_similarity(token_embeds, vocab_embeds)

    sentence_similarities = np.sum(similarities, axis=0)

    label_vec = np.zeros(len(rel_cats))

    for i in range(len(rel_cats)):
        cat = rel_cats[i]
        associated_words = np.array(cats_to_words[cat])
        label_vec[i] = np.sum(sentence_similarities[associated_words])

    return label_vec

def get_empath_labels(utterance, lexicon):
    return list(lexicon.analyze(utterance.lower()).values())

args = get_args()

df = pd.read_csv('{}{}/{}'.format(args.data_dir, args.dataset, args.csv_path),
                 index_col=0).reset_index()

if args.lexicon == 'LIWC':
    lexicon_df = pd.read_csv('{}LIWCVocab/liwc_vocab.csv'.format(args.data_dir))
    parse, category_names = liwc.load_token_parser(
        '../liwc_dict/LIWC2015_English_Flat.dic')

    category_names = category_names[21:]

    if args.word2vec:

        model = api.load('word2vec-google-news-300')

        if os.path.exists('{}LIWCVocab/liwc_cats_to_words.pkl'.format(args.data_dir)):
            with open('{}LIWCVocab/liwc_cats_to_words.pkl'.format(args.data_dir), 'rb') as f:
                cats_to_words = pickle.load(f)

            vocab = np.genfromtxt('{}LIWCVocab/liwc_vocab.npy', dtype='str')

        else:

            vocab = []

            cats_to_words = {cat: [] for cat in category_names}
            for word in tqdm(lexicon_df['text'].values):
                if model.__contains__(word):
                    vocab.append(word)

            for i in tqdm(range(len(vocab))):
                vocab_word = vocab[i]
                cats = list(parse(vocab_word))
                for cat in cats:
                    if cat in category_names:
                        cats_to_words[cat].append(i)

            with open('{}LIWCVocab/liwc_cats_to_words.pkl'.format(args.data_dir), 'wb') as f:
                    pickle.dump(cats_to_words, f)
            
            np.savetxt('{}LIWCVocab/liwc_vocab.npy', vocab, fmt='%s')

        vocab_embeds = model[vocab]

        df['label_count'] = df[args.text_name].progress_apply(get_word2vec_labels, args=(category_names, vocab, vocab_embeds, cats_to_words, model, ))
        mid = 'word2vec_liwc_counts'

    else:
        df['label_count'] = df[args.text_name].progress_apply(get_liwc_labels, args=(category_names, False, ))

        mid = 'liwc_counts'

elif args.lexicon == 'Empath':
    lexicon_df = pd.read_csv('{}EmpathVocab/empath_vocab.csv'.format(args.data_dir))
    lexicon = Empath()
    category_names = list(lexicon.cats.keys())

    if args.word2vec:

        model = api.load('word2vec-google-news-300')

        if os.path.exists('{}EmpathVocab/empath_cats_to_words.pkl'.format(args.data_dir)):
            with open('{}EmpathVocab/empath_cats_to_words.pkl'.format(args.data_dir), 'rb') as f:
                cats_to_words = pickle.load(f)

            vocab = np.genfromtxt('{}EmpathVocab/empath_vocab.npy'.format(args.data_dir), dtype='str')
        
        else:

            vocab = []

            cats_to_words = {cat: [] for cat in category_names}
            for word in tqdm(lexicon_df['text'].values):
                if model.__contains__(word):
                    vocab.append(word)

            for i in tqdm(range(len(vocab))):
                vocab_word = vocab[i]
                empath_output = lexicon.analyze(vocab_word.lower())
                cats = np.array(list(empath_output.keys()))
                counts = np.array(list(empath_output.values()))
                cats = cats[counts > 0]

                for cat in cats:
                    if cat in category_names:
                        cats_to_words[cat].append(i)
            
            with open('{}EmpathVocab/empath_cats_to_words.pkl'.format(args.data_dir), 'wb') as f:
                    pickle.dump(cats_to_words, f)
            
            np.savetxt('{}EmpathVocab/empath_vocab.npy'.format(args.data_dir), vocab, fmt='%s')

        vocab_embeds = model[vocab]

        df['label_count'] = df[args.text_name].progress_apply(get_word2vec_labels, args=(category_names, vocab, vocab_embeds, cats_to_words, model, ))
        mid = 'word2vec_empath_counts'

    else:
        df['label_count'] = df[args.text_name].progress_apply(get_empath_labels, args=(lexicon, ))

        mid = 'empath_counts'

count_df = pd.DataFrame(np.stack(df['label_count'].values, axis=0),
            columns=category_names)
if args.dataset == 'MELD':
    count_df[['Emotion', 'Sentiment']] = df[['Emotion', 'Sentiment']]
else:
    count_df[args.target] = df[args.target].values

stem = args.csv_path.split('.')[0]

count_df.to_csv('{}/{}/{}_{}.csv'.format(args.data_dir, args.dataset, stem, mid))