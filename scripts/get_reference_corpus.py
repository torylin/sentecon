import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
from empath import Empath
import pdb
from sentence_transformers import SentenceTransformer, models
import argparse
import os
import liwc
import pdb
import pickle
import gc

gc.collect()
torch.cuda.empty_cache()

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data/', help='path to data directory')
    parser.add_argument('--dataset', default='MELD')
    parser.add_argument('--text-name', default='Utterance')
    parser.add_argument('--csv-path', default='train_sent_emo.csv')
    parser.add_argument('--lexicon', default='empath')
    parser.add_argument('--lm', default='all-MiniLM-L6-v2', help='language model for sentence embeddings')
    parser.add_argument('--lm-name')
    parser.add_argument('--lm-library', default='sentence-transformers')

    args = parser.parse_args()

    return args

def get_word_embeddings(utterance, tokenizer, model, device):
    try:
        input_ids = torch.tensor(tokenizer.encode(utterance, truncation=True)).unsqueeze(0).to(device)
    except:
        input_ids = torch.tensor(tokenizer.encode(' ', truncation=True)).unsqueeze(0).to(device)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    embeddings = last_hidden_states[:, 1:-1, :]
    return embeddings

def get_pairs(utterance, lexicon, tokenizer, model, device, args, category_names, words_to_cats):

    # if args.lm != 'word2vec':
    #     embeddings = get_word_embeddings(utterance, tokenizer, model, device)
    try:
        tokens = tokenizer.tokenize(utterance[:512])
    except:
        tokens = tokenizer.tokenize(' ')

    pairs = []

    # if args.lexicon == 'empath':
    #     # pdb.set_trace()
    #     for i in range(len(tokens)):
    #         counts = lexicon.analyze(tokens[i])
    #         for category, value in counts.items():
    #             if value > 0:
    #                 # if args.lm == 'word2vec':
    #                 pairs.append((category, tokens[i], i, utterance))
                    # else:
                        # pairs.append((category, tokens[i], i, utterance, embeddings[:,i,:].detach().cpu().numpy().tolist()))
    # elif args.lexicon == 'LIWC':
    for i in range(len(tokens)):
        for category in words_to_cats.get(tokens[i], []):
        # for category in parse(tokens[i]):
            if category in category_names:
                # if args.lm == 'word2vec':
                pairs.append((category, tokens[i], i, utterance))
                    # else:
                        # pairs.append((category, tokens[i], i, utterance, embeddings[:,i,:].detach().cpu().numpy().tolist()))

    return pairs

def create_df(pairs, args, i):
    pairs_df = pd.DataFrame(pairs, 
                        columns=['category', 'word', 'word_idx', 'sentence', 'word_embedding']
                       ).drop_duplicates(subset = ['category', 'word', 'word_idx', 'sentence'])

    sentence_embeds = sentence_model.encode(pairs_df['sentence'].values,
                      show_progress_bar=True)
    pairs_df['sentence_embedding'] = pd.Series(list(sentence_embeds))

    pairs_df.to_pickle('{}{}/{}_corpus_{}_{}_{}.pkl'.format(args.data_dir, args.dataset, args.dataset, args.lexicon, 
        args.lm_name, i))


args = get_args()
if args.lm_name == None:
    args.lm_name = args.lm
train_sent = pd.read_csv('{}{}/{}'.format(args.data_dir, args.dataset, args.csv_path),
                        index_col=0).reset_index()


lexicon = Empath()
if args.lexicon == 'empath':
    category_names = list(lexicon.cats.keys())
elif args.lexicon == 'LIWC':
    parse, category_names = liwc.load_token_parser('../liwc_dict/LIWC2015_English_Flat.dic')
    category_names = category_names[21:]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

model = model.to(device)

if args.lm == 'word2vec':

    print('word2vec')
    pairs = []
    words_to_cats = {word: [] for word in vocab}
    for i in tqdm(train_sent.index):
        pairs += get_pairs(train_sent[args.text_name][i], lexicon, tokenizer, model, device, args, category_names, words_to_cats)

    pairs_df = pd.DataFrame(pairs, 
                    columns=['category', 'word', 'word_idx', 'sentence']
                   ).drop_duplicates(subset = ['category', 'word', 'word_idx', 'sentence'])

    sentence_model = api.load('word2vec-google-news-300')
    sentence_embeds = sentence_model[pairs_df['sentence'].values]
    pairs_df['sentence_embedding'] = pd.Series(list(sentence_embeds))

    pairs_df.reset_index(drop=True, inplace=True)

    pairs_df.to_pickle('{}{}/{}_corpus_{}_{}.pkl'.format(args.data_dir, args.dataset, args.dataset, args.lexicon, args.lm_name))

else:
    
    if args.lm_library == 'sentence-transformers':
        sentence_model = SentenceTransformer(args.lm)
    elif args.lm_library == 'transformers':
        word_model = models.Transformer(args.lm)
        pooling = models.Pooling(word_model.get_word_embedding_dimension())
        sentence_model = SentenceTransformer(modules=[word_model, pooling])

    sentence_model = sentence_model.to(device)
    print("# parameters: {}".format(sum(p.numel() for p in sentence_model.parameters())))

    train_sentence_embeds = sentence_model.encode(train_sent[args.text_name].values,
                          show_progress_bar=True)
    train_sent['sentence_embedding'] = pd.Series(list(train_sentence_embeds))

    if args.lexicon == 'LIWC':

        if os.path.exists('{}LIWCVocab/liwc_words_to_cats.pkl'.format(args.data_dir)):
            with open('{}LIWCVocab/liwc_words_to_cats.pkl'.format(args.data_dir), 'rb') as f:
                words_to_cats = pickle.load(f)

        else:
            lexicon_df = pd.read_csv('{}LIWCVocab/liwc_vocab.csv'.format(args.data_dir))
            vocab = lexicon_df['text'].values
            words_to_cats = {word: [] for word in vocab}
            for i in tqdm(range(len(vocab))):
                vocab_word = vocab[i]
                cats = [cat for cat in parse(vocab_word) if cat in category_names]
                words_to_cats[vocab_word] = cats

            with open('{}LIWCVocab/liwc_words_to_cats.pkl'.format(args.data_dir), 'wb') as f:
                    pickle.dump(words_to_cats, f)

    elif args.lexicon == 'empath':

        if os.path.exists('{}EmpathVocab/empath_words_to_cats.pkl'.format(args.data_dir)):
            with open('{}EmpathVocab/empath_words_to_cats.pkl'.format(args.data_dir), 'rb') as f:
                words_to_cats = pickle.load(f)
        else:

            lexicon_df = pd.read_csv('{}EmpathVocab/empath_vocab.csv'.format(args.data_dir))
            lexicon_df.dropna(axis=0, inplace=True)
            
            vocab = lexicon_df['text'].values
            words_to_cats = {word: [] for word in vocab}
            for i in tqdm(range(len(vocab))):
                vocab_word = vocab[i]
                try:
                    empath_output = lexicon.analyze(vocab_word)
                except:
                    pdb.set_trace()
                cats = np.array(list(empath_output.keys()))
                counts = np.array(list(empath_output.values()))
                cats = cats[counts > 0]
                words_to_cats[vocab_word] = cats
            
            with open('{}EmpathVocab/empath_words_to_cats.pkl'.format(args.data_dir), 'wb') as f:
                pickle.dump(words_to_cats, f)

    pairs = []

    for i in tqdm(train_sent.index):
        pairs += get_pairs(train_sent[args.text_name][i], lexicon, tokenizer, model, device, args, category_names, words_to_cats)
        # if (i % 3000 == 0) and (i != 0):
        #     create_df(pairs, args, int(i/3000))
        #     pairs = []

    pairs_df = pd.DataFrame(pairs, 
                            columns=['category', 'word', 'word_idx', 'sentence']
                           ).drop_duplicates(subset = ['category', 'word', 'word_idx', 'sentence'])

    train_sent = train_sent[[args.text_name, 'sentence_embedding']]
    train_sent.columns = ['sentence', 'sentence_embedding']
    pairs_df = pairs_df.merge(train_sent, on='sentence')

    # pdb.set_trace()

    # sentence_embeds = sentence_model.encode(pairs_df['sentence'].values,
    #                       show_progress_bar=True)
    # pairs_df.reset_index(drop=True, inplace=True)
    # pairs_df['sentence_embedding'] = pd.Series(list(sentence_embeds))

    # df_list = []
    # for i in range(1, 4):
    #     if 3000*i < np.max(train_sent.index):
    #         sub_df = pd.read_pickle('{}{}/{}_corpus_{}_{}_{}.pkl'.format(args.data_dir, args.dataset, args.dataset, 
    #             args.lexicon, args.lm, i))
    #         df_list.append(sub_df)

    # df_list += [pairs_df]
    # pairs_df = pd.concat(df_list)

    pairs_df.reset_index(drop=True, inplace=True)

    pairs_df.to_pickle('{}{}/{}_corpus_{}_{}.pkl'.format(args.data_dir, args.dataset, args.dataset, args.lexicon, args.lm_name))

    # for i in range(1, 4):
    #     if 3000*i < np.max(train_sent.index):
    #         if os.path.exists('{}{}/{}_corpus_{}_{}_{}.pkl'.format(args.data_dir, args.dataset, args.dataset, args.lexicon, args.lm, i)):
    #             os.remove('{}{}/{}_corpus_{}_{}_{}.pkl'.format(args.data_dir, args.dataset, args.dataset, args.lexicon, args.lm, i))