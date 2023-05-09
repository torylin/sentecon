import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, scale, minmax_scale
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from tqdm import tqdm
from scipy import stats
import argparse
import pdb
import liwc
from empath import Empath

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data/', help='data directory')
    parser.add_argument('--dataset', help='task dataset')
    parser.add_argument('--lm-name', default='all-MiniLM-L6-v2', help='language model for sentence embeddings')
    parser.add_argument('--distance', default='cosine', help='distance metric between sentence embeddings')
    parser.add_argument('--text-name', help='name of column containing sentences')
    parser.add_argument('--representation', help='LIWC or Empath or Sentence-LIWC or Sentence-Empath?')
    parser.add_argument('--ref', default='MELD', help='reference corpus used')
    parser.add_argument('--target', help='name of target in task dataset')
    parser.add_argument('--task', default='classification', help='classification or regression?')
    parser.add_argument('--model', help='SVM or LR?')
    parser.add_argument('--num-centroids', type=int, default=1)
    parser.add_argument('--row-norm', action='store_true', help='add per-row mean/var normalization')

    args = parser.parse_args()

    return args

args = get_args()

if args.representation == 'LIWC':
    mid = 'liwc_counts'
elif args.representation == 'Empath':
    mid = 'empath_counts'
elif args.representation == 'Sentence-LIWC':
    if args.num_centroids == 1:
        mid = 'sentence_LIWC_centroid_{}_{}_ref{}'.format(args.distance, args.lm_name, args.ref)
    else:
        mid = 'sentence_LIWC_centroid_{}_{}_{}_ref{}'.format(args.num_centroids, args.distance, args.lm_name, args.ref)
elif args.representation == 'Sentence-Empath':
    if args.num_centroids == 1:
        mid = 'sentence_empath_centroid_{}_{}_ref{}'.format(args.distance, args.lm_name, args.ref)
    else:
        mid = 'sentence_empath_centroid_{}_{}_{}_ref{}'.format(args.num_centroids, args.distance, args.lm_name, args.ref)
elif args.representation == 'word2vec-LIWC':
    mid = 'word2vec_liwc_counts'
elif args.representation == 'word2vec-Empath':
    mid = 'word2vec_empath_counts'


scaler = StandardScaler()

test_df = pd.read_csv('{}{}/test_{}.csv'.format(args.data_dir, args.dataset, mid), index_col=0)


if args.dataset == 'MELD' and args.representation == 'Empath':
    dev_df = pd.read_csv('{}{}/dev_empath_counts.csv'.format(args.data_dir, args.dataset), index_col=0)
elif args.dataset == 'MELD' and args.representation == 'LIWC':
    dev_df = pd.read_csv('{}{}/dev_liwc_counts.csv'.format(args.data_dir, args.dataset), index_col=0)
    if args.target == 'Emotion':
        args.target = 'emotion'
    if args.target == 'Sentiment':
        args.target = 'sentiment'
elif args.dataset == 'MELD' and args.representation == 'word2vec-LIWC':
    dev_df = pd.read_csv('{}{}/dev_word2vec_liwc_counts.csv'.format(args.data_dir, args.dataset), index_col=0)
elif args.dataset == 'MELD' and args.representation == 'word2vec-Empath':
    dev_df = pd.read_csv('{}{}/dev_word2vec_empath_counts.csv'.format(args.data_dir, args.dataset), index_col=0)
else:
    dev_df = pd.read_csv('{}{}/train_{}.csv'.format(args.data_dir, args.dataset, mid), index_col=0)

if args.dataset == 'MELD':
    if args.representation == 'LIWC':
        X_dev = dev_df.drop(['emotion', 'sentiment'], axis=1)
        X_test = test_df.drop(['emotion', 'sentiment'], axis=1)
    else:
        X_dev = dev_df.drop(['Emotion', 'Sentiment'], axis=1)
        X_test = test_df.drop(['Emotion', 'Sentiment'], axis=1)

else:
    X_dev = dev_df.drop([args.target], axis=1)
    X_test = test_df.drop([args.target], axis=1)

if args.representation == 'Sentence-LIWC' or args.representation == 'LIWC' or args.representation == 'word2vec-LIWC':
    parse, category_names = liwc.load_token_parser('../liwc_dict/LIWC2015_English_Flat.dic')
    category_names = category_names[21:]
    X_dev = X_dev[category_names].sort_index(axis=1)
    X_test = X_test[category_names].sort_index(axis=1)

elif args.representation == 'Sentence-Empath' or args.representation == 'Empath' or args.representation == 'word2vec-Empath':
    lexicon = Empath()
    category_names = list(lexicon.cats.keys())
    X_dev = X_dev[category_names].sort_index(axis=1)
    X_test = X_test[category_names].sort_index(axis=1)

y_dev = dev_df[args.target]
y_test = test_df[args.target]

if args.row_norm:
    X_dev = scale(X_dev, axis=1, with_mean=True, with_std=True, copy=True)
    X_test = scale(X_test, axis=1, with_mean=True, with_std=True, copy=True)

if args.task == 'classification':
    skf = StratifiedKFold(n_splits=5)
    if args.model == 'SVM':
        clf = SVC()
    elif args.model == 'LR':
        clf = LogisticRegression(random_state=20211213, multi_class='ovr',
                                solver='saga', max_iter=5000)

    base_accs = []
    test_accs = []

    for train_index, test_index in tqdm(skf.split(X_dev, y_dev)):

        if not args.row_norm:
            X_train_cv, X_test_cv = X_dev.iloc[train_index], X_dev.iloc[test_index]
        else:
            X_train_cv, X_test_cv = X_dev[train_index], X_dev[test_index]

        y_train_cv, y_test_cv = y_dev.values[train_index], y_dev.values[test_index]

        base_accs.append(np.mean(y_test_cv == stats.mode(y_test_cv)[0][0]))
        scaler.fit(X_train_cv)
        clf.fit(scaler.transform(X_train_cv), y_train_cv)
        test_accs.append(clf.score(scaler.transform(X_test_cv), y_test_cv))

    print('Majority baseline 5-fold CV accuracy:', np.mean(base_accs))
    print('{} 5-fold CV accuracy:'.format(args.model), np.mean(test_accs))

if args.task == 'regression':
    kf = KFold(n_splits=5)
    if args.model == 'SVM':
        reg = SVR()
    elif args.model == 'LR':
        reg = LinearRegression()
    
    base_r2 = []
    test_r2 = []
    base_rmse = []
    test_rmse = []

    for train_index, test_index in tqdm(kf.split(X_dev, y_dev)):
        
        if not args.row_norm:
            X_train_cv, X_test_cv = X_dev.iloc[train_index], X_dev.iloc[test_index]
        else:
            X_train_cv, X_test_cv = X_dev[train_index], X_dev[test_index]
        
        y_train_cv, y_test_cv = y_dev.values[train_index], y_dev.values[test_index]

        base_r2.append(r2_score(y_test_cv, [np.mean(y_train_cv)]*len(y_test_cv)))
        base_rmse.append(np.sqrt(mean_squared_error(y_test_cv, [np.mean(y_train_cv)]*len(y_test_cv))))

        scaler.fit(X_train_cv)
        reg.fit(scaler.transform(X_train_cv), y_train_cv)
        test_r2.append(reg.score(scaler.transform(X_test_cv), y_test_cv))
        test_rmse.append(np.sqrt(mean_squared_error(y_test_cv,
                                                reg.predict(scaler.transform(X_test_cv)))))

    print('Majority baseline 5-fold CV R^2:', np.mean(base_r2))
    print('{} 5-fold CV R^2:'.format(args.model), np.mean(test_r2))
    
    print('Majority baseline RMSE:', np.mean(base_rmse))
    print('{} 5-fold CV RMSE:'.format(args.model), np.mean(test_rmse))

if args.task == 'classification':
    print('Majority baseline test accuracy:', np.mean(y_test == stats.mode(y_dev)[0][0]))
    
    if args.model == 'SVM':
        clf = SVC()
    elif args.model == 'LR':
        clf = LogisticRegression(random_state=20211213, multi_class='ovr',
                                solver='saga', max_iter=5000)
    scaler.fit(X_dev)
    clf.fit(scaler.transform(X_dev), y_dev)
    print('{} test accuracy:'.format(args.model), clf.score(scaler.transform(X_test), y_test))

if args.task == 'regression':
    if args.model == 'SVM':
        reg = SVR()
    elif args.model == 'LR':
        reg = LinearRegression()
    scaler.fit(X_dev)
    reg.fit(scaler.transform(X_dev), y_dev)
    
    print('Majority baseline test R^2:', r2_score(y_test, [np.mean(y_dev)]*len(y_test)))
    print('{} test R^2:'.format(args.model), reg.score(scaler.transform(X_test), y_test))
    
    print('Majority baseline test RMSE:', 
          np.sqrt(mean_squared_error(y_test, [np.mean(y_dev)]*len(y_test))))
    print('{} test RMSE:'.format(args.model), 
          np.sqrt(mean_squared_error(y_test, reg.predict(scaler.transform(X_test)))))