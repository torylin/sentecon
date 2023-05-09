import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from tqdm import tqdm
from scipy import stats
import argparse
import pdb
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data/', help='data directory')
    parser.add_argument('--dataset', help='task dataset')
    parser.add_argument('--lm', default='all-MiniLM-L6-v2', help='language model for sentence embeddings')
    parser.add_argument('--target', help='name of target in task dataset')
    parser.add_argument('--task', default='classification', help='classification or regression?')
    parser.add_argument('--model', help='SVM or LR?')
    parser.add_argument('--plot-preds', action='store_true', help='plot predictions against labels')
    parser.add_argument('--scale', action='store_true', help='column normalization?')

    args = parser.parse_args()

    return args

scaler = StandardScaler()
args = get_args()

if args.dataset == 'MELD':
    dev_sent = pd.read_csv('{}{}/dev.csv'.format(args.data_dir, args.dataset))
    dev_embeds = np.load('{}{}/dev_embeds_{}.npy'.format(
        args.data_dir, args.dataset, args.lm))
else:
    dev_sent = pd.read_csv('{}{}/train.csv'.format(args.data_dir, args.dataset))
    dev_embeds = np.load('{}{}/train_embeds_{}.npy'.format(
        args.data_dir, args.dataset, args.lm))

test_sent = pd.read_csv('{}{}/test.csv'.format(args.data_dir, args.dataset))
test_embeds = np.load('{}{}/test_embeds_{}.npy'.format(
    args.data_dir, args.dataset, args.lm))
y_dev = dev_sent[args.target]
y_test = test_sent[args.target]

if args.task == 'classification':
    skf = StratifiedKFold(n_splits=5)
    if args.model == 'SVM':
        clf = SVC()
    elif args.model == 'LR':
        clf = LogisticRegression(random_state=20211213, multi_class='ovr',
                                solver='saga', max_iter=5000)
    base_accs = []
    test_accs = []

    for train_index, test_index in tqdm(skf.split(dev_embeds, y_dev)):
        X_train_cv, X_test_cv = dev_embeds[train_index], dev_embeds[test_index]
        y_train_cv, y_test_cv = y_dev.values[train_index], y_dev.values[test_index]

        base_accs.append(np.mean(y_test_cv == stats.mode(y_test_cv)[0][0]))
        if args.scale:
            clf.fit(scaler.fit_transform(X_train_cv), y_train_cv)
            test_accs.append(clf.score(scaler.transform(X_test_cv), y_test_cv))
        else:
            clf.fit(X_train_cv, y_train_cv)
            test_accs.append(clf.score(X_test_cv, y_test_cv))

    print('Majority baseline 5-fold CV accuracy:', np.mean(base_accs))
    print('{} 5-fold CV accuracy:'.format(args.model), np.mean(test_accs))

    print('Majority baseline test accuracy:', np.mean(y_test == stats.mode(y_dev)[0][0]))
    if args.model == 'SVM':
        clf = SVC()
    elif args.model == 'LR':
        clf = LogisticRegression(random_state=20211213, multi_class='ovr',
                                solver='saga', max_iter=5000)

    if args.scale:
        clf.fit(scaler.fit_transform(dev_embeds), y_dev)
        print('{} test accuracy:'.format(args.model), clf.score(scaler.transform(test_embeds), y_test))
    else:
        clf.fit(dev_embeds, y_dev)
        print('{} test accuracy:'.format(args.model), clf.score(test_embeds, y_test))


elif args.task == 'regression':
    kf = KFold(n_splits=5)

    if args.model == 'SVM':
        reg = SVR()
    elif args.model == 'LR':
        reg = LinearRegression()

    base_r2 = []
    test_r2 = []
    base_rmse = []
    test_rmse = []
    i = 1

    for train_index, test_index in tqdm(kf.split(dev_embeds, y_dev)):
        X_train_cv, X_test_cv = dev_embeds[train_index], dev_embeds[test_index]
        y_train_cv, y_test_cv = y_dev.values[train_index], y_dev.values[test_index]
        
        base_r2.append(r2_score(y_test_cv, [np.mean(y_train_cv)]*len(y_test_cv)))
        base_rmse.append(np.sqrt(mean_squared_error(y_test_cv, [np.mean(y_train_cv)]*len(y_test_cv))))

        if args.scale:
            scaler.fit(X_train_cv)
            reg.fit(scaler.transform(X_train_cv), y_train_cv)
            y_pred_cv = reg.predict(scaler.transform(X_test_cv))
        else:
            reg.fit(X_train_cv, y_train_cv)
            y_pred_cv = reg.predict(X_test_cv)

        test_r2.append(r2_score(y_test_cv, y_pred_cv))
        test_rmse.append(np.sqrt(mean_squared_error(y_test_cv, y_pred_cv)))

        if args.plot_preds:
            plt.scatter(y_test_cv, y_pred_cv)
            plt.xlabel('label')
            plt.ylabel('predicted')
            plt.title('k-fold split {}'.format(i))
            plt.show()
            plt.savefig('../plots/{}_{}_{}_bert_preds_kfold_{}.png'.format(args.dataset, args.model, args.lm, i))
            plt.clf()

        i += 1

    print('Majority baseline 5-fold CV R^2:', np.mean(base_r2))
    print('{} 5-fold CV R^2:'.format(args.model), np.mean(test_r2))
    
    print('Majority baseline RMSE:', np.mean(base_rmse))
    print('{} 5-fold CV RMSE:'.format(args.model), np.mean(test_rmse))

    if args.model == 'SVM':
        reg = SVR()
    elif args.model == 'LR':
        reg = LinearRegression()

    if args.scale:
        scaler.fit(dev_embeds)
        reg.fit(scaler.transform(dev_embeds), y_dev)
        y_pred = reg.predict(scaler.transform(test_embeds))
    else:
        reg.fit(dev_embeds, y_dev)
        y_pred = reg.predict(test_embeds)

    print('Majority baseline test R^2:', r2_score(y_test, [np.mean(y_dev)]*len(y_test)))
    print('{} test R^2:'.format(args.model), r2_score(y_test, y_pred))

    print('Majority baseline test RMSE:', np.sqrt(mean_squared_error(y_test, [np.mean(y_dev)]*len(y_test))))
    print('{} test RMSE:'.format(args.model), np.sqrt(mean_squared_error(y_test, y_pred)))

    if args.plot_preds:
        plt.scatter(y_test, y_pred)
        plt.xlabel('label')
        plt.ylabel('predicted')
        plt.title('test set')
        plt.show()
        plt.savefig('../plots/{}_{}_{}_bert_preds_test.png'.format(args.dataset, args.model, args.lm))