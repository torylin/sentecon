from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, load_metric
import pandas as pd
import pdb
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import gc

gc.collect()
torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data/', help='data directory')
    parser.add_argument('--train-csv')
    parser.add_argument('--test-csv')
    parser.add_argument('--tokenizer')
    parser.add_argument('--lm', default='sentence-transformers/all-mpnet-base-v2',
                        help='language model for sentence embeddings')
    parser.add_argument('--text-name', help='name of column containing sentences')
    parser.add_argument('--target', help='name of target in task dataset')
    parser.add_argument('--task', default='classification', help='classification or regression?')
    parser.add_argument('--output-dir')
    parser.add_argument('--lr', default='1e-5', type=float)
    parser.add_argument('--num-epochs', default=5, type=int)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)

    args = parser.parse_args()

    return args

args = get_args()

def preprocess(df):
    return tokenizer(df[args.text_name], truncation=True, padding="max_length")

if args.task == 'classification':

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    metric = load_metric('accuracy')

else:

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        rmse = np.sqrt(mean_squared_error(labels, predictions))
        return {"rmse": rmse}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

df = pd.read_csv('{}{}'.format(args.data_dir, args.train_csv))
train_df, val_df = train_test_split(df, random_state=203948)
test_df = pd.read_csv('{}{}'.format(args.data_dir, args.test_csv))

train_df.rename(columns={args.target: 'label'}, inplace=True)
train_df.dropna(subset=[args.text_name, 'label'], inplace=True)
val_df.rename(columns={args.target: 'label'}, inplace=True)
val_df.dropna(subset=[args.text_name, 'label'], inplace=True)
test_df.rename(columns={args.target: 'label'}, inplace=True)
test_df.dropna(subset=[args.text_name, 'label'], inplace=True)

if args.task == 'classification':
    map_dict = dict(zip(list(sorted(set(train_df['label'].values))), list(range(len(sorted(set(train_df['label'])))))))
    train_df['label'] = [map_dict[v] for v in train_df['label']]
    val_df['label'] = [map_dict[v] for v in val_df['label']]
    test_df['label'] = [map_dict[v] for v in test_df['label']]

if args.task == 'regression':
    model = AutoModelForSequenceClassification.from_pretrained(args.lm, num_labels=1)
    # model = AutoModelForSequenceClassification.from_pretrained('../models/mpnet_mosi_pretrained_1e2/checkpoint-2160')
else:
    model = AutoModelForSequenceClassification.from_pretrained(args.lm,
                                                               num_labels=len(set(train_df['label'])))
    # model = AutoModelForSequenceClassification.from_pretrained('../models/mpnet_meld_sentiment_pretrained_1e2/checkpoint-1248/')

model.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
# pipe = pipeline(task='text-classification', model=model, tokenizer=tokenizer, device=0)

train_dataset = Dataset.from_pandas(train_df[[args.text_name, 'label']])
train_dataset = train_dataset.map(preprocess, batched=True).remove_columns(args.text_name)
val_dataset = Dataset.from_pandas(val_df[[args.text_name, 'label']])
val_dataset = val_dataset.map(preprocess, batched=True).remove_columns(args.text_name)
test_dataset = Dataset.from_pandas(test_df[[args.text_name, 'label']])
test_dataset = test_dataset.map(preprocess, batched=True).remove_columns(args.text_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

if args.freeze:
    for param in model.base_model.parameters():
        param.requires_grad = False

training_args = TrainingArguments(
    learning_rate=args.lr,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    num_train_epochs=args.num_epochs,
    output_dir=args.output_dir,
    logging_strategy='epoch',
    load_best_model_at_end=True,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    seed=525
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

if not args.eval:
    trainer.train()
    trainer.save_model('{}best_model/'.format(args.output_dir))

train_out = trainer.predict(train_dataset)
test_out = trainer.predict(test_dataset)

if args.task == 'classification':
    train_scores = torch.nn.functional.softmax(torch.tensor(train_out.predictions), dim=1)
    train_preds = torch.max(train_scores, dim=1).indices.tolist()

    test_scores = torch.nn.functional.softmax(torch.tensor(test_out.predictions), dim=1)
    test_preds = torch.max(test_scores, dim=1).indices.tolist()

    train_acc = accuracy_score(train_out.label_ids, train_preds)
    test_acc = accuracy_score(test_out.label_ids, test_preds)
    train_f1 = f1_score(train_out.label_ids, train_preds, average='weighted')
    test_f1 = f1_score(test_out.label_ids, test_preds, average='weighted')

    print('Train acc: {}'.format(train_acc))
    print('Test acc: {}'.format(test_acc))
else:
    train_rmse = np.sqrt(mean_squared_error(train_df['label'].values, train_out.predictions))
    test_rmse = np.sqrt(mean_squared_error(test_df['label'].values, test_out.predictions))

    train_r2 = r2_score(train_df['label'].values, train_out.predictions)
    test_r2 = r2_score(test_df['label'].values, test_out.predictions)

    print('Train R^2: {}'.format(train_r2))
    print('Train RMSE: {}'.format(train_rmse))
    print('Test R^2: {}'.format(test_r2))
    print('Test RMSE: {}'.format(test_rmse))