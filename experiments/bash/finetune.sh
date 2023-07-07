#!/bin/bash

cd $1

lm_name=all-mpnet-base-v2
eval_datasets=(MELD MELD SST LargeMovieReviewDataset MOSI)
targets=(Emotion Sentiment label sentiment sentiment)
tasks=(classification classification classification classification regression)
texts=(Utterance Utterance sentence review utterance)
eval_idxs=(${!eval_datasets[@]})
batch_size=2

printf "Fine-tuning $lm_name on MELD for Emotion task \n"
python mlm.py --data-dir ../data/MELD/ --train-csv dev.csv --test-csv test.csv --tokenizer $lm_name \
    --lm $lm_name --text-name Utterance --target Emotion --task classification --batch-size $batch_size \
    --output-dir /home/victorialin/Documents/2021-2022/sentence_liwc/models/${lm_name}_MELD_Emotion_finetune/

printf "Fine-tuning $lm_name on MELD for Sentiment task \n"
python mlm.py --data-dir ../data/MELD/ --train-csv dev.csv --test-csv test.csv --tokenizer $lm_name \
    --lm $lm_name --text-name Utterance --target Sentiment --task classification --batch-size $batch_size \
    --output-dir /home/victorialin/Documents/2021-2022/sentence_liwc/models/${lm_name}_MELD_Sentiment_finetune/

for i in ${eval_idxs[@]:2}
do
    printf "Fine-tuning $lm_name on ${eval_datasets[i]} for ${targets[i+1]} task \n"
    python mlm.py --data-dir ../data/${eval_datasets[i]}/ --train-csv train.csv --test-csv test.csv --batch-size $batch_size \
        --tokenizer $lm_name --lm $lm_name --text-name ${texts[i+1]} --target ${targets[i+1]} --task ${tasks[i]} \
        --output-dir /home/victorialin/Documents/2021-2022/sentence_liwc/models/${lm_name}_${eval_datasets[i]}_${targets[i+1]}_finetune/
done
