#!/bin/bash

cd /home/victorialin/Documents/2021-2022/Research/sentence_liwc/scripts/

num_centroids=(1 2 3 4)
text_names=(Utterance Utterance review)
targets=(Emotion Sentiment sentiment)
datasets=(MELD MELD LargeMovieReviewDataset)
ref_corps=(MELD MELD LargeMovieReviewDataset)
ft_models=(all-mpnet-base-v2-MELD-finetuned-emotion all-mpnet-base-v2-MELD-finetuned-sentiment all-mpnet-base-v2-IMDB-finetuned)
# text_names=(utterance sentence)
# targets=(sentiment label)
# datasets=(MOSI SST)
# ft_models=(all-mpnet-base-v2-MOSI-finetuned all-mpnet-base-v2-SST-finetuned)
tasks=(classification classification classification)

for i in ${!datasets[@]}
do
    for centroid in ${num_centroids[@]}
    do
        printf "Dataset: ${datasets[i]} \t Target: ${targets[i]} \t Model: Pretrained MPNet \t Centroids: $centroid \t Lexicon: LIWC \t Ref corp: None \n"
        python sentence_category_prediction.py --ref LIWCVocab --target ${targets[i]} --text-name ${text_names[i]} \
            --lexicon LIWC --dataset ${datasets[i]} --lm-name all-mpnet-base-v2 --num-centroids $centroid
        python evaluation.py --dataset ${datasets[i]} --lm-name all-mpnet-base-v2 --text-name ${text_names[i]} \
            --representation Sentence-LIWC --ref LIWCVocab --target ${targets[i]} --task ${tasks[i]} --model LR \
            --num-centroids $centroid --row-norm
        printf "\n"
    done
    printf "\n"
    for centroid in ${num_centroids[@]}
    do
        printf "Dataset: ${datasets[i]} \t Model: Fine-tuned MPNet \t Centroids: $centroid \t Lexicon: LIWC \t Ref corp: None \n"
        python sentence_category_prediction.py --ref LIWCVocab --target ${targets[i]} --text-name ${text_names[i]} \
            --lexicon LIWC --dataset ${datasets[i]} --lm-name ${ft_models[i]} --num-centroids $centroid
        python evaluation.py --dataset ${datasets[i]} --lm-name ${ft_models[i]} --text-name ${text_names[i]} \
            --representation Sentence-LIWC --ref LIWCVocab --target ${targets[i]} --task ${tasks[i]} --model LR \
            --num-centroids $centroid --row-norm
        printf "\n"
    done
    for centroid in ${num_centroids[@]}
    do
        printf "Dataset: ${datasets[i]} \t Model: Pretrained MPNet \t Centroids: $centroid \t Lexicon: LIWC \t Ref corp: ${ref_corps[i]} \n"
        python sentence_category_prediction.py --ref ${ref_corps[i]} --target ${targets[i]} --text-name ${text_names[i]} \
            --lexicon LIWC --dataset ${datasets[i]} --lm-name all-mpnet-base-v2 --num-centroids $centroid
        python evaluation.py --dataset ${datasets[i]} --lm-name all-mpnet-base-v2 --text-name ${text_names[i]} \
            --representation Sentence-LIWC --ref ${ref_corps[i]} --target ${targets[i]} --task ${tasks[i]} --model LR \
            --num-centroids $centroid --row-norm
        printf "\n"
    done
    printf "\n"
    for centroid in ${num_centroids[@]}
    do
        printf "Dataset: ${datasets[i]} \t Model: Fine-tuned MPNet \t Centroids: $centroid \t Lexicon: LIWC \t Ref corp: ${ref_corps[i]} \n"
        python sentence_category_prediction.py --ref ${ref_corps[i]} --target ${targets[i]} --text-name ${text_names[i]} \
            --lexicon LIWC --dataset ${datasets[i]} --lm-name ${ft_models[i]} --num-centroids $centroid
        python evaluation.py --dataset ${datasets[i]} --lm-name ${ft_models[i]} --text-name ${text_names[i]} \
            --representation Sentence-LIWC --ref ${ref_corps[i]} --target ${targets[i]} --task ${tasks[i]} --model LR \
            --num-centroids $centroid --row-norm
        printf "\n"
    done
done