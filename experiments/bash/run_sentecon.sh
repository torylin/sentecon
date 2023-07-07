#!/bin/bash

cd $1

lm_name=all-mpnet-base-v2
ref_corps=(LIWCVocab MELD SST LargeMovieReviewDataset MOSI)
ref_csvs=(liwc_vocab.csv train.csv val.csv ref.csv ref.csv)
ref_textnames=(text Utterance sentence review utterance)
val_csvs=(dev.csv val.csv ref.csv ref.csv)
eval_datasets=(MELD MELD SST LargeMovieReviewDataset MOSI)
targets=(Emotion Sentiment label sentiment sentiment)
tasks=(classification classification classification classification regression)
texts=(Utterance Utterance sentence review utterance)
lexicon=LIWC
representation=Sentence-LIWC
eval_idxs=(${!eval_datasets[@]})

# for i in ${!ref_corps[@]}
# do
#     printf "Generating ${ref_corps[i]} reference corpus \t LM: $lm_name \n"
#     python get_reference_corpus.py --dataset ${ref_corps[i]} --text-name ${ref_textnames[i]} \
#         --csv-path ${ref_csvs[i]} --lexicon $lexicon --lm $lm_name --lm-library transformers
# done

# for i in ${eval_idxs[@]:1}
# do
#     printf "Getting embeddings \t Dataset: ${eval_datasets[i]} \t CSV: train.csv \t Text: ${ref_textnames[i]} \t LM: $lm_name \n"
#     python get_sentence_embeds.py --data-dir ../data/${eval_datasets[i]}/ --csv-path train.csv --lm $lm_name \
#         --text-name ${ref_textnames[i]} --lm-library transformers
    
#     printf "Getting embeddings \t Dataset: ${eval_datasets[i]} \t CSV: ${val_csvs[i-1]} \t Text: ${ref_textnames[i]} \t LM: $lm_name \n"
#     python get_sentence_embeds.py --data-dir ../data/${eval_datasets[i]}/ --csv-path ${val_csvs[i-1]} --lm $lm_name \
#         --text-name ${ref_textnames[i]} --lm-library transformers
    
#     printf "Getting embeddings \t Dataset: ${eval_datasets[i]} \t CSV: test.csv \t Text: ${ref_textnames[i]} \t LM: $lm_name \n"
#     python get_sentence_embeds.py --data-dir ../data/${eval_datasets[i]}/ --csv-path test.csv --lm $lm_name \
#         --text-name ${ref_textnames[i]} --lm-library transformers
# done

# for i in ${!eval_datasets[@]}
# do
#     printf "Evaluating Dataset: ${eval_datasets[i]} \t Target: ${targets[i]} \t Representation: Pretrained $lm_name \t Col scaling: No \n"
#     python eval_bert.py --dataset ${eval_datasets[i]} --lm $lm_name --target ${targets[i]} \
#         --task ${tasks[i]} --model LR
#     # printf "Evaluating Dataset: ${eval_datasets[i]} \t Target: ${targets[i]} \t Representation: Pretrained $lm_name \t Col scaling: Yes \n"
#     # python eval_bert.py --dataset ${eval_datasets[i]} --lm $lm_name --target ${targets[i]} \
#     #     --task ${tasks[i]} --model LR --scale
# done

for j in ${eval_idxs[@]:1}
do
    printf "Generating SenteCon representation for ${eval_datasets[j]} \n"
    python sentence_category_prediction.py --ref LIWCVocab --lm $lm_name --target ${targets[j]} \
        --text-name ${texts[j]} --lexicon $lexicon --dataset ${eval_datasets[j]}
    printf "Generating SenteCon+ representation for ${eval_datasets[j]} \n"
    python sentence_category_prediction.py --ref ${eval_datasets[j]} --lm $lm_name --target ${targets[j]} \
        --text-name ${texts[j]} --lexicon $lexicon --dataset ${eval_datasets[j]}
done

for i in ${!eval_datasets[@]}
do
    printf "Evaluating Dataset: ${eval_datasets[i]} \t Target: ${targets[i]} \t Representation: $representation \t Ref corp: None \n"
    python evaluation.py --dataset ${eval_datasets[i]} --lm-name $lm_name --text-name ${texts[i]} \
        --representation $representation --ref LIWCVocab --target ${targets[i]} --task ${tasks[i]} \
        --model LR --row-norm
done

for i in ${!eval_datasets[@]}
do
    printf "Evaluating Dataset: ${eval_datasets[i]} \t Target: ${targets[i]} \t Representation: $representation \t Ref corp: ${eval_datasets[i]} \n"
    python evaluation.py --dataset ${eval_datasets[i]} --lm-name $lm_name --text-name ${texts[i]} \
        --representation $representation --ref ${eval_datasets[i]} --target ${targets[i]} --task ${tasks[i]} \
        --model LR --row-norm
done
