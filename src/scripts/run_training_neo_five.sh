#!/bin/bash
# Start training on two models, and two datasets

CUDA_VISIBLE_DEVICES=0 nohup python -u baseline2_five_hyp.py \
    -p django_deep_stack \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo chre2 \
    &> log.django.chre2.deep &

CUDA_VISIBLE_DEVICES=1 nohup python -u baseline2_five_hyp.py \
    -p atis_deep_stack \
    --translator atis_rank_char \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo chre2 \
    &> log.atis.chre2.deep &

