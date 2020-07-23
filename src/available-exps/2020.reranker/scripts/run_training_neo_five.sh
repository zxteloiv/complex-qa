#!/bin/bash
# Start training on two models, and two datasets

CUDA_VISIBLE_DEVICES=0 nohup python -u baseline2_five_hyp.py \
    -p django_deeper \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo chre2-deeper \
    &> log.django.chre2.deeper &

CUDA_VISIBLE_DEVICES=1 nohup python -u baseline2_five_hyp.py \
    -p atis_deeper \
    --translator atis_rank_char \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo chre2-deeper \
    &> log.atis.chre2.deeper &

CUDA_VISIBLE_DEVICES=1 nohup python -u baseline2_five_hyp.py \
    -p django_neo_five_dropout \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo chre2 \
    &> log.django.chre2 &

CUDA_VISIBLE_DEVICES=0 nohup python -u baseline2_five_hyp.py \
    -p atis_neo_five_dropout \
    --translator atis_rank_char \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo chre2 \
    &> log.atis.chre2 &

