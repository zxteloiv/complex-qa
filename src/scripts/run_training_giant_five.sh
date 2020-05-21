#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 nohup python -u baseline2_five_hyp_giant.py \
    -p django_giant_five_dropout \
    --translator django_rank \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo giant-dropout \
    &> log.django.giant.five &

CUDA_VISIBLE_DEVICES=1 nohup python -u baseline2_five_hyp_giant.py \
    -p atis_giant_five_dropout \
    --translator atis_rank \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo giant-dropout \
    &> log.atis.giant.five &

