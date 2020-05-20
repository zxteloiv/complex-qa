#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 nohup python -u baseline2_five_hyp_giant.py \
    -p django_giant_five \
    --translator django_rank \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo giant \
    &> log.django.giant.five &

CUDA_VISIBLE_DEVICES=1 nohup python -u baseline2_five_hyp_giant.py \
    -p atis_giant_five \
    --translator atis_rank \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo giant \
    &> log.atis.giant.five &

