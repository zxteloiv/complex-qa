#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 nohup python -u baseline2_five_hyp_giant.py \
    -p django_giant_five_dropout \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo giant-chgiant \
    &> log.django.chgiant &

CUDA_VISIBLE_DEVICES=1 nohup python -u baseline2_five_hyp_giant.py \
    -p atis_giant_five_dropout \
    --translator atis_rank_char \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo chgiant \
    &> log.atis.chgiant &

