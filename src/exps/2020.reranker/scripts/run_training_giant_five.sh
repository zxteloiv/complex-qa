#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=2 nohup python -u baseline2_five_hyp_giant.py \
    -p django_deep_giant \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo cgiant-radam \
    &> log.django.cgiant-radam &

CUDA_VISIBLE_DEVICES=3 nohup python -u baseline2_five_hyp_giant.py \
    -p atis_deep_giant \
    --translator atis_rank_char \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo cgiant-radam \
    &> log.atis.cgiant-radam &

exit 0;
CUDA_VISIBLE_DEVICES=0 python -u baseline2_five_hyp_giant.py \
    -p django_deep_giant \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo debug