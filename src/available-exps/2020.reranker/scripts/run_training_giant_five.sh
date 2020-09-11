#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 nohup python -u baseline2_five_hyp_giant.py \
    -p django_deep_giant \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo complete-giant \
    &> log.django.comp-giant &

CUDA_VISIBLE_DEVICES=1 nohup python -u baseline2_five_hyp_giant.py \
    -p atis_deep_giant \
    --translator atis_rank_char \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo complete-giant \
    &> log.atis.comp-giant &

exit 0;
CUDA_VISIBLE_DEVICES=0 python -u baseline2_five_hyp_giant.py \
    -p django_deep_giant \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo debug