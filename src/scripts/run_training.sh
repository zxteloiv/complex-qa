#!/bin/bash 
# Start training on two models, and two datasets

CUDA_VISIBLE_DEVICES=1 nohup python -u baseline2_five_hyp.py \
    -p django_neo_v2 \
    --translator django_rank \
    --dataset django_fifteen_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo re2-base \
    &> log.django.re2-base &

CUDA_VISIBLE_DEVICES=3 nohup python -u baseline2_five_hyp.py \
    -p atis_neo \
    --translator atis_rank \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo fixed-seed \
    &> log.atis.re2-base &

CUDA_VISIBLE_DEVICES=0 nohup python -u baseline2_five_hyp_giant.py \
    -p django_neo_giant_v2 \
    --translator django_rank \
    --dataset django_fifteen_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo giant-base \
    &> log.django.giant-base &

CUDA_VISIBLE_DEVICES=2 nohup python -u baseline2_five_hyp_giant.py \
    -p atis_neo_giant \
    --translator atis_rank \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo giant-base \
    &> log.atis.giant-base &
