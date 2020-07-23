#!/bin/bash 
# Start training on two models, and two datasets

CUDA_VISIBLE_DEVICES=0 nohup python -u baseline1.py \
    -p django_neo_none \
    --translator django_rank \
    --dataset django_none_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo neore2 \
    &> log.django.neo.none &

CUDA_VISIBLE_DEVICES=1 nohup python -u baseline1.py \
    -p atis_neo_none \
    --translator atis_rank \
    --dataset atis_none_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo neore2 \
    &> log.atis.neo.none &
