#!/bin/bash
# Start training on two models, and two datasets

CUDA_VISIBLE_DEVICES=0 nohup python -u baseline2_five_hyp.py \
    -p django_neo_five_dropout \
    --translator django_rank \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo neore2-dropout \
    &> log.django.neore2.five &

CUDA_VISIBLE_DEVICES=1 nohup python -u baseline2_five_hyp.py \
    -p atis_neo_five_dropout \
    --translator atis_rank \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo neore2-dropout \
    &> log.atis.neore2.five &

