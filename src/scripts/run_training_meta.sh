#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 nohup python -u meta_ranker.py \
    -p atis_lf_ted \
    --translator atis_rank_char \
    --dataset atis_five_hyp \
    --device 0 \
    --vocab-dump atis_vocab \
    --memo metaranker \
    &> log.atis.meta.lfted &

CUDA_VISIBLE_DEVICES=0 nohup python -u meta_ranker.py \
    -p django_lf_ted \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab \
    --memo pretrained-chgiant \
    ../../snapshots/django_five_hyp/baseline2_giant/20200527-133959-chgiant/model_state_31.th \
    &> log.atis.lfted.pretrained &

