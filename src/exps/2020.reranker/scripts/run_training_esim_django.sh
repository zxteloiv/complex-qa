#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 nohup python -u meta_ranker_esim.py \
    -p django_nl_bert \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo esim-nlbert-rank \
    &> log.django.nlbert.esim.rank &

CUDA_VISIBLE_DEVICES=1 nohup python -u meta_ranker_esim.py \
    -p django_nl_ngram \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo esim-nlngram-rank \
    &> log.django.nlngram.esim.rank &

CUDA_VISIBLE_DEVICES=2 nohup python -u meta_ranker_esim.py \
    -p django_lf_ngram \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo esim-lfngram-rank \
    &> log.django.lfngram.esim.rank &

CUDA_VISIBLE_DEVICES=3 nohup python -u meta_ranker_esim.py \
    -p django_lf_ted \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo esim-lfted-rank \
    &> log.django.lfted.esim.rank &

