#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 nohup python -u meta_ranker.py \
    -p django_nl_bert \
    --translator django_rank_char \
    --dataset django_fifteen_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    ../../snapshots/django_five_hyp/baseline2_giant/20200527-133959-chgiant/model_state_31.th \
    --memo qmaml-nlbert-15 \
    &> log.django.nlbert.qmaml &

CUDA_VISIBLE_DEVICES=1 nohup python -u meta_ranker.py \
    -p django_nl_ngram \
    --translator django_rank_char \
    --dataset django_fifteen_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    ../../snapshots/django_five_hyp/baseline2_giant/20200527-133959-chgiant/model_state_31.th \
    --memo qmaml-nlngram-15 \
    &> log.django.nlngram.qmaml &

CUDA_VISIBLE_DEVICES=2 nohup python -u meta_ranker.py \
    -p django_lf_ngram \
    --translator django_rank_char \
    --dataset django_fifteen_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    ../../snapshots/django_five_hyp/baseline2_giant/20200527-133959-chgiant/model_state_31.th \
    --memo qmaml-lfngram-15 \
    &> log.django.lfngram.qmaml &

CUDA_VISIBLE_DEVICES=3 nohup python -u meta_ranker.py \
    -p django_lf_ted \
    --translator django_rank_char \
    --dataset django_fifteen_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    ../../snapshots/django_five_hyp/baseline2_giant/20200527-133959-chgiant/model_state_31.th \
    --memo qmaml-lfted-15-fixed \
    &> log.django.lfted.qmaml.fixed &

