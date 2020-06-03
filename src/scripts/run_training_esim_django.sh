#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 nohup python -u meta_ranker_esim.py \
    -p django_nl_bert \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo esim-nlbert-cont10 \
    ../../snapshots/django_five_hyp/meta_ranker/20200603-172343-esim-nlbert/model_0_state_10.th \
    ../../snapshots/django_five_hyp/meta_ranker/20200603-172343-esim-nlbert/model_1_state_10.th \
    &> log.django.nlbert.esim &

CUDA_VISIBLE_DEVICES=1 nohup python -u meta_ranker_esim.py \
    -p django_nl_ngram \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo esim-nlngram-cont10 \
    ../../snapshots/django_five_hyp/meta_ranker/20200603-172342-esim-nlngram/model_0_state_10.th \
    ../../snapshots/django_five_hyp/meta_ranker/20200603-172342-esim-nlngram/model_1_state_10.th \
    &> log.django.nlngram.esim.cont10 &

CUDA_VISIBLE_DEVICES=2 nohup python -u meta_ranker_esim.py \
    -p django_lf_ngram \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo esim-lfngram-cont10 \
    ../../snapshots/django_five_hyp/meta_ranker/20200603-172343-esim-lfngram/model_0_state_10.th \
    ../../snapshots/django_five_hyp/meta_ranker/20200603-172343-esim-lfngram/model_1_state_10.th \
    &> log.django.lfngram.esim.cont10 &

CUDA_VISIBLE_DEVICES=3 nohup python -u meta_ranker_esim.py \
    -p django_lf_ted \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo esim-lfted-cont10 \
    ../../snapshots/django_five_hyp/meta_ranker/20200603-172343-esim-lfted/model_0_state_10.th \
    ../../snapshots/django_five_hyp/meta_ranker/20200603-172343-esim-lfted/model_1_state_10.th \
    &> log.django.lfted.esim.cont10 &

