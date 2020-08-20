#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 nohup python -u meta_ranker.py \
    -p django_nl_bert \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo qmaml-nlbert \
    &> log.django.nlbert.qmaml &

CUDA_VISIBLE_DEVICES=1 nohup python -u meta_ranker.py \
    -p django_nl_ngram \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo qmaml-nlngram \
    &> log.django.nlngram.qmaml &

CUDA_VISIBLE_DEVICES=2 nohup python -u meta_ranker.py \
    -p django_lf_ngram \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo qmaml-lfngram \
    &> log.django.lfngram.qmaml &

CUDA_VISIBLE_DEVICES=3 nohup python -u meta_ranker.py \
    -p django_lf_ted \
    --translator django_rank_char \
    --dataset django_five_hyp \
    --device 0 \
    --vocab-dump django_vocab_15 \
    --memo qmaml-lfted \
    &> log.django.lfted.qmaml &

#CUDA_VISIBLE_DEVICES=0 nohup python -u meta_ranker.py \
#    -p atis_nl_bert \
#    --translator atis_rank_char \
#    --dataset atis_five_hyp \
#    --device 0 \
#    --vocab-dump atis_vocab \
#    --memo qmaml-nlbert \
#    &> log.atis.nlbert.qmaml &
#
#CUDA_VISIBLE_DEVICES=1 nohup python -u meta_ranker.py \
#    -p atis_nl_ngram \
#    --translator atis_rank_char \
#    --dataset atis_five_hyp \
#    --device 0 \
#    --vocab-dump atis_vocab \
#    --memo qmaml-nlngram \
#    &> log.atis.nlngram.qmaml &
#
#CUDA_VISIBLE_DEVICES=2 nohup python -u meta_ranker.py \
#    -p atis_lf_ngram \
#    --translator atis_rank_char \
#    --dataset atis_five_hyp \
#    --device 0 \
#    --vocab-dump atis_vocab \
#    --memo qmaml-lfngram \
#    &> log.atis.lfngram.qmaml &
#
#CUDA_VISIBLE_DEVICES=3 nohup python -u meta_ranker.py \
#    -p atis_lf_ted \
#    --translator atis_rank_char \
#    --dataset atis_five_hyp \
#    --device 0 \
#    --vocab-dump atis_vocab \
#    --memo qmaml-lfted \
#    &> log.atis.lfted.qmaml &


