#!/bin/bash 
# Start training on two models, and two datasets

# ../../atis_none_hyp/reranking_baseline1/20200518-224257-neore2/model_state_200.th
# ../../django_none_hyp/reranking_baseline1/20200518-224257-neore2/model_state_60.th

RESULT_ROOT=../../result_evaluation

function eval_django_neo () {
    if [ "$1" == "" ]; then
        return 1;
    fi

    MODEL_PREFIX=../../snapshots/django_none_hyp/reranking_baseline1/20200518-224257-neore2
    STAT_DIR="$RESULT_ROOT/$1"
    DECODE_DIR=${STAT_DIR}/decodes

    mkdir -p ${DECODE_DIR}
    STAT_OUTFILE=${STAT_DIR}/django.neore2.top
    DECODE_PREFIX=${DECODE_DIR}/decode.django.neore2
    TEST_FILE=../../data/django_rank/django_rank.five_hyp.test.jsonl
    for ((i=1; i<501; i++));
    do
        if [ ! -f "$MODEL_PREFIX/model_state_$i.th" ]; then
            continue
        fi
        CUDA_VISIBLE_DEVICES=0 nohup python -u baseline1.py \
            -p django_neo_none \
            --translator django_rank \
            --dataset django_five_hyp \
            --device 0 \
            --vocab-dump django_vocab_15 \
            --test $MODEL_PREFIX/model_state_$i.th \
            > $DECODE_PREFIX.$i;

        for ((j=2; j<6; j++));
        do
            echo $i >> ${STAT_OUTFILE}.$j;
            python evaluator.py $TEST_FILE $DECODE_PREFIX.$i --max-hyp-rank $j >> ${STAT_OUTFILE}.$j;
        done
    done;
}

function eval_atis_neo () {
    if [ "$1" == "" ]; then
        return 1;
    fi

    MODEL_PREFIX=../../snapshots/atis_none_hyp/reranking_baseline1/20200518-224257-neore2

    STAT_DIR="$RESULT_ROOT/$1"
    DECODE_DIR=${STAT_DIR}/decodes
    mkdir -p ${DECODE_DIR}
    STAT_OUTFILE=${STAT_DIR}/atis.neore2.top
    DECODE_PREFIX=${DECODE_DIR}/decode.atis.neore2
    TEST_FILE=../../data/atis_rank/atis_rank.five_hyp.test.jsonl
    for ((i=1; i<501; i++));
    do
        if [ ! -f "$MODEL_PREFIX/model_state_$i.th" ]; then
            continue
        fi
        CUDA_VISIBLE_DEVICES=0 nohup python -u baseline1.py \
            -p atis_neo_none \
            --translator atis_rank \
            --dataset atis_five_hyp \
            --device 0 \
            --vocab-dump atis_vocab_15 \
            --test $MODEL_PREFIX/model_state_$i.th \
            > $DECODE_PREFIX.$i;

        for ((j=2; j<6; j++));
        do
            echo $i >> ${STAT_OUTFILE}.$j;
            python evaluator.py $TEST_FILE $DECODE_PREFIX.$i --max-hyp-rank $j >> ${STAT_OUTFILE}.$j;
        done
    done;
}

eval_django_neo 05base1 &> log.eval.django.neore2;
eval_atis_neo 05base1 &> log.eval.atis.neore2;
