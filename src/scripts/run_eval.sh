#!/bin/bash 
# Start training on two models, and two datasets

# ../../snapshots/atis_five_hyp/baseline2_giant/20200116-183245-giant-base/model_state_200.th
# ../../snapshots/atis_five_hyp/reranking_baseline2/20200116-183245-re2-base/model_state_200.th
# ../../snapshots/django_fifteen_hyp/baseline2_giant/20200116-183245-giant-base/model_state_60.th
# ../../snapshots/django_fifteen_hyp/reranking_baseline2/20200116-183245-re2-base/model_state_60.th

function eval_django_neo_re2 () {
    STAT_OUTFILE=django.re2-base.log
    DECODE_PREFIX=/tmp/decode.django.re2-base
    MODEL_PREFIX=../../snapshots/django_fifteen_hyp/reranking_baseline2/20200116-183245-re2-base
    TEST_FILE=../../data/django_rank/django_rank.five_hyp.test.jsonl
    for ((i=1; i<501; i++));
    do
        if [ ! -f "$MODEL_PREFIX/model_state_$i.th" ]; then
            continue
        fi
        echo $i >> $STAT_OUTFILE;
        CUDA_VISIBLE_DEVICES=3 nohup python -u baseline2_five_hyp.py \
            -p django_neo_v2 \
            --translator django_rank \
            --dataset django_five_hyp \
            --device 0 \
            --vocab-dump django_vocab_15 \
            --test $MODEL_PREFIX/model_state_$i.th \
            > $DECODE_PREFIX.$i;
        python evaluator.py $TEST_FILE $DECODE_PREFIX.$i --max-hyp-rank 3 >> $STAT_OUTFILE;
    done;
}

function eval_atis_neo_re2 () {
    STAT_OUTFILE=atis.re2-base.log
    DECODE_PREFIX=/tmp/decode.atis.re2-base
    MODEL_PREFIX=../../snapshots/atis_five_hyp/reranking_baseline2/20200116-183245-re2-base
    TEST_FILE=../../data/atis_rank/atis_rank.five_hyp.test.jsonl
    for ((i=1; i<501; i++));
    do
        if [ ! -f "$MODEL_PREFIX/model_state_$i.th" ]; then
            continue
        fi
        echo $i >> $STAT_OUTFILE;
        CUDA_VISIBLE_DEVICES=1 nohup python -u baseline2_five_hyp.py \
            -p atis_neo \
            --translator atis_rank \
            --dataset atis_five_hyp \
            --device 0 \
            --vocab-dump atis_vocab \
            --test $MODEL_PREFIX/model_state_$i.th \
            > $DECODE_PREFIX.$i;
        python evaluator.py $TEST_FILE $DECODE_PREFIX.$i --max-hyp-rank 3 >> $STAT_OUTFILE;
    done;
}

function eval_django_neo_giant () {
    STAT_OUTFILE=django.giant-base.log
    DECODE_PREFIX=/tmp/decode.django.giant-base
    MODEL_PREFIX=../../snapshots/django_fifteen_hyp/baseline2_giant/20200116-183245-giant-base
    TEST_FILE=../../data/django_rank/django_rank.five_hyp.test.jsonl
    for ((i=1; i<501; i++));
    do
        if [ ! -f "$MODEL_PREFIX/model_state_$i.th" ]; then
            continue
        fi
        echo $i >> $STAT_OUTFILE;
        CUDA_VISIBLE_DEVICES=2 nohup python -u baseline2_five_hyp_giant.py \
            -p django_neo_giant_v2 \
            --translator django_rank \
            --dataset django_five_hyp \
            --device 0 \
            --vocab-dump django_vocab_15 \
            --test $MODEL_PREFIX/model_state_$i.th \
            > $DECODE_PREFIX.$i;
        python evaluator.py $TEST_FILE $DECODE_PREFIX.$i --max-hyp-rank 3 >> $STAT_OUTFILE;
    done;
}

function eval_atis_neo_giant () {
    STAT_OUTFILE=atis.giant-base.log
    DECODE_PREFIX=/tmp/decode.atis.giant-base
    MODEL_PREFIX=../../snapshots/atis_five_hyp/baseline2_giant/20200116-183245-giant-base
    TEST_FILE=../../data/atis_rank/atis_rank.five_hyp.test.jsonl
    for ((i=1; i<501; i++));
    do
        if [ ! -f "$MODEL_PREFIX/model_state_$i.th" ]; then
            continue
        fi
        echo $i >> $STAT_OUTFILE;
        CUDA_VISIBLE_DEVICES=0 nohup python -u baseline2_five_hyp_giant.py \
            -p atis_neo_giant \
            --translator atis_rank \
            --dataset atis_five_hyp \
            --device 0 \
            --vocab-dump atis_vocab \
            --test $MODEL_PREFIX/model_state_$i.th \
            > $DECODE_PREFIX.$i;
        python evaluator.py $TEST_FILE $DECODE_PREFIX.$i --max-hyp-rank 3 >> $STAT_OUTFILE;
    done;
}

eval_django_neo_giant &> log.eval.django.giant-base;
eval_django_neo_re2 &> log.eval.django.re2-base;
eval_atis_neo_giant &> log.eval.atis.giant-base;
eval_atis_neo_re2 &> log.eval.atis.re2-base;
