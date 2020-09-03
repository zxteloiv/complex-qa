#!/usr/bin/env bash

RESULT_ROOT=~/deploy/complex_qa/result_evaluation

function eval_template () {
    if [ "$1" == "" ]; then
        return 1;
    fi

    DATASET_TYPE=$1 # atis or django
    STAT_DIRNAME=$2 # any folder name for stat outputs
    MODEL_NAME=$3
    MODEL_PREFIX=$4 # model directory
    HPARAM_SET=$5 # any parameter sets
    EXECUTABLE=$6

    STAT_DIR="$RESULT_ROOT/$STAT_DIRNAME"
    DECODE_DIR=${STAT_DIR}/decodes

    mkdir -p ${DECODE_DIR}
    STAT_OUTFILE=${STAT_DIR}/${DATASET_TYPE}.${MODEL_NAME}.top
    DECODE_PREFIX=${DECODE_DIR}/decode.${DATASET_TYPE}.${MODEL_NAME}
    TEST_FILE=~/deploy/complex_qa/data/${DATASET_TYPE}_rank/${DATASET_TYPE}_rank.five_hyp.test.jsonl
    for ((i=1; i<501; i++));
    do
        if [ ! -f "$MODEL_PREFIX/model_state_$i.th" ]; then
            continue
        fi
        CUDA_VISIBLE_DEVICES=0 nohup python -u $EXECUTABLE \
            -p $HPARAM_SET \
            --translator ${DATASET_TYPE}_rank_char \
            --dataset ${DATASET_TYPE}_five_hyp \
            --device 0 \
            --vocab-dump $MODEL_PREFIX/vocab \
            --test $MODEL_PREFIX/model_state_$i.th \
            > $DECODE_PREFIX.$i;

        for ((j=2; j<6; j++));
        do
            echo $i >> ${STAT_OUTFILE}.$j;
            python evaluator.py $TEST_FILE $DECODE_PREFIX.$i --max-hyp-rank $j >> ${STAT_OUTFILE}.$j;
        done
    done;
}

# parameters must include:
#   on which dataset, with which model (name and dump parameters), in which hyper-parameters
#
# $1 atis or django, dataset name
# $2 stat output folder name
# $3 model name, used in output files
# $4 snapshot prefix, where the trained model are saved
# $5 hyperparameter used for tests
# $6 python executable script name, based on trialbot
snapshot=~/deploy/complex_qa/snapshots/django_five_hyp/trans_giant_base
eval_template django 25tgiant-group-v4 lfted "$snapshot" django_lf_ted transfer_giant.py
eval_template django 25tgiant-group-v4 lfngram "$snapshot" django_lf_ngram transfer_giant.py
eval_template django 25tgiant-group-v4 nlbert "$snapshot" django_nl_bert transfer_giant.py
eval_template django 25tgiant-group-v4 nlngram "$snapshot" django_nl_ngram transfer_giant.py

snapshot=~/deploy/complex_qa/snapshots/atis_five_hyp/trans_giant_base
eval_template atis 25tgiant-group-v4 lfted "$snapshot" atis_lf_ted transfer_giant.py
eval_template atis 25tgiant-group-v4 lfngram "$snapshot" atis_lf_ngram transfer_giant.py
eval_template atis 25tgiant-group-v4 nlbert "$snapshot" atis_nl_bert transfer_giant.py
eval_template atis 25tgiant-group-v4 nlngram "$snapshot" atis_nl_ngram transfer_giant.py

