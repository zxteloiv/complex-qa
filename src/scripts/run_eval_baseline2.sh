#!/usr/bin/env bash

RESULT_ROOT=../../result_evaluation

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
    TEST_FILE=../../data/${DATASET_TYPE}_rank/${DATASET_TYPE}_rank.five_hyp.test.jsonl
    for ((i=1; i<501; i++));
    do
        if [ ! -f "$MODEL_PREFIX/model_state_$i.th" ]; then
            continue
        fi
        CUDA_VISIBLE_DEVICES=0 nohup python -u $EXECUTABLE \
            -p $HPARAM_SET \
            --translator ${DATASET_TYPE}_rank \
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
snapshot=../../snapshots/django_five_hyp/reranking_baseline2/20200521-220647-neore2-dropout
eval_template django 06base2-dropout neore2 "$snapshot" django_neo_five_dropout baseline2_five_hyp.py

snapshot=../../snapshots/django_five_hyp/baseline2_giant/20200521-220742-giant-dropout
eval_template django 06base2-dropout giant "$snapshot" django_giant_five_dropout baseline2_five_hyp_giant.py

snapshot=../../snapshots/atis_five_hyp/reranking_baseline2/20200521-220647-neore2-dropout
eval_template atis 06base2-dropout neore2 "$snapshot" atis_neo_five_dropout baseline2_five_hyp.py

snapshot=../../snapshots/atis_five_hyp/baseline2_giant/20200521-220742-giant-dropout
eval_template atis 06base2-dropout giant "$snapshot" atis_giant_five_dropout baseline2_five_hyp_giant.py

