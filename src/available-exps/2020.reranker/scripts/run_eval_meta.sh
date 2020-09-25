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
        if [ ! -f "$MODEL_PREFIX/model_0_state_$i.th" ]; then
            continue
        fi
        CUDA_VISIBLE_DEVICES=1 nohup python -u $EXECUTABLE \
            -p $HPARAM_SET \
            --translator ${DATASET_TYPE}_rank_char \
            --dataset ${DATASET_TYPE}_five_hyp \
            --device 0 \
            --vocab-dump $MODEL_PREFIX/vocab \
            --test \
            $MODEL_PREFIX/model_0_state_$i.th \
            $MODEL_PREFIX/model_1_state_$i.th \
            > $DECODE_PREFIX.$i;

        for ((j=2; j<6; j++));
        do
            echo $i >> ${STAT_OUTFILE}.$j;
            python evaluator.py $TEST_FILE $DECODE_PREFIX.$i --max-hyp-rank $j >> ${STAT_OUTFILE}.$j;
        done
    done;
}

snapshot=../../snapshots/atis_five_hyp/meta_ranker/20200814-123223-qmaml-lfngram
eval_template atis 23qmaml-deep-chgiant-rank lfngram "$snapshot" atis_lf_ngram meta_ranker.py

snapshot=../../snapshots/atis_five_hyp/meta_ranker/20200814-123228-qmaml-lfted
eval_template atis 23qmaml-deep-chgiant-rank lfted "$snapshot" atis_lf_ted meta_ranker.py

snapshot=../../snapshots/atis_five_hyp/meta_ranker/20200814-123210-qmaml-nlbert
eval_template atis 23qmaml-deep-chgiant-rank nlbert "$snapshot" atis_nl_bert meta_ranker.py

snapshot=../../snapshots/atis_five_hyp/meta_ranker/20200814-123216-qmaml-nlngram
eval_template atis 23qmaml-deep-chgiant-rank nlngram "$snapshot" atis_nl_ngram meta_ranker.py

snapshot=../../snapshots/django_five_hyp/meta_ranker/20200814-020456-qmaml-lfngram
eval_template django 23qmaml-deep-chgiant-rank lfngram "$snapshot" django_lf_ngram meta_ranker.py

snapshot=../../snapshots/django_five_hyp/meta_ranker/20200814-020504-qmaml-lfted
eval_template django 23qmaml-deep-chgiant-rank lfted "$snapshot" django_lf_ted meta_ranker.py

snapshot=../../snapshots/django_five_hyp/meta_ranker/20200814-020336-qmaml-nlbert
eval_template django 23qmaml-deep-chgiant-rank nlbert "$snapshot" django_nl_bert meta_ranker.py

snapshot=../../snapshots/django_five_hyp/meta_ranker/20200814-020449-qmaml-nlngram
eval_template django 23qmaml-deep-chgiant-rank nlngram "$snapshot" django_nl_ngram meta_ranker.py

