#!/usr/bin/env bash

RESULT_ROOT=../../result_evaluation

function eval_template () {
    DATASET_TYPE=$1 # atis or django
    STAT_DIRNAME=$2 # any folder name for stat outputs
    MODEL_NAME=$3
    MODEL_PREFIX=$4 # model directory
    HPARAM_SET=$5 # any parameter sets
    EXECUTABLE=$6
    MODEL_ID=$7

    STAT_DIR="$RESULT_ROOT/$STAT_DIRNAME"
    DECODE_DIR=${STAT_DIR}/decodes

    mkdir -p ${DECODE_DIR}
    STAT_OUTFILE=${STAT_DIR}/${DATASET_TYPE}.${MODEL_NAME}.top
    DECODE_PREFIX=${DECODE_DIR}/decode.${DATASET_TYPE}.${MODEL_NAME}
    TEST_FILE=../../data/${DATASET_TYPE}_rank/${DATASET_TYPE}_rank.five_hyp.test.jsonl

    if [ ! -f "$MODEL_PREFIX/model_0_state_${MODEL_ID}.th" ]; then
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 nohup python -u $EXECUTABLE \
        -p $HPARAM_SET \
        --translator ${DATASET_TYPE}_rank_char \
        --dataset ${DATASET_TYPE}_five_hyp \
        --device 0 \
        --vocab-dump $MODEL_PREFIX/vocab \
        --test \
        $MODEL_PREFIX/model_0_state_${MODEL_ID}.th \
        $MODEL_PREFIX/model_1_state_${MODEL_ID}.th \
        > $DECODE_PREFIX.${MODEL_ID};

    for ((j=2; j<6; j++));
    do
        echo ${MODEL_ID} >> ${STAT_OUTFILE}.$j;
        python evaluator.py $TEST_FILE $DECODE_PREFIX.${MODEL_ID} --max-hyp-rank $j >> ${STAT_OUTFILE}.$j;
    done
}

snapshot=../../snapshots/django_fifteen_hyp/meta_ranker/20200601-052611-qmaml-lfngram-15
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 1
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 2
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 3
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 4
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 5
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 1-2000
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 2-4000
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 2-6000
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 3-8000
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 3-10000
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 4-12000
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 4-14000
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 5-16000
eval_template django 10qmaml-deep-chgiant-django-15 lfngram "$snapshot" django_lf_ngram meta_ranker.py 5-18000

snapshot=../../snapshots/django_fifteen_hyp/meta_ranker/20200601-052611-qmaml-nlbert-15
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 1
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 2
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 3
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 4
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 5
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 1-2000
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 2-4000
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 2-6000
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 3-8000
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 3-10000
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 4-12000
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 4-14000
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 5-16000
eval_template django 10qmaml-deep-chgiant-django-15 nlbert "$snapshot" django_nl_bert meta_ranker.py 5-18000

snapshot=../../snapshots/django_fifteen_hyp/meta_ranker/20200601-052611-qmaml-nlngram-15
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 1
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 2
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 3
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 4
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 5
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 1-2000
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 2-4000
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 2-6000
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 3-8000
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 3-10000
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 4-12000
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 4-14000
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 5-16000
eval_template django 10qmaml-deep-chgiant-django-15 nlngram "$snapshot" django_nl_ngram meta_ranker.py 5-18000

snapshot=../../snapshots/django_fifteen_hyp/meta_ranker/20200601-140348-qmaml-lfted-15-fixed
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 1
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 2
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 3
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 4
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 5
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 1-2000
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 2-4000
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 2-6000
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 3-8000
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 3-10000
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 4-12000
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 4-14000
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 5-16000
eval_template django 10qmaml-deep-chgiant-django-15 lfted "$snapshot" django_lf_ted meta_ranker.py 5-18000
