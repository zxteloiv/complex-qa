function eval_atis_five () {
    # trained on ten-hyp data, evaluated on five-hyp, however
    OUTFILE=atis_meta_five_hyp.log
    TMPFILE=/tmp/am5
    MODEL_PREFIX=../../snapshots/atis_five_hyp/meta_ranker/20191206-173430-fixed

    # crude testing
    for model_file in $(ls -rt $MODEL_PREFIX/model_state_*-*.th);
    do
        echo $model_file >> $OUTFILE;
        CUDA_VISIBLE_DEVICES=1 python -u meta_ranker.py \
            -p atis_five_crude_testing \
            --dataset atis_five_hyp \
            --device 0 \
            --test $model_file \
            --vocab-dump $MODEL_PREFIX/vocab \
            > $TMPFILE 2>/dev/null;
        python evaluator.py ../../data/atis_rank/atis_rank.five_hyp.test.jsonl $TMPFILE >> $OUTFILE;
        rm $TMPFILE;
    done;

    # meta test
    for ((i=1; i<100; i++));
    do
        if [ ! -f "$MODEL_PREFIX/model_state_$i.th" ]; then
            continue
        fi
        echo $MODEL_PREFIX/model_state_$i.th >> $OUTFILE;
        CUDA_VISIBLE_DEVICES=2 python -u meta_ranker.py \
            -p atis_five_testing_no_dropout \
            --dataset atis_five_hyp \
            --device 0 \
            --test $MODEL_PREFIX/model_state_$i.th \
            --vocab-dump $MODEL_PREFIX/vocab \
            > $TMPFILE;
        python evaluator.py ../../data/atis_rank/atis_rank.five_hyp.test.jsonl $TMPFILE >> $OUTFILE;
        rm $TMPFILE;
    done;
}

if [ "$1" == "atis_5" ]; then
    eval_atis_five;
fi

