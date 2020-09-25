function eval_template () {
    if [ "$1" == "" ]; then
        return 1;
    fi

    MODEL_NAME=$1
    MODEL_PREFIX=$2 # model directory
    HPARAM_SET=$3 # any parameter sets


    DECODE_PREFIX=decode.compwebq.${MODEL_NAME}
    for ((i=1; i<101; i++));
    do
        if [ ! -f "$MODEL_PREFIX/model_state_$i.th" ]; then
            continue
        fi
        CUDA_VISIBLE_DEVICES=0 nohup python -u s2s_qa.py \
            -p $HPARAM_SET \
            --device 0 \
            --vocab-dump $MODEL_PREFIX/vocab \
            --test $MODEL_PREFIX/model_state_$i.th \
            > $DECODE_PREFIX.$i;

    done;
}

snapshot=~/deploy/complex_qa/snapshots/CompWebQ/seq2seq/20200825-174225-mq
eval_template mq $snapshot machine_question_to_answer

snapshot=~/deploy/complex_qa/snapshots/CompWebQ/seq2seq/20200825-162012-q
eval_template q $snapshot question_to_answer
