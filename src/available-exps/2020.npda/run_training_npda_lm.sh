#!/bin/bash

function run_on_device {
  device=$1
  start=$2
  ends=$3

  for ((i=${start}; i<${ends}; i++))
  do
    CUDA_VISIBLE_DEVICES="$device" nohup python -u npda_lm.py \
        -p ntnorm_ntnum_$i \
        --device 0 \
        --vocab-dump vocab \
        --memo ntnorm_ntnum_$i
  done;
}

run_on_device 0 0 3 &> log.npda.lm.ntnorm_ntnum.dev0 &
run_on_device 1 3 6 &> log.npda.lm.ntnorm_ntnum.dev1 &
