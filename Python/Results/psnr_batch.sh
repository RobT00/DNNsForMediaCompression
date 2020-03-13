#!/bin/bash

batch_input=$1
output="${2}.json"
run_batch="python3 $Git/vmaf/run_psnr_in_batch"

$run_batch $batch_input --out-fmt json --parallelize >> $output
