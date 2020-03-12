#!/bin/bash

batch_input=$1
output="${2}.json"
run_vmaf_batch="python3 $Git/vmaf/run_vmaf_in_batch"

$run_vmaf_batch $batch_input --out-fmt json --parallelize >> $output
