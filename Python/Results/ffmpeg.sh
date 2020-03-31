#!/bin/bash

metric=$1  # psnr or ssim
distorted=$2
reference=$3

ffmpeg -v quiet -i "${distorted}" -i "${reference}" -lavfi ${metric}=${metric}.txt -f null -
