#!/bin/bash

metric=$1  # psnr or ssim
distorted=$2
reference=$3

ffmpeg -i $distorted -i $reference -lavfi $metric -f null -
# ffmpeg -i $distorted -i $reference -lavfi $metric=$metric.text -f null -
