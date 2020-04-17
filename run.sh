#!/bin/bash
# Sample script for running 5E1 project

home_dir=$(pwd)

# Activate virtual env for project
source $Git/Envs/5E1/bin/activate

# cd to Project dir
cd $Git/5E1

# Set up alias for training data location
data_dir=$HOME/5E1/DB

# Set up runtime args
# -ci = compressed_input
ci=$data_dir/Video/Xiph/Colour/LowQual/H264
# ci=$data_dir/Video/Xiph/Colour/No_deblocking/H264 
# -oi = original_input
oi=$data_dir/Video/Xiph/Colour/Original
# -od = output_dir
od=$data_dir/Out
# -m = model (path or name)
m="Attempt1"
# -nt = no-train [Include if only loading model output]
nt="nt"
# -s = sequences [Include for video]
s="s"
# -e = epochs
e=500
# -b = batch_size
b=2
# -d = dims (w, h, c)
d="512, 768, 3"
# -cs = colourspace (YUV, BGR, RGB) [YUV is default]
cs="YUV"
# -kl = keep loss function [Default is to change when loading model], add to preserve loss function (MSE)
kl="kl"

# Run script
python Python/Code/main.py -ci "$ci" -oi "$oi" -od "$od" -m "$m" -e $e -b $b -d "$d" -cs "$cs"

# Deactivate Python virtual env
deactivate

# Return to home dir
cd $home_dir
