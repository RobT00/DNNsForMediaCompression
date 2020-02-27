#!/bin/bash
# Script to run 5e1 on Ubuntu

home_dir=$(pwd)

# Activate environment
source /home/rob/Git/Envs/5E1/bin/activate

# cd to Project dir
cd /home/rob/Git/5E1

# Set up runtime args
data_dir=/home/rob/Desktop/5E1_DB/DB
# -ci = compressed_input
ci=$data_dir/Video/Xiph/Colour/H264
# -oi = original_input
oi=$data_dir/Video/Xiph/Original
# -od = output_dir
od=$data_dir/Out
# -m = model (path or name)
m=$data_dir/Training/Video/Attempt1_3D/optimiser=Adam\ epochs=457\ batch_size=8\ metrics=mse\ model=Attempt1_3D\ precision=float32/Model/Attempt1_3D.h5
# -p = precision (float32 or 16)
p="float32"
# -g = generator (use -ng for no generator) [Boolean]
g="g"
# -s = sequences (us -ns for images) [Boolean]
s="s"

# Run script

echo $m

echo "$m"

python Python/Code/main.py -ci "$ci" -oi "$oi" -od "$od" -m "$m" -p $p -$g -$s

# Deativate 
deactivate

# Return to home dir
cd $home_dir
