#!/bin/bash
Original="Original"
Target="VP9"
Files=Original/*.y4m
Extension=".mp4"

[ -z "$1" ] && quality=30 || quality=$1

echo "Quality: $quality"
echo "Path: ${Target}/${f}_${quality}${Extension}"

# https://trac.ffmpeg.org/wiki/Encode/VP9
for f in $Files
do
	bf=${f##*/}  # Get base path
	echo "bf: $bf"
	f=${bf%.y4m}  # Remove .y4m extension
	# Cmd
	ffmpeg -y -i $Original/$bf -c:v libvpx-vp9 -crf $quality -b:v 0 -pix_fmt yuv420p -movflags +faststart -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ${Target}/${f}_${quality}${Extension}
done
