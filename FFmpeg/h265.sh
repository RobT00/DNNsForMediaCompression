#!/bin/bash
Original="Original"
Target="H265"
Files=Original/*.y4m
Extension=".mp4"

[ -z "$1" ] && quality=28 || quality=$1

echo "Quality: $quality"
echo "Path: ${Target}/${f}_${quality}${Extension}"

# https://trac.ffmpeg.org/wiki/Encode/H.265
for f in $Files
do
	bf=${f##*/}  # Get base path
	echo "bf: $bf"
	f=${bf%.y4m}  # Remove .y4m extension
	# Cmd
	ffmpeg -y -i $Original/$bf -map_metadata -1 -c:a libfdk_aac -c:v libx265 -crf $quality -preset veryslow -profile:v main -pix_fmt yuv420p -movflags +faststart -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ${Target}/${f}_${quality}${Extension}
done
