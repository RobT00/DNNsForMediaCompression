#!/bin/bash
Original="Original"
Target="AV1"
Files=Original/*.y4m
Extension=".mp4"

[ -z "$1" ] && quality=34 || quality=$1

echo "Quality: $quality"
echo "Path: ${Target}/${f}_${quality}${Extension}"

for f in $Files
do
	bf=${f##*/}  # Get base path
	echo "bf: $bf"
	f=${bf%.y4m}  # Remove .y4m extension
	# f=${f%.*}  # Remove any extension
	# Cmd
	ffmpeg -y -i $Original/$bf -map_metadata -1 -c:a libopus -c:v libaom-av1 -crf $quality -b:v 0 -pix_fmt yuv420p -movflags +faststart -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -strict experimental ${Target}/${f}_${quality}${Extension}
done
