#!/bin/bash
Original="Original"
Target="H264"
Files=Original/*.y4m
Extension=".mp4"

[ -z "$1" ] && method="crf" || method="$1"

[ -z "$2" ] && quality=24 || quality=$2

echo "Quality: $quality"
file_path=${Target}
mkdir -p $file_path
echo "Path: ${file_path}"

for f in $Files
do
	bf=${f##*/}  # Get base path
	echo "bf: $bf"
	f=${bf%.y4m}  # Remove .y4m extension
	# f=${f%.*}  # Remove any extension
	# Cmd
	ffmpeg -y -i $Original/$bf -map_metadata -1 -c:v libx264 -crf $quality -preset veryslow -profile:v main -pix_fmt yuv420p -movflags +faststart ${file_path}/${f}_${quality}${Extension}
done
