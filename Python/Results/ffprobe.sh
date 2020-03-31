#!/bin/bash

vid=$1
out_file=${2}.json

ffprobe -v quiet -i "$vid" -print_format json -show_format > "${out_file}"
