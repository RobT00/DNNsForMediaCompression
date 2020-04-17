#!/bin/bash

# Bash script for running FFmpeg scripts multiple times for quality settings
call_script () {
	echo "call: ${q}"
 	./${script} ${method} ${q}
}

if [ ! -z "$1" ]
then
	method=$1
	echo "Compression method is ${method}"
else
	echo "Compression method not set, using crf"
	method="crf"
fi

if [ ! -z "$2" ]
then
	format=$2
	echo "Encoding type is ${format}"
else
	echo "Encoding type is empty!"
	exit 1
fi

if [ ! -z "$3" ]
then
	multi=0  # true
else
	echo "No quality params set, running default"
fi

# Get the script
script=${format}.sh
if test -f "${script}"
then
	echo "Script: ${script} found"
else
	echo "Script: ${script} not found!"
fi

if [ ! -z "$multi" ]
then
	for q in "${@:3}"
	do
		call_script
	done
else
	call_script
fi
