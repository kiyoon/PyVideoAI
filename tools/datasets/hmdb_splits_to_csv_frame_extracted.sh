#!/bin/bash

# Author: Kiyoon Kim (yoonkr33@gmail.com)
# Description: Split HMDB51 dataset with provided Three Splits text files.

if [ $# -lt 3 ]
then
	echo "Usage: $0 [Three Splits directory] [Frame extracted location] [Output directory]"
	echo "Split HMDB51 dataset with provided Three Splits text files."
	echo "Author: Kiyoon Kim (yoonkr33@gmail.com)"
	exit 0
fi

split_dir=$(realpath "$1")
data_dir=$(realpath "$2")

mkdir -p "$3"
out_dir=$(realpath "$3")

classes=$(find "$data_dir" -mindepth 1 -maxdepth 1 -type d | sort)
if [ $(echo "$classes" | wc -l) -ne 51 ]
then
	printf '\xF0\x9F\x98\xA1 '   # pouting face
	echo "Error: dataset contains less or more than 51 classes." 1>&2
	exit 1
fi

# write header
for i in {1..3}
do
	# num_classes, but always 0 for single label classification
	echo "0" > "$out_dir/train$i.csv"
	echo "0" > "$out_dir/test$i.csv"
done

class_id=0
video_id=0
while read line
do
	name=$(basename "$line")
	echo "$name"

	for i in {1..3}
	do
		file_content=$(cat "$split_dir/${name}_test_split${i}.txt")

		file=$(echo "$file_content" | grep ' 1 $' | awk '{print $1}')
		if [ $(echo "$file" | wc -l) -ne 70 ]
		then 
			printf '\xF0\x9F\x98\xA1 '   # pouting face
			echo "Error: $split_dir/${name}_test_split${i}.txt doesn't consist of 70 training data." 1>&2
		fi

		while read line_file
		do
			num_frames=$(ls "$data_dir/$name/${line_file%.avi}" | wc -l)
			echo "$name/${line_file%.avi}/{:05d}.jpg $video_id $class_id 0 $((num_frames-1))" >> "$out_dir/train$i.csv"
			(( video_id++ ))
		done <<< "$file"

		file=$(echo "$file_content" | grep ' 2 $' | awk '{print $1}')
		if [ $(echo "$file" | wc -l) -ne 30 ]
		then 
			printf '\xF0\x9F\x98\xA1 '   # pouting face
			echo "Error: $split_dir/${name}_test_split${i}.txt doesn't consist of 30 test data." 1>&2
		fi

		while read line_file
		do
			num_frames=$(ls "$data_dir/$name/${line_file%.avi}" | wc -l)
			echo "$name/${line_file%.avi}/{:05d}.jpg $video_id $class_id 0 $((num_frames-1))" >> "$out_dir/test$i.csv"
			(( video_id++ ))
		done <<< "$file"
	done

	(( class_id++ ))
done <<< "$classes"
