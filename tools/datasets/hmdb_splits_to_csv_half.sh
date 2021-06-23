#!/bin/bash

# Author: Kiyoon Kim (yoonkr33@gmail.com)
# Description: Split HMDB51 dataset with provided Three Splits text files.

if [ $# -lt 3 ]
then
	echo "Usage: $0 [Three Splits directory] [Dataset location] [Output directory]"
	echo "Split HMDB51 dataset with provided Three Splits text files."
	echo "Author: Kiyoon Kim (yoonkr33@gmail.com)"
	exit 0
fi

split_dir=$(realpath "$1")
data_dir=$(realpath "$2")

mkdir -p "$3"
out_dir=$(realpath "$3")

classes=$(find "$data_dir" -mindepth 1 -maxdepth 1 -type d)
if [ $(echo "$classes" | wc -l) -ne 51 ]
then
	printf '\xF0\x9F\x98\xA1 '   # pouting face
	echo "Error: dataset contains less or more than 51 classes." 1>&2
	exit 1
fi

# filter 25 classes randomly
classes_filtered=$(echo "$classes" | shuf | head -n 25)
classes="$classes_filtered"

class_id=0
video_id=0
echo "$classes" | while read line
do
	name=$(basename "$line")

	for i in {1..3}
	do
		file_content=$(cat "$split_dir/${name}_test_split${i}.txt")

		file=$(echo "$file_content" | grep ' 1 $' | awk '{print $1}')
		if [ $(echo "$file" | wc -l) -ne 70 ]
		then 
			printf '\xF0\x9F\x98\xA1 '   # pouting face
			echo "Error: $split_dir/${name}_test_split${i}.txt doesn't consist of 70 training data." 1>&2
		fi

		num_files=$(echo "$file" | wc -l)
		half_num_files=$((num_files / 2))
		file_filtered=$(echo "$file" | shuf | head -n "${half_num_files}")
		file="$file_filtered"
		while read line_file
		do
			echo "$name/$line_file $video_id $class_id" >> "$out_dir/train$i.csv"
			(( video_id++ ))
		done <<< "$file"

		file=$(echo "$file_content" | grep ' 2 $' | awk '{print $1}')
		if [ $(echo "$file" | wc -l) -ne 30 ]
		then 
			printf '\xF0\x9F\x98\xA1 '   # pouting face
			echo "Error: $split_dir/${name}_test_split${i}.txt doesn't consist of 30 test data." 1>&2
		fi

		num_files=$(echo "$file" | wc -l)
		half_num_files=$((num_files / 2))
		file_filtered=$(echo "$file" | shuf | head -n "${half_num_files}")
		file="$file_filtered"
		while read line_file
		do
			echo "$name/$line_file $video_id $class_id" >> "$out_dir/test$i.csv"
			(( video_id++ ))
		done <<< "$file"
	done

	(( class_id++ ))
done	
