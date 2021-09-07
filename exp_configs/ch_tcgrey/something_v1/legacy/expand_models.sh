# You need 6 files
# trn_resnet50-RGB_8frame.py
# TCswap, Grey
# All of the above with 9 frames
trn_files=$(ls trn_resnet50-*.py)

while read line
do
	suffix=$(echo "$line" | awk -F \- '{print $2}')
	ln -s "$line" "tsn_resnet50-$suffix"
	ln -s "$line" "mtrn_resnet50-$suffix"
	ln -s "$line" "tsm_resnet50-$suffix"
done <<< "$trn_files"
