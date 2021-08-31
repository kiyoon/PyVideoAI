for model in tsn_resnet50 trn_resnet50 mtrn_resnet50 tsm_resnet50
do
	for sampling_mode in RGB TC GreyST
	do
		ln -s ../sparsesample_video_${sampling_mode}.py $model-${sampling_mode}_8frame_video.py
	done
done

