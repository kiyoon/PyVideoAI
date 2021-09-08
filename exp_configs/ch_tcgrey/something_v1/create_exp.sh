for model in tsn_resnet50 trn_resnet50 mtrn_resnet50 
do
	for sampling_mode in RGB TC TCrgb TCred GreyST
	do
		ln -s ../sparsesample_${sampling_mode}_crop224_8frame_largejit_plateau_10scrop.py $model-${sampling_mode}_8frame.py
	done
done

cp ../something_v2/tsm_resnet50_nopartialbn-* .

for model in i3d_resnet50
do
	for sampling_mode in RGB TC TCrgb GreyST
	do
		ln -s ../densesample_${sampling_mode}_crop224_8x8_largejit_plateau_3scrop10tcrop.py $model-${sampling_mode}_8frame.py
	done
done
