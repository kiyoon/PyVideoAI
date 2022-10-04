# Capturing Temporal Information in a Single Frame: Channel Sampling Strategies for Action Recognition

Kiyoon Kim, Shreyank N Gowda, Oisin Mac Aodha, Laura Sevilla-Lara  
In BMVC 2022. [arXiv](http://arxiv.org/abs/2201.10394)

<img src="https://user-images.githubusercontent.com/12980409/151038213-12bdad91-7895-40e7-9304-126079fed637.png" alt="8-frame TC Reordering" width="800">  
<img src="https://user-images.githubusercontent.com/12980409/151038200-6f32cea8-6a2b-40bf-9d95-50ba860114be.png" alt="3-frame GrayST" width="800">  

## Getting started

Core implementation of reordering methods is in [`pyvideoai/utils/tc_reordering.py`](../../../pyvideoai/utils/tc_reordering.py).  
See [`exp_configs/ch_tcgrey`](../../../exp_configs/ch_tcgrey) for experiment settings.  
For example, in order to run TSM model, GreyST method on the Something-Something-V1 dataset, you should run [`exp_configs/ch_tcgrey/something_v1/tsm_resnet50_nopartialbn-GreyST_8frame.py`](../../../exp_configs/ch_tcgrey/something_v1/tsm_resnet50_nopartialbn-GreyST_8frame.py), the command of which would be:

```bash
# Run training
tools/run_singlenode.sh train 4 -R ~/experiment_root -D something_v1 -M tsm_resnet_nopartialbn -E GreyST_8frame -c:e tcgrey
# Run evaluation
tools/run_singlenode.sh eval 4 -R ~/experiment_root -D something_v1 -M tsm_resnet_nopartialbn -E GreyST_8frame -c:e tcgrey
```

## Citing the paper

If you find our work or code useful, please cite:

```BibTeX
@inproceedings{kim2022capturing,
  title={Capturing Temporal Information in a Single Frame: Channel Sampling Strategies for Action Recognition},
  author={Kim, Kiyoon and Gowda, Shreyank N and Mac Aodha, Oisin and Sevilla-Lara, Laura},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
