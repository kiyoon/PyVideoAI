# Capturing Temporal Information in a Single Frame: Channel Sampling Strategies for Action Recognition

Kiyoon Kim, Shreyank N Gowda, Oisin Mac Aodha, Laura Sevilla-Lara  
In BMVC 2022. [arXiv](http://arxiv.org/abs/2201.10394)

<img src="https://user-images.githubusercontent.com/12980409/151038213-12bdad91-7895-40e7-9304-126079fed637.png" alt="8-frame TC Reordering" width="800">  
<img src="https://user-images.githubusercontent.com/12980409/151038200-6f32cea8-6a2b-40bf-9d95-50ba860114be.png" alt="3-frame GrayST" width="800">  

## Getting started

### PyVideoAI checklist
- Install PyVideoAI correctly. See [README.md](../../../README.md) and [INSTALL.md](../../../docs/INSTALL.md).
- Make sure datasets are available in `PyVideoAI/data` directory. Generate splits using scripts in [`PyVideoAI/tools/datasets`](../../../tools/datasets). See [DATASET.md](../../../docs/DATASET.md).
- Following [PyVideoAI-examples](https://github.com/kiyoon/PyVideoAI-examples) help understand the overall workflow.

### About the paper
- Core implementation of reordering methods is in [`pyvideoai/utils/tc_reordering.py`](../../../pyvideoai/utils/tc_reordering.py).  
- See [`exp_configs/ch_tcgrey`](../../../exp_configs/ch_tcgrey) for experiment settings.  
For example, in order to run TSM model, GreyST method on the Something-Something-V1 dataset, you should run [`exp_configs/ch_tcgrey/something_v1/tsm_resnet50_nopartialbn-GreyST_8frame.py`](../../../exp_configs/ch_tcgrey/something_v1/tsm_resnet50_nopartialbn-GreyST_8frame.py), the command of which would be:

```bash
# Run training
tools/run_singlenode.sh train 4 -R ~/experiment_root -D something_v1 -M tsm_resnet_nopartialbn -E GreyST_8frame -c:e tcgrey
# Run evaluation
tools/run_singlenode.sh eval 4 -R ~/experiment_root -D something_v1 -M tsm_resnet_nopartialbn -E GreyST_8frame -c:e tcgrey
```

### Lists of datasets, models, experiments available
See [`exp_configs/ch_tcgrey`](../../../exp_configs/ch_tcgrey) for the available experiment settings.  
Not all combinations are available, but below are main settings.

#### Datasets (-D option)
- [cater_task2](../../../dataset_configs/cater_task2.py)
- [cater_task2_cameramotion](../../../dataset_configs/cater_task2_cameramotion.py)
- [something_v1](../../../dataset_configs/something_v1.py)
- [something_v2](../../../dataset_configs/something_v2.py)

#### Models (-M option)
- [tsn_resnet50](../../../model_configs/tsn_resnet50.py)
- [trn_resnet50](../../../model_configs/trn_resnet50.py)
- [mtrn_resnet50](../../../model_configs/mtrn_resnet50.py)
- [tsm_resnet50](../../../model_configs/tsn_resnet50.py) or [tsm_resnet50_nopartialbn](../../../model_configs/tsn_resnet50_nopartialbn.py)
- [mvf_resnet50](../../../model_configs/mvf_resnet50.py) or [mvf_resnet50_nopartialbn](../../../model_configs/mvf_resnet50_nopartialbn.py)

#### Experiments (-E option)
For `something_v1` and `something_v2`,  
- RGB_8frame
- TC_8frame
- TCPlus2_8frame
- GreyST_8frame

For `cater_task2` and `cater_task2_cameramotion`,  
- RGB_32frame
- TC_32frame
- TCPlus2_32frame
- GreyST_32frame

## Citing the paper

If you find our work or code useful, please cite:

```BibTeX
@inproceedings{kim2022capturing,
  title={Capturing Temporal Information in a Single Frame: Channel Sampling Strategies for Action Recognition},
  author={Kim, Kiyoon and Gowda, Shreyank N and Mac Aodha, Oisin and Sevilla-Lara, Laura},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
