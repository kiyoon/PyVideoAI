# I3D, I3D Nonlocal
Use following model configs.  
- `model_configs/i3d_resnet50.py`
- `model_configs/i3dnonlocal_resnet50.py`

## Kinetics

Reference: [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)  
| architecture | depth |  pretrain |  frame length x sample rate | top1 |  top5  |  model | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| I3D | R50 | - | 8 x 8 | 73.5 | 90.8 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl) | Kinetics/c2/I3D_8x8_R50 |
| I3D NLN | R50 | - | 8 x 8 | 74.0 | 91.1 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_NLN_8x8_R50.pkl) | Kinetics/c2/I3D_NLN_8x8_R50 |


Reference: [facebookresearch/video-nonlocal-net](https://github.com/facebookresearch/video-nonlocal-net)  
| <sub>script</sub> | <sub>input frames</sub> | <sub>freeze bn?</sub> | <sub>3D conv?</sub> | <sub>non-local?</sub> | <sub>top1</sub> | <sub>in paper</sub> | <sub>top5</sub> | <sub>model</sub> | <sub>logs</sub> |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| <sub>run_i3d_baseline_400k_32f.sh</sub> | 32 | - | Yes | - | 73.6 | <sub>73.3</sub> | 90.8 | [`link`](https://dl.fbaipublicfiles.com/video-nonlocal/i3d_baseline_32x2_IN_pretrain_400k.pkl) | [`link`](https://dl.fbaipublicfiles.com/video-nonlocal/i3d_baseline_32x2_IN_pretrain_400k.log) |
| <sub>run_i3d_nlnet_400k_32f.sh</sub> | 32 | - | Yes | Yes | 74.9 | <sub>74.9</sub> | 91.6 | [`link`](https://dl.fbaipublicfiles.com/video-nonlocal/i3d_nonlocal_32x2_IN_pretrain_400k.pkl) | [`link`](https://dl.fbaipublicfiles.com/video-nonlocal/i3d_nonlocal_32x2_IN_pretrain_400k.log) |

## HMDB-51

This model is trained with PyVideoAI.  
Top1/5 accuracy is calculated using 1 spatial centre crop and 5 temporal crops.  
`dataset_configs/hmdb.py`  
`model_configs/i3d_resnet50.py`  
`exp_configs/hmdb/i3d_resnet50-crop224_lr0001_batch8_8x8_largejit_plateau_1scrop5tcrop_split1.py`  
| architecture | Pretrain | frame length x sampling stride |  Top1 (highest/last) |  Top5 (highest/last)  | config, log | model (last)  | TensorBoard |
| ------------- | ---- | ------------- | ------------- | ------------- | ------------- | ---- | ---- |
| I3D-ResNet50 | Kinetics | 8 x 8 | 73.20 / 72.94 | 94.05 / 94.05 | [`link`](https://uoe-my.sharepoint.com/:u:/g/personal/s1884147_ed_ac_uk/EdCBfzdWnitNtH475UaCJdEB8R4MhJ9sQCnEEdFabk2xSQ?e=yl20fF&download=1) | [`link`](https://uoe-my.sharepoint.com/:u:/g/personal/s1884147_ed_ac_uk/EefmKjHu_iRPvN2JTqG2QNYBoCs18kbX0ajidiKOuWEgZQ?e=9oGJ0v&download=1) | [`link`](https://tensorboard.dev/experiment/mGSBcdZfQmWJNd658zHLbQ) |


## ImageNet

Reference: [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)  

| architecture | depth |  Top1 |  Top5  |  model  |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ResNet | R50 | 23.6 | 6.8 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/R50_IN1K.pyth) |
