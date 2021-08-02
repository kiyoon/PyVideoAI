# PyVideoAI: Action Recognition Framework

The only framework that completes your computer vision, action recognition research environment.  

** Key features **  
- Supports multi-gpu, multi-node training.  
- STOA models such as I3D, Non-local, TSN, TRN, TSM, ..., and even ImageNet training!
- Many datasets such as Kinetics-400, EPIC-Kitchens-55, Something-Something-V1/V2, HMDB-51, UCF-101, Diving48, CATER, ...
- Supports both video decoding (straight from .avi/mp4) and frame extracted (.jpg/png) dataloaders, sparse-sample and dense-sample.
- Any popular LR scheduling like Cosine Annealing with Warm Restart, Step LR, and Reduce LR on Plateau.
- Early stopping when training doesn't improve (customise your condition)
- **Easily add custom model, optimiser, scheduler, loss and dataloader!**
- Telegram bot reporting experiment status.  
- TensorBoard reporting stats.  
- Colour logging  
- All of the above come with no extra setup. Trust me and try some [examples](https://github.com/kiyoon/PyVideoAI-examples.git).

This package is motivated by PySlowFast from Facebook AI. The PySlowFast is a cool framework, but it depends too much on their config system and it was difficult to add new models (other codes) or reuse part of the modules from the framework.  
This framework by Kiyoon, is designed to replace all the configuration systems to Python files, which enables **easy-addition of custom models/LR scheduling/dataloader** etc.  
Just modify the function bodies in the config files!

Difference between the two config systems can be found in [CONFIG_SYSTEM.md](docs/CONFIG_SYSTEM.md).

# Getting Started
Jupyter Notebook examples to run:  
- HMDB-51 data preparation
- Inference on pre-trained model from the model zoo, and visualise model/dataloader/per-class performance.
- Training I3D using Kinetics pretrained model
- Using image model and ImageNet dataset  

is provided in the [examples](https://github.com/kiyoon/PyVideoAI-examples)!


# Structure

All of the executable files are in `tools/`.  
`dataset_configs/` directory configures datasets. For example, where is the dataset stored, number of classes, single-label or multi-label training, dataset-specific visualisation settings (confusion matrix has different output sizes)  
`model_configs/` directory configures model architectures. For example, model definition, input preprocessing mean/std.  
`exp_configs/` directory configures other training settings like optimiser, scheduling, dataloader, number of frames as input. The config file path has to be in `exp_configs/[dataset_name]/[model_name]_[experiment_name].py` format.

# Usage

## Preparing datasets

This package supports many action recognition datasets such as HMDB-51, EPIC-Kitchens-55, Something-Something-V1, CATER, etc.  
Refer to [DATASET.md](docs/DATASET.md).

## Training command
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_train.py -D {dataset_config_name} -M {model_config_name} -E {exp_config_name} --local_world_size {num_GPUs} -e {num_epochs}
```
`--local_world_size` denotes the number of GPUs per computing node.

## Telegram Bot
You can preview experiment results using Telegram bots!  
<img src="https://user-images.githubusercontent.com/12980409/122335586-7cb10a80-cf76-11eb-950f-af08c20055d4.png" alt="Telegram bot stat report example" width="400">

If your code raises an exception, it will report you too.  
<img src="https://user-images.githubusercontent.com/12980409/122337458-5476db00-cf79-11eb-8d71-3e8ecc9faa9a.png" alt="Telegram error report example" width="400">

You can quickly take a look at example video inputs (as GIF or JPEGs) from the dataloader.  
Use [tools/visualisations/model_and_dataloader_visualiser.py](tools/visualisations/model_and_dataloader_visualiser.py)  
<img src="https://user-images.githubusercontent.com/12980409/122337617-8a1bc400-cf79-11eb-8c48-b0d52a2c49c5.png" alt="Telegram video input report example" width="200">



- Talk to BotFather and make a bot.  
- Go to your bot and type anything (/start)  
- Find chat_id at https://api.telegram.org/bot{token}/getUpdates (without braces)  
- Add your token and chat_id to `tools/key.ini`.  

```INI
[Telegram0]
token=
chat_id=
```


# Model Zoo and Baselines
Refer to [MODEL_ZOO.md](docs/MODEL_ZOO.md)

# Installation
Refer to [INSTALL.md](docs/INSTALL.md).

TL;DR,

```bash
conda create -n videoai python=3.8
conda activate videoai
conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.2 -c pytorch
### For RTX 30xx GPUs,
#conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=11.1 -c pytorch -c nvidia
 

git clone --recurse-submodules https://github.com/kiyoon/PyVideoAI.git
cd PyVideoAI
git checkout v0.1
git submodule update --recursive
cd submodules/video_datasets_api
pip install -e .
cd ../experiment_utils
pip install -e .
cd ../..
pip install -e .
```


# Experiment outputs

The experiment results (log, training stats, weights, tensorboard, plots, etc.) are saved to `data/experiments` by default. This can be huge, so make sure you **make a softlink of a directory you really want to use. (recommended)**  

Otherwise, you can change `pyvideoai/config.py`'s `DEFAULT_EXPERIMENT_ROOT` value. Or, you can also set `--experiment_root` argument manually when executing.  

