Samples here are illustrative. For a runnable walkthrough, see [examples](https://github.com/kiyoon/PyVideoAI-examples.git).

## PySlowFast config vs PyVideoAI config
Let's take a look at the one from PySlowFast (partial):  
```
DATA:
  NUM_FRAMES: 8
  SAMPLING_RATE: 8
  TRAIN_JITTER_SCALES: [256, 320]
  TRAIN_CROP_SIZE: 224
  TEST_CROP_SIZE: 256
  INPUT_CHANNEL_NUM: [3]
RESNET:
  ZERO_INIT_FINAL_BN: True
  WIDTH_PER_GROUP: 64
  NUM_GROUPS: 1
  DEPTH: 50
  TRANS_FUNC: bottleneck_transform
  STRIDE_1X1: False
  NUM_BLOCK_TEMP_KERNEL: [[3], [4], [6], [3]]
NONLOCAL:
  LOCATION: [[[]], [[]], [[]], [[]]]
  GROUP: [[1], [1], [1], [1]]
  INSTANTIATION: softmax
```

This looks intuitive at first glance, but it has many problems.  
First, it is impossible to change the dataloader style to different sampling / augmentation strategy.  
Second, some parameters need to be hard-coded and memorised.  
Third, adding a new model architecture requires re-designing of the entire config system.  

Therefore, PyVideoAI provides Python config structure that enables intuitive full customisation!  
It's not that complicated if you know what to do. You only need to change 3 config files (`dataset_configs`, `model_configs`, `exp_configs`).  
Most of the `dataset_configs` and `model_configs` settings are optional, and what matters is `exp_configs`. However, let's not hardcode everything, but rather split the configs.  
From `exp_configs`, you can access to `dataset_cfg` and `model_cfg` environment.

For example (simplified),  
`dataset_configs/hmdb.py`  
```python
num_classes = 51
task = 'singlelabel_classification'

# ...
```

`model_configs/i3d_resnet50.py`  
```python
# Define your own model by modifying the function body!
from model import ResNetModel
def load_model(num_classes, input_frame_length, crop_size, input_channel_num):
	model = ResNetModel('i3d', depth=50, output=num_classes, ...)
	return model

# ...
```

`exp_configs/hmdb/i3d_resnet50-testrun.py`  
```python
batch_size = 4	# per process (per GPU)

def optimiser(params):
    return torch.optim.SGD(params, lr = 0.001, momentum = 0.9, weight_decay = 5e-4)

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    return torch.optim.lr_scheduler.StepLR(optimiser, step_size = 50 * iters_per_epoch, gamma = 0.1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.

# ...
```

## Running the example
To run the above example,  
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_train.py -D hmdb -M i3d_resnet50 -E testrun --local_world_size 4 -e 200
```

To resume, add `-l -1` argument.
