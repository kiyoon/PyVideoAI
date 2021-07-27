import os

from pyvideoai.dataloaders.frames_densesample_dataset import FramesDensesampleDataset

import torch

#batch_size = 8  # per process (per GPU)
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    devices=list(range(torch.cuda.device_count()))
    vram = min([torch.cuda.get_device_properties(device).total_memory for device in devices])
    if vram > 20e+9:
        return 32 
    elif vram > 10e+9:
        return 16
    return 8

"""By default, val_batch_size is the same as batch_size
"""
#def val_batch_size():
#    return batch_size() if callable(batch_size) else batch_size

input_frame_length = 8
input_sample_rate = 3
crop_size = 224
train_jitter_min = 224
train_jitter_max = 336
val_scale = 224
val_num_ensemble_views = 1
val_num_spatial_crops = 1
test_scale = 224
test_num_ensemble_views = 10
test_num_spatial_crops = 3

input_channel_num=[3]   # RGB


#### OPTIONAL
#def criterion():
#    return torch.nn.CrossEntropyLoss()
#
#def epoch_start_script(epoch, exp, args, rank, world_size, train_kit):
#    return None
#
#def get_optim_policies(model):
#    """
#    You can set different learning rates on different blocks of the network.
#    Refer to `get_optim_policies()` in pyvideoai/models/epic/tsn.py
#    """
#    conv_weight = []
#    conv_bias = []
#    for m in model.parameters():
#        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
#            ps = list(m.parameters())
#            conv_weight.append(ps[0])
#            if len(ps) == 2:
#                conv_bias.append(ps[1])
#        # ...
#    return [
#        {   
#            "params": conv_weight,
#            "lr_mult": 1,
#            "decay_mult": 1,
#            "name": "conv_weight",
#        },
#        {   
#            "params": conv_bias,
#            "lr_mult": 2,
#            "decay_mult": 0,
#            "name": "conv_bias",
#        },
#    ]
#
import logging
logger = logging.getLogger(__name__)
from pyvideoai.utils.early_stopping import min_value_within_lastN, best_value_within_lastN
# optional
def early_stopping_condition(exp, metric_info):
    patience=20
    if not min_value_within_lastN(exp.summary['val_loss'], patience):
        best_metric_fieldname = metric_info['best_metric_fieldname']
        best_metric_is_better = metric_info['best_metric_is_better_func']
        if not best_value_within_lastN(exp.summary[best_metric_fieldname], patience, best_metric_is_better):
            logger.info(f"Validation loss and {best_metric_fieldname} haven't gotten better for {patience} epochs. Stopping training..")
            return True

    return False



from pyvideoai.utils.distributed import get_world_size
def optimiser(params):
    """
    LR should be proportional to the total batch size.
    When distributing, LR should be multiplied by the number of processes (# GPUs)
    Thus, LR = base_LR * batch_size_per_proc * (num_GPUs**2)
    """
    base_learning_rate = 1e-6      # when batch_size == 1 and #GPUs == 1

    batchsize = batch_size() if callable(batch_size) else batch_size
    world_size = get_world_size()
    learning_rate = base_learning_rate * batchsize * (world_size**2)

    return torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)

from pyvideoai.utils.lr_scheduling import ReduceLROnPlateauMultiple
def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    #return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0 = 100 * iters_per_epoch, T_mult = 1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    #return torch.optim.lr_scheduler.StepLR(optimiser, step_size = 50 * iters_per_epoch, gamma = 0.1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    #return torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.
    return ReduceLROnPlateauMultiple(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.
    #return None

def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length, crop_size, input_channel_num)

# optional
def load_pretrained(model):
    model_cfg.load_pretrained_kinetics400(model, model_cfg.kinetics400_pretrained_path_8x8)

def _dataloader_shape_to_model_input_shape(inputs):
    N, C, T, H, W = inputs.shape        # C = 1
    GreyST = inputs.view(N,3,T//3,H,W).reshape(N, T//3, 3, H, W).permute(0,2,1,3,4)
    return model_cfg.NCTHW_to_model_input_shape(GreyST)

def get_input_reshape_func(split):
    '''
    if split == 'train':
        return _dataloader_shape_to_model_input_shape
    elif split == 'val':
        return _dataloader_shape_to_model_input_shape
    elif split == 'multicropval':
        return _dataloader_shape_to_model_input_shape
    else:
        raise ValueError(f'Unknown split: {split}')
    '''
    return _dataloader_shape_to_model_input_shape


def _unpack_data(data):
    '''
    From dataloader returning values to (inputs, uids, labels, [reserved]) format
    '''
    inputs, uids, labels, spatial_idx, temporal_idx, _, _ = data
    return inputs, uids, labels, {"spatial_idx": spatial_idx, "temporal_idx": temporal_idx}

def get_data_unpack_func(split):
    '''
    if split == 'train':
        return _unpack_data
    elif split == 'val':
        return _unpack_data
    elif split == 'multicropval':
        return _unpack_data
    else:
        raise ValueError(f'Unknown split: {split}')
    '''
    return _unpack_data

def _get_torch_dataset(csv_path, split):
    mode = dataset_cfg.split2mode[split]

    if split == 'val':
        _test_scale = val_scale
        _test_num_ensemble_views = val_num_ensemble_views
        _test_num_spatial_crops = val_num_spatial_crops
    else:
        _test_scale = test_scale
        _test_num_ensemble_views = test_num_ensemble_views
        _test_num_spatial_crops = test_num_spatial_crops
    return FramesDensesampleDataset(csv_path, mode,
            input_frame_length*3, input_sample_rate,
            train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
            test_scale=_test_scale, test_num_ensemble_views=_test_num_ensemble_views, test_num_spatial_crops=_test_num_spatial_crops,
            crop_size=crop_size,
            mean = [model_cfg.input_mean[0]], std = [model_cfg.input_std[0]],
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
            greyscale=True,
            path_prefix=dataset_cfg.frames_dir)

def get_torch_dataset(split):

    csv_path = os.path.join(dataset_cfg.frames_split_file_dir, dataset_cfg.split_file_basename[split])

    return _get_torch_dataset(csv_path, split)



"""
OPTIONAL: change metrics, plotting figures and reporting text.
when you resume from checkpoint, load optimiser/scheduler state?
Default values are True.
"""
#load_optimiser_state = True
#load_scheduler_state = True

"""
OPTIONAL but important: configure DDP and AMP
Uncommenting following lines can speed up the training but also can introduce exceptions depending on your model.
Recommeded to change this settings in model_configs
"""
#ddp_find_unused_parameters = False
#use_amp = True

"""
## For both train & val
Changing last_activation and leaving metrics/predictions_gatherers commented out will still change the default metrics and predictions_gatherers' activation function
"""
#last_activation = 'softmax'   # or, you can pass a callable function like `torch.nn.Softmax(dim=1)`

"""
## For training, (tools/run_train.py)
how to calculate metrics
"""
#from pyvideoai.metrics.accuracy import ClipAccuracyMetric, VideoAccuracyMetric
#best_metric = ClipAccuracyMetric()
#metrics = {'train': [ClipAccuracyMetric()],
#        'val': [best_metric],
#        'multicropval': [ClipAccuracyMetric(), VideoAccuracyMetric(topk=(1,5), activation=last_activation)],
#        }
#
"""
## For validation, (tools/run_val.py)
how to gather predictions when --save_predictions is set
"""
#from pyvideoai.metrics.metric import ClipPredictionsGatherer, VideoPredictionsGatherer
#predictions_gatherers = {'val': ClipPredictionsGatherer(last_activation),
#        'multicropval': VideoPredictionsGatherer(last_activation),
#        }
#
"""How will you plot"""
#from pyvideoai.visualisations.metric_plotter import DefaultMetricPlotter
#metric_plotter = DefaultMetricPlotter()
#from pyvideoai.visualisations.telegram_reporter import DefaultTelegramReporter
#telegram_reporter = DefaultTelegramReporter()
