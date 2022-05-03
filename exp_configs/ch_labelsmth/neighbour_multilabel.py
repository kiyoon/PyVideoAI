import os
import pickle

from pyvideoai.dataloaders import FramesSparsesampleDataset, VideoSparsesampleDataset, GulpSparsesampleDataset
#from pyvideoai.utils.losses.proselflc import ProSelfLC, InstableCrossEntropy
#from pyvideoai.utils.losses.loss import LabelSmoothCrossEntropyLoss
from pyvideoai.utils.losses.softlabel import SoftlabelRegressionLoss
from pyvideoai.utils.losses.softlabel import MaskedSoftlabelRegressionLoss
from pyvideoai.utils import loader
from exp_configs.ch_labelsmth.epic100_verb.loss import MinCEMultilabelLoss, MinRegressionCombinationLoss
from pyvideoai.utils.losses.masked_crossentropy import MaskedCrossEntropy
from pyvideoai.utils.losses.proselflc import MaskedProSelfLC
from pyvideoai.utils.stdout_logger import OutputLogger

import torch
import numpy as np

input_frame_length = 8
input_type = 'RGB_video' # RGB_video / flow / gulp_rgb / gulp_flow

num_epochs = int(os.getenv('VAI_NUM_EPOCHS', default=500))

#batch_size = 8  # per process (per GPU)
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    if input_type == 'RGB_video':
        divide_batch_size = 1
    elif input_type == 'flow':
        divide_batch_size = 4       # For optical flow, you read 5 times as many frames, so it will be a bottleneck if you use too big batch size.
    elif input_type in ['gulp_rgb', 'gulp_flow']:
        divide_batch_size = 2
    else:
        raise ValueError(f'Wrong input_type {input_type}')

    devices=list(range(torch.cuda.device_count()))
    vram = min([torch.cuda.get_device_properties(device).total_memory for device in devices])

    if vram > 20e+9:
        return input_frame_length * 4 // divide_batch_size
    elif vram > 10e+9:
        return input_frame_length * 2 // divide_batch_size
    return input_frame_length // divide_batch_size


def val_batch_size():
    return batch_size() if callable(batch_size) else batch_size


crop_size = 224
train_jitter_min = 224
train_jitter_max = 336
val_scale = 256
val_num_spatial_crops = 1
test_scale = 256
test_num_spatial_crops = 10 if dataset_cfg.horizontal_flip else 1

train_classifier_balanced_retraining_epochs = 0     # How many epochs to re-train the classifier from random weights, in a class-balanced way, at the end. Bingyi Kang et al. 2020, (cRT)
early_stop_patience = 20        # or None to turn off

pretrained = 'epic100'          # epic100 / imagenet / random

sample_index_code = 'pyvideoai'
#clip_grad_max_norm = 5

base_learning_rate = float(os.getenv('VAI_BASELR', 5e-6))      # when batch_size == 1 and #GPUs == 1

loss_type = 'mince'     # mince / maskce / minregcomb / maskproselflc
                        # mask_binary_ce

l2_norm = True
save_features = True

#proselflc_total_time = 2639 * 60 # 60 epochs
#proselflc_total_time = 263 * 40 # 60 epochs
def proselflc_total_time():
    train_dataset = get_torch_dataset('train')
    N = batch_size()
    train_samples = len(train_dataset)
    num_iters_per_epoch = train_samples // N
    total_time = num_iters_per_epoch * num_epochs
    logger.info(f'ProSelfLC total time = {num_iters_per_epoch} * {num_epochs} = {total_time}')
    return total_time


proselflc_exp_base = 1.

#### OPTIONAL
def get_criterion(split):
    if split == 'train':
        if loss_type == 'mince':
            return MinCEMultilabelLoss()
        elif loss_type == 'maskce':
            return MaskedCrossEntropy()
        elif loss_type == 'mask_binary_ce':
            return MaskedSoftlabelRegressionLoss()
        elif loss_type == 'minregcomb':
            return MinRegressionCombinationLoss()
        elif loss_type == 'maskproselflc':
            return MaskedProSelfLC(proselflc_total_time(), proselflc_exp_base)
        else:
            raise ValueError(f'{loss_type=} not recognised.')
    else:
        return torch.nn.CrossEntropyLoss()

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import pyvideoai.utils.distributed as du
from pyvideoai.train_and_eval import extract_features
from exp_configs.ch_labelsmth.epic100_verb.features_study import get_neighbours
import exp_configs
train_testmode_dataloader = None
video_id_to_label = None
def epoch_start_script(epoch, exp, args, rank, world_size, train_kit):
    global train_testmode_dataloader
    if train_testmode_dataloader is None or epoch % 5 == 0:
        if 'partial' in dataset_cfg.__name__:
            feature_extract_split = 'trainpartialdata_testmode'
        else:
            feature_extract_split = 'traindata_testmode'

        if train_testmode_dataloader is None:
            train_testmode_dataset = get_torch_dataset(feature_extract_split)
            sampler = DistributedSampler(train_testmode_dataset, shuffle=False) if world_size > 1 else None
            train_testmode_dataloader = torch.utils.data.DataLoader(train_testmode_dataset, batch_size=val_batch_size(), shuffle=False, sampler=sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=False, worker_init_fn = du.seed_worker)

        data_unpack_func = get_data_unpack_func(feature_extract_split)
        input_reshape_func = get_input_reshape_func(feature_extract_split)
        feature_data, _, _, _, eval_log_str = extract_features(model_cfg.feature_extract_model(train_kit["model"], 'features'), train_testmode_dataloader, data_unpack_func, dataset_cfg.num_classes, feature_extract_split, rank, world_size, input_reshape_func=input_reshape_func, refresh_period=args.refresh_period)
        # feature_data['video_ids'] feature_data['labels'] feature_data['clip_features']

        if rank == 0:
            if feature_data['clip_features'].shape[1] == input_frame_length:
                # features are not averaged. Average now
                feature_data['clip_features'] = np.mean(feature_data['clip_features'], axis=1)

            assert feature_data['clip_features'].shape[1] == 2048
            kn = 10
            thr = 0.2
            with OutputLogger(exp_configs.ch_labelsmth.epic100_verb.features_study.__name__, 'INFO'):
                nc_freq, _, _ = get_neighbours(feature_data['clip_features'], feature_data['clip_features'], feature_data['labels'], feature_data['labels'], kn, l2_norm=l2_norm)
            #neighbours_ids = []
            soft_labels = []
            target_ids = feature_data['video_ids']
            source_ids = feature_data['video_ids']

            for target_idx, target_id in enumerate(target_ids):
                #n_ids = [source_ids[x] for x in features_neighbours[target_idx]]
                sl = nc_freq[target_idx]
                sl = sl / sl.sum()
                #neighbours_ids.append(n_ids)
                soft_labels.append(sl)

            #neighbours_ids = np.array(neighbours_ids)
            soft_labels = np.array(soft_labels)

            multilabels = MinCEMultilabelLoss.generate_multilabels_numpy(soft_labels, thr, feature_data['labels'])
            if loss_type in ['maskce', 'maskproselflc']:
                logger.info('Turning multilabels to mask out sign (-1) and including one single label')
                multilabels = -multilabels      # negative numbers mean masks. Mask out the relevant verbs.
                for idx, label in enumerate(feature_data['labels']):
                    multilabels[idx, label] = 1     # Make the actual single label GT the only label.
            else:
                logger.info('Including all multilabels and singlelabel')

            if save_features:
                logger.info(f"Saving features, neighbours, and multilabels to {os.path.join(exp.predictions_dir, 'features_neighbours')}")
                os.makedirs(os.path.join(exp.predictions_dir, 'features_neighbours'), exist_ok = True)

                feature_data['kn'] = kn
                feature_data['thr'] = thr
                feature_data['nc_freq'] = nc_freq
                feature_data['multilabels'] = multilabels
                with open(os.path.join(exp.predictions_dir, 'features_neighbours', f'feature_neighbours_epoch_{epoch:04d}.pkl'), 'wb') as f:
                    pickle.dump(feature_data, f)

        logger.info("Syncing multilabels over processes.")
        # If distributed, sync data
        if world_size > 1:
            cur_device = torch.cuda.current_device()
            if rank == 0:
                num_samples = torch.LongTensor([multilabels.shape[0]]).to(device=cur_device, non_blocking=True)
            else:
                num_samples = torch.LongTensor([0]).to(device=cur_device, non_blocking=True)
            dist.broadcast(num_samples, 0)
            num_samples = num_samples.item()

            if rank != 0:
                multilabels = np.zeros((num_samples,dataset_cfg.num_classes), dtype=int)
                source_ids = np.zeros((num_samples), dtype=int)

            multilabels_dist = torch.from_numpy(multilabels).to(cur_device, non_blocking=True)
            dist.broadcast(multilabels_dist, 0)

            video_ids_dist = torch.from_numpy(source_ids).to(cur_device, non_blocking=True)
            dist.broadcast(video_ids_dist, 0)

            du.synchronize()
            multilabels = multilabels_dist.cpu().numpy()
            source_ids = video_ids_dist.cpu().numpy()

        # Update video_id_to_label
        logger.info(f'New {multilabels = }')
        logger.info(f'For video IDs = {source_ids}')
        global video_id_to_label
        video_id_to_label = {}
        for video_id, label in zip(source_ids, multilabels):
            video_id_to_label[video_id] = label

        # Update train_dataloader with the new labels
        train_dataset = get_torch_dataset('train')
        train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
        train_kit['train_dataloader'] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size(), shuffle=False if train_sampler else True, sampler=train_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=True, worker_init_fn = du.seed_worker)


    if epoch >= num_epochs - train_classifier_balanced_retraining_epochs:
        # freeze base model parameters
        # model state dict doesn't save requires_grad parameters.
        # When resuming, it has to be re-done. That's why we just call it every epoch.
        logger.info(f'Classifier re-training with balanced samples (cRT) for the last {train_classifier_balanced_retraining_epochs} epochs.')
        logger.info(f'cRT: freezing base model')
        train_kit['model'] = model_cfg.freeze_base_model(train_kit['model'])

        logger.info(f'cRT: switching to a balanced dataloader')
        train_dataset = get_torch_dataset('train', class_balanced_sampling=True)
        train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
        train_kit['train_dataloader'] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size(), shuffle=False if train_sampler else True, sampler=train_sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=True, worker_init_fn = du.seed_worker)

    if epoch == num_epochs - train_classifier_balanced_retraining_epochs:
        # re-initialise classifier weights
        logger.info(f'cRT: re-initialising classifier weights')
        train_kit['model'] = model_cfg.initialise_classifier(train_kit['model'])



# optional
def get_optim_policies(model):
    """
    You can set different learning rates on different blocks of the network.
    Refer to `get_optim_policies()` in pyvideoai/models/epic/tsn.py
    """
    return model_cfg.get_optim_policies(model)

import logging
logger = logging.getLogger(__name__)
from pyvideoai.utils.early_stopping import min_value_within_lastN, best_value_within_lastN
# optional
def early_stopping_condition(exp, metric_info):
    if early_stop_patience is None:
        return False

    if not min_value_within_lastN(exp.summary['val_loss'], early_stop_patience):
        best_metric_fieldname = metric_info['best_metric_fieldname']
        best_metric_is_better = metric_info['best_metric_is_better_func']
        if not best_value_within_lastN(exp.summary[best_metric_fieldname], early_stop_patience, best_metric_is_better):
            logger.info(f"Validation loss and {best_metric_fieldname} haven't gotten better for {early_stop_patience} epochs. Stopping training..")
            return True

    return False


from pyvideoai.utils.distributed import get_world_size
def optimiser(params):
    """
    LR should be proportional to the total batch size.
    When distributing, LR should be multiplied by the number of processes (# GPUs)
    Thus, LR = base_LR * batch_size_per_proc * (num_GPUs**2)
    """

    batchsize = batch_size() if callable(batch_size) else batch_size
    world_size = get_world_size()

    max_lr = base_learning_rate * 16 * (4**2)
    learning_rate = min(base_learning_rate * batchsize * (world_size**2), max_lr)

    return torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)

from pyvideoai.utils.lr_scheduling import ReduceLROnPlateauMultiple, GradualWarmupScheduler
def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    if base_learning_rate < 1e-5:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.
    else:
        after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.

        return GradualWarmupScheduler(optimiser, multiplier=1, total_epoch=10, after_scheduler=after_scheduler)

def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length, pretrained = pretrained)

# If you need to extract features, use this. It can be defined in model_cfg too.
#def feature_extract_model(model):
#    from torch.nn import Module
#    class FeatureExtractModel(Module):
#        def __init__(self, model):
#            super().__init__()
#            self.model = model
#        def forward(self, x):
#            return self.model.features(x)
#
#    return FeatureExtractModel(model)

# optional
#def load_pretrained(model):
#    loader.model_load_weights(model, pretrained_path)

def _dataloader_shape_to_model_input_shape(inputs):
    return model_cfg.NCTHW_to_model_input_shape(inputs)

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
    inputs, uids, labels, spatial_idx, _, _ = data
    return inputs, uids, labels, {'spatial_idx': spatial_idx, 'temporal_idx': -1 *torch.ones_like(labels)}


def get_data_unpack_func(split):
    '''
    if split == 'train':
        return _unpack_data
    elif split == 'val':
        return _unpack_data
    elif split == 'multicropval':
        return _unpack_data
    else:
        assert False, 'unknown split'
    '''
    return _unpack_data


def _get_torch_dataset(csv_path, split, class_balanced_sampling):
    mode = dataset_cfg.split2mode[split]



    if split.startswith('multicrop'):
        _test_scale = test_scale
        _test_num_spatial_crops = test_num_spatial_crops
    else:
        _test_scale = val_scale
        _test_num_spatial_crops = val_num_spatial_crops

    if split == 'train':
        global video_id_to_label
    else:
        video_id_to_label = None

    if input_type == 'RGB_video':
        path_prefix = dataset_cfg.video_dir
        return VideoSparsesampleDataset(csv_path, mode,
                input_frame_length,
                train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
                train_horizontal_flip=dataset_cfg.horizontal_flip,
                test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
                crop_size=crop_size,
                mean = model_cfg.input_mean,
                std = model_cfg.input_std,
                normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
                greyscale=False,
                path_prefix=path_prefix,
                sample_index_code=sample_index_code,
                video_id_to_label = video_id_to_label,
                )

    elif input_type == 'flow':
        path_prefix=dataset_cfg.flowframes_dir
        flow = 'RR'
        flow_neighbours = 5
        flow_folder_x = 'u'
        flow_folder_y = 'v'
        return FramesSparsesampleDataset(csv_path, mode,
                input_frame_length,
                train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
                train_horizontal_flip=dataset_cfg.horizontal_flip,
                test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
                crop_size=crop_size,
                mean = model_cfg.input_mean,
                std = model_cfg.input_std,
                normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
                greyscale=False,
                path_prefix=path_prefix,
                sample_index_code=sample_index_code,
                flow = flow,
                flow_neighbours = flow_neighbours,
                flow_folder_x = flow_folder_x,
                flow_folder_y = flow_folder_y,
                video_id_to_label = video_id_to_label,
                )
    elif input_type == 'gulp_rgb':
        gulp_dir_path = os.path.join(dataset_cfg.dataset_root, dataset_cfg.gulp_rgb_dirname[split])

        return GulpSparsesampleDataset(csv_path, mode,
                input_frame_length, gulp_dir_path,
                train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
                train_horizontal_flip=dataset_cfg.horizontal_flip,
                train_class_balanced_sampling=class_balanced_sampling,
                test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
                crop_size=crop_size,
                mean = model_cfg.input_mean,
                std = model_cfg.input_std,
                normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
                greyscale=False,
                sample_index_code=sample_index_code,
                processing_backend = 'pil',
                video_id_to_label = video_id_to_label,
                )
    elif input_type == 'gulp_flow':
        gulp_dir_path = os.path.join(dataset_cfg.dataset_root, dataset_cfg.gulp_flow_dirname[split])
        flow_neighbours = 5

        return GulpSparsesampleDataset(csv_path, mode,
                input_frame_length, gulp_dir_path,
                train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
                train_horizontal_flip=dataset_cfg.horizontal_flip,
                train_class_balanced_sampling=class_balanced_sampling,
                test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
                crop_size=crop_size,
                mean = model_cfg.input_mean,
                std = model_cfg.input_std,
                normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
                greyscale=False,
                sample_index_code=sample_index_code,
                processing_backend = 'pil',
                flow = 'grey',
                flow_neighbours = flow_neighbours,
                video_id_to_label = video_id_to_label,
                )
    else:
        raise ValueError(f'Wrong input_type {input_type}')


def get_torch_dataset(split, class_balanced_sampling=False):
    if input_type == 'RGB_video':
        split_dir = dataset_cfg.video_split_file_dir
    elif input_type == 'flow':
        split_dir = dataset_cfg.flowframes_split_file_dir
    elif input_type == 'gulp_rgb':
        split_dir = dataset_cfg.gulp_rgb_split_file_dir
    elif input_type == 'gulp_flow':
        split_dir = dataset_cfg.gulp_flow_split_file_dir
    else:
        raise ValueError(f'Wrong input_type {input_type}')
    csv_path = os.path.join(split_dir, dataset_cfg.split_file_basename[split])

    return _get_torch_dataset(csv_path, split, class_balanced_sampling)



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
last_activation = 'softmax'   # or, you can pass a callable function like `torch.nn.Softmax(dim=1)`

"""
## For training, (tools/run_train.py)
how to calculate metrics
"""
from pyvideoai.metrics.mAP import Clip_mAPMetric
from pyvideoai.metrics.accuracy import ClipAccuracyMetric, VideoAccuracyMetric
from pyvideoai.metrics.mean_perclass_accuracy import ClipMeanPerclassAccuracyMetric
from pyvideoai.metrics.grouped_class_accuracy import ClipGroupedClassAccuracyMetric
from pyvideoai.metrics.multilabel_accuracy import ClipMultilabelAccuracyMetric
from pyvideoai.metrics.top1_multilabel_accuracy import ClipTop1MultilabelAccuracyMetric, ClipTopkMultilabelAccuracyMetric
from exp_configs.ch_labelsmth.epic100_verb.read_multilabel import read_multilabel, get_val_holdout_set, get_singlemultilabel
from video_datasets_api.epic_kitchens_100.read_annotations import get_verb_uid2label_dict
video_id_to_multilabel = read_multilabel()
_, epic_video_id_to_label = get_verb_uid2label_dict(dataset_cfg.annotations_root)
holdout_video_id_to_label = get_val_holdout_set(epic_video_id_to_label, video_id_to_multilabel)
video_id_to_singlemultilabel = get_singlemultilabel(epic_video_id_to_label, video_id_to_multilabel)

best_metric = ClipAccuracyMetric(video_id_to_label = holdout_video_id_to_label, video_id_to_label_missing_action = 'skip', split='holdoutval')
metrics = {'train': [ClipAccuracyMetric(), ClipMeanPerclassAccuracyMetric(),
            ClipMeanPerclassAccuracyMetric(exclude_classes_less_sample_than=20),
            ClipGroupedClassAccuracyMetric([dataset_cfg.head_classes, dataset_cfg.tail_classes], ['head', 'tail'])],
        'val': [best_metric,
            ClipAccuracyMetric(topk=(5,), video_id_to_label = holdout_video_id_to_label, video_id_to_label_missing_action = 'skip', split='holdoutval'),
            ClipMeanPerclassAccuracyMetric(video_id_to_label = holdout_video_id_to_label, video_id_to_label_missing_action = 'skip', split='holdoutval'),
            ClipMeanPerclassAccuracyMetric(video_id_to_label = holdout_video_id_to_label, video_id_to_label_missing_action = 'skip', split='holdoutval', exclude_classes_less_sample_than=20),
            ClipGroupedClassAccuracyMetric([dataset_cfg.head_classes, dataset_cfg.tail_classes], ['head', 'tail'], video_id_to_label = holdout_video_id_to_label, video_id_to_label_missing_action = 'skip', split='holdoutval'),
            ClipAccuracyMetric(topk=(1,5)),
            ClipMeanPerclassAccuracyMetric(),
            ClipMeanPerclassAccuracyMetric(exclude_classes_less_sample_than=20),
            ClipGroupedClassAccuracyMetric([dataset_cfg.head_classes, dataset_cfg.tail_classes], ['head', 'tail']),
            ClipMultilabelAccuracyMetric(video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip', split='multilabelval'),
            ClipTop1MultilabelAccuracyMetric(video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip', split='multilabelval'),
            ClipTopkMultilabelAccuracyMetric(video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip', split='multilabelval'),
            Clip_mAPMetric(activation='sigmoid', video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip', split='multilabelval'),
            ClipMultilabelAccuracyMetric(video_id_to_label = video_id_to_singlemultilabel, video_id_to_label_missing_action = 'skip', split='singlemultilabelval'),
            ClipTop1MultilabelAccuracyMetric(video_id_to_label = video_id_to_singlemultilabel, video_id_to_label_missing_action = 'skip', split='singlemultilabelval'),
            ClipTopkMultilabelAccuracyMetric(video_id_to_label = video_id_to_singlemultilabel, video_id_to_label_missing_action = 'skip', split='singlemultilabelval'),
            Clip_mAPMetric(activation='sigmoid', video_id_to_label = video_id_to_singlemultilabel, video_id_to_label_missing_action = 'skip', split='singlemultilabelval'),
            ],
        'traindata_testmode': [ClipAccuracyMetric()],
        'trainpartialdata_testmode': [ClipAccuracyMetric()],
        'multicropval': [ClipAccuracyMetric(), VideoAccuracyMetric(topk=(1,5), activation=last_activation)],
        }

"""
## For validation, (tools/run_val.py)
how to gather predictions when --save_predictions is set
"""
from pyvideoai.metrics.metric import ClipPredictionsGatherer, VideoPredictionsGatherer
predictions_gatherers = {'val': ClipPredictionsGatherer(last_activation),
        'traindata_testmode': ClipPredictionsGatherer(last_activation),
        'trainpartialdata_testmode': ClipPredictionsGatherer(last_activation),
        'multicropval': VideoPredictionsGatherer(last_activation),
        }

"""How will you plot"""
#from pyvideoai.visualisations.metric_plotter import DefaultMetricPlotter
#metric_plotter = DefaultMetricPlotter()
from pyvideoai.visualisations.telegram_reporter import DefaultTelegramReporter
telegram_reporter = DefaultTelegramReporter(include_exp_rootdir=True)
