import os
import pickle
import numpy as np
import pandas as pd

import torch

from pyvideoai.dataloaders.feature_dataset import FeatureDataset
from pyvideoai.utils.losses.softlabel import MaskedBinaryCrossEntropyLoss
from pyvideoai.utils.losses.single_positive_multilabel import AssumeNegativeLossWithLogits, WeakAssumeNegativeLossWithLogits, BinaryLabelSmoothLossWithLogits, BinaryNegativeLabelSmoothLossWithLogits, EntropyMaximiseLossWithLogits, BinaryFocalLossWithLogits
from pyvideoai.utils.stdout_logger import OutputLogger
import multi_label_ar

from pyvideoai.utils.distributed import get_rank, get_world_size


num_epochs = int(os.getenv('VAI_NUM_EPOCHS', default=1000))

#batch_size = 8  # per process (per GPU)
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    return int(os.getenv('VAI_BATCHSIZE', 64))

def val_batch_size():
    def default():
        devices=list(range(torch.cuda.device_count()))
        vram = min([torch.cuda.get_device_properties(device).total_memory for device in devices])
        if vram > 20e+9:
            return 2048
        elif vram > 10e+9:
            return 1024
        return 64
    return default()

feature_input_type = 'concat_RGB_flow'          # RGB / flow / concat_RGB_flow
loss_type = 'crossentropy'  # assume_negative, weak_assume_negative, binary_labelsmooth, binary_negative_labelsmooth, binary_focal
                            # entropy_maximise
                            # mask_binary_ce
                            # pseudo_single_binary_ce

loss_types_pseudo_generation = ['mask_binary_ce',
            'pseudo_single_binary_ce',
            ]
loss_types_masked_pseudo = ['mask_binary_ce']

# Pseudo label types
pseudo_label_type = 'neighbours'    # neighbours, multilabel_cooccurance

# Neighbour search settings for pseudo labelling
# int
num_neighbours = 10
thr = 0.2
# If you want the parameters to be different per class.
# dictionary with key being class index, with size num_classes.
# If key not found, use the default setting above.
num_neighbours_per_class = {}
thr_per_class = {}


l2_norm = False


#clip_grad_max_norm = 20
learning_rate = float(os.getenv('VAI_LR', 5e-6))


def get_input_feature_dim():
    input_feature_dim = 2048
    if feature_input_type == 'concat_RGB_flow':
        input_feature_dim *= 2
    return input_feature_dim


import torch.distributed as dist
import pyvideoai.utils.distributed as du
from multi_label_ar.neighbours import get_neighbours

def get_features(split):
    input_feature_dim = get_input_feature_dim()
    if feature_input_type == 'RGB':
        feature_pickle_path = dataset_cfg.RGB_features_pickle_path[split]

        with open(feature_pickle_path, 'rb') as f:
            d = pickle.load(f)

        video_ids, labels, features = d['video_ids'], d['labels'], d['clip_features']
        if features.shape[1] != input_feature_dim:
            # features are not averaged. Average now
            features = np.mean(features, axis=1)

    elif feature_input_type == 'flow':
        feature_pickle_path = dataset_cfg.flow_features_pickle_path[split]

        with open(feature_pickle_path, 'rb') as f:
            d = pickle.load(f)

        video_ids, labels, features = d['video_ids'], d['labels'], d['clip_features']
        if features.shape[1] != input_feature_dim:
            # features are not averaged. Average now
            features = np.mean(features, axis=1)


    elif feature_input_type == 'concat_RGB_flow':
        RGB_feature_pickle_path = dataset_cfg.RGB_features_pickle_path[split]
        flow_feature_pickle_path = dataset_cfg.flow_features_pickle_path[split]

        with open(RGB_feature_pickle_path, 'rb') as f:
            d = pickle.load(f)

        RGB_video_ids, RGB_labels, RGB_features = d['video_ids'], d['labels'], d['clip_features']
        if RGB_features.shape[1] != 2048:
            # features are not averaged. Average now
            RGB_features = np.mean(RGB_features, axis=1)

        with open(flow_feature_pickle_path, 'rb') as f:
            d = pickle.load(f)

        flow_video_ids, flow_labels, flow_features = d['video_ids'], d['labels'], d['clip_features']
        assert RGB_video_ids.shape[0] == RGB_labels.shape[0] == RGB_features.shape[0] == flow_video_ids.shape[0] == flow_labels.shape[0] == flow_features.shape[0]

        if flow_features.shape[1] != 2048:
            # features are not averaged. Average now
            flow_features = np.mean(flow_features, axis=1)

        # Concatenate RGB and flow features.
        flow_video_id_to_idx = {}
        for idx, video_id in enumerate(flow_video_ids):
            assert video_id not in flow_video_id_to_idx
            flow_video_id_to_idx[video_id] = idx

        # Loop over RGB features and find flow features. The array ordering may be different.
        concat_RGB_flow_features = np.zeros((RGB_features.shape[0], input_feature_dim), dtype=RGB_features.dtype)
        for idx, (video_id, RGB_feature) in enumerate(zip(RGB_video_ids, RGB_features)):
            flow_idx = flow_video_id_to_idx[video_id]
            concat_feature = np.concatenate((RGB_feature, flow_features[flow_idx]))
            concat_RGB_flow_features[idx] = concat_feature

        video_ids, labels, features = RGB_video_ids, RGB_labels, concat_RGB_flow_features

    else:
        raise ValueError(f'Not recognised {feature_input_type = }')

    assert features.shape[1] == input_feature_dim
    assert video_ids.shape[0] == labels.shape[0] == features.shape[0]

    logger.info(f'{video_ids.shape[0]} samples in {split = } before filtering samples.')

    # Keep only the videos according to the CSV.
    keep_video_ids = []

    split_file = os.path.join(dataset_cfg.gulp_rgb_split_file_dir, dataset_cfg.split_file_basename[split])
    logger.info(f'Reading {split_file} to get the video IDs to keep.')
    with open(split_file, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        video_id = int(line.split(' ')[1])
        keep_video_ids.append(video_id)

    mask = np.isin(video_ids, keep_video_ids)
    video_ids = video_ids[mask]
    labels = labels[mask]
    features = features[mask]

    logger.info(f'{video_ids.shape[0]} samples in {split = } after filtering samples.')

    return video_ids, labels, features


def generate_train_pseudo_labels():

    rank = get_rank()
    world_size = get_world_size()

    video_ids, labels, features = get_features('train')
    feature_data = {'video_ids': video_ids,
            'labels': labels,
            'clip_features': features,
            }

    if loss_type in loss_types_pseudo_generation:
        if rank == 0:
            if pseudo_label_type == 'neighbours':
                logger.info(f'Generating train pseudo labels using {feature_input_type} features.')
                soft_labels_per_num_neighbour = {}    # key: num_neighbours
                set_num_neighbours = set(num_neighbours_per_class.values()) | {num_neighbours}
                for num_neighbour in set_num_neighbours:
                    with OutputLogger(multi_label_ar.neighbours.__name__, 'INFO'):
                        nc_freq, _, _ = get_neighbours(feature_data['clip_features'], feature_data['clip_features'], feature_data['labels'], feature_data['labels'], num_neighbour, l2_norm=l2_norm)
                    #neighbours_ids = []
                    soft_label = []
                    target_ids = feature_data['video_ids']
                    #source_ids = feature_data['video_ids']

                    for target_idx, target_id in enumerate(target_ids):
                        #n_ids = [source_ids[x] for x in features_neighbours[target_idx]]
                        sl = nc_freq[target_idx]
                        sl = sl / sl.sum()
                        #neighbours_ids.append(n_ids)
                        soft_label.append(sl)

                    #neighbours_ids = np.array(neighbours_ids)
                    soft_labels_per_num_neighbour[num_neighbour] = np.array(soft_label)

                # generate multilabels
                # For each class, use different soft labels generated with different num_neighbours and thr.
                #multilabels = MinCEMultilabelLoss.generate_multilabels_numpy(soft_labels, thr, feature_data['labels'])
                multilabels = []
                for target_idx, (video_id, singlelabel) in enumerate(zip(feature_data['video_ids'], feature_data['labels'])):
                    if singlelabel in num_neighbours_per_class:
                        soft_label = soft_labels_per_num_neighbour[num_neighbours_per_class[singlelabel]][target_idx]
                    else:
                        soft_label = soft_labels_per_num_neighbour[num_neighbours][target_idx]

                    if singlelabel in thr_per_class:
                        th = thr_per_class[singlelabel]
                    else:
                        th = thr

                    multilabel = (soft_label > th).astype(int)
                    multilabel[singlelabel] = 1
                    multilabels.append(multilabel)

                multilabels = np.stack(multilabels)


            elif pseudo_label_type == 'multilabel_cooccurance':
                logger.info('Using multilabel co-occurance data to generate per-class pseudo labels.')
                cooc_csv = os.path.join(multi_label_ar.MODULE_DIR, '..', 'notebooks', 'kiyoon', 'multilabel_cooccurance_normalised_20220427.csv')
                cooc_df = pd.read_csv(cooc_csv)
                cooc_df.set_index('key', inplace=True)

                def thr_multilabel_cooccurance(df_cooc_normalised, thr):
                    """
                    Read multilabel_cooccurance_normalised_20220427.csv file and threshold for each class (row).
                    For generating pseudo labels.
                    Return: dict[str, list[str]]
                    """
                    ret_dict = {}
                    for index, row in df_cooc_normalised.iterrows():
                        ret_dict[row.name] = list(row[row > thr].keys())
                    return ret_dict

                cooc_thresholded = thr_multilabel_cooccurance(cooc_df, 0.5)

                multilabels = []
                for target_idx, (video_id, singlelabel) in enumerate(zip(feature_data['video_ids'], feature_data['labels'])):
                    multilabel = np.zeros(dataset_cfg.num_classes, dtype=int)
                    singlelabel_str = dataset_cfg.class_keys[singlelabel]

                    if singlelabel_str in cooc_thresholded.keys():
                        for label_str in cooc_thresholded[singlelabel_str]:
                            multilabel[dataset_cfg.class_keys_to_label_idx[label_str]] = 1

                    multilabel[singlelabel] = 1
                    multilabels.append(multilabel)

                multilabels = np.stack(multilabels)


            else:
                raise ValueError(f'{pseudo_label_type = } not recognised.')

            assert multilabels.shape == (len(feature_data['labels']), dataset_cfg.num_classes)

            # format multilabels properly based on loss
            if loss_type in loss_types_masked_pseudo:
                logger.info('Turning multilabels to mask out sign (-1) and including one single label')
                multilabels = -multilabels      # negative numbers mean masks. Mask out the relevant verbs.
                for idx, label in enumerate(feature_data['labels']):
                    multilabels[idx, label] = 1     # Make the actual single label GT the only label.
            elif loss_type.lower().startswith('mask'):
                logger.error(f'You are using the mask loss {loss_type} but the labels are not in mask format!!')
            else:
                logger.info('Including all multilabels and singlelabel')


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
#                source_ids = np.zeros((num_samples), dtype=int)

            multilabels_dist = torch.from_numpy(multilabels).to(cur_device, non_blocking=True)
            dist.broadcast(multilabels_dist, 0)

#            video_ids_dist = torch.from_numpy(source_ids).to(cur_device, non_blocking=True)
#            dist.broadcast(video_ids_dist, 0)

            du.synchronize()
            multilabels = multilabels_dist.cpu().numpy()
#            source_ids = video_ids_dist.cpu().numpy()

        # Update video_id_to_label
        logger.info(f'New {multilabels = }')
        logger.info(f'For video IDs = {video_ids}')

        return video_ids, multilabels, features

    elif loss_type.lower().startswith('mask') or loss_type.lower().startswith('pseudo'):
        logger.error(f'loss_type is {loss_type} but not generating pseudo labels?')
        return video_ids, labels, features
    else:
        logger.info('NOT generating pseudo labels')
        return video_ids, labels, features




labelsmooth_factor = 0.1

#### OPTIONAL
def get_criterion(split):
    if loss_type == 'assume_negative':
        return AssumeNegativeLossWithLogits()
    elif loss_type == 'weak_assume_negative':
        return WeakAssumeNegativeLossWithLogits(num_classes = dataset_cfg.num_classes)
    elif loss_type == 'binary_labelsmooth':
        return BinaryLabelSmoothLossWithLogits(smoothing=labelsmooth_factor)
    elif loss_type == 'binary_negative_labelsmooth':
        return BinaryNegativeLabelSmoothLossWithLogits(smoothing=labelsmooth_factor)
    elif loss_type == 'binary_focal':
        return BinaryFocalLossWithLogits()
    elif loss_type == 'entropy_maximise':
        return EntropyMaximiseLossWithLogits()
    elif loss_type == 'mask_binary_ce':
        if split == 'train':
            return MaskedBinaryCrossEntropyLoss()
        else:
            return AssumeNegativeLossWithLogits()
    elif loss_type == 'pseudo_single_binary_ce':
        # make sure you pass pseudo+single labels
        return AssumeNegativeLossWithLogits()
    else:
        return ValueError(f'Wrong loss type: {loss_type}')

#
#def epoch_start_script(epoch, exp, args, rank, world_size, train_kit):
#    return None

# optional
#def get_optim_policies(model):
#    """
#    You can set different learning rates on different blocks of the network.
#    Refer to `get_optim_policies()` in pyvideoai/models/epic/tsn.py
#    """
#    return model_cfg.get_optim_policies(model)

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


def optimiser(params):
    """
    LR should be proportional to the total batch size.
    When distributing, LR should be multiplied by the number of processes (# GPUs)
    Thus, LR = base_LR * batch_size_per_proc * (num_GPUs**2)
    """

    #return torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)
    return torch.optim.Adam(params, lr = learning_rate)


def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    return None


def load_model():
    num_layers = int(os.getenv('VAI_NUMLAYERS', 2))
    num_units = int(os.getenv('VAI_NUMUNITS', 1024))

    return model_cfg.load_model(dataset_cfg.num_classes, get_input_feature_dim(), num_layers=num_layers, num_units=num_units)

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
#    return

def _dataloader_shape_to_model_input_shape(inputs):
    return inputs

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
    inputs, uids, labels, _ = data
    return inputs, uids, labels, {'spatial_idx': 0, 'temporal_idx': -1 *torch.ones_like(labels)}


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


def get_torch_dataset(split):

    if split == 'train':
        video_ids, labels, features = generate_train_pseudo_labels()

    elif split == 'val':
        video_ids, labels, features = get_features('val')

    else:
        raise ValueError(f'Only train and val splits are supported. Got {split}')

    return FeatureDataset(video_ids, labels, features)


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
from pyvideoai.metrics import ClipIOUAccuracyMetric, ClipF1MeasureMetric
from pyvideoai.metrics.mAP import Clip_mAPMetric
from pyvideoai.metrics.accuracy import ClipAccuracyMetric, VideoAccuracyMetric
from pyvideoai.metrics.multilabel_accuracy import ClipMultilabelAccuracyMetric
from pyvideoai.metrics.top1_multilabel_accuracy import ClipTop1MultilabelAccuracyMetric
from exp_configs.ch_labelsmth.epic100_verb.read_multilabel import read_multiverb, get_val_holdout_set
from video_datasets_api.epic_kitchens_100.read_annotations import get_verb_uid2label_dict
video_id_to_multilabel = read_multiverb(dataset_cfg.narration_id_to_video_id)
_, epic_video_id_to_label = get_verb_uid2label_dict(dataset_cfg.annotations_root)
holdout_video_id_to_label = get_val_holdout_set(epic_video_id_to_label, video_id_to_multilabel)

best_metric = ClipAccuracyMetric(video_id_to_label = holdout_video_id_to_label, video_id_to_label_missing_action = 'skip', split='holdoutval')
metrics = {'train': [ClipAccuracyMetric(),
                    ],
        'val': [best_metric,
            ClipAccuracyMetric(topk=(5,), video_id_to_label = holdout_video_id_to_label, video_id_to_label_missing_action = 'skip', split='holdoutval'),
            ClipAccuracyMetric(topk=(1,5)),
            ClipMultilabelAccuracyMetric(video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip', split='multilabelval'),
            ClipTop1MultilabelAccuracyMetric(video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip', split='multilabelval'),
            ClipIOUAccuracyMetric(activation='sigmoid', video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip', split='multilabelval'),
            ClipF1MeasureMetric(activation='sigmoid', video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip', split='multilabelval'),
            Clip_mAPMetric(activation='sigmoid', video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip', split='multilabelval'),
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
predictions_gatherers = {'val': ClipPredictionsGatherer(),
        'multicropval': VideoPredictionsGatherer(),
        }

"""How will you plot"""
#from pyvideoai.visualisations.metric_plotter import DefaultMetricPlotter
#metric_plotter = DefaultMetricPlotter()
from pyvideoai.visualisations.telegram_reporter import DefaultTelegramReporter
telegram_reporter = DefaultTelegramReporter(include_exp_rootdir = True, ignore_figures = True)
