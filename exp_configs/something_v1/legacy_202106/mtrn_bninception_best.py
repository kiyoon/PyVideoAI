import os
import torch
from collections import OrderedDict
import dataset_configs
import model_configs

_script_basename = os.path.basename(os.path.abspath( __file__ ))
_script_basename_wo_ext = os.path.splitext(_script_basename)[0]
_script_name_split = _script_basename_wo_ext.split('-')
dataset_cfg = dataset_configs.load_cfg(_script_name_split[0])
model_cfg = model_configs.load_cfg(_script_name_split[1])

from ...dataloaders.frames_sparsesample_dataset import FramesSparsesampleDataset

from ...dataloaders.trn.transforms import *
from ...dataloaders.trn.datasets_video import return_dataset
from ...dataloaders.trn.dataset import TSNDataSet

input_frame_length = 8


def load_model():
    model = model_cfg.load_model(dataset_cfg.num_classes, input_frame_length)
#    model = torch.nn.DataParallel(model).cuda()
    weights_path = "TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar"
    #weights_path = "TRN_something_RGB_BNInception_TRNmultiscale_segment8_checkpoint.pth.tar"
    checkpoint = torch.load(weights_path)
    
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    #optimiser.load_state_dict(checkpoint["optimiser_state"])

    return model


#def model_predict(model, inputs):
#    verb_outputs = model(inputs)
#    return verb_outputs

def model_predict(model, inputs):
    N, C, T, H, W = inputs.shape
    
    # N, C, T, H, W -> N, T, C, H, W -> N, TC, H, W
    #verb_outputs, noun_outputs = model(inputs.permute(0,2,1,3,4).reshape((N, -1, H, W)))
    verb_outputs = model(inputs.permute(0,2,1,3,4).reshape((N, -1, H, W)))
    return verb_outputs


def unpack_data(data):
    inputs, uids, labels, spatial_idx, _, _, _ = data
    return dataset_cfg._data_to_gpu(inputs, uids, labels) + (spatial_idx,)
    #inputs, labels = data
    #return dataset_cfg._data_to_gpu(inputs, labels, labels) + (labels,)


def get_torch_dataset(csv_path, mode):
    return FramesSparsesampleDataset(csv_path, mode,
            input_frame_length, frame_start_idx=1, test_num_spatial_crops=5,
            mean = model_cfg.input_mean, std = model_cfg.input_std,
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr)


def _get_torch_dataset_split(split):

    mode = dataset_cfg.split2mode[split]
    csv_path = os.path.join(dataset_cfg.frames_split_file_dir, dataset_cfg.split_file_basename[split])
    print(dataset_cfg.frames_split_file_dir)

    return get_torch_dataset(csv_path, mode)
    
'''
def get_torch_datasets(splits=['train', 'val', 'multicropval']):
    assert splits=="val"

    scale_size = 256
    crop_size = 224
    categories, train_list, val_list, root_path, prefix = return_dataset("something", "RGB")
    return TSNDataSet(root_path, val_list, num_segments=input_frame_length,
                   new_length=1,
                   modality="RGB",
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       GroupNormalize(model_cfg.input_mean, model_cfg.input_std)
                   ]))
'''
 
                                                                                                                    
def get_torch_datasets(splits=['train', 'val', 'multicropval']):

    if type(splits) == list:
        torch_datasets = {}
        for split in splits:
            dataset = _get_torch_dataset_split(split)
            torch_datasets[split] = dataset

        return torch_datasets

    elif type(splits) == str:
        return _get_torch_dataset_split(splits)

    else:
        raise ValueError('Wrong type of splits argument. Must be list or string')
