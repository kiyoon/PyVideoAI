#from dataloaders.DALI_video_loader import DALILoader
#from dataloaders.video_classification_dataset import VideoClassificationDataset
#from dataloaders.video_sparsesample_dataset import VideoSparsesampleDataset
#from dataloaders.frames_sparsesample_dataset import FramesSparsesampleDataset

import argparse
from pyvideoai import config
import dataset_configs, model_configs, exp_configs
import os
import pickle

import cv2
from PIL import Image

import csv

from experiment_utils.argparse_utils import add_exp_arguments
from experiment_utils import ExperimentBuilder

import numpy as np

import torch

# org
x0 = 20
y0 = 20
dy = 30

# fontScale
fontScale = 0.90

# White colour in BGR
default_colour = (255, 255, 255)

# Line thickness of 2 px
thickness = 1

#gif_duration = round(1000 / (dataset_cfg.input_target_fps / dataset_cfg.input_frame_stride))
gif_duration = 200

from video_datasets_api.ucf101.video_id import get_unique_video_ids_from_videos

def get_parser():
    parser = argparse.ArgumentParser(description="Visualise success and failures. For now, only support EPIC",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_exp_arguments(parser, dataset_configs.available_datasets, model_configs.available_models, root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='ucf101data_infer_kinetics400model', model_default='i3d_resnet50', name_default='3scrop_5tcrop_split1')
    parser.add_argument("-l", "--load_epoch", type=int, default=None, help="Load from checkpoint. Set to -1 to load from the last checkpoint.")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Evaluate using 1 clip or 30 clips.")
    return parser


def save_dataloader_csv(csv_path, frames_dir, video_ids, video_labels):
    video_list, _ = get_unique_video_ids_from_videos(frames_dir, True)
    with open(csv_path, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow([0])

        for video_id, video_label in zip(video_ids, video_labels):
            sample_frames_dir = os.path.join(frames_dir, f'{video_list[video_id]}')
            num_frames = len([name for name in os.listdir(sample_frames_dir) if os.path.isfile(os.path.join(sample_frames_dir, name))])
            writer.writerow([f"{sample_frames_dir}/{{:05d}}.jpg", video_id, video_label, 0, num_frames-1])


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)


def video_to_image_grid(video, num_frames_row, row_horizontal=True):
    if row_horizontal:
        stack_1 = np.hstack
        stack_2 = np.vstack
    else:
        # vertical order
        stack_1 = np.vstack
        stack_2 = np.hstack

    rows = []
    for frames in grouped(video, num_frames_row):
        rows.append(stack_1(frames))

    return stack_2(rows)

def get_pred_info(video_predictions, video_ids, k=5):
    assert len(video_predictions) == len(video_ids)
    num_samples, num_classes = video_predictions.shape

    topk_labels = np.zeros((num_samples,k), dtype=int)
    topk_preds = np.zeros((num_samples,k), dtype=float)

    for i, (video_prediction, video_id) in enumerate(zip(video_predictions, video_ids)):
        topk_labels[i] = video_prediction.argsort()[-k:][::-1]
        topk_preds[i] = video_prediction[topk_labels[i]]

    return topk_labels, topk_preds

def visualise_pred_info(image, video_label, video_id, topk_label, topk_pred, data_class_keys, model_class_keys):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX


    assert len(topk_label) == len(topk_pred)
    k = len(topk_label)

    line=0
    video_info = 'Video ID: {:d}'.format(video_id)

    image = cv2.putText(image, video_info, (x0, y0+dy*line), font,
                       fontScale, default_colour, thickness, cv2.LINE_AA)

    line += 1
    image = cv2.putText(image, 'UCF-101 Ground truth: {:s}'.format(data_class_keys[video_label]), (x0, y0+dy*line), font,
                       fontScale, default_colour, thickness, cv2.LINE_AA)

    
    for i, (label, pred) in enumerate(zip(topk_label, topk_pred)):
        line += 1
        colour=default_colour
        image = cv2.putText(image, '{:s} {:.4f}%'.format(model_class_keys[label], pred * 100), (x0, y0+dy*line), font,
                           fontScale, colour, thickness, cv2.LINE_AA)
        
    return image

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()


    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, telegram_key_ini = config.KEY_INI_PATH)
    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name)

    if args.load_epoch is None:
        predictions_file_path = os.path.join(exp.predictions_dir, 'pretrained_%sval.pkl' % (args.mode))
    else:
        predictions_file_path = os.path.join(exp.predictions_dir, 'epoch_%04d_%sval.pkl' % (args.load_epoch, args.mode))

    with open(predictions_file_path, 'rb') as f:
        predictions = pickle.load(f)

    if cfg.input_frame_length == 32:
        vis_frame_x = 8
        vis_frame_y = 4
    elif cfg.input_frame_length == 8:
        vis_frame_x = 4
        vis_frame_y = 2
    else:
        raise ValueError('only support num_frames == 32 and 8')

    vis_width = cfg.crop_size * vis_frame_x
    vis_height = vis_width // 4 * 3        # 4:3 aspect ratio
    assert vis_height >= cfg.crop_size * vis_frame_y, 'Not enough pixels to put the frames'

    upscale_factor = 2
    vis_width_gif = cfg.crop_size * upscale_factor
    vis_height_gif = vis_width_gif // 2 * 3             # 2:3 aspect ratio

    video_predictions = predictions['video_predictions']
    video_labels = predictions['video_labels']
    video_ids = predictions['video_ids']

#    max_pred_per_sample = np.max(video_predictions, axis=1)


    ################################################################
    # Get video-based info

    topk_labels, topk_preds = get_pred_info(video_predictions, video_ids)

    best_idxs = topk_preds[:,0].argsort()[-20:][::-1]
    worst_idxs = topk_preds[:,0].argsort()[:20]

    if args.load_epoch is None:
        output_dir = os.path.join(exp.plots_dir, 'most_and_least_confident_pretrained')
    else:
        output_dir = os.path.join(exp.plots_dir, 'most_and_least_confident_epoch_{:04d}'.format(args.load_epoch))
    output_dir_jpg = os.path.join(output_dir, 'jpg')
    output_dir_gif = os.path.join(output_dir, 'gif')
    output_dir_jpg_best = os.path.join(output_dir_jpg, 'most_confident')
    output_dir_gif_best = os.path.join(output_dir_gif, 'most_confident')
    output_dir_jpg_worst = os.path.join(output_dir_jpg, 'least_confident')
    output_dir_gif_worst = os.path.join(output_dir_gif, 'least_confident')
    os.makedirs(output_dir_jpg_best, exist_ok=True)
    os.makedirs(output_dir_gif_best, exist_ok=True)
    os.makedirs(output_dir_jpg_worst, exist_ok=True)
    os.makedirs(output_dir_gif_worst, exist_ok=True)
    
    ################################################################
    print("Saving centre clip visualisations with multicrop prediction results.")


    for mode in ['most_confident', 'least_confident']:
        print("{:s} predictions..".format(mode))

        csv_path = os.path.join(output_dir, '{:s}_samples.csv'.format(mode))

        if mode == 'most_confident':
            filter_idxs = best_idxs
        else:
            filter_idxs = worst_idxs

        best_video_ids = video_ids[filter_idxs].tolist()   # for later, to get best video rank from video id.
        save_dataloader_csv(csv_path, cfg.dataset_cfg.frames_dir, video_ids[filter_idxs], video_labels[filter_idxs])

        dataset = cfg.get_torch_dataset(csv_path, 'val')
        data_unpack_func = cfg.unpack_data
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=4, pin_memory=True, drop_last=False)


        with torch.no_grad():

            for it, data in enumerate(dataloader):
                print("Iteration {:d} / {:d}".format(it+1, len(dataloader)))
                inputs, labels, uids, curr_batch_size, spatial_temporal_idxs = data_unpack_func(data)

                labels = labels.cpu().numpy()
                uids = uids.cpu().numpy()

                inputs = inputs.cpu().numpy()
                inputs = inputs.transpose((0,2,3,4,1))      # B, F, H, W, C
                inputs = (inputs * dataset.std + dataset.mean)
                if dataset.normalise:
                    inputs = inputs * 255
                inputs = inputs.astype(np.uint8)
                inputs = inputs[...,::-1]   # BGR

                for input_vid, label, uid in zip(inputs, labels, uids):
                    best_idx = best_video_ids.index(uid)
                    topk_idx = np.where(video_ids == uid)
                    assert len(topk_idx) == 1
                    topk_idx = int(topk_idx[0])
                    topk_label = topk_labels[topk_idx]
                    topk_pred = topk_preds[topk_idx]

                    clip_based_info = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
                    visualise_pred_info(clip_based_info, label, uid, topk_label, topk_pred, cfg.dataset_cfg.ucf101_class_keys, cfg.dataset_cfg.kinetics400_class_keys)

                    info_overlay_rgb = clip_based_info.copy()[...,::-1]         # only texts and no images. Used for gif frames.
                    info_overlay_rgb = cv2.resize(info_overlay_rgb[:,:vis_width//2,:], (vis_width_gif, vis_height_gif), interpolation=cv2.INTER_CUBIC)  # cut the right half out and downsize


                    x_offset=0
                    y_offset=vis_height-cfg.crop_size*vis_frame_y

                    image_grid = video_to_image_grid(input_vid, vis_frame_x)
                    clip_based_info[y_offset:,x_offset:,:] = image_grid

                    #cv2.imshow('a', clip_based_info)
                    #cv2.waitKey(0)

                    cv2.imwrite(os.path.join(output_dir_jpg, mode, '{:s}_{:02d}-uid_{:05d}.jpg'.format(mode, best_idx, uid)), clip_based_info)

                    # GIF
                    x_offset=0
                    y_offset=vis_height_gif-cfg.crop_size*upscale_factor

                    gif_frames = []
                    for frame in input_vid:
                        frame_rgb = frame[...,::-1]
                        upscaled_size = tuple(upscale_factor*x for x in frame_rgb.shape[:2])     # height, width
                        frame_rgb_upscaled = cv2.resize(frame_rgb, upscaled_size[::-1], interpolation=cv2.INTER_CUBIC)  # upscaled_size[::-1] = (width, height)
                        gif_frame = info_overlay_rgb.copy()
                        gif_frame[y_offset:y_offset+upscaled_size[0],x_offset:x_offset+upscaled_size[1],:] = frame_rgb_upscaled
                        gif_frames.append(Image.fromarray(gif_frame))

                    save_name = '{:s}_{:02d}-uid_{:05d}.gif'.format(mode, best_idx, uid)
                    gif_frames[0].save(os.path.join(output_dir_gif, mode, save_name),
                            save_all=True, append_images=gif_frames[1:], optimize=True, duration=gif_duration, loop=0)

