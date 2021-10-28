#from dataloaders.DALI_video_loader import DALILoader
#from dataloaders.video_classification_dataset import VideoClassificationDataset
#from dataloaders.video_sparsesample_dataset import VideoSparsesampleDataset
#from dataloaders.frames_sparsesample_dataset import FramesSparsesampleDataset

import argparse
import os
import pickle

import cv2
from PIL import Image

import csv

from experiment_utils.argparse_utils import add_exp_arguments
from experiment_utils import ExperimentBuilder

import numpy as np

import torch
from pyvideoai import config
import model_configs, dataset_configs, exp_configs

from pyvideoai.utils import misc

import logging
logger = logging.getLogger(__name__)

# org
x0 = 20
y0 = 20
dy = 30

# fontScale
fontScale = 0.55

# White colour in BGR
default_colour = (255, 255, 255)

# Line thickness of 2 px
thickness = 1

#gif_duration = round(1000 / (dataset_cfg.input_target_fps / dataset_cfg.input_frame_stride))
gif_duration = 200


def get_parser():
    parser = argparse.ArgumentParser(description="Visualise success and failures. For now, only support EPIC",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_exp_arguments(parser, 
            root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='hmdb', model_default='i3d_resnet50', name_default='crop224_8x8_largejit_plateau_1scrop5tcrop_split1',
            dataset_channel_choices=dataset_configs.available_channels, model_channel_choices=model_configs.available_channels, exp_channel_choices=exp_configs.available_channels)
    parser.add_argument("-l", "--load_epoch", type=int, default=None, help="Load from checkpoint. Set to -1 to load from the last checkpoint.")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Evaluate using 1 clip or 30 clips.")
    parser.add_argument("-s", "--split", type=str, default=None,  help="Which split to use to evaluate? Default is auto (None)")
    parser.add_argument("-i", "--input_size", type=int, default=None, help="Input size to the model.")
    parser.add_argument('-v', '--version', default='last', help='ExperimentBuilder version')
    parser.add_argument("-n", "--num_samples", type=int, default=20, help="How many best and worst each to sample.")
    parser.add_argument("--skip_novisual", action='store_true', help="For multicrop, it will generate visualisations with overall prediction scores with no visuals in it. Skip them all.")
    parser.add_argument("--ignore_top5", action='store_true', help="For failures, it will only visualise samples that failed to predict correct class within top5 predictions.")
    return parser


#def save_dataloader_csv(csv_path, sample_path_prefix, video_ids, video_labels, dataloader_type='video_clip'):
#    with open(csv_path, 'w') as f:
#        writer = csv.writer(f, delimiter=' ')
#
#        if dataloader_type in ['video_clip', 'sparse_video']:
#            for video_id, video_label in zip(video_ids, video_labels):
#                writer.writerow([os.path.join(sample_path_prefix, '{:05d}.mp4'.format(video_id)), video_id, video_label])
#        elif dataloader_type == 'sparse_frames':
#            for video_id, video_label in zip(video_ids, video_labels):
#                frames_dir = os.path.join(sample_path_prefix, '{:05d}'.format(video_id))
#                num_frames = len([name for name in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, name))])
#                writer.writerow([frames_dir, video_id, video_label, num_frames])


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

def get_pred_info(video_predictions, video_labels, video_ids, k=5):
    assert len(video_predictions) == len(video_labels) == len(video_ids)
    num_samples, num_classes = video_predictions.shape

    predicted_scores_for_ground_truth_class = np.zeros(num_samples, dtype=np.float64)
    topk_labels = np.zeros((num_samples,k), dtype=np.int64)
    topk_preds = np.zeros((num_samples,k), dtype=np.float64)

    for i, (video_prediction, video_label, video_id) in enumerate(zip(video_predictions, video_labels, video_ids)):
        predicted_scores_for_ground_truth_class[i] = video_prediction[video_label]
        topk_labels[i] = video_prediction.argsort()[-k:][::-1]
        topk_preds[i] = video_prediction[topk_labels[i]]

    return predicted_scores_for_ground_truth_class, topk_labels, topk_preds

def visualise_pred_info(image, video_label, video_id, predicted_score_for_ground_truth_class, topk_label, topk_pred, class_keys, spatial_idx = None, temporal_idx = None):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX


    assert len(topk_label) == len(topk_pred)
    k = len(topk_label)

    line=0
    video_info = 'Video ID: {:d}'.format(video_id)
    if spatial_idx is not None:
        if temporal_idx is not None:
            spatial_idx_to_str = ['centre', 'left', 'right']
        else:
            spatial_idx_to_str = ['centre', 'top left', 'top right', 'bottom left', 'bottom right', 'centre flipped', 'top left flipped', 'top right flipped', 'bottom left flipped', 'bottom right flipped']
        video_info += ', spatially {:s}'.format(spatial_idx_to_str[spatial_idx])
    if temporal_idx is not None:
        video_info += ', temporal index {:d}'.format(temporal_idx)

    image = cv2.putText(image, video_info, (x0, y0+dy*line), font,
                       fontScale, default_colour, thickness, cv2.LINE_AA)

    line += 1
    image = cv2.putText(image, 'Ground truth: {:s} (prediction: {:.4f}%)'.format(class_keys[video_label], predicted_score_for_ground_truth_class * 100), (x0, y0+dy*line), font,
                       fontScale, default_colour, thickness, cv2.LINE_AA)

    
    for i, (label, pred) in enumerate(zip(topk_label, topk_pred)):
        line += 1
        if i==0:
            if label == video_label:
                # correct prediction
                colour=(0,255,0)
            else:
                # incorrect prediction
                colour=(0,0,255)

        else:
            colour=default_colour
        image = cv2.putText(image, '{:s} {:.4f}%'.format(class_keys[label], pred * 100), (x0, y0+dy*line), font,
                           fontScale, colour, thickness, cv2.LINE_AA)
        
    return image

def main():
    parser = get_parser()
    args = parser.parse_args()

    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name, args.dataset_channel, args.model_channel, args.experiment_channel)
    dataset_cfg = cfg.dataset_cfg
    model_cfg = cfg.model_cfg

    metrics = cfg.dataset_cfg.task.get_metrics(cfg)
    summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_from_metrics(metrics)

    if args.version == 'last':
        _expversion = -2    # choose the last version
    else:
        _expversion = int(args.version)


    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name,
            summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes,
            version=_expversion,
            telegram_key_ini = config.KEY_INI_PATH)

    if args.load_epoch == -1:
        exp.load_summary()
        load_epoch = int(exp.summary['epoch'].iloc[-1])
    elif args.load_epoch == -2:
        exp.load_summary()
        best_metric, best_metric_fieldname = metrics.get_best_metric_and_fieldname()
        logger.info(f'Using the best metric from CSV field `{best_metric_fieldname}`')
        load_epoch = int(exp.get_best_model_stat(best_metric_fieldname, best_metric.is_better)['epoch'])
    elif args.load_epoch >= 0:
        load_epoch = args.load_epoch
    else:
        raise ValueError(f"Wrong args.load_epoch value: {args.load_epoch}")

    # Dataset
    if args.split is not None:
        split = args.split
    else:   # set split automatically
        if args.mode == 'oneclip':
            split = 'val'
        else:   # multicrop
            split = 'multicropval'
    
    predictions_file_path = os.path.join(exp.predictions_dir, f'epoch_{load_epoch:04d}_{split}_{args.mode}.pkl')

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

    if args.input_size is None:
        args.input_size = cfg.crop_size
        
    vis_width = args.input_size * vis_frame_x
    vis_height = vis_width // 4 * 3        # 4:3 aspect ratio
    assert vis_height >= args.input_size * vis_frame_y, 'Not enough pixels to put the frames'

    upscale_factor = 2
    vis_width_gif = args.input_size * upscale_factor
    vis_height_gif = vis_width_gif // 2 * 3             # 2:3 aspect ratio

    video_predictions = predictions['video_predictions']
    video_labels = predictions['video_labels']
    video_ids = predictions['video_ids']

#    max_pred_per_sample = np.max(video_predictions, axis=1)


    ################################################################
    # Print video-based info
    print("Saving video-based visualisations.")

    #get_pred_info(video_predictions[best_idx], video_labels[best_idx], video_ids[best_idx])
    #predicted_scores_for_ground_truth_class, topk_labels, topk_preds = get_pred_info(video_predictions[worst_idx], video_labels[worst_idx], video_ids[worst_idx])
    predicted_scores_for_ground_truth_class, topk_labels, topk_preds = get_pred_info(video_predictions, video_labels, video_ids)

    best_idxs = predicted_scores_for_ground_truth_class.argsort()[-args.num_samples:][::-1]
    worst_idxs = predicted_scores_for_ground_truth_class.argsort()[:args.num_samples]

    if args.ignore_top5:
        # Ignore the samples that are within top5. Good when looking at failures that are not in top5.
        temp = []
        for worst_idx in worst_idxs:
            if video_labels[worst_idx] not in topk_labels[worst_idx]:
                temp.append(worst_idx)
        worst_idxs = np.array(temp)


    output_dir = os.path.join(exp.plots_dir, 'success_failures_epoch_{:04d}'.format(load_epoch))
    output_dir_jpg = os.path.join(output_dir, 'jpg')
    output_dir_gif = os.path.join(output_dir, 'gif')
    output_dir_jpg_best = os.path.join(output_dir_jpg, 'best')
    output_dir_gif_best = os.path.join(output_dir_gif, 'best')
    output_dir_jpg_worst = os.path.join(output_dir_jpg, 'worst')
    output_dir_gif_worst = os.path.join(output_dir_gif, 'worst')
    os.makedirs(output_dir_jpg_best, exist_ok=True)
    os.makedirs(output_dir_gif_best, exist_ok=True)
    os.makedirs(output_dir_jpg_worst, exist_ok=True)
    os.makedirs(output_dir_gif_worst, exist_ok=True)

    if not args.skip_novisual:
        for best_idx, (video_label, video_id, predicted_score_for_ground_truth_class, topk_label, topk_pred) in enumerate(zip(video_labels[best_idxs], video_ids[best_idxs], predicted_scores_for_ground_truth_class[best_idxs], topk_labels[best_idxs], topk_preds[best_idxs])):
            video_based_info = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)

            video_based_info = visualise_pred_info(video_based_info, video_label, video_id, predicted_score_for_ground_truth_class, topk_label, topk_pred, dataset_cfg.class_keys)
            video_based_info_gif = cv2.resize(video_based_info[:,:vis_width//2,:], (vis_width_gif, vis_height_gif), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_dir_jpg_best, 'best_{:02d}-uid_{:05d}-multicrop.jpg'.format(best_idx, video_id)), video_based_info)
            cv2.imwrite(os.path.join(output_dir_gif_best, 'best_{:02d}-uid_{:05d}-multicrop.jpg'.format(best_idx, video_id)), video_based_info_gif)


        for worst_idx, (video_label, video_id, predicted_score_for_ground_truth_class, topk_label, topk_pred) in enumerate(zip(video_labels[worst_idxs], video_ids[worst_idxs], predicted_scores_for_ground_truth_class[worst_idxs], topk_labels[worst_idxs], topk_preds[worst_idxs])):
            video_based_info = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)

            video_based_info = visualise_pred_info(video_based_info, video_label, video_id, predicted_score_for_ground_truth_class, topk_label, topk_pred, dataset_cfg.class_keys)
            video_based_info_gif = cv2.resize(video_based_info[:,:vis_width//2,:], (vis_width_gif, vis_height_gif), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_dir_jpg_worst, 'worst_{:02d}-uid_{:05d}-multicrop.jpg'.format(worst_idx, video_id)), video_based_info)
            cv2.imwrite(os.path.join(output_dir_gif_worst, 'worst_{:02d}-uid_{:05d}-multicrop.jpg'.format(worst_idx, video_id)), video_based_info_gif)
    
    ################################################################
    # evaluate each clip again to get the clip-based predictions
    print("Saving clip-based visualisations.")

    dataloader_type = 'sparse_frames'

    for mode in ['worst']:
    #for mode in ['best', 'worst']:
        print("{:s} predictions..".format(mode))

        #csv_path = os.path.join(output_dir, '{:s}_samples.csv'.format(mode))

        if mode == 'best':
            filter_idxs = best_idxs
        else:
            filter_idxs = worst_idxs

        best_video_ids = video_ids[filter_idxs].tolist()   # for later, to get best video rank from video id.
        #save_dataloader_csv(csv_path, dataset_cfg.frames_dir, video_ids[filter_idxs], video_labels[filter_idxs], dataloader_type=model_cfg.dataloader_type)

        dataset = cfg.get_torch_dataset(split)
        dataset.filter_samples(best_video_ids)


        data_unpack_func = cfg.get_data_unpack_func(split)
        input_reshape_func = cfg.get_input_reshape_func(split)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=4, pin_memory=True, drop_last=False)

        model = cfg.load_model()
        cur_device = torch.cuda.current_device()
        # Transfer the model to the current GPU device
        model = model.to(device=cur_device, non_blocking=True)

        weights_path = exp.get_checkpoint_path(load_epoch)
        #logger.info("Loading weights: " + weights_path) 
        checkpoint = torch.load(weights_path, map_location = "cuda:{}".format(cur_device))
        model.load_state_dict(checkpoint["model_state"])

    #    criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            model.eval()

            for it, data in enumerate(dataloader):
                print("Iteration {:d} / {:d}".format(it+1, len(dataloader)))
                inputs, uids, labels, spatial_temporal_idxs = data_unpack_func(data)
                inputs, uids, labels, curr_batch_size = misc.data_to_gpu(inputs, uids, labels)

                outputs = model(input_reshape_func(inputs))
                softmaxed_outputs = torch.nn.Softmax(dim=1)(outputs).cpu().numpy()

                labels = labels.cpu().numpy()
                uids = uids.cpu().numpy()
                _, topk_labels, topk_preds = get_pred_info(softmaxed_outputs, labels, uids)

                inputs = inputs.cpu().numpy()
                inputs = inputs.transpose((0,2,3,4,1))      # B, F, H, W, C
                inputs = (inputs * dataset.std.tolist() + dataset.mean.tolist()) * 255
                inputs = inputs.astype(np.uint8)
                if not dataset.bgr:
                    # The inputs has to be BGR if it was RGB.
                    # b.c. we're using OpenCV here.
                    inputs = inputs[...,::-1]   # BGR


                for input_vid, softmaxed_output, label, uid, spatial_temporal_idx, topk_label, topk_pred in zip(inputs, softmaxed_outputs, labels, uids, spatial_temporal_idxs, topk_labels, topk_preds):
                    if dataloader_type == 'video_clip':
                        spatial_idx = spatial_temporal_idx % 3
                        temporal_idx = spatial_temporal_idx // 3
                    elif dataloader_type.startswith('sparse'):
                        spatial_idx = spatial_temporal_idx % 10
                        temporal_idx = None
                    best_idx = best_video_ids.index(uid)

                    clip_based_info = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
                    visualise_pred_info(clip_based_info, label, uid, softmaxed_output[label], topk_label, topk_pred, dataset_cfg.class_keys, spatial_idx = spatial_idx, temporal_idx = temporal_idx)

                    info_overlay_rgb = clip_based_info.copy()[...,::-1]         # only texts and no images. Used for gif frames.
                    info_overlay_rgb = cv2.resize(info_overlay_rgb[:,:vis_width//2,:], (vis_width_gif, vis_height_gif), interpolation=cv2.INTER_CUBIC)  # cut the right half out and downsize


                    x_offset=0
                    y_offset=vis_height-args.input_size*vis_frame_y

                    image_grid = video_to_image_grid(input_vid, vis_frame_x)
                    clip_based_info[y_offset:,x_offset:,:] = image_grid

                    #cv2.imshow('a', clip_based_info)
                    #cv2.waitKey(0)

                    if temporal_idx is not None:
                        cv2.imwrite(os.path.join(output_dir_jpg, mode, '{:s}_{:02d}-uid_{:05d}-temporal_{:d}-spatial_{:d}.jpg'.format(mode, best_idx, uid, temporal_idx, spatial_idx)), clip_based_info)
                    else:
                        cv2.imwrite(os.path.join(output_dir_jpg, mode, '{:s}_{:02d}-uid_{:05d}-spatial_{:d}.jpg'.format(mode, best_idx, uid, spatial_idx)), clip_based_info)

                    # GIF
                    x_offset=0
                    y_offset=vis_height_gif-args.input_size*upscale_factor

                    gif_frames = []
                    for frame in input_vid:
                        frame_rgb = frame[...,::-1]
                        upscaled_size = tuple(upscale_factor*x for x in frame_rgb.shape[:2])     # height, width
                        frame_rgb_upscaled = cv2.resize(frame_rgb, upscaled_size[::-1], interpolation=cv2.INTER_CUBIC)  # upscaled_size[::-1] = (width, height)
                        gif_frame = info_overlay_rgb.copy()
                        gif_frame[y_offset:y_offset+upscaled_size[0],x_offset:x_offset+upscaled_size[1],:] = frame_rgb_upscaled
                        gif_frames.append(Image.fromarray(gif_frame))

                    if temporal_idx is not None:
                        save_name = '{:s}_{:02d}-uid_{:05d}-temporal_{:d}-spatial_{:d}.gif'.format(mode, best_idx, uid, temporal_idx, spatial_idx)
                    else:
                        save_name = '{:s}_{:02d}-uid_{:05d}-spatial_{:d}.gif'.format(mode, best_idx, uid, spatial_idx)
                    gif_frames[0].save(os.path.join(output_dir_gif, mode, save_name),
                            save_all=True, append_images=gif_frames[1:], optimize=True, duration=gif_duration, loop=0)


if __name__ == '__main__':
    main()
