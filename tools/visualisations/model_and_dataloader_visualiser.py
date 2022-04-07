#!/usr/bin/env python3
'''
cfg needs dataloader_shape_to_model_input_shape(inputs) function
model_cfg needs model_input_shape_to_NTHWC(inputs) function
model_cfg needs input_std, input_mean, input_bgr, input_normalise
'''
import argparse
import logging
import numpy as np
import os



import torch
from torch.utils.tensorboard import SummaryWriter
from experiment_utils.argparse_utils import add_exp_arguments
from experiment_utils.telegram_post import send_numpy_video_as_gif, send_numpy_photo
from experiment_utils import ExperimentBuilder
import dataset_configs, model_configs, exp_configs
import tqdm

from pyvideoai import config
from pyvideoai.utils import misc

import coloredlogs, verboselogs
logger = verboselogs.VerboseLogger(__name__)

def video_ids_to_dataset_index(video_ids: list, dataset):
    return [index for index, video_id in enumerate(dataset._video_ids) if video_id in video_ids]



def main():
    parser = argparse.ArgumentParser(
        description='Randomly sample some training videos and send to Telegram/TensorBoard as GIF. Run with no GPUs (CUDA_VISIBLE_DEVICES=)')

    add_exp_arguments(parser,
            root_default=config.DEFAULT_EXPERIMENT_ROOT, dataset_default='hmdb', model_default='trn_resnet50', name_default='crop224_8frame_largejit_plateau_5scrop_split1',
            dataset_channel_choices=dataset_configs.available_channels, model_channel_choices=model_configs.available_channels, exp_channel_choices=exp_configs.available_channels)
    parser.add_argument("--seed", type=int, default=12, help="Random seed for np, torch, torch.cuda, DALI. Actual seed will be seed+rank.")
    parser.add_argument("--telegram", action='store_true', help="Send to Telegram instead of TensorBoard (default: TensorBoard)")
    parser.add_argument("-B", "--telegram_bot_idx", type=int, default=0, help="Which Telegram bot to use defined in key.ini?")
    parser.add_argument("-w", "--dataloader_num_workers", type=int, default=4, help="num_workers for PyTorch Dataset loader.")
    parser.add_argument("-s", "--split", type=str, default='train', choices=['train', 'val', 'multicropval'], help="Which split to use")
    parser.add_argument("-b", "--batch_size", type=int, help="How many videos to visualise")
    parser.add_argument("-i", "--video_ids", type=int, nargs='*', help="Video IDs to visualise. If not specified, randomly visualise one iteration from the dataloader.")
    parser.add_argument("--jpgs", action='store_true', help="Send as JPGs instead of GIFs")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    coloredlogs.install(fmt='%(name)s: %(lineno)4d - %(levelname)s - %(message)s', level='INFO')
    logging.getLogger('slowfast.utils.checkpoint').setLevel(logging.WARNING)

    cfg = exp_configs.load_cfg(args.dataset, args.model, args.experiment_name, args.dataset_channel, args.model_channel, args.experiment_channel)
    metrics = cfg.dataset_cfg.task.get_metrics(cfg)
    summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_from_metrics(metrics)
    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, args.subfolder_name, summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes, version = 'new', telegram_key_ini = config.KEY_INI_PATH, telegram_bot_idx = args.telegram_bot_idx)

    exp.make_dirs_for_training()


    try:
        if not args.telegram:
            writer = SummaryWriter(os.path.join(exp.tensorboard_runs_dir, 'train_model_and_data'), comment='train_model_and_data')

            # Network
            # Construct the model
            model = cfg.load_model()
            misc.log_model_info(model)

            #criterion = cfg.dataset_cfg.task.get_criterion(cfg)
            #optimiser = cfg.optimiser(policies)


        # Construct dataloader
        dataset = cfg.get_torch_dataset(args.split)
        data_unpack_func = cfg.get_data_unpack_func(args.split)
        if args.batch_size is None:
            batch_size = cfg.batch_size() if callable(cfg.batch_size) else cfg.batch_size
        else:
            batch_size = args.batch_size

        if args.video_ids:
            subset_indices = video_ids_to_dataset_index(args.video_ids, dataset)
            dataset = torch.utils.data.Subset(dataset, subset_indices)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True if args.split=='train' else False, sampler=None, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=False)
        class_keys = cfg.dataset_cfg.class_keys

        if args.telegram:
            message = 'Video from dataloader: reshaped to fit the model shape, and reshaped back to video'
            logger.info(message)
            exp.tg_send_text_with_expname(message)

        #data = next(iter(dataloader))
        for it, data in enumerate(tqdm.tqdm(dataloader)):
            inputs, uids, labels, _ = data_unpack_func(data)
            inputs = cfg.get_input_reshape_func(args.split)(inputs)

            if not args.telegram and it == 0:
                # Write model graph
                # if inputs is a list(or tuple), add_graph will consider it as args and unpack the list.
                # So we wrap it as a tuple
                writer.add_graph(model, (inputs,))

            inputs = cfg.model_cfg.model_input_shape_to_NTHWC(inputs)
            inputs *= torch.tensor(cfg.model_cfg.input_std)
            inputs += torch.tensor(cfg.model_cfg.input_mean)
            if cfg.model_cfg.input_normalise:
                inputs *= 255
            if cfg.model_cfg.input_bgr:
                inputs = inputs[...,::-1]

            inputs = inputs.to(torch.uint8)

            uids = uids.numpy()
            labels = labels.numpy()

            for idx, (video, uid, label) in enumerate(zip(inputs, uids, labels)):
                # T, H, W, C
                class_key = class_keys[label]
                caption = f"uid: {uid}, label: {label}, class_key: {class_key}, video shape: {video.shape}"
                logger.info(caption)
                if args.jpgs:
                    if args.telegram:
                        for jpg in video:
                            # Telegram
                            send_numpy_photo(exp.tg_token, exp.tg_chat_id, jpg.numpy(), caption=caption)
                            caption=None    # only send caption on the first frame
                    else:
                        # Tensorboard
                        # Add in grid format
                        writer.add_images('dataloader_imgs', video, global_step = idx, dataformats='NHWC')
                        writer.add_text('dataloader_imgs', caption, global_step=idx)

                        # Also, add in series format
                        tag = f'dataloader_imgs-uid{uid}'
                        for frame_idx, jpg in enumerate(video):
                            writer.add_image(tag, jpg, global_step=frame_idx, dataformats='HWC')



                else:
                    if args.telegram:
                        # Telegram
                        send_numpy_video_as_gif(exp.tg_token, exp.tg_chat_id, video.numpy(), caption=caption)
                    else:
                        # Tensorboard
                        # T, H, W, C -> 1, T, C, H, W
                        video_tensorboard = video.unsqueeze(0).permute((0,1,4,2,3))
                        writer.add_video('dataloader_gif', video_tensorboard, global_step=idx)
                        writer.add_text('dataloader_gif', caption, global_step=idx)

                if not args.video_ids:
                    # only one iteration
                    break


        if args.telegram:
            logger.success('Finished. Visualisation sent on Telegram')
        else:
            logger.success('Finished. Visualisation sent on TensorBoard')

    except Exception as e:
        logger.exception("Exception occurred")
        exp.tg_send_text_with_expname('Exception occurred\n\n' + repr(e))



if __name__ == '__main__':
    main()
