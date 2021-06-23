import sys
import argparse
from pyvideoai import config
import dataset_configs
from pyvideoai.dataloaders.utils import retry_load_images
import os

from PIL import Image

import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="Visualise success and failures. For now, only support EPIC",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Evaluate using 1 clip or 30 clips.")
    return parser

def rgb_to_greyscale_image(imgs):
    #T, H, W, C = imgs.shape
    rgb_weights = [0.2989, 0.5870, 0.1140]
    return np.dot(imgs[...,:3], rgb_weights)

def video_TC_channel_mix(imgs):
    T, H, W, C = imgs.shape
    imgs = imgs.transpose((0,3,1,2)).reshape(-1, H, W)    # TC, H, W
    imgs = imgs.reshape(C, T, H, W).transpose((1,2,3,0))    # TC, H, W
    return imgs

def video_frame_blend(imgs, weighted=False):
    if weighted:
        weights = np.arange(len(imgs)) + 1
    else:
        weights = None
    return np.average(imgs, axis=0, weights=weights).astype(np.uint8)

def rgb_save_jpg(img, save_path):
    Image.fromarray(img).save(save_path)

def rgb_save_jpgs(imgs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(imgs):
        path = os.path.join(save_dir, f'{i:05d}.jpg')
        rgb_save_jpg(frame, path)

def rgb_save_gif(imgs, save_path, optimize=True, duration=100, loop=0 ):
    #T, H, W, C = imgs.shape
    gif_frames = []
    for gif_frame in imgs:
        gif_frames.append(Image.fromarray(gif_frame))

    gif_frames[0].save(save_path, save_all=True, append_images=gif_frames[1:], optimize=optimize, duration=duration, loop=loop)

def alter_img_and_save(imgs, output_dir):
    greyscale_imgs = rgb_to_greyscale_image(imgs)
    TCmixed_imgs = video_TC_channel_mix(imgs)
    blended_img = video_frame_blend(imgs[10:15])
    weighted_blended_img = video_frame_blend(imgs[10:15], weighted=True)

    rgb_save_gif(imgs, os.path.join(output_dir, 'orig.gif'))
    rgb_save_gif(greyscale_imgs, os.path.join(output_dir, 'grey.gif'))
    rgb_save_gif(TCmixed_imgs, os.path.join(output_dir, 'TCmixed.gif'))
    rgb_save_jpgs(TCmixed_imgs, os.path.join(output_dir, 'TCmixed'))
    rgb_save_jpg(imgs[15], os.path.join(output_dir, 'frame.jpg'))
    rgb_save_jpg(blended_img, os.path.join(output_dir, 'blended.jpg'))
    rgb_save_jpg(weighted_blended_img, os.path.join(output_dir, 'weighted_blended.jpg'))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    base_output_dir = os.path.join(config.DATA_DIR, 'visualisations', 'frame_blend_grey_scale')

    # CATER
    output_dir = os.path.join(base_output_dir, 'CATER_new_000014')
    os.makedirs(output_dir, exist_ok=True)

    cater_cfg = dataset_configs.load_cfg('cater_task2')
    cater_frames_dir = os.path.join(cater_cfg.dataset_root, 'frames', 'CATER_new_000014.avi')
    frames_indices = list(range(35))
    frames_paths = [os.path.join(cater_frames_dir, f'{idx:05d}.jpg') for idx in frames_indices]
    imgs= retry_load_images(frames_paths, backend='cv2', bgr=False)

    alter_img_and_save(imgs, output_dir)


    # diving48
    output_dir = os.path.join(base_output_dir, 'diving48-iv0Gu1VXAgc_00225')
    os.makedirs(output_dir, exist_ok=True)

    cfg = dataset_configs.load_cfg('diving48')
    frames_dir = os.path.join(cfg.image_frames_dir, 'iv0Gu1VXAgc_00225.mp4')
    frames_indices = list(range(35))
    frames_paths = [os.path.join(frames_dir, f'{idx:05d}.jpg') for idx in frames_indices]
    imgs= retry_load_images(frames_paths, backend='cv2', bgr=False)
    alter_img_and_save(imgs, output_dir)
    # diving48 full
    output_dir = os.path.join(base_output_dir, 'diving48-iv0Gu1VXAgc_00225-full')
    os.makedirs(output_dir, exist_ok=True)

    cfg = dataset_configs.load_cfg('diving48')
    frames_dir = os.path.join(cfg.image_frames_dir, 'iv0Gu1VXAgc_00225.mp4')
    frames_indices = list(range(80))
    frames_paths = [os.path.join(frames_dir, f'{idx:05d}.jpg') for idx in frames_indices]
    imgs= retry_load_images(frames_paths, backend='cv2', bgr=False)
    alter_img_and_save(imgs, output_dir)

    # epic
    output_dir = os.path.join(base_output_dir, 'epic-13612')
    os.makedirs(output_dir, exist_ok=True)

    cfg = dataset_configs.load_cfg('epic_verb')
    frames_dir = os.path.join(cfg.dataset_root, 'segments324_15fps_frames', '13612')
    frames_indices = list(range(35))
    frames_paths = [os.path.join(frames_dir, f'{idx:05d}.jpg') for idx in frames_indices]
    imgs= retry_load_images(frames_paths, backend='cv2', bgr=False)
    alter_img_and_save(imgs, output_dir)
