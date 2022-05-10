#!/usr/bin/env python3
from __future__ import annotations


import logging
import numpy as np
import time
import torch
import random
import itertools

import cv2

from . import transform

logger = logging.getLogger(__name__)


def retry_load_images(image_paths, retry=10, backend="pytorch", bgr=False, greyscale=False):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`. For pytorch backend, it returns RGB torch tensor. For cv2, it returns BGR list of numpy images.

    Returns:
        imgs (list): list of loaded images.
    """
    assert not (bgr == greyscale == True), "Cannot be BGR and Greyscale at the same time"
    for i in range(retry):
        if greyscale:
            imgs = [np.expand_dims(cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE), axis=-1) for image_path in image_paths]
        else:
            imgs = [cv2.imread(image_path) for image_path in image_paths]

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                if bgr or greyscale:
                    imgs = torch.as_tensor(np.stack(imgs))
                else:
                    imgs = torch.as_tensor(np.stack(imgs)[...,[2,1,0]])
            else:
                # backend == 'cv2'
                imgs = np.stack(imgs)
                if not bgr and not greyscale:
                    imgs = imgs[...,[2,1,0]]
            return imgs
        else:
            logger.warn("Reading failed. Will retry. %s", str(image_paths))
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.

    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames

    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames, _, scale_factor_width, scale_factor_height = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        frames, _, x_offset, y_offset = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _, is_flipped = transform.random_horizontal_flip(0.5, frames)
        else:
            is_flipped = False
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _, scale_factor_width, scale_factor_height = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _, x_offset, y_offset = transform.uniform_crop(frames, crop_size, spatial_idx)
        is_flipped = False
    return frames, scale_factor_width, scale_factor_height, x_offset, y_offset, is_flipped


def spatial_sampling_5(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
    5 different crop locations (0: centre, 1: top left, 2: top right, 3: bottom left, 4: bottom right)
    Plus, horizontal flip indices (5: centre and horizontal flip, ...)

    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is in [0, 10), perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if spatial_idx == -1:
        if min_scale is not None and max_scale is not None:
            frames, _, scale_factor_width, scale_factor_height = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
        else:
            scale_factor_width = scale_factor_height = None
        frames, _, x_offset, y_offset = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _, is_flipped = transform.random_horizontal_flip(0.5, frames)
        else:
            is_flipped = False
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale and max_scale are expect to be the same.
        if min_scale is not None and max_scale is not None:
            assert len({min_scale, max_scale}) == 1
            frames, _, scale_factor_width, scale_factor_height = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
        else:
            scale_factor_width = scale_factor_height = None

        frames, _, x_offset, y_offset = transform.uniform_crop_5(frames, crop_size, spatial_idx % 5)
        if spatial_idx >= 5:
            frames, _ = transform.horizontal_flip(frames)
            is_flipped = True
        else:
            is_flipped = False


    return frames, scale_factor_width, scale_factor_height, x_offset, y_offset, is_flipped


def tensor_normalize(tensor, mean, std, normalise=True):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor): mean value to subtract.
        std (tensor): std to divide.
    """
    if normalise:
        tensor = tensor / 255.0
    elif tensor.dtype == torch.uint8:
        tensor = tensor.float()
    tensor = tensor - mean
    tensor = tensor / std
    return tensor



def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


def _add_neighbour_frames(frame_indices: list[int], num_neighbours: int) -> list[int]:
    return [idx + n for idx in frame_indices for n in range(num_neighbours)]


def strided_frame_indices(num_video_frames, num_sample_frames, sampling_rate, clip_idx, stride, tight=True, num_neighbours=1):
    """
    params:
        tight (bool): if True, even for sampling_rate > 1, the last frame can be sampled.
                        if False, last {sampling_rate-1} frames are not sampled.
    """
    if tight:
        sample_span = (num_sample_frames-1) * sampling_rate + num_neighbours          # e.g. 8x4 -> 29 frames span
    else:
        assert num_neighbours == 1, 'Not implemented'
        sample_span = num_sample_frames * sampling_rate                  # e.g. 8x4 -> 32 frames span
    video_frame_indices = list(range(num_video_frames))
    if num_video_frames < sample_span:
        num_repeats = sample_span // num_video_frames if sample_span % num_video_frames == 0 else sample_span // num_video_frames + 1
        logger.debug(f"Cannot sample {sample_span:d} frames from {num_video_frames:d}. Duplicating the frames {num_repeats} times.")
        video_frame_indices = list(itertools.chain.from_iterable(itertools.repeat(x, num_repeats) for x in video_frame_indices))
        num_video_frames *= num_repeats     # equal to: len(video_frame_indices)

    # from now on, num_video_frames >= sample_span
    start_idx = stride * clip_idx
    sampled_frame_indices = [video_frame_indices[idx] for idx in range(start_idx, start_idx+sampling_rate*num_sample_frames, sampling_rate)]

    return _add_neighbour_frames(sampled_frame_indices, num_neighbours)


def dense_frame_indices(num_video_frames, num_sample_frames, sampling_rate, clip_idx, num_clips, tight=True, num_neighbours=1):
    """
    params:
        tight (bool): if True, even for sampling_rate > 1, the last frame can be sampled.
                        if False, last {sampling_rate-1} frames are not sampled.
    """
    assert -1 <= clip_idx < num_clips, f"Wrong clip_idx given: {clip_idx}"

    if tight:
        sample_span = (num_sample_frames-1) * sampling_rate + num_neighbours          # e.g. 8x4 -> 29 frames span  (with num_neighbours=1)
    else:
        assert num_neighbours == 1, 'Not implemented'
        sample_span = num_sample_frames * sampling_rate                  # e.g. 8x4 -> 32 frames span
    video_frame_indices = list(range(num_video_frames))
    if num_video_frames < sample_span:
        num_repeats = sample_span // num_video_frames if sample_span % num_video_frames == 0 else sample_span // num_video_frames + 1
        logger.debug(f"Cannot sample {sample_span:d} frames from {num_video_frames:d}. Duplicating the frames {num_repeats} times.")
        video_frame_indices = list(itertools.chain.from_iterable(itertools.repeat(x, num_repeats) for x in video_frame_indices))
        num_video_frames *= num_repeats     # equal to: len(video_frame_indices)

    # from now on, num_video_frames >= sample_span

    start_idx_choices = list(range(num_video_frames - sample_span + 1))
    if clip_idx == -1:
        # random temporal sampling
        start_idx = random.choice(start_idx_choices)
    else:
        if num_clips == 1:
            # sample the middle clip for 1 clip sampling
            start_idx = len(start_idx_choices) // 2
        else:
            # sample the whole temporal coverage (include clip starting from frame 0 and the last frame possible)
            start_idx = round((len(start_idx_choices)-1) / (num_clips-1) * clip_idx)

    sampled_frame_indices = [video_frame_indices[idx] for idx in range(start_idx, start_idx+sampling_rate*num_sample_frames, sampling_rate)]

    #assert len(sampled_frame_indices) == num_sample_frames

    return _add_neighbour_frames(sampled_frame_indices, num_neighbours)



def sparse_frame_indices(num_input_frames, num_output_frames, uniform=True, num_neighbours=1):
    """
    Actual number of output frames would be num_output_frames * num_neighbours.
    num_neighbours samples neighbouring frames (used for optical flow)

    Difference between TSN code: for TSN, when input video is short it just returns zeros. This code efficiently selects more meaningful frames
    """

    # Assume that input is shorter. At the end, we just add neighbours per each frame index.
    num_input_frames_wo_neighbour_length = num_input_frames - num_neighbours + 1

    video_frame_indices = list(range(num_input_frames_wo_neighbour_length))
    if num_input_frames_wo_neighbour_length < num_output_frames:
        num_repeats = num_output_frames // num_input_frames_wo_neighbour_length if num_output_frames % num_input_frames_wo_neighbour_length == 0 else num_output_frames // num_input_frames_wo_neighbour_length + 1
        logger.debug(f"Cannot sample {num_output_frames:d} frames with {num_neighbours} neighbouring frames from {num_input_frames:d}. Duplicating frames {num_repeats} times.")
        video_frame_indices = list(itertools.chain.from_iterable(itertools.repeat(x, num_repeats) for x in video_frame_indices))
        num_input_frames_wo_neighbour_length *= num_repeats     # equal to: len(video_frame_indices)

    segment_start_indices = []      # [segment_start_indices[i], segment_start_indices[i+1]) is the range to sample 1 frame (snippet) from the i-th segment.
    for output_frame_id in range(num_output_frames):
        segment_start_indices.append(int(num_input_frames_wo_neighbour_length * output_frame_id / num_output_frames))

    if uniform:
        sampled_frame_indices = [video_frame_indices[idx] for idx in segment_start_indices]
    else:
        # Random sampling from each segment
        segment_start_indices.append(num_input_frames_wo_neighbour_length)      # since last frame is num_input_frames_wo_neighbour_length - 1
        frame_indices = []
        for output_frame_id in range(num_output_frames):
            frame_indices.append(random.randrange(*segment_start_indices[output_frame_id:output_frame_id+2]))

        sampled_frame_indices = [video_frame_indices[idx] for idx in frame_indices]

    # Add neighbours
    return _add_neighbour_frames(sampled_frame_indices, num_neighbours)


# num_input_frames = num_frames
# num_output_frames = num_segments
def TSN_sample_indices(num_input_frames, num_output_frames, mode="train", new_length = 1):
    """
    :return: list (count from zero)
    """

    if mode == "train":
        average_duration = (num_input_frames - new_length + 1) // num_output_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(num_output_frames)), average_duration) + np.random.randint(average_duration, size=num_output_frames)
        elif num_input_frames > num_output_frames:
            offsets = np.sort(np.random.randint(num_input_frames - new_length + 1, size=num_output_frames))
        else:
            logger.warning("Cannot sample %d frames from %d. Returning all first frames.", num_output_frames, num_input_frames)
            offsets = np.zeros((num_output_frames,), dtype=np.int32)

    else:
        if num_input_frames > num_output_frames + new_length - 1:
            tick = (num_input_frames - new_length + 1) / float(num_output_frames)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_output_frames)])
        else:
            offsets = np.zeros((num_output_frames,), dtype=np.int32)

    return offsets


# num_input_frames = num_frames
# num_output_frames = num_segments
# modified by Kiyoon: indexing from zero (not one)
from numpy.random import randint
def TDN_sample_indices(num_input_frames, num_output_frames, mode='train', new_length=5):
    if mode == 'train' : # TSN uniformly sampling for TDN
        if((num_input_frames - new_length + 1) < num_output_frames):
            average_duration = (num_input_frames - 5 + 1) // (num_output_frames)
        else:
            average_duration = (num_input_frames - new_length + 1) // (num_output_frames)
        offsets = []
        if average_duration > 0:
            offsets += list(np.multiply(list(range(num_output_frames)), average_duration) + randint(average_duration,size=num_output_frames))
        elif num_input_frames > num_output_frames:
            if((num_input_frames - new_length + 1) >= num_output_frames):
                offsets += list(np.sort(randint(num_input_frames - new_length + 1, size=num_output_frames)))
            else:
                offsets += list(np.sort(randint(num_input_frames - 5 + 1, size=num_output_frames)))
        else:
            offsets += list(np.zeros((num_output_frames,)))
    elif mode == 'test':
        if num_input_frames > num_output_frames + new_length - 1:
            tick = (num_input_frames - new_length + 1) / float(num_output_frames)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_output_frames)])
        else:
            offsets = np.zeros((num_output_frames,))
    elif mode == 'dense_train': # i3d type sampling for training
        sample_pos = max(1, 1 + num_input_frames - new_length - 64)
        t_stride = 64 // num_output_frames
        start_idx1 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx1) % num_input_frames for idx in range(num_output_frames)]
    elif mode == 'dense_test':  # i3d dense sample for test
        sample_pos = max(1, 1 + num_input_frames - new_length - 64)
        t_stride = 64 // num_output_frames
        start_idx1 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        start_idx2 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        start_idx3 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        start_idx4 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        start_idx5 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        start_idx6 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        start_idx7 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        start_idx8 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        start_idx9 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        start_idx10 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx1) % num_input_frames for idx in range(num_output_frames)]+[(idx * t_stride + start_idx2) % num_input_frames for idx in range(num_output_frames)]+[(idx * t_stride + start_idx3) % num_input_frames for idx in range(num_output_frames)]+[(idx * t_stride + start_idx4) % num_input_frames for idx in range(num_output_frames)]+[(idx * t_stride + start_idx5) % num_input_frames for idx in range(num_output_frames)]+[(idx * t_stride + start_idx6) % num_input_frames for idx in range(num_output_frames)]+[(idx * t_stride + start_idx7) % num_input_frames for idx in range(num_output_frames)]+[(idx * t_stride + start_idx8) % num_input_frames for idx in range(num_output_frames)]+[(idx * t_stride + start_idx9) % num_input_frames for idx in range(num_output_frames)]+[(idx * t_stride + start_idx10) % num_input_frames for idx in range(num_output_frames)]
    else:
        raise ValueError(f'Not supported mode {mode}. Use train, test, dense_train, dense_test.')

    '''from function TDN.ops.dataset.get
    For each segment starting indices, sample 5 more frames, but duplicate last frame if no more can be found.
    '''
    frames_idx = []
    for seg_ind in offsets:
        p = int(seg_ind)
        for i in range(0,new_length,1):
            frames_idx.append(p)
            if p < num_input_frames-1:  # if p is the last frame already, do not increase
                p += 1

    '''Below seems unnecessary.. Removed the if statement'''
#            if((len(video_list)-self.new_length*1+1)>=8):
#                if p < (len(video_list)):
#                    p += 1
#            else:
#                if p < (len(video_list)):
#                    p += 1

    return frames_idx
