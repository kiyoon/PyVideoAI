#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import random
import torch
import torch.utils.data

#import slowfast.utils.logging as logging
import logging

from . import decoder as decoder
from . import transform as transform
from . import utils as utils


logger = logging.getLogger(__name__)


class FramesSparsesampleDatasetBalanced(torch.utils.data.Dataset):
    """
    Video loader. Construct the video loader, then sample
    clips from the videos.
    """

    def __init__(self, csv_file, num_frames, num_classes, num_samples_per_class=100,
            train_jitter_min=256, train_jitter_max=320,
            crop_size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],
            normalise = True,           # divide pixels by 255
            bgr = False,
            frame_start_idx = 0,        # count from zero or one?
            num_zero_pad_in_filenames = 5,      # %05d
            file_ext = 'jpg',
            path_prefix = "",
            num_retries = 10):
        """
        Construct the video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_frames_dir_1 video_id_1 label_1 num_sample_frames_1
        path_to_frames_dir_2 video_id_2 label_2 num_sample_frames_2
        ...
        path_to_frames_dir_N video_id_N label_N num_sample_frames_N
        ```
        Args:
        """
        self._csv_file = csv_file
        self._path_prefix = path_prefix
        self._num_retries = num_retries

        self.train_jitter_min = train_jitter_min
        self.train_jitter_max = train_jitter_max

        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.num_frames = num_frames

        self.frame_start_idx = frame_start_idx
        self.num_zero_pad_in_filenames = num_zero_pad_in_filenames
        self.file_ext = file_ext

        self.crop_size = crop_size

        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

        self.normalise = normalise
        self.bgr = bgr

        logger.info("Constructing balanced video dataset...")
        self._construct_loader()

    def _construct_loader(self):#{{{
        """
        Construct the video loader.
        """
        assert os.path.exists(self._csv_file), "{} not found".format(
            self._csv_file
        )

        self._video_info = [[]] * self.num_classes
        num_samples_total = 0
        with open(self._csv_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 4
                path, video_id, label, num_frames = path_label.split()
                video_id, label, num_frames = int(video_id), int(label), int(num_frames)
                self._video_info[label].append((os.path.join(self._path_prefix, path),
                    video_id,
                    num_frames))
                num_samples_total += 1
        assert (
            num_samples_total > 0
        ), "Failed to load video loader from {}".format(
            self._csv_file
        )
        logger.info(
            "Constructing video dataloader (size: {}) from {}".format(
                num_samples_total, self._csv_file
            )
        )#}}}

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            video_id (int): the ID of the current video.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        crop_size = self.crop_size

        # -1 indicates random sampling.
        spatial_sample_index = -1
        min_scale = self.train_jitter_min
        max_scale = self.train_jitter_max
        sample_uniform = False

        label = index % self.num_classes
        video_info = random.choice(self._video_info[label])
        path, video_id, num_sample_frames = video_info

        # Decode video. Meta info is used to perform selective decoding.
#        frame_indices = utils.TRN_sample_indices(self._num_sample_frames[index], self.num_frames, mode = self.mode)
        frame_indices = utils.sparse_frame_indices(num_sample_frames, self.num_frames, uniform=sample_uniform)
        frame_indices = [idx+self.frame_start_idx for idx in frame_indices]     # add offset (frame number start from zero or one)

        frame_paths = [os.path.join(path, str(frame_idx).zfill(self.num_zero_pad_in_filenames) + "." + self.file_ext) for frame_idx in frame_indices]

#        # make sure to close the images to avoid memory leakage.
#        frames = [0] * len(frame_paths)
#        for i, frame_path in enumerate(frame_paths):
#            with Image.open(frame_path) as frame:
#                frames[i] = np.array(frame)
        #frames = [np.array(Image.open(frame_path)) for frame_path in frame_paths]
        frames = utils.retry_load_images(frame_paths, retry=self._num_retries, backend='pytorch', bgr=self.bgr)
        #frames = torch.as_tensor(np.stack(frames))

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.mean, self.std, normalise = self.normalise
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames, scale_factor_width, scale_factor_height, x_offset, y_offset, is_flipped = utils.spatial_sampling_5(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=True,
        )


        return frames, video_id, label, spatial_sample_index, index, np.array(frame_indices), {}

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_classes * self.num_samples_per_class

