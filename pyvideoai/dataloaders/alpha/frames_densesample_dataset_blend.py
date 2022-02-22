#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import random
import torch
import torch.utils.data

#import slowfast.utils.logging as logging
import logging

from . import transform as transform
from . import utils as utils

logger = logging.getLogger(__name__)


class FramesDensesampleDatasetBlend(torch.utils.data.Dataset):
    """
    Video loader. Construct the video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, csv_file, mode, num_frames, sampling_rate,
            train_jitter_min=256, train_jitter_max=320, test_scale=256,
            test_num_ensemble_views=10, test_num_spatial_crops=3, 
            crop_size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],
            normalise = True,           # divide pixels by 255
            bgr = False,
            path_prefix = "",
            blend_mode = 'all_frames',
            blend_prob = [0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05],   # list of size {sampling_rate}. Index refers to how many frames to blend.
            num_retries = 10):
        """
        Construct the video loader with a given csv file. The format of
        the csv file is:
        ```
        num_classes     # set it to zero for single label. Only needed for multilabel.
        path_to_frames_dir_1/{:05d}.jpg video_id_1 label_1 start_frame_1 end_frame_1
        path_to_frames_dir_2/{:05d}.jpg video_id_2 label_2 start_frame_2 end_frame_2
        ...
        path_to_frames_dir_N/{:05d}.jpg video_id_N label_N start_frame_N end_frame_N
        ```
        Args:
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
        """
        # Only support train, and test mode.
        assert mode in [
            "train",
            "test",
        ], "Split '{}' not supported".format(mode)
        assert blend_mode in ['all_frames'], f"blend_mode {blend_mode} not supported"

        assert len(blend_prob) == sampling_rate, "len(blend_prob) has to be {sampling_rate}."
        if mode == 'test':
            if blend_prob[0] < 1:
                logger.warning(f'Using blend_prob {blend_prob} in test mode. Test is not deterministic.')
        self._csv_file = csv_file
        self._path_prefix = path_prefix
        self._num_retries = num_retries
        self.mode = mode
        self.blend_mode = blend_mode
        self.blend_prob = blend_prob

        self.train_jitter_min = train_jitter_min
        self.train_jitter_max = train_jitter_max
        self.test_scale = test_scale

        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.test_num_emsemble_views = test_num_ensemble_views

        self.crop_size = crop_size

        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

        self.normalise = normalise
        self.bgr = bgr

        # For training, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = test_num_ensemble_views * test_num_spatial_crops

        assert test_num_spatial_crops in [1, 3], "1 for centre, 3 for centre and left,right/top,bottom"
        self.test_num_spatial_crops = test_num_spatial_crops

        logger.info("Constructing video dataset {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):#{{{
        """
        Construct the video loader.
        """
        assert os.path.exists(self._csv_file), "{} not found".format(
            self._csv_file
        )

        self._path_to_frames = []
        self._video_ids = []
        self._labels = []
        self._start_frames = []    # number of sample video frames
        self._end_frames = []    # number of sample video frames
        self._spatial_temporal_idx = []
        with open(self._csv_file, "r") as f:
            self.num_classes = int(f.readline())
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 5
                path, video_id, label, start_frame, end_frame = path_label.split()

                if self.num_classes > 0:
                    label_list = label.split(",")
                    label = np.zeros(self.num_classes, dtype=np.float32)
                    for label_idx in label_list:
                        label[int(label_idx)] = 1.0       # one hot encoding
                else:
                    label = int(label)

                for idx in range(self._num_clips):
                    self._path_to_frames.append(
                        os.path.join(self._path_prefix, path)
                    )
                    self._video_ids.append(int(video_id))
                    self._labels.append(label)
                    self._start_frames.append(int(start_frame))
                    self._end_frames.append(int(end_frame))
                    self._spatial_temporal_idx.append(idx)
        assert (
            len(self._path_to_frames) > 0
        ), "Failed to load video loader split {} from {}".format(
            self._split_idx, self._csv_file
        )
        logger.info(
            "Constructing video dataloader (size: {}) from {}".format(
                len(self._path_to_frames), self._csv_file
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
        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.train_jitter_min
            max_scale = self.train_jitter_max
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.test_num_spatial_crops
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to centre, left,
            # or right if width is larger than height, and middle, top,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.test_num_spatial_crops
            )
            min_scale, max_scale = [self.test_scale] * 2
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )


        num_video_frames = self._end_frames[index] - self._start_frames[index] + 1
        frame_indices = utils.dense_frame_indices(num_video_frames, self.num_frames, self.sampling_rate, clip_idx = temporal_sample_index, num_clips = self.test_num_emsemble_views, tight=False)
        frame_indices = [idx+self._start_frames[index] for idx in frame_indices]     # add offset (frame number start)

        frame_paths = [self._path_to_frames[index].format(frame_idx) for frame_idx in frame_indices]

#        # make sure to close the images to avoid memory leakage.
#        frames = [0] * len(frame_paths)
#        for i, frame_path in enumerate(frame_paths):
#            with Image.open(frame_path) as frame:
#                frames[i] = np.array(frame)
        #frames = [np.array(Image.open(frame_path)) for frame_path in frame_paths]
        frames = utils.retry_load_images(frame_paths, retry=self._num_retries, backend='pytorch', bgr=self.bgr)
        #frames = torch.as_tensor(np.stack(frames))

        if self.blend_mode == 'all_frames':
            num_blend_frames = np.random.choice(self.sampling_rate, p = self.blend_prob)    # 0 means no blend (see 1 frame)
            if num_blend_frames > 0:
                # blend if enough frames, otherwise skip
                if num_video_frames >= self.num_frames * self.sampling_rate:
                    frames = frames/(num_blend_frames + 1)
                    for idx_offset in range(1, num_blend_frames+1):
                        frame_blend_indices = [idx+idx_offset for idx in frame_indices]     # add offset (frame number start)
                        frame_blend_paths = [self._path_to_frames[index].format(frame_idx) for frame_idx in frame_blend_indices]
                        frames_blend = utils.retry_load_images(frame_blend_paths, retry=self._num_retries, backend='pytorch', bgr=self.bgr)

                        frames += frames_blend/(num_blend_frames+1)
                else:
                    logger.warning(f'skipping blending because of lack of frames. Video framelength = {num_video_frames} but require {self.num_frames*self.sampling_rate}')

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.mean, self.std, normalise = self.normalise
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames, scale_factor_width, scale_factor_height, x_offset, y_offset, is_flipped = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=True,    # only applied when "train" mode (spatial_sample_index == -1)
        )


        video_id = self._video_ids[index]
        label = self._labels[index]
        #frames = utils.pack_pathway_output(self.cfg, frames)
        return frames, video_id, label, spatial_sample_index, temporal_sample_index, index, np.array(frame_indices)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_frames)

