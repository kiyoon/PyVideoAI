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


#@DATASET_REGISTRY.register()
class FramesSparsesampleDataset(torch.utils.data.Dataset):
    """
    Video loader. Construct the video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, csv_file, mode, num_frames, 
            train_jitter_min=256, train_jitter_max=320, val_scale=256, test_scale=256,
            val_num_spatial_crops=1, test_num_spatial_crops=10, 
            crop_size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],
            normalise = True,           # divide pixels by 255
            bgr = False,
            frame_start_idx = 0,        # count from zero or one?
            num_zero_pad_in_filenames = 5,      # %05d
            file_ext = 'jpg',
            path_prefix = "",
            multilabel_numclasses = 0,         # if >0, multilabel. label comma separated
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
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported".format(mode)
        self._csv_file = csv_file
        self._path_prefix = path_prefix
        self._num_retries = num_retries
        self.mode = mode

        self.train_jitter_min = train_jitter_min
        self.train_jitter_max = train_jitter_max
        self.val_scale = val_scale
        self.test_scale = test_scale

        self.multilabel_numclasses = multilabel_numclasses
        self.num_frames = num_frames

        self.frame_start_idx = frame_start_idx
        self.num_zero_pad_in_filenames = num_zero_pad_in_filenames
        self.file_ext = file_ext

        self.crop_size = crop_size

        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

        self.normalise = normalise
        self.bgr = bgr

        # For training mode, one single clip is sampled from every
        # video. For val and testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train"]:
            self._num_clips = 1
        if self.mode in ["val"]:
            self._num_clips = val_num_spatial_crops
        elif self.mode in ["test"]:
            self._num_clips = test_num_spatial_crops

        assert val_num_spatial_crops in [1, 5, 10], "1 for centre, 5 for centre and four corners, 10 for their horizontal flips"
        assert test_num_spatial_crops in [1, 5, 10], "1 for centre, 5 for centre and four corners, 10 for their horizontal flips"
        self.val_num_spatial_crops = val_num_spatial_crops
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

        self._path_to_dirs = []
        self._video_ids = []
        self._labels = []
        self._num_sample_frames = []    # number of sample video frames
        self._spatial_temporal_idx = []
        with open(self._csv_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 4
                path, video_id, label, num_frames = path_label.split()

                # multilabel?
                if self.multilabel_numclasses > 0:
                    label_list = label.split(",")
                    label = np.zeros(self.multilabel_numclasses, dtype=np.float32)
                    for label_idx in label_list:
                        label[int(label_idx)] = 1.0       # one hot encoding
                else:
                    label = int(label)


                for idx in range(self._num_clips):
                    self._path_to_dirs.append(
                        os.path.join(self._path_prefix, path)
                    )
                    self._video_ids.append(int(video_id))
                    self._labels.append(label)
                    self._num_sample_frames.append(int(num_frames))
                    self._spatial_temporal_idx.append(idx)
        assert (
            len(self._path_to_dirs) > 0
        ), "Failed to load video loader split {} from {}".format(
            self._split_idx, self._csv_file
        )
        logger.info(
            "Constructing video dataloader (size: {}) from {}".format(
                len(self._path_to_dirs), self._csv_file
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
            spatial_sample_index = -1
            min_scale = self.train_jitter_min
            max_scale = self.train_jitter_max
            sample_uniform = False
        elif self.mode in ["val"]:
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.val_num_spatial_crops
            )
            min_scale, max_scale = [self.val_scale] * 2
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale are expect to be the same.
            assert len({min_scale, max_scale}) == 1
            sample_uniform = True
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.test_num_spatial_crops
            )
            min_scale, max_scale = [self.test_scale] * 2
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale are expect to be the same.
            assert len({min_scale, max_scale}) == 1
            sample_uniform = True
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )


        # Decode video. Meta info is used to perform selective decoding.
#        frame_indices = utils.TRN_sample_indices(self._num_sample_frames[index], self.num_frames, mode = self.mode)
        frame_indices = utils.sparse_frame_indices(self._num_sample_frames[index], self.num_frames, uniform=sample_uniform)
        frame_indices = [idx+self.frame_start_idx for idx in frame_indices]     # add offset (frame number start from zero or one)

        frame_paths = [os.path.join(self._path_to_dirs[index], str(frame_idx).zfill(self.num_zero_pad_in_filenames) + "." + self.file_ext) for frame_idx in frame_indices]

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


        video_id = self._video_ids[index]
        label = self._labels[index]
        #frames = utils.pack_pathway_output(self.cfg, frames)
        return frames, video_id, label, spatial_sample_index, index, np.array(frame_indices), {}

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_dirs)

