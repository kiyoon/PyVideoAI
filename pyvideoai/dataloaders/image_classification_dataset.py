#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import random
import torch
import torch.utils.data

import logging

from . import transform as transform
from . import utils as utils

logger = logging.getLogger(__name__)

def count_class_frequency(csv_file):
    assert os.path.exists(csv_file), "{} not found".format(
        csv_file
    )

    labels = []
    with open(csv_file, "r") as f:
        self.num_classes = int(f.readline())
        for clip_idx, path_label in enumerate(f.read().splitlines()):
            assert len(path_label.split()) == 3
            path, video_id, label = path_label.split()
            label = int(label)
            labels.append(label)

    labels = np.array(labels)
    assert labels.min() == 0, 'class label has to start with 0'

    class_frequency = np.bincount(labels, minlength=labels.max())
    return class_frequency

class ImageClassificationDataset(torch.utils.data.Dataset):
    """
    Image loader. For training, a single clip is
    randomly sampled from every image with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    image with uniform cropping.
    """

    def __init__(self, csv_file, mode,
            train_jitter_min=256, train_jitter_max=320,
            test_scale=256, test_num_spatial_crops=10, 
            crop_size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],
            normalise = True,           # divide pixels by 255
            bgr = False,
            greyscale = False,
            path_prefix = "",
            num_retries = 10):
        """
        Construct the video loader with a given csv file. The format of
        the csv file is:
        ```
        num_classes     # set it to zero for single label. Only needed for multilabel.
        path_to_image_1.jpg image_id_1 label_1
        path_to_image_2.jpg image_id_2 label_2
        ...
        path_to_image_N.jpg image_id_N label_N
        ```
        Args:
            mode (string): Options includes `train`, or `test` mode.
                For the train, the data loader will take data
                from the train set, and sample one clip per image.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per image.
        """
        # Only support train, and test mode.
        assert mode in [
            "train",
            "test",
        ], "Split '{}' not supported".format(mode)
        self._csv_file = csv_file
        self._path_prefix = path_prefix
        self._num_retries = num_retries
        self.mode = mode

        self.train_jitter_min = train_jitter_min
        self.train_jitter_max = train_jitter_max
        self.test_scale = test_scale

        self.crop_size = crop_size

        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

        self.normalise = normalise
        self.bgr = bgr
        self.greyscale = greyscale 

        # For training mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = test_num_spatial_crops

        assert test_num_spatial_crops in [1, 5, 10], "1 for centre, 5 for centre and four corners, 10 for their horizontal flips"
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

        self._path_to_images = []
        self._image_ids = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(self._csv_file, "r") as f:
            self.num_classes = int(f.readline())
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 3
                path, image_id, label = path_label.split()

                if self.num_classes > 0:
                    label_list = label.split(",")
                    label = np.zeros(self.num_classes, dtype=np.float32)
                    for label_idx in label_list:
                        label[int(label_idx)] = 1.0       # one hot encoding
                else:
                    label = int(label)

                for idx in range(self._num_clips):
                    self._path_to_images.append(
                        os.path.join(self._path_prefix, path)
                    )
                    self._image_ids.append(int(image_id))
                    self._labels.append(label)
                    self._spatial_temporal_idx.append(idx)
        assert (
            len(self._path_to_images) > 0
        ), "Failed to load video loader split {} from {}".format(
            self._split_idx, self._csv_file
        )
        logger.info(
            "Constructing video dataloader (size: {}) from {}".format(
                len(self._path_to_images), self._csv_file
            )
        )#}}}

    def __getitem__(self, index):
        crop_size = self.crop_size
        if self.mode in ["train"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.train_jitter_min
            max_scale = self.train_jitter_max
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
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Treat as 1-frame video
        frame_paths = [self._path_to_images[index]]
        frames = utils.retry_load_images(frame_paths, retry=self._num_retries, backend='pytorch', bgr=self.bgr, greyscale=self.greyscale)
        frames = utils.tensor_normalize(
            frames, self.mean, self.std, normalise = self.normalise
        )

        # T, H, W, C -> C, T, H, W
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

        C, T, H, W = frames.shape   # T == 1
        frames = frames.view(C, H, W)

        image_id = self._image_ids[index]
        label = self._labels[index]
        #frames = utils.pack_pathway_output(self.cfg, frames)
        return frames, image_id, label, spatial_sample_index, index

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_images)

