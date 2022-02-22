#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import random
import torch
import torch.utils.data
import re

from . import utils as utils
#import slowfast.utils.logging as logging
import logging

logger = logging.getLogger(__name__)


import json

object_shape_to_idx = {'spl': 0, 'sphere': 1, 'cube': 2, 'cone': 3, 'cylinder': 4}
object_size_to_idx = {'small': 0, 'medium': 1, 'large': 2}
object_colour_to_idx = {'red': 0, 'green': 1, 'blue': 2, 'cyan': 3, 'yellow': 4, 'purple': 5, 'brown': 6, 'gray': 7, 'gold': 8}
object_material_to_idx = {'metal': 0, 'rubber': 1}
NUM_FRAMES = 301

def cater_load_scene(file_path, frame_indices: list=None, noise_others=None):
    with open(file_path, 'r') as f:
        scene = json.load(f)


    num_objects = len(scene['objects'])

    if frame_indices is None:
        # All frames
        frame_indices = range(NUM_FRAMES)

    data = np.ones((len(frame_indices), 10, 7), dtype=np.float32) * (-1)  # num_frames, num_objects, [material, size, shape, color, location_xyz]

    for objidx in range(num_objects):
        if objidx != 0 and noise_others == 'noise':
            for i, frameidx in enumerate(frame_indices):
                data[i, objidx, 0] = np.random.randint(2)
                data[i, objidx, 1] = np.random.randint(3)
                data[i, objidx, 2] = np.random.randint(5)
                data[i, objidx, 3] = np.random.randint(9)
                data[i, objidx, 4] = np.random.rand() * 4 - 2
                data[i, objidx, 5] = np.random.rand() * 4 - 2
                data[i, objidx, 6] = np.random.rand() * 4 - 2
        elif objidx != 0 and noise_others == 'pad':
                data[i, objidx, :] = np.ones(7) * (-1)
        else:
            for i, frameidx in enumerate(frame_indices):
                data[i, objidx, 0] = object_material_to_idx[scene['objects'][objidx]['material']]
                data[i, objidx, 1] = object_size_to_idx[scene['objects'][objidx]['size']]
                data[i, objidx, 2] = object_shape_to_idx[scene['objects'][objidx]['shape']]
                data[i, objidx, 3] = object_colour_to_idx[scene['objects'][objidx]['color']]
                data[i, objidx, 4:] = scene['objects'][objidx]['locations'][str(frameidx)]

    return data



#@DATASET_REGISTRY.register()
class CaterTask3Dataset(torch.utils.data.Dataset):
    """
    Video loader. Construct the video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, csv_file, scenes_dir, mode, num_frames, _noise_others=None, _lastframeonly=False):
        """
        Construct the video loader with a given csv file. The format of
        the csv file is:
        ```
        video_name_1 label_1
        video_name_2 label_2
        ...
        video_name_N label_N
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
        self._scenes_dir = scenes_dir
        self.mode = mode
        self.num_frames = num_frames

        self._noise_others = _noise_others
        self._lastframeonly = _lastframeonly
        if _lastframeonly:
            assert num_frames == 1

        self._video_meta = {}
        if self.mode in ["train"]:
            self._num_clips = 1
        if self.mode in ["val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = 1

        logger.info("Constructing CATER dataset for task 3 {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):#{{{
        """
        Construct the video loader.
        """
        assert os.path.exists(self._csv_file), "{} not found".format(
            self._csv_file
        )

        self._path_to_jsons = []
        self._video_ids = []
        self._labels = []
        self._num_sample_frames = []    # number of sample video frames
        self._spatial_temporal_idx = []
        with open(self._csv_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 2
                path, label = path_label.split()
                for idx in range(self._num_clips):
                    video_id_str = re.search('CATER_new_(\d+).avi', path).group(1)
                    self._path_to_jsons.append(
                        os.path.join(self._scenes_dir, f'CATER_new_{video_id_str}.json')
                    )
                    self._labels.append(int(label))
                    self._video_ids.append(int(video_id_str))
                    self._spatial_temporal_idx.append(idx)
        assert (
            len(self._path_to_jsons) > 0
        ), "Failed to load video loader from {}".format(
            self._csv_file
        )
        logger.info(
            "Constructing video dataloader (size: {}) from {}".format(
                len(self._path_to_jsons), self._csv_file
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
        if self.mode in ["train"]:
            sample_uniform = False
        elif self.mode in ["val"]:
            sample_uniform = True
        elif self.mode in ["test"]:
            sample_uniform = True
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )


#        frame_indices = utils.TRN_sample_indices(self._num_sample_frames[index], self.num_frames, mode = self.mode)
        if self._lastframeonly:
            frame_indices = [300]
        else:
            frame_indices = utils.sparse_frame_indices(NUM_FRAMES, self.num_frames, uniform=sample_uniform)

        data = cater_load_scene(self._path_to_jsons[index], frame_indices, noise_others=self._noise_others)

        video_id = self._video_ids[index]
        label = self._labels[index]
        spatial_temporal_index = self._spatial_temporal_idx[index]
        #frames = utils.pack_pathway_output(self.cfg, frames)
        return data, video_id, label, spatial_temporal_index, index, np.array(frame_indices)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_jsons)

