#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data

#import slowfast.utils.logging as logging
import logging

from . import decoder as decoder
from . import transform as transform
#from . import utils as utils
from . import video_container as container
#from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


#@DATASET_REGISTRY.register()
class VideoClassificationDataset(torch.utils.data.Dataset):
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
            target_fps = 30000 / 1001,
            train_jitter_min=256, train_jitter_max=320, 
            test_num_ensemble_views=10, test_num_spatial_crops=3,
            crop_size = 224, mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225],
            enable_multi_thread_decode = False,
            decoding_backend = 'pyav',
            path_prefix = "", num_retries=10):
        """
        Construct the video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 video_id_1 label_1
        path_to_video_2 video_id_2 label_2
        ...
        path_to_video_N video_id_N label_N
        ```
        Args:
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported".format(mode)
        self._csv_file = csv_file
        self._path_prefix = path_prefix
        self.mode = mode

        self.train_jitter_min = train_jitter_min
        self.train_jitter_max = train_jitter_max
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.target_fps = target_fps

        self.crop_size = crop_size
        self.enable_multi_thread_decode = enable_multi_thread_decode
        self.decoding_backend = decoding_backend

        self.mean = mean
        self.std = std

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                test_num_ensemble_views * test_num_spatial_crops
            )

        self.test_num_ensemble_views = test_num_ensemble_views
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

        self._path_to_videos = []
        self._video_ids = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(self._csv_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 3
                path, video_id, label = path_label.split()
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self._path_prefix, path)
                    )
                    self._video_ids.append(int(video_id))
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load video loader split {} from {}".format(
            self._split_idx, self._csv_file
        )
        logger.info(
            "Constructing video dataloader (size: {}) from {}".format(
                len(self._path_to_videos), self._csv_file
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
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.train_jitter_min
            max_scale = self.train_jitter_max
            crop_size = self.crop_size
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.test_num_spatial_crops
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.test_num_spatial_crops
            )
            min_scale, max_scale, crop_size = [self.crop_size] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.enable_multi_thread_decode,
                    self.decoding_backend,
                )
            except Exception as e:
                logger.error(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                self.sampling_rate,
                self.num_frames,
                temporal_sample_index,
                self.test_num_ensemble_views,
                video_meta=self._video_meta[index],
                target_fps=self.target_fps,
                backend=self.decoding_backend,
                max_spatial_scale=max_scale,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning('Failed to decode the video index {}'.format(index))
                index = random.randint(0, len(self._path_to_videos) - 1)
                logger.warning('Choosing random video of index {} instead'.format(index))
                continue

            # Perform color normalization.
            frames = frames.float()
            frames = frames / 255.0
            frames = frames - torch.tensor(self.mean)
            frames = frames / torch.tensor(self.std)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
            )

            video_id = self._video_ids[index]
            label = self._labels[index]
            #frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, video_id, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
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
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
