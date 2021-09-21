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

import pickle

logger = logging.getLogger(__name__)

from .frames_sparsesample_dataset import count_class_frequency

class FramesSparsesampleDatasetDynamicLabel(torch.utils.data.Dataset):
    """
    Video loader. Construct the video loader, then sample
    clips from the videos. For training, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the center
    and four corners.
    """

    def __init__(self, csv_file, mode, num_frames, 
            num_classes, student_predictions_pkl, student_weight = 0.5,  # here, student means the model, teacher means the label.
            train_jitter_min=256, train_jitter_max=320,
            train_horizontal_flip = True,
            test_scale=256, test_num_spatial_crops=10, 
            crop_size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],
            normalise = True,           # divide pixels by 255
            bgr = False,
            greyscale = False,
            path_prefix = "",
            num_retries = 10,
            sample_index_code = 'pyvideoai',
            ):
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
            mode (str): Options includes `train`, or `test` mode.
                For the train, the data loader will take data
                from the train set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            sample_index_code (str): Options include `pyvideoai`, `TSN` and `TDN`.
                Slightly different implementation of how video is sampled (pyvideoai and TSN),
                and for the TDN, it is completely different as it samples num_frames*5 frames.
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
        self.sample_index_code = sample_index_code.lower()

        self.train_jitter_min = train_jitter_min
        self.train_jitter_max = train_jitter_max
        self.test_scale = test_scale

        self.train_horizontal_flip = train_horizontal_flip

        self.num_frames = num_frames

        self.crop_size = crop_size

        if greyscale:
            assert len(mean) == 1
            assert len(std) == 1
            assert not bgr, "Greyscale and BGR can't be set at the same time."
        else:
            assert len(mean) in [1, 3]
            assert len(std) in [1, 3]
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


        # Dynamic label smoothing: listen to student (model)
        self.student_weight = student_weight
        self.student_predictions_pkl = student_predictions_pkl    # when set to None, skip label smoothing and use one-hot instead
        if not (student_weight == 0.0 or student_predictions_pkl is None):
            with open(student_predictions_pkl, 'rb') as f:
                predictions = pickle.load(f)
            self.student_predictions = {}   # format: key=video_id, value=prediction
            for video_id, pred in zip(predictions['video_ids'], predictions['video_predictions']):
                assert video_id not in self.student_predictions.keys(), f'Already added {video_id = }'
                self.student_predictions[video_id] = pred

        #self.num_classes = predictions['video_predictions'].shape[1]
        self.num_classes = num_classes

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
            header_num_classes = int(f.readline())
            assert header_num_classes == 0, 'Cannot use label smoothing with multilabel datasets.'

            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 5
                path, video_id, label, start_frame, end_frame = path_label.split()
                video_id = int(video_id)
                label = int(label)

                label_smooth = np.zeros(self.num_classes, dtype=np.float32)
                label_smooth[label] = 1.0

                if self.student_predictions_pkl is not None:
                    label_smooth = (1-self.student_weight)*label_smooth + self.student_weight*self.student_predictions[video_id]

                for idx in range(self._num_clips):
                    self._path_to_frames.append(
                        os.path.join(self._path_prefix, path)
                    )
                    self._video_ids.append(int(video_id))
                    self._labels.append(label_smooth)
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

    def filter_samples(self, video_ids: list):
        """Given a video_ids list, filter the samples.
        """
        indices_of_video_ids = [x for x, v in enumerate(self._video_ids) if v in video_ids]

        self._path_to_frames = [self._path_to_frames[x] for x in indices_of_video_ids]
        self._video_ids = [self._video_ids[x] for x in indices_of_video_ids]
        self._labels = [self._labels[x] for x in indices_of_video_ids]
        self._start_frames = [self._start_frames[x] for x in indices_of_video_ids]
        self._end_frames = [self._end_frames[x] for x in indices_of_video_ids]
        self._spatial_temporal_idx = [self._spatial_temporal_idx[x] for x in indices_of_video_ids]


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
        num_video_frames = self._end_frames[index] - self._start_frames[index] + 1
        if self.sample_index_code == 'pyvideoai':
            frame_indices = utils.sparse_frame_indices(num_video_frames, self.num_frames, uniform=sample_uniform)
        elif self.sample_index_code == 'tsn':
            frame_indices = utils.TSN_sample_indices(num_video_frames, self.num_frames, mode = self.mode)
        elif self.sample_index_code == 'tdn':
            frame_indices = utils.TDN_sample_indices(num_video_frames, self.num_frames, mode = self.mode)
        elif self.sample_index_code == 'tdn_greyst':
            frame_indices = utils.TDN_sample_indices(num_video_frames, self.num_frames, mode = self.mode, new_length=15)
        else:
            raise ValueError(f'Wrong self.sample_index_code: {self.sample_index_code}. Should be pyvideoai, TSN, TDN')
        frame_indices = [idx+self._start_frames[index] for idx in frame_indices]     # add offset (frame number start)

        frame_paths = [self._path_to_frames[index].format(frame_idx) for frame_idx in frame_indices]

#        # make sure to close the images to avoid memory leakage.
#        frames = [0] * len(frame_paths)
#        for i, frame_path in enumerate(frame_paths):
#            with Image.open(frame_path) as frame:
#                frames[i] = np.array(frame)
        #frames = [np.array(Image.open(frame_path)) for frame_path in frame_paths]
        frames = utils.retry_load_images(frame_paths, retry=self._num_retries, backend='pytorch', bgr=self.bgr, greyscale=self.greyscale)
        #frames = torch.as_tensor(np.stack(frames))

        # Perform color normalization.
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
            random_horizontal_flip=self.train_horizontal_flip,
        )


        video_id = self._video_ids[index]
        label = self._labels[index]
        return frames, video_id, label, spatial_sample_index, index, np.array(frame_indices)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_frames)

