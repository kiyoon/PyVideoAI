#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import torch
import torch.utils.data
from gulpio2 import GulpDirectory

#import slowfast.utils.logging as logging
import logging

from . import utils as utils

from .transforms import GroupScale, GroupNDarrayToPILImage, GroupRandomCrop, GroupRandomHorizontalFlip, GroupPILImageToNDarray
from .transforms import GroupOneOfFiveCrops
from .transforms import GroupGrayscale, IdentityTransform
from torchvision.transforms import Compose

import random

logger = logging.getLogger(__name__)

def count_class_frequency(csv_file):
    assert os.path.exists(csv_file), "{} not found".format(
        csv_file
    )

    labels = []
    with open(csv_file, "r") as f:
        num_classes = int(f.readline())
        for clip_idx, path_label in enumerate(f.read().splitlines()):
            assert len(path_label.split()) == 5
            path, video_id, label, start_frame, end_frame = path_label.split()
            label = int(label)
            labels.append(label)

    labels = np.array(labels)
    assert labels.min() == 0, 'class label has to start with 0'

    class_frequency = np.bincount(labels, minlength=labels.max())
    return class_frequency

class GulpSparsesampleDataset(torch.utils.data.Dataset):
    """
    It uses GulpIO2 instead of reading directly from jpg frames to speed up the IO!
    It will ignore the gulp meta data, and read meta from the CSV instead.

    Video loader. Construct the video loader, then sample
    clips from the videos. For training, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the center
    and four corners.
    """

    def __init__(self, csv_file, mode, num_frames, gulp_dir_path: str,
            train_jitter_min=256, train_jitter_max=320,
            train_horizontal_flip = True,
            train_class_balanced_sampling = False,      # index of the __getitem__ will be completely ignored, meaning any sampler like
                                                        # DistributedSampler, RandomSampler etc. will have no effect.
            test_scale=256, test_num_spatial_crops=10, 
            crop_size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],
            normalise = True,           # divide pixels by 255
            bgr = False,
            greyscale = False,
            sample_index_code = 'pyvideoai',
            processing_backend = 'pil',         # torch, pil
                                                # Note that they will produce different result as pillow performs LPF before downsampling.
                                                # https://stackoverflow.com/questions/60949936/why-bilinear-scaling-of-images-with-pil-and-pytorch-produces-different-results
                                                # Also note that in pil backend, horizontal flip of flow will invert the x direction.
                                                # Using Pillow-SIMD, pil backend is much faster than torch.
            flow = None,        # If "grey", each image is a 2D array of shape (H, W).
                                #           Optical flow has to be saved like
                                #           (u1, v1, u2, v2, u3, v3, u4, v4, ...)
                                #           So when indexing the frame dimension,
                                #           it should read [frame*2, frame*2+1] for each frame. 
                                #           Also, the CSV file has to have the actual number of frames,
                                #           instead of doubled number of frames because of the two channels.
                                # If "RG", each image is an 3D array of shape (H, W, 3),
                                #           and we're using R and G channels for the u and v optical flow channels.
            flow_neighbours = 5,    # How many flow frames to stack.
            video_id_to_label: dict = None,     # Pass a dictionary of mapping video ID to labels, and it will ignore the label in the CSV and get labels from here. Useful when using unsupported label types such as soft labels.
            ):
        """
        Construct the video loader with a given csv file. The format of
        the csv file is:
        ```
        num_classes     # set it to zero for single label. Only needed for multilabel.
        gulp_key_1 video_id_1 label_1 start_frame_1 end_frame_1
        gulp_key_2 video_id_2 label_2 start_frame_2 end_frame_2
        ...
        gulp_key_N video_id_N label_N start_frame_N end_frame_N
        ```

        `gulp_key` are the gulp dictionary key to access the video segment. Must be string.
        It will access something like this.
        ```
        gulpdata = GulpDirectory(gulp_dir_path)
        frames = gulpdata[gulp_key, [0, 1, 2]][0]   # It will ignore the meta data.
        ```

        Note that the `video_id` must be an integer.

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
        self._gulp_dir_path = gulp_dir_path
        self.gulp_dir = GulpDirectory(gulp_dir_path)
        self.mode = mode
        self.sample_index_code = sample_index_code.lower()
        
        self.processing_backend = processing_backend.lower()
        assert self.processing_backend in ['torch', 'pil']

        self.train_jitter_min = train_jitter_min
        self.train_jitter_max = train_jitter_max

        if mode == 'train':
            if train_class_balanced_sampling:
                logger.info('Class balanced sampling is ON for training.')
            self.train_class_balanced_sampling = train_class_balanced_sampling
        else:
            assert not train_class_balanced_sampling, 'train_class_balanced_sampling should only be set in train mode but used in test mode.'
            self.train_class_balanced_sampling = False

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

        if flow is not None:
            self.flow = flow.lower()
            assert self.flow in ['grey', 'rg'], f'Optical flow mode must be either grey or RG but got {flow}'

            self.flow_neighbours = flow_neighbours

            assert len(mean) in [1, 2]
            assert len(std)  in [1, 2]
            assert not greyscale, 'For optical flow data, it is impossible to use greyscale.'
            assert not bgr, 'For optical flow data, it is impossible to use BGR channel ordering.'
        else:
            self.flow = None

        self.video_id_to_label = video_id_to_label
        if video_id_to_label is not None:
            logger.info('video_id_to_label is provided. It will replace the labels in the CSV file.')

        # For training mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode == "train":
            self._num_clips = 1
        elif self.mode == "test":
            self._num_clips = test_num_spatial_crops

        assert test_num_spatial_crops in [1, 5, 10], "1 for centre, 5 for centre and four corners, 10 for their horizontal flips"
        self.test_num_spatial_crops = test_num_spatial_crops

        logger.info(f"Constructing gulp video dataset {mode=}...")
        self._construct_loader()

    def _construct_loader(self):#{{{
        """
        Construct the video loader.
        """
        assert os.path.exists(self._csv_file), "{} not found".format(
            self._csv_file
        )

        self._gulp_keys = []
        self._video_ids = []
        self._labels = []
        self._start_frames = []    # number of sample video frames
        self._end_frames = []    # number of sample video frames
        self._spatial_temporal_idx = []

        if self.train_class_balanced_sampling:
            self._indices_per_class = {}   # Dict[int, List[int]]

        with open(self._csv_file, "r") as f:
            self.num_classes = int(f.readline())
            if self.num_classes > 0:
                assert not self.train_class_balanced_sampling, "We don't know how to class-balance a multilabel dataset."

            for clip_idx, key_label in enumerate(f.read().splitlines()):
                assert len(key_label.split()) == 5
                gulp_key, video_id, label, start_frame, end_frame = key_label.split()

                if self.video_id_to_label is None:
                    if self.num_classes > 0:
                        label_list = label.split(",")
                        label = np.zeros(self.num_classes, dtype=np.float32)
                        for label_idx in label_list:
                            label[int(label_idx)] = 1.0       # one hot encoding
                    else:
                        label = int(label)
                else:
                    label = self.video_id_to_label[int(video_id)]

                for idx in range(self._num_clips):
                    self._gulp_keys.append(gulp_key)
                    self._video_ids.append(int(video_id))
                    self._labels.append(label)
                    self._start_frames.append(int(start_frame))
                    self._end_frames.append(int(end_frame))
                    self._spatial_temporal_idx.append(idx)

                    if self.train_class_balanced_sampling:
                        if label in self._indices_per_class.keys():
                            self._indices_per_class[label].append(len(self._video_ids)-1)  # last added index
                        else:
                            self._indices_per_class[label] = [len(self._video_ids)-1]       # last added index

        assert (
            len(self._gulp_keys) > 0
        ), f"Failed to load gulp video loader from {self._csv_file}"
        
        logger.info(f"Constructing gulp video dataloader (size: {len(self)}) from {self._csv_file}")
        #}}}

    def filter_samples(self, video_ids: list):
        """Given a video_ids list, filter the samples.
        Used for visualisation.
        """
        indices_of_video_ids = [x for x, v in enumerate(self._video_ids) if v in video_ids]

        self._gulp_keys = [self._gulp_keys[x] for x in indices_of_video_ids]
        self._video_ids = [self._video_ids[x] for x in indices_of_video_ids]
        self._labels = [self._labels[x] for x in indices_of_video_ids]
        self._start_frames = [self._start_frames[x] for x in indices_of_video_ids]
        self._end_frames = [self._end_frames[x] for x in indices_of_video_ids]
        self._spatial_temporal_idx = [self._spatial_temporal_idx[x] for x in indices_of_video_ids]
        if self.train_class_balanced_sampling:
            raise NotImplementedError()


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
            index (int): Note that it will change from the index argument if self.train_class_balanced_sampling is True.
        """

        if self.train_class_balanced_sampling:
            # Completely ignore the index, and return new index in a class-balanced way.
            random_label = random.choice(list(self._indices_per_class.keys()))
            random_index = random.choice(self._indices_per_class[random_label])
            index = random_index
            

        crop_size = self.crop_size
        if self.mode == "train":
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.train_jitter_min
            max_scale = self.train_jitter_max
            sample_uniform = False
        elif self.mode == "test":
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
            frame_indices = utils.sparse_frame_indices(num_video_frames, self.num_frames, uniform=sample_uniform, num_neighbours=1 if self.flow is None else self.flow_neighbours)
        elif self.sample_index_code == 'tsn':
            frame_indices = utils.TSN_sample_indices(num_video_frames, self.num_frames, mode = self.mode, new_length=1 if self.flow is None else self.flow_neighbours)
        elif self.sample_index_code == 'tdn':
            frame_indices = utils.TDN_sample_indices(num_video_frames, self.num_frames, mode = self.mode)
        elif self.sample_index_code == 'tdn_greyst':
            frame_indices = utils.TDN_sample_indices(num_video_frames, self.num_frames, mode = self.mode, new_length=15)
        else:
            raise ValueError(f'Wrong self.sample_index_code: {self.sample_index_code}. Should be pyvideoai, TSN, TDN')

        frame_indices = [idx+self._start_frames[index] for idx in frame_indices]     # add offset (frame number start)

        if self.processing_backend == 'torch':
            if self.flow == 'grey':
                # Frames are saved as (u0, v0, u1, v1, ...)
                # Read pairs of greyscale images.
                frame_indices = [idx*2+uv for idx in frame_indices for uv in range(2)]
                frames = np.stack(self.gulp_dir[self._gulp_keys[index], frame_indices][0])     # (T*2, H, W)
                TC, H, W = frames.shape
                frames = np.reshape(frames, (TC//2, 2, H, W))    # (T, C=2, H, W)
                frames = np.transpose(frames, (0,2,3,1))         # (T, H, W, C=2) 
            else:
                frames = np.stack(self.gulp_dir[self._gulp_keys[index], frame_indices][0])     # (T, H, W, C=3)
                                                                                            # or if greyscale images, (T, H, W)
                if frames.ndim == 3:
                    # Greyscale images. (T, H, W) -> (T, H, W, 1)
                    frames = np.expand_dims(frames, axis=-1)

                if self.flow == 'rg':
                    frames = frames[..., 0:2]   # Use R and G as u and v (x,y). Discard B channel.


            if self.bgr:
                frames = frames[..., ::-1]

            if self.greyscale:
                raise NotImplementedError()

            frames = torch.from_numpy(frames)

            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.mean, self.std, normalise = self.normalise
            )

            if self.flow is not None:
                # Reshape so that neighbouring frames go in the channel dimension.
                _, H, W, _ = frames.shape
                # T*neighbours, H, W, C -> T*neighbours, C, H, W
                frames = frames.permute(0, 3, 1, 2)
                frames = frames.reshape(self.num_frames, 2*self.flow_neighbours, H, W)  # T, C=2*neighbours, H, W
                # T, C, H, W -> C, T, H, W
                frames = frames.permute(1, 0, 2, 3)
            else:
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

        elif self.processing_backend == 'pil':
            is_flow = self.flow is not None
            pil_transform = Compose(
                    [
                        GroupNDarrayToPILImage(),
                        GroupGrayscale() if self.greyscale else IdentityTransform(),
                        GroupScale(round(np.random.uniform(min_scale, max_scale))),
                        GroupRandomCrop(crop_size) if spatial_sample_index < 0 else GroupOneOfFiveCrops(crop_size, spatial_sample_index, is_flow = is_flow),
                        GroupRandomHorizontalFlip(is_flow = is_flow) if spatial_sample_index < 0 and self.train_horizontal_flip else IdentityTransform(),
                        GroupPILImageToNDarray(),
                        np.stack,
                    ]
                )
            if self.flow == 'grey':
                # Frames are saved as (u0, v0, u1, v1, ...)
                # Read pairs of greyscale images.
                frame_indices = [idx*2+uv for idx in frame_indices for uv in range(2)]
                frames = self.gulp_dir[self._gulp_keys[index], frame_indices][0]     # list(ndarray(H, W)) size T*2

                frames = pil_transform(frames)
                TC, H, W = frames.shape
                frames = np.reshape(frames, (TC//2, 2, H, W))    # (T, C=2, H, W)
                frames = np.transpose(frames, (0,2,3,1))         # (T, H, W, C=2) 
            else:
                frames = self.gulp_dir[self._gulp_keys[index], frame_indices][0]

                frames = pil_transform(frames)      # (T, H, W, C) or (T, H, W) if greyscaled images

                if frames.ndim == 3:
                    # Greyscale images. (T, H, W) -> (T, H, W, 1)
                    frames = np.expand_dims(frames, axis=-1)

                if self.flow == 'rg':
                    frames = frames[..., 0:2]   # Use R and G as u and v (x,y). Discard B channel.


            if self.bgr:
                frames = frames[..., ::-1]

            frames = torch.from_numpy(frames)

            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.mean, self.std, normalise = self.normalise
            )

            if is_flow:
                # Reshape so that neighbouring frames go in the channel dimension.
                _, H, W, _ = frames.shape
                # T*neighbours, H, W, C -> T*neighbours, C, H, W
                frames = frames.permute(0, 3, 1, 2)
                frames = frames.reshape(self.num_frames, 2*self.flow_neighbours, H, W)  # T, C=2*neighbours, H, W
                # T, C, H, W -> C, T, H, W
                frames = frames.permute(1, 0, 2, 3)
            else:
                # T, H, W, C -> C, T, H, W
                frames = frames.permute(3, 0, 1, 2)

        video_id = self._video_ids[index]
        label = self._labels[index]
        return frames, video_id, label, spatial_sample_index, index, np.array(frame_indices)


    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._gulp_keys)

