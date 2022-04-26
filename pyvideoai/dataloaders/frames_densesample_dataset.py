from __future__ import annotations
import numpy as np
import os
import torch
import torch.utils.data

#import slowfast.utils.logging as logging
import logging

from . import utils as utils

logger = logging.getLogger(__name__)


class FramesDensesampleDataset(torch.utils.data.Dataset):
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
            train_jitter_min=256, train_jitter_max=320,
            train_horizontal_flip=True,
            test_scale=256,
            test_ensemble_mode: str = 'fixed_crops',
            test_num_ensemble_views: int = 10,
            test_ensemble_stride: int = None,
            test_num_spatial_crops: int = 3,
            crop_size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],
            normalise = True,           # divide pixels by 255
            bgr = False,
            greyscale = False,
            path_prefix = "",
            num_retries = 10,
            flow = None,        # If 'RG', treat R and G channels as u and v. Discard B.
                                # If 'RR', load two greyscale images and use R channel only.
                                #     Need to define flow_folder_x and flow_folder_y,
                                #     and the path in CSV needs to have {flow_direction}
                                #     which will be replaced to the definitions.
            flow_neighbours = 5,    # How many flow frames to stack.
            flow_folder_x = 'u',
            flow_folder_y = 'v',    # These are only needed for flow='RR'
            video_id_to_label: dict = None,     # Pass a dictionary of mapping video ID to labels, and it will ignore the label in the CSV and get labels from here. Useful when using unsupported label types such as soft labels.
            ):
        """
        Construct the video loader with a given csv file. The format of
        the csv file is:
        ```
        num_classes     # set it to zero for single label. Only needed for multilabel.
        path_to_frames_dir_1/{frame:05d}.jpg video_id_1 label_1 start_frame_1 end_frame_1
        path_to_frames_dir_2/{frame:05d}.jpg video_id_2 label_2 start_frame_2 end_frame_2
        ...
        path_to_frames_dir_N/{frame:05d}.jpg video_id_N label_N start_frame_N end_frame_N
        ```
        Args:
            mode (string): Options includes `train`, or `test` mode.
                For the train mode, the data loader will sample one clip per video.
                For the test mode, the data loader will sample multiple clips per video.
            test_ensemble_mode: 'fixed_crops', or 'fixed_stride'.
                'fixed_crops' will sample fixed number of temporal crops per segment.
                'fixed_stride' will sample videos in fixed stride.
            test_num_ensemble_views: If int type, sample fixed number of temperal views
                per segment no matter how long the segment is.
            test_ensemble_stride: When `test_ensemble_mode` is 'fixed_stride', this is the stride of the sliding window.
                For example, when frame range is [0, 20], num_frames=8, sampling_rate=1 and test_ensemble_stride=4 it will sample
                clips with frame number [0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11], [8, 9, 10, 11, 12, 13, 14, 15], and [12, 13, 14, 15, 16, 17, 18, 19].
                (Note it won't sample frame 20.)
                If it can't sample 1 clip, it will still sample 1 clip by duplicating the last frame.
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

        self.num_frames = num_frames
        self.sampling_rate = sampling_rate

        if test_ensemble_mode == 'fixed_crops':
            assert test_num_ensemble_views > 0, f'Unsupported {test_num_ensemble_views = }'
            self.test_num_ensemble_views = test_num_ensemble_views
        elif test_ensemble_mode == 'fixed_stride':   # 'fixed_stride'
            assert isinstance(test_ensemble_stride, int)
            assert test_ensemble_stride > 0, f'Unsupported {test_ensemble_stride = }'
            self.test_ensemble_stride = test_ensemble_stride
        else:
            raise ValueError(f'Unknown {test_ensemble_mode = }')
        self.test_ensemble_mode = test_ensemble_mode

        assert test_num_spatial_crops in [1, 3], "1 for centre, 3 for centre and left,right/top,bottom"
        self.test_num_spatial_crops = test_num_spatial_crops

        self.train_horizontal_flip = train_horizontal_flip

        self.crop_size = crop_size

        if greyscale:
            assert len(mean) == 1
            assert len(std) == 1
        else:
            assert len(mean) in [1, 3]
            assert len(std) in [1, 3]
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

        self.normalise = normalise
        self.bgr = bgr
        self.greyscale = greyscale

        if flow is None:
            self.flow = None
        else:
            # string
            self.flow = flow.lower()
            self.flow_neighbours = flow_neighbours

            if self.flow == 'rg':
                assert len(mean) in [1, 2]
                assert len(std)  in [1, 2]
                assert not greyscale, 'For optical flow RG data, it is impossible to use greyscale.'
                assert not bgr, 'For optical flow RG data, it is impossible to use BGR channel ordering.'
            elif self.flow == 'rr':
                assert len(mean) in [1, 2]
                assert len(std)  in [1, 2]
                assert not greyscale, 'For optical flow RR data, it is impossible to use greyscale. It actually just uses R channels and ignore the rest.'
                assert not bgr, 'For optical flow RR data, it is impossible to use BGR channel ordering.'
                self.flow_folder_x = flow_folder_x
                self.flow_folder_y = flow_folder_y
            else:
                raise ValueError(f'Not recognised flow format: {self.flow}. Choose one of RG (use red and green channels), RR (two greyscale images representing x and y directions)')

        self.video_id_to_label = video_id_to_label
        if video_id_to_label is not None:
            logger.info('video_id_to_label is provided. It will replace the labels in the CSV file.')

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
        self._spatial_idx = []
        self._temporal_idx = []
        self._num_temporal_crops = []
        with open(self._csv_file, "r") as f:
            self.num_classes = int(f.readline())
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 5
                path, video_id, label, start_frame, end_frame = path_label.split()
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                assert start_frame <= end_frame

                if self.num_classes > 0:
                    label_list = label.split(",")
                    label = np.zeros(self.num_classes, dtype=np.float32)
                    for label_idx in label_list:
                        label[int(label_idx)] = 1.0       # one hot encoding
                else:
                    label = int(label)

                # calculate number of crops for this clip.
                if self.mode == 'train':
                    num_spatial_crops = 1
                    num_temporal_crops = 1
                else:
                    num_spatial_crops = self.test_num_spatial_crops
                    if self.test_ensemble_mode == 'fixed_stride':
                        num_frames_segment = end_frame - start_frame + 1
                        if self.flow is None:
                            sample_span = (self.num_frames - 1) * self.sampling_rate + 1
                        else:
                            sample_span = (self.num_frames - 1) * self.sampling_rate + self.flow_neighbours

                        num_temporal_crops = (num_frames_segment - sample_span) // self.test_ensemble_stride + 1
                        if num_temporal_crops <= 0:
                            num_temporal_crops = 1
                    else:
                        num_temporal_crops = self.test_num_ensemble_views

                for spatial_idx in range(num_spatial_crops):
                    for temporal_idx in range(num_temporal_crops):
                        self._path_to_frames.append(
                            os.path.join(self._path_prefix, path)
                        )
                        self._video_ids.append(int(video_id))
                        self._labels.append(label)
                        self._start_frames.append(int(start_frame))
                        self._end_frames.append(int(end_frame))
                        self._spatial_idx.append(spatial_idx)
                        self._temporal_idx.append(temporal_idx)
                        self._num_temporal_crops.append(num_temporal_crops)
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
            temporal_sample_index = self._temporal_idx[index]
            # spatial_sample_index is in [0, 1, 2]. Corresponding to centre, left,
            # or right if width is larger than height, and middle, top,
            # or bottom if height is larger than width.
            spatial_sample_index = self._spatial_idx[index]

            min_scale, max_scale = [self.test_scale] * 2
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        num_video_frames = self._end_frames[index] - self._start_frames[index] + 1
        if self.mode == 'test' and self.test_ensemble_mode == 'fixed_stride':
            frame_indices = utils.strided_frame_indices(num_video_frames, self.num_frames, self.sampling_rate, clip_idx = temporal_sample_index, stride = self.test_ensemble_stride,
                    num_neighbours = 1 if self.flow is None else self.flow_neighbours)
        else:
            frame_indices = utils.dense_frame_indices(num_video_frames, self.num_frames, self.sampling_rate, clip_idx = temporal_sample_index, num_clips = self._num_temporal_crops[index],
                    num_neighbours = 1 if self.flow is None else self.flow_neighbours)
        frame_indices = [idx+self._start_frames[index] for idx in frame_indices]     # add offset (frame number start)

        if any(idx < self._start_frames[index] or idx > self._end_frames[index] for idx in frame_indices):
            raise NotImplementedError(f'Implementation is wrong. Trying to sample {frame_indices} but range of the segment is {self._start_frames[index]} to {self._end_frames[index]}.')

        if self.flow == 'rr':
            frame_paths_x = [self._path_to_frames[index].format(flow_direction=self.flow_folder_x, frame=frame_idx) for frame_idx in frame_indices]
            frame_paths_y = [self._path_to_frames[index].format(flow_direction=self.flow_folder_y, frame=frame_idx) for frame_idx in frame_indices]
            frames_x = utils.retry_load_images(frame_paths_x, retry=self._num_retries, backend='pytorch', bgr=False, greyscale=False)
            frames_y = utils.retry_load_images(frame_paths_y, retry=self._num_retries, backend='pytorch', bgr=False, greyscale=False)
            frames = torch.cat((frames_x[...,0:1], frames_y[...,0:1]), dim=-1)
        else:
            try:
                # {frame:05d} format (new)
                frame_paths = [self._path_to_frames[index].format(frame=frame_idx) for frame_idx in frame_indices]
            except IndexError:
                # {:05d} format (old)
                frame_paths = [self._path_to_frames[index].format(frame_idx) for frame_idx in frame_indices]

            frames = utils.retry_load_images(frame_paths, retry=self._num_retries, backend='pytorch', bgr=self.bgr, greyscale=self.greyscale)
            if self.flow == 'rg':
                frames = frames[..., 0:2]   # Use R and G as u and v (x,y). Discard B channel.


        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.mean, self.std, normalise = self.normalise
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        if self.mode == 'test' and self.test_num_spatial_crops != 3:
            spatial_sampling_func = utils.spatial_sampling_5
        else:
            spatial_sampling_func = utils.spatial_sampling
        frames, scale_factor_width, scale_factor_height, x_offset, y_offset, is_flipped = spatial_sampling_func(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.train_horizontal_flip,    # only applied when "train" mode (spatial_sample_index == -1)
        )


        video_id = self._video_ids[index]
        label = self._labels[index]
        return frames, video_id, label, spatial_sample_index, temporal_sample_index, index, np.array(frame_indices)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_frames)
