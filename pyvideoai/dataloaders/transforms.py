"""
Original code from TSN, but brought from EPIC-Kitchens repo.

Added by Kiyoon:
    GroupPILImageToNDarray
    GroupGrayscale
    GroupHorizontalFlip
    GroupOneOfFiveCrops

Bug fix by Kiyoon:
    GroupRandomHorizontalFlip always returning original image
    Warning fix from GroupScale

Removed the ones that assumes greyscale is optical flow (which is not true anymore because of GroupGrayscale):
    GroupOversample
    Stack   (instead, use GroupPILImageToNDarray and np.stack)
    ToTorchFormatTensor (instead, use torch.fromarray)

"""
import math
import numbers
import random
from typing import List

import numpy as np
import torchvision
from torchvision.transforms import InterpolationMode
from PIL import Image
from PIL import ImageOps

# line_profiler injects a "profile" into __builtins__. When not running under
# line_profiler we need to inject our own passthrough
if type(__builtins__) is not dict or "profile" not in __builtins__:
    profile = lambda f: f


class GroupNDarrayToPILImage:
    def __call__(self, imgs):
        return [Image.fromarray(img) for img in imgs]

class GroupPILImageToNDarray:
    def __call__(self, imgs):
        return [np.array(img) for img in imgs]

class GroupRandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @profile
    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop:
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    @profile
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupOneOfFiveCrops:
    """
    Return one of five crops without oversampling. Also possible to return horizontal flipped version of them.
    spatial_idx: 0: centre 
                1: top left 
                2: top right
                3: bottom left
                4: bottom right
                5: centre, horizontally flipped
                6: top left, horizontally flipped
                7: top right, horizontally flipped
                8: bottom left, horizontally flipped
                9: bottom right, horizontally flipped
    """
    def __init__(self, size, spatial_idx, is_flow=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.flip_worker = GroupHorizontalFlip(is_flow)
        self.spatial_idx = spatial_idx
    
    def __call__(self, img_group):
        crop_w, crop_h = self.size
        ret = []
        for img in img_group:
            offset_w, offset_h = GroupOneOfFiveCrops.crop_offset(*img.size, *self.size, self.spatial_idx)
            ret.append(img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)))

        if self.spatial_idx >= 5:
            ret = self.flip_worker(ret)

        return ret
        
    @staticmethod
    def crop_offset(image_w, image_h, crop_w, crop_h, spatial_idx: int):
        assert 0 <= spatial_idx < 10

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        if spatial_idx % 5 == 0:
            return 2 * w_step, 2 * h_step   # centre
        elif spatial_idx % 5 == 1:
            return 0, 0                     # top left
        elif spatial_idx % 5 == 2:
            return 4 * w_step, 0            # top right
        elif spatial_idx % 5 == 3:
            return 0, 4 * h_step            # bottom left
        elif spatial_idx % 5 == 4:
            return 4 * w_step, 4 * h_step   # bottom right

class GroupGrayscale:
    def __init__(self, size):
        self.worker = torchvision.transforms.Grayscale()

    @profile
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupHorizontalFlip:
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    @profile
    def __call__(self, img_group):
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        if self.is_flow:
            for i in range(0, len(ret), 2):
                ret[i] = ImageOps.invert(
                    ret[i]
                )  # invert flow pixel values when flipping
        return ret 

class GroupRandomHorizontalFlip:
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __init__(self, is_flow=False, prob=0.5):
        self.prob = prob
        self.worker = GroupHorizontalFlip(is_flow)

    @profile
    def __call__(self, img_group):
        v = random.random()
        if v < self.prob:
            return self.worker(img_group)
        return img_group


class GroupNormalize:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    @profile
    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, mean, std_dev in zip(tensor, rep_mean, rep_std):
            t.sub_(mean).div_(std_dev)

        return tensor


class GroupScale:
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        def _interpolation_modes_from_int(i: int) -> InterpolationMode:
            inverse_modes_mapping = {
                Image.NEAREST: InterpolationMode.NEAREST,
                Image.BILINEAR: InterpolationMode.BILINEAR,
                Image.BICUBIC: InterpolationMode.BICUBIC,
                Image.BOX: InterpolationMode.BOX,
                Image.HAMMING: InterpolationMode.HAMMING,
                Image.LANCZOS: InterpolationMode.LANCZOS,
            }
            return inverse_modes_mapping[i]
        self.worker = torchvision.transforms.Resize(size, _interpolation_modes_from_int(interpolation))

    @profile
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]




class GroupMultiScaleCrop:
    def __init__(
        self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True
    ):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = (
            input_size if not isinstance(input_size, int) else [input_size, input_size]
        )
        self.interpolation = Image.BILINEAR

    @profile
    def __call__(self, img_group):
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
            for img in crop_img_group
        ]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1]
            )

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop:
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    @profile
    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3.0 / 4, 4.0 / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class IdentityTransform:
    def __call__(self, data):
        return data



class ExtractTimeFromChannel:
    def __init__(self, channels: int):
        self.channels = channels

    def __call__(self, xs):
        return xs.reshape(-1, self.channels, xs.shape[1], xs.shape[2])
