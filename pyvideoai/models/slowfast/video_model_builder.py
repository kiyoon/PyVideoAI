#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn

from . import weight_init_helper as init_helper
#from slowfast.models import head_helper, resnet_helper, stem_helper
from . import head_helper, resnet_helper, stem_helper

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slowonly": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slowonly": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}


class FuseFastToSlow(nn.Module):#{{{
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = nn.BatchNorm3d(
            dim_in * fusion_conv_channel_ratio, eps=eps, momentum=bn_mmt
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


class SlowFastModel(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFastModel, self).__init__()
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES
                    // cfg.SLOWFAST.ALPHA
                    // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                ],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x#}}}


class ResNetModel(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, SlowOnly).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, model_arch, resnet_depth, num_classes, num_frames, fc_init_std = 0.01, zero_init_final_bn = True, crop_size=224, input_channel_num=[3],
            nonlocal_location = [[[]], [[]], [[]], [[]]],
            nonlocal_group = [[1], [1], [1], [1]]):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            fc_init_std (float): The std to initialise fc layer(s).
            zero_init_final_bn (bool): If True, initialise the gamma of the final BN of each block to zero.
        """
        super(ResNetModel, self).__init__()
        self.num_pathways = 1
        self._construct_network(model_arch, resnet_depth, num_classes, num_frames, crop_size, input_channel_num=input_channel_num,
                nonlocal_location = nonlocal_location,
                nonlocal_group = nonlocal_group)
        init_helper.init_weights(
            self, fc_init_std, zero_init_final_bn
        )

    def _construct_network(self, model_arch, resnet_depth, num_classes,
            num_frames = 8, crop_size = 224, dropout_rate = 0.5,
            num_groups = 1, width_per_group = 64, input_channel_num = [3],
            spatial_strides = [[1], [2], [2], [2]],
            spatial_dilations = [[1], [1], [1], [1]],
            num_block_temp_kernel = [[3], [4], [6], [3]],
            trans_func = "bottleneck_transform",
            stride_1x1 = False,
            inplace_relu = True,
            nonlocal_location = [[[]], [[]], [[]], [[]]],
            nonlocal_group = [[1], [1], [1], [1]],
            nonlocal_pool = [[[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]]],
            nonlocal_instantiation = "softmax",
            ):
        """
        Builds a single pathway ResNet model.

        Args:
            model_arch (str): c2d, c2dnopool, i3d, slowonly, slowfast
            resnet_depth (int): The number of weight layers. 50 or 101.
            num_classes (int): The number of classes for the output layer.
            num_frames (int) The number of frames of the input clip.
            crop_size (int): The spatial crop size of the input clip.
            dropout_rate (float): Always 0.5 in FAIR settings.
            num_groups (int): 1 for ResNet, larger than 1 for ResNeXt.
            width_per_group (int): 64 -> ResNet, 4 -> ResNeXt.
            input_channel_num (list of int): [3, 3] for SlowFast, [3] for others.
            spatial_strides (list of list of int): Size of stride on different res stages. SlowFast requires more values in each list.
            spatial_dilations (list of list of int): Size of dilation on different res stages. SlowFast requires more values in each list.
            num_block_temp_kernel (list of list of int): If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal kernel of 1 for the rest of the blocks. Default value may not always be suitable.
            trans_func (str): Always use default.
            stride_1x1 (bool): Apply stride to 1x1 conv. Always False.
            inplace_relu (bool): Apply ReLU in a inplace manner. Always True.

            nonlocal_location (list of list of list of int): [[[]], [[1, 3]], [[1, 3, 5]], [[]]] -> I3D NonLocal, [[[]], [[]], [[]], [[]]] -> No NonLocal. SlowFast requires more values in each list.
            nonlocal_group: SlowFast: [[1, 1], [1, 1], [1, 1], [1, 1]], others: [[1], [1], [1], [1]].
            nonlocal_pool: Size of pooling layers. Always use default unless AVA SlowFast.
            nonlocal_instantiation (str): softmax or dot_product.
        """
        assert model_arch in _POOL1.keys()
        pool_size = _POOL1[model_arch]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert resnet_depth in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[resnet_depth]

        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[model_arch]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=input_channel_num,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=spatial_strides[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=num_block_temp_kernel[0],
            nonlocal_inds=nonlocal_location[0],
            nonlocal_group=nonlocal_group[0],
            nonlocal_pool=nonlocal_pool[0],
            instantiation=nonlocal_instantiation,
            trans_func_name=trans_func,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=spatial_dilations[0],
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=spatial_strides[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=num_block_temp_kernel[1],
            nonlocal_inds=nonlocal_location[1],
            nonlocal_group=nonlocal_group[1],
            nonlocal_pool=nonlocal_pool[1],
            instantiation=nonlocal_instantiation,
            trans_func_name=trans_func,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=spatial_dilations[1],
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=spatial_strides[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=num_block_temp_kernel[2],
            nonlocal_inds=nonlocal_location[2],
            nonlocal_group=nonlocal_group[2],
            nonlocal_pool=nonlocal_pool[2],
            instantiation=nonlocal_instantiation,
            trans_func_name=trans_func,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=spatial_dilations[2],
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=spatial_strides[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=num_block_temp_kernel[3],
            nonlocal_inds=nonlocal_location[3],
            nonlocal_group=nonlocal_group[3],
            nonlocal_pool=nonlocal_pool[3],
            instantiation=nonlocal_instantiation,
            trans_func_name=trans_func,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=spatial_dilations[3],
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=[width_per_group * 32],
            num_classes=num_classes,
            pool_size=[
                [
                    num_frames // pool_size[0][0],
                    crop_size // 32 // pool_size[0][1],
                    crop_size // 32 // pool_size[0][2],
                ]
            ],
            dropout_rate=dropout_rate,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
#        if self.enable_detection:
#            x = self.head(x, bboxes)
#        else:
        x = self.head(x)
        return x


def ResNetI3DNonlocal(resnet_depth, num_classes, num_frames, fc_init_std = 0.01, zero_init_final_bn = True, crop_size=224, input_channel_num=[3]):
    return ResNetModel('i3d', resnet_depth, num_classes, num_frames, fc_init_std, zero_init_final_bn, crop_size, input_channel_num,
            nonlocal_location = [[[]], [[1, 3]], [[1, 3, 5]], [[]]],
            nonlocal_group = [[1], [1], [1], [1]])
