"""
Author: Kiyoon Kim (yoonkr33@gmail.com)

Since torchvision 0.13 (PyTorch 1.12), the new `weights` parameter is introduced and the original `pretrained` parameter is now deprecated.
It supports more pretrained weights but it is more difficult to find the right one.
This script provides a backward compatible but supporting new version in ease, using strings.
"""
from __future__ import annotations
from typing import Callable, get_type_hints, get_args
from enum import Enum
from packaging import version
import torchvision
from torch import nn


def _torchvision_0_13() -> bool:
    """
    Return True if Torchvision is new version that deprecates `pretrained` in favour of `weights` parameter.
    """
    return version.parse('0.13.0') <= version.parse(torchvision.__version__)



def get_model_weights_enum(model_func: Callable):
    """
    Since torchvision 0.13 (PyTorch 1.12), the new `weights` parameter is introduced and the original `pretrained` parameter is now deprecated.
    It supports more pretrained weights but it is more difficult to find the right one.
    This function helps getting the correct weight link.

    ```
    weights = get_model_weights_enum(torchvision.models.resnet50)
    print(weights)
    print(weights.IMAGENET1K_V1)

    weights = get_model_weights_enum(torchvision.models.vgg16)
    print(weights)
    print(weights.IMAGENET1K_V1)
    ```
    """
    assert _torchvision_0_13(), 'This function only works with torchvision >= 0.13'

    possible_types = get_args(get_type_hints(model_func)['weights'])
    for possible_type in possible_types:
        if possible_type is None:
            continue
        elif issubclass(possible_type, Enum):
            return possible_type



def get_torchvision_model(model: str, pretrained: str | None) -> nn.Module:
    """
    Use string instead of annoying enum for torchvision 0.13.
    Backward compatible with lower version of torchvision.

    Torchvision code:
    ```
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    ```

    Using this function:
    ```
    # Below two lines are equivalent.
    model = get_torchvision_model("resnet50", "IMAGENET1K_V1")
    model = get_torchvision_model("resnet50", "imagenet")
    ```
    """
    model_func = getattr(torchvision.models, model.lower())
    if _torchvision_0_13():
        if pretrained is None:
            backbone_pretrained = None
        else:
            pretrained = pretrained.lower()
            if pretrained in ["imagenet", "imagenet1k_v1"]:
                backbone_pretrained = get_model_weights_enum(model_func).IMAGENET1K_V1
            elif pretrained == 'imagenet1k_v2':
                backbone_pretrained = get_model_weights_enum(model_func).IMAGENET1K_V2
            elif pretrained == 'default':
                backbone_pretrained = get_model_weights_enum(model_func).DEFAULT
            else:
                raise ValueError(f'Not recognised {pretrained = } with the {model = }.')

        return model_func(weights=backbone_pretrained)
    else:
        if pretrained is None:
            backbone_pretrained = None
        else:
            pretrained = pretrained.lower()
            if pretrained in ["imagenet", "imagenet1k_v1", "default"]:
                backbone_pretrained = "imagenet"
            else:
                raise ValueError(f'Not recognised {pretrained = } with the {model = }. Maybe torchvision version is too low?')

        return model_func(pretrained=backbone_pretrained)


def main():
    #weights = get_model_weights_enum(torchvision.models.resnet50)
    #print(weights)
    #print(weights.IMAGENET1K_V1)

    #weights = get_model_weights_enum(torchvision.models.vgg16)
    #print(weights)
    #print(weights.IMAGENET1K_V1)

    model = get_torchvision_model("resnet50", None)
    print(model)
    model = get_torchvision_model("resnet50", "IMAGENET1K_V1")
    print(model)
    model = get_torchvision_model("resnet50", "imagenet")
    print(model)
    model = get_torchvision_model("resnet50", "DEFAULT")
    print(model)
    model = get_torchvision_model("inception_v3", "DEFAULT")
    print(model)

    print("Below would not work with torchvision < 0.13")
    model = get_torchvision_model("resnet50", "IMAGENET1K_V2")
    print(model)



if __name__ == '__main__':
    main()
