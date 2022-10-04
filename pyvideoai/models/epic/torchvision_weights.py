from typing import Callable, get_type_hints, get_args
from enum import Enum
from packaging import version
import torchvision


def get_model_weights_enum(model_func: Callable):
    """
    Since torchvision 0.13, the new weights parameter is introduced.
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

    possible_types = get_args(get_type_hints(model_func)['weights'])
    for possible_type in possible_types:
        if possible_type is None:
            continue
        elif issubclass(possible_type, Enum):
            return possible_type



def _torchvision_0_13() -> bool:
    """
    Return True if Torchvision is new version that deprecates `pretrained` in favour of `weights` parameter.
    """
    return version.parse('0.13.0') <= version.parse(torchvision.__version__)


def get_torchvision_model(model: str, pretrained: str):
    pretrained = pretrained.lower()
    model_func = getattr(torchvision.models, model.lower())
    if _torchvision_0_13():
        if pretrained in ["imagenet", "imagenet1k_v1"]:
            backbone_pretrained = get_model_weights_enum(model_func).IMAGENET1K_V1
        elif pretrained == 'imagenet1k_v2':
            backbone_pretrained = get_model_weights_enum(model_func).IMAGENET1K_V2
        elif pretrained == 'default':
            backbone_pretrained = get_model_weights_enum(model_func).DEFAULT
        elif pretrained is None:
            backbone_pretrained = None
        else:
            assert ValueError(f'Not recognised {pretrained = } with the {model = }.')

        return model_func(
            weights=backbone_pretrained
        )
    else:
        if pretrained in ["imagenet", "imagenet1k_v1"]:
            backbone_pretrained = "imagenet"
        elif pretrained is None:
            backbone_pretrained = None
        else:
            assert ValueError(f'Not recognised {pretrained = } with the {model = }. Maybe torchvision version is too low?')
        return model_func(
            pretrained=backbone_pretrained
        )


def main():
    weights = get_model_weights_enum(torchvision.models.resnet50)
    print(weights)
    print(weights.IMAGENET1K_V1)

    weights = get_model_weights_enum(torchvision.models.vgg16)
    print(weights)
    print(weights.IMAGENET1K_V1)

if __name__ == '__main__':
    main()
