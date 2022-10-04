from typing import Callable, get_type_hints, get_args
from enum import Enum
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
    

def main():
    weights = get_model_weights_enum(torchvision.models.resnet50)
    print(weights)
    print(weights.IMAGENET1K_V1)

    weights = get_model_weights_enum(torchvision.models.vgg16)
    print(weights)
    print(weights.IMAGENET1K_V1)

if __name__ == '__main__':
    main()
