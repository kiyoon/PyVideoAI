import torch
from torch import nn
from torchvision import models

from BatchRelationalModule import BatchRelationalModule

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ResnetRN(nn.Module):
    def __init__(self):
        super(ResnetRN, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.downsample_conv = nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)
        self.spatial_RN = BatchRelationalModule((64,7,7))
    
    def forward(self, x):
        # Let x = (N, 3, 224, 224)
        x = self.model(x)   # (N, 2048, 7, 7)
        x = self.downsample_conv(x)     # (N, 64, 7, 7)
        x = self.spatial_RN(x)          # (N, 64)
        return x


if __name__ == '__main__':

    model = ResnetRN()


    #model.fc = Identity()
    #model.avgpool = Identity()
    print(model)
    x = torch.randn(10, 3, 224, 224)
    output = model(x)
    print(output.shape)
