import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from base.registry import MODEL

@MODEL.register_module()
class Model(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.backbone = resnet.resnet101()
        self.decoder = nn.Conv2d(2048, 12, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = nn.Softmax(dim=1)(x)

        return x

