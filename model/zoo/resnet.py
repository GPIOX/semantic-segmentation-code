import torch
import torch.nn as nn
import torch.nn.functional as F

from base.registry import MODEL

@MODEL.register_module()
class Model(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

