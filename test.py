import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
# from base.registry import MODEL

class Model(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.backbone = resnet.resnet18()
        self.decoder = nn.Conv2d(512, 12, 1)

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

x = torch.randn(4, 3, 480, 320).cuda()
model = Model(1,2).cuda() # Model(1, 2).cuda()
# model.fc = nn.Conv2d(256, 12, 1).cuda()
y = model(x)
print(y.shape)