import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 修复警告，使用新的 weights 接口
        net = resnet50(weights="IMAGENET1K_V1")
        self.layer1 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2 = net.layer2
        self.layer3 = net.layer3

    def forward(self, x):
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return [f2, f3]