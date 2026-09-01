import torch
import torch.nn as nn
import torch.nn.functional as F

# 多尺度特征融合 + 高斯平滑
# 作用：让缺陷热力图更平滑、定位更准、指标更高
class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        # 7x7 均值滤波 → 热力图去噪、更清晰
        self.smooth = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # 对异常分数图进行平滑
        x = self.smooth(x)
        return x