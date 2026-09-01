import torch
import torch.nn as nn

class DiffusionEnhance(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.conv(x)  