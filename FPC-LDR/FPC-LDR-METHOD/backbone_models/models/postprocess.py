# import torch
# import torch.nn as nn

# class PostProcess(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.smooth = nn.AvgPool2d(5, 1, 2)
#     def forward(self, x):
#         return self.smooth(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class PostProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = nn.AvgPool2d(5, 1, 2)

    def forward(self, x):
        # 
        x = self.smooth(x)
        return x