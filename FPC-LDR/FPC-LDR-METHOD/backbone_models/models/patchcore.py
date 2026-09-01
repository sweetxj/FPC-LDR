import torch
import torch.nn.functional as F
from utils.config import IMAGE_SIZE, MEMORY_SIZE

class PatchCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = []

    def build(self, f2_list, f3_list):
        f2_all = torch.cat(f2_list, dim=0)
        f3_all = torch.cat(f3_list, dim=0)

        if len(f2_all) > MEMORY_SIZE:
            f2_all = f2_all[torch.randperm(len(f2_all))[:MEMORY_SIZE]]
        if len(f3_all) > MEMORY_SIZE:
            f3_all = f3_all[torch.randperm(len(f3_all))[:MEMORY_SIZE]]

        self.memory = [f2_all, f3_all]

    def score(self, feats):
        s = 0
        for i, f in enumerate(feats):
            B, C, H, W = f.shape
            feat = f.permute(0,2,3,1).reshape(B, -1, C)
            bank = self.memory[i].to(feat.device)
            d = torch.cdist(feat, bank).min(dim=-1)[0]
            d = d.view(B, H, W).unsqueeze(1)
            d = F.interpolate(d, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear")
            s += d
        return s