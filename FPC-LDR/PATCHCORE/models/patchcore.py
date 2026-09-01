import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import IMAGE_SIZE

class PatchCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_bank = []  # 保存多层特征

    # 分别存储各层特征，不拼接
    def build_memory(self, features_list):
        feats_dict = {}
        for i, f in enumerate(features_list):
            b, c, h, w = f.shape
            feat = f.permute(0, 2, 3, 1).reshape(-1, c)
            # 随机采样 10 万点，避免爆显存
            if len(feat) > 100000:
                idx = torch.randperm(len(feat))[:100000]
                feat = feat[idx]
            feats_dict[f"layer_{i}"] = feat
        self.memory_bank = feats_dict

    # 分别计算每一层的异常分数
    def compute_score(self, features):
        score = 0
        for idx, f in enumerate(features):
            b, c, h, w = f.shape
            feat = f.permute(0, 2, 3, 1).reshape(b, -1, c)
            bank = self.memory_bank[f"layer_{idx}"].to(feat.device)
            dist = torch.cdist(feat, bank)  # 计算距离
            min_dist, _ = torch.min(dist, dim=-1)
            min_dist = min_dist.view(b, h, w).unsqueeze(1)
            min_dist = F.interpolate(min_dist, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
            score += min_dist
        return score