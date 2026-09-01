import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torchvision.models import wide_resnet50_2

class PatchCore(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.backbone = wide_resnet50_2(weights="IMAGENET1K_V1").to(self.device)
        self.backbone.eval()
        self.feature_maps = []
        self._register_hooks()
        self.memory_bank = []
        self.n_neighbors = 5

    def _register_hooks(self):
        def hook(module, input, output):
            self.feature_maps.append(output)
        self.backbone.layer2[-1].register_forward_hook(hook)
        self.backbone.layer3[-1].register_forward_hook(hook)

    def extract_features(self, x):
        self.feature_maps.clear()
        with torch.no_grad():
            self.backbone(x)
        feat1 = F.interpolate(self.feature_maps[0], size=64, mode='bilinear', align_corners=False)
        feat2 = F.interpolate(self.feature_maps[1], size=64, mode='bilinear', align_corners=False)
        feat = torch.cat([feat1, feat2], dim=1)
        return feat

    def build_memory_bank(self, dataloader):
        print("[PatchCore] Building memory bank...")
        for imgs, _ in dataloader:
            imgs = imgs.to(self.device)
            feat = self.extract_features(imgs)
            b, c, h, w = feat.shape
            patches = feat.permute(0, 2, 3, 1).reshape(-1, c)
            self.memory_bank.append(patches.cpu().numpy())
        self.memory_bank = np.concatenate(self.memory_bank, axis=0)
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='minkowski', p=2)
        self.knn.fit(self.memory_bank)

    def predict_score(self, x):
        feat = self.extract_features(x)
        b, c, h, w = feat.shape
        patches = feat.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
        dists, _ = self.knn.kneighbors(patches)
        score = np.mean(dists, axis=1)
        score = score.reshape(b, h, w)
        score = torch.tensor(score).to(self.device)
        score = F.interpolate(score.unsqueeze(1), size=256, mode='bilinear', align_corners=False)
        return score.squeeze(1)