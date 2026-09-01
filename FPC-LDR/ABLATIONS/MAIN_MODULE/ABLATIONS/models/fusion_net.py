import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.patchcore import PatchCore
from models.diffusion_ldm import SimpleLDM


class AdaptiveFusionModule(nn.Module):
    """自适应加权融合模块：基于两路得分统计特征动态生成融合权重"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.weight_net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        ).to(self.device)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def _get_stats(self, score_map):
        mean_s = torch.mean(score_map, dim=(1, 2))
        std_s = torch.std(score_map, dim=(1, 2))
        max_s = torch.max(score_map.flatten(1), dim=1)[0]
        cv_s = std_s / (mean_s + 1e-8)
        # 强制转为float32，避免与Linear层权重类型不匹配
        return torch.stack([mean_s, std_s, max_s, cv_s], dim=1).float()

    def forward(self, score_mem, score_diff):
        stats_mem = self._get_stats(score_mem)
        stats_diff = self._get_stats(score_diff)
        stats_concat = torch.cat([stats_mem, stats_diff], dim=1)
        alpha = self.weight_net(stats_concat).squeeze(-1)
        alpha_unsq = alpha.unsqueeze(-1).unsqueeze(-1)
        fused_score = alpha_unsq * score_mem + (1 - alpha_unsq) * score_diff
        return fused_score, alpha


class FusionModel(nn.Module):
    def __init__(self, device, mode="full"):
        """
        消融模式对应表：
        - memory_only    : 仅记忆库分支 (配置A)
        - diffusion_only : 仅扩散分支 (配置B)
        - fixed_fusion   : 固定权重融合 (配置C)
        - adaptive_fusion: 自适应权重融合 (配置D)
        - full           : 完整方法 = 自适应融合 + 图像级得分优化 (配置E)
        """
        super().__init__()
        self.device = device
        self.mode = mode

        self.patchcore = None
        self.diffusion = None
        self.adaptive_fusion = None
        self.fixed_w_mem = 0.6
        self.fixed_w_diff = 0.4

        # 按需初始化分支
        if mode in ["memory_only", "fixed_fusion", "adaptive_fusion", "full"]:
            self.patchcore = PatchCore(device)
        if mode in ["diffusion_only", "fixed_fusion", "adaptive_fusion", "full"]:
            self.diffusion = SimpleLDM(device=device).to(device)
        if mode in ["adaptive_fusion", "full"]:
            self.adaptive_fusion = AdaptiveFusionModule(device)

        self.enable_score_opt = (mode == "full")

    def train_diffusion(self, dataloader, epochs=10, lr=1e-4):
        if self.diffusion is None:
            print("[Warning] 当前模式无扩散分支，跳过训练")
            return
        print(f"[Fusion] 训练扩散分支 (模式: {self.mode})...")
        opt = torch.optim.Adam(self.diffusion.parameters(), lr=lr)
        mse_loss = nn.MSELoss()
        self.diffusion.train()

        for ep in range(epochs):
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch [{ep+1}/{epochs}]", leave=False)
            for imgs, _ in pbar:
                imgs = imgs.to(self.device)
                t = torch.randint(0, self.diffusion.num_timesteps, (imgs.shape[0],), device=self.device).long()
                pred, _ = self.diffusion(imgs, t)
                loss = mse_loss(pred, imgs)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {ep+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
        self.diffusion.eval()

    def _min_max_norm(self, score_map):
        """逐图归一化到[0,1]，仅用于像素级指标与可视化"""
        s_min = score_map.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        s_max = score_map.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        return (score_map - s_min) / (s_max - s_min + 1e-8)

    def _compute_image_score(self, pixel_score_raw):
        """图像级得分：从原始得分计算，保证全局可比性"""
        # 关键修复：detach() 切断梯度，再转 numpy
        score_np = pixel_score_raw[0].detach().cpu().numpy()
        if self.enable_score_opt:
            # 完整模式：多百分位加权优化
            p90 = np.percentile(score_np, 90)
            p95 = np.percentile(score_np, 95)
            p99 = np.percentile(score_np, 99)
            return 0.2 * p90 + 0.5 * p95 + 0.3 * p99
        else:
            # 基线模式：单一95分位
            return np.percentile(score_np, 95)

    def predict(self, x):
        """
        Returns:
            score_raw: [B, H, W] 原始融合得分（全局可比，用于图像级指标）
            score_norm: [B, H, W] 逐图归一化得分（用于像素级指标、可视化）
            recon_img: [B, 3, H, W] 重建图像（无扩散分支返回None）
            img_score: float 图像级异常得分
        """
        score_mem_raw = None
        score_diff_raw = None
        recon_img = None

        # 1. 计算两路原始得分
        if self.patchcore is not None:
            score_mem_raw = self.patchcore.predict_score(x)
        if self.diffusion is not None:
            recon_img = self.diffusion.reconstruct(x)
            score_diff_raw = torch.mean((x - recon_img) ** 2, dim=1)

        # 2. 两路得分融合（先归一化再融合，再还原原始量级）
        if self.mode == "memory_only":
            score_raw = score_mem_raw
        elif self.mode == "diffusion_only":
            score_raw = score_diff_raw
        elif self.mode == "fixed_fusion":
            mem_norm = self._min_max_norm(score_mem_raw)
            diff_norm = self._min_max_norm(score_diff_raw)
            fused_norm = self.fixed_w_mem * mem_norm + self.fixed_w_diff * diff_norm
            raw_min = score_mem_raw.min()
            raw_max = score_mem_raw.max()
            score_raw = fused_norm * (raw_max - raw_min) + raw_min
        elif self.mode in ["adaptive_fusion", "full"]:
            mem_norm = self._min_max_norm(score_mem_raw)
            diff_norm = self._min_max_norm(score_diff_raw)
            fused_norm, _ = self.adaptive_fusion(mem_norm, diff_norm)
            raw_min = score_mem_raw.min()
            raw_max = score_mem_raw.max()
            score_raw = fused_norm * (raw_max - raw_min) + raw_min
        else:
            raise ValueError(f"未知消融模式: {self.mode}")

        # 关键修复：推理阶段切断计算图，避免后续转 numpy 报错
        score_raw = score_raw.detach()

        # 3. 生成逐图归一化的像素级得分
        score_norm = self._min_max_norm(score_raw)

        # 4. 计算图像级得分
        img_score = self._compute_image_score(score_raw)

        return score_raw, score_norm, recon_img, img_score
