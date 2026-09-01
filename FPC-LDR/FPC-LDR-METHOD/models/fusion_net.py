import torch
import torch.nn as nn
from models.patchcore import PatchCore
from models.diffusion_ldm import SimpleLDM

class FusionModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.patchcore = PatchCore(device)
        self.diffusion = SimpleLDM(device=device).to(device)
        self.w1 = 0.6
        self.w2 = 0.4

    def train_diffusion(self, dataloader, epochs=10, lr=1e-4):
        print("[Fusion] Training diffusion branch...")
        opt = torch.optim.Adam(self.diffusion.parameters(), lr=lr)
        mse = nn.MSELoss()
        self.diffusion.train()
        for ep in range(epochs):
            total_loss = 0
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                t = torch.randint(0, self.diffusion.num_timesteps, (imgs.shape[0],), device=self.device).long()
                pred, noise = self.diffusion(imgs, t)
                loss = mse(pred, imgs)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f"Diffusion Epoch {ep+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        self.diffusion.eval()

    def predict(self, x):
        score_p = self.patchcore.predict_score(x)
        rec = self.diffusion.reconstruct(x)
        score_d = torch.mean((x - rec)**2, dim=1)
        score_d = (score_d - score_d.min()) / (score_d.max() - score_d.min() + 1e-8)
        score = self.w1 * score_p + self.w2 * score_d
        return score, rec