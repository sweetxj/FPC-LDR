import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLDM(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, device=None):
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 4, 2, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.num_timesteps = 50
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def diffuse(self, x, t):
        noise = torch.randn_like(x).to(self.device)
        a_bar = self.alpha_bar[t][:, None, None, None]
        noisy = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * noise
        return noisy, noise

    def forward(self, x, t):
        noisy, noise = self.diffuse(x, t)
        latent = self.encoder(noisy)
        pred = self.decoder(latent)
        return pred, noise

    def reconstruct(self, x):
        with torch.no_grad():
            t = torch.full((x.shape[0],), self.num_timesteps-1, device=self.device).long()
            pred, _ = self.forward(x, t)
        return pred