import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLDM(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, device=None, 
                 noise_schedule="linear", loss_type="mse"):
        super().__init__()
        self.device = device
        self.loss_type = loss_type
        
        # =====================  =====================
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 4, 2, 1),
        ).to(self.device)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        ).to(self.device)

        # ===================== =====================
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.num_timesteps = 50
        
        if noise_schedule == "linear":
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(self.device)
        elif noise_schedule == "cosine":
            t = torch.linspace(0, self.num_timesteps, self.num_timesteps).to(self.device)
            self.beta = 0.5 * (1 - torch.cos(t * torch.pi / self.num_timesteps)) * self.beta_end
        elif noise_schedule == "quadratic":
            self.beta = torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_timesteps)**2
            self.beta = self.beta.to(self.device)
        elif noise_schedule == "constant":
            self.beta = torch.ones(self.num_timesteps).to(self.device) * 0.01
        else:
            # linear
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(self.device)
        
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def diffuse(self, x, t):
        noise = torch.randn_like(x).to(self.device)
        a_bar = self.alpha_bar[t][:, None, None, None]
        noisy = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * noise
        return noisy, noise

    def forward(self, x, t):
        x = x.to(self.device)
        noisy, noise = self.diffuse(x, t)
        latent = self.encoder(noisy)
        pred = self.decoder(latent)
        return pred, noise

    def get_loss(self, pred, noise):
        # =====================  =====================
        if self.loss_type == "mse":
            return F.mse_loss(pred, noise)
        elif self.loss_type == "l1":
            return F.l1_loss(pred, noise)
        elif self.loss_type == "perceptual":
            return F.mse_loss(pred, noise) * 1.2
        elif self.loss_type == "mse_perceptual":
            return 0.8 * F.mse_loss(pred, noise) + 0.2 * F.l1_loss(pred, noise)
        else:
            #  mse
            return F.mse_loss(pred, noise)

    def reconstruct(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            t = torch.full((x.shape[0],), self.num_timesteps-1, device=self.device).long()
            pred, _ = self.forward(x, t)
        return pred