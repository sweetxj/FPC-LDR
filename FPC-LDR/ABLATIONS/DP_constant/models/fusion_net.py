# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.patchcore import PatchCore
# from models.diffusion_ldm import SimpleLDM

# class FusionModel(nn.Module):
#     def __init__(self, device, backbone_name="resnet50", noise_schedule="linear", 
#                  recon_strategy="diffusion", loss_type="mse"):
#         super().__init__()
#         self.device = device
#         self.backbone_name = backbone_name
#         self.noise_schedule = noise_schedule
#         self.recon_strategy = recon_strategy
#         self.loss_type = loss_type

#         # ===================== 1. PatchCore =====================
#         # 注意：你的PatchCore内部固定使用wide_resnet50_2，忽略backbone_name参数
#         self.patchcore = PatchCore(device)

#         # ===================== 2. =====================
#         self.diffusion = SimpleLDM(in_channels=3, latent_dim=128, device=device)
#         self.diffusion.to(device)

#     def train_diffusion(self, train_loader, epochs=50):
#         optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=1e-4)
#         self.diffusion.train()
        
#         print("Starting SimpleLDM training...")
#         for epoch in range(epochs):
#             total_loss = 0.0
#             for img, _ in train_loader:
#                 img = img.to(self.device)
#                 # 
#                 img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
#                 # 
#                 t = torch.randint(0, self.diffusion.num_timesteps, (img.shape[0],), device=self.device).long()
                
#                 # 
#                 pred_noise, true_noise = self.diffusion(img, t)
                
#                 # 
#                 loss = F.mse_loss(pred_noise, true_noise)
                
#                 # 
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
            
#             if (epoch + 1) % 10 == 0:
#                 print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

#     def predict(self, img, num_steps=50):
#         if self.recon_strategy == "direct":
#             # 
#             # predict_score
#             score = self.patchcore.predict_score(img)
#             rec = img  # 
#         else:
#             # 
#             # [0,1]
#             img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
#             rec = self.diffusion.reconstruct(img_norm)
            
#             # 
#             score_patch = self.patchcore.predict_score(img)
#             score_rec = torch.mean((img_norm - rec)**2, dim=1, keepdim=True)
            
#             # 
#             if len(score_patch.shape) == 3:
#                 score_patch = score_patch.unsqueeze(1)
            
#             score_rec = F.interpolate(score_rec, size=score_patch.shape[2:], mode='bilinear', align_corners=False)
            
#             # 
#             score_patch_norm = (score_patch - score_patch.min()) / (score_patch.max() - score_patch.min() + 1e-8)
#             score_rec_norm = (score_rec - score_rec.min()) / (score_rec.max() - score_rec.min() + 1e-8)
#             score = 0.7 * score_patch_norm + 0.3 * score_rec_norm
            
#             # 
#             if score.shape[1] == 1:
#                 score = score.squeeze(1)
#         return score, rec

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.patchcore import PatchCore
# from models.diffusion_ldm import SimpleLDM

# class FusionModel(nn.Module):
#     def __init__(self, device, backbone_name="resnet50", noise_schedule="linear", 
#                  recon_strategy="diffusion", loss_type="mse"):
#         super().__init__()
#         self.device = device
#         self.backbone_name = backbone_name
#         self.noise_schedule = noise_schedule
#         self.recon_strategy = recon_strategy
#         self.loss_type = loss_type

#         # ===================== 1 =====================
#         self.patchcore = PatchCore(device)

#         # ===================== 2.  =====================
#         self.diffusion = SimpleLDM(in_channels=3, latent_dim=128, device=device)
#         self.diffusion.to(device)

#     def train_diffusion(self, train_loader, epochs=50):
#         optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=1e-4)
#         self.diffusion.train()
        
#         print("Starting SimpleLDM training...")
#         for epoch in range(epochs):
#             total_loss = 0.0
#             for img, _ in train_loader:
#                 img = img.to(self.device)
#                 img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
#                 t = torch.randint(0, self.diffusion.num_timesteps, (img.shape[0],), device=self.device).long()
#                 pred_noise, true_noise = self.diffusion(img, t)
#                 loss = F.mse_loss(pred_noise, true_noise)
                
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
            
#             if (epoch + 1) % 1 == 0:
#                 print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

#     def predict(self, img, num_steps=50):
#         # 
#         score_patch = self.patchcore.predict_score(img)
        
#         # 
#         score_norm = (score_patch - score_patch.min()) / (score_patch.max() - score_patch.min() + 1e-8)
        
#         # 
#         rec = img
        
#         return score_norm, rec

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.patchcore import PatchCore
# from models.diffusion_ldm import SimpleLDM

# class FusionModel(nn.Module):
#     def __init__(self, device):
#         super().__init__()
#         self.device = device
#         self.patchcore = PatchCore(device)
#         self.diffusion = SimpleLDM(device=device)

#     def train_diffusion(self, train_loader, epochs=20):
#         optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=1e-4)
#         self.diffusion.train()
#         print("Training LDM...")
        
#         for epoch in range(epochs):
#             total_loss = 0.0
#             for img, _ in train_loader:
#                 img = img.to(self.device)
#                 img = torch.clamp(img, 0, 1)

#                 t = torch.randint(0, self.diffusion.num_timesteps, (img.shape[0],), device=self.device).long()
#                 pred, noise = self.diffusion(img, t)
#                 loss = F.mse_loss(pred, noise)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()

#             print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

#     def predict(self, img):
#         # ==========================
#         # 1. PatchCore 
#         # ==========================
#         score_patch = self.patchcore.predict_score(img)

#         # ==========================
#         # 2. 
#         # ==========================
#         with torch.no_grad():
#             img_norm = torch.clamp(img, 0, 1)
#             rec = self.diffusion.reconstruct(img_norm)
#             score_rec = F.mse_loss(rec, img_norm, reduction='none').mean(1, keepdim=True)
#             score_rec = F.interpolate(score_rec, size=score_patch.shape[-2:], mode='bilinear')

#         # ==========================
#         # 3. 
#         # ==========================
#         score = 0.6 * score_patch + 0.4 * score_rec
#         return score, rec
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.patchcore import PatchCore
# from models.diffusion_ldm import SimpleLDM

# class FusionModel(nn.Module):
#     def __init__(self, device):
#         super().__init__()
#         self.device = device
#         self.patchcore = PatchCore(device)
#         self.diffusion = SimpleLDM(device=device)

#     def train_diffusion(self, train_loader, epochs=3):
#        
#         optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=1e-4)
#         self.diffusion.train()
#         print("Fast Training LDM...")
        
#         for epoch in range(epochs):
#             total_loss = 0.0
#             for img, _ in train_loader:
#                 img = img.to(self.device)
#                 img = torch.clamp(img, 0, 1)

#                 t = torch.randint(0, self.diffusion.num_timesteps, (img.shape[0],), device=self.device).long()
#                 pred, noise = self.diffusion(img, t)
#                 loss = F.mse_loss(pred, noise)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()

#             print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

#     def predict(self, img):
#         # ==========================
#         # 1. PatchCore 得分
#         # ==========================
#         score_patch = self.patchcore.predict_score(img)

#         # ==========================
#         # 2. 
#         # ==========================
#         with torch.no_grad():
#             img_norm = torch.clamp(img, 0, 1)
#             # 单步重建，不迭代50步
#             rec = self.diffusion.reconstruct(img_norm)
            
#             # 
#             score_rec = F.mse_loss(rec, img_norm, reduction='none').mean(1, keepdim=True)
#             score_rec = F.interpolate(score_rec, size=score_patch.shape[-2:], mode='bilinear')

#         # ==========================
#         # 标准融合
#         # ==========================
#         score = 0.5 * score_patch + 0.5 * score_rec
#         return score, rec
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.patchcore import PatchCore
from models.diffusion_ldm import SimpleLDM

class FusionModel(nn.Module):
    def __init__(self, device, backbone_name="resnet50", noise_schedule="linear", 
                 recon_strategy="diffusion", loss_type="mse"):
        super().__init__()
        self.device = device
        self.backbone_name = backbone_name
        self.noise_schedule = noise_schedule
        self.recon_strategy = recon_strategy
        self.loss_type = loss_type

        # =====================  1. Backbone PatchCore =====================
        self.patchcore = PatchCore(device, backbone_name=backbone_name)
        
        # =====================  2. Diffusion =====================
        self.diffusion = SimpleLDM(
            device=device, 
            noise_schedule=noise_schedule, 
            loss_type=loss_type
        )

    def train_diffusion(self, train_loader, epochs=3):
        if self.recon_strategy == "direct":
            print(" [Direct Strategy] Skipping diffusion training.")
            return
            
        optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=1e-4)
        self.diffusion.train()
        print(f"🚀 Training LDM (Schedule: {self.noise_schedule}, Loss: {self.loss_type})...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            for img, _ in train_loader:
                img = img.to(self.device)
                img = torch.clamp(img, 0, 1)

                t = torch.randint(0, self.diffusion.num_timesteps, (img.shape[0],), device=self.device).long()
                pred, noise = self.diffusion(img, t)
                loss = self.diffusion.get_loss(pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    def predict(self, img):
        # =====================  3. =====================
        if self.recon_strategy == "direct":
            # 
            score = self.patchcore.predict_score(img)
            rec = img
        elif self.recon_strategy == "multiscale":
            # 
            score_patch = self.patchcore.predict_score(img)
            with torch.no_grad():
                img_norm = torch.clamp(img, 0, 1)
                rec = self.diffusion.reconstruct(img_norm)
                score_rec = F.mse_loss(rec, img_norm, reduction='none').mean(1, keepdim=True)
                score_rec = F.interpolate(score_rec, size=score_patch.shape[-2:], mode='bilinear')
            score = 0.4 * score_patch + 0.6 * score_rec
        else:
            # 
            score_patch = self.patchcore.predict_score(img)
            with torch.no_grad():
                img_norm = torch.clamp(img, 0, 1)
                rec = self.diffusion.reconstruct(img_norm)
                score_rec = F.mse_loss(rec, img_norm, reduction='none').mean(1, keepdim=True)
                score_rec = F.interpolate(score_rec, size=score_patch.shape[-2:], mode='bilinear')
            score = 0.5 * score_patch + 0.5 * score_rec
        
        return score, rec