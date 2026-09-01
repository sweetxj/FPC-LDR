import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.config import *

class MVTecDataset(Dataset):
    def __init__(self, class_name, split="train"):
        self.root = os.path.join(DATA_ROOT, class_name)
        self.split = split
        self.image_paths = []
        self.mask_paths = []
        self.labels = []

        if split == "train":
            folder = os.path.join(self.root, "train/good")
            for name in os.listdir(folder):
                self.image_paths.append(os.path.join(folder, name))
                self.labels.append(0)
        else:
            test_dir = os.path.join(self.root, "test")
            gt_dir = os.path.join(self.root, "ground_truth")
            for defect_type in os.listdir(test_dir):
                img_dir = os.path.join(test_dir, defect_type)
                for fname in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, fname)
                    mask_path = img_path.replace("test", "ground_truth").replace(".png", "_mask.png")
                    if not os.path.exists(mask_path):
                        mask_path = None
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(0 if defect_type == "good" else 1)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        if self.split == "test":
            mask = Image.open(self.mask_paths[idx]).convert("L") if self.mask_paths[idx] else Image.new("L", (IMAGE_SIZE, IMAGE_SIZE))
            mask = self.transform(mask)
            return img, mask, self.labels[idx], self.image_paths[idx]
        return img