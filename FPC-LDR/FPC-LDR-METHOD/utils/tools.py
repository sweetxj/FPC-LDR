import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_mask_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

def get_inv_norm():
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return lambda x: x * std + mean

def create_dirs(root):
    dirs = ['images', 'curves', 'metrics']
    for d in dirs:
        os.makedirs(f"{root}/{d}", exist_ok=True)

def load_mvtec_class(root, cls):
    train_dset = ImageFolder(f"{root}/{cls}/train", transform=get_transform())
    test_dset = ImageFolder(f"{root}/{cls}/test", transform=get_transform())
    return train_dset, test_dset

def load_gt_mask(path):
    try:
        mask = Image.open(path).convert('L')
        mask = get_mask_transform()(mask)
        return mask.squeeze(0)
    except:
        return torch.zeros((256, 256))