import torch
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ["transistor"]
DATA_ROOT = "/home/student/txj/product_defect_detection/defect_detection_project/baseline_project/patchcore_mvtec/datasets/mvtec_anomaly_detection"
IMAGE_SIZE = 256
BATCH_SIZE = 8
FEATURE_LAYERS = [2, 3]  # PatchCore 用 layer2 + 3
MEMORY_BANK_SIZE = 100000