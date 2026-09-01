# import os
# import torch
# import time
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader
# from models.backbone import ResNet50Backbone
# from models.fusion import Fusion
# from utils.dataset import MVTecDataset
# from utils.metrics import image_auroc, pixel_auroc, pro_score
# from utils.visualization import save_result
# from utils.config import *
# from tqdm import tqdm  # 进度条

# os.makedirs("results", exist_ok=True)

# for cls in CLASS_NAMES:
#     save_dir = f"results/{cls}"
#     os.makedirs(f"{save_dir}/images", exist_ok=True)
#     os.makedirs(f"{save_dir}/curves", exist_ok=True)
#     os.makedirs(f"{save_dir}/matrix", exist_ok=True)

#     backbone = ResNet50Backbone().to(DEVICE).eval()
#     # 修复 torch.load 警告
#     patchcore = torch.load(f"results/{cls}/patchcore.pth", weights_only=False)
#     fusion = Fusion().to(DEVICE)

#     dataset = MVTecDataset(cls, "test")
#     loader = DataLoader(dataset, batch_size=1, shuffle=False)

#     scores, labels, masks_list, amaps_list = [], [], [], []
#     total_time = 0

#     with torch.no_grad():
#         # =====================  测试进度条  =====================
#         pbar = tqdm(loader, desc=f"Testing {cls}")
#         for idx, (img, mask, label, path) in enumerate(pbar):
#             img = img.to(DEVICE)
#             t0 = time.time()
#             feats = backbone(img)
            
#             # 【修复核心错误】把 score → compute_score
#             amap = patchcore.compute_score(feats)
            
#             amap = fusion(amap)
#             total_time += time.time() - t0

#             score = amap.max().item()
#             scores.append(score)
#             labels.append(label.item())
#             mask_np = mask.cpu().numpy()
#             masks_list.append(mask_np)
#             amap_np = amap.cpu().numpy()
#             amaps_list.append(amap_np)

#             ratio = mask_np.sum() / (IMAGE_SIZE*IMAGE_SIZE)
#             is_anomaly = label.item() == 1
#             save_result(img[0], mask[0], amap[0], ratio, is_anomaly,
#                         f"{save_dir}/images/{idx:03d}.png")

#     img_auc = image_auroc(labels, scores)
#     pix_auc = pixel_auroc(masks_list, amaps_list)
#     pro = pro_score(masks_list, amaps_list)
#     infer_time = total_time / len(loader)

#     print(f"\n {cls} done!")
#     print(f"Image-AUROC: {img_auc:.4f}")
#     print(f"Pixel-AUROC: {pix_auc:.4f}")
#     print(f"PRO: {pro:.4f}")
#     print(f"Infer: {infer_time:.4f}s")

#     df = pd.DataFrame([{
#         "class": cls, "img_auc": img_auc, "pix_auc": pix_auc,
#         "pro": pro, "infer_time": infer_time
#     }])
#     df.to_csv(f"{save_dir}/metrics.csv", index=False)
import os
import torch
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models.backbone import ResNet50Backbone
from models.fusion import Fusion
from utils.dataset import MVTecDataset
from utils.metrics import image_auroc, pixel_auroc, pro_score, find_best_threshold
from utils.visualization import *
from utils.config import *
from tqdm import tqdm

os.makedirs("results", exist_ok=True)

for cls in CLASS_NAMES:
    save_dir = f"results/{cls}"
    os.makedirs(f"{save_dir}/images", exist_ok=True)
    os.makedirs(f"{save_dir}/curves", exist_ok=True)
    os.makedirs(f"{save_dir}/matrix", exist_ok=True)

    backbone = ResNet50Backbone().to(DEVICE).eval()
    patchcore = torch.load(f"results/{cls}/patchcore.pth", weights_only=False)
    fusion = Fusion().to(DEVICE)

    dataset = MVTecDataset(cls, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    scores, labels, masks_list, amaps_list = [], [], [], []
    ratios = []
    total_time = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Testing {cls}")
        for idx, (img, mask, label, path) in enumerate(pbar):
            img = img.to(DEVICE)
            t0 = time.time()
            feats = backbone(img)
            amap = patchcore.compute_score(feats)
            amap = fusion(amap)
            total_time += time.time() - t0

            score = amap.max().item()
            scores.append(score)
            labels.append(label.item())
            masks_list.append(mask.cpu().numpy())
            amaps_list.append(amap.cpu().numpy())
            ratio = mask.sum().item() / (IMAGE_SIZE*IMAGE_SIZE)
            ratios.append(ratio)
            is_anomaly = label.item() == 1
            save_result(img[0], mask[0], amap[0], ratio, is_anomaly,
                        f"{save_dir}/images/{idx:03d}.png")

    # ================== ==================
    img_auc = image_auroc(labels, scores)
    pix_auc = pixel_auroc(masks_list, amaps_list)
    pro = pro_score(masks_list, amaps_list)
    infer_time = total_time / len(loader)
    threshold, tpr, tnr = find_best_threshold(labels, scores)
    preds = (np.array(scores) > threshold).astype(int)

    # ==================  ==================
    plot_roc(labels, scores, f"{save_dir}/curves/roc.png")
    plot_confusion(labels, preds, f"{save_dir}/matrix/confusion.png")
    plot_score_dist(scores, labels, f"{save_dir}/curves/score_dist.png")
    plot_defect_ratio(ratios, f"{save_dir}/curves/defect_ratio.png")
    plot_similarity(patchcore.memory_bank['layer_0'], f"{save_dir}/matrix/similarity.png")

    # ================== ==================
    print("\n RESULTS:")
    print(f"Image-AUROC: {img_auc:.4f}")
    print(f"Pixel-AUROC: {pix_auc:.4f}")
    print(f"PRO: {pro:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Infer time: {infer_time:.4f}s")

    df = pd.DataFrame([{
        "class": cls, "img_auc": img_auc, "pix_auc": pix_auc,
        "pro": pro, "threshold": threshold, "infer_time": infer_time
    }])
    df.to_csv(f"{save_dir}/metrics.csv", index=False)