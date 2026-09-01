import torch
import numpy as np
import pandas as pd
import os
import time
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from models.fusion_net import FusionModel
from utils.metrics import *
from utils.visualization import save_final_visual
from utils.tools import create_dirs, get_inv_norm, load_mvtec_class, load_gt_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/home/student/txj/product_defect_detection/defect_detection_project/ablation_project/DP_resnet50/datasets/mvtec_anomaly_detection"
CLASS_NAME = "wood"
BATCH_SIZE = 4
EPOCHS = 3



ABLATION_CONFIG = {
    "backbone": "resnet50",#  resnet18, resnet34, resnet50, mobilenetv2, convnext_tiny
    "noise_schedule": "constant",# 可选: linear, cosine, quadratic, constant
    "recon_strategy": "diffusion",# 可选: direct, diffusion, multiscale
    "loss_type": "mse"#  mse, l1, perceptual, mse_perceptual
}

EXP_NAME = f"{ABLATION_CONFIG['backbone']}_{ABLATION_CONFIG['noise_schedule']}_{ABLATION_CONFIG['recon_strategy']}_{ABLATION_CONFIG['loss_type']}"
RESULT_ROOT = f"ablation_results/{CLASS_NAME}/{EXP_NAME}"

if __name__ == "__main__":
    create_dirs(RESULT_ROOT)
    create_dirs(f"{RESULT_ROOT}/images")
    inv_norm = get_inv_norm()

    train_dset, test_dset = load_mvtec_class(DATA_ROOT, CLASS_NAME)
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)

    all_labels = []
    for path, _ in test_dset.samples:
        all_labels.append(0 if "good" in path else 1)
    all_labels = np.array(all_labels)

    # =====================  =====================
    model = FusionModel(
        DEVICE,
        backbone_name=ABLATION_CONFIG["backbone"],
        noise_schedule=ABLATION_CONFIG["noise_schedule"],
        recon_strategy=ABLATION_CONFIG["recon_strategy"],
        loss_type=ABLATION_CONFIG["loss_type"]
    )
    
    model.patchcore.build_memory_bank(train_loader)
    model.train_diffusion(train_loader, epochs=EPOCHS)

    all_img_scores = []
    all_pixel_scores = []
    all_gt_masks = []
    all_times = []

    for idx, (img, _) in enumerate(test_loader):
        img = img.to(DEVICE)
        t0 = time.time()

        score_map, rec = model.predict(img)
        all_times.append(time.time() - t0)

        # 🔥 
        score_np = score_map.cpu().numpy().squeeze()
        img_score = np.percentile(score_np, 85)  
        all_img_scores.append(img_score)

        score_norm = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
        try:
            mask_path = test_dset.samples[idx][0].replace("test", "ground_truth").replace(".png", "_mask.png")
            gt_mask = load_gt_mask(mask_path).to(DEVICE)
        except:
            gt_mask = torch.zeros_like(score_map)

        all_pixel_scores.append(score_norm.cpu().numpy().flatten())
        all_gt_masks.append(gt_mask.cpu().numpy().flatten())

        # 可视化
        img_vis = inv_norm(img[0].cpu())
        score_vis = score_norm[0].squeeze()
        mask_vis = gt_mask[0].squeeze()
        save_final_visual(img_vis, score_vis, mask_vis, 0, f"{RESULT_ROOT}/images/{idx:03d}.png")

    # 指标计算
    img_auroc = calculate_image_auroc(all_img_scores, all_labels)
    pix_auroc = calculate_pixel_auroc(all_pixel_scores, all_gt_masks)
    best_th, pro = get_best_pro_threshold(all_pixel_scores, all_gt_masks)
    f1_th = get_best_threshold(all_img_scores, all_labels)
    avg_time = np.mean(all_times) * 1000
    preds = (np.array(all_img_scores) >= f1_th).astype(int)
    cm = confusion_matrix(all_labels, preds)

    print("\n=====  Final Normal Results =====")
    print(f"Class:         {CLASS_NAME}")
    print(f"Backbone:      {ABLATION_CONFIG['backbone']}")
    print(f"Noise Sched:   {ABLATION_CONFIG['noise_schedule']}")
    print(f"Recon Strat:   {ABLATION_CONFIG['recon_strategy']}")
    print(f"Loss Type:     {ABLATION_CONFIG['loss_type']}")
    print("-" * 40)
    print(f"Image-AUROC:   {img_auroc:.4f}")
    print(f"Pixel-AUROC:   {pix_auroc:.4f}")
    print(f"PRO-Score:     {pro:.4f}")
    print(f"Infer(ms):     {avg_time:.2f}")
    print(f"Confusion Matrix:\n{cm}")

    # CSV
    df = pd.DataFrame([{
        "backbone": ABLATION_CONFIG["backbone"],
        "noise_schedule": ABLATION_CONFIG["noise_schedule"],
        "recon_strategy": ABLATION_CONFIG["recon_strategy"],
        "loss_type": ABLATION_CONFIG["loss_type"],
        "image_auroc": round(img_auroc*100, 2),
        "pixel_auroc": round(pix_auroc*100, 2),
        "pro_score": round(pro*100, 2),
        "threshold": round(f1_th, 4),
        "inference_time_ms": round(avg_time, 2)
    }])
    
    csv_path = f"ablation_results/{CLASS_NAME}_ablation_table.csv"
    if os.path.exists(csv_path):
        df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    df.to_csv(csv_path, index=False)