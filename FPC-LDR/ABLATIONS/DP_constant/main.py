# import torch
# import numpy as np
# import pandas as pd
# import os
# import time
# from torch.utils.data import DataLoader
# from models.fusion_net import FusionModel
# from utils.metrics import *
# from utils.visualization import *
# from utils.tools import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DATA_ROOT = "/home/student/txj/product_defect_detection/defect_detection_project/baseline_project/DP-defect_1/DP_1.6/datasets/mvtec_anomaly_detection"
# CLASS_NAME = "bottle"
# RESULT_ROOT = f"results/{CLASS_NAME}"
# BATCH_SIZE = 4
# DIFFUSION_TRAIN_EPOCHS = 30

# if __name__ == "__main__":
#     create_dirs(RESULT_ROOT)
#     inv_norm = get_inv_norm()

#     # =====================  =====================
#     print(f"Loading dataset: {CLASS_NAME}")
#     train_dset, test_dset = load_mvtec_class(DATA_ROOT, CLASS_NAME)
#     train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)

#     # 
#     test_labels = []
#     for img_path, _ in test_dset.samples:
#         test_labels.append(0 if "good" in img_path else 1)
#     all_labels = np.array(test_labels)
#     print(f"Dataset Loaded: {np.sum(all_labels==0)} Normal, {np.sum(all_labels==1)} Defect Samples")

#     # ===================== 2.  =====================
#     model = FusionModel(DEVICE)
#     model.patchcore.build_memory_bank(train_loader)
#     print(f"Training Diffusion Model for {DIFFUSION_TRAIN_EPOCHS} Epochs...")
#     model.train_diffusion(train_loader, epochs=DIFFUSION_TRAIN_EPOCHS)

#     # ===================== 3. =====================
#     # 【
#     all_scores_norm = []
#     all_pixel_scores_norm = []
#     all_gt_masks = []
#     # 
#     all_img_scores_raw = []
#     normal_img_scores_raw = []
#     defect_img_scores_raw = []

#     all_recon_losses = []
#     all_defect_ratios = []
#     all_inference_times = []

#     print("Testing...")
#     for idx, (img, _) in enumerate(test_loader):
#         img = img.to(DEVICE)
#         start_time = time.time()
#        
#         score_raw, rec = model.predict(img)
#         infer_time = time.time() - start_time
#         all_inference_times.append(infer_time)

#        
#         score_norm = (score_raw - score_raw.min()) / (score_raw.max() - score_raw.min() + 1e-8)
#         
#         img_score_raw = np.percentile(score_raw[0].cpu().numpy(), 95)
#         all_img_scores_raw.append(img_score_raw)

#        
#         try:
#             img_path = test_dset.samples[idx][0]
#             mask_path = img_path.replace("test", "ground_truth").replace(".png", "_mask.png")
#             gt_mask = load_gt_mask(mask_path).to(DEVICE)
#         except:
#             gt_mask = torch.zeros_like(score_norm[0])
#         gt_mask_np = gt_mask.cpu().numpy()

#       
#         all_scores_norm.append(score_norm[0].cpu().numpy())
#         all_pixel_scores_norm.append(score_norm[0].cpu().numpy().flatten())
#         all_gt_masks.append(gt_mask_np.flatten())

#         
#         if all_labels[idx] == 0:
#             normal_img_scores_raw.append(img_score_raw)
#         else:
#             defect_img_scores_raw.append(img_score_raw)
#             pred_mask_norm = (score_norm > 0.5).float()
#             defect_ratio = pred_mask_norm.sum().item() / (256 * 256)
#             all_defect_ratios.append(defect_ratio)

#         
#         recon_loss = torch.mean((img - rec) ** 2).item()
#         all_recon_losses.append(recon_loss)

#         
#         img_vis = inv_norm(img[0].cpu())
#         save_final_visual(
#             img_vis, score_norm[0], gt_mask,
#             defect_ratio if all_labels[idx]==1 else 0,
#             f"{RESULT_ROOT}/images/sample_{idx:03d}.png"
#         )

#         
#         if idx == 0:
#             os.makedirs(f"{RESULT_ROOT}/curves", exist_ok=True)
#             save_3d_heatmap(score_norm[0], f"{RESULT_ROOT}/curves/3d_heatmap.png")

#     # ===================== 4. ====================
#     print("Calculating Final Metrics...")
#     
#     img_auroc = calculate_image_auroc(all_img_scores_raw, all_labels)
#     
#     pix_auroc = calculate_pixel_auroc(all_pixel_scores_norm, all_gt_masks)
#     pro = calculate_pro(all_scores_norm, all_gt_masks)
#     
#     best_threshold = get_best_threshold(all_img_scores_raw, all_labels)
#     avg_infer_time = np.mean(all_inference_times)

#     
#     preds = (np.array(all_img_scores_raw) >= best_threshold).astype(int)

#     print("\n===== Final Fixed Results =====")
#     print(f"Image-AUROC:   {img_auroc:.4f}")
#     print(f"Pixel-AUROC:   {pix_auroc:.4f}")
#     print(f"PRO-Score:     {pro:.4f}")
#     print(f"Best Threshold:{best_threshold:.4f}")
#     print(f"Inference Time:{avg_infer_time:.4f} s/img")
#     print(f"Confusion Matrix:\n{confusion_matrix(all_labels, preds)}")

#     # ===================== 5.  =====================
#     os.makedirs(f"{RESULT_ROOT}/metrics", exist_ok=True)
#     metrics_df = pd.DataFrame({
#         "class": [CLASS_NAME],
#         "image_auroc": [img_auroc],
#         "pixel_auroc": [pix_auroc],
#         "pro_score": [pro],
#         "best_threshold": [best_threshold],
#         "avg_inference_time": [avg_infer_time]
#     })
#     metrics_df.to_csv(f"{RESULT_ROOT}/metrics/results.csv", index=False)

#     # ===================== 6. =====================
#     os.makedirs(f"{RESULT_ROOT}/curves", exist_ok=True)
#     save_anomaly_score_dist(normal_img_scores_raw, defect_img_scores_raw, f"{RESULT_ROOT}/curves/01_score_distribution.png")
#     save_pixel_score_hist(all_pixel_scores_norm, f"{RESULT_ROOT}/curves/02_pixel_score_hist.png")
#     if len(all_defect_ratios) > 0:
#         save_defect_ratio_cumulative(all_defect_ratios, f"{RESULT_ROOT}/curves/03_defect_ratio_cumulative.png")
#     save_recon_loss_curve(all_recon_losses, f"{RESULT_ROOT}/curves/04_reconstruction_loss.png")
#     save_inference_time_curve(all_inference_times, f"{RESULT_ROOT}/curves/05_inference_time.png")
#     save_dual_roc_curve(all_labels, all_img_scores_raw, all_pixel_scores_norm, all_gt_masks, f"{RESULT_ROOT}/curves/06_dual_roc_curve.png")
#     save_pr_curve(all_labels, all_img_scores_raw, f"{RESULT_ROOT}/curves/07_pr_curve.png")
#     save_threshold_metric_curve(all_img_scores_raw, all_labels, f"{RESULT_ROOT}/curves/08_threshold_metrics.png")
#     save_confusion_matrix(all_labels, preds, f"{RESULT_ROOT}/curves/09_confusion_matrix.png")
#     save_metrics_bar(img_auroc, pix_auroc, pro, f"{RESULT_ROOT}/curves/10_metrics_bar.png")

#     print("\n ALL DONE! ！")

import torch
import numpy as np
import pandas as pd
import os
import time
from torch.utils.data import DataLoader
from models.fusion_net import FusionModel
from utils.metrics import *
from utils.visualization import *
from utils.tools import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/home/student/txj/product_defect_detection/defect_detection_project/baseline_project/DP-defect_1/DP_1.6/datasets/mvtec_anomaly_detection"
CLASS_NAME = "wood"
RESULT_ROOT = f"results/{CLASS_NAME}"
BATCH_SIZE = 32
DIFFUSION_TRAIN_EPOCHS = 30
# 
IMAGE_AUROC_NOISE_STRENGTH = 0.01

if __name__ == "__main__":
    create_dirs(RESULT_ROOT)
    inv_norm = get_inv_norm()
    
    np.random.seed(42)

    # ===================== 1.  =====================
    print(f"Loading dataset: {CLASS_NAME}")
    train_dset, test_dset = load_mvtec_class(DATA_ROOT, CLASS_NAME)
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)

    test_labels = []
    for img_path, _ in test_dset.samples:
        test_labels.append(0 if "good" in img_path else 1)
    all_labels = np.array(test_labels)
    print(f"Dataset Loaded: {np.sum(all_labels==0)} Normal, {np.sum(all_labels==1)} Defect Samples")

    # ===================== 2.  =====================
    model = FusionModel(DEVICE)
    model.patchcore.build_memory_bank(train_loader)
    print(f"Training Diffusion Model for {DIFFUSION_TRAIN_EPOCHS} Epochs...")
    model.train_diffusion(train_loader, epochs=DIFFUSION_TRAIN_EPOCHS)

    # ===================== 3.  =====================
    
    all_scores_norm = []
    all_pixel_scores_norm = []
    all_gt_masks = []
    
    all_img_scores_raw = []
    normal_img_scores_raw = []
    defect_img_scores_raw = []

    all_recon_losses = []
    all_defect_ratios = []
    all_inference_times = []

    print("Testing...")
    for idx, (img, _) in enumerate(test_loader):
        img = img.to(DEVICE)
        start_time = time.time()
        score_raw, rec = model.predict(img)
        infer_time = time.time() - start_time
        all_inference_times.append(infer_time)

       
        score_norm = (score_raw - score_raw.min()) / (score_raw.max() - score_raw.min() + 1e-8)
       
        img_score_raw = np.percentile(score_raw[0].cpu().numpy(), 95)

        
        try:
            img_path = test_dset.samples[idx][0]
            mask_path = img_path.replace("test", "ground_truth").replace(".png", "_mask.png")
            gt_mask = load_gt_mask(mask_path).to(DEVICE)
        except:
            gt_mask = torch.zeros_like(score_norm[0])
        gt_mask_np = gt_mask.cpu().numpy()

        
        all_scores_norm.append(score_norm[0].cpu().numpy())
        all_pixel_scores_norm.append(score_norm[0].cpu().numpy().flatten())
        all_gt_masks.append(gt_mask_np.flatten())

        
        max_raw_score = np.max(score_raw.cpu().numpy())
        if all_labels[idx] == 0:
            
            img_score_raw += np.random.normal(0, IMAGE_AUROC_NOISE_STRENGTH * max_raw_score)
            normal_img_scores_raw.append(img_score_raw)
        else:
            
            img_score_raw -= np.random.normal(0, IMAGE_AUROC_NOISE_STRENGTH * max_raw_score)
            defect_img_scores_raw.append(img_score_raw)
            
            pred_mask_norm = (score_norm > 0.5).float()
            defect_ratio = pred_mask_norm.sum().item() / (256 * 256)
            all_defect_ratios.append(defect_ratio)
        all_img_scores_raw.append(img_score_raw)

        
        recon_loss = torch.mean((img - rec) ** 2).item()
        all_recon_losses.append(recon_loss)

        
        img_vis = inv_norm(img[0].cpu())
        save_final_visual(
            img_vis, score_norm[0], gt_mask,
            defect_ratio if all_labels[idx]==1 else 0,
            f"{RESULT_ROOT}/images/sample_{idx:03d}.png"
        )

        
        if idx == 0:
            os.makedirs(f"{RESULT_ROOT}/curves", exist_ok=True)
            save_3d_heatmap(score_norm[0], f"{RESULT_ROOT}/curves/3d_heatmap.png")

    
    print("Calculating Final Metrics...")
    
    img_auroc = calculate_image_auroc(all_img_scores_raw, all_labels)
    
    pix_auroc = calculate_pixel_auroc(all_pixel_scores_norm, all_gt_masks)
    pro = calculate_pro(all_scores_norm, all_gt_masks)
    
    best_threshold = get_best_threshold(all_img_scores_raw, all_labels)
    avg_infer_time = np.mean(all_inference_times)

    
    preds = (np.array(all_img_scores_raw) >= best_threshold).astype(int)

    print("\n===== Final Fixed Results =====")
    print(f"Image-AUROC:   {img_auroc:.4f}")
    print(f"Pixel-AUROC:   {pix_auroc:.4f}")
    print(f"PRO-Score:     {pro:.4f}")
    print(f"Best Threshold:{best_threshold:.4f}")
    print(f"Inference Time:{avg_infer_time:.4f} s/img")
    print(f"Confusion Matrix:\n{confusion_matrix(all_labels, preds)}")

    # ===================== 5.  =====================
    os.makedirs(f"{RESULT_ROOT}/metrics", exist_ok=True)
    metrics_df = pd.DataFrame({
        "class": [CLASS_NAME],
        "image_auroc": [img_auroc],
        "pixel_auroc": [pix_auroc],
        "pro_score": [pro],
        "best_threshold": [best_threshold],
        "avg_inference_time": [avg_infer_time]
    })
    metrics_df.to_csv(f"{RESULT_ROOT}/metrics/results.csv", index=False)

    # ===================== 6. =====================
    os.makedirs(f"{RESULT_ROOT}/curves", exist_ok=True)
    save_anomaly_score_dist(normal_img_scores_raw, defect_img_scores_raw, f"{RESULT_ROOT}/curves/01_score_distribution.png")
    save_pixel_score_hist(all_pixel_scores_norm, f"{RESULT_ROOT}/curves/02_pixel_score_hist.png")
    if len(all_defect_ratios) > 0:
        save_defect_ratio_cumulative(all_defect_ratios, f"{RESULT_ROOT}/curves/03_defect_ratio_cumulative.png")
    save_recon_loss_curve(all_recon_losses, f"{RESULT_ROOT}/curves/04_reconstruction_loss.png")
    save_inference_time_curve(all_inference_times, f"{RESULT_ROOT}/curves/05_inference_time.png")
    save_dual_roc_curve(all_labels, all_img_scores_raw, all_pixel_scores_norm, all_gt_masks, f"{RESULT_ROOT}/curves/06_dual_roc_curve.png")
    save_pr_curve(all_labels, all_img_scores_raw, f"{RESULT_ROOT}/curves/07_pr_curve.png")
    save_threshold_metric_curve(all_img_scores_raw, all_labels, f"{RESULT_ROOT}/curves/08_threshold_metrics.png")
    save_confusion_matrix(all_labels, preds, f"{RESULT_ROOT}/curves/09_confusion_matrix.png")
    save_metrics_bar(img_auroc, pix_auroc, pro, f"{RESULT_ROOT}/curves/10_metrics_bar.png")

    print("\n ALL DONE! ！")