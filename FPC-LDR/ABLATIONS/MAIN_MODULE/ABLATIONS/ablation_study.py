import torch
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.fusion_net import FusionModel
from utils.metrics import *
from utils.visualization import save_final_visual
from utils.tools import *

# ===================== config =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/home/tangxiaojun/product_defect_detection/defect_detection_project/baseline_project/DP-defect_1/DP_1.6/datasets/mvtec_anomaly_detection"
CLASS_NAME = "zipper"
RESULT_ROOT = f"results/ablation_{CLASS_NAME}"
BATCH_SIZE = 32
DIFFUSION_EPOCHS = 10

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 5组消融配置
ABLATION_CONFIGS = [
    # {"id": "A", "mode": "memory_only",    "name": "Only Memory Bank"},
    # {"id": "B", "mode": "diffusion_only", "name": "Only Diffusion Reconstruction"},
    # {"id": "C", "mode": "fixed_fusion",   "name": "Fixed Weight Fusion"},
    # {"id": "D", "mode": "adaptive_fusion","name": "Adaptive Weight Fusion"}
    {"id": "E", "mode": "full",           "name": "Full FPC-LDR (Ours)"},
]


def run_single_ablation(cfg):
    cfg_id = cfg["id"]
    cfg_mode = cfg["mode"]
    cfg_name = cfg["name"]
    exp_dir = f"{RESULT_ROOT}/{cfg_id}"
    os.makedirs(f"{exp_dir}/heatmaps", exist_ok=True)
    inv_norm = get_inv_norm()

    print(f"\n{'='*70}")
    print(f"[{cfg_id}] 开始实验: {cfg_name}")
    print(f"{'='*70}")

    # 1. 加载数据集
    train_dset, test_dset = load_mvtec_class(DATA_ROOT, CLASS_NAME)
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)

    # 修正二分类标签
    test_labels = []
    for img_path, _ in test_dset.samples:
        test_labels.append(0 if "good" in img_path else 1)
    all_labels = np.array(test_labels)
    print(f"数据集加载完成: {np.sum(all_labels==0)} 正常, {np.sum(all_labels==1)} 缺陷")

    # 2. 模型初始化与训练
    model = FusionModel(DEVICE, mode=cfg_mode)
    if model.patchcore is not None:
        model.patchcore.build_memory_bank(train_loader)
    if model.diffusion is not None:
        model.train_diffusion(train_loader, epochs=DIFFUSION_EPOCHS)

    # 3. GPU/KNN 预热（前3张不计入平均时间）
    print("预热模型...")
    dummy_img = next(iter(test_loader))[0].to(DEVICE)
    for _ in range(3):
        _, _, _, _ = model.predict(dummy_img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 4. 正式推理测试
    all_pixel_scores_norm = []  # 归一化得分：像素级指标+可视化
    all_gt_flat = []
    all_img_scores_raw = []     # 原始得分：图像级指标
    infer_times = []

    print("开始推理测试...")
    for idx, (img, _) in enumerate(tqdm(test_loader, desc="Inference Testing", leave=False)):
        img = img.to(DEVICE)

        # 纯推理计时
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()
        score_raw, score_norm, recon, img_score = model.predict(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_times.append((time.time() - t_start) * 1000)  # 毫秒

        # 加载真值掩码
        try:
            img_path = test_dset.samples[idx][0]
            mask_path = img_path.replace("test", "ground_truth").replace(".png", "_mask.png")
            gt_mask = load_gt_mask(mask_path).to(DEVICE)
        except:
            gt_mask = torch.zeros_like(score_norm[0])
        gt_np = gt_mask.cpu().numpy()

        # 保存指标数据
        score_np = score_norm[0].cpu().numpy()
        all_pixel_scores_norm.append(score_np)
        all_gt_flat.append(gt_np.flatten())
        all_img_scores_raw.append(img_score)

        # 仅保存前15张缺陷样本热力图
        if all_labels[idx] == 1 and idx < 15:
            img_vis = inv_norm(img[0].cpu())
            defect_ratio = np.sum(score_np > 0.5) / (256 * 256)
            save_final_visual(
                img_vis, score_norm[0], gt_mask,
                defect_ratio,
                f"{exp_dir}/heatmaps/sample_{idx:03d}.png"
            )

    # 5. 计算指标
    print("计算评价指标...")
    # 图像级：用原始得分
    img_auroc = calculate_image_auroc(all_img_scores_raw, all_labels)
    # 像素级：用归一化得分
    pix_flat = [s.flatten() for s in all_pixel_scores_norm]
    pix_auroc = calculate_pixel_auroc(pix_flat, all_gt_flat)
    pro_score = calculate_pro(all_pixel_scores_norm, all_gt_flat)
    # 阈值基于图像级原始得分计算
    best_th = get_best_threshold(all_img_scores_raw, all_labels)
    # 推理时间：去掉前3个预热样本取平均
    avg_time = np.mean(infer_times[3:])

    result = {
        "Configuration": cfg_id,
        "Method": cfg_name,
        "Image-AUROC(%)": round(img_auroc * 100, 2),
        "Pixel-AUROC(%)": round(pix_auroc * 100, 2),
        "PRO(%)": round(pro_score * 100, 2),
        "Threshold": round(best_th, 4),
        "Inference Time(ms)": round(avg_time, 2)
    }

    print(f"\n[{cfg_id}] 实验结果:")
    print(f"  Image-AUROC: {result['Image-AUROC(%)']:.2f}%")
    print(f"  Pixel-AUROC: {result['Pixel-AUROC(%)']:.2f}%")
    print(f"  PRO-Score:   {result['PRO(%)']:.2f}%")
    print(f"  平均推理时间: {result['Inference Time(ms)']:.2f} ms")

    pd.DataFrame([result]).to_csv(f"{exp_dir}/metrics.csv", index=False)
    return result


if __name__ == "__main__":
    os.makedirs(RESULT_ROOT, exist_ok=True)
    all_results = []

    for cfg in ABLATION_CONFIGS:
        res = run_single_ablation(cfg)
        all_results.append(res)

    # 生成汇总表
    summary_df = pd.DataFrame(all_results)
    summary_path = f"{RESULT_ROOT}/ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print(f"消融实验最终汇总 (MVTec AD - {CLASS_NAME})")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)
    print(f"汇总表保存路径: {summary_path}")
