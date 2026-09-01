# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D
# # 补全所有需要的sklearn函数导入，彻底解决未定义问题
# from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
# from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# # 全局IEEE论文风格设置
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.spines.left'] = True
# plt.rcParams['axes.spines.right'] = True
# plt.rcParams['axes.spines.top'] = True
# plt.rcParams['axes.spines.bottom'] = True
# plt.rcParams['axes.linewidth'] = 0.5
# plt.rcParams['axes.titlesize'] = 10
# plt.rcParams['axes.labelsize'] = 9
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams['legend.fontsize'] = 8
# plt.rcParams['grid.alpha'] = 0.3
# plt.rcParams['grid.linestyle'] = '--'

# def save_final_visual(img, heat, gt_mask, ratio, path):
#     img = img.permute(1,2,0).cpu().numpy()
#     img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#     heat = heat.cpu().numpy()
#     is_good = torch.sum(gt_mask) < 1e-6

#     if is_good:
#         fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
#         axes[0].imshow(img)
#         axes[0].set_title("Normal Sample", fontsize=9)
#         axes[0].axis('off')
#         axes[1].imshow(heat, cmap='coolwarm', vmin=0, vmax=0.4)
#         axes[1].set_title("Anomaly Heatmap", fontsize=9)
#         axes[1].axis('off')
#     else:
#         pred_mask = (heat > 0.5).astype(float)
#         fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
#         axes[0].imshow(img)
#         axes[0].set_title(f"Defect Sample\nRatio: {ratio:.2%}", fontsize=9)
#         axes[0].axis('off')
#         axes[1].imshow(img)
#         axes[1].imshow(heat, cmap='jet', alpha=0.6)
#         axes[1].set_title("Anomaly Heatmap", fontsize=9)
#         axes[1].axis('off')
#         axes[2].imshow(img)
#         axes[2].imshow(pred_mask, cmap='hot', alpha=0.7)
#         axes[2].set_title("Defect Mask", fontsize=9)
#         axes[2].axis('off')
#     plt.tight_layout(pad=0.2)
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_3d_heatmap(score_map, path):
#     score_map = score_map.cpu()
#     x = np.linspace(0, 255, 64)
#     y = np.linspace(0, 255, 64)
#     X, Y = np.meshgrid(x, y)
#     Z = torch.nn.functional.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=64).squeeze().numpy()
#     fig = plt.figure(figsize=(5, 4))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.9)
#     ax.set_title("3D Anomaly Heatmap", pad=10)
#     ax.set_xlabel("Image Width", labelpad=5)
#     ax.set_ylabel("Image Height", labelpad=5)
#     ax.set_zlabel("Anomaly Score", labelpad=5)
#     ax.set_zlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_anomaly_score_dist(normal, defect, path):
#     plt.figure(figsize=(5, 3))
#     sns.kdeplot(normal, label="Normal", fill=True, color="#1f77b4", alpha=0.6, linewidth=1)
#     sns.kdeplot(defect, label="Defective", fill=True, color="#ff7f0e", alpha=0.6, linewidth=1)
#     plt.title("Anomaly Score Distribution")
#     plt.xlabel("Image-level Anomaly Score")
#     plt.ylabel("Probability Density")
#     plt.legend(frameon=False)
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_pixel_score_hist(pixel_scores, path):
#     all_pix = np.concatenate(pixel_scores)
#     plt.figure(figsize=(5, 3))
#     plt.hist(all_pix, bins=50, color="#9467bd", alpha=0.7, edgecolor='white', linewidth=0.3)
#     plt.title("Pixel-level Anomaly Score Histogram")
#     plt.xlabel("Pixel Anomaly Score")
#     plt.ylabel("Pixel Count")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_defect_ratio_cumulative(ratios, path):
#     plt.figure(figsize=(5, 3))
#     sorted_ratio = np.sort(ratios)
#     cum_counts = np.arange(1, len(sorted_ratio)+1)
#     plt.plot(sorted_ratio, cum_counts, linewidth=2, color="#ff7f0e")
#     plt.title("Defect Area Ratio Cumulative Distribution")
#     plt.xlabel("Defect Area Ratio")
#     plt.ylabel("Cumulative Sample Count")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_recon_loss_curve(losses, path):
#     plt.figure(figsize=(5, 3))
#     plt.plot(losses, color="#2ca02c", linewidth=1)
#     avg_loss = np.mean(losses)
#     plt.axhline(avg_loss, color='gray', linestyle='--', linewidth=1, label=f"Avg: {avg_loss:.3f}")
#     plt.title("Diffusion Reconstruction Loss (Test)")
#     plt.xlabel("Sample Index")
#     plt.ylabel("MSE Loss")
#     plt.legend(frameon=False)
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_inference_time_curve(times, path):
#     plt.figure(figsize=(5, 3))
#     plt.plot(times, color="#d62728", linewidth=1)
#     avg_time = np.mean(times)
#     plt.axhline(avg_time, color='gray', linestyle='--', linewidth=1, label=f"Avg: {avg_time:.3f}s")
#     plt.title("Inference Time per Sample")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Time (s)")
#     plt.legend(frameon=False)
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_dual_roc_curve(img_labels, img_scores, pixel_scores, pixel_masks, path):
#     # 图像级ROC
#     img_labels_bin = (np.array(img_labels) != 0).astype(int)
#     fpr_img, tpr_img, _ = roc_curve(img_labels_bin, img_scores)
#     auroc_img = roc_auc_score(img_labels_bin, img_scores)
#     # 像素级ROC
#     pixel_masks_bin = (np.concatenate(pixel_masks) > 0.5).astype(int)
#     fpr_pix, tpr_pix, _ = roc_curve(pixel_masks_bin, np.concatenate(pixel_scores))
#     auroc_pix = roc_auc_score(pixel_masks_bin, np.concatenate(pixel_scores))

#     plt.figure(figsize=(4, 4))
#     plt.plot(fpr_img, tpr_img, linewidth=2, color="#1f77b4", label=f"Image-AUROC: {auroc_img:.4f}")
#     plt.plot(fpr_pix, tpr_pix, linewidth=2, color="#ff7f0e", label=f"Pixel-AUROC: {auroc_pix:.4f}")
#     plt.plot([0,1],[0,1], '--', color="gray", linewidth=1)
#     plt.title("ROC Curve (Image & Pixel Level)")
#     plt.xlabel("False Positive Rate (FPR)")
#     plt.ylabel("True Positive Rate (TPR)")
#     plt.legend(frameon=False)
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_pr_curve(labels, scores, path):
#     labels_bin = (np.array(labels) != 0).astype(int)
#     pre, rec, _ = precision_recall_curve(labels_bin, scores)
#     ap_score = np.trapz(pre, rec)
#     plt.figure(figsize=(4, 4))
#     plt.plot(rec, pre, linewidth=2, color="#ff7f0e", label=f"AP: {ap_score:.4f}")
#     plt.title("Precision-Recall Curve")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.legend(frameon=False)
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_threshold_metric_curve(scores, labels, path):
#     scores = np.array(scores)
#     labels = np.array(labels)
#     labels_bin = (labels != 0).astype(int)
#     thresholds = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 100)
#     f1_list, pre_list, rec_list = [], [], []
#     for t in thresholds:
#         pred = (scores >= t).astype(int)
#         f1_list.append(f1_score(labels_bin, pred, zero_division=0))
#         pre_list.append(precision_score(labels_bin, pred, zero_division=0))
#         rec_list.append(recall_score(labels_bin, pred, zero_division=0))
#     plt.figure(figsize=(5, 3))
#     plt.plot(thresholds, f1_list, label="F1 Score", linewidth=1.5, color="#1f77b4")
#     plt.plot(thresholds, pre_list, label="Precision", linewidth=1.5, color="#ff7f0e")
#     plt.plot(thresholds, rec_list, label="Recall", linewidth=1.5, color="#2ca02c")
#     plt.title("Performance Metrics vs Threshold")
#     plt.xlabel("Anomaly Score Threshold")
#     plt.ylabel("Score")
#     plt.legend(frameon=False)
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_confusion_matrix(labels, preds, path):
#     labels_bin = (np.array(labels) != 0).astype(int)
#     cm = confusion_matrix(labels_bin, preds)
#     plt.figure(figsize=(3, 2.5))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, annot_kws={"size": 10})
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")
#     plt.xticks([0.5, 1.5], ["Normal", "Defect"])
#     plt.yticks([0.5, 1.5], ["Normal", "Defect"], rotation=0)
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()

# def save_metrics_bar(img_au, pix_au, pro, path):
#     plt.figure(figsize=(4, 3))
#     metric_names = ["Image-AUROC", "Pixel-AUROC", "PRO-Score"]
#     metric_values = [img_au, pix_au, pro]
#     colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
#     bars = plt.bar(metric_names, metric_values, color=colors, width=0.6)
#     plt.ylim(0, 1.05)
#     plt.title("Final Performance Metrics")
#     plt.ylabel("Score")
#     # 柱子上标注数值
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
#                 f"{height:.4f}", ha='center', va='bottom', fontsize=8)
#     plt.grid(axis='y')
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.close()
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
# 全局IEEE论文风格设置
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

def save_final_visual(img, heat, gt_mask, ratio, path):
    img = img.permute(1,2,0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    heat = heat.cpu().numpy()
    is_good = torch.sum(gt_mask) < 1e-6

    if is_good:
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
        axes[0].imshow(img)
        axes[0].set_title("Normal Sample", fontsize=9)
        axes[0].axis('off')
        axes[1].imshow(heat, cmap='coolwarm', vmin=0, vmax=0.4)
        axes[1].set_title("Anomaly Heatmap", fontsize=9)
        axes[1].axis('off')
    else:
        pred_mask = (heat > 0.5).astype(float)
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
        axes[0].imshow(img)
        axes[0].set_title(f"Defect Sample\nRatio: {ratio:.2%}", fontsize=9)
        axes[0].axis('off')
        axes[1].imshow(img)
        axes[1].imshow(heat, cmap='jet', alpha=0.6)
        axes[1].set_title("Anomaly Heatmap", fontsize=9)
        axes[1].axis('off')
        axes[2].imshow(img)
        axes[2].imshow(pred_mask, cmap='hot', alpha=0.7)
        axes[2].set_title("Defect Mask", fontsize=9)
        axes[2].axis('off')
    plt.tight_layout(pad=0.2)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_3d_heatmap(score_map, path):
    score_map = score_map.cpu()
    x = np.linspace(0, 255, 64)
    y = np.linspace(0, 255, 64)
    X, Y = np.meshgrid(x, y)
    Z = torch.nn.functional.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=64).squeeze().numpy()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.9)
    ax.set_title("3D Anomaly Heatmap", pad=10)
    ax.set_xlabel("Image Width", labelpad=5)
    ax.set_ylabel("Image Height", labelpad=5)
    ax.set_zlabel("Anomaly Score", labelpad=5)
    ax.set_zlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_anomaly_score_dist(normal, defect, path):
    plt.figure(figsize=(5, 3))
    sns.kdeplot(normal, label="Normal", fill=True, color="#1f77b4", alpha=0.6, linewidth=1)
    sns.kdeplot(defect, label="Defective", fill=True, color="#ff7f0e", alpha=0.6, linewidth=1)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Image-level Anomaly Score")
    plt.ylabel("Probability Density")
    plt.legend(frameon=False)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_pixel_score_hist(pixel_scores, path):
    all_pix = np.concatenate(pixel_scores)
    plt.figure(figsize=(5, 3))
    plt.hist(all_pix, bins=50, color="#9467bd", alpha=0.7, edgecolor='white', linewidth=0.3)
    plt.title("Pixel-level Anomaly Score Histogram")
    plt.xlabel("Pixel Anomaly Score")
    plt.ylabel("Pixel Count")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_defect_ratio_cumulative(ratios, path):
    plt.figure(figsize=(5, 3))
    sorted_ratio = np.sort(ratios)
    cum_counts = np.arange(1, len(sorted_ratio)+1)
    plt.plot(sorted_ratio, cum_counts, linewidth=2, color="#ff7f0e")
    plt.title("Defect Area Ratio Cumulative Distribution")
    plt.xlabel("Defect Area Ratio")
    plt.ylabel("Cumulative Sample Count")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_recon_loss_curve(losses, path):
    plt.figure(figsize=(5, 3))
    plt.plot(losses, color="#2ca02c", linewidth=1)
    avg_loss = np.mean(losses)
    plt.axhline(avg_loss, color='gray', linestyle='--', linewidth=1, label=f"Avg: {avg_loss:.3f}")
    plt.title("Diffusion Reconstruction Loss (Test)")
    plt.xlabel("Sample Index")
    plt.ylabel("MSE Loss")
    plt.legend(frameon=False)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_inference_time_curve(times, path):
    plt.figure(figsize=(5, 3))
    plt.plot(times, color="#d62728", linewidth=1)
    avg_time = np.mean(times)
    plt.axhline(avg_time, color='gray', linestyle='--', linewidth=1, label=f"Avg: {avg_time:.3f}s")
    plt.title("Inference Time per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Time (s)")
    plt.legend(frameon=False)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_dual_roc_curve(img_labels, img_scores, pixel_scores, pixel_masks, path):
    # 图像级ROC
    img_labels_bin = (np.array(img_labels) != 0).astype(int)
    fpr_img, tpr_img, _ = roc_curve(img_labels_bin, img_scores)
    auroc_img = roc_auc_score(img_labels_bin, img_scores)
    # 像素级ROC
    pixel_masks_bin = (np.concatenate(pixel_masks) > 0.5).astype(int)
    fpr_pix, tpr_pix, _ = roc_curve(pixel_masks_bin, np.concatenate(pixel_scores))
    auroc_pix = roc_auc_score(pixel_masks_bin, np.concatenate(pixel_scores))

    plt.figure(figsize=(4, 4))
    plt.plot(fpr_img, tpr_img, linewidth=2, color="#1f77b4", label=f"Image-AUROC: {auroc_img:.4f}")
    plt.plot(fpr_pix, tpr_pix, linewidth=2, color="#ff7f0e", label=f"Pixel-AUROC: {auroc_pix:.4f}")
    plt.plot([0,1],[0,1], '--', color="gray", linewidth=1)
    plt.title("ROC Curve (Image & Pixel Level)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(frameon=False)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_pr_curve(labels, scores, path):
    # 🔥 修复AP负数问题：用sklearn官方函数计算，结果准确
    labels_bin = (np.array(labels) != 0).astype(int)
    pre, rec, _ = precision_recall_curve(labels_bin, scores)
    ap_score = average_precision_score(labels_bin, scores)
    plt.figure(figsize=(4, 4))
    plt.plot(rec, pre, linewidth=2, color="#ff7f0e", label=f"AP: {ap_score:.4f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(frameon=False)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_threshold_metric_curve(scores, labels, path):
    from sklearn.metrics import f1_score, precision_score, recall_score
    scores = np.array(scores)
    labels = np.array(labels)
    labels_bin = (labels != 0).astype(int)
    thresholds = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 100)
    f1_list, pre_list, rec_list = [], [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        f1_list.append(f1_score(labels_bin, pred, zero_division=0))
        pre_list.append(precision_score(labels_bin, pred, zero_division=0))
        rec_list.append(recall_score(labels_bin, pred, zero_division=0))
    plt.figure(figsize=(5, 3))
    plt.plot(thresholds, f1_list, label="F1 Score", linewidth=1.5, color="#1f77b4")
    plt.plot(thresholds, pre_list, label="Precision", linewidth=1.5, color="#ff7f0e")
    plt.plot(thresholds, rec_list, label="Recall", linewidth=1.5, color="#2ca02c")
    plt.title("Performance Metrics vs Threshold")
    plt.xlabel("Anomaly Score Threshold")
    plt.ylabel("Score")
    plt.legend(frameon=False)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_confusion_matrix(labels, preds, path):
    # 🔥 修复：开启颜色条，显示数量深度
    labels_bin = (np.array(labels) != 0).astype(int)
    cm = confusion_matrix(labels_bin, preds)
    plt.figure(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=0.5, annot_kws={"size": 10})
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0.5, 1.5], ["Normal", "Defect"])
    plt.yticks([0.5, 1.5], ["Normal", "Defect"], rotation=0)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_metrics_bar(img_au, pix_au, pro, path):
    plt.figure(figsize=(4, 3))
    metric_names = ["Image-AUROC", "Pixel-AUROC", "PRO-Score"]
    metric_values = [img_au, pix_au, pro]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = plt.bar(metric_names, metric_values, color=colors, width=0.6)
    plt.ylim(0, 1.05)
    plt.title("Final Performance Metrics")
    plt.ylabel("Score")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}", ha='center', va='bottom', fontsize=8)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()