import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def calculate_image_auroc(img_scores, labels):
    img_scores = np.array(img_scores)
    labels = np.array(labels)
    labels_bin = (labels != 0).astype(int)
    if len(np.unique(labels_bin)) < 2:
        raise ValueError("标签仅包含单种类别，无法计算AUROC")
    return roc_auc_score(labels_bin, img_scores)

def calculate_pixel_auroc(pixel_scores, gt_masks):
    pixel_scores = np.concatenate(pixel_scores)
    gt_masks = np.concatenate(gt_masks)
    masks_bin = (gt_masks > 0.5).astype(int)
    if len(np.unique(masks_bin)) < 2:
        raise ValueError("掩码仅包含单种类别，无法计算Pixel-AUROC")
    return roc_auc_score(masks_bin, pixel_scores)

def calculate_pro(anomaly_scores, gt_masks, threshold=0.5):
    """计算PRO分数，包含工业级形态学后处理"""
    pro_scores = []
    for score, mask in zip(anomaly_scores, gt_masks):
        mask_flat = mask.flatten()
        if np.sum(mask_flat) == 0:
            continue
        # 生成预测掩码
        pred_mask = (score > threshold).astype(np.uint8)
        # 形态学后处理
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel_open)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel_close)
        
        pred_flat = pred_mask.flatten()
        tp = np.sum(pred_flat * mask_flat)
        fn = np.sum((1 - pred_flat) * mask_flat)
        pro_scores.append(tp / (tp + fn + 1e-8))
    return np.mean(pro_scores) if len(pro_scores) > 0 else 0.0

def get_best_threshold(scores, labels):
    """基于F1分数计算最优阈值"""
    scores = np.array(scores)
    labels = np.array(labels)
    labels_bin = (labels != 0).astype(int)
    best_th = 0.5
    best_f1 = 0.0
    for th in np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 200):
        pred = (scores >= th).astype(int)
        current_f1 = f1_score(labels_bin, pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_th = th
    return round(float(best_th), 4)

def get_best_pro_threshold(anomaly_scores, gt_masks):
    """专门为PRO指标优化的阈值计算函数"""
    best_th = 0.5
    best_pro = 0.0
    for th in np.linspace(0.3, 0.9, 60):
        current_pro = calculate_pro(anomaly_scores, gt_masks, threshold=th)
        if current_pro > best_pro:
            best_pro = current_pro
            best_th = th
    return round(float(best_th), 4), best_pro