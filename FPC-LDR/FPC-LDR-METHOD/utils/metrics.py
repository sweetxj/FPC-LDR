import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

def calculate_image_auroc(img_scores, labels):
    """"""
    img_scores = np.array(img_scores)
    labels = np.array(labels)
    # 
    labels_bin = (labels != 0).astype(int)
    # 
    if len(np.unique(labels_bin)) < 2:
        raise ValueError("error")
    return roc_auc_score(labels_bin, img_scores)

def calculate_pixel_auroc(pixel_scores, gt_masks):
    """"""
    pixel_scores = np.concatenate(pixel_scores)
    gt_masks = np.concatenate(gt_masks)
    
    masks_bin = (gt_masks > 0.5).astype(int)
    
    if len(np.unique(masks_bin)) < 2:
        raise ValueError("ground truth！")
    return roc_auc_score(masks_bin, pixel_scores)

def calculate_pro(anomaly_scores, gt_masks, threshold=0.5):
    """"""
    pro_scores = []
    for score, mask in zip(anomaly_scores, gt_masks):
        mask_flat = mask.flatten()
       
        if np.sum(mask_flat) == 0:
            continue
        pred_flat = (score.flatten() > threshold).astype(np.float32)
        tp = np.sum(pred_flat * mask_flat)
        fn = np.sum((1 - pred_flat) * mask_flat)
        pro_scores.append(tp / (tp + fn + 1e-8))
    return np.mean(pro_scores) if len(pro_scores) > 0 else 0.0

def get_best_threshold(scores, labels):
    """"""
    scores = np.array(scores)
    labels = np.array(labels)
    labels_bin = (labels != 0).astype(int)
    best_th = 0.5
    best_f1 = 0.0
    # 
    for th in np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 200):
        pred = (scores >= th).astype(int)
        current_f1 = f1_score(labels_bin, pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_th = th
    return round(float(best_th), 4)