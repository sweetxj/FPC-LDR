# import numpy as np
# from sklearn.metrics import roc_auc_score

# # 图像级 AUROC
# def image_auroc(y_true, y_score):
#     return roc_auc_score(y_true, y_score)

# # 像素级 AUROC（已修复：强制二值化 + 展平）
# def pixel_auroc(masks_list, amaps_list):
#     y_true = []
#     y_pred = []
    
#     for mask, amap in zip(masks_list, amaps_list):
#         #  关键修复：把浮点数 mask 转成 0/1 整数
#         mask_binary = (mask > 0.5).astype(np.int32).flatten()
        
#         y_true.append(mask_binary)
#         y_pred.append(amap.flatten())
    
#     y_true = np.concatenate(y_true)
#     y_pred = np.concatenate(y_pred)
    
#     return roc_auc_score(y_true, y_pred)

# # PRO 指标
# def pro_score(masks, scores):
#     total = 0.0
#     count = 0
#     for m, s in zip(masks, scores):
#         m = (m > 0.5).astype(np.float32)
#         s = (s > np.percentile(s, 95)).astype(np.float32)
#         inter = np.sum(m * s)
#         union = np.sum(m)
#         if union > 0:
#             total += inter / union
#             count += 1
#     return total / count if count > 0 else 0.0
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# 
def image_auroc(y_true, y_score):
    return roc_auc_score(y_true, y_score)

# 
def pixel_auroc(masks_list, amaps_list):
    y_true = []
    y_pred = []
    for mask, amap in zip(masks_list, amaps_list):
        mask_binary = (mask > 0.5).astype(np.int32).flatten()
        y_true.append(mask_binary)
        y_pred.append(amap.flatten())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return roc_auc_score(y_true, y_pred)

# 
def pro_score(masks, scores):
    total = 0.0
    count = 0
    for m, s in zip(masks, scores):
        m = (m > 0.5).astype(np.float32)
        s = (s > np.percentile(s, 95)).astype(np.float32)
        inter = np.sum(m * s)
        union = np.sum(m)
        if union > 0:
            total += inter / union
            count += 1
    return total / count if count > 0 else 0.0

# 
def find_best_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[optimal_idx]
    best_tpr = tpr[optimal_idx]
    best_tnr = 1 - fpr[optimal_idx]
    return best_thresh, best_tpr, best_tnr

# 
def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)