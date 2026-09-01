# import os
# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from utils.config import *

# def plot_roc(fpr, tpr, auroc, save_path):
#     plt.figure()
#     plt.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
#     plt.plot([0,1],[0,1],'k--')
#     plt.legend()
#     plt.savefig(save_path, dpi=300)
#     plt.close()

# def plot_confusion(y_true, y_pred, save_path):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.imshow(cm, cmap="Blues")
#     plt.savefig(save_path, dpi=300)
#     plt.close()

# def save_result(img, mask, amap, ratio, is_anomaly, save_path):
#     img = (img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
#     mask = (mask[0].cpu().numpy()*255).astype(np.uint8)
#     amap = amap[0].cpu().numpy()
#     amap = (amap - amap.min())/(amap.max()-amap.min()+1e-8)
#     amap = np.uint8(plt.cm.jet(amap)[...,:3]*255)
#     amap = cv2.addWeighted(img, 0.5, amap, 0.5, 0)
#     cv2.putText(amap, f"defect:{ratio:.1%} anomaly:{is_anomaly}",
#                 (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,2550),2)
#     cat = np.hstack([img, cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR), amap])
#     cv2.imwrite(save_path, cat[:,:,::-1])
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import seaborn as sns
# 
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

plt.rcParams['figure.dpi'] = 300

def save_result(img, mask, amap, ratio, is_anomaly, save_path):
    img = (img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    mask = (mask[0].cpu().numpy()*255).astype(np.uint8)
    amap = amap[0].cpu().numpy()
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    heat = np.uint8(plt.cm.jet(amap)[...,:3]*255)
    vis = cv2.addWeighted(img, 0.5, heat, 0.5, 0)
    cv2.putText(vis, f"Defect:{ratio:.1%} Anomaly:{is_anomaly}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    concat = np.hstack((img, mask, vis))
    cv2.imwrite(save_path, concat[...,::-1])

# 
def plot_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# 
def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# 
def plot_similarity(memory, path, n=2000):
    idx = torch.randperm(len(memory))[:n]
    mat = memory[idx].cpu().numpy()
    sim = np.corrcoef(mat)
    plt.figure()
    sns.heatmap(sim, cmap="coolwarm")
    plt.title("Feature Similarity Matrix")
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# 
def plot_score_dist(scores, labels, path):
    scores = np.array(scores)
    normal = scores[np.array(labels)==0]
    anomaly = scores[np.array(labels)==1]
    plt.figure()
    plt.hist(normal, bins=20, alpha=0.5, label="Normal")
    plt.hist(anomaly, bins=20, alpha=0.5, label="Anomaly")
    plt.xlabel("Anomaly Score")
    plt.legend()
    plt.title("Score Distribution")
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# 
def plot_defect_ratio(ratios, path):
    plt.figure()
    plt.hist(ratios, bins=15)
    plt.xlabel("Defect Ratio")
    plt.title("Defect Area Distribution")
    plt.savefig(path, bbox_inches='tight')
    plt.close()