import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

# ---------------------- Loss Functions ----------------------
def dice_loss(gold_mask, pred_mask, smooth=1e-6):
    # Convert pred_mask to binary
    pred_mask_binary = (pred_mask > 0.).int()

    mask = gold_mask != -1

    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]

    intersection = (gold_mask_masked * pred_mask_masked).sum()
    union = gold_mask_masked.sum() + pred_mask_masked.sum()

    return 1 - ((2.0 * intersection + smooth) / (union + smooth))

def WBCE(gold_mask, pred_mask, w0=1, w1=100):
    mask = gold_mask != -1

    gold_mask_masked = gold_mask[mask].float()
    pred_mask_masked = pred_mask[mask].float()

    weights = gold_mask_masked * w1 + (1 - gold_mask_masked) * w0
    criterion = torch.nn.BCEWithLogitsLoss(weight=weights, reduction='mean')
    return criterion(pred_mask_masked, gold_mask_masked)
    #return torch.nn.functional.binary_cross_entropy_with_logits(pred_mask_masked, gold_mask_masked, weights, reduction='mean')

# Pred mask is the predicted fire segmentation mask with probability scores that it is fire (class 1)
# Gold mask is the ground truth fire segmentation mask with values -1 (no data), 0 (no fire), 1 (fire) 
def loss(gold_mask, pred_mask):
    return WBCE(gold_mask, pred_mask) + 2 * dice_loss(gold_mask, pred_mask)

# ---------------------- Evaluation Functions ----------------------
def mean_iou(gold_mask, pred_mask):
    pred_mask_binary = (pred_mask > 0.).int()

    mask = gold_mask != -1

    # Calculate intersection and union of class 1 pixels
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]

    intersection = (gold_mask_masked * pred_mask_masked).sum()
    union = gold_mask_masked.sum() + pred_mask_masked.sum() - intersection

    # Calculate intersection over union of class 0 pixels
    gold_mask_masked = 1 - gold_mask_masked
    pred_mask_masked = 1 - pred_mask_masked

    intersection_0 = (gold_mask_masked * pred_mask_masked).sum()
    union_0 = gold_mask_masked.sum() + pred_mask_masked.sum() - intersection_0

    return ((intersection / union) + (intersection_0 / union_0)) / 2

def accuracy(gold_mask, pred_mask):
    pred_mask_binary = (pred_mask > 0.).int()

    mask = gold_mask != -1

    # Calculate accuracy of class 1 pixels
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]

    accuracy_1 = (gold_mask_masked == pred_mask_masked).sum() / len(gold_mask_masked)

    # Calculate accuracy of class 0 pixels
    gold_mask_masked = 1 - gold_mask_masked
    pred_mask_masked = 1 - pred_mask_masked

    accuracy_0 = (gold_mask_masked == pred_mask_masked).sum() / len(gold_mask_masked)
    
    return (accuracy_1 + accuracy_0) / 2

def distance(gold_mask, pred_mask):
    mask = gold_mask != -1

    gold_mask_masked = gold_mask[mask].float()
    pred_mask_masked = torch.sigmoid(pred_mask[mask].float()) # Clamp values between 0 and 1

    return torch.linalg.norm(gold_mask_masked - pred_mask_masked)

def f1_score(gold_mask, pred_mask):
    """
    Computes the F1 score between the predicted mask and the ground truth mask.
    
    Args:
        gold_mask (torch.Tensor): Ground truth mask with values -1 (no data), 0 (no fire), 1 (fire).
        pred_mask (torch.Tensor): Predicted mask with probability scores.
    
    Returns:
        float: Mean F1 score for both classes (fire and no-fire).
    """
    # Convert pred_mask to binary
    pred_mask_binary = (pred_mask > 0.).int()
    
    # Mask out -1 values in the ground truth
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]
    
    # Calculate TP, FP, FN for class 1 (fire)
    tp_1 = (gold_mask_masked * pred_mask_masked).sum()  # True Positives
    fp_1 = ((1 - gold_mask_masked) * pred_mask_masked).sum()  # False Positives
    fn_1 = (gold_mask_masked * (1 - pred_mask_masked)).sum()  # False Negatives
    
    # Precision and Recall for class 1
    precision_1 = tp_1 / (tp_1 + fp_1 + 1e-6)
    recall_1 = tp_1 / (tp_1 + fn_1 + 1e-6)
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + 1e-6)
    
    # Calculate TP, FP, FN for class 0 (no fire)
    gold_mask_masked_0 = 1 - gold_mask_masked
    pred_mask_masked_0 = 1 - pred_mask_masked
    tp_0 = (gold_mask_masked_0 * pred_mask_masked_0).sum()  # True Positives
    fp_0 = ((1 - gold_mask_masked_0) * pred_mask_masked_0).sum()  # False Positives
    fn_0 = (gold_mask_masked_0 * (1 - pred_mask_masked_0)).sum()  # False Negatives
    
    # Precision and Recall for class 0
    precision_0 = tp_0 / (tp_0 + fp_0 + 1e-6)
    recall_0 = tp_0 / (tp_0 + fn_0 + 1e-6)
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + 1e-6)
    
    # Return the mean F1 score for both classes
    return (f1_1 + f1_0) / 2

from sklearn.metrics import roc_auc_score

def auc_score(gold_mask, pred_mask):
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask].float().cpu().numpy()
    pred_mask_masked = torch.sigmoid(pred_mask[mask]).detach().cpu().numpy()
    
    # Handle cases where all labels are the same (e.g., all 0s or all 1s)
    if len(np.unique(gold_mask_masked)) == 1:
        return np.nan
    return roc_auc_score(gold_mask_masked, pred_mask_masked)

def precision_recall(gold_mask, pred_mask):
    """
    Computes precision and recall between the predicted mask and the ground truth mask.
    
    Args:
        gold_mask (torch.Tensor): Ground truth mask with values -1 (no data), 0 (no fire), 1 (fire).
        pred_mask (torch.Tensor): Predicted mask with probability scores.
    
    Returns:
        float: Mean precision for both classes (fire and no-fire).
        float: Mean recall for both classes (fire and no-fire).
    """
    # Convert pred_mask to binary
    pred_mask_binary = (pred_mask > 0.).int()
    
    # Mask out -1 values in the ground truth
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]
    
    # Calculate TP, FP, FN for class 1 (fire)
    tp_1 = (gold_mask_masked * pred_mask_masked).sum()  # True Positives
    fp_1 = ((1 - gold_mask_masked) * pred_mask_masked).sum()  # False Positives
    fn_1 = (gold_mask_masked * (1 - pred_mask_masked)).sum()  # False Negatives
    
    # Precision and Recall for class 1
    precision_1 = tp_1 / (tp_1 + fp_1 + 1e-6)
    recall_1 = tp_1 / (tp_1 + fn_1 + 1e-6)
    
    # Calculate TP, FP, FN for class 0 (no fire)
    gold_mask_masked_0 = 1 - gold_mask_masked
    pred_mask_masked_0 = 1 - pred_mask_masked
    tp_0 = (gold_mask_masked_0 * pred_mask_masked_0).sum()  # True Positives
    fp_0 = ((1 - gold_mask_masked_0) * pred_mask_masked_0).sum()  # False Positives
    fn_0 = (gold_mask_masked_0 * (1 - pred_mask_masked_0)).sum()  # False Negatives
    
    # Precision and Recall for class 0
    precision_0 = tp_0 / (tp_0 + fp_0 + 1e-6)
    recall_0 = tp_0 / (tp_0 + fn_0 + 1e-6)
    
    # Return the mean precision and recall for both classes
    mean_precision = (precision_1 + precision_0) / 2
    mean_recall = (recall_1 + recall_0) / 2
    return mean_precision, mean_recall


def dice_score(gold_mask, pred_mask, smooth=1e-6):
    """
    Computes the Dice score between the predicted mask and the ground truth mask.
    
    Args:
        gold_mask (torch.Tensor): Ground truth mask with values -1 (no data), 0 (no fire), 1 (fire).
        pred_mask (torch.Tensor): Predicted mask with probability scores.
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: Mean Dice score for both classes (fire and no-fire).
    """
    # Convert pred_mask to binary
    pred_mask_binary = (pred_mask > 0.).int()
    
    # Mask out -1 values in the ground truth
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]
    
    # Calculate Dice score for class 1 (fire)
    intersection_1 = (gold_mask_masked * pred_mask_masked).sum()
    union_1 = gold_mask_masked.sum() + pred_mask_masked.sum()
    dice_1 = (2.0 * intersection_1 + smooth) / (union_1 + smooth)
    
    # Calculate Dice score for class 0 (no fire)
    gold_mask_masked_0 = 1 - gold_mask_masked
    pred_mask_masked_0 = 1 - pred_mask_masked
    intersection_0 = (gold_mask_masked_0 * pred_mask_masked_0).sum()
    union_0 = gold_mask_masked_0.sum() + pred_mask_masked_0.sum()
    dice_0 = (2.0 * intersection_0 + smooth) / (union_0 + smooth)
    
    # Return the mean Dice score for both classes
    return (dice_1 + dice_0) / 2