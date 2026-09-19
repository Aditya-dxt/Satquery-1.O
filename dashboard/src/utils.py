import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes Precision, Recall, F1-Score (Dice), and IoU for binary change detection.
    """
    probs = torch.sigmoid(preds)
    binary_preds = (probs > threshold).float()

    tp = (binary_preds * targets).sum().item()
    fp = (binary_preds * (1 - targets)).sum().item()
    fn = ((1 - binary_preds) * targets).sum().item()
    tn = ((1 - binary_preds) * (1 - targets)).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'accuracy': accuracy
    }

class MetricTracker:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(preds)
        binary_preds = (probs > self.threshold).float()

        self.tp += (binary_preds * targets).sum().item()
        self.fp += (binary_preds * (1.0 - targets)).sum().item()
        self.fn += ((1.0 - binary_preds) * targets).sum().item()
        self.tn += ((1.0 - binary_preds) * (1.0 - targets)).sum().item()

    def compute(self) -> dict:
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-7)
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou': iou
        }

def save_visualization_sample(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    mask: torch.Tensor,
    pred_mask: torch.Tensor,
    save_path: str
):
    """
    Generates a 4-panel comparison plot for project demos and PPT presentations.
    """
    # Un-normalize RGB images
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    img_a_np = (img_a.cpu().numpy().transpose(1, 2, 0) * std + mean).clip(0, 1)
    img_b_np = (img_b.cpu().numpy().transpose(1, 2, 0) * std + mean).clip(0, 1)
    mask_np = mask.cpu().numpy().squeeze()
    pred_np = (torch.sigmoid(pred_mask).cpu().numpy().squeeze() > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_a_np)
    axes[0].set_title("Pre-Change (Time A)")
    axes[0].axis("off")

    axes[1].imshow(img_b_np)
    axes[1].set_title("Post-Change (Time B)")
    axes[1].axis("off")

    axes[2].imshow(mask_np, cmap="gray")
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis("off")

    axes[3].imshow(pred_np, cmap="jet")
    axes[3].set_title("AI Change Prediction")
    axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()