from __future__ import annotations

import torch


def batch_segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > threshold).float()
    targets = (targets > 0.5).float()

    predictions = predictions.reshape(predictions.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    epsilon = 1e-6
    tp = (predictions * targets).sum(dim=1)
    fp = (predictions * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - predictions) * targets).sum(dim=1)

    dice = ((2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)).mean().item()
    iou = ((tp + epsilon) / (tp + fp + fn + epsilon)).mean().item()
    precision = ((tp + epsilon) / (tp + fp + epsilon)).mean().item()
    recall = ((tp + epsilon) / (tp + fn + epsilon)).mean().item()
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }
