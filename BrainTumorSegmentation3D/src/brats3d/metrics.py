"""Patient-volume segmentation metrics."""

from __future__ import annotations

import torch


def _safe_ratio(numerator: float, denominator: float, undefined_value: float) -> float:
    return numerator / denominator if denominator > 0 else undefined_value


def volume_segmentation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> dict[str, float]:
    """Compute metrics over every voxel of one complete patient volume.

    ``prediction`` and ``target`` may include singleton batch/channel dimensions,
    but must have identical shapes. Empty prediction and target volumes are
    treated as a correct segmentation and receive a score of one.
    """

    prediction = torch.as_tensor(prediction).detach()
    target = torch.as_tensor(target).detach()
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target must have identical shapes, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )
    if prediction.numel() == 0:
        raise ValueError("Prediction and target volumes must not be empty tensors")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    probabilities = torch.sigmoid(prediction) if from_logits else prediction
    predicted_positive = probabilities >= threshold
    target_positive = target > 0.5

    true_positive = float(torch.logical_and(predicted_positive, target_positive).sum().item())
    false_positive = float(
        torch.logical_and(predicted_positive, torch.logical_not(target_positive)).sum().item()
    )
    false_negative = float(
        torch.logical_and(torch.logical_not(predicted_positive), target_positive).sum().item()
    )

    dice_denominator = 2.0 * true_positive + false_positive + false_negative
    union = true_positive + false_positive + false_negative
    predicted_count = true_positive + false_positive
    target_count = true_positive + false_negative

    return {
        "dice": _safe_ratio(2.0 * true_positive, dice_denominator, 1.0),
        "iou": _safe_ratio(true_positive, union, 1.0),
        "precision": _safe_ratio(
            true_positive,
            predicted_count,
            1.0 if target_count == 0 else 0.0,
        ),
        "recall": _safe_ratio(true_positive, target_count, 1.0),
    }


# The alias states the intended aggregation unit explicitly at call sites.
patient_volume_metrics = volume_segmentation_metrics

