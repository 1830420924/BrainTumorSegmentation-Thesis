"""Binary segmentation losses for 3-D BraTS experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


def _prepare_targets(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(
            f"Logits and targets must have identical shapes, got {tuple(logits.shape)} "
            f"and {tuple(targets.shape)}"
        )
    return targets.to(device=logits.device, dtype=logits.dtype)


class DiceLoss(nn.Module):
    """Soft Dice loss computed per volume and then averaged across a batch."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError(f"smooth must be positive, got {smooth}")
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _prepare_targets(logits, targets)
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.reshape(probabilities.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice losses."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight must be positive")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _prepare_targets(logits, targets)
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


def build_loss(
    loss_name: str,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> nn.Module:
    """Build BCE, Dice, or the combined BCE-Dice objective."""

    normalized_name = loss_name.strip().lower().replace("-", "_").replace("+", "_")
    if normalized_name in {"bce", "bce_logits", "bce_with_logits"}:
        return nn.BCEWithLogitsLoss()
    if normalized_name == "dice":
        return DiceLoss()
    if normalized_name in {"bce_dice", "dice_bce"}:
        return BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    raise ValueError(
        f"Unsupported loss name {loss_name!r}; expected 'bce', 'dice', or 'bce_dice'"
    )

