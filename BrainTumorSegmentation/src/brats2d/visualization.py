from __future__ import annotations

from pathlib import Path
import io

import matplotlib.pyplot as plt
import numpy as np

from .utils import safe_write_bytes


def save_training_curves(history: list[dict[str, float]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    val_dices = [row["val_dice"] for row in history]
    val_ious = [row["val_iou"] for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_losses, label="train_loss")
    axes[0].plot(epochs, val_losses, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, val_dices, label="val_dice")
    axes[1].plot(epochs, val_ious, label="val_iou")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    safe_write_bytes(output_path, buffer.getvalue())
    return output_path


def save_prediction_panel(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    sample_id: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"{sample_id}\nImage")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("Prediction")
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(mask, cmap="Reds", alpha=0.25)
    axes[3].imshow(prediction, cmap="Greens", alpha=0.25)
    axes[3].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    safe_write_bytes(output_path, buffer.getvalue())
    return output_path
