from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import BrainTumorSliceDataset
from .losses import build_loss
from .metrics import batch_segmentation_metrics
from .models import build_model
from .utils import ensure_dir, resolve_device, safe_torch_save, save_json, write_csv
from .visualization import save_training_curves


def build_dataloader(
    processed_root: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = BrainTumorSliceDataset(processed_root=processed_root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    metrics_accumulator = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        batch_metrics = batch_segmentation_metrics(logits.detach(), masks.detach(), threshold=threshold)
        for key, value in batch_metrics.items():
            metrics_accumulator[key] += value
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("DataLoader produced zero batches.")

    summary = {"loss": total_loss / num_batches}
    for key, value in metrics_accumulator.items():
        summary[key] = value / num_batches
    return summary


def train_experiment(
    config: dict[str, Any],
    experiment: dict[str, str],
    epochs_override: int | None = None,
) -> dict[str, Any]:
    paths = config["paths"]
    training_cfg = config["training"]
    processed_root = Path(paths["processed_root"])
    artifacts_root = ensure_dir(Path(paths["artifacts_root"]) / "experiments" / experiment["name"])

    device = resolve_device(training_cfg["device"])
    batch_size = int(training_cfg["batch_size"])
    epochs = int(epochs_override or training_cfg["epochs"])
    threshold = float(training_cfg["threshold"])
    num_workers = int(training_cfg["num_workers"])
    pin_memory = bool(training_cfg["pin_memory"] and device.type == "cuda")
    base_channels = int(training_cfg["base_channels"])

    train_loader = build_dataloader(
        processed_root=processed_root,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = build_dataloader(
        processed_root=processed_root,
        split="val",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    model = build_model(experiment["model"], base_channels=base_channels).to(device)
    criterion = build_loss(experiment["loss"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))

    history: list[dict[str, float]] = []
    best_dice = -1.0
    best_checkpoint_path = artifacts_root / "best_model.pth"

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, threshold, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, threshold, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(row)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            safe_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": experiment["model"],
                    "loss_name": experiment["loss"],
                    "base_channels": base_channels,
                    "best_val_dice": best_dice,
                    "config_path": config["config_path"],
                },
                best_checkpoint_path,
            )

        print(
            f"[{experiment['name']}] epoch {epoch:03d}/{epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_dice={row['val_dice']:.4f}"
        )

    last_checkpoint_path = artifacts_root / "last_model.pth"
    safe_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": experiment["model"],
            "loss_name": experiment["loss"],
            "base_channels": base_channels,
            "best_val_dice": best_dice,
            "config_path": config["config_path"],
        },
        last_checkpoint_path,
    )

    curves_path = save_training_curves(history, artifacts_root / "training_curves.png")
    write_csv(artifacts_root / "training_history.csv", history)
    metadata = {
        "experiment": experiment,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": float(training_cfg["learning_rate"]),
        "threshold": threshold,
        "best_val_dice": best_dice,
        "best_model_path": str(best_checkpoint_path),
        "last_model_path": str(last_checkpoint_path),
        "curves_path": str(curves_path),
    }
    save_json(artifacts_root / "metadata.json", metadata)
    return metadata
