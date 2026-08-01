from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader

from brats2d.dataset import BrainTumorSliceDataset
from brats2d.losses import build_loss
from brats2d.metrics import batch_segmentation_metrics
from brats2d.models import build_model
from brats2d.utils import ensure_dir, resolve_device, safe_torch_save, save_json, write_csv
from brats2d.visualization import save_training_curves


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def resolve_gpu_ids(training_cfg: dict[str, Any], device: torch.device) -> list[int]:
    configured_gpu_ids = training_cfg.get("gpu_ids", [])
    if not device.type == "cuda":
        return []
    if configured_gpu_ids:
        return [int(gpu_id) for gpu_id in configured_gpu_ids]
    return list(range(torch.cuda.device_count()))


def should_enable_data_parallel(training_cfg: dict[str, Any], device: torch.device, gpu_ids: list[int]) -> bool:
    setting = training_cfg.get("data_parallel", "auto")
    if device.type != "cuda":
        return False
    if len(gpu_ids) <= 1:
        return False
    if isinstance(setting, str) and setting.lower() == "auto":
        return True
    return bool(setting)


def build_runtime_model(
    model_name: str,
    base_channels: int,
    device: torch.device,
    training_cfg: dict[str, Any],
) -> tuple[torch.nn.Module, torch.device, list[int], bool]:
    gpu_ids = resolve_gpu_ids(training_cfg, device)
    enable_data_parallel = should_enable_data_parallel(training_cfg, device, gpu_ids)

    model = build_model(model_name, base_channels=base_channels)
    if enable_data_parallel:
        runtime_device = torch.device(f"cuda:{gpu_ids[0]}")
        model = model.to(runtime_device)
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    else:
        runtime_device = device
        model = model.to(runtime_device)
    return model, runtime_device, gpu_ids, enable_data_parallel


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


def load_torch_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def build_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    experiment: dict[str, str],
    base_channels: int,
    config_path: str,
    epoch: int,
    best_val_dice: float,
    history: list[dict[str, float]],
    cumulative_wall_time_seconds: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_name": experiment["model"],
        "loss_name": experiment["loss"],
        "base_channels": base_channels,
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "history": history,
        "cumulative_wall_time_seconds": cumulative_wall_time_seconds,
        "config_path": config_path,
    }


def write_training_state(
    artifacts_root: Path,
    experiment: dict[str, str],
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    threshold: float,
    history: list[dict[str, float]],
    best_dice: float,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
    epochs_target: int,
    resumed_from_epoch: int,
    cumulative_wall_time_seconds: float,
    status: str,
    gpu_ids: list[int],
    data_parallel_enabled: bool,
) -> dict[str, Any]:
    curves_path = save_training_curves(history, artifacts_root / "training_curves.png")
    write_csv(artifacts_root / "training_history.csv", history)
    metadata = {
        "experiment": experiment,
        "device": str(device),
        "epochs_target": epochs_target,
        "completed_epochs": len(history),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "threshold": threshold,
        "best_val_dice": best_dice,
        "best_model_path": str(best_checkpoint_path),
        "last_model_path": str(last_checkpoint_path),
        "curves_path": str(curves_path),
        "resumed_from_epoch": resumed_from_epoch,
        "cumulative_wall_time_seconds": cumulative_wall_time_seconds,
        "status": status,
        "gpu_ids": gpu_ids,
        "gpu_count": len(gpu_ids),
        "data_parallel_enabled": data_parallel_enabled,
        "last_updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(artifacts_root / "metadata.json", metadata)
    return metadata


def train_experiment_remote(
    config: dict[str, Any],
    experiment: dict[str, str],
    epochs_override: int | None = None,
    resume: bool = False,
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
    learning_rate = float(training_cfg["learning_rate"])

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

    model, runtime_device, gpu_ids, data_parallel_enabled = build_runtime_model(
        model_name=experiment["model"],
        base_channels=base_channels,
        device=device,
        training_cfg=training_cfg,
    )
    criterion = build_loss(experiment["loss"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    best_dice = -1.0
    start_epoch = 1
    resumed_from_epoch = 0
    cumulative_wall_time_seconds = 0.0

    best_checkpoint_path = artifacts_root / "best_model.pth"
    last_checkpoint_path = artifacts_root / "last_model.pth"

    if resume and last_checkpoint_path.exists():
        checkpoint = load_torch_checkpoint(last_checkpoint_path)
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        history = list(checkpoint.get("history", []))
        best_dice = float(checkpoint.get("best_val_dice", -1.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        resumed_from_epoch = start_epoch - 1
        cumulative_wall_time_seconds = float(checkpoint.get("cumulative_wall_time_seconds", 0.0))

    if start_epoch > epochs:
        return write_training_state(
            artifacts_root=artifacts_root,
            experiment=experiment,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            threshold=threshold,
            history=history,
            best_dice=best_dice,
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            epochs_target=epochs,
            resumed_from_epoch=resumed_from_epoch,
            cumulative_wall_time_seconds=cumulative_wall_time_seconds,
            status="completed",
            gpu_ids=gpu_ids,
            data_parallel_enabled=data_parallel_enabled,
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_started_at = perf_counter()

        train_metrics = run_epoch(model, train_loader, criterion, runtime_device, threshold, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, runtime_device, threshold, optimizer=None)

        epoch_wall_time_seconds = perf_counter() - epoch_started_at
        cumulative_wall_time_seconds += epoch_wall_time_seconds

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "epoch_wall_time_seconds": epoch_wall_time_seconds,
        }
        history.append(row)

        checkpoint_payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            experiment=experiment,
            base_channels=base_channels,
            config_path=config["config_path"],
            epoch=epoch,
            best_val_dice=max(best_dice, val_metrics["dice"]),
            history=history,
            cumulative_wall_time_seconds=cumulative_wall_time_seconds,
        )
        safe_torch_save(checkpoint_payload, last_checkpoint_path)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            checkpoint_payload["best_val_dice"] = best_dice
            safe_torch_save(checkpoint_payload, best_checkpoint_path)

        write_training_state(
            artifacts_root=artifacts_root,
            experiment=experiment,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            threshold=threshold,
            history=history,
            best_dice=best_dice,
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            epochs_target=epochs,
            resumed_from_epoch=resumed_from_epoch,
            cumulative_wall_time_seconds=cumulative_wall_time_seconds,
            status="in_progress" if epoch < epochs else "completed",
            gpu_ids=gpu_ids,
            data_parallel_enabled=data_parallel_enabled,
        )

        print(
            f"[{experiment['name']}] epoch {epoch:03d}/{epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_dice={row['val_dice']:.4f} "
            f"epoch_time={epoch_wall_time_seconds:.1f}s "
            f"gpus={gpu_ids if gpu_ids else ['cpu']}"
        )

    return write_training_state(
        artifacts_root=artifacts_root,
        experiment=experiment,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        threshold=threshold,
        history=history,
        best_dice=best_dice,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        epochs_target=epochs,
        resumed_from_epoch=resumed_from_epoch,
        cumulative_wall_time_seconds=cumulative_wall_time_seconds,
        status="completed",
        gpu_ids=gpu_ids,
        data_parallel_enabled=data_parallel_enabled,
    )
