from __future__ import annotations

import csv
import io
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_write_bytes(path: str | Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(payload)
        return path
    except PermissionError:
        with tempfile.NamedTemporaryFile(delete=False, suffix=path.suffix) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        try:
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Copy-Item -LiteralPath '{temp_path}' -Destination '{path}' -Force",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()
        return path


def safe_write_text(path: str | Path, payload: str, encoding: str = "utf-8") -> Path:
    return safe_write_bytes(path, payload.encode(encoding))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    safe_write_text(path, buffer.getvalue())


def safe_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    safe_write_bytes(path, buffer.getvalue())
    return path


def resolve_device(device_setting: str) -> torch.device:
    if device_setting != "auto":
        return torch.device(device_setting)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
