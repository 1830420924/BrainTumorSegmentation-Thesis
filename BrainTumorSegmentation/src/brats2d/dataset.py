from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BrainTumorSliceDataset(Dataset):
    def __init__(self, processed_root: str | Path, split: str) -> None:
        self.processed_root = Path(processed_root)
        self.split = split
        self.images_dir = self.processed_root / split / "images"
        self.masks_dir = self.processed_root / split / "masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        self.sample_ids = sorted(path.stem for path in self.images_dir.glob("*.png"))
        if not self.sample_ids:
            raise RuntimeError(f"No PNG samples found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        image_path = self.images_dir / f"{sample_id}.png"
        mask_path = self.masks_dir / f"{sample_id}.png"

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample_id,
        }
