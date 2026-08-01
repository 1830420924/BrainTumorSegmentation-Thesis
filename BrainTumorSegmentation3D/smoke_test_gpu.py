from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats3d.config import load_config  # noqa: E402
from brats3d.dataset import BraTS3DPatchDataset  # noqa: E402
from brats3d.losses import build_loss  # noqa: E402
from brats3d.models import build_model  # noqa: E402
from brats3d.utils import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real BraTS 3D forward/backward step on the local GPU."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default_config.yaml"),
    )
    parser.add_argument(
        "--model",
        default="attention_unet",
        choices=("unet", "attention_unet"),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        default=None,
        help="Optional D H W override used only by this smoke test.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="Optional model width override used only by this smoke test.",
    )
    return parser.parse_args()


def gib(value: int) -> float:
    return float(value) / (1024.0**3)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this check is intended for the local RTX 3060.")

    config = load_config(args.config, project_root=PROJECT_ROOT)
    manifest_path = Path(config["paths"]["manifest_file"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"3D manifest not found: {manifest_path}. Run prepare_3d_splits.py first."
        )

    seed_everything(int(config["dataset"].get("random_seed", 42)))
    dataset = BraTS3DPatchDataset(
        manifest_path,
        split="train",
        config=config,
        patch_size=args.patch_size,
        patches_per_patient=1,
        positive_patch_ratio=1.0,
        augment=False,
        cache_size=0,
    )
    sample = dataset[0]
    image = sample["image"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)

    training = config["training"]
    device = torch.device("cuda")
    model = build_model(
        args.model,
        in_channels=int(training.get("in_channels", 1)),
        out_channels=int(training.get("out_channels", 1)),
        base_channels=int(args.base_channels or training.get("base_channels", 8)),
    ).to(device)
    criterion = build_loss("bce_dice")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    use_amp = bool(training.get("mixed_precision", True))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    image = image.to(device)
    mask = mask.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logits = model(image)
        loss = criterion(logits, mask)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    properties = torch.cuda.get_device_properties(device)
    print("=" * 68)
    print("Local 3D GPU smoke test: OK")
    print("=" * 68)
    print(f"GPU                 : {properties.name}")
    print(f"GPU memory total    : {gib(properties.total_memory):.2f} GiB")
    print(f"Model               : {args.model}")
    print(f"Base channels       : {int(args.base_channels or training.get('base_channels', 8))}")
    print(f"Input shape         : {tuple(image.shape)}")
    print(f"Output shape        : {tuple(logits.shape)}")
    print(f"Loss                : {float(loss.detach().item()):.6f}")
    print(f"AMP                 : {use_amp}")
    print(f"Peak allocated      : {gib(torch.cuda.max_memory_allocated(device)):.2f} GiB")
    print(f"Peak reserved       : {gib(torch.cuda.max_memory_reserved(device)):.2f} GiB")


if __name__ == "__main__":
    main()
