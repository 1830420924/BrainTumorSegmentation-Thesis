from __future__ import annotations

import argparse
import sys
from pathlib import Path

REMOTE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REMOTE_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
REMOTE_SRC_ROOT = REMOTE_ROOT / "src"

for candidate in (SRC_ROOT, REMOTE_SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from brats2d.config import ensure_directories, load_config  # noqa: E402
from brats2d.utils import set_seed  # noqa: E402
from brats2d_remote.trainer import train_experiment_remote  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote-friendly trainer with resume support.")
    parser.add_argument(
        "--config",
        default=str(REMOTE_ROOT / "configs" / "v100_remote_config.yaml"),
        help="Path to the remote YAML configuration file.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Run only a single experiment by name. Defaults to all configured experiments.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the target total epoch count.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the experiment's last_model.pth when available.",
    )
    parser.add_argument(
        "--processed-root",
        default=None,
        help="Override the processed dataset root.",
    )
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="Override the remote artifacts root.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override training.device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override training.num_workers.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated CUDA GPU ids, for example 0,1 . Defaults to auto-detect.",
    )
    parser.add_argument(
        "--disable-data-parallel",
        action="store_true",
        help="Force single-GPU training even when multiple CUDA GPUs are available.",
    )
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.processed_root is not None:
        config["paths"]["processed_root"] = str(Path(args.processed_root).resolve())
    if args.artifacts_root is not None:
        config["paths"]["artifacts_root"] = str(Path(args.artifacts_root).resolve())
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.gpu_ids is not None:
        gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
        config["training"]["gpu_ids"] = gpu_ids
    if args.disable_data_parallel:
        config["training"]["data_parallel"] = False
    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    ensure_directories(config)
    set_seed(int(config["dataset"]["random_seed"]))

    experiments = config["experiments"]
    if args.experiment is not None:
        experiments = [exp for exp in experiments if exp["name"] == args.experiment]
        if not experiments:
            raise ValueError(f"Experiment not found: {args.experiment}")

    for experiment in experiments:
        metadata = train_experiment_remote(
            config=config,
            experiment=experiment,
            epochs_override=args.epochs,
            resume=args.resume,
        )
        print(f"Finished experiment: {experiment['name']}")
        print(f"Best checkpoint : {metadata['best_model_path']}")
        print(f"Best val Dice   : {metadata['best_val_dice']:.4f}")
        print(f"Status          : {metadata['status']}")


if __name__ == "__main__":
    main()
