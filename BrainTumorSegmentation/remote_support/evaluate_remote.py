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

from brats2d.config import load_config  # noqa: E402
from brats2d_remote.evaluator import evaluate_all_remote  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote-friendly evaluator.")
    parser.add_argument(
        "--config",
        default=str(REMOTE_ROOT / "configs" / "v100_remote_config.yaml"),
        help="Path to the remote YAML configuration file.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Evaluate only the given experiment name. Repeat to evaluate multiple experiments.",
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
        help="Force single-GPU evaluation even when multiple CUDA GPUs are available.",
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
    experiment_names = args.experiment or [experiment["name"] for experiment in config["experiments"]]
    summaries = evaluate_all_remote(config, experiment_names)

    print("=" * 60)
    print("Remote evaluation completed")
    print("=" * 60)
    for summary in summaries:
        print(
            f"{summary['experiment_name']}: "
            f"Dice={summary['dice']:.4f}, "
            f"IoU={summary['iou']:.4f}, "
            f"Precision={summary['precision']:.4f}, "
            f"Recall={summary['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
