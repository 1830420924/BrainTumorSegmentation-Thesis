from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats3d.config import ensure_directories, load_config  # noqa: E402
from brats3d.trainer import train_experiment  # noqa: E402
from brats3d.utils import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train independent 3D U-Net/Attention U-Net BraTS experiments."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default_config.yaml"),
        help="YAML configuration path (default: local 6 GB GPU profile).",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment name to train. Repeat for multiple names; default runs all three.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs, for example --epochs 1 for a smoke run.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a last_model.pth checkpoint (requires exactly one experiment).",
    )
    return parser.parse_args()


def _select_experiments(
    configured: list[dict[str, object]], requested: list[str] | None
) -> list[dict[str, object]]:
    if not requested:
        return configured
    duplicate_names = {name for name in requested if requested.count(name) > 1}
    if duplicate_names:
        raise ValueError(f"Duplicate --experiment values: {sorted(duplicate_names)}")
    by_name = {str(experiment["name"]): experiment for experiment in configured}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(
            f"Unknown experiment(s): {unknown}. Available: {', '.join(sorted(by_name))}"
        )
    return [by_name[name] for name in requested]


def main() -> None:
    args = parse_args()
    config = load_config(args.config, project_root=PROJECT_ROOT)
    ensure_directories(config)
    manifest_path = Path(config["paths"]["manifest_file"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"3D manifest not found: {manifest_path}\n"
            "Run `python prepare_3d_splits.py --config configs/default_config.yaml` first."
        )
    seed_everything(int(config["dataset"].get("random_seed", 42)))
    experiments = _select_experiments(config["experiments"], args.experiment)
    if args.resume is not None and len(experiments) != 1:
        raise ValueError("--resume requires exactly one --experiment value.")

    for experiment in experiments:
        metadata = train_experiment(
            config=config,
            experiment=experiment,
            epochs_override=args.epochs,
            resume_path=args.resume,
        )
        print(f"Finished: {experiment['name']}")
        print(f"  best checkpoint: {metadata['best_model_path']}")
        print(f"  best patient-level validation Dice: {metadata['best_val_dice']:.4f}")


if __name__ == "__main__":
    main()

