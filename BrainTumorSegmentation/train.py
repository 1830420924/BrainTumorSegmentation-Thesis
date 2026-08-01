from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats2d.config import ensure_directories, load_config  # noqa: E402
from brats2d.trainer import train_experiment  # noqa: E402
from brats2d.utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net and Attention U-Net experiments.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default_config.yaml"),
        help="Path to YAML configuration file.",
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
        help="Override the default number of epochs for quick smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config)
    set_seed(int(config["dataset"]["random_seed"]))

    experiments = config["experiments"]
    if args.experiment is not None:
        experiments = [exp for exp in experiments if exp["name"] == args.experiment]
        if not experiments:
            raise ValueError(f"Experiment not found: {args.experiment}")

    for experiment in experiments:
        metadata = train_experiment(config, experiment, epochs_override=args.epochs)
        print(f"Finished experiment: {experiment['name']}")
        print(f"Best checkpoint : {metadata['best_model_path']}")
        print(f"Best val Dice   : {metadata['best_val_dice']:.4f}")


if __name__ == "__main__":
    main()
