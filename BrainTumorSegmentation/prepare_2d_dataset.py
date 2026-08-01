from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats2d.config import ensure_directories, load_config  # noqa: E402
from brats2d.preprocess import prepare_2d_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 2D BraTS PNG slices from the training dataset.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default_config.yaml"),
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config)
    summary = prepare_2d_dataset(config)

    print("=" * 60)
    print("BraTS 2020 preprocessing completed")
    print("=" * 60)
    for split_name in ("train", "val", "test"):
        patient_count = len(summary["patients"][split_name])
        slice_count = summary["slice_counts"][split_name]
        print(f"{split_name:>5} patients: {patient_count:3d} | slices: {slice_count}")
    print(f"Splits file saved: {config['paths']['splits_file']}")


if __name__ == "__main__":
    main()
