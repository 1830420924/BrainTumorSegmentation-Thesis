from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats2d.config import ensure_directories, load_config  # noqa: E402
from brats2d.preprocess import check_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check BraTS 2020 dataset structure and preview slices.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default_config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--show", action="store_true", help="Display the preview figure after saving it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config)
    summary = check_dataset(config, show=args.show)

    print("=" * 60)
    print("BraTS 2020 dataset check completed")
    print("=" * 60)
    print(f"Training patients   : {summary['training_patients']}")
    print(f"Validation patients : {summary['validation_patients']}")
    print(f"Preview patient     : {summary['preview_patient']}")
    print(f"FLAIR shape         : {tuple(summary['flair_shape'])}")
    print(f"Mask labels         : {summary['mask_labels']}")
    print(f"Preview slices      : {summary['preview_slices']}")
    print(f"Preview image saved : {summary['preview_path']}")
    print(f"Preview written     : {summary['preview_saved']}")


if __name__ == "__main__":
    main()
