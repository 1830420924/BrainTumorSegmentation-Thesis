from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats2d.config import load_config  # noqa: E402
from brats2d.evaluator import evaluate_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained experiments on the test split.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default_config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Evaluate only the given experiment name. Repeat to evaluate multiple experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment_names = args.experiment or [experiment["name"] for experiment in config["experiments"]]
    summaries = evaluate_all(config, experiment_names)

    print("=" * 60)
    print("Evaluation completed")
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
