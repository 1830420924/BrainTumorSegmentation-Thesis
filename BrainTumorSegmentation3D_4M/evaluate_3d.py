from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats3d.config import ensure_directories, load_config  # noqa: E402
from brats3d.evaluator import evaluate_all, evaluate_experiment  # noqa: E402
from brats3d.utils import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate four-modality 3D checkpoints on complete patient volumes."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default_config.yaml"),
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment name. Repeat to evaluate several; default evaluates all configured runs.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint path (requires exactly one --experiment).",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=("train", "val", "test"),
        help="Manifest split; default comes from evaluation.split (normally test).",
    )
    nifti_group = parser.add_mutually_exclusive_group()
    nifti_group.add_argument(
        "--save-nifti",
        action="store_true",
        dest="save_nifti",
        help="Save binary 3D predictions with the source affine.",
    )
    nifti_group.add_argument(
        "--no-save-nifti",
        action="store_false",
        dest="save_nifti",
        help="Disable NIfTI output even if enabled in the configuration.",
    )
    parser.set_defaults(save_nifti=None)
    return parser.parse_args()


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

    available = {str(item["name"]): item for item in config["experiments"]}
    names = args.experiment or list(available)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown experiment(s): {unknown}. Available: {', '.join(sorted(available))}"
        )
    if args.checkpoint is not None:
        if len(names) != 1:
            raise ValueError("--checkpoint requires exactly one --experiment value.")
        summaries = [
            evaluate_experiment(
                config=config,
                experiment_name=names[0],
                checkpoint_path=args.checkpoint,
                split=args.split,
                save_nifti=args.save_nifti,
            )
        ]
    else:
        summaries = evaluate_all(
            config=config,
            experiment_names=names,
            split=args.split,
            save_nifti=args.save_nifti,
        )

    print("Evaluation completed (macro average over complete 3D patients)")
    for summary in summaries:
        print(
            f"  {summary['experiment_name']}: "
            f"Dice={summary['dice']:.4f}, IoU={summary['iou']:.4f}, "
            f"Precision={summary['precision']:.4f}, Recall={summary['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
