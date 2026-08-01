from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats3d.config import ensure_directories, load_config  # noqa: E402
from brats3d.manifest import prepare_manifest  # noqa: E402


def prepare_3d_splits(
    config: Mapping[str, Any],
    reuse_legacy: bool | None = None,
) -> dict[str, Any]:
    """Create only a JSON manifest; raw NIfTI files remain in their original root."""

    ensure_directories(config)
    return prepare_manifest(config, reuse_legacy=reuse_legacy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lightweight 3D NIfTI manifest, reusing the old 2D patient split by default."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default_config.yaml"),
        help="Path to the independent 3D YAML configuration.",
    )
    parser.add_argument(
        "--fresh-split",
        action="store_true",
        help="Ignore the old 2D split and generate a deterministic split from the configured ratios.",
    )
    parser.add_argument(
        "--legacy-splits",
        default=None,
        help="Optional path to a legacy 2D splits.json file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.legacy_splits:
        config["paths"]["legacy_splits_file"] = str(Path(args.legacy_splits).expanduser().resolve())
    manifest = prepare_3d_splits(config, reuse_legacy=False if args.fresh_split else None)

    print("=" * 72)
    print("BraTS 3D manifest prepared")
    print("=" * 72)
    print(f"Manifest        : {config['paths']['manifest_file']}")
    print(f"Split counts    : {manifest['counts']}")
    print(f"Split source    : {manifest['split_source']['kind']}")
    if manifest["split_source"].get("path"):
        print(f"Reused split    : {manifest['split_source']['path']}")
    print("Raw data copied : False (manifest references original NIfTI paths)")


if __name__ == "__main__":
    main()

