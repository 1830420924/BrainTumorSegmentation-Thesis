from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a top-level mapping.")

    config = deepcopy(config)
    path_section = config.get("paths", {})
    for key, value in list(path_section.items()):
        if value is None:
            continue
        resolved = Path(value)
        if not resolved.is_absolute():
            resolved = (config_path.parent / resolved).resolve()
        path_section[key] = str(resolved)
    config["paths"] = path_section
    config["config_path"] = str(config_path)
    return config


def ensure_directories(config: dict[str, Any]) -> None:
    for key in ("project_root", "processed_root", "artifacts_root"):
        path = Path(config["paths"][key])
        path.mkdir(parents=True, exist_ok=True)
