"""
Configuration loading utilities.

This module loads JSON config files and merges base configuration
with dataset-specific configuration files.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """
    Save a dictionary as a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a dictionary.

    Nested dictionaries are merged. Other values are overwritten.
    """
    result = deepcopy(base)

    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_config(
    dataset_name: Optional[str] = None,
    extra_config_path: Optional[str | Path] = None
) -> Dict[str, Any]:
    """
    Load project configuration.

    Parameters
    ----------
    dataset_name:
        Optional dataset name. Supported values are:
        - "skab"
        - "batadal"

    extra_config_path:
        Optional extra config file. Values in this file override previous ones.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    config = load_json(CONFIG_DIR / "base_config.json")

    if dataset_name is not None:
        dataset_name = dataset_name.lower().strip()

        if dataset_name == "skab":
            dataset_config = load_json(CONFIG_DIR / "skab_config.json")
        elif dataset_name == "batadal":
            dataset_config = load_json(CONFIG_DIR / "batadal_config.json")
        else:
            raise ValueError(
                f"Unknown dataset_name='{dataset_name}'. "
                "Expected 'skab' or 'batadal'."
            )

        config = deep_update(config, dataset_config)

    if extra_config_path is not None:
        extra_config = load_json(extra_config_path)
        config = deep_update(config, extra_config)

    return config


def get_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.

    Example
    -------
    get_value(config, "training.batch_size")
    """
    current: Any = config

    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default

        current = current[key]

    return current


if __name__ == "__main__":
    loaded_config = load_config("skab")
    print(json.dumps(loaded_config, indent=4, ensure_ascii=False))