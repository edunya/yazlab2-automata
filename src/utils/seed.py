"""
Random seed utilities.

This module makes experiments more reproducible across Python,
NumPy and PyTorch.
"""

from __future__ import annotations

import os
import random
from typing import Dict, Any

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for Python, NumPy and PyTorch.

    Parameters
    ----------
    seed:
        Seed value.

    deterministic:
        If True, PyTorch uses deterministic behavior when possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    except ImportError:
        pass


def set_seed_from_config(config: Dict[str, Any], seed_index: int = 0) -> int:
    """
    Read one seed value from config and apply it.

    Parameters
    ----------
    config:
        Project configuration dictionary.

    seed_index:
        Index of the seed in config["random_seeds"].

    Returns
    -------
    int
        Applied seed value.
    """
    seeds = config.get("random_seeds", [])

    if not seeds:
        raise ValueError("No random seeds found in config.")

    if seed_index < 0 or seed_index >= len(seeds):
        raise IndexError(
            f"seed_index={seed_index} is out of range. "
            f"Available seed count: {len(seeds)}"
        )

    selected_seed = int(seeds[seed_index])
    set_seed(selected_seed)

    return selected_seed