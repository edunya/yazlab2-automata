"""
Timer utilities.

This module measures elapsed time for training, inference and
other experiment steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


def format_seconds(seconds: float) -> str:
    """
    Format seconds as a readable string.
    """
    if seconds < 60:
        return f"{seconds:.4f} sec"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    return f"{minutes} min {remaining_seconds:.2f} sec"


@dataclass
class Timer:
    """
    Context manager for measuring elapsed time.

    Example
    -------
    with Timer("training") as timer:
        train_model()

    print(timer.elapsed)
    """

    name: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    elapsed: Optional[float] = None
    verbose: bool = False

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.end_time = time.perf_counter()

        if self.start_time is None:
            self.elapsed = None
        else:
            self.elapsed = self.end_time - self.start_time

        if self.verbose and self.name is not None and self.elapsed is not None:
            print(f"[TIMER] {self.name}: {format_seconds(self.elapsed)}")