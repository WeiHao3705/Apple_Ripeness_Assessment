from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:

    target_size: tuple[int, int] = (224, 224)

    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)

    grabcut_iterations: int = 5

    min_foreground_ratio: float = 0.01