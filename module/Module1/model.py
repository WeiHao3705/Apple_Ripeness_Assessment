from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreprocessingResult:

    original: np.ndarray
    resized: np.ndarray
    clahe: np.ndarray
    hsv_candidate_mask: np.ndarray
    grabcut_mask: np.ndarray
    refined_mask: np.ndarray
    final: np.ndarray
    segmentation_success: bool
    foreground_ratio: float
    fallback_reason: str | None = None