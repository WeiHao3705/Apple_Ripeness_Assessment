from __future__ import annotations  # Defer type-hint evaluation for modern annotations.

from dataclasses import dataclass  # Generates the result container's boilerplate methods.

import numpy as np  # Supplies the ndarray type used for every stored image and mask.


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
