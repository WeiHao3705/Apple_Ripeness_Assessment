from __future__ import annotations  # Defer type-hint evaluation for modern annotations.

import cv2  # OpenCV supplies colour conversion, resizing, masks, and GrabCut.
import numpy as np  # NumPy stores images as arrays and creates/manipulates masks.

from .config import PreprocessingConfig  # Central settings for the pipeline.
from .model import PreprocessingResult  # Structured container for every output step.


# ============================================================
# Image Validation
# ============================================================

def _validate_bgr(image: np.ndarray) -> None:
    """
    Validate that the input is a non-empty 3-channel BGR image.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a BGR image with three channels."
        )

# Step 1 - Image Resizing
def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize an image to fit a centered, white target-size canvas."""

    _validate_bgr(image)

    target_width, target_height = target_size

    height, width = image.shape[:2]

    scale = min(
        target_width / float(width),
        target_height / float(height),
    )

    resized_width = max(
        1,
        int(round(width * scale))
    )

    resized_height = max(
        1,
        int(round(height * scale))
    )

    if scale < 1.0:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    # OpenCV performs the actual resampling using the interpolation selected above.
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    # Create 224 x 224 white canvas
    # NumPy creates a three-channel array filled with 255 (white).
    canvas = np.full(
        (
            target_height,
            target_width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    x_offset = (
        target_width - resized_width
    ) // 2

    y_offset = (
        target_height - resized_height
    ) // 2

    canvas[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized

    return canvas


# Step 2 - CLAHE
def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Improve local contrast by applying CLAHE to the LAB lightness channel."""

    _validate_bgr(image)

    # LAB separates brightness from colour, allowing contrast enhancement without
    # directly changing the apple's red, green, or yellow colour information.
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    # OpenCV separates the LAB image so CLAHE affects only the lightness channel.
    lightness, channel_a, channel_b = cv2.split(
        lab
    )

    # CLAHE enhances small image regions while limiting amplified sensor noise.
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=grid_size,
    )

    enhanced_lightness = clahe.apply(
        lightness
    )

    # Recombine the enhanced lightness with the untouched colour channels.
    enhanced_lab = cv2.merge(
        (
            enhanced_lightness,
            channel_a,
            channel_b,
        )
    )

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

# Morphological Operations
def apply_opening(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Remove small isolated foreground noise.
    """

    if mask is None or mask.size == 0:
        raise ValueError("Input mask is empty.")

    if mask.ndim != 2:
        raise ValueError(
            "Opening expects a single-channel binary mask."
        )

    # An elliptical OpenCV kernel better matches rounded apple regions than a square.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    # MORPH_OPEN erodes and then dilates, removing small white specks.
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )


def apply_closing(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Fill small gaps and holes in the foreground mask.
    """

    if mask is None or mask.size == 0:
        raise ValueError("Input mask is empty.")

    if mask.ndim != 2:
        raise ValueError(
            "Closing expects a single-channel binary mask."
        )

    # A slightly larger ellipse closes small breaks within the apple region.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    # MORPH_CLOSE dilates and then erodes, filling narrow gaps and holes.
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

# Step 3A - HSV Apple Candidate Mask
def create_apple_candidate_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Create an initial apple candidate mask using red,
    green and yellow HSV colour ranges.
    """

    _validate_bgr(image)

    # HSV makes colour-range selection more direct than raw BGR channel values.
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    # Red
    # inRange returns 255 for pixels inside the lower red hue range and 0 otherwise.
    red_mask_1 = cv2.inRange(
        hsv,
        np.array(
            [0, 70, 40],
            dtype=np.uint8,
        ),
        np.array(
            [12, 255, 255],
            dtype=np.uint8,
        ),
    )

    # Red wraps around the HSV hue scale, so a second upper range is required.
    red_mask_2 = cv2.inRange(
        hsv,
        np.array(
            [165, 70, 40],
            dtype=np.uint8,
        ),
        np.array(
            [179, 255, 255],
            dtype=np.uint8,
        ),
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2,
    )

    # Green
    # Build a binary mask for green pixels that may represent unripe apples.
    green_mask = cv2.inRange(
        hsv,
        np.array(
            [30, 45, 35],
            dtype=np.uint8,
        ),
        np.array(
            [90, 255, 255],
            dtype=np.uint8,
        ),
    )

    # Yellow
    # Build a binary mask for yellow pixels that may represent ripening apples.
    yellow_mask = cv2.inRange(
        hsv,
        np.array(
            [15, 60, 60],
            dtype=np.uint8,
        ),
        np.array(
            [38, 255, 255],
            dtype=np.uint8,
        ),
    )

    # Combine masks
    candidate_mask = cv2.bitwise_or(
        red_mask,
        green_mask,
    )

    candidate_mask = cv2.bitwise_or(
        candidate_mask,
        yellow_mask,
    )

    # Remove isolated noise
    candidate_mask = apply_opening(
        candidate_mask
    )

    # Fill small gaps
    candidate_mask = apply_closing(
        candidate_mask
    )

    return candidate_mask


# Compatibility Function for Module 2
def create_apple_colour_mask(
    image: np.ndarray,
) -> np.ndarray:
    """Return the apple-colour mask through the Module 2 compatibility API."""

    return create_apple_candidate_mask(
        image
    )


# Compatibility Mask Refinement
def refine_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Refine a binary mask for use by Module 2.
    """

    if (
        mask is None
        or mask.size == 0
        or mask.ndim != 2
    ):
        raise ValueError(
            "Expected a non-empty single-channel mask."
        )

    if kernel_size == 3:
        return apply_opening(mask)

    if kernel_size == 5:
        opened = apply_opening(mask)

        return apply_closing(
            opened
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    return cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
    )

# Step 3B - GrabCut Background Segmentation
def segment_background_with_steps(
    image: np.ndarray,
    iterations: int = 5,
    min_foreground_ratio: float = 0.01,
) -> dict:
    """
    Segment the apple using an HSV-guided GrabCut mask.
    """

    _validate_bgr(image)

    height, width = image.shape[:2]

    candidate_mask = create_apple_candidate_mask(
        image
    )

    empty_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    # Check candidate availability
    candidate_pixels = cv2.countNonZero(
        candidate_mask
    )

    if candidate_pixels == 0:
        return {
            "candidate_mask": candidate_mask,
            "grabcut_mask": empty_mask,
            "closed_mask": empty_mask,
            "segmented": image.copy(),
            "segmentation_success": False,
            "foreground_ratio": 0.0,
            "fallback_reason": (
                "No red, green or yellow apple "
                "candidate was detected."
            ),
        }

    # Initialise GrabCut labels
    grabcut_labels = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8,
    )

    grabcut_labels[
        candidate_mask > 0
    ] = cv2.GC_PR_FGD

    # Force image border as background
    border = max(
        1,
        int(min(height, width) * 0.01),
    )

    grabcut_labels[
        :border,
        :
    ] = cv2.GC_BGD

    grabcut_labels[
        -border:,
        :
    ] = cv2.GC_BGD

    grabcut_labels[
        :,
        :border
    ] = cv2.GC_BGD

    grabcut_labels[
        :,
        -border:
    ] = cv2.GC_BGD

    # GrabCut models
    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    fallback_reason = None

    # Run GrabCut safely
    try:
        # GrabCut uses the colour mask as foreground guidance and learns statistical
        # foreground/background colour models over the requested iterations.
        cv2.grabCut(
            image,
            grabcut_labels,
            None,
            background_model,
            foreground_model,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )

        # NumPy converts GrabCut's four label types into a standard 0/255 mask.
        grabcut_mask = np.where(
            (
                grabcut_labels == cv2.GC_FGD
            )
            |
            (
                grabcut_labels == cv2.GC_PR_FGD
            ),
            255,
            0,
        ).astype(np.uint8)

    except cv2.error as exc:
        grabcut_mask = empty_mask.copy()

        fallback_reason = (
            f"GrabCut could not converge: {exc}"
        )

    # Morphological refinement
    grabcut_mask = apply_opening(
        grabcut_mask
    )

    closed_mask = apply_closing(
        grabcut_mask
    )

    # Segmentation validation
    foreground_pixels = cv2.countNonZero(
        closed_mask
    )

    total_pixels = height * width

    foreground_ratio = (
        float(foreground_pixels)
        / float(total_pixels)
    )

    segmentation_success = (
        foreground_ratio
        >= min_foreground_ratio
    )

    if segmentation_success:

        # Keep source pixels only where the final binary mask marks foreground.
        segmented_image = cv2.bitwise_and(
            image,
            image,
            mask=closed_mask,
        )

    else:

        segmented_image = image.copy()

        if fallback_reason is None:
            fallback_reason = (
                "Detected foreground was below "
                "the minimum reliable ratio. "
                "The CLAHE image was preserved."
            )

    return {
        "candidate_mask": candidate_mask,
        "grabcut_mask": grabcut_mask,
        "closed_mask": closed_mask,
        "segmented": segmented_image,
        "segmentation_success": segmentation_success,
        "foreground_ratio": foreground_ratio,
        "fallback_reason": fallback_reason,
    }


# Simple Segmentation Function
def segment_background(
    image: np.ndarray,
) -> np.ndarray:
    """Segment the apple from the background and return only the final image."""

    result = segment_background_with_steps(
        image
    )

    return result["segmented"]


# Complete Preprocessing Pipeline
def preprocess_image_with_steps(
    image: np.ndarray,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """
    Run the complete Module 1 preprocessing pipeline.

    Flow:
    Original
        -> Resize
        -> CLAHE
        -> HSV candidate mask
        -> GrabCut
        -> Morphological refinement
        -> Final segmented image
    """

    if config is None:
        config = PreprocessingConfig()

    _validate_bgr(image)

    # Step 1
    resized = resize_image(
        image,
        config.target_size,
    )

    # Step 2
    enhanced = apply_clahe(
        resized,
        clip_limit=config.clahe_clip_limit,
        grid_size=config.clahe_grid_size,
    )

    # Step 3
    segmentation = segment_background_with_steps(
        enhanced,
        iterations=config.grabcut_iterations,
        min_foreground_ratio=(
            config.min_foreground_ratio
        ),
    )

    return PreprocessingResult(
        original=image.copy(),

        resized=resized,

        clahe=enhanced,

        hsv_candidate_mask=segmentation[
            "candidate_mask"
        ],

        grabcut_mask=segmentation[
            "grabcut_mask"
        ],

        refined_mask=segmentation[
            "closed_mask"
        ],

        final=segmentation[
            "segmented"
        ],

        segmentation_success=bool(
            segmentation[
                "segmentation_success"
            ]
        ),

        foreground_ratio=float(
            segmentation[
                "foreground_ratio"
            ]
        ),

        fallback_reason=segmentation[
            "fallback_reason"
        ],
    )


# Final Image Only
def preprocess_image(
    image: np.ndarray,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Run the preprocessing pipeline and return only its final image."""

    result = preprocess_image_with_steps(
        image,
        config,
    )

    return result.final
