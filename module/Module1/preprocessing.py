from __future__ import annotations  # Defer type-hint evaluation for modern annotations.

import cv2  # OpenCV supplies colour conversion, resizing, masks, and GrabCut.
import numpy as np  # NumPy stores images as arrays and creates/manipulates masks.

from .config import PreprocessingConfig  # Central settings for the pipeline.
from .model import PreprocessingResult  # Structured container for every output step.

# Images in this module follow OpenCV's BGR channel order. Binary masks use a
# single uint8 channel where 0 means background and 255 means foreground.


# ============================================================
# Image Validation
# ============================================================

def _validate_bgr(image: np.ndarray) -> None:
    """
    Validate that the input is a non-empty 3-channel BGR image.

    Args:
        image: NumPy array to validate before OpenCV processing.

    Raises:
        ValueError: If the image is empty or is not a three-channel image.
    """

    # Checking both None and size prevents later OpenCV calls from failing with
    # less helpful low-level assertion errors.
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    # A colour image must have dimensions (height, width, channels), and this
    # pipeline specifically expects the three BGR channels used by OpenCV.
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a BGR image with three channels."
        )

# Step 1 - Image Resizing
def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Resize an image to fit a centered, white target-size canvas.

    The original aspect ratio is preserved, so the apple is not stretched.
    Empty space around the resized image is padded in white.

    Args:
        image: Source image in OpenCV BGR format.
        target_size: Required output width and height in pixels.

    Returns:
        A BGR image whose dimensions exactly match ``target_size``.
    """

    # Validate early because image.shape and cv2.resize require a real BGR array.
    _validate_bgr(image)

    # target_size follows OpenCV's conventional (width, height) ordering.
    target_width, target_height = target_size

    # NumPy image shapes use the opposite spatial order: (height, width, channels).
    height, width = image.shape[:2]

    # Use the smaller ratio so both resized dimensions fit inside the canvas.
    # This is the calculation that preserves the input aspect ratio.
    scale = min(
        target_width / float(width),
        target_height / float(height),
    )

    # Rounding converts the scaled dimensions to whole pixels. max(1, ...)
    # prevents an extremely narrow image from becoming zero pixels wide.
    resized_width = max(
        1,
        int(round(width * scale))
    )

    resized_height = max(
        1,
        int(round(height * scale))
    )

    # INTER_AREA generally retains detail when reducing an image; 
    # INTER_LINEAR gives smoother results when an image must be enlarged.
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

    # Half of the unused width and height places the resized image in the center.
    x_offset = (
        target_width - resized_width
    ) // 2

    y_offset = (
        target_height - resized_height
    ) // 2

    # NumPy slicing replaces only the centered rectangular area of the canvas.
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
    """
    Improve local contrast by applying CLAHE to the LAB lightness channel.

    Args:
        image: Source image in OpenCV BGR format.
        clip_limit: Maximum contrast amplification applied within each tile.
        grid_size: Number of local tiles across the image width and height.

    Returns:
        A contrast-enhanced BGR image with the same size as the input.
    """

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

    # apply() redistributes brightness values independently within the local tiles.
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

    # Convert back to BGR because every later function expects OpenCV BGR input.
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

    Args:
        mask: Single-channel binary mask containing values 0 and 255.

    Returns:
        A cleaned binary mask with small foreground specks removed.
    """

    # Morphological operations require actual pixels and cannot process None.
    if mask is None or mask.size == 0:
        raise ValueError("Input mask is empty.")

    # Two dimensions confirm that this is one mask channel rather than a BGR image.
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

    Args:
        mask: Single-channel binary mask containing values 0 and 255.

    Returns:
        A binary mask with nearby foreground areas joined and holes filled.
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

    Args:
        image: Source image in OpenCV BGR format.

    Returns:
        A binary mask whose white pixels have likely apple colours.
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

    # bitwise_or keeps a pixel when it belongs to either red hue interval.
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
    # Union the independent colour masks so any supported apple colour is kept.
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


# Step 3B - GrabCut Background Segmentation
def segment_background_with_steps(
    image: np.ndarray,
    iterations: int = 5,
    min_foreground_ratio: float = 0.01,
) -> dict:
    """
    Segment the apple using an HSV-guided GrabCut mask.

    Args:
        image: Contrast-enhanced source image in BGR format.
        iterations: Number of GrabCut model-refinement passes.
        min_foreground_ratio: Smallest accepted foreground fraction.

    Returns:
        A dictionary containing intermediate masks, the segmented image,
        validation measurements, and any fallback explanation.
    """

    _validate_bgr(image)

    # Store spatial dimensions for masks, border calculations, and area checks.
    height, width = image.shape[:2]

    # The HSV result supplies GrabCut with likely foreground pixels.
    candidate_mask = create_apple_candidate_mask(
        image
    )

    # Prepare a correctly sized all-background mask for safe fallback results.
    empty_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    # Check candidate availability
    # countNonZero counts white/foreground pixels without scanning in Python.
    candidate_pixels = cv2.countNonZero(
        candidate_mask
    )

    # GrabCut cannot learn foreground appearance when the colour mask is empty.
    # Return the original image instead of raising an error or returning black.
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
    # Start every pixel as probable background. GrabCut distinguishes definite
    # background/foreground from probable background/foreground using four labels.
    grabcut_labels = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8,
    )

    # Pixels selected by HSV become probable foreground seeds for GrabCut.
    grabcut_labels[
        candidate_mask > 0
    ] = cv2.GC_PR_FGD

    # Force image border as background
    # Treat the outermost 1% as definite background. max(1, ...) ensures at
    # least one border pixel is available even for small images.
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
    # OpenCV requires two temporary 1x65 float arrays for GrabCut's internal
    # Gaussian mixture models. grabCut fills these arrays during processing.
    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    # None indicates that processing has not encountered a recoverable failure.
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

    # Convert an OpenCV convergence/argument failure into a controlled fallback.
    except cv2.error as exc:
        grabcut_mask = empty_mask.copy()

        fallback_reason = (
            f"GrabCut could not converge: {exc}"
        )

    # Morphological refinement
    # Remove isolated points that GrabCut may have misclassified as foreground.
    grabcut_mask = apply_opening(
        grabcut_mask
    )

    # Fill small holes to produce a more continuous apple silhouette.
    closed_mask = apply_closing(
        grabcut_mask
    )

    # Segmentation validation
    # Measure the refined mask rather than the noisier raw GrabCut output.
    foreground_pixels = cv2.countNonZero(
        closed_mask
    )

    # Total image area is used to normalize foreground size across resolutions.
    total_pixels = height * width

    # Convert both values to float so the result is an explicit proportion from 0 to 1.
    foreground_ratio = (
        float(foreground_pixels)
        / float(total_pixels)
    )

    # Very small regions are likely noise, so reject them using the configured limit.
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

        # Preserve the enhanced input when segmentation is not reliable. copy()
        # prevents callers from accidentally modifying the original array later.
        segmented_image = image.copy()

        if fallback_reason is None:
            fallback_reason = (
                "Detected foreground was below "
                "the minimum reliable ratio. "
                "The CLAHE image was preserved."
            )

    # Expose intermediate outputs so the UI and later modules can inspect each stage.
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
    """
    Segment the apple from the background and return only the final image.

    This convenience wrapper hides intermediate diagnostic masks when a caller
    only needs the image that proceeds to later feature-extraction stages.
    """

    # Reuse the detailed function so both public APIs follow identical logic.
    result = segment_background_with_steps(
        image
    )

    # Select only the final segmented/fallback image from the detailed dictionary.
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

    Args:
        image: Original apple image in OpenCV BGR format.
        config: Optional pipeline settings; defaults are used when omitted.

    Returns:
        A ``PreprocessingResult`` containing the original and every pipeline stage.
    """

    # Creating the default here avoids sharing mutable configuration state.
    if config is None:
        config = PreprocessingConfig()

    # Validate once at the public pipeline boundary before starting any work.
    _validate_bgr(image)

    # Step 1: standardize model input dimensions without distorting the apple.
    resized = resize_image(
        image,
        config.target_size,
    )

    # Step 2: reduce uneven-lighting effects and reveal local surface detail.
    enhanced = apply_clahe(
        resized,
        clip_limit=config.clahe_clip_limit,
        grid_size=config.clahe_grid_size,
    )

    # Step 3: locate apple colours, separate foreground, and refine the mask.
    segmentation = segment_background_with_steps(
        enhanced,
        iterations=config.grabcut_iterations,
        min_foreground_ratio=(
            config.min_foreground_ratio
        ),
    )

    # Package named outputs in a dataclass instead of returning an ambiguous tuple.
    return PreprocessingResult(
        # Keep a defensive copy of the caller's untouched input.
        original=image.copy(),

        # Fixed-size, aspect-preserving image from Step 1.
        resized=resized,

        # Locally contrast-enhanced image from Step 2.
        clahe=enhanced,

        # Initial HSV colour selection used to seed GrabCut.
        hsv_candidate_mask=segmentation[
            "candidate_mask"
        ],

        # Foreground mask produced directly by GrabCut and opening.
        grabcut_mask=segmentation[
            "grabcut_mask"
        ],

        # Closed mask containing the final continuous apple region.
        refined_mask=segmentation[
            "closed_mask"
        ],

        # Segmented apple, or the enhanced image when segmentation was unreliable.
        final=segmentation[
            "segmented"
        ],

        # Explicit conversions keep the dataclass fields as normal Python scalars.
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
