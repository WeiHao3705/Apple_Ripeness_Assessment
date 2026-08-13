import cv2
import numpy as np


# ============================================================
# 1. IMAGE RESIZING
# ============================================================

def resize_image(
    image: np.ndarray,
    size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Resize an input image to a fixed resolution.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    size : tuple[int, int]
        Target image size.
        Default is (224, 224).

    Returns
    -------
    np.ndarray
        Resized BGR image.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "resize_image expects a BGR image with 3 channels."
        )

    resized_image = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )

    return resized_image


# ============================================================
# 2. CLAHE CONTRAST ENHANCEMENT
# ============================================================

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalisation
    (CLAHE) to the lightness channel of the image.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    clip_limit : float
        CLAHE contrast limiting value.

    tile_grid_size : tuple[int, int]
        Number of tiles used by CLAHE.

    Returns
    -------
    np.ndarray
        Contrast-enhanced BGR image.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "apply_clahe expects a BGR image with 3 channels."
        )

    # Convert BGR image to LAB colour space.
    lab_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    # Separate lightness and colour channels.
    l_channel, a_channel, b_channel = cv2.split(
        lab_image
    )

    # Create CLAHE object.
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    # Apply CLAHE only to the lightness channel.
    enhanced_l = clahe.apply(
        l_channel
    )

    # Merge the enhanced lightness channel
    # with the original colour channels.
    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel
        )
    )

    # Convert back to BGR colour space.
    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image


# ============================================================
# 3. GRABCUT BACKGROUND SEGMENTATION
# ============================================================

def grabcut_segmentation(
    image: np.ndarray,
    inset_ratio: float = 0.08,
    iterations: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform GrabCut background segmentation.

    GrabCut separates the likely foreground region from
    the surrounding background.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    inset_ratio : float
        Percentage of the image border excluded from the
        initial GrabCut rectangle.

    iterations : int
        Number of GrabCut iterations.

    Returns
    -------
    segmented_image : np.ndarray
        BGR image containing the retained foreground.

    foreground_mask : np.ndarray
        Binary mask where:
        255 = foreground
        0   = background
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "grabcut_segmentation expects a BGR image "
            "with 3 channels."
        )

    height, width = image.shape[:2]

    # Initialise GrabCut mask.
    grabcut_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Calculate margins around the image.
    inset_x = max(
        1,
        int(width * inset_ratio)
    )

    inset_y = max(
        1,
        int(height * inset_ratio)
    )

    rectangle_width = max(
        1,
        width - (2 * inset_x)
    )

    rectangle_height = max(
        1,
        height - (2 * inset_y)
    )

    # Initial GrabCut rectangle.
    rectangle = (
        inset_x,
        inset_y,
        rectangle_width,
        rectangle_height
    )

    # Models required internally by GrabCut.
    background_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    # Perform GrabCut.
    cv2.grabCut(
        image,
        grabcut_mask,
        rectangle,
        background_model,
        foreground_model,
        iterations,
        cv2.GC_INIT_WITH_RECT
    )

    # Convert GrabCut labels into a binary mask.
    foreground_mask = np.where(
        (
            grabcut_mask == cv2.GC_FGD
        )
        |
        (
            grabcut_mask == cv2.GC_PR_FGD
        ),
        255,
        0
    ).astype(np.uint8)

    # Apply mask to the image.
    segmented_image = cv2.bitwise_and(
        image,
        image,
        mask=foreground_mask
    )

    return segmented_image, foreground_mask


def segment_background(image: np.ndarray) -> np.ndarray:
    """Return a background-segmented image for legacy callers."""

    segmented_image, foreground_mask = grabcut_segmentation(image)
    closed_mask = apply_closing(foreground_mask)
    return apply_mask(image, closed_mask)


# ============================================================
# 4. MORPHOLOGICAL CLOSING
# ============================================================

def apply_closing(
    mask: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5)
) -> np.ndarray:
    """
    Apply morphological closing to a binary mask.

    Closing consists of:

    Dilation
        followed by
    Erosion

    It is useful for filling small holes and connecting
    small gaps inside segmented foreground regions.

    Parameters
    ----------
    mask : np.ndarray
        Binary foreground mask.

    kernel_size : tuple[int, int]
        Size of the elliptical structuring element.

    Returns
    -------
    np.ndarray
        Refined binary mask.
    """

    if mask is None or mask.size == 0:
        raise ValueError("Input mask is empty.")

    if mask.ndim != 2:
        raise ValueError(
            "apply_closing expects a single-channel binary mask."
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        kernel_size
    )

    closed_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closed_mask


# ============================================================
# 5. APPLY MASK TO IMAGE
# ============================================================

def apply_mask(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Apply a binary mask to a BGR image.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    mask : np.ndarray
        Binary mask.

    Returns
    -------
    np.ndarray
        Masked BGR image.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if mask is None or mask.size == 0:
        raise ValueError("Input mask is empty.")

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "Image and mask dimensions must match."
        )

    masked_image = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return masked_image


# ============================================================
# 6. COMPLETE MODULE 1 PREPROCESSING PIPELINE
# ============================================================

def preprocess_image(
    image: np.ndarray
) -> np.ndarray:
    """
    Perform the complete Module 1 preprocessing pipeline.

    Pipeline
    --------
    1. Resize image to 224 x 224
    2. Apply CLAHE contrast enhancement
    3. Perform GrabCut background segmentation
    4. Apply morphological closing
    5. Apply the refined mask to the CLAHE image

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.

    Returns
    -------
    np.ndarray
        Final preprocessed BGR image.
    """

    # Step 1: Resize image.
    resized_image = resize_image(
        image
    )

    # Step 2: Apply CLAHE.
    clahe_image = apply_clahe(
        resized_image
    )

    # Step 3: Perform GrabCut segmentation.
    _, foreground_mask = grabcut_segmentation(
        clahe_image
    )

    # Step 4: Refine the mask using morphological closing.
    closed_mask = apply_closing(
        foreground_mask
    )

    # Step 5: Apply final refined mask.
    final_image = apply_mask(
        clahe_image,
        closed_mask
    )

    return final_image


# ============================================================
# 7. COMPLETE PIPELINE WITH INTERMEDIATE RESULTS
# ============================================================

def preprocess_image_with_steps(
    image: np.ndarray
) -> dict:
    """
    Perform the preprocessing pipeline and return all
    intermediate processing results.

    This function is useful for:
    - Streamlit visualisation
    - Testing
    - Demonstration
    - Assignment screenshots

    Returns
    -------
    dict
        Dictionary containing all processing stages.
    """

    # Step 1
    resized_image = resize_image(
        image
    )

    # Step 2
    clahe_image = apply_clahe(
        resized_image
    )

    # Step 3
    segmented_image, grabcut_mask = grabcut_segmentation(
        clahe_image
    )

    # Step 4
    closed_mask = apply_closing(
        grabcut_mask
    )

    # Step 5
    final_image = apply_mask(
        clahe_image,
        closed_mask
    )

    return {
        "original": image,
        "resized": resized_image,
        "clahe": clahe_image,
        "grabcut": segmented_image,
        "grabcut_mask": grabcut_mask,
        "closed_mask": closed_mask,
        "final": final_image,
    }
