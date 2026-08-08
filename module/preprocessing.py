import cv2
import numpy as np

# 1. IMAGE RESIZING
def resize_image(image: np.ndarray) -> np.ndarray:
    """
    Resize the input image to 224 x 224 pixels.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

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
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    return resized_image

# 2. CLAHE CONTRAST ENHANCEMENT
def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to improve local contrast.

    CLAHE is applied only to the Lightness (L) channel
    in LAB colour space so that colour information is
    preserved.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

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

    # Convert BGR image to LAB colour space
    lab_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(
        lab_image
    )

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Apply CLAHE to Lightness channel
    enhanced_l = clahe.apply(
        l_channel
    )

    # Merge enhanced L channel with original A and B channels
    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel
        )
    )

    # Convert LAB back to BGR
    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image

# 3. MORPHOLOGICAL CLOSING
def apply_closing(mask: np.ndarray) -> np.ndarray:
    """
    Apply morphological closing to a binary mask.

    Closing = Dilation followed by Erosion.

    The purpose is to fill small holes and gaps inside
    the segmented apple region while maintaining the
    overall object shape.

    Parameters
    ----------
    mask : np.ndarray
        Binary foreground mask.

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

    # Elliptical kernel works well with rounded apple shapes
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    closed_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closed_mask

# 4. BACKGROUND SEGMENTATION
def segment_background(image: np.ndarray) -> np.ndarray:
    """
    Remove the background using GrabCut.

    Processing:
        1. Initialise GrabCut mask
        2. Define foreground region
        3. Run GrabCut
        4. Generate binary foreground mask
        5. Apply morphological closing
        6. Apply refined mask to image

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    Returns
    -------
    np.ndarray
        BGR image with background removed.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "segment_background expects a BGR image "
            "with 3 channels."
        )

    height, width = image.shape[:2]

    # Initialise GrabCut mask
    grabcut_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Define GrabCut rectangle
    # Assume the apple is mainly located inside the image,
    # while a small border around the image is background.
    inset_x = max(
        1,
        int(width * 0.08)
    )

    inset_y = max(
        1,
        int(height * 0.08)
    )

    rect_width = max(
        1,
        width - (2 * inset_x)
    )

    rect_height = max(
        1,
        height - (2 * inset_y)
    )

    rectangle = (
        inset_x,
        inset_y,
        rect_width,
        rect_height
    )

    # GrabCut internal models
    background_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    # Run GrabCut
    cv2.grabCut(
        image,
        grabcut_mask,
        rectangle,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    # Convert GrabCut output to binary mask
    # Foreground:
    # GC_FGD = definite foreground
    # GC_PR_FGD = probable foreground
    # Everything else becomes background.
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

    # Morphological Closing
    foreground_mask = apply_closing(
        foreground_mask
    )

    # Apply mask to image
    segmented_image = cv2.bitwise_and(
        image,
        image,
        mask=foreground_mask
    )

    return segmented_image

# COMPLETE MODULE 1 PREPROCESSING PIPELINE
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Complete preprocessing pipeline for Module 1.

    Flow:
        Original Image
              ↓
        Resize to 224 x 224
              ↓
        CLAHE
              ↓
        GrabCut Background Segmentation
              ↓
        Morphological Closing
              ↓
        Preprocessed Image

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.

    Returns
    -------
    np.ndarray
        Final preprocessed BGR image.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    # Step 1: Image Resizing
    resized_image = resize_image(
        image
    )

    # Step 2: CLAHE Enhancement
    clahe_image = apply_clahe(
        resized_image
    )

    # Step 3: GrabCut Background Segmentation
    # Morphological Closing is performed internally on
    # the GrabCut binary mask.
    segmented_image = segment_background(
        clahe_image
    )

    return segmented_image