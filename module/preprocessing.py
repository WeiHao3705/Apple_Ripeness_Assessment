import cv2
import numpy as np

def segment_background(image: np.ndarray) -> np.ndarray:
    """
    Remove the background and keep the apple as the dominant foreground object.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    Returns
    -------
    np.ndarray
        BGR image with the background removed and set to black.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("segment_background expects a BGR image with 3 channels.")

    height, width = image.shape[:2]
    if height < 20 or width < 20:
        return image.copy()

    mask = np.zeros((height, width), np.uint8)

    inset_x = max(1, int(width * 0.08))
    inset_y = max(1, int(height * 0.08))
    rect = (
        inset_x,
        inset_y,
        max(1, width - 2 * inset_x),
        max(1, height - 2 * inset_y),
    )

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return image.copy()

    foreground_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype("uint8")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        foreground_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return image.copy()

    largest_contour = max(contours, key=cv2.contourArea)
    cleaned_mask = np.zeros_like(foreground_mask)
    cv2.drawContours(cleaned_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    segmented = cv2.bitwise_and(image, image, mask=cleaned_mask)
    return segmented


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to improve image contrast.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    Returns
    -------
    np.ndarray
        CLAHE enhanced BGR image.
    """
    
    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    
    # Apply CLAHE to Lightness channel
    l = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge((l, a, b))
    
    # Convert back to BGR
    enhanced_image = cv2.cvtColor(
        enhanced_lab, 
        cv2.COLOR_LAB2BGR
    )
    
    return enhanced_image
    