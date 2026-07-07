import cv2
import numpy as np

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
    