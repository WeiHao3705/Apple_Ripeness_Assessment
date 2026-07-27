import numpy as np
import cv2 as cv

# Object Detection
def detect_objects(segmented_img: np.ndarray):
    if segmented_img is None or segmented_img.size == 0:
        raise ValueError("Input image is empty.")
    
    if segmented_img.ndim != 3 or segmented_img.shape[2] != 3:
        raise ValueError("detect_objects expects a BGR image with 3 channels.")
    
    gray = cv.cvtColor(segmented_img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # The background is already black after segmentation, so a low threshold
    # is more stable than Otsu for preserving the apple region.
    _, binary_mask = cv.threshold(
        blurred,
        5,
        255,
        cv.THRESH_BINARY,
    )
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel)
    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(
        binary_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        no_object_image = segmented_img.copy()
        cv.putText(
            no_object_image,
            "No apple found in the image",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return None, no_object_image, binary_mask, "No apple found in the image"
    
    largest_contour = max(contours, key=cv.contourArea)

    contour_area = cv.contourArea(largest_contour)
    perimeter = cv.arcLength(largest_contour, True)
    x, y, w, h = cv.boundingRect(largest_contour)
    bounding_area = max(1, w * h)
    image_area = segmented_img.shape[0] * segmented_img.shape[1]

    circularity = 0.0
    if perimeter > 0:
        circularity = (4.0 * np.pi * contour_area) / (perimeter * perimeter)

    aspect_ratio = w / float(h) if h > 0 else 0.0
    extent = contour_area / float(bounding_area)
    solidity = contour_area / float(cv.contourArea(cv.convexHull(largest_contour)) or 1)

    roundness_ratio = 1.0
    if len(largest_contour) >= 5:
        (_, _), (axis_a, axis_b), _ = cv.fitEllipse(largest_contour)
        major = max(axis_a, axis_b)
        minor = min(axis_a, axis_b)
        if major > 0:
            roundness_ratio = minor / major

    contour_mask = np.zeros_like(gray, dtype=np.uint8)
    cv.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv.FILLED)

    texture_edges = cv.Canny(blurred, 60, 140)
    texture_edges = cv.bitwise_and(texture_edges, texture_edges, mask=contour_mask)
    edge_pixels = np.count_nonzero(texture_edges)
    object_pixels = max(1, np.count_nonzero(contour_mask))
    edge_density = edge_pixels / float(object_pixels)

    is_apple_like = (
        contour_area > 0.005 * image_area
        and circularity >= 0.30
        and 0.75 <= aspect_ratio <= 1.35
        and extent >= 0.25
        and solidity >= 0.82
        and roundness_ratio >= 0.72
        and edge_density <= 0.18
    )

    annotated_image = segmented_img.copy()
    if not is_apple_like:
        cv.putText(
            annotated_image,
            "No apple found in the image",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return None, annotated_image, binary_mask, "No apple found in the image"

    cv.drawContours(annotated_image, [largest_contour], -1, (0, 255, 0), 2)
    cv.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return largest_contour, annotated_image, binary_mask, "Apple found in the image"


# Edge detection
def detect_edges(segmented_img: np.ndarray):
    if segmented_img is None or segmented_img.size == 0:
        raise ValueError("Input image is empty.")

    if segmented_img.ndim != 3 or segmented_img.shape[2] != 3:
        raise ValueError("detect_edges expects a BGR image with 3 channels.")

    gray = cv.cvtColor(segmented_img, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)

    smoothed = cv.bilateralFilter(normalized, d=7, sigmaColor=50, sigmaSpace=50)

    nonzero_pixels = smoothed[smoothed > 0]
    if nonzero_pixels.size == 0:
        nonzero_pixels = smoothed.flatten()

    median_intensity = float(np.median(nonzero_pixels))
    lower = int(max(20, 0.66 * median_intensity))
    upper = int(min(220, 1.33 * median_intensity))

    edges = cv.Canny(smoothed, lower, upper)

    foreground_mask = np.where(gray > 5, 255, 0).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_OPEN, kernel)
    foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_CLOSE, kernel)

    edges = cv.bitwise_and(edges, edges, mask=foreground_mask)

    edge_overlay = segmented_img.copy()
    edge_overlay[edges > 0] = (0, 0, 255)

    return edges, edge_overlay


# Watershed segmentation
def apply_watershed(segmented_img: np.ndarray):
    if segmented_img is None or segmented_img.size == 0:
        raise ValueError("Input image is empty.")

    if segmented_img.ndim != 3 or segmented_img.shape[2] != 3:
        raise ValueError("apply_watershed expects a BGR image with 3 channels.")

    gray = cv.cvtColor(segmented_img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_image = segmented_img.copy()
    markers = cv.watershed(watershed_image, markers)
    watershed_image[markers == -1] = (0, 0, 255)

    return watershed_image, markers, binary