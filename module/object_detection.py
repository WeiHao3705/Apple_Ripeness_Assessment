import numpy as np
import cv2 as cv

# ==========================================
# 1. ADVANCED OBJECT DETECTION 
# (Integrated with Friend's Config Heuristics)
# ==========================================
def detect_objects(segmented_img: np.ndarray):
    if segmented_img is None or segmented_img.size == 0:
        raise ValueError("Input image is empty.")
    
    if segmented_img.ndim != 3 or segmented_img.shape[2] != 3:
        raise ValueError("detect_objects expects a BGR image with 3 channels.")
    
    image = np.ascontiguousarray(segmented_img, dtype=np.uint8)
    
    # 1. Candidate Mask 
    smoothed_image = cv.medianBlur(image, 5) # (Consider increasing to 15 if water droplets cause issues again)
    hsv = cv.cvtColor(smoothed_image, cv.COLOR_BGR2HSV)
    lower_bound = np.array([0, 40, 40])
    upper_bound = np.array([180, 255, 255])
    colour_mask = cv.inRange(hsv, lower_bound, upper_bound)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    colour_mask = cv.morphologyEx(colour_mask, cv.MORPH_OPEN, kernel, iterations=2)
    candidate_mask = cv.morphologyEx(colour_mask, cv.MORPH_CLOSE, kernel, iterations=2)

    # 2. Internal Watershed Separation
    sure_bg = cv.dilate(candidate_mask, kernel, iterations=3)
    distance = cv.distanceTransform(candidate_mask, cv.DIST_L2, 5)
    
    valid_contours = []
    image_area = image.shape[0] * image.shape[1]
    
    if distance.max() > 0:
        _, sure_fg = cv.threshold(distance, 0.8 * distance.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        
        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        watershed_image = image.copy()
        markers = cv.watershed(watershed_image, markers)
        
        for label in np.unique(markers):
            if label <= 1: 
                continue
            
            label_mask = np.uint8(markers == label) * 255
            contours, _ = cv.findContours(label_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv.contourArea)
                
                # Smooth the contour slightly to get rid of minor hand/leaf jaggedness
                epsilon = 0.005 * cv.arcLength(c, True)
                c = cv.approxPolyDP(c, epsilon, True)
                
                area = cv.contourArea(c)
                if not (0.005 * image_area <= area <= 0.70 * image_area):
                    continue
                    
                x, y, w, h = cv.boundingRect(c)
                aspect = w / float(h) if h > 0 else 0.0
                
                hull = cv.convexHull(c)
                hull_area = cv.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0.0
                
                perimeter = cv.arcLength(c, True)
                circularity = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
                
                # Adjusted thresholds to safely tolerate hands and stems
                if (0.30 <= aspect <= 1.60) and (solidity >= 0.60) and (circularity >= 0.30):
                    valid_contours.append(c)

    # --- NEW: relative-size filtering ---
    # Drop candidates that are much smaller than the largest candidate found.
    if valid_contours:
        areas = [cv.contourArea(c) for c in valid_contours]
        max_area = max(areas)
        # 0.25 threshold ensures we drop small leaf fragments while keeping real secondary apples
        size_keep = [c for c, a in zip(valid_contours, areas) if a >= 0.25 * max_area]
        valid_contours = size_keep

    # ❌ REMOVED THE MERGING LOOP THAT WAS EATING THE IMAGE ❌

    annotated_image = image.copy()

    if not valid_contours:
        cv.putText(annotated_image, "No apples found", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        return [], annotated_image, candidate_mask, "No apple found in the image"

    detected_apples = []

    # Annotate all valid apples
    for i, c in enumerate(valid_contours, start=1):
        x, y, w, h = cv.boundingRect(c)
        cv.drawContours(annotated_image, [c], -1, (0, 255, 0), 2)
        cv.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(annotated_image, f"Apple {i}", (x, max(20, y - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv.LINE_AA)

        # We must also calculate confidence to prevent the UI from crashing
        # Re-calculating stats for the dictionary
        area = cv.contourArea(c)
        perimeter = cv.arcLength(c, True)
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
        aspect = w / float(h) if h > 0 else 0.0
        hull_area = cv.contourArea(cv.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        detected_apples.append({
            "id": i,
            "bbox": (x, y, w, h),
            "contour": c,
            # Pass a baseline confidence score based on solidity so app.py doesn't throw a KeyError
            "confidence": min(solidity, 1.0),
            "individual_img": image.copy() 
        })

    total_apples = len(valid_contours)
    status_message = f"Apple detected! Found {total_apples} apple(s)"
    
    return detected_apples, annotated_image, candidate_mask, status_message


# ==========================================
# 2. EDGE DETECTION
# ==========================================
def detect_edges(segmented_img: np.ndarray):
    if segmented_img is None or segmented_img.size == 0:
        raise ValueError("Input image is empty.")

    if segmented_img.ndim != 3 or segmented_img.shape[2] != 3:
        raise ValueError("detect_edges expects a BGR image with 3 channels.")

    segmented_img = np.ascontiguousarray(segmented_img, dtype=np.uint8)
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


# ==========================================
# 3. WATERSHED SEGMENTATION (Comparison)
# ==========================================
def apply_watershed(segmented_img: np.ndarray):
    if segmented_img is None or segmented_img.size == 0:
        raise ValueError("Input image is empty.")

    if segmented_img.ndim != 3 or segmented_img.shape[2] != 3:
        raise ValueError("apply_watershed expects a BGR image with 3 channels.")

    segmented_img = np.ascontiguousarray(segmented_img, dtype=np.uint8)

    hsv = cv.cvtColor(segmented_img, cv.COLOR_BGR2HSV)
    lower_bound = np.array([0, 40, 40])
    upper_bound = np.array([180, 255, 255])
    binary = cv.inRange(hsv, lower_bound, upper_bound)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    
    _, sure_fg = cv.threshold(dist_transform, 0.8 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    total_item = len(np.unique(markers)) - 1
    
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_image = segmented_img.copy()
    markers = cv.watershed(watershed_image, markers)
    watershed_image[markers == -1] = (0, 0, 255) 
    
    return watershed_image, markers, binary