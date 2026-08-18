import numpy as np
import cv2 as cv

# ==========================================
# 1. ADVANCED OBJECT DETECTION
# (Uses Watershed internally to split touching apples)
# ==========================================
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
 
    smoothed_image = cv.medianBlur(image, 5)
    
    # 1. Candidate Mask 
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
        # Using 0.4 peak ratio for robust internal separation
        _, sure_fg = cv.threshold(distance, 0.4 * distance.max(), 255, 0)
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
                
                # Adjusted thresholds to safely tolerate hands and stems while rejecting backgrounds
                if (0.30 <= aspect <= 1.60) and (solidity >= 0.60) and (circularity >= 0.30):
                    valid_contours.append(c)
 
    # --- NEW: relative-size filtering ---
    # Drop candidates that are much smaller than the largest candidate found.
    # A real apple dominates the frame; stray leaf/bush/background fragments
    # that slipped through the color mask are typically much smaller.
    if valid_contours:
        areas = [cv.contourArea(c) for c in valid_contours]
        max_area = max(areas)
        size_keep = [c for c, a in zip(valid_contours, areas) if a >= 0.25 * max_area]
        valid_contours = size_keep
 
    # --- NEW: merge overlapping fragments ---
    # Watershed can over-split a single apple (e.g. where a leaf/stem creates
    # a "waist") into two adjacent pieces. Merge contours whose bounding
    # boxes overlap significantly into one contour via convex hull.
    def _bbox_iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xa, ya = max(x1, x2), max(y1, y2)
        xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0
 
    merged = True
    while merged and len(valid_contours) > 1:
        merged = False
        for i in range(len(valid_contours)):
            for j in range(i + 1, len(valid_contours)):
                bi = cv.boundingRect(valid_contours[i])
                bj = cv.boundingRect(valid_contours[j])
                if _bbox_iou(bi, bj) > 0.15:
                    combined_pts = np.vstack([valid_contours[i], valid_contours[j]])
                    new_c = cv.convexHull(combined_pts)
                    new_list = [c for k, c in enumerate(valid_contours) if k not in (i, j)]
                    new_list.append(new_c)
                    valid_contours = new_list
                    merged = True
                    break
            if merged:
                break
 
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
 
        detected_apples.append({
            "id": i,
            "bbox": (x, y, w, h),
            "contour": c
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
    
    _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
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