import cv2
import streamlit as st

from module.Module1.image_acquisition import (
    upload_single_image,
    upload_batch_images,
    camera_capture,
    save_processed_image,
)

from module.Module1.preprocessing import preprocess_image_with_steps

from module.object_detection import (
    detect_objects,
    detect_edges,
    apply_watershed,
)
from module.classification import predict_ripeness


st.set_page_config(
    page_title="Apple Ripeness System",
    layout="centered",
)

st.title("Apple Ripeness System")

CLASS_ORDER = ["20%", "40%", "60%", "80%", "100%", "Overripe"]


def use_white_segmented_background(image, foreground_mask):
    """Composite a segmented BGR image onto a white background."""
    white_background = image.copy()
    white_background[:] = 255
    return cv2.copyTo(image, foreground_mask, white_background)


def get_display_order(present_classes):
    """Return class labels ordered by ripeness progression."""
    ordered = [label for label in CLASS_ORDER if label in present_classes]
    extras = [label for label in present_classes if label not in CLASS_ORDER]
    return ordered + extras


def process_image(img, original_path):

    # =========================================================
    # MODULE 1 — PREPROCESSING
    # =========================================================

    try:
        steps = preprocess_image_with_steps(img)
        preprocessed_img = steps.final

        if steps.segmentation_success:
            preprocessed_img = use_white_segmented_background(
                preprocessed_img,
                steps.refined_mask,
            )

    except (ValueError, cv2.error) as exc:
        st.error(f"Image preprocessing failed: {exc}")
        return

    save_processed_image(
        preprocessed_img,
        original_path
    )

    # =========================================================
    # DISPLAY EVERY PREPROCESSING STAGE
    # =========================================================

    st.subheader("Complete Preprocessing Results")
    st.caption(
        "The stages below are shown in processing order. The final image is "
        "the input passed to apple detection and ripeness classification."
    )

    preprocessing_stages = [
        ("1. Original image", steps.original, "BGR"),
        ("2. Resized and letterboxed (224 × 224)", steps.resized, "BGR"),
        ("3. CLAHE contrast enhancement", steps.clahe, "BGR"),
        ("4. HSV apple-colour candidate mask", steps.hsv_candidate_mask, None),
        ("5. GrabCut foreground mask", steps.grabcut_mask, None),
        ("6. Refined mask after morphological closing", steps.refined_mask, None),
        ("7. Segmented foreground", steps.final, "BGR"),
        ("8. Final preprocessing output", preprocessed_img, "BGR"),
    ]

    for row_start in range(0, len(preprocessing_stages), 3):
        columns = st.columns(3)
        row_stages = preprocessing_stages[row_start:row_start + 3]

        for column, (caption, stage_image, channels) in zip(columns, row_stages):
            with column:
                image_options = {
                    "caption": caption,
                    "use_container_width": True,
                }
                if channels is None:
                    image_options["clamp"] = True
                else:
                    image_options["channels"] = channels
                st.image(stage_image, **image_options)

    foreground_percent = steps.foreground_ratio * 100.0
    if steps.segmentation_success:
        st.success(
            f"Background segmentation succeeded — foreground occupies "
            f"{foreground_percent:.1f}% of the processed image."
        )
    else:
        reason = steps.fallback_reason or "No reliable foreground was found."
        st.warning(
            f"Background segmentation used the fallback image "
            f"({foreground_percent:.1f}% foreground): {reason}"
        )

    # =========================================================
    # MODULE 2 — OBJECT DETECTION
    # =========================================================

    try:
        detected_apples, contour_img, binary_mask, status_message = (
            detect_objects(preprocessed_img)
        )

    except Exception as exc:
        st.error(f"Object detection failed: {exc}")
        return

    if status_message:
        if detected_apples:
            st.success(status_message)
        else:
            st.warning(status_message)

    if not detected_apples:
        st.image(
            contour_img,
            channels="BGR",
            caption="Detection Result"
        )
        return


# =========================================================
    # MODULE 3 — INDIVIDUAL APPLE DETECTIONS
    # =========================================================
    st.header("🍎 Individual Detection Results")
    
    # Create Table Headers (2 Columns)
    header_col1, header_col2 = st.columns([1, 4])
    with header_col1:
        st.markdown("**Apple ID**")
    with header_col2:
        st.markdown("**Detected Apple (Crop)**")
        
    st.divider()

    # Create one row per apple
    for apple in detected_apples:
        row_col1, row_col2 = st.columns([1, 4])
        
        x, y, w, h = apple["bbox"]
        apple_id = apple["id"]
        
        # Crop the specific apple from the original preprocessed image
        # Crop from `steps.resized` (letterboxed, but BEFORE CLAHE/GrabCut),
        # not from `preprocessed_img` (which already went through CLAHE + GrabCut
        # + white-background compositing). detect_objects() needs the fully
        # processed image to find the apple, but predict_ripeness() needs a
        # clean crop to run ITS OWN single-pass preprocessing on - otherwise the
        # apple gets CLAHE'd and GrabCut'd twice, which does not match training
        # and was producing a systematic bias toward Overripe.
        apple_roi = steps.resized[y:y+h, x:x+w]

        classification = {
            "label": "Unknown",
            "confidence": 0.0,
            "probabilities": {},
        }

        try:
            classification = predict_ripeness(apple_roi)
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            st.warning(f"Ripeness classification unavailable for Apple #{apple_id}: {exc}")

        with row_col1:
            st.subheader(f"#{apple_id}")
            st.caption(f"Size: {w} x {h} px")
            st.caption(f"Ripeness: {classification['label']}")
            st.caption(f"Confidence: {classification['confidence']:.2%}")
             
        with row_col2:
            st.image(apple_roi, channels="BGR", use_container_width=False)

            if classification.get("probabilities"):
                st.write("Class probabilities:")
                ordered_labels = get_display_order(classification["probabilities"].keys())
                for label in ordered_labels:
                    probability = classification["probabilities"][label]
                    st.caption(f"- {label}: {probability:.2%}")
             
        st.divider()

    # =========================================================
    # OTHER DETECTION METHODS
    # =========================================================

    edge_mask, edge_img = detect_edges(
        preprocessed_img
    )

    watershed_img, markers, watershed_mask = apply_watershed(
        preprocessed_img
    )

    # =========================================================
    # DISPLAY DETECTION RESULTS
    # =========================================================

    st.subheader("Object Detection Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            contour_img,
            channels="BGR",
            caption="Contour Detection",
            use_container_width=True
        )

    with col2:
        st.image(
            edge_img,
            channels="BGR",
            caption="Edge Detection",
            use_container_width=True
        )

    with col3:
        st.image(
            watershed_img,
            channels="BGR",
            caption="Watershed Segmentation",
            use_container_width=True
        )


# =============================================================
# INPUT METHOD
# =============================================================

option = st.selectbox(
    "Select Input Method",
    [
        "Single Upload",
        "Batch Upload",
        "Camera",
        "Live Camera",
    ]
)


# =============================================================
# SINGLE UPLOAD
# =============================================================

if option == "Single Upload":

    result = upload_single_image()

    if result is not None:
        img, original_path = result
        process_image(img, original_path)


# =============================================================
# BATCH UPLOAD
# =============================================================

elif option == "Batch Upload":

    images = upload_batch_images()

    for i, (img, original_path) in enumerate(
        images,
        start=1
    ):
        st.subheader(f"Image {i}")

        process_image(
            img,
            original_path
        )


# =============================================================
# CAMERA
# =============================================================

elif option == "Camera":

    result = camera_capture()

    if result is not None:
        img, original_path = result
        process_image(img, original_path)


# =============================================================
# LIVE CAMERA
# =============================================================

elif option == "Live Camera":

    try:
        from module.Module1.live_camera import live_camera_classification
    except ImportError:
        st.error(
            "Live Camera requires additional packages. "
            "Please install them with: pip install streamlit-webrtc av"
        )
    else:
        live_camera_classification()