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


st.title("Apple Ripeness System")

CLASS_ORDER = ["20%", "40%", "60%", "80%", "100%", "Overripe"]


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

    except (ValueError, cv2.error) as exc:
        st.error(f"Image preprocessing failed: {exc}")
        return

    save_processed_image(
        preprocessed_img,
        original_path
    )

    # =========================================================
    # DISPLAY PREPROCESSING RESULTS
    # =========================================================

    st.subheader("Preprocessing Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            steps.resized,
            channels="BGR",
            caption="Resized (224 × 224)",
            use_container_width=True
        )

    with col2:
        st.image(
            steps.clahe,
            channels="BGR",
            caption="CLAHE Enhanced",
            use_container_width=True
        )

    with col3:
        st.image(
            preprocessed_img,
            channels="BGR",
            caption="Final Preprocessed Image",
            use_container_width=True
        )

    # =========================================================
    # SHOW GRABCUT + CLOSING MASK
    # =========================================================

    with st.expander("Show preprocessing masks"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                steps.hsv_candidate_mask,
                caption="Apple Colour Candidate Mask",
                clamp=True,
                use_container_width=True
            )

        with col2:
            st.image(
                steps.grabcut_mask,
                caption="GrabCut Mask",
                clamp=True,
                use_container_width=True
            )

        with col3:
            st.image(
                steps.refined_mask,
                caption="Mask After Closing",
                clamp=True,
                use_container_width=True
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
        apple_roi = preprocessed_img[y:y+h, x:x+w]

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
        "Camera"
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