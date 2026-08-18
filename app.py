import cv2
import streamlit as st

from module.image_acquisition import (
    upload_single_image,
    upload_batch_images,
    camera_capture,
    save_processed_image,
)

from module.preprocessing import preprocess_image_with_steps

from module.object_detection import (
    detect_objects,
    detect_edges,
    apply_watershed,
)


st.title("Apple Ripeness System")


def process_image(img, original_path):

    # =========================================================
    # MODULE 1 — PREPROCESSING
    # =========================================================

    try:
        steps = preprocess_image_with_steps(img)
        preprocessed_img = steps["final"]

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
            steps["resized"],
            channels="BGR",
            caption="Resized (224 × 224)",
            use_container_width=True
        )

    with col2:
        st.image(
            steps["clahe"],
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
                steps["candidate_mask"],
                caption="Apple Colour Candidate Mask",
                clamp=True,
                use_container_width=True
            )

        with col2:
            st.image(
                steps["grabcut_mask"],
                caption="GrabCut Mask",
                clamp=True,
                use_container_width=True
            )

        with col3:
            st.image(
                steps["closed_mask"],
                caption="Mask After Closing",
                clamp=True,
                use_container_width=True
            )

    # =========================================================
    # MODULE 2 — OBJECT DETECTION
    # =========================================================

    try:
        contour, contour_img, binary_mask, status_message = (
            detect_objects(preprocessed_img)
        )

    except Exception as exc:
        st.error(f"Object detection failed: {exc}")
        return

    if status_message:
        if contour is not None:
            st.success(status_message)
        else:
            st.warning(status_message)

    if contour is None:
        st.image(
            contour_img,
            channels="BGR",
            caption="Detection Result"
        )
        return

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

    col1, col2 = st.columns(2)

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