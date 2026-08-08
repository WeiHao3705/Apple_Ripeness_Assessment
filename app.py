import streamlit as st
from module.image_acquisition import (
    upload_single_image,
    upload_batch_images,
    camera_capture,
    UPLOAD_DIR,
    save_processed_image,
)
from module.preprocessing import apply_clahe, segment_background
from module.object_detection import detect_objects, detect_edges, apply_watershed

st.title("Apple Ripeness System")


def show_detection_results(segmented_img, contour_img, edge_img, watershed_img, binary_mask, edge_mask, watershed_mask):
    st.subheader("Object Detection Results")

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            segmented_img,
            channels="BGR",
            caption="Segmented Image",
            use_container_width=True,
        )
    with col2:
        st.image(
            contour_img,
            channels="BGR",
            caption="Contour Detection",
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.image(
            edge_img,
            channels="BGR",
            caption="Edge Detection Overlay",
            use_container_width=True,
        )
    with col4:
        st.image(
            watershed_img,
            channels="BGR",
            caption="Watershed Segmentation",
            use_container_width=True,
        )

    with st.expander("Show binary masks"):
        mask_col1, mask_col2, mask_col3 = st.columns(3)
        with mask_col1:
            st.image(binary_mask, caption="Contour Mask", clamp=True, use_container_width=True)
        with mask_col2:
            st.image(edge_mask, caption="Edge Mask", clamp=True, use_container_width=True)
        with mask_col3:
            st.image(watershed_mask, caption="Watershed Mask", clamp=True, use_container_width=True)


def process_image(img, original_path, label=None):
    segmented_img = segment_background(img)

    try:
        contour, contour_img, binary_mask, status_message = detect_objects(segmented_img)
    except Exception as exc:
        st.error(f"Object detection failed: {exc}")
        st.image(segmented_img, channels="BGR", caption="Segmented Image")
        return

    if status_message:
        if status_message == "Apple found in the image":
            st.success(status_message)
        else:
            st.warning(status_message)

    if contour is None:
        st.image(segmented_img, channels="BGR", caption="Segmented Image")
        st.image(contour_img, channels="BGR", caption="Detection Result")
        return

    edge_mask, edge_img = detect_edges(segmented_img)
    watershed_img, markers, watershed_mask = apply_watershed(segmented_img)

    show_detection_results(
        segmented_img,
        contour_img,
        edge_img,
        watershed_img,
        binary_mask,
        edge_mask,
        watershed_mask,
    )

    if contour is not None:
        clahe_img = apply_clahe(segmented_img)
        save_processed_image(clahe_img, original_path)
        st.subheader("Preprocessed Image")
        st.image(
            clahe_img,
            channels="BGR",
            caption="Result after preprocessing",
            use_container_width=True,
        )

option = st.selectbox(
    "Select Input Method",
    ["Single Upload", "Batch Upload", "Camera"],
)

if option == "Single Upload":
    upload_result = upload_single_image()

    if upload_result is not None:
        img, original_path = upload_result
        process_image(img, original_path)
    
elif option == "Batch Upload":
    imgs = upload_batch_images()
    
    for i, (img, original_path) in enumerate(imgs, start=1):
        st.subheader(f"Image {i}")
        st.caption(f"Processing: {original_path.name}")
        process_image(img, original_path, label=f"CLAHE Comparison - Image {i}")

elif option == "Camera":
    upload_result = camera_capture()

    if upload_result is not None:
        img, original_path = upload_result
        process_image(img, original_path)
    