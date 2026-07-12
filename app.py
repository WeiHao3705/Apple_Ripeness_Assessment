import streamlit as st
from module.image_acquisition import (
    upload_single_image,
    upload_batch_images,
    camera_capture,
    UPLOAD_DIR,
    save_processed_image,
)
from module.preprocessing import apply_clahe, segment_background

st.title("Apple Ripeness System")
AFTER_CLAHE_DIR = UPLOAD_DIR / "after_clahe"


def show_comparison(original, processed, title="CLAHE Comparison"):
    st.subheader(title)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original,
            channels="BGR",
            caption="Original Image",
            use_container_width=True,
        )

    with col2:
        st.image(
            processed,
            channels="BGR",
            caption="After CLAHE",
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
        segmented_img = segment_background(img)
        clahe_img = apply_clahe(segmented_img)
        clahe_path = save_processed_image(clahe_img, original_path)
        show_comparison(img, clahe_img)
        st.caption(f"CLAHE saved to: {clahe_path.as_posix()}")
    
elif option == "Batch Upload":
    imgs = upload_batch_images()
    
    for i, (img, original_path) in enumerate(imgs, start=1):
        segmented_img = segment_background(img)
        clahe_img = apply_clahe(segmented_img)
        clahe_path = save_processed_image(clahe_img, original_path)
        show_comparison(img, clahe_img, f"Image {i}")
        st.caption(f"CLAHE saved to: {clahe_path.as_posix()}")

elif option == "Camera":
    upload_result = camera_capture()

    if upload_result is not None:
        img, original_path = upload_result
        segmented_img = segment_background(img)
        clahe_img = apply_clahe(segmented_img)
        clahe_path = save_processed_image(clahe_img, original_path)
        show_comparison(img, clahe_img)
        st.caption(f"CLAHE saved to: {clahe_path.as_posix()}")
    