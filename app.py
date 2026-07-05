import streamlit as st
from module.image_acquisition import (
    upload_single_image,
    upload_batch_images,
    camera_capture,
)

st.title("Apple Ripeness System")

option = st.selectbox(
    "Select Input Method",
    ["Single Upload", "Batch Upload", "Camera"],
)

if option == "Single Upload":
    img = upload_single_image()
elif option == "Batch Upload":
    imgs = upload_batch_images()
elif option == "Camera":
    img = camera_capture()