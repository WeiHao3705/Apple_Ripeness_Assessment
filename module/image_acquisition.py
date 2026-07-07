from __future__ import annotations

import hashlib
import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MIN_IMAGE_WIDTH = 300
MIN_IMAGE_HEIGHT = 300
UPLOAD_DIR = Path("uploads")
BEFORE_CLAHE_DIR = UPLOAD_DIR / "before_clahe"
AFTER_CLAHE_DIR = UPLOAD_DIR / "after_clahe"


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    rgb_img = img.convert("RGB")
    return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)


def _read_uploaded_image(uploaded_file) -> tuple[np.ndarray, Image.Image]:
    file_bytes = uploaded_file.getvalue()
    pil_image = Image.open(io.BytesIO(file_bytes))
    pil_image.load()
    cv2_image = pil_to_cv2(pil_image)
    return cv2_image, pil_image


def _save_uploaded_image(uploaded_file, pil_image: Image.Image) -> Path:
    BEFORE_CLAHE_DIR.mkdir(parents=True, exist_ok=True)
    original_name = Path(uploaded_file.name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{original_name.stem}_{timestamp}{original_name.suffix.lower()}"
    saved_path = BEFORE_CLAHE_DIR / saved_name
    pil_image.save(saved_path)
    return saved_path


def save_processed_image(processed_image: np.ndarray, source_path: Path) -> Path:
    AFTER_CLAHE_DIR.mkdir(parents=True, exist_ok=True)
    saved_name = f"{source_path.stem}_clahe{source_path.suffix}"
    saved_path = AFTER_CLAHE_DIR / saved_name
    cv2.imwrite(saved_path.as_posix(), processed_image)
    return saved_path


def _maybe_save_uploaded_image(uploaded_file, pil_image: Image.Image) -> Path | None:
    file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    saved_paths = st.session_state.setdefault("saved_upload_paths", {})

    if file_hash in saved_paths:
        return Path(saved_paths[file_hash])

    saved_path = _save_uploaded_image(uploaded_file, pil_image)
    saved_paths[file_hash] = saved_path.as_posix()
    return saved_path


def _display_uploaded_image(uploaded_file, image: np.ndarray, pil_image: Image.Image) -> None:
    st.image(image, channels="BGR", caption="Uploaded Image")
    st.caption(
        f"Filename: {uploaded_file.name} | Resolution: {pil_image.width} x {pil_image.height} | "
        f"Format: {pil_image.format or 'Unknown'}"
    )


def _validate_uploaded_file(uploaded_file) -> str | None:
    file_extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if file_extension not in SUPPORTED_EXTENSIONS:
        st.error("Unsupported image format.")
        return None

    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error("Image file is too large. Maximum size is 10 MB.")
        return None

    return file_extension


def _process_uploaded_file(uploaded_file) -> tuple[np.ndarray, Path] | None:
    if _validate_uploaded_file(uploaded_file) is None:
        return None

    try:
        image, pil_image = _read_uploaded_image(uploaded_file)
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Unable to read image.")
        return None

    if pil_image.width < MIN_IMAGE_WIDTH or pil_image.height < MIN_IMAGE_HEIGHT:
        st.error("Image resolution is too low.")
        return None

    _display_uploaded_image(uploaded_file, image, pil_image)

    saved_path = _maybe_save_uploaded_image(uploaded_file, pil_image)
    if saved_path is None:
        st.error("Unable to save the uploaded image.")
        return None

    st.caption(f"Saved to: {saved_path.as_posix()}")

    return image, saved_path


def upload_single_image():
    uploaded_file = st.file_uploader(
        "Upload an apple image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Please upload an image.")
        return None

    return _process_uploaded_file(uploaded_file)


def upload_batch_images():
    uploaded_files = st.file_uploader(
        "Upload apple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Please upload images.")
        return []

    images = []
    for uploaded_file in uploaded_files:
        image_info = _process_uploaded_file(uploaded_file)
        if image_info is not None:
            images.append(image_info)

    if images:
        st.success(f"{len(images)} image(s) uploaded successfully.")

    return images


def camera_capture():
    captured_file = st.camera_input("Take a picture of an apple")

    if captured_file is None:
        st.info("Please capture an image.")
        return None

    return _process_uploaded_file(captured_file)