"""Module 1 image acquisition: upload, batch upload and camera capture."""

from __future__ import annotations  # Defer type-hint evaluation for modern annotations.

import hashlib  # Creates content hashes so the same upload is not saved twice.
import io  # Wraps uploaded bytes in a file-like object for Pillow.
from datetime import datetime  # Adds a timestamp to saved image names.
from pathlib import Path  # Builds portable upload paths and file names.

import cv2  # OpenCV converts colour formats and writes processed images.
import numpy as np  # Represents OpenCV images as multidimensional arrays.
import streamlit as st  # Provides upload, camera, message, and image UI widgets.
from PIL import Image, UnidentifiedImageError  # Pillow decodes and validates images.


# ============================================================
# Configuration
# ============================================================

SUPPORTED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
}

MAX_FILE_SIZE_BYTES = (
    10 * 1024 * 1024
)

MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

UPLOAD_DIR = Path("uploads")

ORIGINAL_DIR = (
    UPLOAD_DIR / "original"
)

PROCESSED_DIR = (
    UPLOAD_DIR / "processed"
)


# ============================================================
# PIL -> OpenCV
# ============================================================

def pil_to_cv2(
    image: Image.Image,
) -> np.ndarray:
    """
    Convert a PIL image into an OpenCV BGR image.
    """

    # Pillow normalizes inputs such as grayscale or RGBA images to three-channel RGB.
    rgb_image = image.convert(
        "RGB"
    )

    # NumPy changes the Pillow object into the array representation OpenCV expects.
    np_image = np.array(
        rgb_image
    )

    # Pillow uses RGB order while OpenCV conventionally uses BGR order.
    return cv2.cvtColor(
        np_image,
        cv2.COLOR_RGB2BGR,
    )


# ============================================================
# Read Uploaded Image
# ============================================================

def _read_uploaded_image(
    uploaded_file,
) -> tuple[np.ndarray, Image.Image]:
    """Decode an uploaded file and return OpenCV and PIL image versions."""

    file_bytes = (
        uploaded_file.getvalue()
    )

    # BytesIO lets Pillow read Streamlit's in-memory upload like a normal file.
    pil_image = Image.open(
        io.BytesIO(file_bytes)
    )

    pil_image.load()

    cv2_image = pil_to_cv2(
        pil_image
    )

    return (
        cv2_image,
        pil_image,
    )


# ============================================================
# Save Original Image
# ============================================================

def _save_uploaded_image(
    uploaded_file,
    pil_image: Image.Image,
) -> Path:
    """Save an uploaded image under a timestamped name and return its path."""

    ORIGINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_name = Path(
        uploaded_file.name
    )

    # strftime produces a filesystem-safe, sortable date-and-time suffix.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    extension = (
        original_name.suffix.lower()
    )

    if not extension:
        extension = ".jpg"

    saved_name = (
        f"{original_name.stem}_"
        f"{timestamp}"
        f"{extension}"
    )

    saved_path = (
        ORIGINAL_DIR
        / saved_name
    )

    pil_image.save(
        saved_path
    )

    return saved_path


# ============================================================
# Save Processed Image
# ============================================================

def save_processed_image(
    processed_image: np.ndarray,
    source_path: Path,
) -> Path:
    """
    Save the final preprocessing output.

    Kept as a compatibility function for app.py.
    """

    if (
        processed_image is None
        or processed_image.size == 0
    ):
        raise ValueError(
            "Processed image is empty."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_name = (
        f"{source_path.stem}"
        "_processed"
        f"{source_path.suffix}"
    )

    saved_path = (
        PROCESSED_DIR
        / saved_name
    )

    # OpenCV encodes the array according to the extension in saved_path.
    success = cv2.imwrite(
        saved_path.as_posix(),
        processed_image,
    )

    if not success:
        raise OSError(
            "Unable to save processed image."
        )

    return saved_path


# ============================================================
# Prevent Duplicate Saving
# ============================================================

def _maybe_save_uploaded_image(
    uploaded_file,
    pil_image: Image.Image,
) -> Path:
    """Save an upload once per session, reusing its path when seen again."""

    # A SHA-256 digest identifies file content even when filenames are identical.
    file_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    saved_paths = (
        st.session_state.setdefault(
            "saved_upload_paths",
            {},
        )
    )

    if file_hash in saved_paths:
        return Path(
            saved_paths[file_hash]
        )

    saved_path = _save_uploaded_image(
        uploaded_file,
        pil_image,
    )

    saved_paths[
        file_hash
    ] = saved_path.as_posix()

    return saved_path


# ============================================================
# Display Uploaded Image
# ============================================================

def _display_uploaded_image(
    uploaded_file,
    image: np.ndarray,
    pil_image: Image.Image,
) -> None:
    """Display an uploaded image and its basic file details in Streamlit."""

    # channels="BGR" tells Streamlit to display the OpenCV channel order correctly.
    st.image(
        image,
        channels="BGR",
        caption="Uploaded Image",
    )

    st.caption(
        f"Filename: {uploaded_file.name} | "
        f"Resolution: "
        f"{pil_image.width} × "
        f"{pil_image.height} | "
        f"Format: "
        f"{pil_image.format or 'Unknown'}"
    )


# ============================================================
# File Validation
# ============================================================

def _validate_uploaded_file(
    uploaded_file,
) -> bool:
    """Check an upload's extension and size, displaying errors when invalid."""

    file_extension = (
        Path(
            uploaded_file.name
        )
        .suffix
        .lower()
        .lstrip(".")
    )

    if (
        file_extension
        not in SUPPORTED_EXTENSIONS
    ):
        st.error(
            "Unsupported image format. "
            "Please use JPG, JPEG or PNG."
        )

        return False

    file_size = len(
        uploaded_file.getvalue()
    )

    if (
        file_size
        > MAX_FILE_SIZE_BYTES
    ):
        st.error(
            "Image file is too large. "
            "Maximum size is 10 MB."
        )

        return False

    return True


# ============================================================
# Common Upload Processing
# ============================================================

def _process_uploaded_file(
    uploaded_file,
) -> tuple[np.ndarray, Path] | None:
    """Validate, decode, display, and save one uploaded image."""

    if not _validate_uploaded_file(
        uploaded_file
    ):
        return None

    try:
        (
            image,
            pil_image,
        ) = _read_uploaded_image(
            uploaded_file
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        st.error(
            "Unable to read the image."
        )

        return None

    # --------------------------------------------------------
    # Resolution validation
    # --------------------------------------------------------

    if (
        pil_image.width
        < MIN_IMAGE_WIDTH
        or
        pil_image.height
        < MIN_IMAGE_HEIGHT
    ):
        st.error(
            "Image resolution is too low. "
            "Minimum resolution is "
            "100 × 100 pixels."
        )

        return None

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    _display_uploaded_image(
        uploaded_file,
        image,
        pil_image,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    try:
        saved_path = (
            _maybe_save_uploaded_image(
                uploaded_file,
                pil_image,
            )
        )

    except OSError:
        st.error(
            "Unable to save the uploaded image."
        )

        return None

    st.caption(
        f"Saved to: "
        f"{saved_path.as_posix()}"
    )

    return (
        image,
        saved_path,
    )


# ============================================================
# Single Image Upload
# ============================================================

def upload_single_image():
    """Render a single-image uploader and return the processed upload details."""

    # Streamlit returns an in-memory UploadedFile, or None before a selection.
    uploaded_file = st.file_uploader(
        "Upload an apple image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info(
            "Please upload an image."
        )

        return None

    return _process_uploaded_file(
        uploaded_file
    )


# ============================================================
# Batch Image Upload
# ============================================================

def upload_batch_images():
    """Render a multi-image uploader and return all valid uploaded images."""

    # Multiple-file mode returns a list of UploadedFile objects.
    uploaded_files = st.file_uploader(
        "Upload apple images",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info(
            "Please upload images."
        )

        return []

    images = []

    for uploaded_file in uploaded_files:

        image_info = (
            _process_uploaded_file(
                uploaded_file
            )
        )

        if image_info is not None:
            images.append(
                image_info
            )

    if images:
        st.success(
            f"{len(images)} "
            "image(s) uploaded successfully."
        )

    return images


# ============================================================
# Camera Capture
# ============================================================

def camera_capture():
    """Capture one camera image and process it like a regular upload."""

    # camera_input returns a file-like image captured by the browser camera.
    captured_file = st.camera_input(
        "Take a picture of an apple"
    )

    if captured_file is None:
        st.info(
            "Please capture an image."
        )

        return None

    return _process_uploaded_file(
        captured_file
    )
