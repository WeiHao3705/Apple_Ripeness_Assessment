from __future__ import annotations

import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.Module1.model import PreprocessingResult
from module.Module1.preprocessing import preprocess_image_with_steps
from module.object_detection import detect_objects


MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


@dataclass
class AppleResult:
    apple_id: int
    crop: np.ndarray
    label: str
    confidence: float
    probabilities: dict[str, float]
    classification_error: str | None = None


@dataclass
class ImageAnalysis:
    source: np.ndarray
    processed: np.ndarray
    annotated: np.ndarray
    preprocessing: PreprocessingResult
    apples: list[AppleResult]
    detection_message: str


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a three-channel OpenCV BGR image."""
    if not image_bytes:
        raise ValueError("The selected image is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError("The selected image is larger than the 10 MB limit.")

    try:
        with Image.open(BytesIO(image_bytes)) as pil_image:
            corrected = ImageOps.exif_transpose(pil_image).convert("RGB")
            rgb_image = np.asarray(corrected)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The selected file could not be read as an image.") from exc

    if rgb_image.shape[0] < 100 or rgb_image.shape[1] < 100:
        raise ValueError("The image must be at least 100 × 100 pixels.")

    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def _white_segmented_background(
    image: np.ndarray,
    foreground_mask: np.ndarray,
) -> np.ndarray:
    white_background = np.full_like(image, 255)
    return cv2.copyTo(image, foreground_mask, white_background)


def analyse_image(image_bytes: bytes) -> ImageAnalysis:
    """Run Modules 1–3 on one camera capture or uploaded image."""
    source = decode_image(image_bytes)
    preprocessing = preprocess_image_with_steps(source)
    processed = preprocessing.final

    if preprocessing.segmentation_success:
        processed = _white_segmented_background(
            processed,
            preprocessing.refined_mask,
        )

    detected_apples, annotated, _mask, message = detect_objects(processed)
    apple_results: list[AppleResult] = []

    predict_ripeness = None
    classifier_import_error = None
    try:
        from module.classification import predict_ripeness
    except (ImportError, ValueError) as exc:
        classifier_import_error = (
            "The classification dependencies could not be loaded: "
            f"{exc}"
        )

    for detected_apple in detected_apples:
        x, y, width, height = detected_apple["bbox"]
        apple_id = int(detected_apple["id"])

        # Classification performs its own preprocessing. Use the clean,
        # resized crop so contrast enhancement and segmentation are not
        # accidentally applied twice.
        crop = np.ascontiguousarray(
            preprocessing.resized[y:y + height, x:x + width]
        )

        label = "Unavailable"
        confidence = 0.0
        probabilities: dict[str, float] = {}
        classification_error = classifier_import_error

        if predict_ripeness is not None:
            try:
                prediction = predict_ripeness(crop)
                label = str(prediction["label"])
                confidence = float(prediction["confidence"])
                probabilities = {
                    str(class_label): float(probability)
                    for class_label, probability in prediction.get(
                        "probabilities", {}
                    ).items()
                }
            except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
                classification_error = str(exc)

        apple_results.append(
            AppleResult(
                apple_id=apple_id,
                crop=crop,
                label=label,
                confidence=confidence,
                probabilities=probabilities,
                classification_error=classification_error,
            )
        )

    return ImageAnalysis(
        source=source,
        processed=processed,
        annotated=annotated,
        preprocessing=preprocessing,
        apples=apple_results,
        detection_message=message,
    )
