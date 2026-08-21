from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import graycomatrix, graycoprops


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "Model Training" / "svm_final_model.pkl"
IMAGE_SIZE = (256, 256)


def _validate_image(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("predict_ripeness expects a BGR image with 3 channels.")


def _prepare_image(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return extract_features(preprocess_image(resized)).reshape(1, -1)


def extract_color_features(image: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist = hist / (hist.sum() + 1e-7)

    stats = np.array([
        hsv[:, :, 0].mean(), hsv[:, :, 0].std(),
        hsv[:, :, 1].mean(), hsv[:, :, 1].std(),
        hsv[:, :, 2].mean(), hsv[:, :, 2].std(),
    ])

    return np.concatenate([hist, stats])


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    return np.array([
        graycoprops(glcm, "contrast").mean(),
        graycoprops(glcm, "dissimilarity").mean(),
        graycoprops(glcm, "homogeneity").mean(),
        graycoprops(glcm, "energy").mean(),
        graycoprops(glcm, "correlation").mean(),
        graycoprops(glcm, "ASM").mean(),
    ])


def extract_features(image: np.ndarray) -> np.ndarray:
    return np.concatenate([extract_color_features(image), extract_texture_features(image)])


def preprocess_image(image: np.ndarray) -> np.ndarray:
    from module.Module1.preprocessing import apply_clahe, segment_background

    return apply_clahe(segment_background(image))


@lru_cache(maxsize=1)
def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"SVM model file not found: {MODEL_PATH}")

    payload = joblib.load(MODEL_PATH)
    model = payload["model"] if isinstance(payload, dict) else payload
    encoder = payload["label_encoder"] if isinstance(payload, dict) else None
    if encoder is None:
        raise ValueError(f"Label encoder missing from SVM model file: {MODEL_PATH}")
    return model, encoder


def predict_ripeness(image: np.ndarray) -> dict:
    _validate_image(image)
    model, encoder = _load_artifacts()

    model_input = _prepare_image(image)
    probabilities = model.predict_proba(model_input)[0]
    predicted_index = int(np.argmax(probabilities))
    classes = list(encoder.classes_)
    predicted_label = classes[predicted_index]
    confidence = float(probabilities[predicted_index])

    class_probabilities = {
        label: float(probabilities[index]) for index, label in enumerate(classes)
    }

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": class_probabilities,
    }