from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "Model Training" / "CNN" / "cnn_apple_ripeness_model.keras"
ENCODER_PATH = REPO_ROOT / "Model Training" / "CNN" / "cnn_label_encoder.pkl"
IMAGE_SIZE = (224, 224)


def _validate_image(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("predict_ripeness expects a BGR image with 3 channels.")


def _prepare_image(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32")
    processed = preprocess_input(rgb)
    return np.expand_dims(processed, axis=0)


@lru_cache(maxsize=1)
def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN model file not found: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder file not found: {ENCODER_PATH}")

    model = load_model(MODEL_PATH)
    encoder_data = joblib.load(ENCODER_PATH)
    encoder = encoder_data["label_encoder"] if isinstance(encoder_data, dict) else encoder_data
    return model, encoder


def predict_ripeness(image: np.ndarray) -> dict:
    _validate_image(image)
    model, encoder = _load_artifacts()

    model_input = _prepare_image(image)
    probabilities = model.predict(model_input, verbose=0)[0]
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
