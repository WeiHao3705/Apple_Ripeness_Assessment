"""
train_final_model.py
---------------------
Trains the FINAL deployable SVM model on 100% of the dataset, using the
hyperparameters already found via GridSearchCV in SVM.py.

Why this is separate from SVM.py:
    SVM.py's job was to EVALUATE the model (train/test split, cross
    validation, hyperparameter search) - it deliberately holds back data
    so it can report honest performance numbers.

    This script's job is different: now that model selection and evaluation
    are done, we retrain on ALL available images (no held-out split) using
    the hyperparameters already chosen, so the final deployed model learns
    from every image you have. This is standard practice once you're done
    comparing/justifying model choice and are ready to ship.

    This script does NOT report a test accuracy, because it has no held-out
    test set left - that's expected and correct. Your reported accuracy
    figures should come from SVM.py's evaluation runs, not from this script.

Output:
    svm_final_model.pkl - saved via joblib, contains the fitted pipeline
    (scaler + SVM) and the label encoder, ready to be loaded by predict.py.

Requirements:
    pip install opencv-python numpy scikit-learn scikit-image joblib
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Setup: import the shared preprocessing module (searches upward)
# ---------------------------------------------------------------------------
import importlib


def find_preprocessing_module():
    """
    Locate preprocessing.py by searching upward from this script's location
    (and recursively downward at each level), then import it correctly
    whether it is a standalone file or part of a real package (e.g.
    "module/Module1/" with its own __init__.py, config.py, model.py, etc,
    using relative imports internally like "from .config import ...").

    A plain sys.path.append() + "from preprocessing import ..." only works
    for standalone files - if preprocessing.py belongs to a package, that
    approach breaks its internal relative imports with:
        ImportError: attempted relative import with no known parent package

    This function handles both cases automatically.
    """
    start = Path(__file__).resolve().parent
    preprocessing_path = None
    for parent in [start] + list(start.parents):
        matches = list(parent.rglob("preprocessing.py"))
        if matches:
            preprocessing_path = matches[0]
            break

    if preprocessing_path is None:
        raise FileNotFoundError(
            f"Could not find a 'preprocessing.py' file anywhere above {start}. "
            f"Check your folder structure."
        )

    package_dir = preprocessing_path.parent
    package_name = package_dir.name
    parent_of_package = str(package_dir.parent)

    # Try package-style import FIRST (e.g. "Module1.preprocessing"). This
    # works whether or not Module1 has an __init__.py - Python 3 supports
    # "namespace packages" without one - and is REQUIRED if preprocessing.py
    # uses relative imports internally (e.g. "from .config import ...").
    if parent_of_package not in sys.path:
        sys.path.insert(0, parent_of_package)
    try:
        return importlib.import_module(f"{package_name}.preprocessing")
    except ImportError:
        pass

    # Fall back to flat-file import (old style, no internal relative imports).
    dir_str = str(package_dir)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)
    return importlib.import_module("preprocessing")


_preprocessing = find_preprocessing_module()
segment_background = _preprocessing.segment_background
apply_clahe = _preprocessing.apply_clahe


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = Path(
    r"C:\Users\jecsh\OneDrive\Desktop\RSW Y3S1\Apple_Ripeness_Assessment"
    r"\Apple Ripeness Levels Image Dataset"
)

OUTPUT_DIR = Path(__file__).resolve().parent
FINAL_MODEL_PATH = OUTPUT_DIR / "svm_final_model.pkl"  # saved in module/ so predict.py finds it easily

IMAGE_SIZE = (256, 256)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_STATE = 42

# Hyperparameters already found via GridSearchCV in SVM.py - fill in your
# actual best params here if different.
SVM_BEST_PARAMS = {"C": 10, "gamma": 0.01, "kernel": "rbf"}


# ---------------------------------------------------------------------------
# Feature extraction (identical to SVM.py - must never drift from this)
# ---------------------------------------------------------------------------
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
    return apply_clahe(segment_background(image))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.resize(image, IMAGE_SIZE)


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    X, y = [], []
    for class_dir in class_dirs:
        label = class_dir.name
        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
        print(f"  Class '{label}': {len(image_paths)} images")

        for img_path in image_paths:
            image = load_image(img_path)
            if image is None:
                continue
            try:
                processed = preprocess_image(image)
                features = extract_features(processed)
            except Exception:
                continue
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading FULL dataset (no train/test split - this is the final model)...")
    X, y = load_dataset(DATASET_DIR)
    print(f"\nTotal samples used for final training: {len(X)}\n")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=SVM_BEST_PARAMS["C"],
            gamma=SVM_BEST_PARAMS["gamma"],
            kernel=SVM_BEST_PARAMS["kernel"],
            probability=True,   # needed so predict.py can return confidence scores
            random_state=RANDOM_STATE,
        )),
    ])

    print("Training final SVM on 100% of the dataset...")
    pipeline.fit(X, y_encoded)

    joblib.dump({"model": pipeline, "label_encoder": encoder}, FINAL_MODEL_PATH)
    print(f"\nFinal deployable model saved to: {FINAL_MODEL_PATH}")
    print("This model has seen every image in the dataset and is ready for predict.py.")


if __name__ == "__main__":
    main()