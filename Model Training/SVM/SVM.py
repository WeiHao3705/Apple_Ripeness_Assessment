from __future__ import annotations
 
import sys
from pathlib import Path
 
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
 
# ---------------------------------------------------------------------------
# Setup: import the shared preprocessing module
# ---------------------------------------------------------------------------
# Allows "Model Training/SVM.py" to import from the sibling "module/" folder.
import importlib


def find_preprocessing_module():
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
# Adjust this path to match your machine.
DATASET_DIR = Path(
    r"C:\Users\jecsh\OneDrive\Desktop\RSW Y3S1\Apple_Ripeness_Assessment"
    r"\Apple Ripeness Levels Image Dataset"
)
 
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT_PATH = OUTPUT_DIR / "svm_apple_ripeness_model.pkl"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
 
IMAGE_SIZE = (256, 256)          # (width, height) images are resized to before feature extraction
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5  # for stratified k-fold sanity check

# Explicit ripeness progression order for display (confusion matrix, report).
# LabelEncoder sorts classes alphabetically by default, which would put
# "Overripe" in the wrong position (before "100%" alphabetically). This list
# controls the order labels are SHOWN in, independent of how they're encoded.
# Any class folder not listed here is simply appended at the end.
CLASS_ORDER = ["20%", "40%", "60%", "80%", "100%", "Overripe"]


def get_display_order(present_classes) -> list:
    """Return class labels ordered per CLASS_ORDER, with any extras appended."""
    ordered = [c for c in CLASS_ORDER if c in present_classes]
    extras = [c for c in present_classes if c not in CLASS_ORDER]
    return ordered + extras
 
 
# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_color_features(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """HSV color histogram + per-channel mean/std."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist = hist / (hist.sum() + 1e-7)  # normalize so brightness/scale doesn't dominate
 
    stats = np.array([
        hsv[:, :, 0].mean(), hsv[:, :, 0].std(),
        hsv[:, :, 1].mean(), hsv[:, :, 1].std(),
        hsv[:, :, 2].mean(), hsv[:, :, 2].std(),
    ])
 
    return np.concatenate([hist, stats])
 
 
def extract_texture_features(image: np.ndarray) -> np.ndarray:
    """GLCM texture features (contrast, dissimilarity, homogeneity, energy, correlation, ASM)."""
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
    """Combine color + texture features into a single feature vector."""
    color_feats = extract_color_features(image)
    texture_feats = extract_texture_features(image)
    return np.concatenate([color_feats, texture_feats])
 
 
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply the shared preprocessing pipeline (segmentation + contrast enhancement)."""
    segmented = segment_background(image)
    enhanced = apply_clahe(segmented)
    return enhanced
 
 
# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.resize(image, IMAGE_SIZE)
 
 
def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk the dataset directory. Each immediate subfolder is treated as a class
    (e.g. "20%", "40%", ...). Returns (X, y) as numpy arrays.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
 
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subfolders found in {dataset_dir}")
 
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
 
    X, y = [], []
    for class_dir in class_dirs:
        label = class_dir.name
        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
        print(f"  Class '{label}': {len(image_paths)} images")
 
        for img_path in image_paths:
            image = load_image(img_path)
            if image is None:
                print(f"    Skipping unreadable file: {img_path.name}")
                continue
 
            try:
                processed = preprocess_image(image)
                features = extract_features(processed)
            except Exception as exc:
                print(f"    Skipping {img_path.name} due to error: {exc}")
                continue
 
            X.append(features)
            y.append(label)
 
    return np.array(X), np.array(y)
 
 
# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_svm(X: np.ndarray, y: np.ndarray):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
    ])
 
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.001],
        "svm__kernel": ["rbf", "linear"],
    }
 
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=5, scoring="accuracy",
        n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
 
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")
 
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
 
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")

    # Order labels by ripeness progression rather than alphabetically
    display_labels = get_display_order(list(encoder.classes_))
    display_indices = [np.where(encoder.classes_ == c)[0][0] for c in display_labels]

    print(classification_report(
        y_test, y_pred,
        labels=display_indices,
        target_names=display_labels,
    ))

    cm = confusion_matrix(y_test, y_pred, labels=display_indices)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=display_labels, yticklabels=display_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - SVM Apple Ripeness Classifier")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
 
    return best_model, encoder
 
 
# ---------------------------------------------------------------------------
# Cross-validation sanity check
# ---------------------------------------------------------------------------
def cross_validate_svm(X: np.ndarray, y: np.ndarray, best_params: dict) -> None:
    """
    Run stratified k-fold cross-validation over the ENTIRE dataset using the
    best hyperparameters found by GridSearchCV. This checks whether the high
    test accuracy from a single train/test split actually holds up, or was
    just a lucky split / a sign of data leakage (e.g. near-duplicate images
    of the same physical apple appearing in both train and test).
 
    A single split can look perfect (e.g. 99/100) by chance on a small
    dataset; if the per-fold accuracies below vary a lot, or are all
    suspiciously perfect, that's worth investigating further rather than
    trusting the headline number.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=best_params["svm__C"],
            gamma=best_params["svm__gamma"],
            kernel=best_params["svm__kernel"],
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])
 
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = cross_val_score(pipeline, X, y_encoded, cv=skf, scoring="accuracy", n_jobs=-1)
 
    print(f"\n--- Stratified {N_FOLDS}-Fold Cross-Validation (sanity check) ---")
    for i, score in enumerate(fold_scores, start=1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"  Mean accuracy: {fold_scores.mean():.4f}")
    print(f"  Std deviation: {fold_scores.std():.4f}")
 
    if fold_scores.std() > 0.05:
        print(
            "  NOTE: fold accuracies vary noticeably. This suggests the single "
            "train/test split earlier may have been optimistic (lucky split), "
            "or that some folds contain harder/more ambiguous images."
        )
    elif fold_scores.mean() > 0.98 and fold_scores.std() < 0.02:
        print(
            "  NOTE: accuracy is consistently very high across all folds. "
            "Before trusting this, double check for near-duplicate images "
            "(same physical apple photographed multiple times) split across "
            "folds, and check whether classes are trivially separable by "
            "color/background rather than genuine ripeness texture cues."
        )
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading dataset and extracting features...")
    X, y = load_dataset(DATASET_DIR)
    print(f"\nTotal samples: {len(X)} | Feature dimension: {X.shape[1] if len(X) else 0}\n")
 
    model, encoder = train_svm(X, y)
 
    # Retrieve the best hyperparameters found during GridSearchCV so the
    # cross-validation check below uses the same settings, not defaults.
    best_svm_step = model.named_steps["svm"]
    best_params = {
        "svm__C": best_svm_step.C,
        "svm__gamma": best_svm_step.gamma,
        "svm__kernel": best_svm_step.kernel,
    }
    cross_validate_svm(X, y, best_params)
 
    joblib.dump({"model": model, "label_encoder": encoder}, MODEL_OUTPUT_PATH)
    print(f"\nTrained model saved to: {MODEL_OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()