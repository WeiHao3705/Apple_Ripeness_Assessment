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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
 
# ---------------------------------------------------------------------------
# Setup: import the shared preprocessing module
# ---------------------------------------------------------------------------
def find_module_dir(start: Path) -> Path:
    """
    Walk upward from `start` looking for a folder named "module" that
    contains preprocessing.py. This makes the script location-independent -
    it keeps working whether the script sits directly in "Model Training/"
    or in a subfolder like "Model Training/CNN/", as long as "module/" is
    somewhere in a parent directory.
    """
    for parent in [start] + list(start.parents):
        candidate = parent / "module"
        if (candidate / "preprocessing.py").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a 'module' folder containing preprocessing.py "
        f"above {start}. Check your folder structure."
    )
 
 
MODULE_DIR = find_module_dir(Path(__file__).resolve().parent)
sys.path.append(str(MODULE_DIR))
 
from preprocessing import segment_background, apply_clahe  # noqa: E402
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = Path(
    r"C:\Users\jecsh\OneDrive\Desktop\RSW Y3S1\Apple_Ripeness_Assessment"
    r"\Apple Ripeness Levels Image Dataset"
)
 
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT_PATH = OUTPUT_DIR / "rf_apple_ripeness_model.pkl"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix_rf.png"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance_rf.png"
 
IMAGE_SIZE = (256, 256)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
 
# Explicit ripeness progression order for display (confusion matrix, report).
# LabelEncoder sorts classes alphabetically by default, which would put
# "Overripe" in the wrong spot. This list controls display order only.
CLASS_ORDER = ["20%", "40%", "60%", "80%", "100%", "Overripe"]
 
 
def get_display_order(present_classes) -> list:
    """Return class labels ordered per CLASS_ORDER, with any extras appended."""
    ordered = [c for c in CLASS_ORDER if c in present_classes]
    extras = [c for c in present_classes if c not in CLASS_ORDER]
    return ordered + extras
 
 
# ---------------------------------------------------------------------------
# Feature extraction (identical to SVM.py, so results are comparable)
# ---------------------------------------------------------------------------
def extract_color_features(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """HSV color histogram + per-channel mean/std."""
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
 
 
def feature_names() -> list:
    """
    Human-readable names for each feature dimension, in the same order they
    are concatenated in extract_features(). Used for the feature importance
    plot below.
    """
    names = []
    for channel in ("H", "S", "V"):
        for i in range(32):
            names.append(f"{channel}_hist_{i}")
    names += ["H_mean", "H_std", "S_mean", "S_std", "V_mean", "V_std"]
    names += ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    return names
 
 
# ---------------------------------------------------------------------------
# Dataset loading (identical to SVM.py)
# ---------------------------------------------------------------------------
def load_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.resize(image, IMAGE_SIZE)
 
 
def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk the dataset directory. Each immediate subfolder is treated as a
    class (e.g. "20%", "40%", ..., "Overripe"). Returns (X, y) as numpy arrays.
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
def train_random_forest(X: np.ndarray, y: np.ndarray):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
 
    # Random Forest doesn't strictly need feature scaling, but keeping the
    # scaler in the pipeline keeps it consistent with SVM.py and does no harm.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
 
    param_grid = {
        "rf__n_estimators": [100, 200, 400],
        "rf__max_depth": [None, 10, 20, 30],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", "log2"],
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
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=display_labels, yticklabels=display_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Random Forest Apple Ripeness Classifier")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
 
    # Random Forest gives feature importances "for free" - plot the top 20.
    # This is a nice diagnostic SVM can't easily give you: which features
    # (which color bins / texture stats) actually drive the classification.
    rf_step = best_model.named_steps["rf"]
    importances = rf_step.feature_importances_
    names = feature_names()
    order = np.argsort(importances)[::-1][:20]
 
    plt.figure(figsize=(9, 7))
    plt.barh([names[i] for i in order][::-1], importances[order][::-1], color="seagreen")
    plt.xlabel("Feature importance")
    plt.title("Top 20 Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH)
    plt.close()
    print(f"Feature importance plot saved to: {FEATURE_IMPORTANCE_PATH}")
 
    return best_model, encoder
 
 
# ---------------------------------------------------------------------------
# Cross-validation sanity check (same purpose as in SVM.py)
# ---------------------------------------------------------------------------
def cross_validate_rf(X: np.ndarray, y: np.ndarray, best_params: dict) -> None:
    """
    Run stratified k-fold cross-validation over the ENTIRE dataset using the
    best hyperparameters found by GridSearchCV, to check whether the held-out
    test accuracy holds up consistently or was a lucky split.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=best_params["rf__n_estimators"],
            max_depth=best_params["rf__max_depth"],
            min_samples_split=best_params["rf__min_samples_split"],
            min_samples_leaf=best_params["rf__min_samples_leaf"],
            max_features=best_params["rf__max_features"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
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
            "  NOTE: fold accuracies vary noticeably. The single train/test "
            "split earlier may have been optimistic (lucky split), or some "
            "folds contain harder/more ambiguous images."
        )
    elif fold_scores.mean() > 0.98 and fold_scores.std() < 0.02:
        print(
            "  NOTE: accuracy is consistently very high across all folds. "
            "Cross-check against SVM.py's cross-validation result and, if "
            "not already done, verify there's no data leakage (near-duplicate "
            "images of the same apple split across folds)."
        )
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading dataset and extracting features...")
    X, y = load_dataset(DATASET_DIR)
    print(f"\nTotal samples: {len(X)} | Feature dimension: {X.shape[1] if len(X) else 0}\n")
 
    model, encoder = train_random_forest(X, y)
 
    rf_step = model.named_steps["rf"]
    best_params = {
        "rf__n_estimators": rf_step.n_estimators,
        "rf__max_depth": rf_step.max_depth,
        "rf__min_samples_split": rf_step.min_samples_split,
        "rf__min_samples_leaf": rf_step.min_samples_leaf,
        "rf__max_features": rf_step.max_features,
    }
    cross_validate_rf(X, y, best_params)
 
    joblib.dump({"model": model, "label_encoder": encoder}, MODEL_OUTPUT_PATH)
    print(f"\nTrained model saved to: {MODEL_OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()