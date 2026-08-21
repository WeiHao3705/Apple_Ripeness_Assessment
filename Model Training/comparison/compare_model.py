from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# ---------------------------------------------------------------------------
# Setup: import the shared preprocessing module (searches upward)
# ---------------------------------------------------------------------------
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
DATASET_DIR = Path(
    r"C:\Users\jecsh\OneDrive\Desktop\RSW Y3S1\Apple_Ripeness_Assessment"
    r"\Apple Ripeness Levels Image Dataset"
)

OUTPUT_DIR = Path(__file__).resolve().parent
BOXPLOT_PATH = OUTPUT_DIR / "svm_vs_rf_paired_folds.png"
SVM_AGG_CM_PATH = OUTPUT_DIR / "confusion_matrix_svm_aggregated_50folds.png"
RF_AGG_CM_PATH = OUTPUT_DIR / "confusion_matrix_rf_aggregated_50folds.png"

IMAGE_SIZE = (256, 256)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_STATE = 42

N_SPLITS = 5        # folds per repeat
N_REPEATS = 10       # -> 50 total paired scores per model (more repeats = more stable estimate)
ALPHA = 0.05         # significance threshold

# Explicit ripeness progression order for display, matching SVM.py / RandomForest.py.
CLASS_ORDER = ["20%", "40%", "60%", "80%", "100%", "Overripe"]


def get_display_order(present_classes) -> list:
    """Return class labels ordered per CLASS_ORDER, with any extras appended."""
    ordered = [c for c in CLASS_ORDER if c in present_classes]
    extras = [c for c in present_classes if c not in CLASS_ORDER]
    return ordered + extras

# --- Fill these in from your GridSearchCV results in SVM.py / RandomForest.py ---
# You can paste the "Best params: {...}" dict directly, either as plain keys
# (e.g. "C") or with the pipeline prefix GridSearchCV prints
# (e.g. "svm__C") - both formats work, the prefix is stripped automatically.
SVM_BEST_PARAMS = {'svm__C': 10, 'svm__gamma': 0.01, 'svm__kernel': 'rbf'}

RF_BEST_PARAMS = {'rf__max_depth': None, 'rf__max_features': 'sqrt', 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 400}


# ---------------------------------------------------------------------------
# Feature extraction (identical to SVM.py / RandomForest.py)
# ---------------------------------------------------------------------------
def extract_color_features(image: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist = hist / (hist.sum() + 1e-7)

    stats_arr = np.array([
        hsv[:, :, 0].mean(), hsv[:, :, 0].std(),
        hsv[:, :, 1].mean(), hsv[:, :, 1].std(),
        hsv[:, :, 2].mean(), hsv[:, :, 2].std(),
    ])

    return np.concatenate([hist, stats_arr])


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
# Paired repeated k-fold comparison
# ---------------------------------------------------------------------------
def build_svm_pipeline() -> Pipeline:
    params = strip_prefix(SVM_BEST_PARAMS, "svm__")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=params["C"],
            gamma=params["gamma"],
            kernel=params["kernel"],
            random_state=RANDOM_STATE,
        )),
    ])


def build_rf_pipeline() -> Pipeline:
    params = strip_prefix(RF_BEST_PARAMS, "rf__")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def cohens_d_paired(diffs: np.ndarray) -> float:
    """Effect size for paired samples: mean difference / std of differences."""
    return diffs.mean() / (diffs.std(ddof=1) + 1e-12)


def strip_prefix(params: dict, prefix: str) -> dict:
    """
    Normalize a params dict so it works whether you pasted plain keys
    (e.g. {"C": 10}) or GridSearchCV's pipeline-prefixed keys straight from
    the "Best params: {...}" printout in SVM.py/RandomForest.py
    (e.g. {"svm__C": 10}). Strips the given prefix from any key that has it.
    """
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in params.items()
    }


def run_paired_comparison(X: np.ndarray, y: np.ndarray):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    display_labels = get_display_order(list(encoder.classes_))
    display_indices = [np.where(encoder.classes_ == c)[0][0] for c in display_labels]

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    svm_scores, rf_scores = [], []
    # Per-fold, per-class precision/recall/f1 - shape will end up
    # (total_folds, num_classes) for each metric, for each model.
    svm_precision, svm_recall, svm_f1 = [], [], []
    rf_precision, rf_recall, rf_f1 = [], [], []

    num_classes = len(display_labels)
    svm_cm_sum = np.zeros((num_classes, num_classes), dtype=int)
    rf_cm_sum = np.zeros((num_classes, num_classes), dtype=int)

    total_folds = N_SPLITS * N_REPEATS
    print(f"Running {total_folds} paired folds ({N_SPLITS}-fold x {N_REPEATS} repeats)...\n")

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y_encoded), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        svm_model = build_svm_pipeline()
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_pred)
        svm_scores.append(svm_acc)

        p, r, f, _ = precision_recall_fscore_support(
            y_test, svm_pred, labels=display_indices, zero_division=0
        )
        svm_precision.append(p)
        svm_recall.append(r)
        svm_f1.append(f)
        svm_cm_sum += confusion_matrix(y_test, svm_pred, labels=display_indices)

        rf_model = build_rf_pipeline()
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_scores.append(rf_acc)

        p, r, f, _ = precision_recall_fscore_support(
            y_test, rf_pred, labels=display_indices, zero_division=0
        )
        rf_precision.append(p)
        rf_recall.append(r)
        rf_f1.append(f)
        rf_cm_sum += confusion_matrix(y_test, rf_pred, labels=display_indices)

        if fold_idx % N_SPLITS == 0:
            print(f"  Repeat {fold_idx // N_SPLITS}: "
                  f"SVM mean so far = {np.mean(svm_scores):.4f}, "
                  f"RF mean so far = {np.mean(rf_scores):.4f}")

    svm_scores = np.array(svm_scores)
    rf_scores = np.array(rf_scores)
    diffs = svm_scores - rf_scores  # positive => SVM better on that fold

    print("\n=== Summary ===")
    print(f"SVM: mean = {svm_scores.mean():.4f}, std = {svm_scores.std():.4f}")
    print(f"RF:  mean = {rf_scores.mean():.4f}, std = {rf_scores.std():.4f}")
    print(f"Mean paired difference (SVM - RF): {diffs.mean():.4f}")

    # --- Paired t-test (parametric) ---
    t_stat, t_pvalue = stats.ttest_rel(svm_scores, rf_scores)
    print(f"\nPaired t-test: t = {t_stat:.4f}, p = {t_pvalue:.4f}")

    # --- Wilcoxon signed-rank test (non-parametric backup, robust to
    #     non-normal accuracy distributions, which is common near ceiling) ---
    try:
        w_stat, w_pvalue = stats.wilcoxon(svm_scores, rf_scores)
        print(f"Wilcoxon signed-rank test: W = {w_stat:.4f}, p = {w_pvalue:.4f}")
    except ValueError as exc:
        # Wilcoxon fails if all differences are exactly zero
        w_pvalue = None
        print(f"Wilcoxon signed-rank test could not be computed: {exc}")

    d = cohens_d_paired(diffs)
    print(f"Cohen's d (paired effect size): {d:.4f}")

    # --- Interpretation ---
    print("\n=== Interpretation ===")
    if t_pvalue < ALPHA:
        better = "SVM" if diffs.mean() > 0 else "Random Forest"
        print(
            f"p = {t_pvalue:.4f} < {ALPHA} -> statistically significant difference. "
            f"{better} performs significantly better on this dataset."
        )
    else:
        print(
            f"p = {t_pvalue:.4f} >= {ALPHA} -> NOT statistically significant. "
            f"There is no reliable evidence that SVM and Random Forest differ "
            f"in accuracy on this dataset - treat them as performing equivalently."
        )

    if abs(d) < 0.2:
        effect_desc = "negligible"
    elif abs(d) < 0.5:
        effect_desc = "small"
    elif abs(d) < 0.8:
        effect_desc = "medium"
    else:
        effect_desc = "large"
    print(f"Effect size is {effect_desc} (|d| = {abs(d):.4f}), "
          f"regardless of statistical significance.")

    # --- Plot paired fold accuracies ---
    plt.figure(figsize=(8, 6))
    plt.boxplot([svm_scores, rf_scores], tick_labels=["SVM", "Random Forest"])
    for i in range(len(svm_scores)):
        plt.plot([1, 2], [svm_scores[i], rf_scores[i]], color="gray", alpha=0.3, linewidth=0.8)
    plt.scatter(np.ones(len(svm_scores)), svm_scores, alpha=0.6, color="steelblue")
    plt.scatter(np.full(len(rf_scores), 2), rf_scores, alpha=0.6, color="seagreen")
    plt.ylabel("Accuracy")
    plt.title(f"SVM vs Random Forest - {total_folds} Paired Folds\n(p={t_pvalue:.4f}, Cohen's d={d:.3f})")
    plt.tight_layout()
    plt.savefig(BOXPLOT_PATH)
    plt.close()
    print(f"\nComparison plot saved to: {BOXPLOT_PATH}")

    # --- Aggregated per-class report across ALL folds (mean +/- std) ---
    # This is the robust equivalent of sklearn's classification_report, but
    # averaged over every fold instead of coming from a single train/test
    # split - use this instead of the original one-off report when
    # justifying per-class performance in your write-up.
    svm_precision = np.array(svm_precision)  # shape: (total_folds, num_classes)
    svm_recall = np.array(svm_recall)
    svm_f1 = np.array(svm_f1)
    rf_precision = np.array(rf_precision)
    rf_recall = np.array(rf_recall)
    rf_f1 = np.array(rf_f1)

    def print_aggregated_report(name, labels, precision, recall, f1):
        print(f"\n--- Aggregated per-class report - {name} (mean +/- std across {total_folds} folds) ---")
        header = f"{'Class':<12}{'Precision':<18}{'Recall':<18}{'F1-score':<18}"
        print(header)
        print("-" * len(header))
        for i, label in enumerate(labels):
            print(
                f"{label:<12}"
                f"{precision[:, i].mean():.4f} +/- {precision[:, i].std():.4f}   "
                f"{recall[:, i].mean():.4f} +/- {recall[:, i].std():.4f}   "
                f"{f1[:, i].mean():.4f} +/- {f1[:, i].std():.4f}"
            )
        print("-" * len(header))
        print(
            f"{'macro avg':<12}"
            f"{precision.mean():.4f} +/- {precision.std():.4f}   "
            f"{recall.mean():.4f} +/- {recall.std():.4f}   "
            f"{f1.mean():.4f} +/- {f1.std():.4f}"
        )

    print_aggregated_report("SVM", display_labels, svm_precision, svm_recall, svm_f1)
    print_aggregated_report("Random Forest", display_labels, rf_precision, rf_recall, rf_f1)

    # --- Aggregated confusion matrices (summed raw counts across all 50 folds) ---
    # Far more robust than a single train/test split's confusion matrix, since
    # each cell here reflects 10x more predictions per image (10 repeats).
    for name, cm_sum, cmap, path in [
        ("SVM", svm_cm_sum, "Blues", SVM_AGG_CM_PATH),
        ("Random Forest", rf_cm_sum, "Greens", RF_AGG_CM_PATH),
    ]:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_sum, annot=True, fmt="d", cmap=cmap,
            xticklabels=display_labels, yticklabels=display_labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Aggregated Confusion Matrix - {name}\n(summed over {total_folds} folds)")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Aggregated confusion matrix ({name}) saved to: {path}")

    return svm_scores, rf_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading dataset and extracting features...")
    X, y = load_dataset(DATASET_DIR)
    print(f"\nTotal samples: {len(X)} | Feature dimension: {X.shape[1] if len(X) else 0}\n")

    run_paired_comparison(X, y)


if __name__ == "__main__":
    main()