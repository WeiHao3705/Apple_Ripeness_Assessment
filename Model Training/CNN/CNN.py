from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Setup: import the shared preprocessing module (searches upward, so it
# works regardless of how deep this script is nested)
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
MODEL_OUTPUT_PATH = OUTPUT_DIR / "cnn_apple_ripeness_model.keras"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix_cnn.png"
TRAINING_CURVES_PATH = OUTPUT_DIR / "training_curves_cnn.png"

IMAGE_SIZE = (224, 224)          # MobileNetV2's standard expected input size
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 16                  # smaller than before since 224x224 uses more memory
PHASE1_EPOCHS = 25               # frozen-backbone training
PHASE2_EPOCHS = 15               # fine-tuning with top layers unfrozen
FINE_TUNE_AT_LAYER = -30         # unfreeze roughly the last 30 layers of MobileNetV2
N_FOLDS = 5
CV_EPOCHS = 12                    # transfer learning converges faster, so fewer epochs/fold needed

CLASS_ORDER = ["20%", "40%", "60%", "80%", "100%", "Overripe"]


def get_display_order(present_classes) -> list:
    """Return class labels ordered per CLASS_ORDER, with any extras appended."""
    ordered = [c for c in CLASS_ORDER if c in present_classes]
    extras = [c for c in present_classes if c not in CLASS_ORDER]
    return ordered + extras


# ---------------------------------------------------------------------------
# Preprocessing (identical segmentation/CLAHE step to SVM.py / RandomForest.py)
# ---------------------------------------------------------------------------
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply the shared preprocessing pipeline (segmentation + contrast enhancement)."""
    segmented = segment_background(image)
    enhanced = apply_clahe(segmented)
    return enhanced


def load_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.resize(image, IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk the dataset directory. Each immediate subfolder is treated as a
    class. Returns:
        X : np.ndarray of shape (N, 224, 224, 3), float32, RAW pixel values
            in [0, 255] - MobileNetV2's preprocess_input() is applied later,
            NOT here, since it needs to be applied consistently at both
            training and inference time.
        y : np.ndarray of string labels, shape (N,)
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
            except Exception as exc:
                print(f"    Skipping {img_path.name} due to error: {exc}")
                continue

            # BGR (OpenCV) -> RGB. Keep raw [0, 255] float values here;
            # MobileNetV2's preprocess_input() handles scaling to [-1, 1].
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype("float32")
            X.append(rgb)
            y.append(label)

    return np.array(X, dtype="float32"), np.array(y)


# ---------------------------------------------------------------------------
# Model architecture: MobileNetV2 backbone + custom classification head
# ---------------------------------------------------------------------------
def build_transfer_model(input_shape: tuple, num_classes: int):
    """
    Returns (model, base_model). base_model is the MobileNetV2 backbone,
    kept as a separate reference so it can be unfrozen later for fine-tuning.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,       # drop MobileNetV2's original 1000-class ImageNet head
        weights="imagenet",
    )
    base_model.trainable = False  # freeze for Phase 1

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


# ---------------------------------------------------------------------------
# Augmentation + tf.data pipeline (replaces the old ImageDataGenerator,
# which was removed in Keras 3 / TensorFlow 2.16+)
# ---------------------------------------------------------------------------
def build_augmentation_layer() -> tf.keras.Sequential:
    """
    Modest augmentation - apples don't flip upside down in practice, but
    rotation/shift/zoom/brightness variation helps generalize across
    lighting and camera angle differences. Operates on raw [0, 255] images,
    BEFORE preprocess_input is applied.
    """
    return tf.keras.Sequential(
        [
            layers.RandomRotation(20 / 360, fill_mode="nearest"),
            layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
            layers.RandomZoom(0.15, fill_mode="nearest"),
            layers.RandomFlip("horizontal"),
            layers.RandomBrightness(0.2, value_range=(0, 255)),
        ],
        name="augmentation",
    )


def make_dataset(
    X: np.ndarray, y: np.ndarray, batch_size: int,
    augment: bool = False, shuffle: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline: batch -> (optional) augment -> MobileNetV2
    preprocess_input -> prefetch. Replaces ImageDataGenerator.flow().
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=RANDOM_STATE, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    if augment:
        aug_layer = build_augmentation_layer()
        ds = ds.map(lambda imgs, labels: (aug_layer(imgs, training=True), labels),
                    num_parallel_calls=AUTOTUNE)

    ds = ds.map(lambda imgs, labels: (preprocess_input(imgs), labels),
                num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_cnn(X: np.ndarray, y: np.ndarray):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_categorical,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    y_temp_labels = np.argmax(y_temp, axis=1)
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=y_temp_labels,
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    y_train_labels = np.argmax(y_train, axis=1)
    class_weights_array = compute_class_weight(
        "balanced", classes=np.unique(y_train_labels), y=y_train_labels
    )
    class_weights = dict(enumerate(class_weights_array))

    model, base_model = build_transfer_model(input_shape=X.shape[1:], num_classes=num_classes)
    model.summary()

    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, augment=True, shuffle=True)
    val_ds = make_dataset(X_val, y_val, BATCH_SIZE, augment=False, shuffle=False)

    # Test set: kept as plain preprocessed arrays (not a tf.data pipeline)
    # since it's only used once for a final model.predict() call.
    X_test_processed = preprocess_input(X_test.copy())

    callbacks_phase1 = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    print("\n=== Phase 1: training classification head (MobileNetV2 frozen) ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    # --- Phase 2: fine-tuning ---
    print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False  # keep earlier (more generic) layers frozen

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # much lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7),
        ModelCheckpoint(str(MODEL_OUTPUT_PATH), monitor="val_accuracy", save_best_only=True),
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    # --- Plot training curves across both phases ---
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]
    phase_boundary = len(history1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(acc, label="Train")
    axes[0].plot(val_acc, label="Validation")
    axes[0].axvline(phase_boundary, color="gray", linestyle="--", label="Fine-tuning starts")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(loss, label="Train")
    axes[1].plot(val_loss, label="Validation")
    axes[1].axvline(phase_boundary, color="gray", linestyle="--", label="Fine-tuning starts")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(TRAINING_CURVES_PATH)
    plt.close()
    print(f"Training curves saved to: {TRAINING_CURVES_PATH}")

    # --- Evaluate on held-out test set ---
    y_pred_probs = model.predict(X_test_processed)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    test_acc = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {test_acc:.4f}\n")

    display_labels = get_display_order(list(encoder.classes_))
    display_indices = [np.where(encoder.classes_ == c)[0][0] for c in display_labels]

    print(classification_report(
        y_true, y_pred,
        labels=display_indices,
        target_names=display_labels,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=display_indices)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=display_labels, yticklabels=display_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - CNN (MobileNetV2 Transfer Learning)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    return model, encoder


# ---------------------------------------------------------------------------
# Lightweight cross-validation sanity check
# ---------------------------------------------------------------------------
def cross_validate_cnn(X: np.ndarray, y: np.ndarray) -> None:
    """
    Stratified k-fold sanity check. Uses only a frozen-backbone phase (no
    fine-tuning) per fold to keep this fast - it's a consistency check, not
    a full retrain.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    print(f"\n--- Stratified {N_FOLDS}-Fold Cross-Validation (sanity check, {CV_EPOCHS} epochs/fold, frozen backbone only) ---")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), start=1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold = to_categorical(y_encoded[train_idx], num_classes=num_classes)
        y_val_fold = to_categorical(y_encoded[val_idx], num_classes=num_classes)

        fold_model, _ = build_transfer_model(input_shape=X.shape[1:], num_classes=num_classes)
        train_ds = make_dataset(X_train_fold, y_train_fold, BATCH_SIZE, augment=True, shuffle=True)
        val_ds = make_dataset(X_val_fold, y_val_fold, BATCH_SIZE, augment=False, shuffle=False)

        fold_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=CV_EPOCHS,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0,
        )

        _, val_acc = fold_model.evaluate(val_ds, verbose=0)
        fold_scores.append(val_acc)
        print(f"  Fold {fold_idx}: {val_acc:.4f}")

    fold_scores = np.array(fold_scores)
    print(f"  Mean accuracy: {fold_scores.mean():.4f}")
    print(f"  Std deviation: {fold_scores.std():.4f}")

    if fold_scores.std() > 0.05:
        print(
            "  NOTE: fold accuracies vary noticeably. The single train/test "
            "split earlier may have been optimistic, or some folds contain "
            "harder/more ambiguous images."
        )
    elif fold_scores.mean() > 0.98 and fold_scores.std() < 0.02:
        print(
            "  NOTE: accuracy is consistently very high across all folds. "
            "Cross-check against SVM.py / RandomForest.py results and verify "
            "there's no data leakage (near-duplicate images split across folds)."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading dataset and preprocessing images...")
    X, y = load_dataset(DATASET_DIR)
    print(f"\nTotal samples: {len(X)} | Image shape: {X.shape[1:] if len(X) else 'N/A'}\n")

    model, encoder = train_cnn(X, y)

    # This retrains a fresh (frozen-backbone) model per fold, so it will
    # take a while. Comment out if you just want the main result quickly.
    cross_validate_cnn(X, y)

    joblib.dump({"label_encoder": encoder}, OUTPUT_DIR / "cnn_label_encoder.pkl")
    print(f"\nTrained model saved to: {MODEL_OUTPUT_PATH}")
    print(f"Label encoder saved to: {OUTPUT_DIR / 'cnn_label_encoder.pkl'}")


if __name__ == "__main__":
    main()