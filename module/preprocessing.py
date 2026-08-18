import cv2
import numpy as np


def resize_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "resize_image expects a BGR image with 3 channels."
        )

    return cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )


def apply_clahe(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "apply_clahe expects a BGR image with 3 channels."
        )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge(
        (enhanced_l, a, b)
    )

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )


def apply_closing(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        raise ValueError("Input mask is empty.")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


def apply_opening(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


def create_apple_candidate_mask(
    image: np.ndarray
) -> np.ndarray:

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    red_lower_1 = np.array(
        [0, 70, 40],
        dtype=np.uint8
    )

    red_upper_1 = np.array(
        [12, 255, 255],
        dtype=np.uint8
    )

    red_lower_2 = np.array(
        [165, 70, 40],
        dtype=np.uint8
    )

    red_upper_2 = np.array(
        [179, 255, 255],
        dtype=np.uint8
    )

    red_mask_1 = cv2.inRange(
        hsv,
        red_lower_1,
        red_upper_1
    )

    red_mask_2 = cv2.inRange(
        hsv,
        red_lower_2,
        red_upper_2
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )

    green_lower = np.array(
        [30, 45, 35],
        dtype=np.uint8
    )

    green_upper = np.array(
        [90, 255, 255],
        dtype=np.uint8
    )

    green_mask = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    yellow_lower = np.array(
        [15, 60, 60],
        dtype=np.uint8
    )

    yellow_upper = np.array(
        [38, 255, 255],
        dtype=np.uint8
    )

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    candidate_mask = cv2.bitwise_or(
        red_mask,
        green_mask
    )

    candidate_mask = cv2.bitwise_or(
        candidate_mask,
        yellow_mask
    )

    candidate_mask = apply_opening(
        candidate_mask
    )

    candidate_mask = apply_closing(
        candidate_mask
    )

    return candidate_mask


def segment_background_with_steps(
    image: np.ndarray
) -> dict:

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    height, width = image.shape[:2]

    candidate_mask = create_apple_candidate_mask(
        image
    )

    grabcut_labels = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    grabcut_labels[
        candidate_mask > 0
    ] = cv2.GC_PR_FGD

    border = max(
        1,
        int(min(height, width) * 0.01)
    )

    grabcut_labels[
        :border,
        :
    ] = cv2.GC_BGD

    grabcut_labels[
        -border:,
        :
    ] = cv2.GC_BGD

    grabcut_labels[
        :,
        :border
    ] = cv2.GC_BGD

    grabcut_labels[
        :,
        -border:
    ] = cv2.GC_BGD

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    candidate_pixels = cv2.countNonZero(
        candidate_mask
    )

    if candidate_pixels == 0:
        empty_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        return {
            "candidate_mask": candidate_mask,
            "grabcut_mask": empty_mask,
            "closed_mask": empty_mask,
            "segmented": image.copy(),
            "segmentation_success": False
        }

    cv2.grabCut(
        image,
        grabcut_labels,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    grabcut_mask = np.where(
        (
            grabcut_labels == cv2.GC_FGD
        )
        |
        (
            grabcut_labels == cv2.GC_PR_FGD
        ),
        255,
        0
    ).astype(np.uint8)

    grabcut_mask = apply_opening(
        grabcut_mask
    )

    closed_mask = apply_closing(
        grabcut_mask
    )

    foreground_pixels = cv2.countNonZero(
        closed_mask
    )

    total_pixels = height * width

    foreground_ratio = (
        foreground_pixels
        / total_pixels
    )

    MIN_FOREGROUND_RATIO = 0.01

    if foreground_ratio < MIN_FOREGROUND_RATIO:
        segmented_image = image.copy()
        segmentation_success = False

    else:
        segmented_image = cv2.bitwise_and(
            image,
            image,
            mask=closed_mask
        )

        segmentation_success = True

    return {
        "candidate_mask": candidate_mask,
        "grabcut_mask": grabcut_mask,
        "closed_mask": closed_mask,
        "segmented": segmented_image,
        "segmentation_success": segmentation_success
    }


def segment_background(
    image: np.ndarray
) -> np.ndarray:

    results = segment_background_with_steps(
        image
    )

    return results["segmented"]


def preprocess_image_with_steps(
    image: np.ndarray
) -> dict:

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    resized = resize_image(
        image
    )

    clahe = apply_clahe(
        resized
    )

    segmentation = segment_background_with_steps(
        clahe
    )

    return {
        "resized": resized,
        "clahe": clahe,
        "candidate_mask": segmentation[
            "candidate_mask"
        ],
        "grabcut_mask": segmentation[
            "grabcut_mask"
        ],
        "closed_mask": segmentation[
            "closed_mask"
        ],
        "segmentation_success": segmentation[
            "segmentation_success"
        ],
        "final": segmentation[
            "segmented"
        ]
    }


def preprocess_image(
    image: np.ndarray
) -> np.ndarray:

    return preprocess_image_with_steps(
        image
    )["final"]