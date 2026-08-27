from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import streamlit as st


UI_DIRECTORY = Path(__file__).resolve().parent
if str(UI_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(UI_DIRECTORY))

from image_pipeline import ImageAnalysis, analyse_image

st.set_page_config(
    page_title="Apple Ripeness System",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> None:
    """Display the prototype navigation menu."""
    with st.sidebar:
        st.title("🍎 Apple Ripeness")
        st.caption("Recognition System")
        st.divider()

        st.radio(
            "Navigation",
            ["Scan Apple", "History", "About"],
            index=0,
            disabled=True,
            help="Additional pages will be connected later.",
        )

        st.divider()
        st.caption("Image processing connected")


def render_scan_input() -> bytes | None:
    """Display input controls and return the selected still-image bytes."""
    st.subheader("Choose an input method")

    camera_tab, upload_tab, live_camera_tab = st.tabs(
        ["📷 Take a photo", "🖼️ Upload an image", "🎥 Live camera"]
    )

    camera_image = None
    uploaded_image = None

    with camera_tab:
        st.write("Use your device camera to take a clear picture of the apple.")
        camera_image = st.camera_input("Take an apple photo")

    with upload_tab:
        st.write("Select a JPG or PNG image from your device.")
        uploaded_image = st.file_uploader(
            "Upload an apple image",
            type=["jpg", "jpeg", "png"],
            key="apple_image",
        )
        if uploaded_image is not None:
            st.image(
                uploaded_image,
                caption="Selected image",
                use_container_width=True,
            )

    with live_camera_tab:
        st.write(
            "Start the live camera to detect and classify apples continuously."
        )
        try:
            from module.Module1.live_camera import live_camera_classification
        except (ImportError, ValueError) as exc:
            st.error("The live camera dependencies are not available.")
            st.caption(str(exc))
            st.code(
                "python -m pip install streamlit-webrtc av scikit-image",
                language="powershell",
            )
        else:
            live_camera_classification()

    # An uploaded image takes priority if an older camera capture is also
    # retained by Streamlit's widget state.
    selected_image = uploaded_image or camera_image
    if selected_image is None:
        return None

    return selected_image.getvalue()


def _ordered_probabilities(probabilities: dict[str, float]):
    class_order = ["20%", "40%", "60%", "80%", "100%", "Overripe"]
    known = [label for label in class_order if label in probabilities]
    extras = [label for label in probabilities if label not in class_order]
    return [(label, probabilities[label]) for label in known + extras]


def render_processing_stages(analysis: ImageAnalysis) -> None:
    """Show Module 1 preprocessing output for inspection."""
    steps = analysis.preprocessing

    with st.expander("View preprocessing stages"):
        stages = [
            ("Original", steps.original, "BGR"),
            ("Resized", steps.resized, "BGR"),
            ("CLAHE", steps.clahe, "BGR"),
            ("HSV candidate mask", steps.hsv_candidate_mask, None),
            ("GrabCut mask", steps.grabcut_mask, None),
            ("Refined mask", steps.refined_mask, None),
            ("Processed output", analysis.processed, "BGR"),
        ]

        for row_start in range(0, len(stages), 2):
            columns = st.columns(2)
            for column, (caption, image, channels) in zip(
                columns,
                stages[row_start:row_start + 2],
            ):
                options = {
                    "caption": caption,
                    "use_container_width": True,
                }
                if channels is None:
                    options["clamp"] = True
                else:
                    options["channels"] = channels
                column.image(image, **options)

        foreground_percent = steps.foreground_ratio * 100
        if steps.segmentation_success:
            st.success(
                f"Background segmentation succeeded ({foreground_percent:.1f}% foreground)."
            )
        else:
            reason = steps.fallback_reason or "No reliable foreground was found."
            st.warning(f"Segmentation fallback used: {reason}")


def render_completed_analysis(analysis: ImageAnalysis) -> None:
    """Display detection and classification results."""
    if not analysis.apples:
        st.warning(analysis.detection_message or "No apple was detected.")
        st.image(
            analysis.annotated,
            channels="BGR",
            caption="Detection result",
            use_container_width=True,
        )
        render_processing_stages(analysis)
        return

    first_apple = analysis.apples[0]
    apple_count_column, ripeness_column, confidence_column = st.columns(3)
    apple_count_column.metric("Apples found", len(analysis.apples))
    ripeness_column.metric("First result", first_apple.label)
    confidence_column.metric("Confidence", f"{first_apple.confidence:.1%}")

    st.success(analysis.detection_message)
    st.image(
        analysis.annotated,
        channels="BGR",
        caption="Detected apples",
        use_container_width=True,
    )

    for apple in analysis.apples:
        with st.container(border=True):
            crop_column, details_column = st.columns([2, 3])

            with crop_column:
                st.image(
                    apple.crop,
                    channels="BGR",
                    caption=f"Apple #{apple.apple_id}",
                    use_container_width=True,
                )

            with details_column:
                st.subheader(f"Apple #{apple.apple_id}")

                if apple.classification_error:
                    st.error("Ripeness classification is currently unavailable.")
                    st.caption(apple.classification_error)
                    continue

                st.metric("Ripeness level", apple.label)
                st.write(f"Confidence: **{apple.confidence:.1%}**")
                st.progress(min(max(apple.confidence, 0.0), 1.0))

                if apple.probabilities:
                    with st.expander("Class probabilities"):
                        for label, probability in _ordered_probabilities(
                            apple.probabilities
                        ):
                            st.write(f"{label}: {probability:.1%}")

    render_processing_stages(analysis)


def render_result_panel(image_bytes: bytes | None) -> None:
    """Run the pipeline on demand and display its latest result."""
    st.subheader("Analysis result")

    image_hash = (
        hashlib.sha256(image_bytes).hexdigest()
        if image_bytes is not None
        else None
    )

    if image_hash != st.session_state.get("analysis_input_hash"):
        st.session_state["analysis_input_hash"] = image_hash
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_error", None)

    with st.container(border=True):
        if image_bytes is None:
            st.info("Capture or upload an apple image to begin.")

        analyse_clicked = st.button(
            "Analyse apple",
            type="primary",
            use_container_width=True,
            disabled=image_bytes is None,
        )

        if analyse_clicked and image_bytes is not None:
            try:
                with st.spinner("Preprocessing, detecting, and classifying the apple…"):
                    st.session_state["analysis_result"] = analyse_image(
                        image_bytes
                    )
                st.session_state.pop("analysis_error", None)
            except (ValueError, cv2.error, KeyError) as exc:
                st.session_state.pop("analysis_result", None)
                st.session_state["analysis_error"] = str(exc)
            except Exception as exc:
                st.session_state.pop("analysis_result", None)
                st.session_state["analysis_error"] = (
                    "The analysis could not be completed. "
                    f"Technical detail: {exc}"
                )

        analysis_error = st.session_state.get("analysis_error")
        if analysis_error:
            st.error(analysis_error)

        analysis = st.session_state.get("analysis_result")
        if analysis is not None:
            render_completed_analysis(analysis)


def main() -> None:
    """Start the standalone user-interface prototype."""
    render_sidebar()

    st.title("Apple Ripeness Recognition")
    st.write(
        "Take a photo, upload an image, or use the live camera to identify "
        "an apple's ripeness level."
    )

    st.divider()

    input_column, result_column = st.columns([3, 2], gap="large")

    with input_column:
        image_bytes = render_scan_input()

    with result_column:
        render_result_panel(image_bytes)

    st.divider()
    st.caption(
        "Photo preprocessing, apple detection, ripeness classification, and live "
        "camera analysis are connected. Authentication and database storage will "
        "be added later."
    )


if __name__ == "__main__":
    main()
