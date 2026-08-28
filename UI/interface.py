from __future__ import annotations

import hashlib
import html
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import cv2
import streamlit as st


UI_DIRECTORY = Path(__file__).resolve().parent
if str(UI_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(UI_DIRECTORY))

from image_pipeline import ImageAnalysis, analyse_image
from analysis_repository import (
    AnalysisRepositoryError,
    download_analysis_image,
    list_analyses,
    save_analysis,
)
from login import (
    current_user_name,
    is_logged_in,
    logout,
    require_login_or_guest,
)


@dataclass(frozen=True)
class SelectedImage:
    data: bytes
    input_method: str
    name: str


def load_app_styles() -> None:
    """Load the application stylesheet on every Streamlit rerun."""
    st.html(UI_DIRECTORY / "styles.css")


APPLE_MARK_HTML = '<span class="apple-shape"><i></i></span>'

def render_sidebar() -> str:
    """Display the prototype navigation menu."""
    with st.sidebar:
        st.html(
            '<div class="brand-lockup">'
            f'<div class="brand-mark">{APPLE_MARK_HTML}</div>'
            '<div><strong>Apple Ripeness</strong>'
            '<span>Vision assessment</span></div>'
            '</div>'
        )
        st.divider()

        if is_logged_in():
            st.html(
                '<div class="session-chip session-chip--signed">'
                '<span></span><div><b>Signed in</b>'
                f'<small>{html.escape(current_user_name())}</small></div></div>'
            )
            st.button("Log out", width="stretch", on_click=logout)
        else:
            st.html(
                '<div class="session-chip"><span></span>'
                '<div><b>Guest mode</b><small>History off</small></div></div>'
            )
            st.button(
                "Sign in",
                width="stretch",
                on_click=_return_guest_to_login,
            )

        st.divider()

        navigation_labels = {
            "scan": ":material/camera_enhance: Scan apple",
            "history": ":material/history: History",
            "about": ":material/info: About",
        }
        selected_page = st.radio(
            "Navigation",
            options=navigation_labels,
            format_func=navigation_labels.get,
            index=0,
            label_visibility="collapsed",
        )

    return selected_page


def _return_guest_to_login() -> None:
    """Leave guest mode so the login screen is shown on the next rerun."""
    st.session_state.guest_mode = False


def _camera_context_is_secure() -> bool:
    """Return whether browser camera APIs may run on the current app URL."""
    try:
        parsed_url = urlparse(str(st.context.url))
    except (AttributeError, RuntimeError, ValueError):
        return True

    local_hosts = {"localhost", "127.0.0.1", "::1"}
    return parsed_url.scheme == "https" or parsed_url.hostname in local_hosts


def _render_camera_security_notice() -> None:
    if not _camera_context_is_secure():
        st.warning(
            "Your browser blocks camera access on this HTTP network address. "
            "Open the app through an HTTPS URL to use photo capture or the "
            "live camera. Image upload is still available."
        )


def _clear_analysis_state() -> None:
    """Remove still-image results when the user changes input context."""
    st.session_state["analysis_input_hash"] = None
    st.session_state.pop("analysis_batch_results", None)
    st.session_state.pop("analysis_batch_errors", None)
    st.session_state.pop("analysis_saved_count", None)
    st.session_state.pop("analysis_save_errors", None)


def render_scan_input() -> list[SelectedImage]:
    """Display the selected input control and return its still images."""
    st.subheader("Input")

    input_method = st.segmented_control(
        "Input method",
        options=["photo", "upload", "live"],
        default="photo",
        format_func={
            "photo": ":material/photo_camera: Take a photo",
            "upload": ":material/upload_file: Upload images",
            "live": ":material/videocam: Live camera",
        }.get,
        key="scan_input_method",
        on_change=_clear_analysis_state,
        label_visibility="collapsed",
        width="stretch",
    )

    if input_method == "photo":
        camera_image = None
        _render_camera_security_notice()

        photo_camera_state_key = "apple_photo_camera_requested"
        if photo_camera_state_key not in st.session_state:
            st.session_state[photo_camera_state_key] = False

        def start_photo_camera() -> None:
            st.session_state[photo_camera_state_key] = True

        def stop_photo_camera() -> None:
            st.session_state[photo_camera_state_key] = False

        start_column, stop_column = st.columns(2)
        with start_column:
            st.button(
                "Start camera",
                key="apple-photo-start",
                on_click=start_photo_camera,
                disabled=st.session_state[photo_camera_state_key],
                width="stretch",
                type="primary",
            )
        with stop_column:
            st.button(
                "Stop camera",
                key="apple-photo-stop",
                on_click=stop_photo_camera,
                disabled=not st.session_state[photo_camera_state_key],
                width="stretch",
            )

        st.html(
            '<div class="camera-guide"><span></span>'
            "Center one apple</div>"
        )
        if st.session_state[photo_camera_state_key]:
            with st.container(key="camera-scan-stage"):
                camera_image = st.camera_input(
                    "Take an apple photo",
                    key="apple-photo-camera",
                    resolution="720p",
                )
        if camera_image is not None:
            return [
                SelectedImage(
                    data=camera_image.getvalue(),
                    input_method="camera",
                    name="Camera photo",
                )
            ]
        return []

    if input_method == "upload":
        uploaded_images = st.file_uploader(
            "Drop apple images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="apple_images",
        )
        if uploaded_images:
            st.caption(f"{len(uploaded_images)} image(s) selected")
            preview_columns = st.columns(min(len(uploaded_images), 3))
            for index, uploaded_image in enumerate(uploaded_images):
                with preview_columns[index % len(preview_columns)]:
                    st.image(
                        uploaded_image,
                        caption=uploaded_image.name,
                        width="stretch",
                    )

            return [
                SelectedImage(
                    data=uploaded_image.getvalue(),
                    input_method="upload",
                    name=uploaded_image.name,
                )
                for uploaded_image in uploaded_images
            ]
        return []

    if input_method == "live":
        _render_camera_security_notice()
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

    return []


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
                    "width": "stretch",
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
            width="stretch",
        )
        render_processing_stages(analysis)
        return

    first_apple = analysis.apples[0]
    safe_first_label = html.escape(first_apple.label)
    st.html(
        '<div class="result-summary">'
        '<div class="result-stat">'
        '<span>Apples found</span>'
        f'<strong>{len(analysis.apples)}</strong>'
        "</div>"
        '<div class="result-stat result-stat--primary">'
        '<span>First result</span>'
        f'<strong>{safe_first_label}</strong>'
        "</div>"
        '<div class="result-stat">'
        '<span>Confidence</span>'
        f'<strong>{first_apple.confidence:.1%}</strong>'
        "</div>"
        "</div>"
    )

    st.success(analysis.detection_message)
    st.image(
        analysis.annotated,
        channels="BGR",
        caption="Detected apples",
        width="stretch",
    )

    for apple in analysis.apples:
        with st.container(border=True):
            crop_column, details_column = st.columns([2, 3])

            with crop_column:
                st.image(
                    apple.crop,
                    channels="BGR",
                    caption=f"Apple #{apple.apple_id}",
                    width="stretch",
                )

            with details_column:
                st.subheader(f"Apple #{apple.apple_id}")

                if apple.classification_error:
                    st.error("Ripeness classification is currently unavailable.")
                    st.caption(apple.classification_error)
                    continue

                st.html(
                    '<div class="ripeness-pill">'
                    f"{html.escape(apple.label)}"
                    "</div>"
                )
                st.write(f"Confidence: **{apple.confidence:.1%}**")
                st.progress(min(max(apple.confidence, 0.0), 1.0))

                if apple.probabilities:
                    with st.expander("Class probabilities"):
                        for label, probability in _ordered_probabilities(
                            apple.probabilities
                        ):
                            st.write(f"{label}: {probability:.1%}")

    render_processing_stages(analysis)


def _selected_images_hash(selected_images: list[SelectedImage]) -> str | None:
    if not selected_images:
        return None

    digest = hashlib.sha256()
    for selected_image in selected_images:
        digest.update(selected_image.input_method.encode("utf-8"))
        digest.update(selected_image.name.encode("utf-8", errors="replace"))
        digest.update(selected_image.data)
    return digest.hexdigest()


def render_result_panel(selected_images: list[SelectedImage]) -> None:
    """Run the pipeline on demand for one image or a complete upload batch."""
    st.subheader("Result")

    image_hash = _selected_images_hash(selected_images)

    if image_hash != st.session_state.get("analysis_input_hash"):
        st.session_state["analysis_input_hash"] = image_hash
        st.session_state.pop("analysis_batch_results", None)
        st.session_state.pop("analysis_batch_errors", None)
        st.session_state.pop("analysis_saved_count", None)
        st.session_state.pop("analysis_save_errors", None)

    with st.container(border=True):
        if not selected_images:
            st.info("Waiting for an image")
        elif len(selected_images) > 1:
            st.info(f"Ready to analyse {len(selected_images)} uploaded images.")

        analyse_clicked = st.button(
            (
                f"Analyse {len(selected_images)} images"
                if len(selected_images) > 1
                else "Analyse"
            ),
            type="primary",
            width="stretch",
            disabled=not selected_images,
        )

        if analyse_clicked and selected_images:
            scan_effect = st.empty()
            scan_effect.html(
                '<div class="analysis-scanner">Scanning colour, shape, and '
                "ripeness features…</div>"
            )
            completed_results: list[ImageAnalysis | None] = []
            analysis_errors: list[str | None] = []
            save_errors: list[str] = []
            saved_count = 0

            progress = st.progress(0.0) if len(selected_images) > 1 else None
            for index, selected_image in enumerate(selected_images, start=1):
                try:
                    with st.spinner(
                        f"Analysing image {index} of {len(selected_images)}…"
                    ):
                        completed_analysis = analyse_image(selected_image.data)
                    completed_results.append(completed_analysis)
                    analysis_errors.append(None)

                    if is_logged_in():
                        try:
                            saved_id = save_analysis(
                                image_bytes=selected_image.data,
                                input_method=selected_image.input_method,
                                analysis=completed_analysis,
                            )
                            if saved_id:
                                saved_count += 1
                        except AnalysisRepositoryError as exc:
                            save_errors.append(f"{selected_image.name}: {exc}")
                except (ValueError, cv2.error, KeyError) as exc:
                    completed_results.append(None)
                    analysis_errors.append(str(exc))
                except Exception as exc:
                    completed_results.append(None)
                    analysis_errors.append(
                        "The analysis could not be completed. "
                        f"Technical detail: {exc}"
                    )
                finally:
                    if progress is not None:
                        progress.progress(index / len(selected_images))

            st.session_state["analysis_batch_results"] = completed_results
            st.session_state["analysis_batch_errors"] = analysis_errors
            st.session_state["analysis_saved_count"] = saved_count
            st.session_state["analysis_save_errors"] = save_errors
            if progress is not None:
                progress.empty()
            scan_effect.empty()

        saved_count = st.session_state.get("analysis_saved_count", 0)
        if saved_count:
            st.success(
                f"{saved_count} analysis result(s) saved to your private History."
            )

        for save_error in st.session_state.get("analysis_save_errors", []):
            st.warning(save_error)

        completed_results = st.session_state.get("analysis_batch_results", [])
        analysis_errors = st.session_state.get("analysis_batch_errors", [])

        for index, selected_image in enumerate(selected_images):
            analysis = (
                completed_results[index] if index < len(completed_results) else None
            )
            analysis_error = (
                analysis_errors[index] if index < len(analysis_errors) else None
            )

            if len(selected_images) > 1 and (analysis is not None or analysis_error):
                st.divider()
                st.subheader(f"Image {index + 1}: {selected_image.name}")

            if analysis_error:
                st.error(analysis_error)
            elif analysis is not None:
                render_completed_analysis(analysis)


def _format_history_time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        malaysia_time = parsed.astimezone(timezone(timedelta(hours=8)))
        return malaysia_time.strftime("%d %b %Y, %I:%M %p")
    except (TypeError, ValueError):
        return value


def render_history_page() -> None:
    """Show the signed-in user's saved images and analysis results."""
    st.title("Analysis History")
    st.write("Review your most recent saved apple assessments.")

    if not is_logged_in():
        st.info("Sign in with Google to save and retrieve analysis history.")
        st.button(
            "Return to sign in",
            type="primary",
            on_click=_return_guest_to_login,
        )
        return

    try:
        with st.spinner("Loading your private history…"):
            history = list_analyses()
    except AnalysisRepositoryError as exc:
        st.error(str(exc))
        return

    if not history:
        st.info("No saved analyses yet. Scan an apple to create your first one.")
        return

    st.caption(f"Showing your {len(history)} most recent analyses")

    for record in history:
        with st.container(border=True):
            image_column, details_column = st.columns([2, 3], gap="large")

            with image_column:
                try:
                    image_bytes = download_analysis_image(record["image_path"])
                    st.image(image_bytes, width="stretch")
                except AnalysisRepositoryError as exc:
                    st.warning(str(exc))

            with details_column:
                st.subheader(_format_history_time(record["created_at"]))
                st.caption(
                    f"Input: {record['input_method'].replace('_', ' ').title()}"
                )
                st.metric("Apples found", record["apple_count"])

                if record.get("detection_message"):
                    st.write(record["detection_message"])

                results = sorted(
                    record.get("analysis_results") or [],
                    key=lambda item: item["apple_number"],
                )
                if not results:
                    st.info("No apples were detected in this image.")

                for result in results:
                    st.markdown(
                        f"**Apple #{result['apple_number']}: "
                        f"{result['ripeness_label']}**"
                    )
                    if result.get("classification_error"):
                        st.caption(result["classification_error"])
                    else:
                        confidence = float(result["confidence"])
                        st.write(f"Confidence: {confidence:.1%}")
                        st.progress(min(max(confidence, 0.0), 1.0))

                        probabilities = result.get("probabilities") or {}
                        if probabilities:
                            with st.expander(
                                f"Apple #{result['apple_number']} probabilities"
                            ):
                                for label, probability in _ordered_probabilities(
                                    probabilities
                                ):
                                    st.write(f"{label}: {float(probability):.1%}")


def render_about_page() -> None:
    st.title("About the System")
    st.write(
        "This application combines image preprocessing, apple detection, "
        "ripeness classification, Google authentication, and private "
        "Supabase history storage."
    )
    st.info(
        "Guest scans remain temporary. Signed-in scans are stored under the "
        "user's own account and protected by database and Storage policies."
    )


def main() -> None:
    """Start the standalone user-interface prototype."""
    # This module is imported by the root ``app.py`` entry point and remains
    # cached between Streamlit widget reruns. Keep every Streamlit page setup
    # call inside ``main`` so it is emitted again on each rerun.
    st.set_page_config(
        page_title="Apple Ripeness System",
        page_icon="🍎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Do not render classifier controls until the visitor signs in with Google
    # or explicitly chooses guest mode.
    require_login_or_guest()
    load_app_styles()

    selected_page = render_sidebar()

    if selected_page == "history":
        render_history_page()
        return
    if selected_page == "about":
        render_about_page()
        return

    st.html(
        '<section class="app-hero">'
        '<div class="hero-copy">'
        '<div class="hero-kicker">Computer vision</div>'
        "<h1>Apple Ripeness</h1>"
        "<p>Capture. Detect. Assess.</p>"
        "</div>"
        '<div class="hero-visual" aria-hidden="true">'
        '<div class="scan-ring scan-ring--outer"></div>'
        '<div class="scan-ring scan-ring--inner"></div>'
        f'<div class="hero-apple">{APPLE_MARK_HTML}</div>'
        '<div class="vision-chip">AI vision <b>online</b></div>'
        '</div>'
        "</section>"
    )

    with st.container(key="scan-workspace"):
        input_column, result_column = st.columns([1.42, 1], gap="large")

        with input_column:
            selected_images = render_scan_input()

        with result_column:
            render_result_panel(selected_images)
