from __future__ import annotations

import queue
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

import av
import cv2
import numpy as np
import streamlit as st
from aiortc.codecs import h264 as aiortc_h264
from aiortc.codecs import vpx as aiortc_vpx
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from module.classification import predict_ripeness
from module.Module1.preprocessing import create_apple_candidate_mask


# ============================================================
# Configuration
# ============================================================

PREDICTION_INTERVAL_SECONDS = 0.5      # keep cloud CPU free for real-time video delivery
CONFIDENCE_THRESHOLD = 0.55            # temporal agreement supplies an additional confidence gate
STABILITY_WINDOW = 3                   # recent predictions used to suppress flicker
MIN_CANDIDATE_RATIO = 0.05             # FR-LC-14: reject empty / no-apple ROI
MAX_TRACK_DISTANCE = 180               # px: tolerate normal movement between inference cycles
MAX_MISSED_CYCLES = 3                  # detection cycles an apple can vanish before its track is dropped
DETECTION_MAX_DIMENSION = 448          # detect on a small, aspect-preserving copy
MAX_APPLES_PER_CYCLE = 4               # cap classification work per cycle so worst-case latency stays bounded
# Keep an HD live preview while the detection worker runs at a lower rate so
# classification does not block video delivery.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
WEBRTC_DEFAULT_BITRATE = 800_000
WEBRTC_MAX_BITRATE = 1_500_000


def _get_rtc_configuration() -> dict:
    """Build ICE configuration for local and deployed WebRTC connections.

    Public STUN servers are sufficient on most networks. Deployments behind a
    restrictive firewall or symmetric NAT can additionally provide TURN_URL,
    TURN_USERNAME and TURN_CREDENTIAL as environment variables.
    """
    ice_servers = [
        {
            "urls": [
                "stun:stun.l.google.com:19302",
                "stun:stun.cloudflare.com:3478",
            ]
        }
    ]

    turn_url = os.getenv("TURN_URL", "").strip()
    turn_username = os.getenv("TURN_USERNAME", "").strip()
    turn_credential = os.getenv("TURN_CREDENTIAL", "").strip()
    if turn_url and turn_username and turn_credential:
        ice_servers.append(
            {
                "urls": turn_url,
                "username": turn_username,
                "credential": turn_credential,
            }
        )

    return {"iceServers": ice_servers}


# Keep the return-path bitrate appropriate for the lightweight live preview.
# This avoids buffering several seconds of video on slower mobile links.
aiortc_vpx.DEFAULT_BITRATE = WEBRTC_DEFAULT_BITRATE
aiortc_vpx.MAX_BITRATE = WEBRTC_MAX_BITRATE
aiortc_h264.DEFAULT_BITRATE = WEBRTC_DEFAULT_BITRATE
aiortc_h264.MAX_BITRATE = WEBRTC_MAX_BITRATE


@dataclass
class ApplePrediction:
    """Outcome of the stabilization/confidence check for one tracked apple."""

    track_id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h — in full-frame pixel coordinates
    status: str    # "analyzing" | "low_confidence" | "confirmed"
    message: str = ""
    label: str | None = None
    confidence: float = 0.0


class _Track:
    """Per-apple state carried across detection cycles for stabilization (FR-LC-12)."""

    __slots__ = ("id", "bbox", "recent", "missed")

    def __init__(self, track_id: int, bbox: tuple[int, int, int, int]) -> None:
        self.id = track_id
        self.bbox = bbox
        self.recent: deque[tuple[str, float] | None] = deque(maxlen=STABILITY_WINDOW)
        self.missed = 0


def _normalize_live_frame(img: np.ndarray) -> np.ndarray:
    """Return a low-latency 16:9 frame at the configured output resolution.

    Browsers can occasionally deliver a fallback size despite the requested
    media constraints. Centre-cropping before resizing preserves the image's
    proportions while keeping encoding cost and network use predictable.
    """

    height, width = img.shape[:2]
    target_ratio = CAMERA_WIDTH / CAMERA_HEIGHT
    source_ratio = width / height

    if source_ratio > target_ratio:
        crop_width = int(round(height * target_ratio))
        x_offset = (width - crop_width) // 2
        img = img[:, x_offset:x_offset + crop_width]
    elif source_ratio < target_ratio:
        crop_height = int(round(width / target_ratio))
        y_offset = (height - crop_height) // 2
        img = img[y_offset:y_offset + crop_height, :]

    if img.shape[1] != CAMERA_WIDTH or img.shape[0] != CAMERA_HEIGHT:
        img = cv2.resize(
            img,
            (CAMERA_WIDTH, CAMERA_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    return np.ascontiguousarray(img)


class ApplePredictionProcessor:
    """streamlit-webrtc video processor: detects, classifies and stabilizes every apple in frame.

    Classification (colour-candidate check + SVM inference) is dispatched to a
    short-lived background thread on each due cycle so that ``recv`` — which
    gates how quickly frames reach the browser preview — never blocks on it.
    Without this, the preview visibly lags/stutters whenever the camera moves,
    since ``recv`` would otherwise stall for the full inference duration.
    """

    def __init__(self) -> None:
        self._busy = False
        self._last_predict_time = 0.0
        self._state_lock = threading.Lock()
        self._tracks: dict[int, _Track] = {}
        self._display_predictions: list[ApplePrediction] = []
        self.result_queue: "queue.Queue[list[ApplePrediction]]" = queue.Queue(maxsize=1)

    # --------------------------------------------------------
    # Multi-apple detection across the whole camera frame (FR-LC-06)
    # --------------------------------------------------------

    def _detect_apple_boxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if frame is None or frame.size == 0:
            return []

        # Detect connected colour regions instead of watershed peaks. In live
        # video, reflections and camera noise can create many watershed seeds
        # inside one apple, which causes duplicate boxes and rapidly changing
        # track IDs.
        height, width = frame.shape[:2]
        scale = min(1.0, DETECTION_MAX_DIMENSION / max(width, height))
        detection_frame = frame
        if scale < 1.0:
            detection_frame = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        try:
            smoothed = cv2.medianBlur(detection_frame, 5)
            candidate_mask = create_apple_candidate_mask(smoothed)
        except (ValueError, cv2.error):
            return []

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        candidate_mask = cv2.morphologyEx(
            candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        contours, _ = cv2.findContours(
            candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        image_area = detection_frame.shape[0] * detection_frame.shape[1]
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (0.007 * image_area <= area <= 0.75 * image_area):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / float(h) if h else 0.0
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area else 0.0
            extent = area / float(w * h) if w and h else 0.0
            perimeter = cv2.arcLength(contour, True)
            circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0

            # Apples remain compact and approximately round even when their
            # colour mask has small gaps. These checks reject hands, leaves,
            # furniture and narrow background regions.
            if (
                0.50 <= aspect <= 1.60
                and solidity >= 0.72
                and extent >= 0.48
                and circularity >= 0.25
            ):
                candidates.append((area, (x, y, w, h)))

        if not candidates:
            return []

        # Ignore small coloured fragments relative to the principal apples.
        candidates.sort(key=lambda item: item[0], reverse=True)
        largest_area = candidates[0][0]
        boxes = [box for area, box in candidates if area >= 0.18 * largest_area]

        if scale < 1.0:
            inv_scale = 1.0 / scale
            boxes = [
                (
                    int(x * inv_scale), int(y * inv_scale),
                    int(w * inv_scale), int(h * inv_scale),
                )
                for x, y, w, h in boxes
            ]

        # A tight watershed box can cut off the apple edge. Add a small margin
        # before classification and clamp it to the source frame.
        padded_boxes = []
        for x, y, w, h in boxes:
            pad = max(4, int(round(0.08 * max(w, h))))
            left = max(0, x - pad)
            top = max(0, y - pad)
            right = min(width, x + w + pad)
            bottom = min(height, y + h + pad)
            padded_boxes.append((left, top, right - left, bottom - top))

        return padded_boxes

    # --------------------------------------------------------
    # Classification of one cropped ROI (FR-LC-06, FR-LC-07, FR-LC-08)
    # --------------------------------------------------------

    def _classify_roi(self, roi: np.ndarray) -> tuple[str, float] | None:
        if roi is None or roi.size == 0:
            return None

        roi = np.ascontiguousarray(roi, dtype=np.uint8)

        # Cheap HSV colour-candidate check first (FR-LC-14): avoids running the
        # full GrabCut + feature-extraction pipeline on frames with no apple.
        try:
            candidate_mask = create_apple_candidate_mask(roi)
        except (ValueError, cv2.error):
            return None

        candidate_ratio = cv2.countNonZero(candidate_mask) / candidate_mask.size
        if candidate_ratio < MIN_CANDIDATE_RATIO:
            return None

        try:
            # Use the same GrabCut preprocessing used during model training.
            # Detection is now limited to a few stable boxes, so this accuracy
            # improvement does not block the preview thread.
            outcome = predict_ripeness(roi)
        except (FileNotFoundError, ValueError, OSError, RuntimeError):
            return None

        return outcome["label"], outcome["confidence"]

    # --------------------------------------------------------
    # Background inference (runs off the video-frame thread)
    # --------------------------------------------------------

    def _run_detection(self, frame: np.ndarray) -> None:
        try:
            try:
                boxes = self._detect_apple_boxes(frame)
            except Exception:
                boxes = []
            # Tracking and classification are owned by this single worker.
            # Keep them outside the state lock so recv() can continue sending
            # preview frames while inference is running.
            predictions = self._update_tracks(boxes, frame)
        except Exception:
            predictions = []
        finally:
            # Never leave detection permanently disabled after an unexpected
            # error in tracking or classification.
            with self._state_lock:
                if "predictions" in locals():
                    self._display_predictions = predictions
                self._busy = False

        if self.result_queue.full():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.result_queue.put_nowait(predictions)
        except queue.Full:
            pass

    # --------------------------------------------------------
    # Tracking: match this cycle's boxes to existing apples so each one keeps
    # its own stabilization window across frames (FR-LC-12).
    # --------------------------------------------------------

    def _match_track(
        self, box: tuple[int, int, int, int], candidate_ids: set[int]
    ) -> int | None:
        bx, by, bw, bh = box
        cx, cy = bx + bw / 2, by + bh / 2

        best_id: int | None = None
        best_dist = float("inf")

        for track_id in candidate_ids:
            tx, ty, tw, th = self._tracks[track_id].bbox
            tcx, tcy = tx + tw / 2, ty + th / 2
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            intersection_width = max(0, min(bx + bw, tx + tw) - max(bx, tx))
            intersection_height = max(0, min(by + bh, ty + th) - max(by, ty))
            intersection = intersection_width * intersection_height
            union = bw * bh + tw * th - intersection
            iou = intersection / union if union else 0.0
            # Larger/nearer apples naturally move farther in pixel space.
            allowed_distance = max(MAX_TRACK_DISTANCE, 0.5 * max(bw, bh, tw, th))
            if (iou >= 0.10 or dist <= allowed_distance) and dist < best_dist:
                best_dist = dist
                best_id = track_id

        return best_id

    def _update_tracks(
        self, boxes: list[tuple[int, int, int, int]], frame: np.ndarray
    ) -> list[ApplePrediction]:
        if len(boxes) > MAX_APPLES_PER_CYCLE:
            boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:MAX_APPLES_PER_CYCLE]

        unmatched_ids = set(self._tracks)
        assignments: dict[int, tuple[int, int, int, int]] = {}

        for box in boxes:
            track_id = self._match_track(box, unmatched_ids)
            if track_id is None:
                # Reuse the lowest available display ID. Temporary detection
                # loss should never turn two physical apples into Apple #20.
                available_ids = [
                    candidate_id
                    for candidate_id in range(1, MAX_APPLES_PER_CYCLE + 1)
                    if candidate_id not in self._tracks
                ]
                if available_ids:
                    track_id = available_ids[0]
                elif unmatched_ids:
                    # All display slots are occupied by stale, unmatched
                    # tracks. Replace the stalest one with this detection.
                    track_id = max(
                        unmatched_ids, key=lambda candidate_id: self._tracks[candidate_id].missed
                    )
                    unmatched_ids.discard(track_id)
                else:
                    continue
                self._tracks[track_id] = _Track(track_id, box)
            else:
                unmatched_ids.discard(track_id)
            assignments[track_id] = box

        # Age out apples that briefly drop out of detection instead of
        # discarding their stabilization history on the very next frame.
        for track_id in unmatched_ids:
            track = self._tracks[track_id]
            track.missed += 1
            if track.missed > MAX_MISSED_CYCLES:
                del self._tracks[track_id]

        predictions = []
        for track_id, box in assignments.items():
            track = self._tracks[track_id]
            # Smooth the overlay and crop coordinates to reduce webcam jitter.
            old = track.bbox
            track.bbox = tuple(
                int(round(0.65 * previous + 0.35 * current))
                for previous, current in zip(old, box)
            )
            track.missed = 0

            # Classify the current detection, not the smoothed overlay box;
            # the latter intentionally trails motion and can clip the apple.
            x, y, w, h = box
            sub_roi = np.ascontiguousarray(frame[y:y + h, x:x + w])
            try:
                outcome = self._classify_roi(sub_roi)
            except Exception:
                outcome = None
            track.recent.append(outcome)

            predictions.append(self._track_prediction(track))

        return predictions

    # --------------------------------------------------------
    # Stabilization (FR-LC-12) + confidence gate (FR-LC-13, FR-LC-14)
    # --------------------------------------------------------

    def _track_prediction(self, track: _Track) -> ApplePrediction:
        valid = [item for item in track.recent if item is not None]

        if len(valid) < 2:
            return ApplePrediction(
                track.id, track.bbox, "low_confidence",
                "Identifying... move closer / ensure sufficient lighting.",
            )

        labels = [label for label, _ in valid]
        majority_label, majority_count = Counter(labels).most_common(1)[0]

        if majority_count < 2:
            return ApplePrediction(
                track.id, track.bbox, "analyzing", "Identifying... hold the apple steady.",
            )

        confidences = [conf for label, conf in valid if label == majority_label]
        avg_confidence = sum(confidences) / len(confidences)

        if avg_confidence < CONFIDENCE_THRESHOLD:
            return ApplePrediction(
                track.id, track.bbox, "low_confidence", "Unable to identify clearly.",
            )

        return ApplePrediction(
            track.id, track.bbox, "confirmed",
            label=majority_label, confidence=avg_confidence,
        )

    # --------------------------------------------------------
    # Overlay drawing (FR-LC-04, FR-LC-09)
    # --------------------------------------------------------

    def _draw_overlay(
        self,
        img: np.ndarray,
        predictions: list[ApplePrediction],
    ) -> None:
        height, width = img.shape[:2]

        # Show the fixed processing/output size independently of how small the
        # browser must render the preview on the page.
        cv2.putText(
            img, f"Live: {width} x {height}", (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )

        if not predictions:
            cv2.putText(
                img, "No apples detected.", (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA,
            )
            return

        for pred in predictions:
            x, y, w, h = pred.bbox
            color = (0, 200, 0) if pred.status == "confirmed" else (0, 165, 255)

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            text = f"{pred.label} - {pred.confidence:.0%}" if pred.status == "confirmed" else pred.message
            cv2.putText(
                img, text, (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Frame callback (FR-LC-05, FR-LC-06, FR-LC-10, FR-LC-11, AF-03)
    # --------------------------------------------------------

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        img = _normalize_live_frame(img)

        now = time.monotonic()

        with self._state_lock:
            due = (now - self._last_predict_time) >= PREDICTION_INTERVAL_SECONDS and not self._busy
            if due:
                self._busy = True
                self._last_predict_time = now
            display_predictions = self._display_predictions

        # Hand the frame off to a background thread and return immediately —
        # inference never blocks the frame being sent to the browser.
        if due:
            threading.Thread(
                target=self._run_detection, args=(img.copy(),), daemon=True
            ).start()

        self._draw_overlay(img, display_predictions)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# Streamlit entry point
# ============================================================

def live_camera_classification() -> None:
    """Render the Live Camera screen (FR-LC-01 .. FR-LC-15)."""

    st.info("Point the camera at one or more apples — every apple in view will be detected and classified.")

    playing_state_key = "apple_live_camera_requested"
    if playing_state_key not in st.session_state:
        st.session_state[playing_state_key] = False

    def start_camera() -> None:
        st.session_state[playing_state_key] = True

    def stop_camera() -> None:
        st.session_state[playing_state_key] = False

    start_column, stop_column = st.columns(2)
    with start_column:
        st.button(
            "Start camera",
            key="apple-live-start",
            on_click=start_camera,
            disabled=st.session_state[playing_state_key],
            use_container_width=True,
            type="primary",
        )
    with stop_column:
        st.button(
            "Stop camera",
            key="apple-live-stop",
            on_click=stop_camera,
            disabled=not st.session_state[playing_state_key],
            use_container_width=True,
        )

    ctx = webrtc_streamer(
        key="apple-live-camera",
        mode=WebRtcMode.SENDRECV,
        desired_playing_state=st.session_state[playing_state_key],
        rtc_configuration=_get_rtc_configuration(),
        video_processor_factory=ApplePredictionProcessor,
        media_stream_constraints={
            "video": {
                # Phones should open the rear camera. Keep this preference
                # non-mandatory so PCs, which do not expose a "back" camera,
                # can fall back to their normal webcam.
                "facingMode": {"ideal": "environment"},
                "width": {"ideal": CAMERA_WIDTH},
                "height": {"ideal": CAMERA_HEIGHT},
                "frameRate": {"ideal": CAMERA_FPS},
            },
            "audio": False,
        },
        # streamlit-webrtc defaults to width: 100%, which can enlarge a
        # low-resolution stream and make it look blurrier. Keep the camera's
        # intrinsic display size and only shrink it when the page is narrower.
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "playsInline": True,
            "style": {
                "width": "auto",
                "maxWidth": "100%",
                "height": "auto",
            },
        },
        # recv() only resizes/draws; detection already runs in its own worker.
        # Processing frames in order prevents an async backlog from making the
        # phone preview drift several seconds behind the real camera.
        async_processing=False,
    )

    if not ctx.state.playing:
        if st.session_state[playing_state_key]:
            st.caption("Connecting to the camera…")
        else:
            st.caption(
                "Click **Start camera** above and allow camera access to begin. "
                "Use the fixed **Stop camera** button to end the stream."
            )
        return

    result_placeholder = st.empty()

    # Loop while the stream is active; stops (and releases the camera via
    # streamlit-webrtc) once the user stops the stream or leaves the screen.
    while ctx.state.playing:
        if ctx.video_processor is None:
            break

        try:
            results: list[ApplePrediction] = ctx.video_processor.result_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        with result_placeholder.container():
            if not results:
                st.info("No apples detected. Point the camera at one or more apples.")
            for pred in results:
                if pred.status == "confirmed":
                    st.success(f"Apple {pred.track_id}: **{pred.label}** — {pred.confidence:.0%}")
                else:
                    st.warning(f"Apple {pred.track_id}: {pred.message}")
