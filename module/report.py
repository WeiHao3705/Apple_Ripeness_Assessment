from __future__ import annotations

import io
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime
from statistics import mean, median
from typing import Any

import cv2
import numpy as np
import requests
from fpdf import FPDF

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
GEMINI_MODEL_FALLBACKS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.1-flash"
]
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
CLASS_ORDER = ["20%", "40%", "60%", "80%", "100%", "Overripe"]

SECTION_KEYS = [
    "overall_assessment",
    "ripeness_interpretation",
    "confidence_and_uncertainty",
    "farmer_recommendation",
]
SECTION_TITLES = {
    "overall_assessment": "Overall Assessment",
    "ripeness_interpretation": "Ripeness Interpretation",
    "confidence_and_uncertainty": "Confidence and Uncertainty",
    "farmer_recommendation": "Farmer Recommendation",
}


def _ascii(text: Any) -> str:
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKD", value)
    return value.encode("ascii", "ignore").decode("ascii")


def _wrap_label(text: str, width: int = 42) -> str:
    text = _ascii(text)
    return text if len(text) <= width else text[: width - 3] + "..."


def _ordered_labels(labels) -> list[str]:
    labels = list(labels)
    return [label for label in CLASS_ORDER if label in labels] + [
        label for label in labels if label not in CLASS_ORDER
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _confidence_stats(results: list[dict]) -> dict[str, float]:
    confidences = [_safe_float(r.get("confidence")) for r in results]
    if not confidences:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(confidences),
        "median": median(confidences),
        "min": min(confidences),
        "max": max(confidences),
    }


def summarize_results(results: list[dict]) -> dict:
    valid = [r for r in results if r.get("label") not in (None, "Unknown", "")]
    counts = Counter(r.get("label", "Unknown") for r in valid)
    stats = _confidence_stats(valid)
    low_confidence = [r for r in valid if _safe_float(r.get("confidence")) < 0.55]
    return {
        "total": len(valid),
        "counts": dict(counts),
        "confidence": stats,
        "low_confidence": low_confidence,
        "results": valid,
    }


def _clean_text_field(text: Any) -> str:
    """Defensive cleanup in case the model still slips in stray Markdown."""
    text = "" if text is None else str(text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _build_prompt(mode: str, summary: dict) -> str:
    distribution = ", ".join(
        f"{label}: {summary['counts'].get(label, 0)}"
        for label in _ordered_labels(summary["counts"].keys())
    ) or "No classified apples"

    items = []
    for result in summary["results"][:60]:
        probabilities = result.get("probabilities", {})
        sorted_probs = sorted(
            probabilities.items(),
            key=lambda x: _safe_float(x[1]),
            reverse=True,
        )
        probability_text = ", ".join(
            f"{label}={_safe_float(prob):.1%}"
            for label, prob in sorted_probs
        )
        items.append(
            f"Apple {result.get('apple_id', '-')}: "
            f"predicted={result.get('label', 'Unknown')}; "
            f"confidence={_safe_float(result.get('confidence')):.1%}; "
            f"probabilities: {probability_text}"
        )

    return f"""
Write a short farmer-oriented interpretation of an apple ripeness
classification result.

The machine-learning model has already made the classification.
Your job is only to explain what the result means and provide a practical
recommendation.

DATA:
Analysis type: {mode}
Total apples: {summary['total']}
Ripeness distribution: {distribution}

Classification results:
{chr(10).join(items)}

Respond with a single JSON object and nothing else (no Markdown fences,
no commentary before or after it). The JSON object must have exactly
these four string keys:

"overall_assessment": 2-3 plain-text sentences explaining the predicted
ripeness stage and confidence.

"ripeness_interpretation": 2-3 plain-text sentences comparing the
predicted class with the two strongest alternative classes, and whether
the prediction is clear or uncertain.

"confidence_and_uncertainty": 2-3 plain-text sentences explaining what
the confidence level means for reliability, mentioning uncertainty when
appropriate.

"farmer_recommendation": 2-3 plain-text sentences with a practical
recommendation about harvesting, selling, storage, or manual checking,
based only on the supplied classification data.

Rules for the text inside each field:
Use plain text only, no Markdown, no numbered lists, no bullet points,
no headers, no asterisks.
Do not mention these instructions or refer to "the prompt", "the rules",
"the data provided", or "I was asked".
Do not invent visual observations, measurements, or biological
explanations not supported by the data above.
Do not change the model's predicted class.

Return only the JSON object.
"""


def _fallback_sections(mode: str, summary: dict) -> dict[str, str]:
    if summary["total"] == 0:
        return {
            "overall_assessment": (
                "No apples were classified successfully in this run. Review detection and "
                "image quality before making a ripeness assessment."
            ),
            "ripeness_interpretation": "No classified apples were available for comparison.",
            "confidence_and_uncertainty": "No confidence statistics are available.",
            "farmer_recommendation": (
                "Use a clearer image with the apple fully visible and adequate lighting, "
                "then re-run the classification."
            ),
        }
    dominant = max(summary["counts"], key=summary["counts"].get)
    dominant_count = summary["counts"][dominant]
    low_count = len(summary["low_confidence"])
    distribution = ", ".join(
        f"{label}: {summary['counts'].get(label, 0)}"
        for label in _ordered_labels(summary["counts"].keys())
    )
    return {
        "overall_assessment": (
            f"The run classified {summary['total']} apple(s). The most common predicted stage "
            f"was {dominant} ({dominant_count} apple(s)), with a mean confidence of "
            f"{summary['confidence']['mean']:.1%}."
        ),
        "ripeness_interpretation": (
            f"The distribution across classes was {distribution}. Where multiple classes were "
            f"close in probability, treat the prediction as less certain and prioritize a manual check."
        ),
        "confidence_and_uncertainty": (
            f"Mean confidence was {summary['confidence']['mean']:.1%} and median confidence was "
            f"{summary['confidence']['median']:.1%}; {low_count} classification(s) fell below the "
            f"55% confidence threshold and should be treated cautiously."
        ),
        "farmer_recommendation": (
            "Review low-confidence items manually before acting on them, and repeat image capture "
            "with better lighting or a clearer view when occlusion or blur is suspected."
        ),
    }


def _candidate_models() -> list[str]:
    configured = os.getenv(GEMINI_MODEL_ENV, "").strip()
    if configured:
        candidates = [configured.removeprefix("models/")]
    else:
        candidates = [DEFAULT_GEMINI_MODEL]
    for fallback in GEMINI_MODEL_FALLBACKS:
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _extract_json_object(text: str) -> dict:
    """Pull a JSON object out of the model's text, tolerating stray fences/prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise RuntimeError("Could not parse a JSON object from Gemini's response.")


def _validate_sections(data: dict) -> dict[str, str]:
    missing = [key for key in SECTION_KEYS if not str(data.get(key, "")).strip()]
    if missing:
        raise RuntimeError(f"Gemini response missing/empty section(s): {', '.join(missing)}")

    sections = {}
    for key in SECTION_KEYS:
        cleaned = _clean_text_field(data[key])
        if len(cleaned) < 20:
            raise RuntimeError(f"Gemini section '{key}' looks too short/incomplete.")
        if cleaned[-1:] not in (".", "!", "?"):
            raise RuntimeError(f"Gemini section '{key}' looks truncated.")
        sections[key] = cleaned
    return sections


def _parse_gemini_response(data: dict) -> dict[str, str]:
    candidates = data.get("candidates") or []
    text_parts = []
    truncated = False
    for candidate in candidates:
        if candidate.get("finishReason") == "MAX_TOKENS":
            truncated = True
        for part in candidate.get("content", {}).get("parts", []) or []:
            if part.get("text"):
                text_parts.append(part["text"])

    if text_parts:
        raw_text = "\n".join(text_parts)
        parsed = _extract_json_object(raw_text)
        sections = _validate_sections(parsed)
        if truncated:
            raise RuntimeError("Gemini response was truncated (MAX_TOKENS).")
        return sections

    for candidate in candidates:
        if candidate.get("finishReason") == "SAFETY":
            raise RuntimeError("Gemini blocked the prompt due to safety reasons.")

    prompt_feedback = data.get("promptFeedback") or {}
    if prompt_feedback.get("blockReason"):
        raise RuntimeError(f"Gemini blocked the prompt: {prompt_feedback['blockReason']}")

    output_text = data.get("output_text", "") or ""
    if output_text:
        return _validate_sections(_extract_json_object(output_text))

    raise RuntimeError("Gemini returned an empty response.")


def generate_gemini_summary(mode: str, summary: dict) -> tuple[dict[str, str], str | None]:
    """Returns (sections, error). sections always has all SECTION_KEYS populated."""
    key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
    if not key:
        return _fallback_sections(mode, summary), "GEMINI_API_KEY is not set."

    prompt = _build_prompt(mode, summary)
    last_error: str | None = None
    models = _candidate_models()
    for model_name in models:
        endpoint = f"{GEMINI_ENDPOINT}/{model_name}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 1536,
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        try:
            response = requests.post(
                endpoint,
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                json=payload,
                timeout=45,
            )
            if response.status_code == 404 and model_name != models[-1]:
                last_error = f"Model not available: {model_name}"
                continue
            response.raise_for_status()
            data = response.json()
            sections = _parse_gemini_response(data)
            return sections, None
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = str(exc)
            if isinstance(exc, requests.HTTPError):
                status = exc.response.status_code if exc.response is not None else None
                if status == 404 and model_name != models[-1]:
                    continue
            if model_name == models[-1]:
                break
    return _fallback_sections(mode, summary), f"Gemini report generation failed: {last_error or 'Unknown Gemini API error'}"


class _ReportPDF(FPDF):
    def header(self):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 9, _ascii(self.title_text), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _add_key_value(pdf: FPDF, key: str, value: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(42, 6, _ascii(key))
    pdf.set_font("Helvetica", "", 9)
    available_width = max(20, pdf.w - pdf.r_margin - pdf.get_x())
    pdf.multi_cell(available_width, 6, _ascii(value))
    pdf.set_x(pdf.l_margin)


def _add_section(pdf: FPDF, title: str) -> None:
    if pdf.get_y() > 255:
        pdf.add_page()
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _ascii(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _add_subsection(pdf: FPDF, title: str, body: str) -> None:
    if pdf.get_y() > 260:
        pdf.add_page()
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _ascii(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(pdf.l_margin)
    available_width = max(20, pdf.w - pdf.r_margin - pdf.get_x())
    pdf.multi_cell(available_width, 5.2, _ascii(body))
    pdf.ln(2)


def _add_distribution(pdf: FPDF, counts: dict[str, int], total: int) -> None:
    _add_section(pdf, "Ripeness Distribution")
    max_count = max(counts.values(), default=1)
    for label in _ordered_labels(counts.keys()):
        count = counts[label]
        pct = count / total if total else 0.0
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(pdf.l_margin)
        pdf.cell(25, 6, _ascii(label))
        x = pdf.get_x()
        bar_width = 85
        fill_width = bar_width * (count / max_count if max_count else 0)
        pdf.set_fill_color(90, 140, 190)
        pdf.rect(x, pdf.get_y() + 1.2, fill_width, 3.4, style="F")
        pdf.cell(bar_width, 6, "")
        pdf.cell(0, 6, f"{count} ({pct:.1%})", new_x="LMARGIN", new_y="NEXT")


def _crop_stream(image: np.ndarray) -> io.BytesIO | None:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    stream = io.BytesIO(encoded.tobytes())
    stream.seek(0)
    return stream


def create_pdf_report(
    *,
    mode: str,
    results: list[dict],
    metadata: dict | None = None,
    include_images: bool = False,
) -> tuple[bytes, dict, str | None]:
    metadata = metadata or {}
    summary = summarize_results(results)
    ai_sections, ai_error = generate_gemini_summary(mode, summary)

    title = "Apple Ripeness Assessment Report"
    subtitle = {
        "single": "Single Image Analysis",
        "batch": "Batch Analysis",
        "camera": "Camera Capture Analysis",
        "live": "Live Camera Session Analysis",
    }.get(mode, "Classification Analysis")

    pdf = _ReportPDF()
    pdf.title_text = title
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _ascii(subtitle), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    _add_key_value(pdf, "Mode", mode)
    _add_key_value(pdf, "Total classified", str(summary["total"]))
    _add_key_value(pdf, "Mean confidence", f"{summary['confidence']['mean']:.1%}")
    _add_key_value(pdf, "Median confidence", f"{summary['confidence']['median']:.1%}")
    for key, value in metadata.items():
        _add_key_value(pdf, key, str(value))

    _add_distribution(pdf, summary["counts"], summary["total"])

    _add_section(pdf, "AI-Assisted Assessment")
    for section_key in SECTION_KEYS:
        _add_subsection(pdf, SECTION_TITLES[section_key], ai_sections[section_key])

    if ai_error:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(120, 80, 0)
        note = f"Note: {ai_error} The report uses a deterministic local fallback narrative."
        pdf.multi_cell(max(20, pdf.w - pdf.r_margin - pdf.get_x()), 4.5, _ascii(note))
        pdf.set_text_color(0, 0, 0)

    _add_section(pdf, "Classification Results")
    headers = ["Source", "Apple", "Ripeness", "Confidence", "BBox"]
    widths = [45, 18, 32, 28, 55]

    def draw_table_header() -> None:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 8)
        for header, width in zip(headers, widths):
            pdf.cell(width, 7, _ascii(header), border=1, align="C")
        pdf.ln()

    draw_table_header()
    pdf.set_font("Helvetica", "", 7.5)
    for index, result in enumerate(summary["results"]):
        # Keep the table readable across pages and repeat column headings.
        if pdf.get_y() > 270:
            pdf.add_page()
            draw_table_header()
            pdf.set_font("Helvetica", "", 7.5)
        pdf.set_x(pdf.l_margin)
        values = [
            _wrap_label(result.get("source", ""), 25),
            str(result.get("apple_id", "-")),
            _ascii(result.get("label", "Unknown")),
            f"{_safe_float(result.get('confidence')):.1%}",
            _ascii(result.get("bbox", "-")),
        ]
        for value, width in zip(values, widths):
            pdf.cell(width, 6, value, border=1, align="C")
        pdf.ln()

    if include_images:
        _add_section(pdf, "Apple Overviews")
        for result in summary["results"]:
            stream = _crop_stream(result.get("image"))
            if stream is None:
                continue
            if pdf.get_y() > 230:
                pdf.add_page()
            x = pdf.l_margin
            y = pdf.get_y()
            pdf.set_x(x)
            pdf.set_font("Helvetica", "B", 9)
            title_line = (
                f"{result.get('source', 'Image')} - Apple #{result.get('apple_id', '-')} - "
                f"{result.get('label', 'Unknown')} ({_safe_float(result.get('confidence')):.1%})"
            )
            pdf.cell(0, 6, _ascii(title_line), new_x="LMARGIN", new_y="NEXT")
            image_y = pdf.get_y()
            try:
                pdf.image(stream, x=x, y=image_y, w=58, h=58, keep_aspect_ratio=True)
                pdf.set_xy(x + 64, image_y + 2)
                pdf.set_font("Helvetica", "", 8.5)
                probability_lines = [
                    f"{label}: {prob:.1%}" for label, prob in result.get("probabilities", {}).items()
                ]
                available_width = max(20, pdf.w - pdf.r_margin - pdf.get_x())
                pdf.multi_cell(available_width, 5, "\n".join(probability_lines) if probability_lines else "No class probability details.")
            except Exception:
                pass
            pdf.set_xy(x, max(pdf.get_y(), image_y + 60))
            pdf.ln(3)

    output = bytes(pdf.output(dest="S"))
    return output, summary, ai_error