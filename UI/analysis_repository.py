"""Persist and retrieve authenticated apple analyses with Supabase."""

from __future__ import annotations

from io import BytesIO
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from image_pipeline import ImageAnalysis
from login import authenticated_supabase_client, current_user_id


APPLE_IMAGES_BUCKET = "apple-images"
HISTORY_PAGE_SIZE = 50


class AnalysisRepositoryError(RuntimeError):
    """Raised when an analysis cannot be stored or retrieved."""


def _image_type(image_bytes: bytes) -> tuple[str, str]:
    """Return a safe file extension and MIME type from the actual image bytes."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise AnalysisRepositoryError(
            "The image could not be prepared for secure storage."
        ) from exc

    if image_format in {"JPG", "JPEG"}:
        return "jpg", "image/jpeg"
    if image_format == "PNG":
        return "png", "image/png"

    raise AnalysisRepositoryError(
        "Only JPEG and PNG images can be saved to analysis history."
    )


def save_analysis(
    *,
    image_bytes: bytes,
    input_method: str,
    analysis: ImageAnalysis,
) -> str:
    """Upload an image and insert its analysis and per-apple results."""
    user_id = current_user_id()
    if user_id is None:
        raise AnalysisRepositoryError("Sign in before saving an analysis.")

    extension, content_type = _image_type(image_bytes)
    analysis_id = str(uuid4())
    image_path = f"{user_id}/{analysis_id}/original.{extension}"
    client = authenticated_supabase_client()
    bucket = client.storage.from_(APPLE_IMAGES_BUCKET)

    image_uploaded = False
    analysis_inserted = False

    try:
        bucket.upload(
            image_path,
            image_bytes,
            {
                "content-type": content_type,
                "cache-control": "3600",
                "upsert": "false",
            },
        )
        image_uploaded = True

        client.table("analyses").insert(
            {
                "id": analysis_id,
                "user_id": user_id,
                "image_path": image_path,
                "input_method": input_method,
                "apple_count": len(analysis.apples),
                "detection_message": analysis.detection_message,
            }
        ).execute()
        analysis_inserted = True

        result_rows = [
            {
                "analysis_id": analysis_id,
                "apple_number": apple.apple_id,
                "ripeness_label": apple.label,
                "confidence": apple.confidence,
                "probabilities": apple.probabilities,
                "classification_error": apple.classification_error,
            }
            for apple in analysis.apples
        ]
        if result_rows:
            client.table("analysis_results").insert(result_rows).execute()

        return analysis_id
    except Exception as exc:
        # Supabase REST and Storage calls are separate transactions. Compensate
        # for a partial failure so users do not accumulate orphaned rows/files.
        if analysis_inserted:
            try:
                client.table("analyses").delete().eq(
                    "id", analysis_id
                ).execute()
            except Exception:
                pass
        if image_uploaded:
            try:
                bucket.remove([image_path])
            except Exception:
                pass
        raise AnalysisRepositoryError(
            "The analysis was completed but could not be saved to your history. "
            f"Technical detail: {exc}"
        ) from exc


def list_analyses(limit: int = HISTORY_PAGE_SIZE) -> list[dict[str, Any]]:
    """Return the signed-in user's newest analysis records."""
    if current_user_id() is None:
        raise AnalysisRepositoryError("Sign in to view analysis history.")

    try:
        response = (
            authenticated_supabase_client()
            .table("analyses")
            .select(
                "id,image_path,input_method,apple_count,detection_message,"
                "created_at,analysis_results("
                "id,apple_number,ripeness_label,confidence,probabilities,"
                "classification_error)"
            )
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        raise AnalysisRepositoryError(
            "Your analysis history could not be loaded. "
            f"Technical detail: {exc}"
        ) from exc

    return list(response.data or [])


def download_analysis_image(image_path: str) -> bytes:
    """Download one private history image through the user's RLS session."""
    try:
        return (
            authenticated_supabase_client()
            .storage.from_(APPLE_IMAGES_BUCKET)
            .download(image_path)
        )
    except Exception as exc:
        raise AnalysisRepositoryError(
            "The saved image could not be downloaded."
        ) from exc
