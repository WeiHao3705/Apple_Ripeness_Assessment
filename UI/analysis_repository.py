"""Save and retrieve authenticated apple assessments with Supabase.

This file is the application's data-access layer.  It keeps Supabase code out
of the Streamlit interface and handles two different Supabase products:

* Storage: saves the original JPEG/PNG in the ``apple-images`` bucket.
* Database: saves one parent row in ``analyses`` and one child row for every
  detected apple in ``analysis_results``.

All requests use the signed-in user's access token from ``login.py``.  The
actual privacy boundary must still be enforced by Supabase RLS and Storage
policies; a user ID in a file path is not a security rule by itself.
"""

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
    """Upload one source image and save its parent and per-apple records.

    Returns the new analysis UUID, which is shared by the Storage path and the
    related database rows.
    """
    # History belongs to a Supabase Auth user. Guest analyses remain temporary.
    user_id = current_user_id()
    if user_id is None:
        raise AnalysisRepositoryError("Sign in before saving an analysis.")

    # Generate the identifier in the app so Storage and both tables can refer
    # to the same assessment before any network request is made.
    extension, content_type = _image_type(image_bytes)
    analysis_id = str(uuid4())
    image_path = f"{user_id}/{analysis_id}/original.{extension}"
    client = authenticated_supabase_client()
    bucket = client.storage.from_(APPLE_IMAGES_BUCKET)

    image_uploaded = False
    analysis_inserted = False

    try:
        # Step 1: store the original input image. Images are grouped by user
        # and analysis ID to avoid filename collisions.
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

        # Step 2: insert the parent assessment (one row per input image).
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

        # Step 3: insert the child results (one row per detected apple).
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
    """Return recent parent records with their nested per-apple results.

    The database RLS policy should restrict this query to the authenticated
    user's rows. The UI additionally requires a login before making the query.
    """
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
    """Download one history image using the authenticated Supabase client.

    Whether the object is private is determined by the bucket configuration
    and Storage policies in Supabase.
    """
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
