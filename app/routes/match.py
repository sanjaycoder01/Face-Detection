"""Reference-to-crowd face matching routes."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.routes.detection import MAX_UPLOAD_BYTES, _multipart_mixed_response
from app.schemas.match_schema import FaceMatchResponse
from app.services.face_matcher import get_matcher
from app.services.history_store import append_match_history
from app.utils import image_utils
from app.utils.media_urls import media_url_for_filename

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/faces", tags=["faces"])

# Canonical + Lovable-friendly aliases
REFERENCE_FIELD_NAMES = (
    "reference",
    "reference_image",
    "referenceImage",
    "selfie",
    "person",
    "person_image",
    "personImage",
    "face",
    "query",
)
TARGET_FIELD_NAMES = (
    "target",
    "group",
    "group_image",
    "groupImage",
    "group_photo",
    "groupPhoto",
    "event",
    "event_photo",
    "eventPhoto",
    "photo",
)
NAME_FIELD_NAMES = (
    "name",
    "person_name",
    "personName",
    "target_name",
    "targetName",
)


async def _read_upload(upload: UploadFile | StarletteUploadFile) -> tuple[bytes, str]:
    filename = getattr(upload, "filename", None)
    try:
        image_utils.validate_upload_filename(filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    raw = await upload.read()
    if len(raw) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image upload",
        )
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image exceeds maximum size of {MAX_UPLOAD_BYTES} bytes",
        )
    return raw, filename or "image.jpg"


def _first_upload(form, field_names: tuple[str, ...]) -> UploadFile | StarletteUploadFile | None:
    for key in field_names:
        value = form.get(key)
        if isinstance(value, (UploadFile, StarletteUploadFile)):
            return value
    return None


def _first_text(form, field_names: tuple[str, ...]) -> str | None:
    for key in field_names:
        value = form.get(key)
        if value is None or isinstance(value, (UploadFile, StarletteUploadFile)):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


@router.post(
    "/match",
    responses={
        200: {
            "description": (
                "Default: application/json with matching results and annotated_image_url. "
                "Pass response_format=multipart for legacy multipart/mixed (JSON + JPEG bytes)."
            ),
            "content": {
                "application/json": {"schema": FaceMatchResponse.model_json_schema()},
                "multipart/mixed": {},
            },
        },
        400: {"description": "Invalid image"},
        413: {"description": "Payload too large"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Missing required form fields"},
        503: {"description": "Matcher unavailable"},
    },
    summary="Match a reference face photo against faces in a target image",
)
async def match_faces(
    request: Request,
    max_matches: int | None = Query(
        None,
        ge=0,
        le=20,
        description=(
            "Return at most this many matches (highest similarity first). "
            "0 = no limit. Defaults to COMPREFACE_MATCH_MAX_RESULTS (usually 1)."
        ),
    ),
    response_format: Literal["json", "multipart"] = Query(
        "json",
        description="json (default, frontend-friendly) or multipart (legacy curl/Swagger binary)",
    ),
):
    """
    Multipart form fields:

    - reference image: `reference` (preferred) or aliases `selfie`, `person`, `personImage`, ...
    - group image: `target` (preferred) or aliases `group`, `groupPhoto`, `eventPhoto`, ...
    - person name: `name` (preferred) or aliases `person_name`, `personName`, `target_name`, ...
    """
    matcher = get_matcher()
    if not matcher.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Face matching requires COMPREFACE_SERVICE_TYPE=recognition "
                "and a valid COMPREFACE_API_KEY"
            ),
        )

    form = await request.form()
    reference = _first_upload(form, REFERENCE_FIELD_NAMES)
    target = _first_upload(form, TARGET_FIELD_NAMES)
    person_name_raw = _first_text(form, NAME_FIELD_NAMES)

    missing: list[str] = []
    if reference is None:
        missing.append("reference (or selfie/person/personImage)")
    if target is None:
        missing.append("target (or group/groupPhoto/eventPhoto)")
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Field required: {'; '.join(missing)}",
        )

    ref_raw, ref_name = await _read_upload(reference)
    tgt_raw, tgt_name = await _read_upload(target)

    try:
        ref_bgr = image_utils.decode_image_bytes(ref_raw)
        tgt_bgr = image_utils.decode_image_bytes(tgt_raw)
    except ValueError as exc:
        message = str(exc)
        code = (
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            if "format" in message.lower()
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=message) from exc

    try:
        result = matcher.match_faces(
            ref_bgr,
            tgt_bgr,
            reference_filename=ref_name,
            target_filename=tgt_name,
            max_matches=max_matches,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during face matching")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal error during face matching",
        ) from exc

    annotated = (
        image_utils.draw_match_annotations(tgt_bgr, result.matches)
        if result.matches
        else tgt_bgr
    )
    jpeg_bytes = image_utils.encode_jpeg(annotated)
    saved_path = image_utils.save_processed_jpeg(
        jpeg_bytes,
        tgt_name,
        suffix="_matched",
    )
    annotated_image_url = media_url_for_filename(saved_path.name)
    person_name = person_name_raw

    if result.matches:
        best = result.matches[0]
        pct = int(round(best.similarity * 100))
        who = f" for {person_name}" if person_name else ""
        message = f"Best match{who}: {pct}% similar (face #{best.face_id})"
        if result.candidates_above_threshold > len(result.matches):
            message += (
                f"; {result.candidates_above_threshold} face(s) above threshold, "
                f"showing top {len(result.matches)}"
            )
    else:
        who = f" for {person_name}" if person_name else ""
        message = f"No matching faces found in target image{who}"

    history = append_match_history(
        person_name=person_name,
        match_count=len(result.matches),
        total_faces_in_target=result.total_faces_in_target,
        similarity_threshold=result.similarity_threshold,
        matches=[m.model_dump() for m in result.matches],
        annotated_image_url=annotated_image_url,
        annotated_filename=saved_path.name,
        message=message,
    )

    payload = FaceMatchResponse(
        success=True,
        operation_id=history["id"],
        person_name=person_name,
        match_count=len(result.matches),
        total_faces_in_target=result.total_faces_in_target,
        similarity_threshold=result.similarity_threshold,
        max_matches=result.max_matches,
        candidates_above_threshold=result.candidates_above_threshold,
        matches=result.matches,
        all_candidates=result.all_candidates,
        message=message,
        annotated_image_url=annotated_image_url,
        output_image_path=str(saved_path),
    )

    if response_format == "multipart":
        return _multipart_mixed_response(payload.model_dump(), jpeg_bytes)
    return JSONResponse(content=payload.model_dump(), status_code=status.HTTP_200_OK)
