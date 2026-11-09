"""Face detection HTTP routes."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse, Response

from app.schemas.detection_schema import (
    ErrorResponse,
    FaceDetectionResponse,
    ModelInfoResponse,
)
from app.services.face_detector import get_detector
from app.utils import image_utils
from app.utils.media_urls import media_url_for_filename

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/faces", tags=["faces"])

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


def _multipart_mixed_response(json_payload: dict, jpeg_bytes: bytes) -> Response:
    """
    multipart/mixed: JSON metadata + annotated JPEG.

    Swagger documents the JSON shape; clients that need the binary can opt into
    response_format=multipart. Default API responses are application/json.
    """
    boundary = f"face-detection-{uuid.uuid4().hex}"
    json_bytes = json.dumps(json_payload).encode("utf-8")
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="metadata"\r\n',
            b"Content-Type: application/json; charset=utf-8\r\n\r\n",
            json_bytes,
            b"\r\n",
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="processed_image"\r\n',
            b"Content-Type: image/jpeg\r\n\r\n",
            jpeg_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return Response(
        content=body,
        media_type=f"multipart/mixed; boundary={boundary}",
        status_code=status.HTTP_200_OK,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    detector = get_detector()
    info = detector.model_info()
    if not info["model_loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face detection model is not loaded",
        )
    return ModelInfoResponse(**info)


@router.post(
    "/detect",
    responses={
        200: {
            "description": (
                "Default: application/json with faces and annotated_image_url. "
                "Pass response_format=multipart for legacy multipart/mixed."
            ),
            "content": {
                "application/json": {
                    "schema": FaceDetectionResponse.model_json_schema(),
                },
                "multipart/mixed": {},
            },
        },
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Detect faces in an uploaded image",
)
async def detect_faces(
    image: UploadFile = File(..., description="JPEG or PNG image"),
    response_format: Literal["json", "multipart"] = Query(
        "json",
        description="json (default) or multipart (legacy)",
    ),
):
    detector = get_detector()
    if not detector.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face detection model is not available",
        )

    try:
        image_utils.validate_upload_filename(image.filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    raw = await image.read()
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

    try:
        bgr = image_utils.decode_image_bytes(raw)
    except ValueError as exc:
        message = str(exc)
        code = (
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            if "format" in message.lower()
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=message) from exc

    try:
        faces = detector.detect_faces(bgr)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during face detection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal error during face detection",
        ) from exc

    annotated = image_utils.draw_face_annotations(bgr, faces)
    jpeg_bytes = image_utils.encode_jpeg(annotated)
    saved_path = image_utils.save_processed_jpeg(jpeg_bytes, image.filename)
    annotated_image_url = media_url_for_filename(saved_path.name)

    payload = FaceDetectionResponse(
        success=True,
        total_faces=len(faces),
        faces=faces,
        message="Face detection completed",
        annotated_image_url=annotated_image_url,
        output_image_path=str(saved_path),
    )
    if response_format == "multipart":
        return _multipart_mixed_response(payload.model_dump(), jpeg_bytes)
    return JSONResponse(content=payload.model_dump(), status_code=status.HTTP_200_OK)
