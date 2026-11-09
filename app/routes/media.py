"""Serve annotated images from the outputs directory (no path traversal)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.utils.media_urls import resolve_output_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/media", tags=["media"])


@router.get(
    "/outputs/{filename}",
    summary="Download an annotated output image by filename",
)
def get_output_image(filename: str):
    try:
        path = resolve_output_file(filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found",
        ) from exc

    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=path.name,
    )
