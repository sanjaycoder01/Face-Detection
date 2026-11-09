"""History of face-matching operations for the Lovable frontend."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas.history_schema import FaceHistoryResponse
from app.services.history_store import list_history

router = APIRouter(prefix="/api/v1/faces", tags=["faces"])


@router.get(
    "/history",
    response_model=FaceHistoryResponse,
    summary="List recent face-matching operations",
)
def get_history(limit: int = Query(50, ge=1, le=200)):
    records = list_history(limit=limit)
    return FaceHistoryResponse(success=True, total=len(records), records=records)
