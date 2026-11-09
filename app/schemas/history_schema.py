"""Schemas for match operation history."""

from typing import Any

from pydantic import BaseModel, Field


class FaceHistoryResponse(BaseModel):
    success: bool = True
    total: int
    records: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Newest-first match operations (empty until matches are run)",
    )
