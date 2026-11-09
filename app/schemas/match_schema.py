"""Pydantic models for reference-to-crowd face matching."""

from pydantic import BaseModel, Field

from app.schemas.detection_schema import BoundingBox


class MatchedFace(BaseModel):
    face_id: int
    bounding_box: BoundingBox
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="CompreFace similarity to the reference subject",
    )
    is_match: bool = True


class FaceMatchResponse(BaseModel):
    success: bool = True
    operation_id: str | None = Field(
        None,
        description="History record id for this match operation",
    )
    person_name: str | None = Field(
        None,
        description="Optional name supplied by the frontend",
    )
    match_count: int
    total_faces_in_target: int
    similarity_threshold: float
    max_matches: int = Field(
        ...,
        description="Maximum matches returned (best similarity first); 0 = no limit",
    )
    candidates_above_threshold: int = Field(
        ...,
        description="Faces at or above threshold before applying max_matches",
    )
    matches: list[MatchedFace] = Field(
        ...,
        description="Top match(es) after max_matches limit (drawn on output image)",
    )
    all_candidates: list[MatchedFace] = Field(
        default_factory=list,
        description="Every face above threshold, best similarity first (for review in Swagger)",
    )
    message: str = "Face matching completed"
    annotated_image_url: str | None = Field(
        None,
        description="URL path for the annotated JPEG (prefix with API base URL)",
    )
    output_image_path: str | None = Field(
        None,
        description="Server filesystem path (local tooling only; prefer annotated_image_url)",
    )
