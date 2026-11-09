"""Pydantic models for face detection API requests and responses."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int = Field(..., description="Left edge of the face box (pixels)")
    y1: int = Field(..., description="Top edge of the face box (pixels)")
    x2: int = Field(..., description="Right edge of the face box (pixels)")
    y2: int = Field(..., description="Bottom edge of the face box (pixels)")


class DetectedFace(BaseModel):
    face_id: int
    bounding_box: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)


class FaceDetectionResponse(BaseModel):
    success: bool = True
    total_faces: int
    faces: list[DetectedFace]
    message: str = "Face detection completed"
    annotated_image_url: str | None = Field(
        None,
        description="URL path for the annotated JPEG (prefix with API base URL)",
    )
    output_image_path: str | None = Field(
        None,
        description="Server filesystem path (local tooling only; prefer annotated_image_url)",
    )


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_pack: str
    detection_model: str
    det_size: int
    execution_provider: str
    allowed_modules: list[str]
    model_loaded: bool
    licensing_note: str
