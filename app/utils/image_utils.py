"""Image validation, decoding, annotation, and encoding helpers."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.schemas.detection_schema import DetectedFace

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_PREFIXES = ("image/jpeg", "image/png")


def validate_upload_filename(filename: str | None) -> None:
    if not filename or not filename.strip():
        raise ValueError("Empty filename")
    lower = filename.lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError(
            f"Unsupported image format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def sniff_image_kind(data: bytes) -> str | None:
    """Return 'jpeg' or 'png' from magic bytes, or None if unrecognized."""
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return None


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("Empty image upload")

    kind = sniff_image_kind(data)
    if kind not in ("jpeg", "png"):
        raise ValueError("Unsupported or unrecognized image format")

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Corrupted or unreadable image")
    return image


def draw_face_annotations(
    image: np.ndarray, faces: list[DetectedFace]
) -> np.ndarray:
    """Draw green boxes with face id and confidence; preserves image dimensions."""
    output = image.copy()
    for face in faces:
        box = face.bounding_box
        cv2.rectangle(
            output,
            (box.x1, box.y1),
            (box.x2, box.y2),
            (0, 255, 0),
            2,
        )
        label = f"#{face.face_id} {face.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        y_top = max(box.y1 - text_h - baseline - 4, 0)
        cv2.rectangle(
            output,
            (box.x1, y_top),
            (box.x1 + text_w + 4, y_top + text_h + baseline + 4),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            output,
            label,
            (box.x1 + 2, y_top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    success, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not success:
        raise ValueError("Failed to encode processed image as JPEG")
    return buffer.tobytes()


def save_processed_jpeg(jpeg_bytes: bytes, original_filename: str | None = None) -> Path:
    """Write annotated JPEG to the outputs directory; returns absolute path."""
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = "detected"
    if original_filename:
        stem = Path(original_filename).stem + "_detected"
    filename = f"{stem}_{uuid.uuid4().hex[:8]}.jpg"
    path = (output_dir / filename).resolve()
    path.write_bytes(jpeg_bytes)
    logger.info("Saved annotated image to %s", path)
    return path
