"""InsightFace SCRFD-based face detection (detection only, no recognition)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from insightface.app import FaceAnalysis

from app.schemas.detection_schema import BoundingBox, DetectedFace

logger = logging.getLogger(__name__)

_init_lock = threading.Lock()
_detector: FaceDetectorService | None = None


@dataclass(frozen=True)
class DetectorConfig:
    model_name: str = "buffalo_l"
    det_size: int = 640
    ctx_id: int = -1  # CPU


class FaceDetectorService:
    """Wraps InsightFace FaceAnalysis configured for SCRFD detection only."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._app: FaceAnalysis | None = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self._app is not None

    def load(self) -> None:
        """Load pretrained weights once. Thread-safe for concurrent startup."""
        if self._ready:
            return
        with _init_lock:
            if self._ready:
                return
            logger.info(
                "Loading InsightFace model pack=%s det_size=%s (CPU)",
                self._config.model_name,
                self._config.det_size,
            )
            try:
                # Detection-only: SCRFD from the buffalo pack; no ArcFace / embeddings used.
                app = FaceAnalysis(
                    name=self._config.model_name,
                    allowed_modules=["detection"],
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(
                    ctx_id=self._config.ctx_id,
                    det_size=(self._config.det_size, self._config.det_size),
                )
                self._app = app
                self._ready = True
                logger.info("Face detection model loaded successfully")
            except Exception as exc:
                logger.exception("Failed to initialize face detection model")
                raise RuntimeError("Model initialization failed") from exc

    def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        """
        Detect all faces in a BGR OpenCV image.

        Returns structured results with 1-based face_id, bbox, and confidence.
        """
        if not self.is_ready or self._app is None:
            raise RuntimeError("Face detector is not initialized")

        if image is None or image.size == 0:
            return []

        raw_faces = self._app.get(image)
        detections: list[DetectedFace] = []

        for index, face in enumerate(raw_faces, start=1):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            score = float(getattr(face, "det_score", 0.0))

            detections.append(
                DetectedFace(
                    face_id=index,
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=round(score, 4),
                )
            )

        logger.info("Detected %s face(s) in image", len(detections))
        return detections

    def model_info(self) -> dict[str, Any]:
        return {
            "model_pack": self._config.model_name,
            "detection_model": "SCRFD (via InsightFace buffalo pack)",
            "det_size": self._config.det_size,
            "execution_provider": "CPUExecutionProvider",
            "allowed_modules": ["detection"],
            "model_loaded": self.is_ready,
            "licensing_note": (
                "InsightFace/buffalo model weights have their own license terms. "
                "Review https://github.com/deepinsight/insightface before commercial use."
            ),
        }


def get_detector() -> FaceDetectorService:
    """Return the singleton detector instance (must call load() at startup)."""
    global _detector
    if _detector is None:
        _detector = FaceDetectorService()
    return _detector


def set_detector(service: FaceDetectorService) -> None:
    """Replace the singleton (used at startup with env-based config)."""
    global _detector
    _detector = service
