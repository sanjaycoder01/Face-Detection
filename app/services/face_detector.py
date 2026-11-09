"""Face detection backed by CompreFace REST API (detection only, no recognition)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.schemas.detection_schema import DetectedFace
from app.services.compreface_client import (
    CompreFaceClient,
    CompreFaceConfig,
    CompreFaceError,
)

logger = logging.getLogger(__name__)

_detector: FaceDetectorService | None = None


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the CompreFace-backed detector."""

    compreface_url: str = "http://localhost:8080"
    api_key: str = ""
    service_type: str = "detection"
    det_prob_threshold: float | None = None
    limit: int = 0
    max_detect_size: int = 640
    timeout_seconds: float = 60.0


class FaceDetectorService:
    """Delegates inference to a CompreFace Face Detection service instance."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._client: CompreFaceClient | None = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self._client is not None

    def load(self) -> None:
        """Validate config and verify CompreFace is reachable."""
        if self._ready:
            return

        if not self._config.api_key.strip():
            logger.error(
                "COMPREFACE_API_KEY is missing; face detection will be unavailable"
            )
            return

        service_type = self._config.service_type.strip().lower()
        if service_type not in ("detection", "recognition"):
            raise RuntimeError(
                "COMPREFACE_SERVICE_TYPE must be 'detection' or 'recognition'"
            )

        client = CompreFaceClient(
            CompreFaceConfig(
                base_url=self._config.compreface_url,
                api_key=self._config.api_key.strip(),
                service_type=service_type,
                det_prob_threshold=self._config.det_prob_threshold,
                limit=self._config.limit,
                timeout_seconds=self._config.timeout_seconds,
            )
        )

        if not client.ping():
            logger.warning(
                "CompreFace at %s is not reachable yet; will retry on first detect",
                self._config.compreface_url,
            )

        self._client = client
        self._ready = True
        logger.info(
            "CompreFace face detection configured (url=%s)",
            self._config.compreface_url,
        )

    def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        """
        Detect all faces in a BGR OpenCV image via CompreFace.

        Returns structured results with 1-based face_id, bbox, and confidence.
        """
        if not self.is_ready or self._client is None:
            raise RuntimeError("Face detector is not initialized")

        if image is None or image.size == 0:
            return []

        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Failed to encode image as JPEG for CompreFace")

        try:
            return self._client.detect_faces(buf.tobytes())
        except CompreFaceError as exc:
            raise RuntimeError(str(exc)) from exc

    def model_info(self) -> dict[str, Any]:
        return {
            "model_pack": "CompreFace",
            "detection_model": (
                f"CompreFace {self._config.service_type} service (detect faces only)"
            ),
            "det_size": self._config.max_detect_size,
            "execution_provider": f"CompreFace @ {self._config.compreface_url}",
            "allowed_modules": ["detection"],
            "model_loaded": self.is_ready,
            "licensing_note": (
                "CompreFace is Apache 2.0; underlying ML models may have separate terms. "
                "See https://github.com/exadel-inc/CompreFace"
            ),
        }

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._ready = False


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
