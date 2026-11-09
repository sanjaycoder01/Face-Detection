"""HTTP client for CompreFace Face Detection service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.schemas.detection_schema import BoundingBox, DetectedFace

logger = logging.getLogger(__name__)


ServiceType = str  # "detection" | "recognition"


@dataclass(frozen=True)
class CompreFaceConfig:
    base_url: str = "http://localhost:8080"
    api_key: str = ""
    service_type: ServiceType = "detection"
    det_prob_threshold: float | None = None
    limit: int = 0
    timeout_seconds: float = 60.0

    @property
    def detect_path(self) -> str:
        if self.service_type == "recognition":
            return "/api/v1/recognition/recognize"
        return "/api/v1/detection/detect"


class CompreFaceError(RuntimeError):
    """CompreFace API returned an error or the request failed."""


def _parse_detection_response(payload: dict[str, Any]) -> list[DetectedFace]:
    """Map CompreFace detection JSON to internal face models."""
    raw_faces = payload.get("result")
    if not raw_faces:
        return []

    parsed: list[tuple[int, DetectedFace]] = []
    for item in raw_faces:
        box = item.get("box") or {}
        try:
            x1 = int(box["x_min"])
            y1 = int(box["y_min"])
            x2 = int(box["x_max"])
            y2 = int(box["y_max"])
            confidence = float(box.get("probability", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed CompreFace box: %s", box)
            continue

        parsed.append(
            (
                x1,
                DetectedFace(
                    face_id=0,
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=round(confidence, 4),
                ),
            )
        )

    parsed.sort(key=lambda pair: (pair[0], pair[1].bounding_box.y1))
    return [
        face.model_copy(update={"face_id": index})
        for index, (_, face) in enumerate(parsed, start=1)
    ]


class CompreFaceClient:
    """Calls CompreFace detection or recognition (detect-only) endpoints."""

    def __init__(self, config: CompreFaceConfig) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def ping(self) -> bool:
        """Best-effort reachability check (UI gateway, no API key required)."""
        try:
            response = self._client.get("/")
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    def detect_faces(
        self,
        image_bytes: bytes,
        *,
        filename: str = "image.jpg",
        content_type: str = "image/jpeg",
    ) -> list[DetectedFace]:
        if not self._config.api_key:
            raise CompreFaceError("COMPREFACE_API_KEY is not configured")

        params: dict[str, str | int | float] = {}
        if self._config.limit:
            params["limit"] = self._config.limit
        if self._config.det_prob_threshold is not None:
            params["det_prob_threshold"] = self._config.det_prob_threshold

        headers = {"x-api-key": self._config.api_key}
        files = {"file": (filename, image_bytes, content_type)}

        try:
            response = self._client.post(
                self._config.detect_path,
                params=params,
                headers=headers,
                files=files,
            )
        except httpx.HTTPError as exc:
            logger.exception("CompreFace request failed")
            raise CompreFaceError("Could not reach CompreFace service") from exc

        if response.status_code == 401:
            raise CompreFaceError("Invalid CompreFace API key")
        if response.status_code >= 400:
            detail = response.text[:500]
            raise CompreFaceError(
                f"CompreFace detection failed (HTTP {response.status_code}): {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise CompreFaceError("CompreFace returned non-JSON response") from exc

        faces = _parse_detection_response(payload)
        logger.info("CompreFace detected %s face(s)", len(faces))
        return faces

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int | float] | None = None,
        json_body: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> dict[str, Any]:
        if not self._config.api_key:
            raise CompreFaceError("COMPREFACE_API_KEY is not configured")

        headers = {"x-api-key": self._config.api_key}
        try:
            response = self._client.request(
                method,
                path,
                params=params,
                headers=headers,
                json=json_body,
                files=files,
            )
        except httpx.HTTPError as exc:
            logger.exception("CompreFace request failed")
            raise CompreFaceError("Could not reach CompreFace service") from exc

        if response.status_code == 401:
            raise CompreFaceError("Invalid CompreFace API key")
        if response.status_code >= 400:
            detail = response.text[:500]
            raise CompreFaceError(
                f"CompreFace request failed (HTTP {response.status_code}): {detail}"
            )

        if not response.content.strip():
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise CompreFaceError("CompreFace returned non-JSON response") from exc

    def add_subject_example(
        self,
        subject: str,
        image_bytes: bytes,
        *,
        filename: str = "reference.jpg",
        content_type: str = "image/jpeg",
    ) -> None:
        """Enroll one face image under a subject in the recognition collection."""
        if self._config.service_type != "recognition":
            raise CompreFaceError(
                "Face matching requires COMPREFACE_SERVICE_TYPE=recognition"
            )
        self._request(
            "POST",
            "/api/v1/recognition/faces",
            params={"subject": subject},
            files={"file": (filename, image_bytes, content_type)},
        )

    def delete_subject(self, subject: str) -> None:
        if self._config.service_type != "recognition":
            return
        try:
            self._request(
                "DELETE",
                f"/api/v1/recognition/subjects/{subject}",
            )
        except CompreFaceError:
            logger.warning("Failed to delete temporary subject %s", subject)

    def recognize(
        self,
        image_bytes: bytes,
        *,
        filename: str = "target.jpg",
        content_type: str = "image/jpeg",
        prediction_count: int = 5,
    ) -> dict[str, Any]:
        if self._config.service_type != "recognition":
            raise CompreFaceError(
                "Face matching requires COMPREFACE_SERVICE_TYPE=recognition"
            )

        params: dict[str, str | int | float] = {"prediction_count": prediction_count}
        if self._config.limit:
            params["limit"] = self._config.limit
        if self._config.det_prob_threshold is not None:
            params["det_prob_threshold"] = self._config.det_prob_threshold

        headers = {"x-api-key": self._config.api_key}
        files = {"file": (filename, image_bytes, content_type)}
        try:
            response = self._client.post(
                "/api/v1/recognition/recognize",
                params=params,
                headers=headers,
                files=files,
            )
        except httpx.HTTPError as exc:
            logger.exception("CompreFace recognize failed")
            raise CompreFaceError("Could not reach CompreFace service") from exc

        if response.status_code == 401:
            raise CompreFaceError("Invalid CompreFace API key")
        if response.status_code >= 400:
            detail = response.text[:500]
            raise CompreFaceError(
                f"CompreFace recognition failed (HTTP {response.status_code}): {detail}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise CompreFaceError("CompreFace returned non-JSON response") from exc
