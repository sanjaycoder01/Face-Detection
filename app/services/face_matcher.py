"""Match a reference face photo against faces in a target (event) image."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.detection_schema import BoundingBox, DetectedFace
from app.schemas.match_schema import MatchedFace
from app.services.compreface_client import CompreFaceClient, CompreFaceConfig, CompreFaceError
from app.services.face_detector import DetectorConfig

logger = logging.getLogger(__name__)

_matcher: FaceMatcherService | None = None


@dataclass(frozen=True)
class MatchResult:
    matches: list[MatchedFace]
    all_candidates: list[MatchedFace]
    all_faces: list[DetectedFace]
    total_faces_in_target: int
    similarity_threshold: float
    max_matches: int
    candidates_above_threshold: int


class FaceMatcherService:
    """Enrolls a temporary subject from a reference photo, then recognizes on target."""

    def __init__(
        self,
        config: DetectorConfig,
        similarity_threshold: float = 0.7,
        max_matches: int = 1,
    ) -> None:
        self._config = config
        self._similarity_threshold = similarity_threshold
        self._max_matches = max(0, max_matches)
        self._client: CompreFaceClient | None = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return (
            self._ready
            and self._client is not None
            and self._config.service_type == "recognition"
            and bool(self._config.api_key.strip())
        )

    def load(self) -> None:
        if self._config.service_type != "recognition":
            logger.warning(
                "Face matching disabled: set COMPREFACE_SERVICE_TYPE=recognition"
            )
            return
        if not self._config.api_key.strip():
            logger.warning("Face matching disabled: COMPREFACE_API_KEY is missing")
            return

        self._client = CompreFaceClient(
            CompreFaceConfig(
                base_url=self._config.compreface_url,
                api_key=self._config.api_key.strip(),
                service_type="recognition",
                det_prob_threshold=self._config.det_prob_threshold,
                limit=self._config.limit,
                timeout_seconds=self._config.timeout_seconds,
            )
        )
        self._ready = True
        logger.info(
            "Face matcher ready (threshold=%s, max_matches=%s)",
            self._similarity_threshold,
            self._max_matches,
        )

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._ready = False

    def match_faces(
        self,
        reference_bgr: np.ndarray,
        target_bgr: np.ndarray,
        *,
        reference_filename: str = "reference.jpg",
        target_filename: str = "target.jpg",
        max_matches: int | None = None,
    ) -> MatchResult:
        if not self.is_ready or self._client is None:
            raise RuntimeError(
                "Face matcher is not available (requires recognition service + API key)"
            )

        ref_for_enroll = _reference_single_face_crop(reference_bgr, self._client)
        ref_bytes = _encode_jpeg_bytes(ref_for_enroll)
        target_bytes = _encode_jpeg_bytes(target_bgr)
        subject = f"match-{uuid.uuid4().hex}"

        try:
            self._client.add_subject_example(
                subject,
                ref_bytes,
                filename=reference_filename,
            )
            payload = self._client.recognize(
                target_bytes,
                filename=target_filename,
                prediction_count=5,
            )
        except CompreFaceError as exc:
            raise RuntimeError(str(exc)) from exc
        finally:
            self._client.delete_subject(subject)

        limit = self._max_matches if max_matches is None else max(0, max_matches)
        return _parse_match_payload(
            payload,
            subject_name=subject,
            similarity_threshold=self._similarity_threshold,
            target_shape=target_bgr.shape,
            max_matches=limit,
        )


def _reference_single_face_crop(
    reference_bgr: np.ndarray, client: CompreFaceClient
) -> np.ndarray:
    """
    CompreFace rejects enrollment when multiple faces are in the reference image.
    Detect faces and crop the largest one (with padding) for enrollment.
    """
    try:
        faces = client.detect_faces(_encode_jpeg_bytes(reference_bgr))
    except CompreFaceError:
        faces = []

    if not faces:
        return reference_bgr

    if len(faces) == 1:
        box = faces[0].bounding_box
        return _crop_with_padding(reference_bgr, box.x1, box.y1, box.x2, box.y2)

    largest = max(
        faces,
        key=lambda f: (f.bounding_box.x2 - f.bounding_box.x1)
        * (f.bounding_box.y2 - f.bounding_box.y1),
    )
    box = largest.bounding_box
    logger.info(
        "Reference image has %s faces; enrolling largest face #%s",
        len(faces),
        largest.face_id,
    )
    return _crop_with_padding(reference_bgr, box.x1, box.y1, box.x2, box.y2)


def _crop_with_padding(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding_ratio: float = 0.25,
) -> np.ndarray:
    height, width = image.shape[:2]
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    return image[top:bottom, left:right].copy()


def _encode_jpeg_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Failed to encode image as JPEG")
    return buf.tobytes()


def _scale_box_to_image(
    x1: int, y1: int, x2: int, y2: int, height: int, width: int
) -> tuple[int, int, int, int]:
    """CompreFace boxes can exceed decoded dimensions on large images; scale if needed."""
    max_x = max(x1, x2)
    max_y = max(y1, y2)
    if max_x <= width and max_y <= height:
        return x1, y1, x2, y2
    scale_x = width / max_x if max_x > width else 1.0
    scale_y = height / max_y if max_y > height else 1.0
    scale = min(scale_x, scale_y)
    return (
        int(x1 * scale),
        int(y1 * scale),
        int(x2 * scale),
        int(y2 * scale),
    )


def _parse_match_payload(
    payload: dict,
    *,
    subject_name: str,
    similarity_threshold: float,
    target_shape: tuple[int, ...],
    max_matches: int = 1,
) -> MatchResult:
    height, width = target_shape[0], target_shape[1]
    raw_faces = payload.get("result") or []

    all_faces: list[tuple[int, DetectedFace]] = []
    matches: list[tuple[float, MatchedFace]] = []

    for item in raw_faces:
        box = item.get("box") or {}
        try:
            x1, y1, x2, y2 = _scale_box_to_image(
                int(box["x_min"]),
                int(box["y_min"]),
                int(box["x_max"]),
                int(box["y_max"]),
                height,
                width,
            )
            det_conf = float(box.get("probability", 0.0))
        except (KeyError, TypeError, ValueError):
            continue

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        best_similarity = 0.0
        for subj in item.get("subjects") or []:
            if subj.get("subject") != subject_name:
                continue
            best_similarity = max(best_similarity, float(subj.get("similarity", 0.0)))

        face = DetectedFace(
            face_id=0,
            bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            confidence=round(det_conf, 4),
        )
        all_faces.append((x1, face))

        if best_similarity >= similarity_threshold:
            matches.append(
                (
                    -best_similarity,
                    MatchedFace(
                        face_id=0,
                        bounding_box=face.bounding_box,
                        detection_confidence=face.confidence,
                        similarity=round(best_similarity, 4),
                        is_match=True,
                    ),
                )
            )

    all_faces.sort(key=lambda pair: (pair[0], pair[1].bounding_box.y1))
    numbered_all = [
        f.model_copy(update={"face_id": i})
        for i, (_, f) in enumerate(all_faces, start=1)
    ]

    matches.sort(key=lambda pair: (pair[0], pair[1].bounding_box.x1))
    candidates_above_threshold = len(matches)

    numbered_all_candidates = [
        m.model_copy(update={"face_id": i})
        for i, (_, m) in enumerate(matches, start=1)
    ]

    limited = matches
    if max_matches > 0:
        limited = matches[:max_matches]

    numbered_matches = [
        m.model_copy(update={"face_id": i})
        for i, (_, m) in enumerate(limited, start=1)
    ]

    return MatchResult(
        matches=numbered_matches,
        all_candidates=numbered_all_candidates,
        all_faces=numbered_all,
        total_faces_in_target=len(numbered_all),
        similarity_threshold=similarity_threshold,
        max_matches=max_matches,
        candidates_above_threshold=candidates_above_threshold,
    )


def get_matcher() -> FaceMatcherService:
    global _matcher
    if _matcher is None:
        _matcher = FaceMatcherService(
            DetectorConfig(), similarity_threshold=0.7, max_matches=1
        )
    return _matcher


def set_matcher(service: FaceMatcherService) -> None:
    global _matcher
    _matcher = service
