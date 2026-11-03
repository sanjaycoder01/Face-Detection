"""Pytest fixtures with a mocked face detector (no model download in CI)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.detection_schema import BoundingBox, DetectedFace
from app.services import face_detector as face_detector_module


@pytest.fixture
def mock_detector():
    service = MagicMock()
    service.is_ready = True
    service.model_info.return_value = {
        "model_pack": "buffalo_l",
        "detection_model": "SCRFD (via InsightFace buffalo pack)",
        "det_size": 640,
        "execution_provider": "CPUExecutionProvider",
        "allowed_modules": ["detection"],
        "model_loaded": True,
        "licensing_note": "test",
    }
    face_detector_module.set_detector(service)
    yield service
    face_detector_module.set_detector(face_detector_module.FaceDetectorService())


@pytest.fixture
def client(mock_detector):
    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    app.router.lifespan_context = noop_lifespan
    with TestClient(app) as test_client:
        yield test_client


def make_blank_jpeg(width: int = 64, height: int = 64) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def make_blank_png(width: int = 64, height: int = 64) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()
