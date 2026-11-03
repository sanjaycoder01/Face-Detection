"""API tests for face detection endpoints."""

from __future__ import annotations

import json
import re

import pytest

from app.schemas.detection_schema import BoundingBox, DetectedFace
from tests.conftest import make_blank_jpeg, make_blank_png


def _parse_multipart_json(content: bytes, content_type: str) -> dict:
    match = re.search(r"boundary=([^;\s]+)", content_type)
    assert match, content_type
    boundary = match.group(1).encode()
    parts = content.split(b"--" + boundary)
    for part in parts:
        if b"application/json" in part or b'"json"' in part:
            body = part.split(b"\r\n\r\n", 1)[-1].rstrip(b"\r\n-")
            return json.loads(body.decode())
    raise AssertionError("JSON part not found in multipart response")


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_model_info(client, mock_detector):
    response = client.get("/api/v1/faces/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_pack"] == "buffalo_l"
    assert "SCRFD" in data["detection_model"]


def test_detect_valid_image_with_faces(client, mock_detector):
    mock_detector.detect_faces.return_value = [
        DetectedFace(
            face_id=1,
            bounding_box=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            confidence=0.99,
        )
    ]
    response = client.post(
        "/api/v1/faces/detect",
        files={"image": ("group.jpg", make_blank_jpeg(), "image/jpeg")},
    )
    assert response.status_code == 200
    meta = _parse_multipart_json(response.content, response.headers["content-type"])
    assert meta["success"] is True
    assert meta["total_faces"] == 1
    assert meta["faces"][0]["face_id"] == 1


def test_detect_no_faces(client, mock_detector):
    mock_detector.detect_faces.return_value = []
    response = client.post(
        "/api/v1/faces/detect",
        files={"image": ("empty_scene.jpg", make_blank_jpeg(), "image/jpeg")},
    )
    assert response.status_code == 200
    meta = _parse_multipart_json(response.content, response.headers["content-type"])
    assert meta["total_faces"] == 0
    assert meta["faces"] == []


def test_invalid_file_format(client, mock_detector):
    response = client.post(
        "/api/v1/faces/detect",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_corrupted_image(client, mock_detector):
    response = client.post(
        "/api/v1/faces/detect",
        files={"image": ("bad.jpg", b"\xff\xd8\xff bogus", "image/jpeg")},
    )
    assert response.status_code == 400


def test_empty_upload(client, mock_detector):
    response = client.post(
        "/api/v1/faces/detect",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_png_supported(client, mock_detector):
    mock_detector.detect_faces.return_value = []
    response = client.post(
        "/api/v1/faces/detect",
        files={"image": ("photo.png", make_blank_png(), "image/png")},
    )
    assert response.status_code == 200
