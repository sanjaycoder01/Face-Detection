"""API tests for face matching endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.schemas.detection_schema import BoundingBox
from app.schemas.match_schema import MatchedFace
from app.services import face_matcher as face_matcher_module
from app.services.face_detector import DetectorConfig
from app.services.face_matcher import FaceMatcherService, MatchResult
from tests.conftest import make_blank_jpeg
from tests.test_detection_api import _parse_multipart_json


@pytest.fixture
def mock_matcher():
    service = MagicMock()
    service.is_ready = True
    service.match_faces.return_value = MatchResult(
        matches=[
            MatchedFace(
                face_id=1,
                bounding_box=BoundingBox(x1=10, y1=10, x2=50, y2=50),
                detection_confidence=0.99,
                similarity=0.92,
            )
        ],
        all_faces=[],
        total_faces_in_target=5,
        similarity_threshold=0.75,
        max_matches=1,
        candidates_above_threshold=2,
        all_candidates=[
            MatchedFace(
                face_id=1,
                bounding_box=BoundingBox(x1=10, y1=10, x2=50, y2=50),
                detection_confidence=0.99,
                similarity=0.92,
            ),
            MatchedFace(
                face_id=2,
                bounding_box=BoundingBox(x1=100, y1=10, x2=150, y2=50),
                detection_confidence=0.95,
                similarity=0.80,
            ),
        ],
    )
    face_matcher_module.set_matcher(service)
    yield service
    face_matcher_module.set_matcher(
        FaceMatcherService(DetectorConfig(), similarity_threshold=0.7, max_matches=1)
    )


def test_match_success(client, mock_detector, mock_matcher, tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("HISTORY_FILE", str(tmp_path / "history.json"))
    response = client.post(
        "/api/v1/faces/match",
        data={"name": "Alex"},
        files={
            "reference": ("ref.jpg", make_blank_jpeg(), "image/jpeg"),
            "target": ("group.jpg", make_blank_jpeg(), "image/jpeg"),
        },
    )
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    meta = response.json()
    assert meta["match_count"] == 1
    assert meta["person_name"] == "Alex"
    assert meta["operation_id"]
    assert meta["max_matches"] == 1
    assert meta["candidates_above_threshold"] == 2
    assert meta["matches"][0]["similarity"] == 0.92
    assert meta["annotated_image_url"].startswith("/api/v1/media/outputs/")

    history = client.get("/api/v1/faces/history")
    assert history.status_code == 200
    payload = history.json()
    assert payload["total"] == 1
    assert payload["records"][0]["person_name"] == "Alex"
    assert payload["records"][0]["id"] == meta["operation_id"]

    media = client.get(meta["annotated_image_url"])
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("image/")


def test_match_accepts_lovable_aliases(client, mock_detector, mock_matcher, tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("HISTORY_FILE", str(tmp_path / "history.json"))
    response = client.post(
        "/api/v1/faces/match",
        data={"name": "Sanjay"},
        files={
            "reference_image": ("ref.jpg", make_blank_jpeg(), "image/jpeg"),
            "group_image": ("group.jpg", make_blank_jpeg(), "image/jpeg"),
        },
    )
    assert response.status_code == 200
    meta = response.json()
    assert meta["person_name"] == "Sanjay"
    assert meta["match_count"] == 1


def test_match_multipart_legacy(client, mock_detector, mock_matcher, tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("HISTORY_FILE", str(tmp_path / "history.json"))
    response = client.post(
        "/api/v1/faces/match?response_format=multipart",
        files={
            "reference": ("ref.jpg", make_blank_jpeg(), "image/jpeg"),
            "target": ("group.jpg", make_blank_jpeg(), "image/jpeg"),
        },
    )
    assert response.status_code == 200
    assert "multipart/mixed" in response.headers["content-type"]
    meta = _parse_multipart_json(response.content, response.headers["content-type"])
    assert meta["match_count"] == 1


def test_match_unavailable(client, mock_detector):
    off = MagicMock()
    off.is_ready = False
    face_matcher_module.set_matcher(off)
    try:
        response = client.post(
            "/api/v1/faces/match",
            files={
                "reference": ("ref.jpg", make_blank_jpeg(), "image/jpeg"),
                "target": ("group.jpg", make_blank_jpeg(), "image/jpeg"),
            },
        )
        assert response.status_code == 503
    finally:
        face_matcher_module.set_matcher(
            FaceMatcherService(DetectorConfig(), similarity_threshold=0.7, max_matches=1)
        )


def test_history_empty(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HISTORY_FILE", str(tmp_path / "empty-history.json"))
    response = client.get("/api/v1/faces/history")
    assert response.status_code == 200
    assert response.json() == {"success": True, "total": 0, "records": []}


def test_media_rejects_unsafe_filename(client):
    response = client.get("/api/v1/media/outputs/not%20safe.jpg")
    assert response.status_code == 400


def test_cors_preflight(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "https://id-preview--aec36bb1-da36-4b41-9f17-df680b48ef48.lovable.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code in (200, 204)
    assert (
        response.headers.get("access-control-allow-origin")
        == "https://id-preview--aec36bb1-da36-4b41-9f17-df680b48ef48.lovable.app"
    )
