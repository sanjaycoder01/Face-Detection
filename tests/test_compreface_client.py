"""Unit tests for CompreFace response parsing."""

from app.services.compreface_client import _parse_detection_response


def test_parse_detection_response_sorts_left_to_right():
    payload = {
        "result": [
            {
                "box": {
                    "x_min": 200,
                    "y_min": 10,
                    "x_max": 300,
                    "y_max": 110,
                    "probability": 0.91,
                }
            },
            {
                "box": {
                    "x_min": 20,
                    "y_min": 10,
                    "x_max": 120,
                    "y_max": 110,
                    "probability": 0.99,
                }
            },
        ]
    }
    faces = _parse_detection_response(payload)
    assert len(faces) == 2
    assert faces[0].face_id == 1
    assert faces[0].bounding_box.x1 == 20
    assert faces[1].face_id == 2
    assert faces[1].bounding_box.x1 == 200


def test_parse_empty_result():
    assert _parse_detection_response({}) == []
    assert _parse_detection_response({"result": []}) == []
