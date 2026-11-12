"""Tests for embedding JSON file persistence."""

from __future__ import annotations

import json

from app.services.embedding_store import (
    GROUP_PHOTO_EMBEDDINGS_FILE,
    TARGET_PHOTO_EMBEDDINGS_FILE,
    save_embeddings_file,
)


def test_save_target_and_group_embedding_files(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_DIR", str(tmp_path))

    target_payload = {
        "result": [
            {
                "box": {
                    "x_min": 1,
                    "y_min": 2,
                    "x_max": 3,
                    "y_max": 4,
                    "probability": 0.99,
                },
                "embedding": [0.1, 0.2, 0.3],
            }
        ]
    }
    group_payload = {
        "result": [
            {
                "box": {
                    "x_min": 10,
                    "y_min": 20,
                    "x_max": 30,
                    "y_max": 40,
                    "probability": 0.95,
                },
                "embedding": [0.4, 0.5],
                "subjects": [{"subject": "match-abc", "similarity": 0.9}],
            }
        ]
    }

    target_path = save_embeddings_file(
        kind="target",
        source_filename="Target1.jpg",
        payload=target_payload,
    )
    group_path = save_embeddings_file(
        kind="group",
        source_filename="Fullimage.jpg",
        payload=group_payload,
    )

    assert target_path.name == TARGET_PHOTO_EMBEDDINGS_FILE
    assert group_path.name == GROUP_PHOTO_EMBEDDINGS_FILE

    target_doc = json.loads(target_path.read_text())
    group_doc = json.loads(group_path.read_text())

    assert target_doc["kind"] == "target"
    assert target_doc["total_faces"] == 1
    assert target_doc["faces"][0]["embedding"] == [0.1, 0.2, 0.3]

    assert group_doc["kind"] == "group"
    assert group_doc["faces"][0]["embedding"] == [0.4, 0.5]
    assert group_doc["faces"][0]["subjects"][0]["similarity"] == 0.9
