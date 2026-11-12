"""Persist CompreFace face embeddings to local JSON files for inspection."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fixed filenames so you can open them after each match/detect run.
TARGET_PHOTO_EMBEDDINGS_FILE = "target_photo_embeddings.json"
GROUP_PHOTO_EMBEDDINGS_FILE = "group_photo_embeddings.json"


def embeddings_dir() -> Path:
    path = Path(os.getenv("EMBEDDINGS_DIR", "outputs/embeddings"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _faces_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    faces: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("result") or [], start=1):
        box = item.get("box") or {}
        embedding = item.get("embedding")
        entry: dict[str, Any] = {
            "face_id": index,
            "bounding_box": {
                "x1": box.get("x_min"),
                "y1": box.get("y_min"),
                "x2": box.get("x_max"),
                "y2": box.get("y_max"),
            },
            "detection_confidence": box.get("probability"),
            "embedding_dim": len(embedding) if embedding is not None else 0,
            "embedding": [float(v) for v in embedding] if embedding is not None else None,
        }
        subjects = item.get("subjects")
        if subjects:
            entry["subjects"] = subjects
        faces.append(entry)
    return faces


def save_embeddings_file(
    *,
    kind: str,
    source_filename: str,
    payload: dict[str, Any],
) -> Path:
    """
    Write embeddings for one image.

    kind: "target" (reference/selfie) or "group" (event photo)
    """
    if kind == "target":
        filename = TARGET_PHOTO_EMBEDDINGS_FILE
    elif kind == "group":
        filename = GROUP_PHOTO_EMBEDDINGS_FILE
    else:
        raise ValueError("kind must be 'target' or 'group'")

    faces = _faces_from_payload(payload)
    document = {
        "kind": kind,
        "source_filename": source_filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_faces": len(faces),
        "faces_with_embeddings": sum(1 for f in faces if f.get("embedding") is not None),
        "faces": faces,
    }

    path = embeddings_dir() / filename
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    logger.info(
        "Saved %s embeddings (%s face(s)) → %s",
        kind,
        document["faces_with_embeddings"],
        path.resolve(),
    )
    print(
        f"[embeddings-file] {kind} → {path.resolve()} "
        f"({document['faces_with_embeddings']} embedding(s))",
        flush=True,
    )
    return path
