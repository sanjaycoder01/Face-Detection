"""Local JSON history of face-matching operations (demo persistence)."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _history_path() -> Path:
    path = Path(os.getenv("HISTORY_FILE", "data/history.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_records() -> list[dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read history file %s: %s", path, exc)
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    return []


def _write_records(records: list[dict[str, Any]]) -> None:
    path = _history_path()
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def append_match_history(
    *,
    person_name: str | None,
    match_count: int,
    total_faces_in_target: int,
    similarity_threshold: float,
    matches: list[dict[str, Any]],
    annotated_image_url: str | None,
    annotated_filename: str | None,
    message: str,
) -> dict[str, Any]:
    """Append one match operation. Does not invent past records."""
    record = {
        "id": str(uuid.uuid4()),
        "type": "match",
        "person_name": person_name.strip() if person_name and person_name.strip() else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "match_count": match_count,
        "total_faces_in_target": total_faces_in_target,
        "similarity_threshold": similarity_threshold,
        "matches": matches,
        "annotated_image_url": annotated_image_url,
        "annotated_filename": annotated_filename,
        "message": message,
    }
    with _lock:
        records = _read_records()
        records.insert(0, record)
        _write_records(records)
    logger.info("History recorded id=%s match_count=%s", record["id"], match_count)
    return record


def list_history(limit: int = 50) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 200))
    with _lock:
        records = _read_records()
    return records[:limit]
