"""Safe helpers for output image paths and public media URLs."""

from __future__ import annotations

import os
import re
from pathlib import Path

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


def output_dir() -> Path:
    return Path(os.getenv("OUTPUT_DIR", "outputs")).resolve()


def media_url_for_filename(filename: str) -> str:
    """Relative URL the frontend can prefix with VITE_API_BASE_URL."""
    return f"/api/v1/media/outputs/{filename}"


def resolve_output_file(filename: str) -> Path:
    """
    Resolve a filename under OUTPUT_DIR.

    Rejects path separators and traversal. Raises ValueError if unsafe or missing.
    """
    if not filename or not _SAFE_FILENAME.match(filename):
        raise ValueError("Invalid filename")
    if filename in {".", ".."}:
        raise ValueError("Invalid filename")

    base = output_dir()
    candidate = (base / filename).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("Path traversal rejected") from exc
    if not candidate.is_file():
        raise FileNotFoundError(filename)
    return candidate
