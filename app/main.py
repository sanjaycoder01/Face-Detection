"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes.detection import router as detection_router
from app.routes.history import router as history_router
from app.routes.match import router as match_router
from app.routes.media import router as media_router
from app.services.face_detector import (
    DetectorConfig,
    FaceDetectorService,
    get_detector,
    set_detector,
)
from app.services.face_matcher import FaceMatcherService, get_matcher, set_matcher
from app.services.compreface_client import CompreFaceClient, CompreFaceConfig

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _optional_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return float(raw)


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ALLOWED_ORIGINS",
        ",".join(
            [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "https://id-preview--aec36bb1-da36-4b41-9f17-df680b48ef48.lovable.app",
            ]
        ),
    )
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to CompreFace once at startup."""
    config = DetectorConfig(
        compreface_url=os.getenv("COMPREFACE_URL", "http://localhost:8080"),
        api_key=os.getenv("COMPREFACE_API_KEY", ""),
        service_type=os.getenv("COMPREFACE_SERVICE_TYPE", "detection"),
        det_prob_threshold=_optional_float("COMPREFACE_DET_PROB_THRESHOLD"),
        limit=int(os.getenv("COMPREFACE_FACE_LIMIT", "0")),
        max_detect_size=int(os.getenv("COMPREFACE_MAX_DETECT_SIZE", "640")),
        timeout_seconds=float(os.getenv("COMPREFACE_TIMEOUT_SECONDS", "60")),
    )
    detector = FaceDetectorService(config)
    matcher = FaceMatcherService(
        config,
        similarity_threshold=float(
            os.getenv("COMPREFACE_MATCH_SIMILARITY_THRESHOLD", "0.75")
        ),
        max_matches=int(os.getenv("COMPREFACE_MATCH_MAX_RESULTS", "1")),
    )
    try:
        detector.load()
        set_detector(detector)
        matcher.load()
        set_matcher(matcher)
    except RuntimeError:
        logger.error("Application started without a loaded detection backend")
    yield
    detector.shutdown()
    matcher.shutdown()


app = FastAPI(
    title="Face Detection Service",
    description="Face detection and reference-to-crowd matching via CompreFace.",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(detection_router)
app.include_router(match_router)
app.include_router(history_router)
app.include_router(media_router)


def _compreface_status() -> str:
    api_key = os.getenv("COMPREFACE_API_KEY", "").strip()
    if not api_key:
        return "not_configured"
    url = os.getenv("COMPREFACE_URL", "http://localhost:8080")
    client = CompreFaceClient(
        CompreFaceConfig(
            base_url=url,
            api_key=api_key,
            service_type=os.getenv("COMPREFACE_SERVICE_TYPE", "detection"),
            timeout_seconds=3.0,
        )
    )
    try:
        return "connected" if client.ping() else "unreachable"
    finally:
        client.close()


@app.get("/health", tags=["health"])
def health():
    detector = get_detector()
    matcher = get_matcher()
    ready = detector.is_ready
    compreface = _compreface_status()
    status_value = "healthy" if ready and compreface == "connected" else "degraded"
    return {
        "status": status_value,
        "service": "face-recognition-api",
        "api": "running",
        "compreface": compreface,
        "model_loaded": ready,
        "face_matching_enabled": matcher.is_ready,
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
        },
    )
