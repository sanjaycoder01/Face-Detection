"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from app.routes.detection import router as detection_router
from app.services.face_detector import (
    DetectorConfig,
    FaceDetectorService,
    get_detector,
    set_detector,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the SCRFD detector once at startup."""
    config = DetectorConfig(
        model_name=os.getenv("FACE_MODEL_NAME", "buffalo_l"),
        det_size=int(os.getenv("DET_SIZE", "640")),
    )
    detector = FaceDetectorService(config)
    try:
        detector.load()
        set_detector(detector)
    except RuntimeError:
        logger.error("Application started without a loaded detection model")
    yield


app = FastAPI(
    title="Face Detection Service",
    description="Phase 1: SCRFD face detection via InsightFace (no recognition).",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(detection_router)


@app.get("/health", tags=["health"])
def health():
    detector = get_detector()
    return {
        "status": "ok" if detector.is_ready else "degraded",
        "model_loaded": detector.is_ready,
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
