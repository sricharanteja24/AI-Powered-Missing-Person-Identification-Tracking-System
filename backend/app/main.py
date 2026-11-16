"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import alerts, auth, cases, reports, videos
from app.core.config import settings
from app.core.logging_config import logger
from app.db.database import init_db
from app.ml.detector import person_detector
from app.ml.face_engine import face_engine
from app.ml.reid_engine import reid_engine
from app.services.job_manager import job_manager

settings.ensure_dirs()
init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    init_db()
    job_manager.start()
    logger.info("Application started successfully.")
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount evidence and uploaded photos for authenticated dashboard preview
app.mount("/media/evidence", StaticFiles(directory=str(settings.evidence_dir)), name="evidence")
app.mount("/media/photos", StaticFiles(directory=str(settings.upload_dir / "photos")), name="photos")

app.include_router(auth.router)
app.include_router(cases.router)
app.include_router(videos.router)
app.include_router(alerts.router)
app.include_router(reports.router)


@app.get("/api/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "version": settings.app_version,
        "models": {
            "detector": person_detector.backend,
            "face_engine": face_engine.backend,
            "reid_engine": reid_engine.backend,
        },
    }
