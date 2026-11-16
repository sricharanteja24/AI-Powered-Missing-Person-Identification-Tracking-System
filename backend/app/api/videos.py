"""CCTV video upload, async job dispatch, and progress tracking API routes."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.security import get_current_user, require_roles
from app.db.repositories import create_video_record, get_video_by_id, list_videos, log_audit
from app.services.job_manager import job_manager

router = APIRouter(prefix="/api/videos", tags=["CCTV Videos"])


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
def upload_video(
    camera_id: str = Form(default="CAM-01"),
    file: UploadFile = File(...),
    current_user: dict = Depends(require_roles("admin", "investigator")),
) -> dict:
    ext = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported video format")

    dest = settings.upload_dir / "videos" / f"vid_{uuid.uuid4().hex[:10]}{ext}"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video = create_video_record(
        video_name=file.filename or dest.name,
        camera_id=camera_id,
        file_path=str(dest),
        uploaded_by=current_user["uid"],
    )
    job_manager.enqueue_video(video["id"])
    log_audit(
        "VIDEO_UPLOADED",
        "video",
        str(video["id"]),
        f"camera={camera_id}, file={video['video_name']}",
        current_user["uid"],
        current_user["sub"],
    )
    return {"video": video}


@router.get("")
def get_videos(current_user: dict = Depends(get_current_user)) -> dict:
    return {"videos": list_videos()}


@router.get("/{video_id}")
def get_video_status(video_id: int, current_user: dict = Depends(get_current_user)) -> dict:
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return {"video": video}
