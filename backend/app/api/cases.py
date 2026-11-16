"""Missing person case management API routes."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.security import get_current_user, require_roles
from app.db.repositories import (
    create_case,
    delete_case,
    get_case_by_id,
    list_cases,
    log_audit,
    update_case_photo_and_embeddings,
)
from app.ml.face_engine import face_engine
from app.ml.reid_engine import reid_engine
from app.models.schemas import CaseCreate

router = APIRouter(prefix="/api/cases", tags=["Missing Person Cases"])


@router.post("", status_code=status.HTTP_201_CREATED)
def register_case(
    payload: CaseCreate,
    current_user: dict = Depends(require_roles("admin", "investigator")),
) -> dict:
    try:
        case = create_case(
            case_id=payload.case_id,
            full_name=payload.full_name,
            age=payload.age,
            gender=payload.gender,
            last_known_location=payload.last_known_location,
            last_seen_date=payload.last_seen_date,
            description=payload.description,
            created_by=current_user["uid"],
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Could not create case: {exc}") from exc

    log_audit(
        "CASE_CREATED",
        "missing_case",
        str(case["id"]),
        f"case_id={payload.case_id}",
        current_user["uid"],
        current_user["sub"],
    )
    return {"case": case}


@router.get("")
def get_cases(
    status_filter: str | None = None,
    current_user: dict = Depends(get_current_user),
) -> dict:
    return {"cases": list_cases(status_filter=status_filter)}


@router.post("/{case_db_id}/photo")
def upload_reference_photo(
    case_db_id: int,
    file: UploadFile = File(...),
    current_user: dict = Depends(require_roles("admin", "investigator")),
) -> dict:
    case = get_case_by_id(case_db_id)
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    ext = Path(file.filename or "photo.jpg").suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported image format")

    dest = settings.upload_dir / "photos" / f"case_{case_db_id}_{uuid.uuid4().hex[:8]}{ext}"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(str(dest))
    if image is None:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or unreadable image file")

    face_result = face_engine.extract_reference_embedding(image)
    reid_emb = reid_engine.extract_embedding(image)

    updated = update_case_photo_and_embeddings(
        case_db_id=case_db_id,
        photo_path=str(dest),
        face_embedding=face_result.embedding.tolist() if face_result.embedding is not None else None,
        reid_embedding=reid_emb.tolist() if reid_emb is not None else None,
    )
    log_audit(
        "CASE_PHOTO_UPLOADED",
        "missing_case",
        str(case_db_id),
        f"face_detected={face_result.quality_ok}",
        current_user["uid"],
        current_user["sub"],
    )
    return {
        "case": updated,
        "face_detected": face_result.quality_ok,
        "face_quality_note": face_result.reason,
    }


@router.delete("/{case_db_id}")
def remove_case(
    case_db_id: int,
    current_user: dict = Depends(require_roles("admin", "investigator")),
) -> dict:
    case = get_case_by_id(case_db_id)
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    if case.get("photo_path"):
        Path(case["photo_path"]).unlink(missing_ok=True)
    delete_case(case_db_id)
    log_audit("CASE_DELETED", "missing_case", str(case_db_id), "Data retention deletion", current_user["uid"], current_user["sub"])
    return {"deleted": True, "case_db_id": case_db_id}
