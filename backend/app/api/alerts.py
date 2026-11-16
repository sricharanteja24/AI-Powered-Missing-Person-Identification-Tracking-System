"""Candidate match alerts, human verification, and audit log API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import get_current_user, require_roles
from app.db.repositories import list_alerts, list_audit_logs, log_audit, verify_alert
from app.models.schemas import AlertVerificationRequest

router = APIRouter(prefix="/api/alerts", tags=["Match Alerts & Audit"])


@router.get("")
def get_alerts(
    case_db_id: int | None = None,
    verification_status: str | None = None,
    current_user: dict = Depends(get_current_user),
) -> dict:
    return {"alerts": list_alerts(case_db_id=case_db_id, verification_status=verification_status)}


@router.post("/{alert_id}/verify")
def verify_candidate_alert(
    alert_id: int,
    payload: AlertVerificationRequest,
    current_user: dict = Depends(require_roles("admin", "investigator", "reviewer")),
) -> dict:
    updated = verify_alert(
        alert_id=alert_id,
        verification_status=payload.verification_status,
        verified_by=current_user["uid"],
        notes=payload.verification_notes,
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    log_audit(
        "ALERT_VERIFIED",
        "match_alert",
        str(alert_id),
        f"status={payload.verification_status}",
        current_user["uid"],
        current_user["sub"],
    )
    return {"alert": updated}


@router.get("/audit-logs")
def get_audit_history(
    limit: int = 100,
    current_user: dict = Depends(require_roles("admin", "investigator", "reviewer")),
) -> dict:
    return {"audit_logs": list_audit_logs(limit=limit)}
