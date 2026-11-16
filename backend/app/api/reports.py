"""PDF investigation report download API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.core.security import get_current_user
from app.db.repositories import get_case_by_id, list_alerts, log_audit
from app.services.report_service import generate_case_pdf_report

router = APIRouter(prefix="/api/reports", tags=["Investigation Reports"])


@router.get("/{case_db_id}/pdf")
def download_case_report(case_db_id: int, current_user: dict = Depends(get_current_user)) -> FileResponse:
    case = get_case_by_id(case_db_id)
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    alerts = list_alerts(case_db_id=case_db_id)
    pdf_path = generate_case_pdf_report(case=case, alerts=alerts)
    log_audit(
        "REPORT_GENERATED",
        "missing_case",
        str(case_db_id),
        f"pdf={pdf_path.name}",
        current_user["uid"],
        current_user["sub"],
    )
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
