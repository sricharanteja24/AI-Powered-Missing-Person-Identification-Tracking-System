"""PDF investigation report generator using ReportLab."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.config import settings


def _format_video_time(sec: float) -> str:
    s = float(sec or 0.0)
    mins = int(s // 60)
    rem = int(s % 60)
    whole = round(s)
    mod100 = whole % 100
    suffix = (
        "th"
        if 11 <= mod100 <= 13
        else {1: "st", 2: "nd", 3: "rd"}.get(whole % 10, "th")
    )
    return f"{mins:02d}:{rem:02d} ({whole}{suffix} sec / {s:.2f}s)"


def generate_case_pdf_report(case: dict[str, Any], alerts: list[dict[str, Any]]) -> Path:
    """Generate a formal investigation PDF report for a missing person case."""
    settings.ensure_dirs()
    output_path = settings.reports_dir / f"investigation_report_{case['case_id']}.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#991B1B"),
        backColor=colors.HexColor("#FEF2F2"),
        borderPadding=6,
        spaceAfter=10,
    )
    body_style = styles["Normal"]

    elements: list[Any] = [
        Paragraph("Missing Person Investigation & Candidate Match Report", title_style),
        Paragraph(
            "<b>IMPORTANT NOTICE:</b> AI-generated similarity measurements are investigative leads only and "
            "do <b>NOT</b> establish a person's identity. All candidate matches require human investigator verification.",
            disclaimer_style,
        ),
        Spacer(1, 4 * mm),
        Paragraph("<b>1. Case Information</b>", styles["Heading2"]),
    ]

    case_table_data = [
        ["Case ID", case.get("case_id", "N/A"), "Status", str(case.get("status", "open")).upper()],
        ["Full Name", case.get("full_name", "N/A"), "Age / Gender", f"{case.get('age') or 'N/A'} / {case.get('gender') or 'N/A'}"],
        ["Last Known Location", case.get("last_known_location", "N/A"), "Last Seen Date", case.get("last_seen_date", "N/A")],
    ]
    case_table = Table(case_table_data, colWidths=[35 * mm, 55 * mm, 32 * mm, 50 * mm])
    case_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(case_table)
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph(f"<b>2. Candidate Match Alerts ({len(alerts)})</b>", styles["Heading2"]))
    if not alerts:
        elements.append(Paragraph("No candidate match alerts have been recorded for this case.", body_style))
    else:
        for idx, alert in enumerate(alerts, start=1):
            elements.append(
                Paragraph(
                    f"<b>Candidate #{idx}</b> — Video: {alert.get('video_name', 'N/A')} | "
                    f"Camera: {alert.get('camera_id')} | Track ID: {alert.get('track_id')} | "
                    f"Detected at: {_format_video_time(alert.get('timestamp_sec', 0))} (Frame {alert.get('frame_number')})",
                    styles["Heading3"],
                )
            )
            face_sim = f"{alert['face_similarity']:.3f}" if alert.get("face_similarity") is not None else "N/A"
            reid_sim = f"{alert['reid_similarity']:.3f}" if alert.get("reid_similarity") is not None else "N/A"
            elements.append(
                Paragraph(
                    f"<b>Tier:</b> {alert.get('match_tier')} | "
                    f"<b>Face Cosine Sim:</b> {face_sim} | "
                    f"<b>OSNet Re-ID Sim:</b> {reid_sim} | "
                    f"<b>YOLO Conf:</b> {alert.get('detection_confidence', 0):.2f} | "
                    f"<b>Verification:</b> {alert.get('verification_status')}",
                    body_style,
                )
            )
            elements.append(Paragraph(f"<i>Limitation Note:</i> {alert.get('limitation_note')}", body_style))
            if alert.get("verification_notes"):
                elements.append(Paragraph(f"<i>Investigator Notes:</i> {alert.get('verification_notes')}", body_style))

            ev_path = Path(str(alert.get("evidence_image_path", "")))
            if ev_path.exists() and ev_path.is_file():
                try:
                    elements.append(Spacer(1, 2 * mm))
                    elements.append(RLImage(str(ev_path), width=75 * mm, height=48 * mm, kind="proportional"))
                except Exception:
                    pass
            elements.append(Spacer(1, 4 * mm))

    doc.build(elements)
    return output_path