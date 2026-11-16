"""Parameterized repository layer for database operations."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.db.database import db_session


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_audit(
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    details: str | None = None,
    user_id: int | None = None,
    username: str | None = None,
) -> None:
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO audit_logs (user_id, username, action, resource_type, resource_id, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, action, resource_type, resource_id, details, utc_now()),
        )


def list_audit_logs(limit: int = 100) -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# --- Users ---
def create_user(username: str, full_name: str, email: str, password_hash: str, role: str) -> dict[str, Any]:
    now = utc_now()
    with db_session() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (username, full_name, email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username, full_name, email, password_hash, role, now),
        )
        row = conn.execute("SELECT id, username, full_name, email, role, is_active, created_at FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def count_users() -> int:
    with db_session() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()
        return int(row["cnt"])


# --- Missing Cases ---
def create_case(
    case_id: str,
    full_name: str,
    age: int | None,
    gender: str | None,
    last_known_location: str,
    last_seen_date: str,
    description: str | None,
    created_by: int,
    photo_path: str | None = None,
    face_embedding: list[float] | None = None,
    reid_embedding: list[float] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    with db_session() as conn:
        cur = conn.execute(
            """
            INSERT INTO missing_cases (
                case_id, full_name, age, gender, last_known_location, last_seen_date,
                description, photo_path, face_embedding_json, reid_embedding_json,
                status, created_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
            """,
            (
                case_id,
                full_name,
                age,
                gender,
                last_known_location,
                last_seen_date,
                description,
                photo_path,
                json.dumps(face_embedding) if face_embedding is not None else None,
                json.dumps(reid_embedding) if reid_embedding is not None else None,
                created_by,
                now,
                now,
            ),
        )
        row = conn.execute("SELECT * FROM missing_cases WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _sanitize_case(dict(row))


def _sanitize_case(row: dict[str, Any], include_embeddings: bool = False) -> dict[str, Any]:
    row["has_face_embedding"] = row.get("face_embedding_json") is not None
    row["has_reid_embedding"] = row.get("reid_embedding_json") is not None
    if not include_embeddings:
        row.pop("face_embedding_json", None)
        row.pop("reid_embedding_json", None)
    return row


def list_cases(status_filter: str | None = None) -> list[dict[str, Any]]:
    with db_session() as conn:
        if status_filter:
            rows = conn.execute(
                "SELECT * FROM missing_cases WHERE status = ? ORDER BY id DESC", (status_filter,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM missing_cases ORDER BY id DESC").fetchall()
        return [_sanitize_case(dict(r)) for r in rows]


def get_open_cases_with_embeddings() -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM missing_cases WHERE status = 'open'").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["face_embedding"] = json.loads(d["face_embedding_json"]) if d["face_embedding_json"] else None
            d["reid_embedding"] = json.loads(d["reid_embedding_json"]) if d["reid_embedding_json"] else None
            result.append(d)
        return result


def get_case_by_id(case_db_id: int) -> dict[str, Any] | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM missing_cases WHERE id = ?", (case_db_id,)).fetchone()
        return _sanitize_case(dict(row)) if row else None


def update_case_photo_and_embeddings(
    case_db_id: int,
    photo_path: str,
    face_embedding: list[float] | None,
    reid_embedding: list[float] | None,
) -> dict[str, Any] | None:
    with db_session() as conn:
        conn.execute(
            """
            UPDATE missing_cases
            SET photo_path = ?, face_embedding_json = ?, reid_embedding_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                photo_path,
                json.dumps(face_embedding) if face_embedding is not None else None,
                json.dumps(reid_embedding) if reid_embedding is not None else None,
                utc_now(),
                case_db_id,
            ),
        )
        row = conn.execute("SELECT * FROM missing_cases WHERE id = ?", (case_db_id,)).fetchone()
        return _sanitize_case(dict(row)) if row else None


def delete_case(case_db_id: int) -> bool:
    with db_session() as conn:
        cur = conn.execute("DELETE FROM missing_cases WHERE id = ?", (case_db_id,))
        return cur.rowcount > 0


# --- Videos ---
def create_video_record(video_name: str, camera_id: str, file_path: str, uploaded_by: int) -> dict[str, Any]:
    with db_session() as conn:
        cur = conn.execute(
            """
            INSERT INTO videos (video_name, camera_id, file_path, status, uploaded_by, created_at)
            VALUES (?, ?, ?, 'queued', ?, ?)
            """,
            (video_name, camera_id, file_path, uploaded_by, utc_now()),
        )
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def update_video_progress(
    video_id: int,
    status_val: str,
    progress_pct: float,
    processed_frames: int = 0,
    total_frames: int = 0,
    fps: float = 0.0,
    duration_sec: float = 0.0,
    error_message: str | None = None,
) -> None:
    completed_at = utc_now() if status_val in ("completed", "failed") else None
    with db_session() as conn:
        conn.execute(
            """
            UPDATE videos
            SET status = ?, progress_pct = ?, processed_frames = ?, total_frames = ?,
                fps = ?, duration_sec = ?, error_message = ?,
                completed_at = COALESCE(?, completed_at)
            WHERE id = ?
            """,
            (
                status_val,
                round(progress_pct, 2),
                processed_frames,
                total_frames,
                fps,
                duration_sec,
                error_message,
                completed_at,
                video_id,
            ),
        )


def list_videos() -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM videos ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def get_video_by_id(video_id: int) -> dict[str, Any] | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return dict(row) if row else None


# --- Match Alerts ---
def create_match_alert(
    case_db_id: int,
    video_id: int,
    track_id: int,
    camera_id: str,
    timestamp_sec: float,
    frame_number: int,
    face_similarity: float | None,
    reid_similarity: float | None,
    detection_confidence: float,
    face_quality_ok: bool,
    match_tier: str,
    limitation_note: str,
    evidence_image_path: str,
) -> dict[str, Any]:
    with db_session() as conn:
        cur = conn.execute(
            """
            INSERT INTO match_alerts (
                case_db_id, video_id, track_id, camera_id, timestamp_sec, frame_number,
                face_similarity, reid_similarity, detection_confidence, face_quality_ok,
                match_tier, limitation_note, evidence_image_path, verification_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending_review', ?)
            """,
            (
                case_db_id,
                video_id,
                track_id,
                camera_id,
                round(timestamp_sec, 2),
                frame_number,
                round(face_similarity, 4) if face_similarity is not None else None,
                round(reid_similarity, 4) if reid_similarity is not None else None,
                round(detection_confidence, 4),
                1 if face_quality_ok else 0,
                match_tier,
                limitation_note,
                evidence_image_path,
                utc_now(),
            ),
        )
        row = conn.execute("SELECT * FROM match_alerts WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def list_alerts(case_db_id: int | None = None, verification_status: str | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT a.*, c.case_id AS case_code, c.full_name AS missing_person_name, v.video_name
        FROM match_alerts a
        JOIN missing_cases c ON a.case_db_id = c.id
        JOIN videos v ON a.video_id = v.id
        WHERE 1=1
    """
    params: list[Any] = []
    if case_db_id is not None:
        query += " AND a.case_db_id = ?"
        params.append(case_db_id)
    if verification_status:
        query += " AND a.verification_status = ?"
        params.append(verification_status)
    query += " ORDER BY a.id DESC"

    with db_session() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def verify_alert(
    alert_id: int, verification_status: str, verified_by: int, notes: str | None
) -> dict[str, Any] | None:
    with db_session() as conn:
        conn.execute(
            """
            UPDATE match_alerts
            SET verification_status = ?, verified_by = ?, verification_notes = ?, verified_at = ?
            WHERE id = ?
            """,
            (verification_status, verified_by, notes, utc_now(), alert_id),
        )
        row = conn.execute("SELECT * FROM match_alerts WHERE id = ?", (alert_id,)).fetchone()
        return dict(row) if row else None
