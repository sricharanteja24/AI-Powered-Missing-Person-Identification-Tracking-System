"""SQLite database connection manager with WAL mode and indexed schema."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.core.config import settings


def get_connection() -> sqlite3.Connection:
    settings.ensure_dirs()
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


@contextmanager
def db_session() -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize relational tables and performance indexes."""
    with db_session() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'investigator',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS missing_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                last_known_location TEXT NOT NULL,
                last_seen_date TEXT NOT NULL,
                description TEXT,
                photo_path TEXT,
                face_embedding_json TEXT,
                reid_embedding_json TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                created_by INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (created_by) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                duration_sec REAL DEFAULT 0.0,
                fps REAL DEFAULT 0.0,
                total_frames INTEGER DEFAULT 0,
                processed_frames INTEGER DEFAULT 0,
                progress_pct REAL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'queued',
                error_message TEXT,
                uploaded_by INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (uploaded_by) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS match_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_db_id INTEGER NOT NULL,
                video_id INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                camera_id TEXT NOT NULL,
                timestamp_sec REAL NOT NULL,
                frame_number INTEGER NOT NULL,
                face_similarity REAL,
                reid_similarity REAL,
                detection_confidence REAL NOT NULL,
                face_quality_ok INTEGER NOT NULL DEFAULT 0,
                match_tier TEXT NOT NULL,
                limitation_note TEXT NOT NULL,
                evidence_image_path TEXT NOT NULL,
                verification_status TEXT NOT NULL DEFAULT 'pending_review',
                verified_by INTEGER,
                verification_notes TEXT,
                verified_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (case_db_id) REFERENCES missing_cases(id) ON DELETE CASCADE,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
                FOREIGN KEY (verified_by) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT,
                details TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_cases_case_id ON missing_cases(case_id);
            CREATE INDEX IF NOT EXISTS idx_cases_status ON missing_cases(status);
            CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
            CREATE INDEX IF NOT EXISTS idx_alerts_case ON match_alerts(case_db_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_video ON match_alerts(video_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_status ON match_alerts(verification_status);
            CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC);
            """
        )
