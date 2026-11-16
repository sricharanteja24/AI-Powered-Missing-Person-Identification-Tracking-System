"""End-to-end integration and unit tests for the Missing Person Identification System."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

# Isolate test database and storage in a temporary directory
_TMP_DIR = tempfile.TemporaryDirectory()
os.environ["DATA_DIR"] = str(Path(_TMP_DIR.name) / "data")
os.environ["DB_PATH"] = str(Path(_TMP_DIR.name) / "data" / "test.db")
os.environ["UPLOAD_DIR"] = str(Path(_TMP_DIR.name) / "data" / "uploads")
os.environ["EVIDENCE_DIR"] = str(Path(_TMP_DIR.name) / "data" / "evidence")
os.environ["REPORTS_DIR"] = str(Path(_TMP_DIR.name) / "data" / "reports")
os.environ["LOG_DIR"] = str(Path(_TMP_DIR.name) / "logs")

from app.main import app  # noqa: E402
from app.ml.matcher import matching_engine  # noqa: E402


class TestMissingPersonSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_ctx.__exit__(None, None, None)
        _TMP_DIR.cleanup()

    def _create_synthetic_image(self, path: Path) -> None:
        img = np.zeros((256, 128, 3), dtype=np.uint8)
        cv2.rectangle(img, (34, 40), (94, 220), (40, 140, 220), -1)
        cv2.circle(img, (64, 28), 16, (210, 185, 165), -1)
        cv2.imwrite(str(path), img)

    def _create_synthetic_video(self, path: Path) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 10.0, (320, 240))
        for i in range(15):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            x_offset = 60 + i * 6
            cv2.rectangle(frame, (x_offset, 50), (x_offset + 50, 210), (40, 140, 220), -1)
            cv2.circle(frame, (x_offset + 25, 36), 14, (210, 185, 165), -1)
            writer.write(frame)
        writer.release()

    def test_01_health_check(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("models", data)

    def test_02_auth_case_video_alert_pdf_flow(self) -> None:
        # 1. Register & Login
        reg = self.client.post(
            "/api/auth/register",
            json={
                "username": "investigator1",
                "full_name": "Lead Investigator",
                "email": "inv1@agency.local",
                "password": "StrongPassword123!",
                "role": "investigator",
            },
        )
        self.assertEqual(reg.status_code, 201)

        login = self.client.post(
            "/api/auth/login",
            json={"username": "investigator1", "password": "StrongPassword123!"},
        )
        self.assertEqual(login.status_code, 200)
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Register Missing Person Case
        case_resp = self.client.post(
            "/api/cases",
            headers=headers,
            json={
                "case_id": "MP-2026-001",
                "full_name": "Test Subject",
                "age": 24,
                "gender": "Male",
                "last_known_location": "Central Station Gate 3",
                "last_seen_date": "2026-11-15",
                "description": "Blue jacket, dark trousers",
            },
        )
        self.assertEqual(case_resp.status_code, 201)
        case_db_id = case_resp.json()["case"]["id"]

        # 3. Upload Reference Photo
        photo_path = Path(_TMP_DIR.name) / "ref.jpg"
        self._create_synthetic_image(photo_path)
        with photo_path.open("rb") as f:
            photo_resp = self.client.post(
                f"/api/cases/{case_db_id}/photo",
                headers=headers,
                files={"file": ("ref.jpg", f, "image/jpeg")},
            )
        self.assertEqual(photo_resp.status_code, 200)
        self.assertTrue(photo_resp.json()["case"]["has_reid_embedding"])

        # 4. Upload Synthetic CCTV Video & Wait for Async Worker
        vid_path = Path(_TMP_DIR.name) / "cctv.mp4"
        self._create_synthetic_video(vid_path)
        with vid_path.open("rb") as f:
            vid_resp = self.client.post(
                "/api/videos/upload",
                headers=headers,
                data={"camera_id": "CAM-NORTH-01"},
                files={"file": ("cctv.mp4", f, "video/mp4")},
            )
        self.assertEqual(vid_resp.status_code, 202)
        video_id = vid_resp.json()["video"]["id"]

        status_val = "queued"
        for _ in range(30):
            poll = self.client.get(f"/api/videos/{video_id}", headers=headers)
            self.assertEqual(poll.status_code, 200)
            status_val = poll.json()["video"]["status"]
            if status_val in ("completed", "failed"):
                break
            time.sleep(0.1)
        self.assertEqual(status_val, "completed")

        # 5. Generate PDF Report
        pdf_resp = self.client.get(f"/api/reports/{case_db_id}/pdf", headers=headers)
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertTrue(pdf_resp.content.startswith(b"%PDF"))

        # 6. Audit Logs Check
        audit_resp = self.client.get("/api/alerts/audit-logs", headers=headers)
        self.assertEqual(audit_resp.status_code, 200)
        self.assertGreaterEqual(len(audit_resp.json()["audit_logs"]), 3)

    def test_03_matching_engine_limitation_gate(self) -> None:
        vec = np.ones(512, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        open_cases = [
            {
                "id": 99,
                "case_id": "MP-TEST-99",
                "face_embedding": vec.tolist(),
                "reid_embedding": vec.tolist(),
            }
        ]
        # Appearance-only match when face is unavailable must flag limitation
        cands = matching_engine.match_track_against_cases(
            track_face_embs=[],
            track_reid_embs=[vec],
            open_cases=open_cases,
        )
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].match_tier, "appearance_only_candidate")
        self.assertFalse(cands[0].face_quality_ok)
        self.assertIn("cannot establish identity", cands[0].limitation_note.lower())


if __name__ == "__main__":
    unittest.main()
