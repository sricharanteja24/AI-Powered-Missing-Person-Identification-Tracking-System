"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "AI Missing Person Identification & Tracking System")
    app_version: str = "1.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Security
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production-use-32-byte-random-hex")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
    allowed_origins: list[str] = field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    )

    # Storage paths
    data_dir: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    db_path: Path = Path(os.getenv("DB_PATH", str(BASE_DIR / "data" / "missing_person.db")))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
    evidence_dir: Path = Path(os.getenv("EVIDENCE_DIR", str(BASE_DIR / "data" / "evidence")))
    reports_dir: Path = Path(os.getenv("REPORTS_DIR", str(BASE_DIR / "data" / "reports")))
    log_dir: Path = Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))

    # ML Model Settings
    yolo_model: str = os.getenv("YOLO_MODEL", "yolo11s.pt")
    yolo_conf_threshold: float = float(os.getenv("YOLO_CONF_THRESHOLD", "0.30"))
    insightface_pack: str = os.getenv("INSIGHTFACE_PACK", "buffalo_l")
    reid_model_name: str = os.getenv("REID_MODEL_NAME", "osnet_ain_x1_0")
    use_onnx: bool = os.getenv("USE_ONNX", "true").lower() == "true"
    device: str = os.getenv("DEVICE", "auto")

    # Pipeline & Matching Thresholds (configurable, not calibrated probabilities)
    target_sample_fps: float = float(os.getenv("TARGET_SAMPLE_FPS", "4.0"))
    motion_min_area_ratio: float = float(os.getenv("MOTION_MIN_AREA_RATIO", "0.001"))
    top_k_crops_per_track: int = int(os.getenv("TOP_K_CROPS_PER_TRACK", "8"))
    min_face_size_px: int = int(os.getenv("MIN_FACE_SIZE_PX", "24"))
    min_laplacian_blur: float = float(os.getenv("MIN_LAPLACIAN_BLUR", "15.0"))
    face_match_threshold: float = float(os.getenv("FACE_MATCH_THRESHOLD", "0.35"))
    reid_match_threshold: float = float(os.getenv("REID_MATCH_THRESHOLD", "0.52"))
    min_consistent_frames: int = int(os.getenv("MIN_CONSISTENT_FRAMES", "1"))

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.evidence_dir, self.reports_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "photos").mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "videos").mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()