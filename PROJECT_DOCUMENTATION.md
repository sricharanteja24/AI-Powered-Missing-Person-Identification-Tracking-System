# Project Documentation — AI-Powered Missing Person Identification & Tracking System

## 1. Abstract
This system assists authorized investigators in reviewing CCTV footage for missing persons using a modular computer vision pipeline: motion-gated frame sampling, Ultralytics YOLO11s person detection, ByteTrack multi-object tracking, Top-K sharpness crop selection, InsightFace `buffalo_l` (RetinaFace + ArcFace) facial embedding extraction, and OSNet-AIN (`osnet_ain_x1_0`) cross-camera appearance re-identification. All outputs are presented strictly as candidate leads requiring human verification.

## 2. Problem Statement & Motivation
Manual review of multi-hour CCTV recordings across distributed cameras is slow and error-prone. Conversely, naive automated face-only systems fail under low resolution, motion blur, or non-frontal camera angles. Combining quality-gated facial recognition with cross-domain person Re-ID and track-level temporal pooling accelerates investigative triage while preserving human oversight.

## 3. System Architecture & Optimizations
- **Motion-Gated Frame Sampling:** Skips static empty frames using frame differencing.
- **YOLO11s Person Detection:** Isolates person bounding boxes (`classes=[0]`).
- **ByteTrack Association + Top-K Buffer:** Tracks individuals across frames and retains only the top 5 sharpest, largest crops per track ID for heavy embedding extraction.
- **Decoupled Matching Engine:** Separates `face_and_appearance_candidate` / `face_candidate` from `appearance_only_candidate` with explicit limitation notes when facial comparison is unavailable.
- **Security & Auditability:** PBKDF2-SHA256 password hashing, HS256 JWT tokens, RBAC, SQLite WAL mode, and full audit logging.

## 4. Datasets & Model Strategy (Pretrained Inference vs. Fine-Tuning)
All core models operate via **pretrained inference** out of the box; no fabricated training claims are made. For formal evaluation or optional fine-tuning:

| Dataset | Contents | Target Module | Usage & License Notes |
| :--- | :--- | :--- | :--- |
| **COCO (Person Class)** | Annotated everyday & surveillance-like person bounding boxes | YOLO11s Detection | Pretrained baseline; CC-BY 4.0 |
| **Market-1501** | 32,668 bounding boxes of 1,501 identities across 6 cameras | OSNet-AIN Re-ID Evaluation | Academic research evaluation (Rank-1, mAP) |
| **MSMT17** | 126,441 bounding boxes across 15 indoor/outdoor cameras | Cross-domain Re-ID Evaluation | Academic research license required |
| **DukeMTMC-ReID** | Multi-camera pedestrian Re-ID benchmark | Re-ID (Restricted) | Discontinued / restricted; prefer Market-1501 or MSMT17 |
| **CelebA / LFW** | Aligned face attributes and verification pairs | Face Verification Calibration | Non-commercial research; use consent-based data for local testing |

## 5. Limitations & Future Scope
- **Limitations:** Extreme low light, heavy occlusion, and clothing changes degrade appearance-only Re-ID; similarity thresholds require calibration on local camera conditions.
- **Future Scope:** Multi-camera RTSP live stream ingestion, Celery + Redis distributed GPU workers, and FAISS HNSW indexing for city-scale case registries.
