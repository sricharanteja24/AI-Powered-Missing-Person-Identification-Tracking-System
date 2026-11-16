"""End-to-end CCTV video processing pipeline orchestrating detection, tracking, and matching."""
from __future__ import annotations

import uuid
from pathlib import Path

import cv2

from app.core.config import settings
from app.core.logging_config import logger
from app.db.repositories import (
    create_match_alert,
    get_open_cases_with_embeddings,
    get_video_by_id,
    update_video_progress,
)
from app.ml.detector import person_detector
from app.ml.face_engine import face_engine
from app.ml.frame_sampler import iter_motion_gated_frames
from app.ml.matcher import matching_engine
from app.ml.reid_engine import reid_engine
from app.ml.tracker import ByteTrackBuffer


def process_video_pipeline(video_id: int) -> dict:
    """Process a queued CCTV video and generate candidate match alerts."""
    video = get_video_by_id(video_id)
    if not video:
        raise ValueError(f"Video {video_id} not found")

    logger.info(
        "Starting video pipeline video_id=%s detector=%s face=%s reid=%s",
        video_id,
        person_detector.backend,
        face_engine.backend,
        reid_engine.backend,
    )

    update_video_progress(video_id, status_val="processing", progress_pct=1.0)
    metadata, frame_iter = iter_motion_gated_frames(video["file_path"])
    total_frames = max(1, int(metadata["total_frames"] or 1))
    tracker = ByteTrackBuffer()

    processed_samples = 0
    for sampled in frame_iter:
        detections = person_detector.detect(sampled.frame)
        if detections:
            tracker.update(
                detections=detections,
                frame=sampled.frame,
                frame_number=sampled.frame_number,
                timestamp_sec=sampled.timestamp_sec,
            )
        processed_samples += 1
        pct = min(85.0, (sampled.frame_number / float(total_frames)) * 85.0)
        if processed_samples % 5 == 0:
            update_video_progress(
                video_id=video_id,
                status_val="processing",
                progress_pct=pct,
                processed_frames=sampled.frame_number,
                total_frames=total_frames,
                fps=metadata["fps"],
                duration_sec=metadata["duration_sec"],
            )

    # Track-first Top-K embedding extraction & matching
    open_cases = get_open_cases_with_embeddings()
    alerts_created = 0

    for track_id, track in tracker.tracks.items():
        if track.hit_count < settings.min_consistent_frames and len(tracker.tracks) > 0:
            # Allow single-hit tracks only if video is very short (< 2 samples)
            if processed_samples >= 3:
                continue

        face_embs = []
        reid_embs = []
        best_sample = track.top_samples[0] if track.top_samples else None
        if best_sample is None:
            continue

        for sample in track.top_samples:
            f_res = face_engine.extract_face_embedding(sample.crop)
            if f_res.quality_ok and f_res.embedding is not None:
                face_embs.append(f_res.embedding)
            r_emb = reid_engine.extract_embedding(sample.crop)
            if r_emb is not None:
                reid_embs.append(r_emb)

        candidates = matching_engine.match_track_against_cases(
            track_face_embs=face_embs,
            track_reid_embs=reid_embs,
            open_cases=open_cases,
        )
        for cand in candidates:
            ev_filename = f"alert_v{video_id}_c{cand.case_db_id}_t{track_id}_{uuid.uuid4().hex[:6]}.jpg"
            ev_path = settings.evidence_dir / ev_filename
            cv2.imwrite(str(ev_path), best_sample.annotated_frame)

            create_match_alert(
                case_db_id=cand.case_db_id,
                video_id=video_id,
                track_id=track_id,
                camera_id=video["camera_id"],
                timestamp_sec=best_sample.timestamp_sec,
                frame_number=best_sample.frame_number,
                face_similarity=cand.face_similarity,
                reid_similarity=cand.reid_similarity,
                detection_confidence=best_sample.confidence,
                face_quality_ok=cand.face_quality_ok,
                match_tier=cand.match_tier,
                limitation_note=cand.limitation_note,
                evidence_image_path=str(ev_path),
            )
            alerts_created += 1

    update_video_progress(
        video_id=video_id,
        status_val="completed",
        progress_pct=100.0,
        processed_frames=total_frames,
        total_frames=total_frames,
        fps=metadata["fps"],
        duration_sec=metadata["duration_sec"],
    )
    logger.info("Completed video_id=%s tracks=%d alerts=%d", video_id, len(tracker.tracks), alerts_created)
    return {"video_id": video_id, "tracks": len(tracker.tracks), "alerts_created": alerts_created}
