"""ByteTrack-style multi-object tracker with Top-K sharpest-crop buffering per track."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.config import settings
from app.ml.detector import PersonDetection


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def compute_sharpness(crop: np.ndarray) -> float:
    """Compute Laplacian variance as a fast focus/blur quality metric."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class TrackCropSample:
    frame_number: int
    timestamp_sec: float
    bbox: tuple[int, int, int, int]
    confidence: float
    quality_score: float
    crop: np.ndarray
    annotated_frame: np.ndarray


@dataclass
class TrackState:
    track_id: int
    last_bbox: tuple[int, int, int, int]
    last_seen_frame: int
    hit_count: int = 1
    top_samples: list[TrackCropSample] = field(default_factory=list)

    def add_sample(self, sample: TrackCropSample, top_k: int) -> None:
        self.top_samples.append(sample)
        self.top_samples.sort(key=lambda s: s.quality_score, reverse=True)
        if len(self.top_samples) > top_k:
            self.top_samples = self.top_samples[:top_k]


class ByteTrackBuffer:
    """Two-stage (high/low confidence) ByteTrack-style IoU association with Top-K crop buffer."""

    def __init__(self, max_age_frames: int = 30, iou_threshold: float = 0.25) -> None:
        self.max_age_frames = max_age_frames
        self.iou_threshold = iou_threshold
        self._next_track_id = 1
        self.tracks: dict[int, TrackState] = {}

    def update(
        self,
        detections: list[PersonDetection],
        frame: np.ndarray,
        frame_number: int,
        timestamp_sec: float,
    ) -> list[int]:
        """Associate detections using ByteTrack high-conf then low-conf stages."""
        high_dets = [d for d in detections if d.confidence >= 0.5]
        low_dets = [d for d in detections if d.confidence < 0.5]

        active_ids = [
            tid for tid, t in self.tracks.items() if (frame_number - t.last_seen_frame) <= self.max_age_frames
        ]
        matched_track_ids: set[int] = set()
        assigned_ids: list[int] = []

        # Stage 1: Associate high-confidence detections
        unmatched_high: list[PersonDetection] = []
        for det in high_dets:
            best_tid, best_iou = None, self.iou_threshold
            for tid in active_ids:
                if tid in matched_track_ids:
                    continue
                score = _iou(det.bbox, self.tracks[tid].last_bbox)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid
            if best_tid is not None:
                matched_track_ids.add(best_tid)
                self._record_match(best_tid, det, frame, frame_number, timestamp_sec)
                assigned_ids.append(best_tid)
            else:
                unmatched_high.append(det)

        # Stage 2: Associate remaining tracks with low-confidence detections (occlusion recovery)
        for det in low_dets:
            best_tid, best_iou = None, self.iou_threshold
            for tid in active_ids:
                if tid in matched_track_ids:
                    continue
                score = _iou(det.bbox, self.tracks[tid].last_bbox)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid
            if best_tid is not None:
                matched_track_ids.add(best_tid)
                self._record_match(best_tid, det, frame, frame_number, timestamp_sec)
                assigned_ids.append(best_tid)

        # Create new tracks for unmatched high-confidence detections
        for det in unmatched_high:
            tid = self._next_track_id
            self._next_track_id += 1
            self.tracks[tid] = TrackState(track_id=tid, last_bbox=det.bbox, last_seen_frame=frame_number)
            self._record_match(tid, det, frame, frame_number, timestamp_sec, increment_hits=False)
            assigned_ids.append(tid)

        return assigned_ids

    def _record_match(
        self,
        tid: int,
        det: PersonDetection,
        frame: np.ndarray,
        frame_number: int,
        timestamp_sec: float,
        increment_hits: bool = True,
    ) -> None:
        track = self.tracks[tid]
        track.last_bbox = det.bbox
        track.last_seen_frame = frame_number
        if increment_hits:
            track.hit_count += 1

        x1, y1, x2, y2 = det.bbox
        crop = frame[y1:y2, x1:x2].copy()
        area = max(1, (x2 - x1) * (y2 - y1))
        sharpness = compute_sharpness(crop)
        quality_score = float(np.sqrt(area) * max(1.0, sharpness) * det.confidence)

        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 185, 129), 2)
        cv2.putText(
            annotated,
            f"ID:{tid} ({det.confidence:.2f})",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (16, 185, 129),
            2,
        )

        track.add_sample(
            TrackCropSample(
                frame_number=frame_number,
                timestamp_sec=timestamp_sec,
                bbox=det.bbox,
                confidence=det.confidence,
                quality_score=quality_score,
                crop=crop,
                annotated_frame=annotated,
            ),
            top_k=settings.top_k_crops_per_track,
        )
