"""Motion-gated adaptive frame sampler for CCTV video streams."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from app.core.config import settings


@dataclass
class SampledFrame:
    frame_number: int
    timestamp_sec: float
    frame: np.ndarray
    motion_ratio: float


def iter_motion_gated_frames(
    video_path: str,
    target_fps: float | None = None,
    min_motion_ratio: float | None = None,
) -> tuple[dict, Iterator[SampledFrame]]:
    """Open video, extract metadata, and yield motion-gated frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if native_fps <= 0:
        native_fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / native_fps if total_frames > 0 else 0.0

    sample_fps = target_fps or settings.target_sample_fps
    stride = max(1, int(round(native_fps / max(0.5, sample_fps))))
    motion_thresh = min_motion_ratio if min_motion_ratio is not None else settings.motion_min_area_ratio

    metadata = {
        "fps": native_fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "stride": stride,
    }

    def _generator() -> Iterator[SampledFrame]:
        prev_gray: np.ndarray | None = None
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_small = cv2.resize(gray, (320, 180))
                    gray_blur = cv2.GaussianBlur(gray_small, (5, 5), 0)

                    motion_ratio = 1.0
                    if prev_gray is not None:
                        diff = cv2.absdiff(prev_gray, gray_blur)
                        _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
                        motion_ratio = float(np.count_nonzero(mask)) / float(mask.size)

                    prev_gray = gray_blur
                    # Always yield first frame or frames exceeding motion threshold
                    if frame_idx == 0 or motion_ratio >= motion_thresh:
                        yield SampledFrame(
                            frame_number=frame_idx,
                            timestamp_sec=frame_idx / native_fps,
                            frame=frame,
                            motion_ratio=motion_ratio,
                        )
                frame_idx += 1
        finally:
            cap.release()

    return metadata, _generator()
