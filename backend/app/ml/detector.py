"""Person detection module using Ultralytics YOLO11s with graceful OpenCV HOG fallback."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging_config import logger


@dataclass
class PersonDetection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class PersonDetector:
    """Wraps Ultralytics YOLO11s (class 0 = person) with optional ONNX/GPU acceleration."""

    def __init__(self) -> None:
        self.backend = "uninitialized"
        self._model = None
        self._hog = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(settings.yolo_model)
            self.backend = f"ultralytics:{settings.yolo_model}"
            logger.info("Loaded person detector: %s", self.backend)
        except Exception as exc:
            logger.warning("Ultralytics YOLO11s unavailable (%s); using OpenCV HOG fallback.", exc)
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.backend = "opencv_hog_fallback"

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        h, w = frame.shape[:2]
        detections: list[PersonDetection] = []

        if self._model is not None:
            results = self._model.predict(
                source=frame,
                classes=[0],  # COCO class 0: person
                conf=settings.yolo_conf_threshold,
                verbose=False,
            )
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0].item())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if (x2 - x1) >= 24 and (y2 - y1) >= 48:
                        detections.append(PersonDetection(bbox=(x1, y1, x2, y2), confidence=conf))
            return detections

        # Fallback: OpenCV HOG people detector
        if self._hog is not None:
            rects, weights = self._hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
            for (x, y, rw, rh), weight in zip(rects, weights):
                conf = float(min(0.99, max(0.35, float(weight))))
                if conf >= settings.yolo_conf_threshold:
                    detections.append(
                        PersonDetection(bbox=(max(0, x), max(0, y), min(w, x + rw), min(h, y + rh)), confidence=conf)
                    )
        return detections


person_detector = PersonDetector()
