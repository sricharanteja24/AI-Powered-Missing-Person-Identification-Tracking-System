"""Face detection, alignment, quality gating, and embedding extraction using InsightFace buffalo_l."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging_config import logger


@dataclass
class FaceExtractionResult:
    embedding: np.ndarray | None
    quality_ok: bool
    det_score: float
    reason: str


class FaceEngine:
    """Uses InsightFace (RetinaFace + ArcFace buffalo_l) with quality gating."""

    def __init__(self) -> None:
        self.backend = "uninitialized"
        self._app = None
        self._haar = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if settings.device != "cpu"
                else ["CPUExecutionProvider"]
            )
            self._app = FaceAnalysis(name=settings.insightface_pack, providers=providers)
            self._app.prepare(ctx_id=0 if settings.device != "cpu" else -1, det_size=(640, 640))
            self.backend = f"insightface:{settings.insightface_pack}"
            logger.info("Loaded face engine: %s", self.backend)
        except Exception as exc:
            logger.warning("InsightFace unavailable (%s); using OpenCV Haar quality-gated fallback.", exc)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            self.backend = "opencv_haar_fallback"

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 1e-8 else vec.astype(np.float32)

    def extract_reference_embedding(self, image: np.ndarray) -> FaceExtractionResult:
        return self.extract_face_embedding(image)

    def extract_face_embedding(self, crop: np.ndarray) -> FaceExtractionResult:
        if crop is None or crop.size == 0:
            return FaceExtractionResult(None, False, 0.0, "Empty image crop")

        if self._app is not None:
            faces = self._app.get(crop)
            if not faces:
                return FaceExtractionResult(None, False, 0.0, "No face detected")
            # Select largest / highest-confidence face
            best = max(faces, key=lambda f: float(f.det_score) * max(1.0, (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
            fw = float(best.bbox[2] - best.bbox[0])
            fh = float(best.bbox[3] - best.bbox[1])
            if min(fw, fh) < settings.min_face_size_px:
                return FaceExtractionResult(None, False, float(best.det_score), f"Face too small ({int(fw)}x{int(fh)}px)")
            x1, y1, x2, y2 = [max(0, int(v)) for v in best.bbox[:4]]
            face_roi = crop[y1:y2, x1:x2]
            if face_roi.size > 0:
                blur_val = float(cv2.Laplacian(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if blur_val < settings.min_laplacian_blur:
                    return FaceExtractionResult(None, False, float(best.det_score), f"Face too blurry (Laplacian={blur_val:.1f})")
            emb = self._l2_normalize(np.asarray(best.embedding, dtype=np.float32))
            return FaceExtractionResult(emb, True, float(best.det_score), "OK")

        # Lightweight fallback when InsightFace is not installed (e.g. sandbox tests)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(settings.min_face_size_px, settings.min_face_size_px))
        if len(faces) == 0:
            return FaceExtractionResult(None, False, 0.0, "No frontal face detected")
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        roi = cv2.resize(gray[y : y + h, x : x + w], (16, 16)).astype(np.float32).flatten()
        # Pad/project deterministically to 512-d normalized vector for interface compatibility
        emb = np.zeros(512, dtype=np.float32)
        emb[:256] = roi / 255.0
        emb[256:] = (roi - np.mean(roi)) / (np.std(roi) + 1e-6)
        return FaceExtractionResult(self._l2_normalize(emb), True, 0.75, "OK (fallback)")


face_engine = FaceEngine()
