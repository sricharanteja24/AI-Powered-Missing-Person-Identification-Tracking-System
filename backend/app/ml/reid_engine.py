"""Cross-camera Person Re-Identification engine using OSNet-AIN (osnet_ain_x1_0)."""
from __future__ import annotations

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging_config import logger


class ReIDEngine:
    """Extracts 512-d L2-normalized appearance embeddings using OSNet-AIN."""

    def __init__(self) -> None:
        self.backend = "uninitialized"
        self._extractor = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from torchreid.utils import FeatureExtractor  # type: ignore

            device = "cuda" if settings.device != "cpu" else "cpu"
            self._extractor = FeatureExtractor(
                model_name=settings.reid_model_name,
                device=device,
            )
            self.backend = f"torchreid:{settings.reid_model_name}"
            logger.info("Loaded Re-ID engine: %s", self.backend)
        except Exception as exc:
            logger.warning("Torchreid OSNet-AIN unavailable (%s); using HSV spatial histogram fallback.", exc)
            self.backend = "opencv_hsv_spatial_fallback"

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 1e-8 else vec.astype(np.float32)

    def extract_embedding(self, person_crop: np.ndarray) -> np.ndarray | None:
        if person_crop is None or person_crop.size == 0:
            return None

        if self._extractor is not None:
            rgb = cv2.cvtColor(cv2.resize(person_crop, (128, 256)), cv2.COLOR_BGR2RGB)
            features = self._extractor([rgb])
            vec = features[0].detach().cpu().numpy().astype(np.float32)
            return self._l2_normalize(vec)

        # Fallback: 4-strip spatial HSV color histogram (512-d) for upper/lower clothing appearance
        resized = cv2.resize(person_crop, (128, 256))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        strips = np.array_split(hsv, 4, axis=0)
        parts = []
        for strip in strips:
            hist = cv2.calcHist([strip], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
            parts.append(hist)
        vec = np.concatenate(parts).astype(np.float32)
        return self._l2_normalize(vec)


reid_engine = ReIDEngine()
