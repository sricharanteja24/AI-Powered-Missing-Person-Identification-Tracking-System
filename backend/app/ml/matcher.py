"""Vectorized cosine similarity matching engine with explicit face vs appearance-only gating."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.config import settings


@dataclass
class MatchCandidate:
    case_db_id: int
    case_code: str
    face_similarity: float | None
    reid_similarity: float | None
    face_quality_ok: bool
    match_tier: str
    limitation_note: str


class MatchingEngine:
    """Compares track embeddings against open missing-person cases without claiming identity."""

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    def match_track_against_cases(
        self,
        track_face_embs: list[np.ndarray],
        track_reid_embs: list[np.ndarray],
        open_cases: list[dict[str, Any]],
    ) -> list[MatchCandidate]:
        if not open_cases:
            return []

        # Pool track embeddings via L2-normalized mean over Top-K sharp frames
        pooled_face: np.ndarray | None = None
        if track_face_embs:
            mean_f = np.mean(np.stack(track_face_embs, axis=0), axis=0)
            norm = float(np.linalg.norm(mean_f))
            if norm > 1e-8:
                pooled_face = (mean_f / norm).astype(np.float32)

        pooled_reid: np.ndarray | None = None
        if track_reid_embs:
            mean_r = np.mean(np.stack(track_reid_embs, axis=0), axis=0)
            norm = float(np.linalg.norm(mean_r))
            if norm > 1e-8:
                pooled_reid = (mean_r / norm).astype(np.float32)

        candidates: list[MatchCandidate] = []
        for case in open_cases:
            face_sim: float | None = None
            reid_sim: float | None = None

            if pooled_face is not None and case.get("face_embedding"):
                ref_face = np.asarray(case["face_embedding"], dtype=np.float32)
                if ref_face.shape == pooled_face.shape:
                    face_sim = self.cosine_similarity(pooled_face, ref_face)

            if pooled_reid is not None and case.get("reid_embedding"):
                ref_reid = np.asarray(case["reid_embedding"], dtype=np.float32)
                if ref_reid.shape == pooled_reid.shape:
                    reid_sim = self.cosine_similarity(pooled_reid, ref_reid)

            face_hit = face_sim is not None and face_sim >= settings.face_match_threshold
            reid_hit = reid_sim is not None and reid_sim >= settings.reid_match_threshold

            if face_hit and reid_hit:
                candidates.append(
                    MatchCandidate(
                        case_db_id=case["id"],
                        case_code=case["case_id"],
                        face_similarity=face_sim,
                        reid_similarity=reid_sim,
                        face_quality_ok=True,
                        match_tier="face_and_appearance_candidate",
                        limitation_note=(
                            "Candidate lead combining facial and appearance similarity. "
                            "Scores are uncalibrated heuristics and require human verification."
                        ),
                    )
                )
            elif face_hit:
                candidates.append(
                    MatchCandidate(
                        case_db_id=case["id"],
                        case_code=case["case_id"],
                        face_similarity=face_sim,
                        reid_similarity=reid_sim,
                        face_quality_ok=True,
                        match_tier="face_candidate",
                        limitation_note=(
                            "Candidate lead based on facial similarity; clothing/appearance may have changed. "
                            "Requires human verification."
                        ),
                    )
                )
            elif reid_hit and pooled_face is None:
                candidates.append(
                    MatchCandidate(
                        case_db_id=case["id"],
                        case_code=case["case_id"],
                        face_similarity=None,
                        reid_similarity=reid_sim,
                        face_quality_ok=False,
                        match_tier="appearance_only_candidate",
                        limitation_note=(
                            "Facial comparison unavailable (low resolution, occlusion, or pose). "
                            "Appearance/clothing similarity alone cannot establish identity; human verification required."
                        ),
                    )
                )

        return candidates


matching_engine = MatchingEngine()
