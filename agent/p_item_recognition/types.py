from __future__ import annotations

from dataclasses import dataclass

UNKNOWN_P_ITEM_ID = "UNKNOWN"


@dataclass(frozen=True)
class PItemHit:
    p_item_id: str
    class_name: str
    similarity: float
    source_index: int
    visual_group_id: str
    candidate_p_item_ids: tuple[str, ...]
    candidate_upgrade_states: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedPItemIdentity:
    p_item_id: str
    visual_group_id: str
    plan: str
    upgraded: bool | None
    plan_resolved: bool
    upgrade_resolved: bool


@dataclass(frozen=True)
class PItemPrediction:
    box: tuple[int, int, int, int]
    frame_id: str
    p_item_id: str
    predicted_p_item_id: str
    visual_group_id: str
    confidence: float
    margin: float
    accepted: bool
    reason: str
    upgraded: bool | None
    plan: str
    model_sha256: str
    gallery_sha256: str
    classes_sha256: str
    recognizer: str
    top_k_p_item_ids: tuple[tuple[str, ...], ...]
    top_k_scores: tuple[float, ...]
