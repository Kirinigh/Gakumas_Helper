from __future__ import annotations

from typing import Any
from dataclasses import field, asdict, dataclass

UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CandidateBox:
    """A selectable Maa detector result with an audit-only raw label."""

    slot: int
    box: tuple[int, int, int, int]
    detector_label: str = "cards"
    detector_score: float = 1.0
    visible: bool = True
    clickable: bool = True

    @property
    def center(self) -> tuple[int, int]:
        x, y, width, height = self.box
        return x + width // 2, y + height // 2


@dataclass(frozen=True)
class CardPrediction:
    """One slot classification, including rejected top-class information."""

    slot: int
    box: tuple[int, int, int, int]
    frame_id: str
    card_id: str
    predicted_card_id: str
    class_name: str
    confidence: float
    margin: float
    raw_score: float
    accepted: bool
    reason: str
    model_sha256: str
    classes_sha256: str
    detector_label: str = "cards"
    detector_score: float = 1.0
    visible: bool = True
    clickable: bool = True
    source_frames: tuple[str, ...] = field(default_factory=tuple)
    recognizer: str = "classifier"
    gallery_sha256: str = ""
    top_k_card_ids: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    top_k_scores: tuple[float, ...] = field(default_factory=tuple)
    view_mode: str = "full"
    visible_ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionContext:
    scene: str
    plan: str
    p_idol: str
    stage: str
    deck_card_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionDecision:
    status: str
    reason: str
    chosen_slot: int | None
    click_point: tuple[int, int] | None
    candidates: tuple[dict[str, Any], ...]
    applied_overrides: tuple[str, ...]
    policy_version: int

    @property
    def has_click_intent(self) -> bool:
        return self.status == "SELECT" and self.click_point is not None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_click_intent"] = self.has_click_intent
        return data
