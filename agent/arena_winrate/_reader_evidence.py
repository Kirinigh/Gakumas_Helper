"""Shared immutable OCR evidence types; no controller dependency."""

from typing import Any, NamedTuple

_PItemPanelRow = tuple[
    str,
    tuple[float, float, float, float],
    tuple[tuple[str, tuple[float, float, float, float]], ...],
]


_NormalizedPItemRow = tuple[
    str,
    str,
    tuple[float, float, float, float],
    tuple[tuple[str, str, tuple[float, float, float, float]], ...],
]


class _PItemOcrObservation(NamedTuple):
    """One OCR call, with separate title and size-excluded source evidence."""

    full_text: str
    panel_text: str
    anchors_visible: bool
    panel_rows: tuple[_PItemPanelRow, ...]
    background_rows: tuple[_PItemPanelRow, ...] = ()
    all_items: tuple[Any, ...] = ()


class _DetailIdentityObservation(NamedTuple):
    """One exact-title semantic observation from the active click transaction."""

    transaction_token: int
    transaction_started: float
    source_card_box: tuple[int, int, int, int]
    interaction_box: tuple[int, int, int, int]
    contact_released_at: float
    capture_started_at: float
    detail_image: Any
    source_guard_frames: Any
    restoration_signatures: Any
    identity_frames: Any
    title: str
    card_id: int
    customizations: tuple[tuple[str, int], ...]
    evidence_mode: str


class _DetailIdentityProof(NamedTuple):
    """Exact-title observations bound to one physical click transaction."""

    transaction_token: int
    transaction_started: float
    source_card_box: tuple[int, int, int, int]
    interaction_box: tuple[int, int, int, int]
    contact_released_at: float
    capture_started_at: tuple[float, ...]
    detail_images: tuple[Any, ...]
    source_guard_frames: Any
    restoration_signatures: Any
    identity_frames: Any
    title: str
    card_id: int
    customizations: tuple[tuple[str, int], ...]
    resolution_source: str
    evidence_mode: str
    detail_confirmation_reads: int


class _TitleBoundEffectRoiText(str):
    """Merged ROI text carrying how many non-empty OCR passes produced it."""

    observation_count: int

    def __new__(cls, value: str, observation_count: int) -> "_TitleBoundEffectRoiText":
        instance = str.__new__(cls, value)
        instance.observation_count = observation_count
        return instance


def _effect_roi_observation_count(value: str) -> int:
    count = getattr(value, "observation_count", 1)
    return count if type(count) is int and count > 0 else 1
