from __future__ import annotations

from typing import Any, Iterable
from dataclasses import asdict, dataclass

from .upgrade import LiveUpgradeConsensus, resolve_live_upgrade_consensus
from .partial_card import normalise_live_candidates


class LiveSelectionError(ValueError):
    """Raised before another click when the Live SELECT state is not unique."""

    def __init__(self, message: str, *, events: Iterable[dict[str, Any]] = ()) -> None:
        super().__init__(message)
        self.events = tuple(events)


@dataclass(frozen=True)
class LiveSelectionConsensus:
    """Three-frame proof of either no SELECT or one stable selected slot."""

    accepted: bool
    selected_slot: int | None
    reason: str
    observations: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveTitleView:
    """One directly visible or SELECT-expanded Live card-title region."""

    slot: int
    box: tuple[int, int, int, int]
    suffix_complete: bool
    selected: bool


def _validate_box(
    box: tuple[int, int, int, int], frame_width: int, frame_height: int
) -> None:
    x, y, width, height = box
    if (
        x < 0
        or y < 0
        or width < 32
        or height < 20
        or x + width > frame_width
        or y + height > frame_height
    ):
        raise LiveSelectionError("Live title ROI is outside the frame")


def build_live_title_views(
    frame_shape: tuple[int, ...],
    candidates: Iterable[Any],
    *,
    selected_slot: int | None = None,
) -> tuple[LiveTitleView, ...]:
    """Build title ROIs for one-to-five cards or one SELECT-expanded card.

    An overlapped title can prove enhancement when ``+`` is visible, but the
    absence of ``+`` is accepted as normal only when ``suffix_complete`` is
    true. A selected card expands to roughly 116% of the detector width.
    """

    if len(frame_shape) < 2:
        raise LiveSelectionError("Live frame shape is invalid")
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    cards = normalise_live_candidates(candidates)
    if not 1 <= len(cards) <= 5:
        raise LiveSelectionError("Live title views require one to five cards")
    if selected_slot is not None and selected_slot not in range(len(cards)):
        raise LiveSelectionError("selected Live slot is outside the current hand")

    views = []
    for index, card in enumerate(cards):
        x, y, width, height = card.box
        inset = max(5, round(width * 0.035))
        if selected_slot == card.slot:
            expanded_width = round(width * 1.16)
            box = (
                x + inset,
                y + round(height * 0.74),
                min(expanded_width - 2 * inset, frame_width - x - inset),
                max(60, round(height * 0.256)),
            )
            complete = True
            selected = True
        else:
            full_right = min(x + width - inset, frame_width)
            visible_right = (
                min(full_right, cards[index + 1].box[0] - 4)
                if index + 1 < len(cards)
                else full_right
            )
            box = (
                x + inset,
                y + round(height * 0.82),
                visible_right - (x + inset),
                max(30, round(height * 0.15)),
            )
            complete = len(cards) <= 3 or index + 1 == len(cards)
            selected = False
        _validate_box(box, frame_width, frame_height)
        views.append(
            LiveTitleView(
                slot=card.slot,
                box=box,
                suffix_complete=complete,
                selected=selected,
            )
        )
    return tuple(views)


def live_select_label_box(
    frame_shape: tuple[int, ...], candidate: Any
) -> tuple[int, int, int, int]:
    """Return the target-specific ROI where a selected card renders SELECT."""

    if len(frame_shape) < 2:
        raise LiveSelectionError("Live frame shape is invalid")
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    x, y, width, height = candidate.box
    label_x = x + round(width * 0.20)
    box = (
        label_x,
        y + height - 36,
        min(round(width * 0.58), frame_width - label_x),
        min(58, frame_height - (y + height - 36)),
    )
    _validate_box(box, frame_width, frame_height)
    return box


def resolve_live_selected_slot_consensus(
    selected_slots: Iterable[Iterable[int]],
) -> LiveSelectionConsensus:
    """Distinguish stable unselected, stable selected, and ambiguous evidence."""

    observations = tuple(tuple(sorted(set(slots))) for slots in selected_slots)
    if len(observations) != 3:
        raise LiveSelectionError("Live SELECT consensus requires exactly three frames")
    if any(len(slots) > 1 for slots in observations):
        return LiveSelectionConsensus(
            accepted=False,
            selected_slot=None,
            reason="multiple_select_labels",
            observations=observations,
        )
    frame_slots = tuple(slots[0] if slots else None for slots in observations)
    if len(set(frame_slots)) != 1:
        return LiveSelectionConsensus(
            accepted=False,
            selected_slot=None,
            reason="select_frames_disagree",
            observations=observations,
        )
    selected_slot = frame_slots[0]
    return LiveSelectionConsensus(
        accepted=True,
        selected_slot=selected_slot,
        reason=("selected_unanimous" if selected_slot is not None else "unselected_unanimous"),
        observations=observations,
    )


def remaining_live_card_clicks(
    current_selected_slot: int | None,
    target_slot: int,
    *,
    hand_size: int,
) -> int:
    """Return one click for the selected card, otherwise select then play."""

    if hand_size < 1 or hand_size > 5 or target_slot not in range(hand_size):
        raise LiveSelectionError("Live play target is outside the current hand")
    if current_selected_slot is not None and current_selected_slot not in range(hand_size):
        raise LiveSelectionError("current Live SELECT state is outside the hand")
    return 1 if current_selected_slot == target_slot else 2


def resolve_live_title_consensus(
    images: Any, view: LiveTitleView
) -> LiveUpgradeConsensus:
    """Resolve a title while keeping covered no-marker results fail-closed."""

    consensus = resolve_live_upgrade_consensus(
        images,
        view.slot,
        box=view.box,
        expanded=view.selected,
    )
    if (consensus.upgraded is True or view.suffix_complete) and view.selected:
        return LiveUpgradeConsensus(
            accepted=consensus.accepted,
            upgraded=consensus.upgraded,
            plus_votes=consensus.plus_votes,
            normal_votes=consensus.normal_votes,
            reason=f"selected_{consensus.reason}",
            observations=consensus.observations,
        )
    if consensus.upgraded is True or view.suffix_complete:
        return consensus
    return LiveUpgradeConsensus(
        accepted=False,
        upgraded=None,
        plus_votes=consensus.plus_votes,
        normal_votes=consensus.normal_votes,
        reason="live_upgrade_suffix_occluded",
        observations=consensus.observations,
    )
