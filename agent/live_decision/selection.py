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
    """Observed absence of SELECT or its unique physical slot."""

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


def classify_live_runtime_frame(
    *,
    health_hit: bool,
    playable_hit: bool,
    secondary_hit: bool,
    popup_hit: bool,
) -> str:
    """Classify one frame without treating missing or contradictory evidence as exit."""

    if secondary_hit or popup_hit or health_hit != playable_hit:
        return "STOP"
    if health_hit and playable_hit:
        return "PLAYABLE"
    return "UNKNOWN"


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
    hif_layout: bool = False,
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
    if hif_layout and (frame_shape[:2] != (1280, 720)
                       or any(card.box[2:] != (190, 250) for card in cards)):
        raise LiveSelectionError("HIF title layout requires normalized physical card footprints")

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
            if hif_layout:
                box = (x, y + height - 36, width, 36)
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
            if hif_layout:
                # HIF's title fills the physical card width. An additional
                # inset here plus the marker reader's own border margin can
                # erase a valid suffix near the right edge of a long title.
                full_right = min(x + width, frame_width)
                visible_right = (min(full_right, cards[index + 1].box[0] - 2)
                                 if index + 1 < len(cards) else full_right)
                box = (x, y + height - 36, visible_right - x, 36)
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
    *, single_frame: bool = False,
) -> LiveSelectionConsensus:
    """Distinguish stable unselected, stable selected, and ambiguous evidence."""

    observations = tuple(tuple(sorted(set(slots))) for slots in selected_slots)
    if len(observations) != (1 if single_frame else 3):
        raise LiveSelectionError("Live SELECT observation count does not match the requested read")
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
    image: Any,
    view: LiveTitleView,
    *,
    frame_id: str | None = None,
) -> LiveUpgradeConsensus:
    """Resolve one current title frame; covered or unreadable suffixes stay unknown."""

    consensus = resolve_live_upgrade_consensus(
        image,
        view.slot,
        box=view.box,
        expanded=view.selected,
        frame_id=frame_id,
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


def live_selection_geometry_evidence(
    candidates: tuple[Any, ...],
) -> dict[str, Any]:
    """Describe SELECT expansion, including the observed two-card HIF row."""

    hif_pair = len(candidates) in (1,2) and all(
        abs(card.box[0] - anchor) <= 8 and 830 <= card.box[1] <= 890
        and 180 <= card.box[2] <= 205 and 240 <= card.box[3] <= 310
        for card, anchor in zip(candidates, (265,) if len(candidates)==1 else (157, 374)))
    hif_five = len(candidates) == 5 and all(
        abs(card.box[0] - anchor) <= 10 and 830 <= card.box[1] <= 900
        and 170 <= card.box[2] <= 205 and 230 <= card.box[3] <= 310
        for card, anchor in zip(candidates, (18, 142, 265, 388, 512)))
    if (len(candidates) not in {3, 4} and not hif_pair and not hif_five) or tuple(
        card.slot for card in candidates
    ) != tuple(range(len(candidates))):
        return {
            "available": False,
            "ambiguous": False,
            "raised_slots": (),
            "selected_slot": None,
        }
    tops = sorted(int(card.box[1]) for card in candidates)
    heights = sorted(int(card.box[3]) for card in candidates)
    if any(height <= 0 for height in heights):
        return {
            "available": True,
            "ambiguous": True,
            "raised_slots": (),
            "selected_slot": None,
        }
    row_top = 884 if len(candidates)==1 else tops[len(tops) // 2]
    typical_height = heights[len(heights) // 2]
    minimum_raise = max(18.0, typical_height * 0.07)
    raised_slots = tuple(
        card.slot
        for card in candidates
        if row_top - int(card.box[1]) >= minimum_raise
    )
    ambiguous = len(raised_slots) > 1 or (
        not raised_slots and tops[-1] - tops[0] >= minimum_raise
    )
    return {
        "available": True,
        "ambiguous": ambiguous,
        "raised_slots": raised_slots,
        "selected_slot": (
            raised_slots[0] if len(raised_slots) == 1 and not ambiguous else None
        ),
    }


def same_live_candidate_geometry(
    expected: tuple[Any, ...],
    observed: tuple[Any, ...],
) -> bool:
    if len(expected) != len(observed):
        return False
    for before, current in zip(expected, observed, strict=True):
        before_x, before_y, before_width, before_height = before.box
        current_x, current_y, current_width, current_height = current.box
        before_center = (before_x + before_width / 2.0, before_y + before_height / 2.0)
        current_center = (
            current_x + current_width / 2.0,
            current_y + current_height / 2.0,
        )
        if abs(before_center[0] - current_center[0]) > max(14.0, before_width * 0.12):
            return False
        if abs(before_center[1] - current_center[1]) > max(18.0, before_height * 0.10):
            return False
        if not 0.70 <= current_width / before_width <= 1.35:
            return False
        if not 0.70 <= current_height / before_height <= 1.35:
            return False
    return True
