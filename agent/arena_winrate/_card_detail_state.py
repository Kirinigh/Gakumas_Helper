"""One owner for card-detail transaction state; no UI, clock or persistence."""

from __future__ import annotations

from typing import Any, NamedTuple
from dataclasses import field, dataclass
from collections.abc import Mapping

from ._reader_evidence import _DetailIdentityProof

CardKey = tuple[int, int]
CardBox = tuple[int, int, int, int]
DEFAULT_DETAIL_KIND = "necessary_skill_card_detail_transaction"


class CardDetailOpening(NamedTuple):
    started: float
    token: int
    ocr_started: tuple[int, int]


class ClosedCardDetail(NamedTuple):
    started: float | None
    kind: str
    ocr_started: tuple[int, int] | None


@dataclass
class CardDetailState:
    """Mutable dictionaries are owned here and shared by reference with sessions.

    Opening reserves a new serial before input, but becomes active only after
    the caller confirms the title. Closing invalidates the transaction while
    retaining its text/image for failure reporting. Member reset releases those
    observations separately and never rewinds the serial.
    """

    serial: int = 0
    started: dict[CardKey, float] = field(default_factory=dict)
    tokens: dict[CardKey, int] = field(default_factory=dict)
    contact_counts: dict[CardKey, int] = field(default_factory=dict)
    source_boxes: dict[CardKey, CardBox] = field(default_factory=dict)
    interaction_boxes: dict[CardKey, CardBox] = field(default_factory=dict)
    kinds: dict[CardKey, str] = field(default_factory=dict)
    ocr_started: dict[CardKey, tuple[int, int]] = field(default_factory=dict)
    open_ids: dict[CardKey, int] = field(default_factory=dict)
    rebind_ids: dict[CardKey, int] = field(default_factory=dict)
    texts: dict[CardKey, str] = field(default_factory=dict)
    images: dict[CardKey, Any] = field(default_factory=dict)
    contact_released_at: dict[CardKey, float] = field(default_factory=dict)
    capture_started_at: dict[CardKey, float] = field(default_factory=dict)
    identity_proofs: dict[CardKey, _DetailIdentityProof] = field(default_factory=dict)

    def prepare_open(self, key: CardKey, now: float, counts: Mapping[str, int]) -> CardDetailOpening:
        started = self.started.get(key, now)
        self.serial = int(self.serial) + 1
        self.identity_proofs.pop(key, None)
        baseline = self.ocr_started.get(key, (
            counts.get("detail_capture_broad_ocr_backend_calls", 0),
            counts.get("detail_capture_broad_ocr_cache_hits", 0),
        ))
        return CardDetailOpening(started, self.serial, baseline)

    def record_contact(self, key: CardKey, count: int) -> None:
        self.open_ids.pop(key, None)
        self.contact_counts[key] = count

    def accept_open(
        self, key: CardKey, opening: CardDetailOpening, *, image: Any,
        contact_released_at: float, capture_started_at: float,
        source_box: CardBox, interaction_box: CardBox,
    ) -> None:
        # Text and confirmed-name diagnostics are recorded by the caller first.
        self.images[key] = image
        self.contact_released_at[key] = contact_released_at
        self.capture_started_at[key] = capture_started_at
        self.started[key] = opening.started
        self.tokens[key] = opening.token
        self.source_boxes[key] = tuple(source_box)
        self.interaction_boxes[key] = tuple(interaction_box)
        self.ocr_started[key] = opening.ocr_started
        self.kinds.setdefault(key, DEFAULT_DETAIL_KIND)

    def finish(self, key: CardKey) -> ClosedCardDetail:
        """Invalidate once, after the caller has retained any failed identity."""
        self.open_ids.pop(key, None)
        self.rebind_ids.pop(key, None)
        started = self.started.pop(key, None)
        self.identity_proofs.pop(key, None)
        self.tokens.pop(key, None)
        self.contact_counts.pop(key, None)
        self.source_boxes.pop(key, None)
        self.interaction_boxes.pop(key, None)
        self.contact_released_at.pop(key, None)
        self.capture_started_at.pop(key, None)
        ocr_started = self.ocr_started.pop(key, None)
        kind = self.kinds.pop(key, DEFAULT_DETAIL_KIND)
        return ClosedCardDetail(started, kind, ocr_started)

    def clear_open_identity(self) -> None:
        self.open_ids.clear()
        self.rebind_ids.clear()

    def clear_member_observations(self) -> None:
        self.texts.clear()
        self.images.clear()
        self.contact_released_at.clear()
        self.capture_started_at.clear()

    def clear_transaction_metadata(self) -> None:
        """Run after active transactions have been accounted for by the caller."""
        self.tokens.clear()
        self.contact_counts.clear()
        self.source_boxes.clear()
        self.interaction_boxes.clear()
        self.kinds.clear()
        self.ocr_started.clear()


def card_detail_state(owner: Any) -> CardDetailState:
    """Allow legacy partially constructed test/diagnostic backends to share state."""
    state = getattr(owner, "_card_detail_state", None)
    if state is None:
        state = CardDetailState()
        owner._card_detail_state = state
    return state


def card_detail_field(name: str) -> property:
    """Compatibility alias; never keep a second dictionary on the backend."""
    if name not in CardDetailState.__dataclass_fields__:
        raise ValueError(f"unknown card detail state field: {name}")

    def get(owner: Any) -> Any:
        return getattr(card_detail_state(owner), name)

    def set_(owner: Any, value: Any) -> None:
        setattr(card_detail_state(owner), name, value)

    return property(get, set_)
