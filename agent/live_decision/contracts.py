from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from dataclasses import field, asdict, dataclass

PROTOCOL_VERSION = "1.0"
UNKNOWN = "UNKNOWN"
REQUIRED_STATE_FIELDS = frozenset(
    {
        "turn",
        "turns_remaining",
        "stamina",
        "max_stamina",
        "score",
        "score_multiplier",
        "action_order",
        "items",
        "field_effects",
        "statuses",
        "hand",
    }
)


class ContractError(ValueError):
    """Raised when a versioned Live contract is incomplete or contradictory."""


@dataclass(frozen=True)
class LiveStatus:
    status_id: str
    stacks: int = 0
    remaining_turns: int | None = None


@dataclass(frozen=True)
class LiveCard:
    slot: int
    card_id: str
    cost: int
    playable: bool
    identity_confidence: float
    identity_accepted: bool
    box: tuple[int, int, int, int] | None = None
    features: dict[str, float] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)
    upgraded: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveState:
    snapshot_id: str
    turn: int
    turns_remaining: int
    stamina: int
    max_stamina: int
    score: int
    score_multiplier: float
    action_order: tuple[str, ...]
    items: tuple[str, ...]
    field_effects: tuple[str, ...]
    statuses: tuple[LiveStatus, ...]
    hand: tuple[LiveCard, ...]
    capture_age_ms: float = 0.0
    unknown_fields: tuple[str, ...] = field(default_factory=tuple)
    contradictory_fields: tuple[str, ...] = field(default_factory=tuple)
    protocol_version: str = PROTOCOL_VERSION

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ContractError(f"unsupported LiveState protocol: {self.protocol_version}")
        if not self.snapshot_id:
            raise ContractError("snapshot_id is required")
        if self.turn < 1 or self.turns_remaining < 0:
            raise ContractError("turn values must be non-negative and turn starts at one")
        if self.stamina < 0 or self.max_stamina < 0 or self.stamina > self.max_stamina:
            raise ContractError("stamina is outside its valid range")
        if self.score < 0 or self.score_multiplier <= 0:
            raise ContractError("score values are outside their valid range")
        if self.capture_age_ms < 0:
            raise ContractError("capture_age_ms cannot be negative")
        slots = [card.slot for card in self.hand]
        if len(slots) != len(set(slots)):
            raise ContractError("duplicate hand slot")
        if any(card.cost < 0 for card in self.hand):
            raise ContractError("card cost cannot be negative")
        if any(not 0.0 <= card.identity_confidence <= 1.0 for card in self.hand):
            raise ContractError("card identity confidence must be between zero and one")
        unknown = set(self.unknown_fields)
        if unknown - REQUIRED_STATE_FIELDS:
            raise ContractError(f"unknown_fields contains unsupported names: {sorted(unknown - REQUIRED_STATE_FIELDS)}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LiveState":
        statuses = tuple(LiveStatus(**item) for item in raw.get("statuses", []))
        hand = tuple(
            LiveCard(
                slot=int(item["slot"]),
                card_id=str(item["card_id"]),
                cost=int(item["cost"]),
                playable=bool(item["playable"]),
                identity_confidence=float(item["identity_confidence"]),
                identity_accepted=bool(item["identity_accepted"]),
                box=tuple(item["box"]) if item.get("box") is not None else None,
                features={str(key): float(value) for key, value in item.get("features", {}).items()},
                tags=tuple(str(value) for value in item.get("tags", [])),
                upgraded=(bool(item["upgraded"]) if item.get("upgraded") is not None else None),
            )
            for item in raw.get("hand", [])
        )
        state = cls(
            snapshot_id=str(raw["snapshot_id"]),
            turn=int(raw["turn"]),
            turns_remaining=int(raw["turns_remaining"]),
            stamina=int(raw["stamina"]),
            max_stamina=int(raw["max_stamina"]),
            score=int(raw.get("score", 0)),
            score_multiplier=float(raw["score_multiplier"]),
            action_order=tuple(str(value) for value in raw.get("action_order", [])),
            items=tuple(str(value) for value in raw.get("items", [])),
            field_effects=tuple(str(value) for value in raw.get("field_effects", [])),
            statuses=statuses,
            hand=hand,
            capture_age_ms=float(raw.get("capture_age_ms", 0.0)),
            unknown_fields=tuple(str(value) for value in raw.get("unknown_fields", [])),
            contradictory_fields=tuple(str(value) for value in raw.get("contradictory_fields", [])),
            protocol_version=str(raw.get("protocol_version", PROTOCOL_VERSION)),
        )
        state.validate()
        return state

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveAction:
    kind: str
    card_slot: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionProposal:
    status: str
    reason: str
    action: LiveAction | None
    confidence: float
    provider: str
    provider_version: str
    elapsed_ms: float
    scores: dict[str, float] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafeDecision:
    status: str
    reason: str
    action: LiveAction | None
    proposal: DecisionProposal | None
    executed: bool = False
    protocol_version: str = PROTOCOL_VERSION

    @property
    def has_action_intent(self) -> bool:
        return self.status == "ALLOW" and self.action is not None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_action_intent"] = self.has_action_intent
        return data


@runtime_checkable
class LiveStateProvider(Protocol):
    provider_name: str
    provider_version: str

    def build_state(self, observation: dict[str, Any]) -> LiveState: ...


@runtime_checkable
class LiveDecisionProvider(Protocol):
    provider_name: str
    provider_version: str

    def decide(self, state: LiveState) -> DecisionProposal: ...


@runtime_checkable
class LiveSafetyFilter(Protocol):
    def filter(self, state: LiveState, proposal: DecisionProposal | None) -> SafeDecision: ...
