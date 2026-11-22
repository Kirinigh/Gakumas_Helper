from __future__ import annotations

import math
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
        "capture_age_ms",
        "action_order",
        "items",
        "field_effects",
        "statuses",
        "hand",
    }
)


class ContractError(ValueError):
    """Raised when a versioned Live contract is incomplete or contradictory."""


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _is_strict_json_value(value: Any) -> bool:
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_strict_json_value(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_is_strict_json_value(item) for item in value)
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return _is_finite_number(value)
    return False


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{field_name} must be an object with string keys")
    return value


def _require_sequence(value: Any, field_name: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{field_name} must be an array")
    return value


def _require_field(raw: dict[str, Any], key: str, scope: str) -> Any:
    if key not in raw:
        raise ContractError(f"{scope}.{key} is required")
    return raw[key]


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{field_name} must be a string")
    return value


def _require_integer(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise ContractError(f"{field_name} must be an integer")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if not _is_finite_number(value):
        raise ContractError(f"{field_name} must be a finite number")
    return float(value)


def _require_boolean(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{field_name} must be a boolean")
    return value


def _text_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    values = _require_sequence(value, field_name)
    return tuple(
        _require_string(item, f"{field_name}[{index}]")
        for index, item in enumerate(values)
    )


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
    capture_age_ms: float
    unknown_fields: tuple[str, ...] = field(default_factory=tuple)
    contradictory_fields: tuple[str, ...] = field(default_factory=tuple)
    protocol_version: str = PROTOCOL_VERSION

    def validate(self) -> None:
        if not isinstance(self.protocol_version, str):
            raise ContractError("LiveState protocol_version must be a string")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ContractError(f"unsupported LiveState protocol: {self.protocol_version}")
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id:
            raise ContractError("snapshot_id is required")
        if any(
            type(value) is not int
            for value in (
                self.turn,
                self.turns_remaining,
                self.stamina,
                self.max_stamina,
                self.score,
            )
        ):
            raise ContractError("LiveState integer fields must be integers")
        if self.turn < 1 or self.turns_remaining < 0:
            raise ContractError("turn values must be non-negative and turn starts at one")
        if self.stamina < 0 or self.max_stamina < 0 or self.stamina > self.max_stamina:
            raise ContractError("stamina is outside its valid range")
        if (
            self.score < 0
            or not _is_finite_number(self.score_multiplier)
            or self.score_multiplier <= 0
        ):
            raise ContractError("score values are outside their valid range")
        if not _is_finite_number(self.capture_age_ms) or self.capture_age_ms < 0:
            raise ContractError("capture_age_ms cannot be negative")
        for field_name in (
            "action_order",
            "items",
            "field_effects",
            "unknown_fields",
            "contradictory_fields",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) for value in values
            ):
                raise ContractError(f"{field_name} must contain only strings")
        if not isinstance(self.hand, tuple) or any(
            not isinstance(card, LiveCard) for card in self.hand
        ):
            raise ContractError("hand must contain only LiveCard values")
        if not isinstance(self.statuses, tuple) or any(
            not isinstance(status, LiveStatus) for status in self.statuses
        ):
            raise ContractError("statuses must contain only LiveStatus values")
        unknown = set(self.unknown_fields)
        if not self.hand:
            if "hand" not in unknown:
                raise ContractError("an empty hand must be explicitly unknown")
        elif len(self.hand) > 5:
            raise ContractError("Live hand must contain at most five cards")
        slots = [card.slot for card in self.hand]
        if slots != list(range(len(self.hand))):
            raise ContractError("Live hand slots must be ordered and contiguous from zero")
        if any(
            type(card.slot) is not int
            or type(card.cost) is not int
            or card.slot < 0
            or card.cost < 0
            for card in self.hand
        ):
            raise ContractError("card slot and cost must be non-negative integers")
        if any(
            not isinstance(card.card_id, str)
            or not card.card_id
            or type(card.playable) is not bool
            or type(card.identity_accepted) is not bool
            or (card.identity_accepted and card.card_id == UNKNOWN)
            or (card.identity_accepted and card.upgraded is None)
            or (card.upgraded is not None and type(card.upgraded) is not bool)
            for card in self.hand
        ):
            raise ContractError("card identity and boolean fields have invalid types")
        if any(
            not _is_finite_number(card.identity_confidence)
            or not 0.0 <= card.identity_confidence <= 1.0
            for card in self.hand
        ):
            raise ContractError("card identity confidence must be between zero and one")
        if any(
            not isinstance(card.features, dict)
            or any(not isinstance(key, str) for key in card.features)
            or any(not _is_finite_number(value) for value in card.features.values())
            for card in self.hand
        ):
            raise ContractError("card features must map string keys to finite numbers")
        if any(
            not isinstance(card.tags, tuple)
            or any(not isinstance(tag, str) for tag in card.tags)
            for card in self.hand
        ):
            raise ContractError("card tags must contain only strings")
        if any(
            card.box is not None
            and (
                not isinstance(card.box, tuple)
                or len(card.box) != 4
                or any(type(value) is not int for value in card.box)
                or card.box[0] < 0
                or card.box[1] < 0
                or card.box[2] <= 0
                or card.box[3] <= 0
            )
            for card in self.hand
        ):
            raise ContractError("card click boxes must contain four integers")
        if any(
            not isinstance(status.status_id, str)
            or not status.status_id
            or type(status.stacks) is not int
            or (
                status.remaining_turns is not None
                and type(status.remaining_turns) is not int
            )
            for status in self.statuses
        ):
            raise ContractError("status fields have invalid types")
        if unknown - REQUIRED_STATE_FIELDS:
            raise ContractError(f"unknown_fields contains unsupported names: {sorted(unknown - REQUIRED_STATE_FIELDS)}")
        contradictory = set(self.contradictory_fields)
        if contradictory - REQUIRED_STATE_FIELDS:
            raise ContractError(
                "contradictory_fields contains unsupported names: "
                f"{sorted(contradictory - REQUIRED_STATE_FIELDS)}"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LiveState":
        raw = _require_mapping(raw, "LiveState")
        statuses = []
        for index, value in enumerate(
            _require_sequence(
                _require_field(raw, "statuses", "LiveState"),
                "LiveState.statuses",
            )
        ):
            item = _require_mapping(value, f"LiveState.statuses[{index}]")
            remaining = item.get("remaining_turns")
            statuses.append(
                LiveStatus(
                    status_id=_require_string(
                        _require_field(item, "status_id", f"LiveState.statuses[{index}]"),
                        f"LiveState.statuses[{index}].status_id",
                    ),
                    stacks=_require_integer(
                        item.get("stacks", 0),
                        f"LiveState.statuses[{index}].stacks",
                    ),
                    remaining_turns=(
                        None
                        if remaining is None
                        else _require_integer(
                            remaining,
                            f"LiveState.statuses[{index}].remaining_turns",
                        )
                    ),
                )
            )
        hand = []
        for index, value in enumerate(
            _require_sequence(
                _require_field(raw, "hand", "LiveState"),
                "LiveState.hand",
            )
        ):
            item = _require_mapping(value, f"LiveState.hand[{index}]")
            box_value = item.get("box")
            if box_value is None:
                box = None
            else:
                raw_box = _require_sequence(box_value, f"LiveState.hand[{index}].box")
                if len(raw_box) != 4:
                    raise ContractError(
                        f"LiveState.hand[{index}].box must contain four integers"
                    )
                box = tuple(
                    _require_integer(part, f"LiveState.hand[{index}].box[{part_index}]")
                    for part_index, part in enumerate(raw_box)
                )
            raw_features = _require_mapping(
                item.get("features", {}),
                f"LiveState.hand[{index}].features",
            )
            features = {
                key: _require_number(
                    feature,
                    f"LiveState.hand[{index}].features.{key}",
                )
                for key, feature in raw_features.items()
            }
            upgraded = item.get("upgraded")
            hand.append(
                LiveCard(
                    slot=_require_integer(
                        _require_field(item, "slot", f"LiveState.hand[{index}]"),
                        f"LiveState.hand[{index}].slot",
                    ),
                    card_id=_require_string(
                        _require_field(item, "card_id", f"LiveState.hand[{index}]"),
                        f"LiveState.hand[{index}].card_id",
                    ),
                    cost=_require_integer(
                        _require_field(item, "cost", f"LiveState.hand[{index}]"),
                        f"LiveState.hand[{index}].cost",
                    ),
                    playable=_require_boolean(
                        _require_field(item, "playable", f"LiveState.hand[{index}]"),
                        f"LiveState.hand[{index}].playable",
                    ),
                    identity_confidence=_require_number(
                        _require_field(
                            item,
                            "identity_confidence",
                            f"LiveState.hand[{index}]",
                        ),
                        f"LiveState.hand[{index}].identity_confidence",
                    ),
                    identity_accepted=_require_boolean(
                        _require_field(
                            item,
                            "identity_accepted",
                            f"LiveState.hand[{index}]",
                        ),
                        f"LiveState.hand[{index}].identity_accepted",
                    ),
                    box=box,
                    features=features,
                    tags=_text_tuple(
                        item.get("tags", []),
                        f"LiveState.hand[{index}].tags",
                    ),
                    upgraded=(
                        None
                        if upgraded is None
                        else _require_boolean(
                            upgraded,
                            f"LiveState.hand[{index}].upgraded",
                        )
                    ),
                )
            )
        state = cls(
            snapshot_id=_require_string(
                _require_field(raw, "snapshot_id", "LiveState"),
                "LiveState.snapshot_id",
            ),
            turn=_require_integer(
                _require_field(raw, "turn", "LiveState"), "LiveState.turn"
            ),
            turns_remaining=_require_integer(
                _require_field(raw, "turns_remaining", "LiveState"),
                "LiveState.turns_remaining",
            ),
            stamina=_require_integer(
                _require_field(raw, "stamina", "LiveState"), "LiveState.stamina"
            ),
            max_stamina=_require_integer(
                _require_field(raw, "max_stamina", "LiveState"),
                "LiveState.max_stamina",
            ),
            score=_require_integer(
                _require_field(raw, "score", "LiveState"), "LiveState.score"
            ),
            score_multiplier=_require_number(
                _require_field(raw, "score_multiplier", "LiveState"),
                "LiveState.score_multiplier",
            ),
            action_order=_text_tuple(
                _require_field(raw, "action_order", "LiveState"),
                "LiveState.action_order",
            ),
            items=_text_tuple(
                _require_field(raw, "items", "LiveState"),
                "LiveState.items",
            ),
            field_effects=_text_tuple(
                _require_field(raw, "field_effects", "LiveState"),
                "LiveState.field_effects",
            ),
            statuses=tuple(statuses),
            hand=tuple(hand),
            capture_age_ms=_require_number(
                _require_field(raw, "capture_age_ms", "LiveState"),
                "LiveState.capture_age_ms",
            ),
            unknown_fields=_text_tuple(
                raw.get("unknown_fields", []), "LiveState.unknown_fields"
            ),
            contradictory_fields=_text_tuple(
                raw.get("contradictory_fields", []),
                "LiveState.contradictory_fields",
            ),
            protocol_version=_require_string(
                raw.get("protocol_version", PROTOCOL_VERSION),
                "LiveState.protocol_version",
            ),
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
    candidate_traces: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        for field_name in (
            "status",
            "reason",
            "provider",
            "provider_version",
            "protocol_version",
        ):
            if not isinstance(getattr(self, field_name), str):
                raise ContractError(f"proposal {field_name} must be a string")
        if not _is_finite_number(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ContractError("proposal confidence must be between zero and one")
        if not _is_finite_number(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ContractError("proposal elapsed_ms must be a finite non-negative number")
        if (
            not isinstance(self.scores, dict)
            or any(not isinstance(key, str) for key in self.scores)
            or any(not _is_finite_number(value) for value in self.scores.values())
        ):
            raise ContractError("proposal scores must map string keys to finite numbers")
        if (
            not isinstance(self.candidate_traces, dict)
            or any(not isinstance(key, str) for key in self.candidate_traces)
            or any(not isinstance(value, dict) for value in self.candidate_traces.values())
            or not _is_strict_json_value(self.candidate_traces)
        ):
            raise ContractError("proposal traces must contain only strict JSON values")
        if self.action is not None and not isinstance(self.action, LiveAction):
            raise ContractError("proposal action must use the LiveAction contract")
        if self.action is not None and (
            not isinstance(self.action.kind, str)
            or (
                self.action.card_slot is not None
                and type(self.action.card_slot) is not int
            )
        ):
            raise ContractError("proposal action fields have invalid types")

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
