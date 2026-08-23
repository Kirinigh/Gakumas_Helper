from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Callable

from .state import live_card_from_task100_prediction
from .contracts import REQUIRED_STATE_FIELDS, LiveState, LiveStatus


class DmmLiveVisualStateProvider:
    """Build a fail-closed Live state from versioned DMM visual observations.

    OCR, template matching and card recognition stay outside this pure
    normalizer. Every visual field must carry an explicit ``accepted`` result;
    absent or rejected fields are preserved in ``unknown_fields`` rather than
    being silently promoted to gameplay truth.
    """

    provider_name = "dmm-live-visual"
    provider_version = "1.0"

    def build_state(self, observation: dict[str, Any]) -> LiveState:
        fields = observation.get("fields", {})
        if not isinstance(fields, Mapping):
            fields = {}
        unknown: set[str] = set()
        contradictory: set[str] = set()

        turn = self._field(fields, "turn", int, 1, unknown)
        turns_remaining = self._field(fields, "turns_remaining", int, 0, unknown)
        stamina = self._field(fields, "stamina", int, 0, unknown)
        max_stamina = self._field(fields, "max_stamina", int, stamina, unknown)
        score = self._field(fields, "score", int, 0, unknown)
        score_multiplier = self._field(fields, "score_multiplier", float, 1.0, unknown)
        action_order = self._sequence_field(fields, "action_order", unknown)
        items = self._sequence_field(fields, "items", unknown)
        field_effects = self._sequence_field(fields, "field_effects", unknown)
        statuses = self._status_field(fields, unknown)

        if turn < 1:
            contradictory.add("turn")
            turn = 1
        if turns_remaining < 0:
            contradictory.add("turns_remaining")
            turns_remaining = 0
        if stamina < 0:
            contradictory.add("stamina")
            stamina = 0
        if max_stamina < stamina or max_stamina < 0:
            contradictory.add("max_stamina")
            max_stamina = stamina
        if score < 0:
            contradictory.add("score")
            score = 0
        if score_multiplier <= 0:
            contradictory.add("score_multiplier")
            score_multiplier = 1.0

        hand = []
        raw_cards = observation.get("card_predictions", ())
        if not isinstance(raw_cards, (list, tuple)) or not raw_cards:
            unknown.add("hand")
            raw_cards = ()
        for entry in raw_cards:
            if not isinstance(entry, Mapping) or entry.get("prediction") is None:
                unknown.add("hand")
                continue
            cost = entry.get("cost")
            playable = entry.get("playable")
            if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
                cost = 0
                unknown.add("hand")
            if not isinstance(playable, bool):
                playable = False
                unknown.add("hand")
            features = entry.get("features", {})
            if not isinstance(features, Mapping):
                features = {}
                unknown.add("hand")
            try:
                parsed_features = {str(key): float(value) for key, value in features.items()}
            except (TypeError, ValueError):
                parsed_features = {}
                unknown.add("hand")
            tags = entry.get("tags", ())
            if not isinstance(tags, (list, tuple)):
                tags = ()
                unknown.add("hand")
            hand.append(
                live_card_from_task100_prediction(
                    entry["prediction"],
                    cost=cost,
                    playable=playable,
                    features=parsed_features,
                    tags=tuple(str(value) for value in tags),
                    upgraded=(
                        bool(entry["upgraded"])
                        if entry.get("upgraded") is not None
                        else None
                    ),
                )
            )

        supplied_unknown = observation.get("unknown_fields", ())
        if isinstance(supplied_unknown, (list, tuple)):
            unknown.update(str(value) for value in supplied_unknown)
        if unknown - REQUIRED_STATE_FIELDS:
            unsupported = sorted(unknown - REQUIRED_STATE_FIELDS)
            raise ValueError(f"unsupported visual unknown fields: {unsupported}")
        supplied_contradictory = observation.get("contradictory_fields", ())
        if isinstance(supplied_contradictory, (list, tuple)):
            contradictory.update(str(value) for value in supplied_contradictory)

        state = LiveState(
            snapshot_id=str(observation.get("snapshot_id", "")),
            turn=turn,
            turns_remaining=turns_remaining,
            stamina=stamina,
            max_stamina=max_stamina,
            score=score,
            score_multiplier=score_multiplier,
            action_order=action_order,
            items=items,
            field_effects=field_effects,
            statuses=statuses,
            hand=tuple(hand),
            capture_age_ms=float(observation.get("capture_age_ms", 0.0)),
            unknown_fields=tuple(sorted(unknown)),
            contradictory_fields=tuple(sorted(contradictory)),
        )
        state.validate()
        return state

    @staticmethod
    def _field(
        fields: Mapping[str, Any],
        name: str,
        converter: Callable[[Any], Any],
        fallback: Any,
        unknown: set[str],
    ) -> Any:
        result = fields.get(name)
        if not isinstance(result, Mapping) or result.get("accepted") is not True:
            unknown.add(name)
            return fallback
        try:
            return converter(result["value"])
        except (KeyError, TypeError, ValueError):
            unknown.add(name)
            return fallback

    @classmethod
    def _sequence_field(
        cls,
        fields: Mapping[str, Any],
        name: str,
        unknown: set[str],
    ) -> tuple[str, ...]:
        value = cls._field(fields, name, lambda item: item, (), unknown)
        if not isinstance(value, (list, tuple)):
            unknown.add(name)
            return ()
        return tuple(str(item) for item in value)

    @classmethod
    def _status_field(
        cls,
        fields: Mapping[str, Any],
        unknown: set[str],
    ) -> tuple[LiveStatus, ...]:
        value = cls._field(fields, "statuses", lambda item: item, (), unknown)
        if not isinstance(value, (list, tuple)):
            unknown.add("statuses")
            return ()
        try:
            return tuple(
                item if isinstance(item, LiveStatus) else LiveStatus(**item)
                for item in value
            )
        except (TypeError, ValueError):
            unknown.add("statuses")
            return ()
