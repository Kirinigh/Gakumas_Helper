from __future__ import annotations

from typing import Any

from .contracts import UNKNOWN, LiveCard, LiveState


class SnapshotLiveStateProvider:
    """Strict provider for normalized replay data or a future visual adapter."""

    provider_name = "normalized-snapshot"
    provider_version = "1.0"

    def build_state(self, observation: dict[str, Any]) -> LiveState:
        return LiveState.from_dict(observation)


def live_card_from_task100_prediction(
    prediction: Any,
    *,
    cost: int,
    playable: bool,
    features: dict[str, float] | None = None,
    tags: tuple[str, ...] = (),
    upgraded: bool | None = None,
) -> LiveCard:
    """Map a versioned card-identity result into a Live hand card."""

    card_id = str(getattr(prediction, "card_id", UNKNOWN))
    accepted = bool(getattr(prediction, "accepted", False)) and card_id != UNKNOWN
    raw_box = getattr(prediction, "box", None)
    return LiveCard(
        slot=int(getattr(prediction, "slot")),
        card_id=card_id,
        cost=int(cost),
        playable=bool(playable),
        identity_confidence=float(getattr(prediction, "confidence", 0.0)),
        identity_accepted=accepted,
        box=tuple(int(value) for value in raw_box) if raw_box is not None else None,
        features=dict(features or {}),
        tags=tuple(tags),
        upgraded=upgraded,
    )
