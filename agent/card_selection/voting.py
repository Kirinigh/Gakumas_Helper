from __future__ import annotations

from collections import Counter
from dataclasses import replace
from statistics import mean
from typing import Iterable

from .types import UNKNOWN, CardPrediction


class MultiFrameVoter:
    """Recover a rejected slot only from a unique, agreeing multi-frame majority."""

    def __init__(
        self,
        *,
        minimum_frames: int = 2,
        minimum_agreeing: int = 2,
        acceptance_threshold: float | None = None,
        minimum_margin: float | None = None,
    ) -> None:
        if minimum_frames < 2 or minimum_agreeing < 2 or minimum_agreeing > minimum_frames:
            raise ValueError("multi-frame voting requires 2 <= minimum_agreeing <= minimum_frames")
        if acceptance_threshold is not None and not 0.0 < acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold must be in (0,1]")
        if minimum_margin is not None and not 0.0 <= minimum_margin <= 1.0:
            raise ValueError("minimum_margin must be in [0,1]")
        self.minimum_frames = minimum_frames
        self.minimum_agreeing = minimum_agreeing
        self.acceptance_threshold = acceptance_threshold
        self.minimum_margin = minimum_margin

    def vote(self, predictions: Iterable[CardPrediction]) -> CardPrediction:
        items = tuple(predictions)
        if len(items) < self.minimum_frames:
            raise ValueError("not enough frames for voting")
        slots = {item.slot for item in items}
        if len(slots) != 1:
            raise ValueError("all predictions in a vote must refer to one slot")
        eligible = tuple(
            item
            for item in items
            if item.predicted_card_id not in {UNKNOWN, "0"}
            and item.reason in {"accepted", "below_acceptance_threshold", "below_margin_threshold"}
        )
        counts = Counter(item.predicted_card_id for item in eligible)
        template = max(items, key=lambda item: (item.confidence, item.frame_id))
        if not counts:
            return replace(
                template,
                card_id=UNKNOWN,
                accepted=False,
                reason="multiframe_no_accepted_vote",
                source_frames=tuple(item.frame_id for item in items),
            )
        ranked = counts.most_common()
        top_card, top_count = ranked[0]
        tied = len(ranked) > 1 and ranked[1][1] == top_count
        if tied or top_count < self.minimum_agreeing:
            return replace(
                template,
                card_id=UNKNOWN,
                accepted=False,
                reason="multiframe_inconsistent",
                source_frames=tuple(item.frame_id for item in items),
            )
        winners = tuple(item for item in eligible if item.predicted_card_id == top_card)
        best = max(winners, key=lambda item: (item.confidence, item.frame_id))
        aggregate_confidence = mean(item.confidence for item in winners)
        aggregate_margin = mean(item.margin for item in winners)
        if self.acceptance_threshold is None or self.minimum_margin is None:
            return replace(
                best,
                card_id=UNKNOWN,
                accepted=False,
                reason="multiframe_threshold_not_calibrated",
                source_frames=tuple(item.frame_id for item in items),
            )
        if aggregate_confidence < self.acceptance_threshold or aggregate_margin < self.minimum_margin:
            return replace(
                best,
                card_id=UNKNOWN,
                accepted=False,
                reason="multiframe_below_threshold",
                confidence=aggregate_confidence,
                margin=aggregate_margin,
                source_frames=tuple(item.frame_id for item in items),
            )
        return replace(
            best,
            card_id=top_card,
            confidence=aggregate_confidence,
            margin=aggregate_margin,
            accepted=True,
            reason="multiframe_consensus",
            source_frames=tuple(item.frame_id for item in items),
        )
