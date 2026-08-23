from __future__ import annotations

import time
from typing import Any
from pathlib import Path
from dataclasses import dataclass

from .contracts import LiveState, LiveAction, DecisionProposal


class PolicyError(ValueError):
    """Raised when a weighted Live policy is invalid."""


@dataclass(frozen=True)
class WeightedPolicyProvider:
    preset_name: str
    weights: dict[str, float]
    provider_name: str = "weighted-policy"
    provider_version: str = "1.0"

    @classmethod
    def load(cls, path: str | Path, preset: str) -> "WeightedPolicyProvider":
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("PyYAML is required to load Live policies") from error
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise PolicyError("unsupported Live policy schema")
        presets = raw.get("presets", {})
        if preset not in presets:
            raise PolicyError(f"unknown Live policy preset: {preset}")
        selected = presets[preset]
        weights = selected.get("weights") if isinstance(selected, dict) else None
        if not isinstance(weights, dict) or not weights:
            raise PolicyError("Live policy weights must be a non-empty mapping")
        parsed = {str(key): float(value) for key, value in weights.items()}
        if any(not -1000.0 <= value <= 1000.0 for value in parsed.values()):
            raise PolicyError("Live policy weight is outside the supported range")
        return cls(preset_name=preset, weights=parsed)

    def decide(self, state: LiveState) -> DecisionProposal:
        started = time.perf_counter()
        scores: dict[str, float] = {}
        playable = []
        for card in state.hand:
            if not card.playable or card.cost > state.stamina:
                continue
            score = sum(self.weights.get(name, 0.0) * value for name, value in card.features.items())
            score -= self.weights.get("stamina_cost", 0.0) * card.cost
            scores[str(card.slot)] = score
            playable.append((score, -card.slot, card))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not playable:
            return DecisionProposal(
                status="STOP",
                reason="no_weighted_legal_candidate",
                action=None,
                confidence=1.0,
                provider=self.provider_name,
                provider_version=self.provider_version,
                elapsed_ms=elapsed_ms,
                scores=scores,
            )
        playable.sort(reverse=True)
        best_score, _slot_key, best_card = playable[0]
        gap = best_score - playable[1][0] if len(playable) > 1 else abs(best_score) + 1.0
        confidence = max(0.0, min(1.0, 0.5 + gap / (2.0 * (abs(best_score) + 1.0))))
        return DecisionProposal(
            status="PROPOSE",
            reason=f"weighted_max:{self.preset_name}",
            action=LiveAction(kind="PLAY_CARD", card_slot=best_card.slot),
            confidence=confidence,
            provider=self.provider_name,
            provider_version=self.provider_version,
            elapsed_ms=elapsed_ms,
            scores=scores,
        )
