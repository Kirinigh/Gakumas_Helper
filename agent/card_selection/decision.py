from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .types import UNKNOWN, CardPrediction, SelectionContext, SelectionDecision


class PolicyError(ValueError):
    """Raised for an invalid or ambiguous selection policy."""


def _normalise_id(value: Any) -> str:
    return str(value).strip()


def load_policy(path: str | Path, preset: str) -> "CardSelectionPolicy":
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load card-selection policies") from error
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise PolicyError("unsupported card-selection policy schema")
    presets = data.get("presets", {})
    if preset not in presets:
        raise PolicyError(f"unknown card-selection preset: {preset}")
    return CardSelectionPolicy(schema_version=1, preset_name=preset, data=presets[preset])


@dataclass(frozen=True)
class CardSelectionPolicy:
    schema_version: int
    preset_name: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PolicyError("unsupported policy schema")
        if not isinstance(self.data.get("baseline"), dict):
            raise PolicyError("preset baseline must be a mapping")
        if self.data.get("unranked") not in {"STOP", "LAST"}:
            raise PolicyError("unranked must be STOP or LAST")

    def _base_order(self, plan: str) -> list[str]:
        baseline = self.data["baseline"]
        selected = baseline.get(plan, baseline.get("free", []))
        common = baseline.get("free", []) if plan != "free" else []
        result: list[str] = []
        for value in [*selected, *common]:
            card_id = _normalise_id(value)
            if card_id and card_id not in result:
                result.append(card_id)
        return result

    @staticmethod
    def _matches(when: dict[str, Any], context: SelectionContext) -> bool:
        for field in ("scene", "plan", "p_idol", "stage"):
            expected = _normalise_id(when.get(field, "*"))
            if expected not in {"*", _normalise_id(getattr(context, field))}:
                return False
        required_cards = {_normalise_id(value) for value in when.get("deck_contains_all", [])}
        return required_cards.issubset(set(context.deck_card_ids))

    def _resolved_order(self, context: SelectionContext) -> tuple[list[str], tuple[str, ...]]:
        order = self._base_order(context.plan)
        matches: list[tuple[int, int, str, list[str]]] = []
        for override in self.data.get("overrides", []):
            when = override.get("when", {})
            if not self._matches(when, context):
                continue
            specificity = sum(_normalise_id(when.get(field, "*")) != "*" for field in ("scene", "plan", "p_idol", "stage"))
            specificity += len(when.get("deck_contains_all", []))
            matches.append((int(override.get("priority", 0)), specificity, str(override["id"]), [_normalise_id(v) for v in override["order"]]))
        applied: list[str] = []
        for _priority, _specificity, override_id, override_order in sorted(matches):
            order = [*override_order, *(card_id for card_id in order if card_id not in override_order)]
            applied.append(override_id)
        return order, tuple(applied)

    def decide(self, predictions: Iterable[CardPrediction], context: SelectionContext) -> SelectionDecision:
        candidates = tuple(predictions)
        trace = tuple(item.to_dict() for item in candidates)
        if not candidates:
            return self._stop("candidate_missing", trace)
        if any(item.detector_label not in {"cards", "suggestions"} for item in candidates):
            return self._stop("nonselectable_detector_label", trace)
        slots = [item.slot for item in candidates]
        if len(slots) != len(set(slots)):
            return self._stop("duplicate_slot", trace)
        if any(not item.visible or not item.clickable for item in candidates):
            return self._stop("candidate_not_visible_or_clickable", trace)
        if any(not item.accepted or item.card_id == UNKNOWN for item in candidates):
            return self._stop("candidate_unknown", trace)
        order, applied = self._resolved_order(context)
        ranks = {card_id: index for index, card_id in enumerate(order)}
        unranked = [item for item in candidates if item.card_id not in ranks]
        if unranked and self.data["unranked"] == "STOP":
            return self._stop("unranked_candidate", trace, applied)
        fallback_rank = len(ranks)
        best_rank = min(ranks.get(item.card_id, fallback_rank) for item in candidates)
        finalists = [item for item in candidates if ranks.get(item.card_id, fallback_rank) == best_rank]
        if len(finalists) > 1:
            finalists.sort(key=lambda item: item.confidence, reverse=True)
            required_gap = float(self.data.get("tie_break_min_confidence_gap", 1.0))
            if finalists[0].confidence - finalists[1].confidence < required_gap:
                return self._stop("unresolved_tie", trace, applied)
        selected = finalists[0]
        center_x, center_y = selected.box[0] + selected.box[2] // 2, selected.box[1] + selected.box[3] // 2
        return SelectionDecision(
            status="SELECT",
            reason="unique_ranked_candidate",
            chosen_slot=selected.slot,
            click_point=(center_x, center_y),
            candidates=trace,
            applied_overrides=applied,
            policy_version=self.schema_version,
        )

    def _stop(
        self,
        reason: str,
        candidates: tuple[dict[str, Any], ...],
        applied: tuple[str, ...] = (),
    ) -> SelectionDecision:
        return SelectionDecision(
            status="STOP",
            reason=reason,
            chosen_slot=None,
            click_point=None,
            candidates=candidates,
            applied_overrides=applied,
            policy_version=self.schema_version,
        )
