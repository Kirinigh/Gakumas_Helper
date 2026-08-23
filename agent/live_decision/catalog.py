from __future__ import annotations

import re
import csv
from pathlib import Path
from dataclasses import dataclass

_NUMBER = r"(-?\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class SkillCardEntry:
    card_id: str
    name: str
    card_type: str
    plan: str
    upgraded: bool
    stamina_cost: int
    conditions: str
    actions: str


class SkillCardFeatureCatalog:
    """Frozen card metadata used by the task-local Live heuristic."""

    def __init__(self, entries: dict[str, SkillCardEntry]):
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path) -> "SkillCardFeatureCatalog":
        entries: dict[str, SkillCardEntry] = {}
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                card_id = str(row.get("id", "")).strip()
                if not card_id:
                    continue
                cost_match = re.search(r"cost-=(\d+)", str(row.get("cost", "")))
                entries[card_id] = SkillCardEntry(
                    card_id=card_id,
                    name=str(row.get("name", "")),
                    card_type=str(row.get("type", "")),
                    plan=str(row.get("plan", "free")),
                    upgraded=str(row.get("upgraded", "")).upper() == "TRUE",
                    stamina_cost=int(cost_match.group(1)) if cost_match else 0,
                    conditions=str(row.get("conditions", "")),
                    actions=str(row.get("actions", "")),
                )
        if not entries:
            raise ValueError("skill-card catalog is empty")
        return cls(entries)

    def resolve_upgrade_state(
        self,
        candidate_ids: tuple[str, ...],
        *,
        upgraded: bool,
        plan: str,
    ) -> str | None:
        candidates = [
            self.entries[card_id]
            for card_id in candidate_ids
            if card_id in self.entries and self.entries[card_id].plan in {"free", plan}
        ]
        if len(candidates) == 1:
            return candidates[0].card_id
        matches = [entry.card_id for entry in candidates if entry.upgraded is upgraded]
        return matches[0] if len(matches) == 1 else None

    def resolve_family(self, candidate_ids: tuple[str, ...], *, plan: str) -> str | None:
        candidates = [
            self.entries[card_id]
            for card_id in candidate_ids
            if card_id in self.entries and self.entries[card_id].plan in {"free", plan}
        ]
        if len(candidates) == 1:
            return candidates[0].card_id
        if len(candidates) == 2:
            base, upgraded = sorted(candidates, key=lambda entry: int(entry.card_id))
            if upgraded.name == f"{base.name}+" or base.name == upgraded.name.rstrip("+"):
                return base.card_id
        return None

    def features(self, card_id: str, *, turns_remaining: int) -> dict[str, float]:
        entry = self.entries[card_id]
        actions = entry.actions

        def total(pattern: str) -> float:
            return sum(float(value) for value in re.findall(pattern, actions))

        score_gain = total(r"(?:^|[; {])(?:g\.)?score\+=" + _NUMBER)
        stamina_recovery = total(r"fixedStamina\+=" + _NUMBER)
        stamina_recovery += total(r"genki\+=" + _NUMBER) * 0.25
        status_gain = total(r"fullPowerCharge\+=" + _NUMBER) * 1.5
        status_gain += total(r"(?:concentration|goodConditionTurns|motivation)\+=" + _NUMBER) * 0.5
        status_gain += actions.count("setStance(strength)") * 4.0
        status_gain += actions.count("setStance(preservation)") * 3.0
        multiplier_synergy = actions.count("scoreTimes") * 4.0
        multiplier_synergy += actions.count("target:active") * 2.0
        multiplier_synergy += actions.count("at:cardUsed") * 2.0
        multiplier_synergy += actions.count("goodConditionTurns") * 1.5
        draw_value = total(r"drawCard\(" + _NUMBER + r"\)")
        draw_value += actions.count("holdSelected") * 2.0
        risk = max(0, entry.stamina_cost - 5) * 0.5
        risk += 0.5 if entry.conditions else 0.0

        if turns_remaining <= 3:
            score_gain *= 1.8
            status_gain *= 0.3
            multiplier_synergy *= 0.7
            draw_value *= 0.2
            if entry.card_type == "active":
                score_gain += 2.0
        elif turns_remaining >= 7:
            score_gain *= 0.15
            status_gain *= 1.5
            multiplier_synergy *= 1.35
            draw_value *= 1.4
            if entry.card_type == "mental":
                status_gain += 2.0

        return {
            "score_gain": score_gain,
            "stamina_recovery": stamina_recovery,
            "status_gain": status_gain,
            "multiplier_synergy": multiplier_synergy,
            "draw_value": draw_value,
            "risk": risk,
        }
