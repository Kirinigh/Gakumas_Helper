"""Cached arena candidate domains from accepted facets and the active RIS rows."""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence

PLANS = frozenset({"free", "sense", "logic", "anomaly"})
KINDS = frozenset({"p_idol", "support", "basic", "other"})
RARITIES = frozenset({"L", "N", "R", "SR", "SSR"})
SLOT_KINDS = (frozenset({"p_idol"}), frozenset({"support", "other"}), *(frozenset({"other"}),) * 4)


def classification_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    installed = root / "data/skill_card_classification.json"
    return installed if installed.is_file() else root / "assets/data/skill_card_classification.json"


def _load_rows() -> tuple[dict[int, Mapping[str, Any]], str | None]:
    try:
        payload = json.loads(classification_path().read_text(encoding="utf-8-sig"))
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or payload.get("classification_rule") != "game_facets_v1_all_legend_other"
            or not isinstance(payload.get("cards"), list)
        ):
            raise ValueError("unsupported skill-card classification schema")
        rows = {}
        for row in payload["cards"]:
            if not isinstance(row, Mapping) or type(row.get("business_id")) is not int or row["business_id"] < 1:
                raise ValueError("invalid skill-card classification identity")
            if row["business_id"] in rows:
                raise ValueError("duplicate skill-card classification identity")
            rows[row["business_id"]] = row
        return rows, None
    except (OSError, ValueError, TypeError) as error:
        # Classification is a narrowing aid. A missing package cannot remove
        # active RIS cards from the existing title/detail fallback domain.
        return {}, f"{type(error).__name__}: {error}"


def _source_matches(card: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    source = row.get("source")
    if not isinstance(source, Mapping):
        return False
    return (
        all(key in card and card[key] == row.get(key) for key in ("name", "upgraded", "plan"))
        and all(
            key in card and card[key] == source.get(frozen_key)
            for key, frozen_key in (
                ("sourceType", "catalog_source_type"),
                ("rarity", "catalog_rarity"),
                ("type", "card_type"),
            )
        )
        and "pIdolId" in card
        and (card["pIdolId"] or None) == source.get("catalog_p_idol_id")
    )


def _active_facets(card: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Use only explicit current fields; unknowns remain unconstrained dimensions."""

    plan = card.get("plan")
    plan = plan if isinstance(plan, str) and plan in PLANS else None
    rarity = card.get("rarity")
    # RIS's trouble rarity T is the official N class (the frozen 眠気 row).
    rarity = "N" if rarity == "T" else rarity
    rarity = rarity if isinstance(rarity, str) and rarity in RARITIES else None
    source = card.get("sourceType")
    if rarity == "L":
        kind = "other"
    elif source == "pIdol":
        kind = "p_idol"
    elif source == "support":
        kind = "support"
    elif source in ("produce", "default") and isinstance(card.get("name"), str):
        kind = "basic" if "基本" in card["name"] else "other"
    else:
        kind = None
    return plan, kind, rarity


class ArenaSkillCardEligibility:
    """One catalog-instance snapshot; candidate queries perform no file access."""

    def __init__(self, cards: Sequence[Mapping[str, Any]]) -> None:
        frozen, self.load_error = _load_rows()
        self.facets = {}
        self.sources = {}
        unknown = set()
        for card in cards:
            card_id = int(card["id"])
            facets = _active_facets(card)
            row = frozen.get(card_id)
            from_frozen = False
            if row is not None and _source_matches(card, row):
                labels = tuple(row.get(key) for key in ("plan", "kind", "rarity"))
                if (
                    all(isinstance(label, str) for label in labels)
                    and labels[0] in PLANS and labels[1] in KINDS and labels[2] in RARITIES
                    and labels[0] == facets[0] and labels[2] == facets[2]
                ):
                    facets = labels
                    from_frozen = True
            self.facets[card_id] = facets
            self.sources[card_id] = "classification" if from_frozen else "active_catalog"
            if None in facets:
                unknown.add(card_id)
        self.unknown_ids = frozenset(unknown)
        self.required_ids = tuple(sorted(
            card_id for card_id, (_, _, rarity) in self.facets.items() if rarity not in {"L", "N"}
        ))
        self.candidate_sets = {}
        for plan in ("sense", "logic", "anomaly"):
            plan_ids = frozenset(
                card_id for card_id in self.required_ids if self.facets[card_id][0] in {None, "free", plan}
            )
            self.candidate_sets[(plan, None)] = plan_ids
            for slot, kinds in enumerate(SLOT_KINDS):
                self.candidate_sets[(plan, slot)] = frozenset(
                    card_id for card_id in plan_ids if self.facets[card_id][1] is None or self.facets[card_id][1] in kinds
                )
