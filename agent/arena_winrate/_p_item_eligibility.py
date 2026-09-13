"""Catalog-instance P-item domains from reviewed metadata and active RIS fields."""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence

PLANS = frozenset({"free", "sense", "logic", "anomaly"})
KINDS = frozenset({"p_idol", "support_inheritable", "support_non_inheritable", "other"})


def classification_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    installed = root / "data/p_item_classification.json"
    return installed if installed.is_file() else root / "assets/data/p_item_classification.json"


def _load_rows() -> tuple[dict[int, Mapping[str, Any]], str | None]:
    try:
        payload = json.loads(classification_path().read_text(encoding="utf-8-sig"))
        if (not isinstance(payload, Mapping) or payload.get("schema_version") != 1
                or payload.get("classification_rule") != "p_item_plan_and_four_kinds_v1"
                or not isinstance(payload.get("items"), list)):
            raise ValueError("unsupported P-item classification schema")
        rows = {}
        for row in payload["items"]:
            if (not isinstance(row, Mapping) or type(row.get("business_id")) is not int
                    or row["business_id"] < 1 or row["business_id"] in rows):
                raise ValueError("invalid or duplicate P-item classification identity")
            rows[row["business_id"]] = row
        return rows, None
    except (OSError, ValueError, TypeError) as error:
        return {}, f"{type(error).__name__}: {error}"


def _source_matches(item: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    source = row.get("source")
    return isinstance(source, Mapping) and all(
        key in item and item[key] == row.get(key) for key in ("name", "upgraded")
    ) and all(
        key in item and item[key] == source.get(frozen_key)
        for key, frozen_key in (
            ("plan", "catalog_plan"), ("mode", "catalog_mode"),
            ("sourceType", "catalog_source_type"), ("rarity", "catalog_rarity"),
        )
    ) and "pIdolId" in item and (item["pIdolId"] or None) == source.get("catalog_p_idol_id")


def _active_facets(item: Mapping[str, Any]) -> tuple[str | None, str | None]:
    plan = item.get("plan")
    plan = plan if isinstance(plan, str) and plan in PLANS else None
    source, mode = item.get("sourceType"), item.get("mode")
    if source == "pIdol":
        kind = "p_idol"
    elif source == "support" and mode in {"stage", "produce"}:
        kind = "support_inheritable" if mode == "stage" else "support_non_inheritable"
    elif source == "produce":
        kind = "other"
    else:
        kind = None
    return plan, kind


class ArenaPItemEligibility:
    """Unknown future metadata stays eligible; explicit non-arena items do not."""

    def __init__(self, items: Sequence[Mapping[str, Any]]) -> None:
        frozen, self.load_error = _load_rows()
        self.facets, self.sources = {}, {}
        for item in items:
            item_id = int(item["id"])
            facets = _active_facets(item)
            row = frozen.get(item_id)
            from_frozen = False
            if row is not None and _source_matches(item, row):
                plan, kind = row.get("plan"), row.get("kind")
                if isinstance(plan, str) and plan in PLANS and isinstance(kind, str) and kind in KINDS:
                    facets = plan, kind
                    from_frozen = True
            self.facets[item_id] = facets
            self.sources[item_id] = "classification" if from_frozen else "active_catalog"
        self.unknown_ids = frozenset(item_id for item_id, facets in self.facets.items() if None in facets)
        self.required_ids = tuple(sorted(
            int(item["id"]) for item in items
            if self.facets[int(item["id"])][1] not in {"support_non_inheritable", "other"}
            and item.get("mode") != "produce"
        ))
        self.candidate_sets = {
            plan: frozenset(item_id for item_id in self.required_ids
                            if self.facets[item_id][0] in {None, "free", plan})
            for plan in ("sense", "logic", "anomaly")
        }
        self.candidate_sets[None] = self.candidate_sets["free"] = frozenset(self.required_ids)

        self.slot_candidate_sets = {
            (plan, slot): frozenset(item_id for item_id in ids
                                   if self.facets[item_id][1] in {None, kind})
            for plan, ids in self.candidate_sets.items()
            for slot, kind in enumerate(("p_idol", "support_inheritable", "support_inheritable", "support_inheritable"))
        }
