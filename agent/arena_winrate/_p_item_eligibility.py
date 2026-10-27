"""Catalog-instance P-item domains from reviewed metadata and active RIS fields."""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence

PLANS = frozenset({"free", "sense", "logic", "anomaly"})
KINDS = frozenset({"p_idol", "support_inheritable", "support_non_inheritable", "other"})
PLACEHOLDER_RULE = "catalog-only-blank-reference-v1"


def classification_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    installed = root / "data/p_item_classification.json"
    return installed if installed.is_file() else root / "assets/data/p_item_classification.json"


def _declared_placeholders(payload: Mapping[str, Any], rows: Mapping[int, Mapping[str, Any]]) -> frozenset[int]:
    declaration = payload.get("temporary_missing_references")
    if (not isinstance(declaration, Mapping)
            or declaration.get("rule_version") != PLACEHOLDER_RULE
            or declaration.get("reference_enabled") is not False
            or declaration.get("status") != "USER_AUTHORIZED_TEMPORARY_MISSING_REFERENCES"):
        return frozenset()
    ids = declaration.get("business_ids")
    if (not isinstance(ids, list) or any(type(item_id) is not int or item_id < 1 for item_id in ids)
            or len(ids) != len(set(ids))):
        return frozenset()
    placeholders = set()
    for item_id in ids:
        row = rows.get(item_id, {})
        source = row.get("source", {})
        marker = source.get("temporary_missing_reference") if isinstance(source, Mapping) else None
        if (row.get("reference_available") is False and row.get("kind") == "p_idol"
                and isinstance(row.get("name"), str) and row["name"].rstrip("+＋")
                and type(row.get("upgraded")) is bool
                and row["name"].endswith(("+", "＋")) is row["upgraded"]
                and isinstance(marker, Mapping) and marker.get("rule_version") == PLACEHOLDER_RULE
                and marker.get("reference_enabled") is False
                and marker.get("reason") == "VERIFIED_IDENTITY_AND_UI_PENDING"
                and source.get("official") is None
                and source.get("official_mapping") == "catalog_only_temporary_missing_reference"):
            placeholders.add(item_id)
    paired = set()
    source_fields = ("catalog_plan", "catalog_mode", "catalog_source_type", "catalog_p_idol_id", "catalog_rarity")
    for item_id in placeholders:
        row = rows[item_id]
        partner_id = row.get("paired_business_id")
        if type(partner_id) is not int or partner_id not in placeholders:
            continue
        partner = rows[partner_id]
        if (partner.get("paired_business_id") == item_id and partner["upgraded"] is not row["upgraded"]
                and row.get("plan") == partner.get("plan")
                and row["name"].rstrip("+＋") == partner["name"].rstrip("+＋")
                and all(row["source"].get(key) == partner["source"].get(key) for key in source_fields)):
            paired.add(item_id)
    return frozenset(paired)


def _load_rows() -> tuple[dict[int, Mapping[str, Any]], str | None, frozenset[int]]:
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
        return rows, None, _declared_placeholders(payload, rows)
    except (OSError, ValueError, TypeError) as error:
        return {}, f"{type(error).__name__}: {error}", frozenset()


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
    """Keep unknown rows eligible; exclude explicit non-arena and preview rows."""

    def __init__(self, items: Sequence[Mapping[str, Any]]) -> None:
        frozen, self.load_error, declared_placeholders = _load_rows()
        self.facets, self.sources = {}, {}
        placeholders = set()
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
                    if (item_id in declared_placeholders and item.get("sourceType") == "pIdol"
                            and item.get("mode") == "stage"
                            and type(item.get("pIdolId")) is int and item["pIdolId"] > 0
                            and type(item.get("upgraded")) is bool
                            and row.get("upgraded") is item["upgraded"]
                            and row.get("plan") == item.get("plan")):
                        placeholders.add(item_id)
            self.facets[item_id] = facets
            self.sources[item_id] = "classification" if from_frozen else "active_catalog"
        self.unknown_ids = frozenset(item_id for item_id, facets in self.facets.items() if None in facets)
        # The complete catalog keeps these IDs as inventory, while neither
        # image coverage nor a detail title can enable a preview placeholder.
        self.reference_placeholder_ids = frozenset(
            item_id for item_id in placeholders if frozen[item_id]["paired_business_id"] in placeholders
        )
        self.required_ids = tuple(sorted(
            int(item["id"]) for item in items
            if self.facets[int(item["id"])][1] not in {"support_non_inheritable", "other"}
            and item.get("mode") != "produce"
            and int(item["id"]) not in self.reference_placeholder_ids
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
