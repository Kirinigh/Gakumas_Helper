"""Carry canonical catalog title corrections without changing frozen identities."""

from __future__ import annotations

import unicodedata
from copy import deepcopy

IDENTITY_FIELDS = (
    "catalog_name", "api_name", "internal_id", "upgrade_count", "asset_id",
    "is_character_asset", "asset_variants",
)


def compare_identity(previous, current):
    """Only an unchanged native identity's canonical title may replace its typo."""
    if previous["business_id"] != current["business_id"]:
        raise ValueError("identity fields changed: business_id")
    changed = [field for field in IDENTITY_FIELDS if previous.get(field) != current[field]]
    if not changed:
        return None
    canonical = unicodedata.normalize("NFKC", current["api_name"])
    if changed != ["catalog_name"] or unicodedata.normalize("NFKC", current["catalog_name"]) != canonical:
        raise ValueError(f"identity fields changed: {changed}")
    return {
        "previous": {key: deepcopy(previous[key]) for key in ("business_id", *IDENTITY_FIELDS)},
        "current": {key: deepcopy(current[key]) for key in ("business_id", *IDENTITY_FIELDS)},
    }


def apply_title_corrections(mappings, corrections):
    """Replay exact before/after evidence; never rewrite historical crosswalk files."""
    if not isinstance(corrections, list):
        raise ValueError("catalog title corrections must be a list")
    seen = set()
    for correction in corrections:
        previous, current = correction["previous"], correction["current"]
        business_id = previous["business_id"]
        if business_id in seen or current["business_id"] != business_id or business_id not in mappings:
            raise ValueError("duplicate or unknown catalog title correction")
        seen.add(business_id)
        actual = {key: mappings[business_id][key] for key in ("business_id", *IDENTITY_FIELDS)}
        if actual != previous or compare_identity(previous, current) != correction:
            raise ValueError(f"catalog title correction evidence mismatch: {business_id}")
        mappings[business_id] = {**mappings[business_id], "catalog_name": current["catalog_name"]}
