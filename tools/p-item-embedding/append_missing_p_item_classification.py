"""Append the authorized catalog-only P-item gaps without changing recognition.

The shared white PNG is a visible missing-image asset, never a recognition
reference. Existing gallery bytes, official identities and classification rows
are retained; the normal source-backed builder remains strict.
"""

from __future__ import annotations

import re
import json
import hashlib
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

RULE = "catalog-only-blank-reference-v1"
STATUS = "USER_AUTHORIZED_TEMPORARY_MISSING_REFERENCES"
PLACEHOLDER_IMAGE = "resource/base/model/embedding/p_item_missing-reference.png"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def catalog_rows(path):
    rows = read(path)
    require(isinstance(rows, list) and rows, "catalog must contain ordered rows")
    require([row.get("id") for row in rows] == list(range(1, len(rows) + 1)), "catalog IDs must be contiguous and ordered")
    require(all(type(row["id"]) is int for row in rows), "catalog IDs must be integers")
    for row in rows:
        require(isinstance(row.get("name"), str) and row["name"], "missing catalog name")
        require(type(row.get("upgraded")) is bool, "invalid catalog upgrade state")
    return rows


def gallery_ids(root, old, accepted_count):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = read(manifest_path)
    require(manifest.get("schema_version") == 1 and manifest.get("recognizer") == "p_item_rendered_reference_v1", "unsupported accepted P-item gallery")
    info = manifest["reference_gallery"]
    relative = Path(info["path"])
    gallery_path = (root / relative).resolve()
    require(not relative.is_absolute() and gallery_path.is_relative_to(root), "gallery path escapes root")
    require(digest(gallery_path) == info["sha256"], "gallery digest mismatch")
    with np.load(gallery_path, allow_pickle=False) as arrays:
        raw_ids = arrays["p_item_ids"]
        require(raw_ids.ndim == 1 and raw_ids.dtype.kind in "iu", "invalid gallery IDs")
        ids = set(map(int, raw_ids))
        require(len(raw_ids) == len(ids) == info["business_id_count"] and bool(ids), "accepted gallery reference count mismatch")
        require("reference_enabled" not in arrays.files, "P-item runtime does not support masked reference rows")
    require(ids == set(manifest["source"]["business_ids"]), "gallery source ID mismatch")
    require(manifest["source"]["business_id_count"] == len(ids), "gallery source count mismatch")
    coverage = old["gallery_coverage"]
    require(coverage["component"] == "p_item_reference" and coverage["business_id_count"] == len(ids), "classification gallery coverage mismatch")
    require(coverage["manifest_sha256"] == digest(manifest_path), "classification manifest binding mismatch")
    require(coverage["gallery_sha256"] == info["sha256"], "classification gallery binding mismatch")
    require(ids <= set(range(1, accepted_count + 1)), "accepted gallery contains future catalog IDs")
    return ids


def build(*, catalog, accepted_catalog, classification, gallery_root, output, placeholder_output, catalog_revision):
    require(isinstance(catalog_revision, str) and re.fullmatch(r"[0-9a-f]{40}", catalog_revision), "catalog revision must be a frozen commit")
    catalog, accepted_catalog, classification = map(Path, (catalog, accepted_catalog, classification))
    output, placeholder_output = Path(output).resolve(), Path(placeholder_output).resolve()
    gallery_root = Path(gallery_root).resolve()
    inputs = {path.resolve() for path in (catalog, accepted_catalog, classification)}
    require(output != placeholder_output and output not in inputs and placeholder_output not in inputs, "outputs overlap inputs")
    require(not output.is_relative_to(gallery_root) and not placeholder_output.is_relative_to(gallery_root), "missing-image output must stay outside the recognition gallery")
    require(not output.exists() and not placeholder_output.exists(), "output already exists; refusing overwrite or duplicate append")
    current = catalog_rows(catalog)
    accepted = catalog_rows(accepted_catalog)
    accepted_count = len(accepted)
    require(len(current) > accepted_count, "catalog has no new missing-reference rows")
    require(current[:accepted_count] == accepted, "accepted catalog history changed")
    old = read(classification)
    require(old.get("schema_version") == 1 and old.get("classification_rule") == "p_item_plan_and_four_kinds_v1", "unsupported accepted classification")
    require("temporary_missing_references" not in old and "temporary_placeholder_catalog" not in old.get("sources", {}), "duplicate missing-reference append")
    require([row["business_id"] for row in old["items"]] == list(range(1, accepted_count + 1)), "accepted classification coverage mismatch")
    require(digest(accepted_catalog) == old["sources"]["catalog"]["p_items_sha256"], "accepted catalog/classification digest mismatch")
    for row, item in zip(old["items"], accepted, strict=True):
        require(row["name"] == item["name"] and row["upgraded"] is item["upgraded"], "accepted classification identity changed")
        fields = row["source"]
        require(all(fields.get(key) == item[field] for field, key in (
            ("plan", "catalog_plan"), ("mode", "catalog_mode"), ("sourceType", "catalog_source_type"),
            ("rarity", "catalog_rarity"), ("welfare", "catalog_welfare"),
        )), "accepted classification source fields changed")
        require(fields.get("catalog_p_idol_id") == (item["pIdolId"] or None), "accepted classification owner changed")
    represented = gallery_ids(gallery_root, old, accepted_count)
    require(all(type(row["reference_available"]) is bool and row["reference_available"] == (row["business_id"] in represented) for row in old["items"]), "accepted reference availability mismatch")
    new = current[accepted_count:]
    missing_ids = [row["id"] for row in new]
    require(len(new) % 2 == 0, "missing-reference rows must contain complete pairs")
    for normal, upgraded in zip(new[::2], new[1::2], strict=True):
        require(normal["upgraded"] is False and upgraded["upgraded"] is True, "normal/+ pair state order mismatch")
        require(normal["name"].rstrip("+＋") == upgraded["name"].rstrip("+＋") and not normal["name"].endswith(("+", "＋")) and upgraded["name"].endswith(("+", "＋")), "normal/+ pair name mismatch")
        require(all(normal[key] == upgraded[key] for key in ("plan", "rarity", "sourceType", "mode", "pIdolId", "welfare")), "normal/+ pair source mismatch")
        require(normal["sourceType"] == "pIdol" and normal["mode"] == "stage" and bool(normal["pIdolId"]), "only P-idol stage gaps are authorized")
        require(normal["plan"] in old["facets"]["plan"] and normal["rarity"] in {"N", "R", "SR", "SSR"}, "unsupported temporary P-item facets")
    result = deepcopy(old)
    for item in new:
        bid = item["id"]
        result["items"].append({
            "business_id": bid, "name": item["name"], "upgraded": item["upgraded"],
            "plan": item["plan"], "kind": "p_idol", "reference_available": False,
            "paired_business_id": bid - 1 if item["upgraded"] else bid + 1,
            "source": {
                "catalog_plan": item["plan"], "catalog_mode": item["mode"], "catalog_source_type": item["sourceType"],
                "catalog_p_idol_id": item["pIdolId"] or None, "catalog_rarity": item["rarity"], "catalog_welfare": item["welfare"],
                "plan_basis": "catalog_plan_for_temporary_missing_reference", "kind_basis": "catalog_source_for_temporary_missing_reference",
                "official": None, "official_mapping": "catalog_only_temporary_missing_reference",
                "temporary_missing_reference": {"rule_version": RULE, "reference_enabled": False, "reason": "VERIFIED_IDENTITY_AND_UI_PENDING"},
            },
        })
    result.update(revision=f"task095-temporary-missing-p-items-{len(current)}-v1", status=STATUS)
    result["temporary_missing_references"] = {"rule_version": RULE, "business_ids": missing_ids,
        "placeholder_image": PLACEHOLDER_IMAGE, "reference_enabled": False, "status": STATUS}
    result["sources"]["temporary_placeholder_catalog"] = {
        "repository": "https://github.com/gakumas-tools/gakumas-tools", "path": "packages/gakumas-data/json/p_items.json",
        "revision": catalog_revision, "sha256": digest(catalog),
    }
    result["indexes"] = {facet: {value: [row["business_id"] for row in result["items"] if row[facet] == value] for value in values}
                         for facet, values in result["facets"].items()}
    result["counts"] = {facet: {value: len(ids) for value, ids in groups.items()} for facet, groups in result["indexes"].items()}
    result["official_unmapped_business_ids"] = [row["business_id"] for row in result["items"] if row["source"]["official"] is None]
    result["gallery_coverage"]["without_reference_business_ids"] = sorted(set(range(1, len(current) + 1)) - represented)
    require(result["items"][:accepted_count] == old["items"], "old classification records changed")
    require({key: value for key, value in result["sources"].items() if key != "temporary_placeholder_catalog"} == old["sources"], "accepted source fields changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    placeholder_output.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (255, 255, 255)).save(placeholder_output, format="PNG", compress_level=9)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "accepted-catalog", "classification", "gallery-root", "output", "placeholder-output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--catalog-revision", required=True)
    args = parser.parse_args()
    result = build(**vars(args))
    print(json.dumps({"status": result["status"], "items": len(result["items"]), "active_references": result["gallery_coverage"]["business_id_count"],
                      "missing_business_ids": result["temporary_missing_references"]["business_ids"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
