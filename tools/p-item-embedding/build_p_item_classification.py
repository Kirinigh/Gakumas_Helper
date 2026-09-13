"""Build source-backed P-item facets without changing runtime recognition."""

from __future__ import annotations

import json
import hashlib
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np

FACETS = {
    "plan": {"free": "共通", "sense": "センス", "logic": "ロジック", "anomaly": "アノマリー"},
    "kind": {
        "p_idol": "P 偶像固有",
        "support_inheritable": "可继承支援卡道具",
        "support_non_inheritable": "不可继承支援卡道具",
        "other": "其他",
    },
}
PLANS = dict(zip(("Common", "Plan1", "Plan2", "Plan3"), FACETS["plan"], strict=True))
# These accepted name aliases are also checked against the frozen visual crosswalk.
ALIASES = {
    174: "アイドルパワー測定機",
    242: "「Pっち」",
    243: "「Pっち」+",
    310: "ビシッバシッ竹刀",
    380: "ぴったしコレクション",
    383: "そっくりワンワン",
    419: "夜空を従える花",
    420: "夜空を従える花+",
}
PLAN_DIFFERENCES = {115: ("logic", "free"), 142: ("free", "logic"), 406: ("sense", "free"), 407: ("logic", "free"), 408: ("anomaly", "free")}
SPECIAL_POUCHES = {
    i: (f"ポーチ（{color}）", plan)
    for i, (plan, color) in enumerate(((plan, color) for plan in ("sense", "logic", "anomaly") for color in ("赤", "緑", "黄")), 431)
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def unique(rows, key):
    result = {}
    for row in rows:
        value = key(row)
        require(value not in result, f"duplicate identity: {value}")
        result[value] = row
    return result


def normalized(name):
    return unicodedata.normalize("NFKC", name).replace(" ", "").removesuffix("+")


def classify(item, official):
    source, mode, plan = item["sourceType"], item["mode"], item["plan"]
    require(source in {"pIdol", "support", "produce"}, "unknown source type")
    require(mode in {"stage", "produce"}, "unknown mode")
    require(plan in FACETS["plan"], "unknown plan")
    require(type(item["upgraded"]) is bool, "invalid upgraded state")
    plan_basis = "catalog_special_item"
    if official is not None:
        official_plan = PLANS[official["planType"].removeprefix("ProducePlanType_")]
        require(type(official["isExamEffect"]) is bool, "invalid official exam flag")
        require(official["isExamEffect"] == (mode == "stage"), f"mode conflict: {item['id']}")
        require(official["isUpgraded"] == item["upgraded"], "upgrade conflict")
        require(not official["originIdolCardId"] or source == "pIdol", "idol source conflict")
        require(not official["originSupportCardId"] or source == "support", "support source conflict")
        if plan != official_plan:
            require(PLAN_DIFFERENCES.get(item["id"]) == (plan, official_plan), "unreviewed plan conflict")
        if item["id"] in {406, 407, 408}:
            plan_basis = "existing_business_plan_split"
        else:
            plan = official_plan
            plan_basis = "official_plan"
    else:
        require(
            SPECIAL_POUCHES.get(item["id"]) == (item["name"], plan) and source == "produce" and mode == "produce" and not item["upgraded"],
            "unresolved official mapping",
        )
    kind = "other"
    if source == "pIdol":
        kind = "p_idol"
    elif source == "support":
        require(official is not None, "support inheritance requires official evidence")
        kind = "support_inheritable" if official["isExamEffect"] else "support_non_inheritable"
    return {"plan": plan, "kind": kind}, plan_basis


def select_ids(index, **filters):
    """OR within each facet, AND across facets; free must be explicitly selected."""
    selected = {row["business_id"] for row in index["items"]}
    for facet, values in filters.items():
        require(facet in FACETS, f"unknown facet: {facet}")
        values = [values] if isinstance(values, str) else list(values)
        require(all(v in FACETS[facet] for v in values), f"unknown {facet} filter")
        selected &= set().union(*(index["indexes"][facet][v] for v in values))
    return sorted(selected)


def build(catalog_path, master_path, evidence_path, receipt_path, dataset_path, gallery_root, *, revision):
    require(bool(revision.strip()), "revision is required")
    evidence = read(evidence_path)["catalog"]
    require(digest(catalog_path) == evidence["p_items_sha256"], "catalog digest mismatch")
    receipts = [r for r in read(receipt_path) if r["file"] == "ProduceItem.yaml"]
    require(len(receipts) == 1 and digest(master_path) == receipts[0]["sha256"], "master digest mismatch")
    dataset = read(dataset_path)
    crosswalk_path = Path(dataset_path).parent / dataset["mapping"]["crosswalk_path"]
    require(digest(crosswalk_path) == dataset["mapping"]["crosswalk_sha256"], "crosswalk digest mismatch")
    crosswalk = unique(read(crosswalk_path), lambda r: r["business_id"])
    catalog = unique(read(catalog_path), lambda r: r["id"])
    require(all(type(i) is int and i > 0 for i in catalog), "invalid business ID")
    master = yaml.load(Path(master_path).read_text(encoding="utf-8-sig"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    unique(master, lambda r: r["id"])
    by_name = defaultdict(list)
    for row in master:
        by_name[(normalized(row["name"]), row["isUpgraded"])].append(row)
    manifest = read(Path(gallery_root) / "manifest.json")
    gallery_info = manifest["reference_gallery"]
    gallery_path = Path(gallery_root) / gallery_info["path"]
    require(digest(gallery_path) == gallery_info["sha256"], "gallery digest mismatch")
    with np.load(gallery_path, allow_pickle=False) as arrays:
        raw_ids = arrays["p_item_ids"]
        require(raw_ids.ndim == 1 and raw_ids.dtype.kind in "iu", "invalid gallery IDs")
        gallery_ids = set(map(int, raw_ids))
        require(len(raw_ids) == len(gallery_ids) == gallery_info["business_id_count"], "duplicate gallery ID or count mismatch")
    require(gallery_ids <= set(catalog), "gallery ID absent from catalog")
    require(gallery_ids == set(manifest["source"]["business_ids"]), "gallery source IDs mismatch")
    required_ids = {i for i, r in catalog.items() if r["sourceType"] in {"pIdol", "support"} and r["mode"] == "stage"}
    require(required_ids <= gallery_ids, "missing existing arena reference")
    items, pairs = [], defaultdict(dict)
    for bid, item in sorted(catalog.items()):
        matches = by_name[(normalized(ALIASES.get(bid, item["name"])), item["upgraded"])]
        require(len(matches) <= 1, f"ambiguous official mapping: {bid}")
        official = matches[0] if matches else None
        if bid in crosswalk:
            mapping = crosswalk[bid]
            require(mapping["name"] == item["name"] and mapping["upgraded"] == item["upgraded"], "crosswalk identity mismatch")
            require(official is not None and official["assetId"] == mapping["asset_name"], f"crosswalk art mismatch: {bid}")
        if bid in ALIASES:
            require(bid in crosswalk, "alias requires accepted visual crosswalk")
        labels, basis = classify(item, official)
        source = {
            "catalog_plan": item["plan"],
            "catalog_mode": item["mode"],
            "catalog_source_type": item["sourceType"],
            "catalog_p_idol_id": item["pIdolId"] or None,
            "catalog_rarity": item["rarity"],
            "catalog_welfare": item["welfare"],
            "plan_basis": basis,
            "kind_basis": "catalog_source_and_official_exam_flag" if item["sourceType"] == "support" else "catalog_source",
            "official_mapping": "catalog_only_special_item"
            if official is None
            else "accepted_visual_alias"
            if bid in ALIASES
            else "unique_normalized_name",
        }
        source["official"] = (
            None
            if official is None
            else {
                k: official[k]
                for k in (
                    "id",
                    "assetId",
                    "name",
                    "planType",
                    "isExamEffect",
                    "isUpgraded",
                    "rarity",
                    "originIdolCardId",
                    "originSupportCardId",
                    "libraryHidden",
                    "isChallenge",
                    "isHighScoreRush",
                    "isResearch",
                    "isEasy",
                    "isLimited",
                )
            }
        )
        row = {
            "business_id": bid,
            "name": item["name"],
            "upgraded": item["upgraded"],
            **labels,
            "reference_available": bid in gallery_ids,
            "source": source,
        }
        items.append(row)
        pair_key = (normalized(item["name"]), item["sourceType"], item["pIdolId"], item["plan"])
        require(item["upgraded"] not in pairs[pair_key], f"duplicate pair state: {bid}")
        pairs[pair_key][item["upgraded"]] = row
    for states in pairs.values():
        if True in states:
            require(False in states, "missing normal partner")
            require(all(states[True][f] == states[False][f] for f in FACETS), "normal/+ classification mismatch")
        for state, row in states.items():
            row["paired_business_id"] = states[not state]["business_id"] if (not state) in states else None
    indexes = {f: {v: [r["business_id"] for r in items if r[f] == v] for v in values} for f, values in FACETS.items()}
    return {
        "schema_version": 1,
        "revision": revision,
        "status": "OFFLINE_VALIDATED",
        "classification_rule": "p_item_plan_and_four_kinds_v1",
        "runtime_filtering_enabled": False,
        "sources": {
            "catalog": {k: evidence[k] for k in ("repository", "path", "revision", "p_items_sha256")},
            "official_master": receipts[0],
            "catalog_evidence_sha256": digest(evidence_path),
            "visual_dataset_manifest_sha256": digest(dataset_path),
            "visual_crosswalk_sha256": digest(crosswalk_path),
        },
        "facets": FACETS,
        "items": items,
        "indexes": indexes,
        "counts": {f: {v: len(ids) for v, ids in groups.items()} for f, groups in indexes.items()},
        "official_unmapped_business_ids": [r["business_id"] for r in items if r["source"]["official"] is None],
        "gallery_coverage": {
            "component": "p_item_reference",
            "join_key": "business_id",
            "business_id_count": len(gallery_ids),
            "manifest_sha256": digest(Path(gallery_root) / "manifest.json"),
            "gallery_sha256": gallery_info["sha256"],
            "without_reference_business_ids": sorted(set(catalog) - gallery_ids),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "master", "evidence", "receipt", "dataset", "gallery-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    result = build(args.catalog, args.master, args.evidence, args.receipt, args.dataset, args.gallery_root, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
