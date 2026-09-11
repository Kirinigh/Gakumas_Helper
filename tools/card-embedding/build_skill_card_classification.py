"""Build offline card facets from pinned master/catalog and accepted galleries.

This index does not enable runtime filtering or change recognition artifacts.
Legend always belongs to other, including idol-associated Legend cards.
"""

from __future__ import annotations

import re
import json
import hashlib
import argparse
from pathlib import Path

import numpy as np

PLANS = dict(
    zip(
        ("Common", "Plan1", "Plan2", "Plan3"),
        ("free", "sense", "logic", "anomaly"),
        strict=True,
    )
)
RARITIES = {"N": "N", "R": "R", "Sr": "SR", "Ssr": "SSR", "Legend": "L"}
FACETS = {
    "plan": {"free": "共通", "sense": "センス", "logic": "ロジック", "anomaly": "アノマリー"},
    "kind": {"p_idol": "Pアイドル固有", "support": "サポートカード固有", "basic": "名前に「基本」を含む", "other": "その他"},
    "rarity": {"L": "レジェンド", "SSR": "SSR", "SR": "SR", "R": "R", "N": "N"},
}
ORIGINS = ("originIdolCardId", "originSupportCardId", "originPrimaStellaIdolCardId", "originCharacterId")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def indexed(rows, key):
    result = {}
    for row in rows:
        value = key(row)
        require(value not in result, f"duplicate identity: {value}")
        result[value] = row
    return result


def classify(card, master):
    plan = PLANS[master["planType"].removeprefix("ProducePlanType_")]
    rarity = RARITIES[master["rarity"].removeprefix("ProduceCardRarity_")]
    require(card["plan"] == plan, f"plan conflict: {card['id']}")
    trouble = card["rarity"] == "T" and card["type"] == "trouble" and rarity == "N"
    require(card["rarity"] == rarity or trouble, f"rarity conflict: {card['id']}")
    source = card["sourceType"]
    require(source in {"default", "produce", "pIdol", "support"}, "unknown card source")
    idol = bool(master.get("originIdolCardId")) or source == "pIdol"
    support = bool(master.get("originSupportCardId")) or source == "support"
    require(not (idol and support), f"conflicting ownership: {card['id']}")
    if rarity == "L":
        kind, basis = "other", "user_rule_all_legend_other"
    elif idol:
        kind, basis = "p_idol", "official_idol_origin" if master.get("originIdolCardId") else "catalog_pIdol_source"
    elif support:
        kind, basis = "support", "official_support_origin" if master.get("originSupportCardId") else "catalog_support_source"
    elif "基本" in card["name"]:
        kind, basis = "basic", "name_contains_基本"
    else:
        kind, basis = "other", "remaining_card"
    return {"plan": plan, "kind": kind, "rarity": rarity}, basis


def select_ids(index, **filters):
    """OR within a facet, AND across facets; unknown values never broaden scope.

    This is exact library filtering. A battle consumer must explicitly include
    free cards when applying a known plan and must establish filter evidence.
    """
    selected = {row["business_id"] for row in index["cards"]}
    for facet, values in filters.items():
        require(facet in FACETS, f"unknown facet: {facet}")
        values = [values] if isinstance(values, str) else list(values)
        require(all(v in FACETS[facet] for v in values), f"unknown {facet} filter")
        allowed = set().union(*(index["indexes"][facet][v] for v in values))
        selected &= allowed
    return sorted(selected)


def build(catalog_path, master_path, dataset_paths, component_roots, *, revision, catalog_revision):
    require(bool(re.fullmatch(r"[0-9a-f]{40}", catalog_revision)), "catalog revision must be pinned")
    require(bool(revision.strip()), "index revision is required")
    catalog = indexed(read(catalog_path), lambda r: r["id"])
    require(all(type(i) is int and i > 0 for i in catalog), "invalid business ID")
    master = indexed(read(master_path), lambda r: (r["id"], r["upgradeCount"]))
    mappings, datasets = {}, []
    for path in dataset_paths:
        manifest = read(path)
        mapping = manifest["mapping"]
        crosswalk = Path(path).parent / mapping["crosswalk_path"]
        require(digest(crosswalk) == mapping["crosswalk_sha256"], "crosswalk digest mismatch")
        for row in read(crosswalk):
            bid = row["business_id"]
            require(bid not in mappings, f"duplicate crosswalk ID: {bid}")
            mappings[bid] = row
        datasets.append({"dataset_id": manifest["dataset_id"], "manifest_sha256": digest(path)})
    require(set(mappings) == set(catalog), "catalog/crosswalk coverage mismatch")
    final_source = read(dataset_paths[-1])["sources"]
    require(final_source["catalog_sha256"] == digest(catalog_path), "catalog differs from frozen dataset")
    require(final_source["decoded_pcard_sha256"] == digest(master_path), "master differs from frozen dataset")
    require(final_source["catalog"]["commit"] == catalog_revision, "catalog revision mismatch")
    cards, pairs = [], {}
    for bid, card in sorted(catalog.items()):
        mapping = mappings[bid]
        key = (mapping["internal_id"], mapping["upgrade_count"])
        official = master[key]
        require(card["name"] == mapping["catalog_name"], f"catalog name mismatch: {bid}")
        require(official["name"] == mapping["api_name"], f"master name mismatch: {bid}")
        require(official["assetId"] == mapping["asset_id"], f"master art mismatch: {bid}")
        require(type(card["upgraded"]) is bool and int(card["upgraded"]) == key[1], "upgrade mismatch")
        labels, basis = classify(card, official)
        require(pairs.setdefault(key[0], labels) == labels, f"normal/+ facets disagree: {bid}")
        cards.append(
            {
                "business_id": bid,
                "name": card["name"],
                "upgraded": card["upgraded"],
                "internal_id": key[0],
                "asset_id": official["assetId"],
                **labels,
                "source": {
                    "catalog_source_type": card["sourceType"],
                    "catalog_p_idol_id": card["pIdolId"] or None,
                    "catalog_rarity": card["rarity"],
                    "card_type": card["type"],
                    "official_plan": official["planType"],
                    "official_rarity": official["rarity"],
                    **{k: official.get(k, "") for k in ORIGINS},
                    "kind_basis": basis,
                },
            }
        )
    galleries = []
    require(len(component_roots) == 3, "require base, arena and badge components")
    require(len({Path(p).resolve() for p in component_roots}) == 3, "duplicate component root")
    for root in component_roots:
        root = Path(root)
        manifest = read(root / "manifest.json")
        info = manifest["gallery"]
        gallery_path = root / info["path"]
        require(digest(gallery_path) == info["sha256"], "gallery digest mismatch")
        with np.load(gallery_path, allow_pickle=False) as arrays:
            ids = arrays["card_ids"] if "card_ids" in arrays else arrays["business_ids"]
            names = arrays["class_names"] if "class_names" in arrays else arrays["reference_names"]
            require(set(map(int, ids)) == set(catalog), "gallery/index coverage mismatch")
            require(len(ids) == len(names) == len(set(map(str, names))), "invalid gallery row mapping")
            galleries.append(
                {
                    "component": root.name,
                    "manifest_sha256": digest(root / "manifest.json"),
                    "gallery_sha256": info["sha256"],
                    "row_count": len(ids),
                    "business_id_count": len(catalog),
                    "join_key": "business_id",
                }
            )
    indexes = {facet: {value: [r["business_id"] for r in cards if r[facet] == value] for value in values} for facet, values in FACETS.items()}
    return {
        "schema_version": 1,
        "revision": revision,
        "status": "OFFLINE_VALIDATED",
        "classification_rule": "game_facets_v1_all_legend_other",
        "runtime_filtering_enabled": False,
        "sources": {
            "catalog_revision": catalog_revision,
            "catalog_sha256": digest(catalog_path),
            "decoded_master_sha256": digest(master_path),
            "datasets": datasets,
        },
        "facets": FACETS,
        "cards": cards,
        "indexes": indexes,
        "counts": {facet: {v: len(ids) for v, ids in groups.items()} for facet, groups in indexes.items()},
        "gallery_coverage": galleries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, action="append", required=True)
    parser.add_argument("--component-root", type=Path, action="append", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--catalog-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.catalog, args.master, args.dataset_manifest, args.component_root, revision=args.revision, catalog_revision=args.catalog_revision
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
