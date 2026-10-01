"""Offline append-only identity joins when a decoded master mirror is late.

Catalog labels and public internal IDs are explicitly NOT official master fields.
Every new identity also needs an official asset object and a frozen UI reference.
"""
from __future__ import annotations

import re
import json
import hashlib
from pathlib import Path

import yaml

CONTRACT = "catalog-public-id-official-art-ui-v1"
PLANS = {"free": "Common", "sense": "Plan1", "logic": "Plan2", "anomaly": "Plan3"}
RARITIES = {"SSR": "Ssr", "SR": "Sr", "R": "R", "N": "N", "L": "Legend"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_bound(root, record):
    require(set(record) == {"path", "sha256"}, "invalid joint source binding")
    path = (root / record["path"]).resolve()
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest().upper() == record["sha256"], "joint source digest mismatch")
    return yaml.load(raw, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) if path.suffix == ".yaml" else json.loads(raw)


def load(path):
    path = Path(path)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("source_contract") != CONTRACT:
        return value
    require(set(value) == {"source_contract", "component", "sources", "bindings"}, "invalid joint source envelope")
    kind = value["component"]
    require(kind in {"skill_card", "p_item"}, "invalid joint component")
    sources = value["sources"]
    require(set(sources) == {"base_master", "catalog", "public_ids", "official_manifest", "ui_manifest"}, "incomplete joint evidence")
    inputs = {k: read_bound(path.parent, v) for k, v in sources.items()}
    base = inputs["base_master"]
    require(isinstance(base, list), "joint base must be the frozen master rows")
    catalog = {r["id"]: r for r in inputs["catalog"]}
    require(len(catalog) == len(inputs["catalog"]), "duplicate catalog identity")
    public = inputs["public_ids"]
    require(public.get("source_kind") == "translated_master_fields_not_full_official_master", "public identity source kind mismatch")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", public.get("revision", ""))), "public identity revision must be pinned")
    public_rows = public["skill_rows" if kind == "skill_card" else "p_item_rows"]
    public_keys = {(r["id"], r.get("upgradeCount", 0)) for r in public_rows}
    assets = {r["name"] for r in inputs["official_manifest"]["assetBundleList"]}
    ui = inputs["ui_manifest"]["references"]
    # Reuse the same source and identity validators as the gallery builders.
    from tools.published_ui_reference import validate_ui_source_origin, validate_published_ui_references

    validate_published_ui_references(ui, expected_ids=list(map(int, ui)), catalog_rows=list(catalog.values()), component=kind)
    validate_ui_source_origin(inputs["ui_manifest"]["source_type"], ui)
    keys = {(r["id"], r.get("upgradeCount", 0)) for r in base}
    names = {r["name"].replace("＋", "+") for r in base}
    bindings = value["bindings"]
    ids = [r["business_id"] for r in bindings]
    require(ids == sorted(set(ids)) and ids, "joint IDs must be unique and sorted")
    result = list(base)
    for binding in bindings:
        require(set(binding) == {"business_id", "internal_id", "asset_id"}, "invalid joint binding")
        bid, internal, asset = binding["business_id"], binding["internal_id"], binding["asset_id"]
        card = catalog[bid]
        upgrade = int(card["upgraded"])
        key = (internal, upgrade if kind == "skill_card" else 0)
        require(key in public_keys and key not in keys, "new internal identity absent or already in base")
        require(card["name"].replace("＋", "+") not in names, "joint source cannot replace an existing master row")
        require(asset in assets and not any(a.startswith(asset + "-") for a in assets), "joint art must be one official unsuffixed object")
        if kind == "skill_card":
            match = re.fullmatch(r"p_card-(\d{2})-((?:ido|sup)-[0-9]_[0-9]+)", internal)
            require(match is not None and asset == "img_general_skillcard_" + match[2], "skill internal ID/art mismatch")
        else:
            match = re.fullmatch(r"pitem_(\d{2})-([0-9]-[0-9]+)-([01])", internal)
            require(match is not None and asset == "img_general_pitem_" + match[2] and int(match[3]) == upgrade, "P internal ID/art/upgrade mismatch")
        require(int(match[1]) == list(PLANS).index(card["plan"]), "internal ID/catalog plan conflict")
        ref = ui.get(str(bid))
        require(ref is not None and ref["name"] == card["name"].rstrip("+＋") and ref["upgraded"] == card["upgraded"], "joint identity requires exact full UI")
        evidence = {"contract": CONTRACT, "business_id": bid, "field_basis": "catalog_labels_public_internal_id_official_asset_name", "sources": {k: v["sha256"] for k, v in sources.items()}}
        row = {"id": internal, "name": card["name"], "assetId": asset, "planType": "ProducePlanType_" + PLANS[card["plan"]], "identity_source": evidence}
        if kind == "skill_card":
            row.update(upgradeCount=upgrade, isCharacterAsset=False, rarity="ProduceCardRarity_" + RARITIES[card["rarity"]])
        else:
            row.update(isUpgraded=card["upgraded"], isExamEffect=card["mode"] == "stage", rarity="ProduceItemRarity_" + RARITIES[card["rarity"]])
        result.append(row)
        keys.add(key)
        names.add(card["name"].replace("＋", "+"))
    return result
