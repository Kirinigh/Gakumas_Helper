"""Append inactive, user-authorized card placeholders without replacing references.

The generated rows complete catalog coverage but cannot participate in image
recognition. They are neutral local artwork, never official game UI or master
identities. This tool writes a new candidate directory and never promotes it.
"""

from __future__ import annotations

import re
import sys
import json
import shutil
import hashlib
import argparse
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from card_selection.embedding import OnnxCardEmbedder, extract_upgrade_marker

RULE = "temporary-placeholder-reference-v1"
STATUS = "USER_AUTHORIZED_TEMPORARY_PLACEHOLDERS_REAL_SAMPLES_PENDING"
ORIGINS = ("originIdolCardId", "originSupportCardId", "originPrimaStellaIdolCardId", "originCharacterId")
PLANS = {"free": "Common", "sense": "Plan1", "logic": "Plan2", "anomaly": "Plan3"}
RARITIES = {"N": "N", "R": "R", "SR": "Sr", "SSR": "Ssr", "L": "Legend"}
EMBEDDING_KEYS = {"embeddings", "class_names", "card_ids", "visual_group_ids", "upgrade_counts", "upgrade_markers"}
REFERENCE_KEYS = {"business_ids", "visual_group_ids", "reference_names", "top_coarse", "green_masks"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value, *, newline="\n"):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=newline)


def inside(root, value):
    require(isinstance(value, str) and value, "missing component path")
    relative = Path(value)
    require(not relative.is_absolute(), "component path must be relative")
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root.resolve()), "component path escapes its root")
    return resolved


def catalog_rows(path):
    rows = read(path)
    require(isinstance(rows, list) and rows, "catalog must be a nonempty list")
    ids = [r.get("id") for r in rows]
    require(all(type(i) is int and i > 0 for i in ids), "invalid catalog ID")
    require(ids == list(range(1, len(rows) + 1)), "catalog IDs must be contiguous and ordered")
    for row in rows:
        require(isinstance(row.get("name"), str) and row["name"], "invalid catalog name")
        require(type(row.get("upgraded")) is bool, "invalid catalog upgrade state")
        require(row.get("plan") in PLANS and row.get("rarity") in {*RARITIES, "T"}, "invalid catalog facets")
        require(row.get("sourceType") in {"default", "produce", "pIdol", "support"}, "invalid catalog source")
    return rows


def new_families(rows):
    require(len(rows) % 2 == 0 and rows, "placeholder extension must contain normal/+ pairs")
    result = {}
    for normal, upgraded in zip(rows[::2], rows[1::2], strict=True):
        require(not normal["upgraded"] and upgraded["upgraded"], "normal/+ pair order mismatch")
        require(normal["name"].rstrip("+＋") == upgraded["name"].rstrip("+＋"), "normal/+ name mismatch")
        require(all(normal[k] == upgraded[k] for k in ("plan", "rarity", "sourceType")), "normal/+ facets mismatch")
        result[normal["id"]] = result[upgraded["id"]] = f"provisional:{normal['id']}"
    return result


def placeholder_image():
    image = Image.new("RGB", (130, 130), (224, 224, 224))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 124, 124), outline=(96, 96, 96), width=2)
    font = ImageFont.load_default(size=13)
    draw.text((65, 56), "TEMPORARY", anchor="mm", fill=(64, 64, 64), font=font)
    draw.text((65, 77), "NO GAME ART", anchor="mm", fill=(64, 64, 64), font=ImageFont.load_default(size=10))
    return image


def load_component(root, accepted_ids, *, embedding):
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    require(manifest.get("schema_version") == 1, "input must be an active schema-1 component; duplicate append forbidden")
    runtime = manifest.get("runtime", {})
    require(not any(k.startswith("temporary_") for k in runtime), "existing temporary declaration is not appendable")
    path = inside(root, manifest["gallery"]["path"])
    require(sha(path) == manifest["gallery"]["sha256"], "input gallery digest mismatch")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    require(set(arrays) == (EMBEDDING_KEYS if embedding else REFERENCE_KEYS), "unsupported or already extended input arrays")
    ids = arrays["card_ids"] if embedding else arrays["business_ids"]
    count = len(ids)
    require(all(v.ndim >= 1 and len(v) == count for v in arrays.values()), "input row arrays disagree")
    require(set(map(int, ids)) == set(accepted_ids), "input component/classification coverage mismatch")
    names = arrays["class_names"] if embedding else arrays["reference_names"]
    require(len(set(map(str, names))) == count, "input reference names are not unique")
    groups_by_id = {}
    for bid, group in zip(ids, arrays["visual_group_ids"], strict=True):
        require(groups_by_id.setdefault(int(bid), str(group)) == str(group), "input family mapping is inconsistent")
    if embedding:
        classes = inside(root, manifest["gallery"]["class_table_path"])
        require(sha(classes) == manifest["gallery"]["class_table_sha256"], "input class table digest mismatch")
        require(read(classes) == names.tolist(), "input class table order mismatch")
        model = inside(root, manifest["model"]["path"])
        require(sha(model) == manifest["model"]["sha256"], "input model digest mismatch")
        require(arrays["embeddings"].shape == (count, 128), "invalid embedding dimensions")
        require(arrays["upgrade_markers"].shape == (count, 24, 24, 3), "invalid marker dimensions")
    else:
        require(arrays["top_coarse"].shape == (count, 8, 16, 3), "invalid card reference dimensions")
        require(arrays["green_masks"].shape == (count, 96, 96), "invalid green mask dimensions")
    return root, manifest, arrays


def append_arrays(arrays, rows, groups, image, *, vector=None):
    """Keep every old row and append only disabled placeholder references."""
    embedding = "embeddings" in arrays
    ids = [row["id"] for row in rows]
    old_ids = arrays["card_ids"] if embedding else arrays["business_ids"]
    require(not set(ids) & set(map(int, old_ids)), "placeholder IDs already exist")
    require("reference_enabled" not in arrays, "duplicate placeholder append")
    count = len(ids)
    extra = {"visual_group_ids": np.asarray([groups[i] for i in ids])}
    if embedding:
        require(vector is not None and np.asarray(vector).shape == (128,), "placeholder vector dimension mismatch")
        vector = np.asarray(vector, dtype=np.float32)
        require(np.isfinite(vector).all() and abs(float(np.linalg.norm(vector)) - 1.0) < 1e-3, "invalid normalized placeholder vector")
        extra.update(
            embeddings=np.repeat(vector[None, :], count, axis=0),
            class_names=np.asarray([f"{i}_temp" for i in ids]),
            card_ids=np.asarray(list(map(str, ids))),
            upgrade_counts=np.asarray([int(row["upgraded"]) for row in rows], dtype=arrays["upgrade_counts"].dtype),
            upgrade_markers=np.repeat(extract_upgrade_marker(image)[None, :], count, axis=0),
        )
    else:
        bgr = np.asarray(image)[:, :, ::-1].copy()
        normalized = cv2.resize(bgr, (96, 96), interpolation=cv2.INTER_AREA)
        coarse = cv2.resize(normalized[:48], (16, 8), interpolation=cv2.INTER_AREA)
        extra.update(
            business_ids=np.asarray(ids, dtype=arrays["business_ids"].dtype),
            reference_names=np.asarray([f"T{i:04d}" for i in ids]),
            top_coarse=np.repeat(coarse[None, :], count, axis=0),
            green_masks=np.zeros((count, 96, 96), dtype=arrays["green_masks"].dtype),
        )
    result = {}
    for key, value in arrays.items():
        require(key in extra, f"unsupported row array {key}")
        appended = extra[key].astype(value.dtype, copy=False)
        # Fixed-width string arrays may not silently truncate identity fields.
        if value.dtype.kind in "US":
            require(appended.tolist() == extra[key].tolist(), f"new {key} exceeds inherited string width")
        result[key] = np.concatenate((value, appended), axis=0)
        require(np.array_equal(result[key][:len(value)], value), f"inherited {key} changed")
    result["reference_enabled"] = np.concatenate((np.ones(len(old_ids), dtype=np.bool_), np.zeros(count, dtype=np.bool_)))
    return result


def candidate_component(source, output, rows, groups, image, image_sha, catalog_source, *, embedder_factory=OnnxCardEmbedder):
    root, original, arrays = source
    manifest = deepcopy(original)
    newline = "\r\n" if b"\r\n" in (root / "manifest.json").read_bytes() else "\n"
    embedding = "embeddings" in arrays
    vector = None
    if embedding:
        embedder = embedder_factory(inside(root, original["model"]["path"]), expected_model_sha256=original["model"]["sha256"], source_color_order="RGB")
        # Reuse the runtime encoder preprocessing for this neutral rendered image.
        vector = embedder.embed(np.asarray(image), (0, 0, image.width, image.height))
    extended = append_arrays(arrays, rows, groups, image, vector=vector)
    output.mkdir()
    image_path = output / "temporary-card.png"
    image.save(image_path, format="PNG", compress_level=9)
    require(sha(image_path) == image_sha, "placeholder image serialization changed")
    gallery = inside(output, manifest["gallery"]["path"])
    gallery.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(gallery, **extended)
    manifest["gallery"].update(sha256=sha(gallery), bytes=gallery.stat().st_size)
    if embedding:
        model = inside(output, manifest["model"]["path"])
        model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(root, manifest["model"]["path"]), model)
        classes = inside(output, manifest["gallery"]["class_table_path"])
        classes.parent.mkdir(parents=True, exist_ok=True)
        classes_newline = "\r\n" if b"\r\n" in inside(root, original["gallery"]["class_table_path"]).read_bytes() else "\n"
        write(classes, extended["class_names"].tolist(), newline=classes_newline)
        manifest["gallery"].update(class_table_sha256=sha(classes), class_table_bytes=classes.stat().st_size)
    all_ids = sorted(set(map(int, extended["card_ids"] if embedding else extended["business_ids"])))
    ids = [row["id"] for row in rows]
    manifest["schema_version"] = 2
    manifest.setdefault("runtime", {}).update(temporary_reference_rule_version=RULE, temporary_placeholder_business_ids=ids)
    source_metadata = manifest.setdefault("source", {})
    inherited_domain = source_metadata.get("source_domain")
    source_metadata["inherited_source_domain"] = inherited_domain
    source_metadata["source_domain"] = (inherited_domain or "INHERITED_REFERENCES") + "_WITH_EXCLUDED_TEMPORARY_PLACEHOLDERS"
    source_metadata.update(
        business_card_ids=all_ids,
        business_card_id_count=len(all_ids),
        business_card_id_min=min(all_ids),
        business_card_id_max=max(all_ids),
        business_card_id_set_sha256=hashlib.sha256(json.dumps(all_ids, separators=(",", ":")).encode("ascii")).hexdigest().upper(),
        temporary_placeholders={
            "status": STATUS,
            "source_kind": "USER_AUTHORIZED_NEUTRAL_PLACEHOLDER_ART",
            "historical_provenance_scope": "inherited enabled rows only",
            "catalog": catalog_source,
            "image_path": "temporary-card.png",
            "image_sha256": image_sha,
            "business_ids": ids,
            "references": [{"business_id": row["id"], "name": row["name"], "upgraded": row["upgraded"], "visual_group_id": groups[row["id"]], "reference_enabled": False} for row in rows],
            "replacement_required": "replace placeholders with verified actual card art and matching source identities; rebuild and validate after release",
        },
    )
    if embedding:
        source_metadata.update(image_class_count=len(extended["card_ids"]), visual_identity_count=len(set(map(str, extended["visual_group_ids"]))))
    else:
        source_metadata["reference_file_count"] = len(extended["business_ids"])
        manifest["gallery"].update(business_card_id_count=len(all_ids), visual_group_count=len(set(map(str, extended["visual_group_ids"]))))
    for field in ("promotion", "evaluation"):
        manifest.pop(field, None)
    manifest["build"] = {
        "tool_contract": "task095-temporary-placeholder-append-v1",
        "mode": "GALLERY_ONLY_NO_TRAINING",
        "base_manifest_sha256": sha(root / "manifest.json"),
        "base_gallery_sha256": original["gallery"]["sha256"],
        "preserved_rows": len(next(iter(arrays.values()))),
        "appended_rows": len(rows),
        "enabled_rows": int(extended["reference_enabled"].sum()),
    }
    manifest["validation"] = {
        "state": STATUS,
        "preserved_active_reference_rows": "ARRAY_PREFIX_IDENTICAL",
        "placeholder_recognition_acceptance": "NOT_APPLICABLE_DISABLED",
        "pending": ["verified_card_art_and_identities", "independent_live_recognition", "full_read_latency"],
    }
    manifest["production_handoff"] = {"status": STATUS, "reason": "user authorized inactive catalog placeholders; no new recognition or real-sample acceptance claimed"}
    write(output / "manifest.json", manifest, newline=newline)
    return manifest, extended


def labels(card):
    rarity = "N" if card["rarity"] == "T" and card.get("type") == "trouble" else card["rarity"]
    require(rarity in RARITIES, "invalid normalized card rarity")
    source = card["sourceType"]
    kind = "other" if rarity == "L" else "p_idol" if source == "pIdol" else "support" if source == "support" else "basic" if "基本" in card["name"] else "other"
    return {"plan": card["plan"], "rarity": rarity, "kind": kind}


def classification(old, rows, groups, components, catalog_source, image_sha, *, decoded=None, decoded_sha=None):
    result = deepcopy(old)
    official_by_name = {}
    for row in decoded or []:
        if row.get("upgradeCount") in {0, 1} and "identity_source" not in row:
            key = (row["name"].rstrip("+＋"), row["upgradeCount"])
            require(key not in official_by_name, "ambiguous decoded official identity")
            official_by_name[key] = row
    for card in rows:
        official = official_by_name.get((card["name"].rstrip("+＋"), int(card["upgraded"])))
        facets = labels(card)
        if official:
            require(official["planType"] == "ProducePlanType_" + PLANS[facets["plan"]], "decoded official plan conflict")
            require(official["rarity"] == "ProduceCardRarity_" + RARITIES[facets["rarity"]], "decoded official rarity conflict")
            require(isinstance(official.get("id"), str) and official["id"] and isinstance(official.get("assetId"), str) and official["assetId"], "invalid decoded official identity")
        result["cards"].append({
            "business_id": card["id"], "name": card["name"], "upgraded": card["upgraded"],
            "internal_id": official["id"] if official else None,
            "asset_id": official["assetId"] if official else None,
            **facets,
            "source": {
                "catalog_source_type": card["sourceType"], "catalog_p_idol_id": card.get("pIdolId") or None,
                "catalog_rarity": card["rarity"], "card_type": card["type"],
                "official_plan": official["planType"] if official else None,
                "official_rarity": official["rarity"] if official else None,
                **{key: official.get(key, "") if official else None for key in ORIGINS},
                "kind_basis": "user_rule_all_legend_other" if facets["rarity"] == "L" else "catalog_source_for_temporary_placeholder",
                "temporary_placeholder": {
                    "status": STATUS, "reference_enabled": False, "rule_version": RULE,
                    "provisional_family_id": groups[card["id"]], "image_sha256": image_sha,
                    "catalog": catalog_source,
                    "official_identity_status": "FROZEN_DECODED_MASTER" if official else "UNKNOWN_PENDING",
                    "decoded_master_sha256": decoded_sha if official else None,
                    "gameplay_effects_status": "NOT_VALIDATED_BY_PLACEHOLDER_UPDATE",
                },
            },
        })
    result["revision"] = "task095-temporary-placeholder-" + str(len(result["cards"])) + "-v1"
    result["status"] = STATUS
    result.setdefault("sources", {})["temporary_placeholder_catalog"] = catalog_source
    result["indexes"] = {facet: {value: [r["business_id"] for r in result["cards"] if r[facet] == value] for value in values} for facet, values in result["facets"].items()}
    result["counts"] = {facet: {value: len(ids) for value, ids in groups.items()} for facet, groups in result["indexes"].items()}
    result["gallery_coverage"] = components
    require(result["cards"][:len(old["cards"])] == old["cards"], "old classification records changed")
    return result


def build(*, catalog, accepted_catalog, classification_path, base_root, arena_root, badge_root, output, catalog_revision, decoded_master=None, embedder_factory=OnnxCardEmbedder):
    require(re.fullmatch(r"[0-9a-f]{40}", catalog_revision), "catalog revision must be a frozen commit")
    catalog, accepted_catalog, classification_path = map(Path, (catalog, accepted_catalog, classification_path))
    output = Path(output).resolve()
    require(not output.exists(), "output already exists; refusing overwrite or duplicate append")
    inputs = [catalog, accepted_catalog, classification_path, Path(base_root), Path(arena_root), Path(badge_root)]
    if decoded_master:
        inputs.append(Path(decoded_master))
    require(all(not path.resolve().is_relative_to(output) and not output.is_relative_to(path.resolve()) for path in inputs), "output overlaps an input")
    current, accepted = catalog_rows(catalog), catalog_rows(accepted_catalog)
    require(current[:len(accepted)] == accepted, "accepted catalog history changed")
    new = current[len(accepted):]
    groups = new_families(new)
    old = read(classification_path)
    accepted_ids = list(range(1, len(accepted) + 1))
    require([r["business_id"] for r in old["cards"]] == accepted_ids, "accepted classification coverage mismatch")
    require(old["sources"]["catalog_sha256"] == sha(accepted_catalog), "accepted catalog/classification binding mismatch")
    for index_row, catalog_row in zip(old["cards"], accepted, strict=True):
        require(index_row["name"] == catalog_row["name"] and index_row["upgraded"] == catalog_row["upgraded"], "accepted classification identity changed")
        require(all(index_row[k] == v for k, v in labels(catalog_row).items()), "accepted classification facets changed")
    sources = [load_component(root, accepted_ids, embedding=embedding) for root, embedding in ((base_root, True), (arena_root, True), (badge_root, False))]
    require(len({str(source[0]) for source in sources}) == 3, "duplicate input component roots")
    decoded = read(decoded_master) if decoded_master else None
    require(decoded is None or isinstance(decoded, list), "decoded master must be a frozen row list")
    catalog_source = {"repository": "https://github.com/gakumas-tools/gakumas-tools", "revision": catalog_revision, "sha256": sha(catalog)}
    image = placeholder_image()
    output.mkdir(parents=True)
    image_path = output / "temporary-card.png"
    image.save(image_path, format="PNG", compress_level=9)
    components = []
    for name, source in zip(("skill-base", "skill-arena", "badge-reference"), sources, strict=True):
        candidate_root = output / name
        manifest, arrays = candidate_component(source, candidate_root, new, groups, image, sha(image_path), catalog_source, embedder_factory=embedder_factory)
        ids = arrays["card_ids"] if "card_ids" in arrays else arrays["business_ids"]
        components.append({"component": name, "manifest_sha256": sha(candidate_root / "manifest.json"), "gallery_sha256": manifest["gallery"]["sha256"], "row_count": len(ids), "business_id_count": len(current), "enabled_business_id_count": len(accepted), "temporary_placeholder_business_id_count": len(new), "join_key": "business_id"})
    result = classification(old, new, groups, components, catalog_source, sha(image_path), decoded=decoded, decoded_sha=sha(decoded_master) if decoded_master else None)
    write(output / "skill_card_classification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "accepted-catalog", "classification-path", "base-root", "arena-root", "badge-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--catalog-revision", required=True)
    parser.add_argument("--decoded-master", type=Path)
    args = parser.parse_args()
    result = build(**vars(args))
    print(json.dumps({"status": result["status"], "card_count": len(result["cards"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
