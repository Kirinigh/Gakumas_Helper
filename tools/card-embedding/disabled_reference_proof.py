"""Read an existing disabled-reference component as offline append evidence."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

RULE = "temporary-placeholder-reference-v1"
STATUS = "USER_AUTHORIZED_TEMPORARY_PLACEHOLDERS_REAL_SAMPLES_PENDING"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def _inside(root, relative):
    _require(isinstance(relative, str) and relative, "proof path is missing")
    p = Path(relative)
    _require(not p.is_absolute(), "proof path must be relative")
    resolved = (root / p).resolve()
    _require(resolved.is_relative_to(root), "proof path escapes component")
    return resolved


def load_disabled_reference_proof(root):
    """Verify the existing schema-2 contract, not an arbitrary list of gaps."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    _require(manifest.get("schema_version") == 2, "disabled proof must be schema 2")
    runtime = manifest.get("runtime", {})
    source = manifest.get("source", {})
    declaration = source.get("temporary_placeholders", {})
    ids = runtime.get("temporary_placeholder_business_ids")
    _require(isinstance(ids, list) and ids and all(type(i) is int and i > 0 for i in ids), "invalid disabled proof ID declaration")
    _require(ids == sorted(set(ids)), "disabled proof IDs must be sorted and unique")
    _require(runtime.get("temporary_reference_rule_version") == RULE, "disabled proof rule differs")
    _require(declaration.get("status") == STATUS and declaration.get("business_ids") == ids, "disabled proof source declaration differs")
    _require(manifest.get("production_handoff", {}).get("status") == STATUS, "disabled proof has no promoted placeholder handoff")
    _require(manifest.get("validation", {}).get("state") == STATUS, "disabled proof validation differs")
    _require(manifest.get("build", {}).get("tool_contract") == "task095-temporary-placeholder-append-v1", "disabled proof build contract differs")
    gallery_path = _inside(root, manifest["gallery"]["path"])
    _require(_sha(gallery_path) == manifest["gallery"]["sha256"], "disabled proof gallery digest mismatch")
    with np.load(gallery_path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    _require("reference_enabled" in arrays, "disabled proof flag array is absent")
    raw_ids = arrays.get("card_ids", arrays.get("business_ids"))
    _require(raw_ids is not None and raw_ids.ndim == 1, "disabled proof ID array differs")
    flags = arrays["reference_enabled"]
    _require(flags.dtype == np.bool_ and flags.shape == raw_ids.shape, "disabled proof flag shape/type differs")
    enabled_count = int(flags.sum())
    _require(flags.tolist() == [True] * enabled_count + [False] * len(ids), "disabled proof must retain an enabled prefix and disabled tail")
    _require(all(a.ndim >= 1 and len(a) == len(raw_ids) for a in arrays.values()), "disabled proof row arrays disagree")
    numeric = np.asarray([int(i) for i in raw_ids])
    disabled = set(ids)
    _require(set(numeric[~flags]) == disabled, "disabled proof rows differ from declared IDs")
    _require(not set(numeric[flags]) & disabled, "disabled proof has an enabled declared ID")
    _require(set(numeric) == set(source.get("business_card_ids", [])), "disabled proof scope differs")
    _require(source.get("business_card_id_count") == len(set(numeric)), "disabled proof scope count differs")
    build = manifest["build"]
    _require(build.get("enabled_rows") == enabled_count and build.get("preserved_rows") == enabled_count and build.get("appended_rows") == len(ids), "disabled proof build row counts differ")
    if "embeddings" in arrays:
        classes = _inside(root, manifest["gallery"]["class_table_path"])
        _require(_sha(classes) == manifest["gallery"]["class_table_sha256"], "disabled proof class table digest mismatch")
        _require(json.loads(classes.read_text(encoding="utf-8-sig")) == arrays["class_names"].tolist(), "disabled proof class table order differs")
        model = _inside(root, manifest["model"]["path"])
        _require(_sha(model) == manifest["model"]["sha256"], "disabled proof fixed encoder differs")
    references = declaration.get("references")
    _require(isinstance(references, list) and len(references) == len(ids), "disabled proof reference declarations differ")
    reference_by_id = {r.get("business_id"): r for r in references}
    _require(set(reference_by_id) == disabled and all(r.get("reference_enabled") is False for r in references), "disabled proof references are not strictly disabled")
    for bid in ids:
        rows = np.flatnonzero(numeric == bid)
        _require(len(rows) == 1, "disabled proof requires one preserved placeholder row per ID")
        _require(str(arrays["visual_group_ids"][rows[0]]) == reference_by_id[bid]["visual_group_id"], "disabled proof family differs")
    image = _inside(root, declaration.get("image_path"))
    _require(_sha(image) == declaration.get("image_sha256"), "disabled proof image digest mismatch")
    evidence = {"manifest_sha256": _sha(manifest_path), "gallery_sha256": _sha(gallery_path), "disabled_business_ids": ids,
                "enabled_business_ids": sorted(set(map(int, numeric[flags]))), "rule_version": RULE}
    return {"root": root, "manifest": manifest, "arrays": arrays, "flags": flags, "ids": numeric,
            "disabled_ids": tuple(ids), "references": reference_by_id, "image": image, "evidence": evidence}


def validate_gap_proof(root, *, base_ids, gap_ids, base_gallery_manifest_path):
    proof = load_disabled_reference_proof(root)
    _require(set(proof["disabled_ids"]) == set(gap_ids), "gap IDs must exactly equal the existing disabled IDs")
    _require(set(proof["evidence"]["enabled_business_ids"]) == set(base_ids), "gap proof enabled IDs differ from accepted base")
    path = Path(base_gallery_manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    gallery = _inside(path.parent.resolve(), manifest["gallery"]["path"])
    _require(_sha(gallery) == manifest["gallery"]["sha256"], "accepted base gallery digest mismatch")
    with np.load(gallery, allow_pickle=False) as base:
        _require(set(base.files) == set(proof["arrays"]) - {"reference_enabled"}, "gap proof/base array sets differ")
        _require(all(np.array_equal(base[k], proof["arrays"][k][proof["flags"]]) for k in base.files), "gap proof enabled arrays differ from accepted base")
    _require(manifest.get("model", {}).get("sha256") == proof["manifest"].get("model", {}).get("sha256"), "gap proof changed the fixed encoder")
    return proof["evidence"]
