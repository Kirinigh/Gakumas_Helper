"""Traceable cost-base derivation from the previously promoted template pool."""
from __future__ import annotations

import copy
import json
import hashlib
from pathlib import Path

import numpy as np

CONTRACT = "card-cost-frozen-template-derivation-v1"
BASE_FILES = {
    "card_cost_reference.npz": "5DC6BB337A2379567FC49C85B1DBD0469F7499006F8AF8BE8004817152636DA5",
    "manifest.json": "3574F3C013EEED5F887EB18D1742BF0715E51120C3CB7404BD337CD5CFD14825",
    "production_handoff_evaluation.json": "D73220AB0034ED7FF566C60830CF64879A3C724906817B9413307FFFC4C1545D",
}
BASE_ARRAYS = "171062A2A7A1A7093D601AF62357EC9468BC400C96A63543CAE37F27BD0DBD8F"
ROW_KEYS = {"generic_card_ids", "base_values", "cost_kinds", "base_masks", "base_symbols"}


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        a = arrays[key]
        digest.update(key.encode() + b"\0" + a.dtype.str.encode() + b"\0"
                      + str(a.shape).encode() + b"\0" + a.tobytes())
    return digest.hexdigest().upper()


def validate_derivation(manifest, arrays):
    provenance = manifest["source"].get("template_derivation")
    if manifest.get("build", {}).get("tool_contract") != CONTRACT:
        if provenance is not None:
            raise ValueError("unexpected template derivation")
        return
    if not isinstance(provenance, dict) or provenance.get("base_files") != BASE_FILES:
        raise ValueError("untrusted frozen cost source")
    old = {k: v[:53] if k in ROW_KEYS else v for k, v in arrays.items()}
    if arrays_digest(old) != BASE_ARRAYS:
        raise ValueError("frozen cost rows or templates changed")
    pairs = list(zip(arrays["template_kinds"].tolist(), arrays["template_values"].tolist()))
    expected = []
    for index in range(53, len(arrays["generic_card_ids"])):
        kind, value = str(arrays["cost_kinds"][index]), int(arrays["base_values"][index])
        if (kind, value) not in pairs or (kind, max(0, value - 2)) not in pairs:
            raise ValueError("derived cost requires both nonzero frozen templates")
        ti = pairs.index((kind, value))
        if not (np.array_equal(arrays["base_masks"][index], arrays["template_masks"][ti])
                and np.array_equal(arrays["base_symbols"][index], arrays["template_symbols"][ti])):
            raise ValueError("derived base differs from frozen template")
        expected.append({"business_id": int(arrays["generic_card_ids"][index]),
                         "kind": kind, "value": value, "hypotheses": [value, value - 2]})
    if not expected or provenance.get("rows") != expected:
        raise ValueError("template derivation mapping mismatch")
    binding = {"base_files": BASE_FILES, "catalog_file_set_sha256": manifest["source"]["catalog_file_set_sha256"],
               "rows": expected}
    if manifest["source"]["source_set_sha256"] != hashlib.sha256(canonical(binding)).hexdigest().upper():
        raise ValueError("template derivation source binding mismatch")


def build(base_root, catalog_dir, output, revision):
    from agent.arena_winrate.catalog import ArenaEntityCatalog

    base_root, catalog_dir, output = map(Path, (base_root, catalog_dir, output))
    if output.exists():
        raise FileExistsError(output)
    for name, expected in BASE_FILES.items():
        if sha(base_root / name) != expected:
            raise ValueError("frozen base handoff changed")
    # The fixed hashes bind the previously promoted three-file source. Current
    # catalog differs by design; its complete contract is checked below/promoter.
    manifest = json.loads((base_root / "manifest.json").read_bytes())
    catalog = ArenaEntityCatalog(json.loads((catalog_dir / "skill_cards.json").read_bytes()),
                                 json.loads((catalog_dir / "customizations.json").read_bytes()))
    with np.load(base_root / "card_cost_reference.npz", allow_pickle=False) as payload:
        arrays = {k: payload[k] for k in payload.files}
    ids = arrays["generic_card_ids"].tolist()
    wanted = list(catalog.generic_cost_card_ids())
    if not set(ids) < set(wanted):
        raise ValueError("derivation must strictly extend frozen cost IDs")
    for index, card_id in enumerate(ids):
        kind, value = str(arrays["cost_kinds"][index]), int(arrays["base_values"][index])
        if catalog.card_face_cost_descriptor(card_id) != (kind, value) or catalog.generic_cost_hypotheses(card_id) != (value, max(0, value - 2)):
            raise ValueError("existing cost catalog semantics changed")
    pairs = list(zip(arrays["template_kinds"].tolist(), arrays["template_values"].tolist()))
    rows = []
    for card_id in sorted(set(wanted) - set(ids)):
        kind, value = catalog.card_face_cost_descriptor(card_id)
        hypotheses = catalog.generic_cost_hypotheses(card_id)
        if hypotheses != (value, value - 2) or any((kind, v) not in pairs for v in hypotheses):
            raise ValueError("new cost has missing or incompatible templates")
        ti = pairs.index((kind, value))
        extra = {"generic_card_ids": card_id, "base_values": value, "cost_kinds": kind,
                 "base_masks": arrays["template_masks"][ti], "base_symbols": arrays["template_symbols"][ti]}
        for key in ROW_KEYS:
            arrays[key] = np.concatenate((arrays[key], np.asarray([extra[key]], dtype=arrays[key].dtype)))
        rows.append({"business_id": card_id, "kind": kind, "value": value, "hypotheses": list(hypotheses)})
    digest = hashlib.sha256()
    for name in sorted(("skill_cards.json", "customizations.json")):
        digest.update(name.encode() + b"\0" + bytes.fromhex(sha(catalog_dir / name)))
    manifest = copy.deepcopy(manifest)
    manifest.pop("evaluation", None)
    manifest["source"].update(revision=revision, catalog_file_set_sha256=digest.hexdigest().upper())
    binding = {"base_files": BASE_FILES, "catalog_file_set_sha256": digest.hexdigest().upper(), "rows": rows}
    manifest["source"]["source_set_sha256"] = hashlib.sha256(canonical(binding)).hexdigest().upper()
    manifest["source"]["template_derivation"] = {"base_files": BASE_FILES, "rows": rows}
    manifest["build"] = {"tool_contract": CONTRACT, "deterministic_output": "NPZ_AND_CANONICAL_LF_JSON"}
    manifest["production_handoff"] = {"status": "PENDING_VALIDATION", "reason": "fixed card-cost reference candidate requires container and offline behavior evaluation before promotion"}
    validate_derivation(manifest, arrays)
    output.mkdir(parents=True)
    np.savez_compressed(output / "card_cost_reference.npz", **arrays)
    manifest["gallery"].update(generic_card_count=len(wanted), sha256=sha(output / "card_cost_reference.npz"))
    (output / "manifest.json").write_bytes(canonical(manifest))
