"""Frozen provenance/replay of calibrated, project-derived full P-item images."""
from __future__ import annotations

import io
import re
import json
from typing import Any
from pathlib import Path
from collections.abc import Mapping

import cv2
import numpy as np
from PIL import Image

from tools import calibrated_p_item_reference as renderer
from tools.p_item_derived_qualification import (
    WINDOWS,
    REQUIRED_IDS,
    DerivedQualificationError,
    file_sha256,
    sha256_bytes,
    current_code_bindings,
    canonical_source_sha256,
    validate_qualification_report,
)

EVIDENCE_TOOL_CONTRACT = "p-item-production-source-evidence-v4"
SOURCE_KEY = "derived_rendered_reference"
SOURCE_TYPE = "calibrated_derived_full_reference"
TARGET_DOMAIN = "arena_member_small_p_item_slot"
SOURCE_REVISION_SUFFIX = "+calibrated-derived-full-reference-v1"
PROMOTED_REASON = (
    "frozen arena-member small-slot derived source replay, lineage, container, ranking and scoped independent "
    "two-window calibration passed; global real JJC calibration remains PENDING"
)
GEOMETRY = {
    "overlay_width_ratio": [28, 58], "overlay_left_ratio": [34, 58],
    "overlay_top_ratio": [-4, 58], "rounding": "floor(x+0.5)",
    "resampling": "INTER_AREA_premultiplied_alpha_float64",
    "compositing": "straight_alpha_after_premultiplied_resampling_canvas_intersection",
    "encoding": "Pillow_RGB_PNG_optimize_false_compress_level_9_no_metadata",
}
OVERLAY = {
    "repository": "https://github.com/vertesan/hatsuboshi-library",
    "revision": "686ae0e07e2564ff0a7beec17db13acbeb0d779d",
    "path": "public/img/icon_enhanced.webp",
    "sha256": "8810A684457A96B80EAFBD2D1A626201A9D53EA98F236815C37D5CFC1472C6B4",
    "dimensions": [56, 56], "source_role": "third_party_transparent_primitive_not_rendered_UI",
}
QUALIFICATION_KEYS = {
    "report_sha256", "status", "scope", "roi_contract", "candidate_npz_sha256", "reference_code_sha256",
    "formal_renderer_sha256", "frozen_renderer_sha256", "mathematical_contract_sha256",
    "reference_code_canonical_lf_sha256", "formal_renderer_canonical_lf_sha256",
    "collection_manifest_sha256", "development_manifest_sha256", "windows", "case_count",
    "observation_count", "observations_with_development_identical_bytes", "distinct_observation_image_sha256_count",
    "verified_business_ids_by_window", "independent_enhanced_family_ids", "wrong_accept_count",
    "new_detail_count", "correct_direct_regression_count", "global_live_calibration",
    "runtime_detail_authority", "unavailable_normal_controls",
    "strong_detail_case_count", "manual_visual_control_case_count", "visible_empty_control_case_count",
    "latency_scope", "end_to_end_reader_calibration",
}


class DerivedSourceError(ValueError):
    """Derived source bytes or their declared qualification are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DerivedSourceError(message)


def _hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9A-F]{64}", value) is not None


def _ids(value: object) -> bool:
    return isinstance(value, list) and all(type(x) is int and x > 0 for x in value) and value == sorted(set(value))


def validate_public_qualification(value: object, *, candidate_sha256: str | None = None, verify_current_code: bool = False) -> None:
    _require(isinstance(value, Mapping) and set(value) == QUALIFICATION_KEYS, "derived scoped qualification has missing or non-public fields")
    assert isinstance(value, Mapping)
    _require(all(_hash(value[key]) for key in QUALIFICATION_KEYS if key.endswith("sha256")), "derived qualification digest is invalid")
    _require(value["status"] == "PASS" and value["scope"] == "frozen_candidate_two_window_independent_observations", "derived scoped qualification is not PASS")
    _require(value["roi_contract"] == "arena_reader_member_p_item_grid_v1", "derived scoped qualification target ROI contract is invalid")
    _require(value["global_live_calibration"] == "PENDING" and value["runtime_detail_authority"] is False, "derived qualification must keep global live calibration PENDING")
    _require(value["latency_scope"] == "offline_matching_compute_and_detail_demand_only" and value["end_to_end_reader_calibration"] == "TASK410_PENDING", "derived qualification cannot claim end-to-end reader calibration")
    _require(value["windows"] == list(WINDOWS), "derived qualification needs both actual windows")
    _require(all(value[key] == 0 and type(value[key]) is int for key in ("wrong_accept_count", "new_detail_count", "correct_direct_regression_count")), "derived qualification contains behavior regressions")
    _require(type(value["case_count"]) is int and value["case_count"] >= 16 and value["case_count"] % 4 == 0, "derived qualification physical-slot count is invalid")
    _require(type(value["observation_count"]) is int and value["observation_count"] * 4 == value["case_count"] * 3, "derived qualification actual-observation count is invalid")
    for key in ("observations_with_development_identical_bytes", "distinct_observation_image_sha256_count"):
        _require(type(value[key]) is int and 0 <= value[key] <= value["observation_count"], "derived qualification visual diversity count is invalid")
    _require(value["distinct_observation_image_sha256_count"] > 0, "derived qualification has no source images")
    authority_counts = [value[key] for key in ("strong_detail_case_count", "manual_visual_control_case_count", "visible_empty_control_case_count")]
    _require(all(type(count) is int and count >= 0 for count in authority_counts) and sum(authority_counts) == value["case_count"] and authority_counts[0] >= 10, "derived qualification truth authority counts are invalid")
    coverage = value["verified_business_ids_by_window"]
    _require(isinstance(coverage, Mapping) and set(coverage) == set(WINDOWS), "derived qualification coverage is invalid")
    _require(all(_ids(coverage[w]) and set(REQUIRED_IDS).issubset(coverage[w]) for w in WINDOWS), "derived qualification lacks known defect/control coverage")
    _require(_ids(value["independent_enhanced_family_ids"]) and bool(value["independent_enhanced_family_ids"]), "derived qualification lacks an undeveloped enhanced family")
    _require(_ids(value["unavailable_normal_controls"]) and set(value["unavailable_normal_controls"]).issubset({132, 135}), "derived qualification pending controls are invalid")
    if candidate_sha256 is not None:
        _require(value["candidate_npz_sha256"] == candidate_sha256, "derived qualification candidate mismatch")
    if verify_current_code:
        _require(all(value[key] == expected for key, expected in current_code_bindings().items()), "derived qualification recognizer or renderer source code changed")


def validate_derived_source(value: object, *, expected_ids: tuple[int, ...], catalog_revision: str) -> None:
    _require(isinstance(value, Mapping) and set(value) == {
        "source_type", "target_domain", "renderer", "overlay", "pairing_contract", "generated_business_ids",
        "references", "catalog_production_deployment", "qualification",
    }, "derived source has missing or non-public fields")
    assert isinstance(value, Mapping)
    _require(value["source_type"] == SOURCE_TYPE and value["overlay"] == OVERLAY, "derived source type or primitive attribution is invalid")
    _require(value["target_domain"] == TARGET_DOMAIN, "derived source target domain is invalid")
    declaration = value["renderer"]
    _require(isinstance(declaration, Mapping) and set(declaration) == {"module", "tool_contract", "sha256", "canonical_lf_sha256", "geometry"}, "derived renderer contract is invalid")
    _require(declaration["module"] == "tools/calibrated_p_item_reference.py" and declaration["tool_contract"] == renderer.TOOL_CONTRACT and declaration["geometry"] == GEOMETRY and _hash(declaration["sha256"]), "derived renderer geometry or revision is invalid")
    _require(_hash(declaration["canonical_lf_sha256"]), "derived renderer canonical source digest is invalid")
    _require(value["pairing_contract"] == "catalog_unique_(sourceType,pIdolId)_false_true_v1", "derived pairing contract is invalid")
    generated = value["generated_business_ids"]
    _require(_ids(generated) and bool(generated) and set(expected_ids).issubset(generated), "derived generation inventory is invalid")
    references = value["references"]
    _require(isinstance(references, Mapping) and set(references) == set(map(str, generated)), "derived references are incomplete")
    for identity in generated:
        record = references[str(identity)]
        _require(isinstance(record, Mapping) and set(record) == {"base_business_id", "base_png_sha256", "png_sha256", "rgb_pixel_sha256", "dimensions", "size"}, "derived reference has unsupported fields")
        _require(type(record["base_business_id"]) is int and record["base_business_id"] > 0 and record["base_business_id"] not in generated, "derived base identity is invalid")
        _require(all(_hash(record[key]) for key in ("base_png_sha256", "png_sha256", "rgb_pixel_sha256")), "derived reference hash is invalid")
        _require(isinstance(record["dimensions"], list) and len(record["dimensions"]) == 2 and all(type(x) is int and x >= 8 for x in record["dimensions"]), "derived reference dimensions are invalid")
        _require(type(record["size"]) is int and record["size"] > 0, "derived reference byte size is invalid")
    deployment = value["catalog_production_deployment"]
    _require(isinstance(deployment, Mapping) and set(deployment) == {"deployment_id", "revision", "status"} and type(deployment["deployment_id"]) is int and deployment["deployment_id"] > 0 and deployment["revision"] == catalog_revision and deployment["status"] == "success", "derived catalog production deployment is invalid")
    validate_public_qualification(value["qualification"])
    _require(value["qualification"]["formal_renderer_sha256"] == declaration["sha256"], "derived qualification renderer binding mismatch")
    _require(value["qualification"]["formal_renderer_canonical_lf_sha256"] == declaration["canonical_lf_sha256"], "derived qualification canonical renderer binding mismatch")


def replay_derived_pngs(base_gallery_path: Path, catalog_path: Path, overlay_path: Path) -> tuple[dict[int, bytes], dict[int, dict[str, Any]]]:
    """Return all inherited/generated PNG bytes and exact generated-source records."""
    _require(overlay_path.is_file() and file_sha256(overlay_path) == OVERLAY["sha256"], "derived overlay bytes mismatch")
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
    _require(overlay is not None and list(overlay.shape) == [56, 56, 4], "derived overlay shape is invalid")
    with np.load(base_gallery_path, allow_pickle=False) as arrays:
        _require(set(arrays.files) == {"p_item_ids", "png_bytes", "png_offsets"}, "derived base inventory is invalid")
        ids, payload, offsets = (np.asarray(arrays[key]) for key in ("p_item_ids", "png_bytes", "png_offsets"))
    _require(ids.dtype == np.int32 and ids.ndim == 1 and payload.dtype == np.uint8 and payload.ndim == 1 and offsets.dtype == np.int64 and offsets.shape == (len(ids) + 1,), "derived base arrays are invalid")
    _require(len(ids) > 0 and int(offsets[0]) == 0 and int(offsets[-1]) == len(payload) and np.all(np.diff(offsets) > 0), "derived base offsets are invalid")
    identities = list(map(int, ids))
    _require(_ids(identities), "derived base IDs are invalid")
    encoded = {identity: bytes(payload[int(offsets[i]):int(offsets[i + 1])]) for i, identity in enumerate(identities)}
    rows = json.loads(catalog_path.read_bytes())
    pairs = renderer.pair_upgraded_p_items(rows, identities)
    output = dict(encoded)
    references: dict[int, dict[str, Any]] = {}
    for enhanced_id, base_id in sorted(pairs.items()):
        pixels = cv2.imdecode(np.frombuffer(encoded[base_id], np.uint8), cv2.IMREAD_COLOR)
        composed = renderer.compose_enhanced_reference(pixels, overlay)
        buffer = io.BytesIO()
        Image.fromarray(composed[:, :, ::-1]).save(buffer, format="PNG", optimize=False, compress_level=9)
        raw = buffer.getvalue()
        output[enhanced_id] = raw
        references[enhanced_id] = {
            "base_business_id": base_id, "base_png_sha256": sha256_bytes(encoded[base_id]),
            "png_sha256": sha256_bytes(raw), "rgb_pixel_sha256": sha256_bytes(composed[:, :, ::-1].tobytes()),
            "dimensions": [composed.shape[1], composed.shape[0]], "size": len(raw),
        }
    return output, references


def build_derived_source(
    *, base_gallery_path: Path, catalog_path: Path, overlay_path: Path, source_dir: Path,
    candidate_root: Path, qualification_report: Path, runtime_contract: Mapping[str, Any],
    catalog_production_deployment: Mapping[str, Any],
) -> dict[str, Any]:
    output, references = replay_derived_pngs(base_gallery_path, catalog_path, overlay_path)
    _require({path.name for path in source_dir.iterdir() if path.is_file()} == {f"{identity}.png" for identity in output}, "derived source inventory mismatch")
    _require(all((source_dir / f"{identity}.png").read_bytes() == raw for identity, raw in output.items()), "derived source differs from base/overlay replay or inherited bytes")
    candidate_npz = candidate_root / "p_item_rendered_reference_gallery.npz"
    with np.load(candidate_npz, allow_pickle=False) as arrays:
        _require(set(arrays.files) == {"p_item_ids", "png_bytes", "png_offsets"}, "qualified candidate array inventory is invalid")
        ids, payload, offsets = (np.asarray(arrays[key]) for key in ("p_item_ids", "png_bytes", "png_offsets"))
    _require(ids.dtype == np.int32 and ids.ndim == 1 and payload.dtype == np.uint8 and payload.ndim == 1 and offsets.dtype == np.int64 and offsets.shape == (len(ids) + 1,), "qualified candidate arrays are invalid")
    _require(list(map(int, ids)) == sorted(output) and int(offsets[0]) == 0 and int(offsets[-1]) == len(payload) and np.all(np.diff(offsets) > 0), "qualified candidate inventory or offsets differ from source replay")
    _require(all(bytes(payload[int(offsets[index]):int(offsets[index + 1])]) == output[int(identity)] for index, identity in enumerate(ids)), "qualified candidate PNG bytes differ from full source replay")
    try:
        qualification = validate_qualification_report(qualification_report, candidate_root=candidate_root, base_gallery_path=base_gallery_path, catalog_path=catalog_path, runtime_contract=runtime_contract)
    except DerivedQualificationError as error:
        raise DerivedSourceError(str(error)) from error
    return {
        "source_type": SOURCE_TYPE, "target_domain": TARGET_DOMAIN,
        "renderer": {"module": "tools/calibrated_p_item_reference.py", "tool_contract": renderer.TOOL_CONTRACT, "sha256": file_sha256(Path(renderer.__file__)), "canonical_lf_sha256": canonical_source_sha256(Path(renderer.__file__)), "geometry": GEOMETRY},
        "overlay": OVERLAY, "pairing_contract": "catalog_unique_(sourceType,pIdolId)_false_true_v1",
        "generated_business_ids": sorted(references), "references": {str(identity): record for identity, record in references.items()},
        "catalog_production_deployment": dict(catalog_production_deployment), "qualification": qualification,
    }
