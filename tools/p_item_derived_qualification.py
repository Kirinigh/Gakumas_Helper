"""Private, bounded real-observation replay for derived full P-item references.

Collection records and detail truth require independent acquisition/audit. Hashes
bind that evidence; they cannot establish when a screenshot was taken or its truth.
Only the public aggregate returned here may enter a distributable source proof.
"""
from __future__ import annotations

import json
import hashlib
from typing import Any
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from collections.abc import Mapping

import cv2
import numpy as np

from agent.p_item_recognition.reference import PItemRenderedReferenceGallery

TOOL_CONTRACT = "task095-derived-p-item-scoped-qualification-v1"
REQUIRED_IDS = (133, 136, 442, 443)
WINDOWS = ("large", "small")
ROI_CONTRACT = "arena_reader_member_p_item_grid_v1"
OCR_RECORD_KIND = "ocr_observation_v1"
STABLE_CAPTURE_RECORD_KIND = "stable_source_capture_v1"
STABLE_CAPTURE_ORIGIN = "original stable source generation, independent of detail OCR"
REPORT_KEYS = {"schema_version", "tool_contract", "inputs", "results", "summary"}


class DerivedQualificationError(ValueError):
    """The scoped report is incomplete, stale or fails real-observation replay."""


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_source_sha256(path: Path) -> str:
    """Bind source text while accepting Git's CRLF/LF checkout conversion only."""
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def current_code_bindings() -> dict[str, str]:
    from tools import calibrated_p_item_reference as renderer
    from agent.p_item_recognition import reference
    return {
        "reference_code_canonical_lf_sha256": canonical_source_sha256(Path(reference.__file__)),
        "formal_renderer_canonical_lf_sha256": canonical_source_sha256(Path(renderer.__file__)),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DerivedQualificationError(message)


def _object(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == keys, f"{label} has unsupported or missing fields")
    return value  # type: ignore[return-value]


def _time(value: object) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise DerivedQualificationError("observation time is invalid") from error
    _require(result.tzinfo is not None, "observation time must include timezone")
    return result


def _bound_file(binding: object, label: str) -> Path:
    value = _object(binding, {"path", "sha256"}, label)
    path = Path(str(value["path"]))
    _require(path.is_absolute() and path.is_file(), f"{label} input is unavailable")
    _require(file_sha256(path) == value["sha256"], f"{label} SHA-256 mismatch")
    return path


def _ids(value: object, label: str, *, empty: bool = False) -> list[int]:
    _require(isinstance(value, list), f"{label} must be an ordered ID list")
    _require((bool(value) or empty) and all(type(i) is int and i > 0 for i in value), f"{label} has invalid IDs")
    _require(value == sorted(set(value)), f"{label} must be unique and sorted")
    return value  # type: ignore[return-value]


def _load_gallery_from_npz(path: Path, candidate: PItemRenderedReferenceGallery) -> PItemRenderedReferenceGallery:
    with np.load(path, allow_pickle=False) as arrays:
        ids, payload, offsets = (np.asarray(arrays[key]) for key in ("p_item_ids", "png_bytes", "png_offsets"))
    images = tuple(cv2.imdecode(payload[int(offsets[i]):int(offsets[i + 1])], cv2.IMREAD_COLOR) for i in range(len(ids)))
    _require(all(image is not None for image in images), "base gallery PNG decode failed")
    return PItemRenderedReferenceGallery(tuple(map(int, ids)), images, runtime=candidate.runtime)


def _decision(gallery: PItemRenderedReferenceGallery, frames: tuple[np.ndarray, ...], roi: list[int], plan: str) -> dict[str, Any]:
    result = asdict(gallery.classify(frames, tuple(roi), plan=plan))
    # JSON round trip makes tuple/list spelling identical in generated/replayed reports.
    return json.loads(canonical_bytes(result))


def arena_member_rois(width: int, height: int) -> set[tuple[int, int, int, int]]:
    """The unchanged arena_reader.read_p_item_ids member-slot grid."""
    icon = int(width * 0.09)
    return {(int(width * x), int(height * 0.207), icon, icon) for x in (0.05, 0.147, 0.247, 0.34)}


def evaluate_qualification_inputs(
    inputs: Mapping[str, Any], *, candidate_root: Path, base_gallery_path: Path,
    catalog_path: Path, runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Read original frames, run the unmodified temporal classifier, derive a report.

    The collection manifest is an independently reviewed acquisition record. It
    identifies distinct observations even when a static scene has identical bytes.
    No code is loaded from any input path.
    """
    _object(inputs, {"freeze", "collection_manifest", "development_manifest", "cases"}, "qualification inputs")
    freeze = _object(inputs["freeze"], {
        "candidate_npz_sha256", "frozen_at", "frozen_renderer", "mathematical_contract",
        "reference_code_sha256", "formal_renderer_sha256",
    }, "candidate freeze")
    from tools import calibrated_p_item_reference as renderer
    from agent.p_item_recognition import reference
    _require(file_sha256(Path(renderer.__file__)) == freeze["formal_renderer_sha256"], "formal renderer binding mismatch")
    _require(file_sha256(Path(reference.__file__)) == freeze["reference_code_sha256"], "production reference code binding mismatch")
    _bound_file(freeze["frozen_renderer"], "frozen renderer")
    _bound_file(freeze["mathematical_contract"], "mathematical contract")
    frozen_at = _time(freeze["frozen_at"])
    npz = candidate_root / "p_item_rendered_reference_gallery.npz"
    _require(file_sha256(npz) == freeze["candidate_npz_sha256"], "qualification candidate mismatch")
    manifest = json.loads((candidate_root / "manifest.json").read_bytes())
    _require(manifest.get("runtime") == dict(runtime_contract), "qualification changed the original runtime gates")
    candidate = PItemRenderedReferenceGallery.load(candidate_root)
    original = _load_gallery_from_npz(base_gallery_path, candidate)
    _require(original.p_item_ids == candidate.p_item_ids, "qualification changed the gallery ID inventory")
    catalog_rows = json.loads(catalog_path.read_bytes())
    by_id = {row["id"]: row for row in catalog_rows}

    development_path = _bound_file(inputs["development_manifest"], "development manifest")
    development = _object(json.loads(development_path.read_bytes()), {"frame_sha256", "family_ids", "capture_events", "source_paths"}, "development manifest")
    development_families = _ids(development["family_ids"], "development family IDs")
    excluded_hashes = development["frame_sha256"]
    _require(isinstance(excluded_hashes, list) and bool(excluded_hashes), "development frame exclusion inventory is missing")
    excluded_events = {(row["capture_session"], row["capture_sequence"]) for row in development["capture_events"]}
    excluded_paths = {Path(path).resolve() for path in development["source_paths"]}
    _require(bool(excluded_events) and bool(excluded_paths), "development acquisition exclusion inventory is missing")

    collection_path = _bound_file(inputs["collection_manifest"], "collection manifest")
    collection = _object(json.loads(collection_path.read_bytes()), {
        "schema_version", "candidate_npz_sha256", "windows", "observations", "truths",
    }, "collection manifest")
    _require(type(collection["schema_version"]) is int and collection["schema_version"] == 1 and collection["candidate_npz_sha256"] == freeze["candidate_npz_sha256"], "collection was not bound to the frozen candidate")
    windows = _object(collection["windows"], set(WINDOWS), "actual windows")
    physical_sizes = []
    for window in WINDOWS:
        descriptor = _object(windows[window], {"physical_size", "normalized_size", "dpi", "acquisition_evidence"}, "window descriptor")
        window_evidence = _bound_file(descriptor["acquisition_evidence"], "window acquisition evidence")
        gate = json.loads(window_evidence.read_bytes())
        _require(isinstance(gate, Mapping) and isinstance(gate.get("window"), Mapping), "window acquisition evidence lacks actual window metadata")
        _require(gate["window"].get("physical_client_size") == descriptor["physical_size"] and gate["window"].get("dpi") == descriptor["dpi"], "window dimensions or DPI differ from acquisition evidence")
        for key in ("physical_size", "normalized_size"):
            _require(isinstance(descriptor[key], list) and len(descriptor[key]) == 2 and all(type(x) is int and x > 0 for x in descriptor[key]), "window dimensions are invalid")
        _require(type(descriptor["dpi"]) is int and descriptor["dpi"] > 0, "window DPI is invalid")
        physical_sizes.append(descriptor["physical_size"])
    _require(physical_sizes[0] != physical_sizes[1], "two actual physical windows are required")

    observations: dict[str, Mapping[str, Any]] = {}
    files: set[Path] = set()
    events: set[tuple[str, int]] = set()
    for raw in collection["observations"]:
        record_keys = {"observation_id", "window", "observed_at", "capture_session", "capture_sequence", "image", "capture_evidence", "copied"}
        if isinstance(raw, Mapping) and "capture_record_kind" in raw:
            record_keys.add("capture_record_kind")
        record = _object(raw, record_keys, "observation")
        record_kind = record.get("capture_record_kind", OCR_RECORD_KIND)
        _require(record_kind in {OCR_RECORD_KIND, STABLE_CAPTURE_RECORD_KIND}, "unsupported capture record kind")
        identity = record["observation_id"]
        _require(isinstance(identity, str) and bool(identity) and identity not in observations, "observation IDs must be unique")
        _require(record["window"] in WINDOWS and record["copied"] is False, "copied or unclassified observation")
        _require(_time(record["observed_at"]) > frozen_at, "observation predates candidate freeze")
        _require(isinstance(record["capture_session"], str) and bool(record["capture_session"]) and type(record["capture_sequence"]) is int and record["capture_sequence"] >= 0, "capture record is invalid")
        event = (record["capture_session"], record["capture_sequence"])
        _require(event not in events, "one capture event cannot represent multiple observations")
        _require(event not in excluded_events, "development capture record cannot be an independent holdout")
        events.add(event)
        capture_evidence = _bound_file(record["capture_evidence"], "capture evidence")
        _require(capture_evidence.resolve() not in excluded_paths, "development acquisition evidence cannot be an independent holdout")
        path = _bound_file(record["image"], "source frame")
        _require(path.resolve() not in files, "one source file cannot represent multiple observations")
        _require(path.resolve() not in excluded_paths, "development source path cannot be an independent holdout")
        capture_document = json.loads(capture_evidence.read_bytes())
        _require(isinstance(capture_document, Mapping), "capture evidence must contain an object")
        capture_rows = capture_document.get("frames")
        sequence = record["capture_sequence"]
        if record_kind == STABLE_CAPTURE_RECORD_KIND:
            _require(capture_document.get("frame_origin") == STABLE_CAPTURE_ORIGIN, "stable capture evidence has the wrong frame origin")
            _require(isinstance(capture_rows, list) and bool(capture_rows) and all(isinstance(row, Mapping) and type(row.get("capture_sequence")) is int and row["capture_sequence"] >= 0 for row in capture_rows), "stable capture sequence inventory is invalid")
            original_sequences = [row["capture_sequence"] for row in capture_rows]
            _require(len(set(original_sequences)) == len(original_sequences), "stable capture sequences must be unique")
            matching_rows = [row for row in capture_rows if row["capture_sequence"] == sequence]
            _require(len(matching_rows) == 1, "stable capture sequence is absent from the original records")
            capture_row = matching_rows[0]
            time_field = "captured_at"
        else:
            _require(isinstance(capture_rows, list) and sequence < len(capture_rows), "capture record sequence is outside the original frames")
            capture_row = capture_rows[sequence]
            time_field = "observed_at"
        _require(isinstance(capture_row, Mapping) and isinstance(capture_row.get("image"), str), "capture record image is invalid")
        relative_image = Path(capture_row["image"])
        _require(not relative_image.is_absolute() and (capture_evidence.parent / relative_image).resolve() == path.resolve(), "source frame path differs from the original capture record")
        _require(_time(capture_row.get(time_field)) == _time(record["observed_at"]), "stable capture time differs from the original capture record" if record_kind == STABLE_CAPTURE_RECORD_KIND else "observation time differs from the original capture record")
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        _require(decoded is not None and list(decoded.shape) == capture_row.get("shape"), "source frame shape differs from the original capture record")
        if record_kind == STABLE_CAPTURE_RECORD_KIND:
            boxes = capture_row.get("p_item_boxes")
            _require(isinstance(boxes, list) and len(boxes) == 4 and all(isinstance(box, list) and len(box) == 4 and all(type(x) is int for x in box) for box in boxes), "stable source member-slot boxes are invalid")
            _require({tuple(box) for box in boxes} == arena_member_rois(decoded.shape[1], decoded.shape[0]), "stable source boxes differ from the actual arena member grid")
        files.add(path.resolve())
        observations[identity] = record

    truths: dict[str, Mapping[str, Any]] = {}
    for raw in collection["truths"]:
        truth = _object(raw, {"truth_id", "business_id", "upgraded", "source_image", "truth_authority"}, "truth")
        _require(isinstance(truth["truth_id"], str) and truth["truth_id"] not in truths, "detail truth identity is duplicated")
        business_id = truth["business_id"]
        authority = _object(truth["truth_authority"], {"kind", "evidence"}, "truth authority")
        _require(authority["kind"] in {"independently_audited_detail", "manual_reference_visual", "visible_empty"}, "unsupported truth authority")
        if business_id == 0:
            _require(type(business_id) is int and truth["upgraded"] is False and authority["kind"] == "visible_empty", "empty truth authority is invalid")
        else:
            _require(type(business_id) is int and business_id in by_id and by_id[business_id]["upgraded"] is truth["upgraded"] and authority["kind"] != "visible_empty", "truth state differs from the catalog")
        truth_image = _bound_file(truth["source_image"], "truth source image")
        _require(cv2.imread(str(truth_image), cv2.IMREAD_COLOR) is not None, "truth source image cannot be decoded")
        _bound_file(authority["evidence"], "truth audit evidence")
        truths[truth["truth_id"]] = truth

    cases = inputs["cases"]
    _require(isinstance(cases, list) and bool(cases), "qualification cases are missing")
    results, case_ids, used_observations = [], set(), set()
    page_rois: dict[tuple[str, ...], list[tuple[int, ...]]] = {}
    covered = {window: set() for window in WINDOWS}
    new_families = {window: set() for window in WINDOWS}
    image_cache: dict[str, np.ndarray] = {}
    for raw in cases:
        case = _object(raw, {"case_id", "window", "observation_ids", "roi", "plan", "truth_id", "require_direct"}, "qualification case")
        _require(isinstance(case["case_id"], str) and case["case_id"] not in case_ids, "duplicate case identity")
        case_ids.add(case["case_id"])
        window = case["window"]
        _require(window in WINDOWS, "case window is invalid")
        observation_ids = case["observation_ids"]
        _require(isinstance(observation_ids, list) and len(observation_ids) == 3 and len(set(observation_ids)) == 3, "each case requires three actual observations")
        _require(all(key in observations and observations[key]["window"] == window for key in observation_ids), "case observations do not belong to this window")
        capture_set = [observations[key] for key in observation_ids]
        record_kinds = {row.get("capture_record_kind", OCR_RECORD_KIND) for row in capture_set}
        _require(len(record_kinds) == 1, "three observations must use one capture record kind")
        _require(len({row["capture_session"] for row in capture_set}) == 1 and all(row["capture_evidence"] == capture_set[0]["capture_evidence"] for row in capture_set), "three observations must belong to one original capture session")
        sequences = [row["capture_sequence"] for row in capture_set]
        _require(sequences == list(range(sequences[0], sequences[0] + 3)), "three observations require consecutive original capture records")
        used_observations.update(observation_ids)
        times = [_time(observations[key]["observed_at"]) for key in observation_ids]
        _require(times == sorted(times) and len(set(times)) == 3, "three observation times must be distinct and ordered")
        for key in observation_ids:
            if key not in image_cache:
                image = cv2.imread(str(_bound_file(observations[key]["image"], "source frame")), cv2.IMREAD_COLOR)
                _require(image is not None and [image.shape[1], image.shape[0]] == windows[window]["normalized_size"], "source frame dimensions differ from actual window")
                image_cache[key] = image
        roi = case["roi"]
        _require(isinstance(roi, list) and len(roi) == 4 and all(type(x) is int for x in roi), "case ROI is invalid")
        width, height = windows[window]["normalized_size"]
        _require(tuple(roi) in arena_member_rois(width, height), "case ROI is outside the actual arena member small-slot grid")
        page_rois.setdefault(tuple(observation_ids), []).append(tuple(roi))
        _require(case["plan"] in {"sense", "logic", "anomaly"} and type(case["require_direct"]) is bool, "case plan/direct requirement is invalid")
        _require(case["truth_id"] in truths, "case truth is unavailable")
        truth = truths[case["truth_id"]]
        business_id = truth["business_id"]
        authority_kind = truth["truth_authority"]["kind"]
        if business_id:
            covered[window].add(business_id)
            row = by_id[business_id]
            _require(business_id not in REQUIRED_IDS or authority_kind == "independently_audited_detail", "known defects require independent detail truth")
            if row["upgraded"] and row.get("pIdolId") not in development_families and authority_kind == "independently_audited_detail":
                new_families[window].add(row["pIdolId"])
        frames = tuple(image_cache[key] for key in observation_ids)
        before = _decision(original, frames, roi, case["plan"])
        after = _decision(candidate, frames, roi, case["plan"])
        direct = lambda decision: decision["status"] in {"ACCEPTED", "EMPTY"}
        before_correct = direct(before) and before["p_item_id"] == business_id
        after_correct = direct(after) and after["p_item_id"] == business_id
        _require(not direct(after) or after_correct, f"new wrong acceptance in {case['case_id']}")
        _require(not before_correct or after_correct, f"previously correct slot regressed in {case['case_id']}")
        _require(not direct(before) or direct(after), f"new detail demand in {case['case_id']}")
        _require(not (case["require_direct"] or business_id in REQUIRED_IDS) or after_correct, f"required direct identity failed in {case['case_id']}")
        row_result = {"case_id": case["case_id"], "window": window, "truth_business_id": business_id, "truth_authority": authority_kind, "original": before, "candidate": after}
        if record_kinds == {STABLE_CAPTURE_RECORD_KIND}:
            row_result["capture_record_kind"] = STABLE_CAPTURE_RECORD_KIND
            row_result["event_time_semantics"] = "original_stable_source_captured_at"
        results.append(row_result)
    _require(used_observations == set(observations), "collection has observations omitted from the qualification")
    _require(all(len(rois) == len(set(rois)) == 4 for rois in page_rois.values()), "every observed page requires all four physical slots")
    _require(all(set(rois) == arena_member_rois(*windows[observations[observation_ids[0]]["window"]]["normalized_size"]) for observation_ids, rois in page_rois.items()), "observed page ROI set differs from the complete member small-slot grid")
    _require(all(set(REQUIRED_IDS).issubset(covered[w]) for w in WINDOWS), "known defects and normal control must be observed in both windows")
    independent_families = sorted(new_families["large"] & new_families["small"])
    _require(bool(independent_families), "an undeveloped enhanced family must be observed in both windows")
    # The novel family is a required positive, never rescued by retained details.
    _require(all(row["candidate"]["status"] == "ACCEPTED" for row in results if row["truth_business_id"] and by_id[row["truth_business_id"]].get("pIdolId") in independent_families and by_id[row["truth_business_id"]]["upgraded"]), "independent enhanced family was not accepted directly")
    summary = {
        "status": "PASS", "scope": "frozen_candidate_two_window_independent_observations",
        "roi_contract": ROI_CONTRACT,
        "candidate_npz_sha256": freeze["candidate_npz_sha256"], "reference_code_sha256": freeze["reference_code_sha256"],
        **current_code_bindings(),
        "formal_renderer_sha256": freeze["formal_renderer_sha256"], "frozen_renderer_sha256": freeze["frozen_renderer"]["sha256"],
        "mathematical_contract_sha256": freeze["mathematical_contract"]["sha256"],
        "collection_manifest_sha256": inputs["collection_manifest"]["sha256"], "development_manifest_sha256": inputs["development_manifest"]["sha256"],
        "windows": list(WINDOWS), "case_count": len(results), "observation_count": len(observations),
        "observations_with_development_identical_bytes": sum(row["image"]["sha256"] in excluded_hashes for row in observations.values()),
        "distinct_observation_image_sha256_count": len({row["image"]["sha256"] for row in observations.values()}),
        "verified_business_ids_by_window": {key: sorted(covered[key]) for key in WINDOWS},
        "independent_enhanced_family_ids": independent_families,
        "wrong_accept_count": 0, "new_detail_count": 0, "correct_direct_regression_count": 0,
        "strong_detail_case_count": sum(row["truth_authority"] == "independently_audited_detail" for row in results),
        "manual_visual_control_case_count": sum(row["truth_authority"] == "manual_reference_visual" for row in results),
        "visible_empty_control_case_count": sum(row["truth_authority"] == "visible_empty" for row in results),
        "latency_scope": "offline_matching_compute_and_detail_demand_only",
        "end_to_end_reader_calibration": "TASK410_PENDING",
        "global_live_calibration": "PENDING", "runtime_detail_authority": False,
        "unavailable_normal_controls": [i for i in (132, 135) if not all(i in covered[w] for w in WINDOWS)],
    }
    return {"schema_version": 1, "tool_contract": TOOL_CONTRACT, "inputs": dict(inputs), "results": results, "summary": summary}


def validate_qualification_report(
    report_path: Path, *, candidate_root: Path, base_gallery_path: Path,
    catalog_path: Path, runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    _require(report_path.is_file(), "derived qualification report is required")
    raw = report_path.read_bytes()
    report = _object(json.loads(raw), REPORT_KEYS, "qualification report")
    _require(type(report["schema_version"]) is int and report["schema_version"] == 1 and report["tool_contract"] == TOOL_CONTRACT, "qualification report schema or tool contract is invalid")
    _require(raw == canonical_bytes(report), "qualification report must use canonical LF JSON")
    actual = evaluate_qualification_inputs(report["inputs"], candidate_root=candidate_root, base_gallery_path=base_gallery_path, catalog_path=catalog_path, runtime_contract=runtime_contract)
    _require(report == actual, "qualification report statistics or decisions differ from original-frame replay")
    return {"report_sha256": sha256_bytes(raw), **actual["summary"]}
