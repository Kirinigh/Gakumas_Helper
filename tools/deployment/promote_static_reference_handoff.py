"""Validate and atomically promote static recognition reference galleries.

Builders intentionally emit only ``PENDING_VALIDATION`` candidates.  This
tool replays the component's load and offline reference checks, writes a
public-safe aggregate report, and is the only path that emits the release
eligible handoff used by the release packer.
"""

from __future__ import annotations

import io
import os
import re
import sys
import copy
import json
import shutil
import hashlib
import secrets
import argparse
from typing import Any
from pathlib import Path
from collections.abc import Mapping

import cv2
import numpy as np
from PIL import Image

try:
    from agent.arena_winrate.catalog import ArenaEntityCatalog
    from agent.arena_winrate.card_cost import (
        GenericCostReferenceError,
        GenericCostReferenceGallery,
        stable_generic_cost_value,
    )
    from agent.p_item_recognition.reference import (
        PItemReferenceError,
        PItemRenderedReferenceGallery,
    )
except ModuleNotFoundError:  # Direct script execution from tools/deployment.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.arena_winrate.catalog import ArenaEntityCatalog  # type: ignore[no-redef]
    from agent.arena_winrate.card_cost import (  # type: ignore[no-redef]
        GenericCostReferenceError,
        GenericCostReferenceGallery,
        stable_generic_cost_value,
    )
    from agent.p_item_recognition.reference import (  # type: ignore[no-redef]
        PItemReferenceError,
        PItemRenderedReferenceGallery,
    )

SCHEMA_VERSION = 1
PROMOTION_TOOL_CONTRACT = "task095-static-reference-production-promotion-v1"
REPORT_NAME = "production_handoff_evaluation.json"
PENDING_STATUS = "PENDING_VALIDATION"
PROMOTED_STATUS = "READY_WITH_REAL_SAMPLES_PENDING"
P_ITEM_PRODUCTION_SOURCE_EVIDENCE_TOOL_CONTRACT = (
    "task095-p-item-production-source-evidence-v1"
)
P_ITEM_PRODUCTION_SOURCE_EVIDENCE_SHA256 = (
    "8398CF4041FEED83AEC4EA356351E5F69E8E33101852738070B7CDFB17A20463"
)
P_ITEM_PRODUCTION_REVISION = "d476e78b1d3b9fb8eac61924e3869b647ca82ab2"
P_ITEM_PRODUCTION_DEPLOYMENT_ID = 6156565906
P_ITEM_GK_IMG_GITLINK_REVISION = "4acb592e4dc81acabad7df7ed08fe5b5fb572adc"
P_ITEM_PRODUCTION_REFERENCE_IDS = (475, 476)
SHA1_PATTERN = re.compile(r"[0-9A-Fa-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9A-Fa-f]{64}")
COST_REPLAY_CONTRACT = {
    "card389_explicit_zero_cost.json": {
        "business_id": 389,
        "sha256": "FDBAAC87406B77F472797A54F3200328298FE79651169C3ADDDAA14C0B804EB6",
    },
    "card423_explicit_zero_cost.json": {
        "business_id": 423,
        "sha256": "500554C1DF67EC9A8F78F38958F2199F4E34F2D4199E2D031BD37487587BD1BD",
    },
    "user_crop_explicit_zero_cost.json": {
        "business_id": 423,
        "sha256": "B9D86A5271A15DD74B4AB7BF2BB664A4DC5AC60140398BBE3672D0C6B63FD7CF",
    },
}
COST_REPLAY_SET_SHA256 = "5740B066D36541962602D2D14DAB7C9618FCB565F7031635E673B0CBD1B416F7"
COST_RUNTIME_CONTRACT = {
    "maximum_cutout_error": 0.16,
    "maximum_digit_error": 0.18,
    "maximum_high_error": 0.2,
    "maximum_mask_error": 0.11,
    "maximum_medium_error": 0.15,
    "minimum_class_margin": 0.01,
    "minimum_cutout_margin": 0.04,
    "minimum_digit_margin": 0.05,
    "minimum_medium_margin": 0.04,
    "rule_version": "arena-generic-cost-v8-id-independent-zero-topology",
}
P_ITEM_RUNTIME_CONTRACT = {
    "ambiguous_action": "bounded_detail_or_fail_closed",
    "coarse_fine_profiles": [
        {
            "acceptance_rank_limit": 12,
            "candidate_limit": 24,
            "canonical_slot_size": [64, 64],
            "coarse_delta": -6,
            "minimum_capture_size": [720, 1280],
        }
    ],
    "complete_opposite_half_minimum": 0.05,
    "content_generation_max_mean_abs_error": 0.75,
    "embedding_role": "offline_diagnostic_only",
    "empty_foreground_maximum": 0.05,
    "identity_authority": "fixed_rendered_reference_coarse_fine_with_full_fallback",
    "identity_minimum_margin": 0.01,
    "identity_minimum_similarity": 0.65,
    "maximum_detail_fallbacks_per_member": 2,
    "nonempty_foreground_minimum": 0.15,
    "source_color_order": "BGR",
    "stable_frame_count": 3,
}

COMPONENT_CONTRACTS = {
    "arena_cost_reference": {
        "build_tool_contract": "task095-arena-card-cost-reference-v1",
        "gallery_key": "gallery",
        "gallery_name": "card_cost_reference.npz",
        "candidate_reason": ("fixed card-cost reference candidate requires container and offline behavior evaluation before promotion"),
        "promoted_reason": (
            "deterministic source rebuild, container, and offline behavior gates passed; independent live holdout remains PENDING"
        ),
    },
    "p_item_reference": {
        "build_tool_contract": "task095-p-item-rendered-reference-gallery-v1",
        "gallery_key": "reference_gallery",
        "gallery_name": "p_item_rendered_reference_gallery.npz",
        "candidate_reason": ("fixed rendered-reference candidate requires lineage, container, and ranking evaluation before promotion"),
        "promoted_reason": (
            "authoritative fixed-rendered source, deterministic lineage, container, "
            "and offline ranking gates passed; real JJC calibration remains PENDING"
        ),
    },
}


class StaticReferenceHandoffError(RuntimeError):
    """Raised when a static reference component cannot be promoted or released."""


def _is_sha1(value: object) -> bool:
    return isinstance(value, str) and SHA1_PATTERN.fullmatch(value) is not None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the byte contract used for candidate, report, and manifest hashes."""

    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_p_item_production_source_evidence(
    evidence: Mapping[str, Any],
) -> None:
    if (
        set(evidence)
        != {
            "base_gallery",
            "catalog",
            "official_rendered_source",
            "schema_version",
            "source",
            "tool_contract",
        }
        or evidence.get("schema_version") != 1
        or evidence.get("tool_contract")
        != P_ITEM_PRODUCTION_SOURCE_EVIDENCE_TOOL_CONTRACT
    ):
        raise StaticReferenceHandoffError(
            "P-item production/source evidence contract is invalid"
        )
    base = evidence.get("base_gallery")
    catalog = evidence.get("catalog")
    official = evidence.get("official_rendered_source")
    source = evidence.get("source")
    if not all(isinstance(value, Mapping) for value in (base, catalog, official, source)):
        raise StaticReferenceHandoffError(
            "P-item production/source evidence is incomplete"
        )
    assert isinstance(base, Mapping)
    assert isinstance(catalog, Mapping)
    assert isinstance(official, Mapping)
    assert isinstance(source, Mapping)
    count = source.get("business_id_count")
    if (
        set(source)
        != {
            "business_id_count",
            "revision",
            "sanitized_icons_sha256",
            "source_icons_sha256",
        }
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < len(P_ITEM_PRODUCTION_REFERENCE_IDS)
        or not _is_sha256(source.get("sanitized_icons_sha256"))
        or not _is_sha256(source.get("source_icons_sha256"))
    ):
        raise StaticReferenceHandoffError(
            "P-item production/source evidence source contract is invalid"
        )
    if (
        set(base)
        != {
            "business_id_count",
            "inherited_business_id_count",
            "replaced_business_ids",
            "sha256",
        }
        or base.get("business_id_count") != count
        or base.get("inherited_business_id_count")
        != count - len(P_ITEM_PRODUCTION_REFERENCE_IDS)
        or base.get("replaced_business_ids")
        != list(P_ITEM_PRODUCTION_REFERENCE_IDS)
        or not _is_sha256(base.get("sha256"))
    ):
        raise StaticReferenceHandoffError(
            "P-item production/source evidence base gallery is invalid"
        )
    if (
        set(catalog)
        != {
            "excluded_current_rows",
            "included_business_ids",
            "p_items_git_blob_sha1",
            "p_items_sha256",
            "path",
            "repository",
            "revision",
            "stage_scope_contract",
        }
        or catalog.get("repository")
        != "https://github.com/surisuririsu/gakumas-tools"
        or catalog.get("path") != "packages/gakumas-data/json/p_items.json"
        or catalog.get("revision") != P_ITEM_PRODUCTION_REVISION
        or catalog.get("included_business_ids")
        != list(P_ITEM_PRODUCTION_REFERENCE_IDS)
        or not isinstance(catalog.get("excluded_current_rows"), list)
        or not isinstance(catalog.get("stage_scope_contract"), str)
        or not catalog["stage_scope_contract"]
        or not _is_sha1(catalog.get("p_items_git_blob_sha1"))
        or not _is_sha256(catalog.get("p_items_sha256"))
    ):
        raise StaticReferenceHandoffError(
            "P-item production/source evidence catalog contract is invalid"
        )
    expected_source_revision = (
        f"base-gallery-sha256@{base['sha256']}+"
        f"gakumas-tools@{P_ITEM_PRODUCTION_REVISION}"
    )
    deployment = official.get("production_deployment")
    references = official.get("references")
    if (
        source.get("revision") != expected_source_revision
        or set(official)
        != {
            "first_added_revision",
            "first_successful_production_revision",
            "gk_img_gitlink_path",
            "gk_img_gitlink_repository",
            "gk_img_gitlink_revision",
            "package_path",
            "production_deployment",
            "references",
            "repository",
        }
        or official.get("repository")
        != "https://github.com/surisuririsu/gakumas-tools"
        or official.get("package_path") != "packages/gakumas-images"
        or official.get("gk_img_gitlink_path") != "gk-img"
        or official.get("gk_img_gitlink_repository")
        != "https://github.com/surisuririsu/gk-img"
        or official.get("gk_img_gitlink_revision")
        != P_ITEM_GK_IMG_GITLINK_REVISION
        or not _is_sha1(official.get("first_added_revision"))
        or not _is_sha1(official.get("first_successful_production_revision"))
        or deployment
        != {
            "deployment_id": P_ITEM_PRODUCTION_DEPLOYMENT_ID,
            "revision": P_ITEM_PRODUCTION_REVISION,
            "status": "success",
        }
        or not isinstance(references, Mapping)
        or set(references) != set(map(str, P_ITEM_PRODUCTION_REFERENCE_IDS))
    ):
        raise StaticReferenceHandoffError(
            "P-item production/source evidence deployment contract is invalid"
        )
    assert isinstance(references, Mapping)
    for business_id in P_ITEM_PRODUCTION_REFERENCE_IDS:
        reference = references.get(str(business_id))
        if (
            not isinstance(reference, Mapping)
            or set(reference)
            != {
                "dimensions",
                "git_blob_sha1",
                "path",
                "png_sha256",
                "rgb_pixel_sha256",
                "size",
            }
            or reference.get("dimensions") != [130, 130]
            or reference.get("path")
            != f"packages/gakumas-images/images/pItems/icons/{business_id}.png"
            or not _is_sha1(reference.get("git_blob_sha1"))
            or not _is_sha256(reference.get("png_sha256"))
            or not _is_sha256(reference.get("rgb_pixel_sha256"))
            or isinstance(reference.get("size"), bool)
            or not isinstance(reference.get("size"), int)
            or int(reference["size"]) < 1
        ):
            raise StaticReferenceHandoffError(
                "P-item production/source evidence reference contract is invalid"
            )


def _load_p_item_production_source_evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if (
        not resolved.is_file()
        or sha256_file(resolved) != P_ITEM_PRODUCTION_SOURCE_EVIDENCE_SHA256
    ):
        raise StaticReferenceHandoffError(
            "P-item production/source evidence is unavailable or not frozen"
        )
    evidence = _load_object(resolved, label="P-item production/source evidence")
    if resolved.read_bytes() != canonical_json_bytes(evidence):
        raise StaticReferenceHandoffError(
            "P-item production/source evidence is not canonical LF JSON"
        )
    _validate_p_item_production_source_evidence(evidence)
    return evidence


def _validate_p_item_candidate_source_evidence(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise StaticReferenceHandoffError("P-item candidate source is invalid")
    provenance = source.get("extension_provenance")
    if not isinstance(provenance, Mapping):
        raise StaticReferenceHandoffError(
            "P-item candidate extension provenance is invalid"
        )
    candidate_source = {
        key: source.get(key)
        for key in (
            "business_id_count",
            "revision",
            "sanitized_icons_sha256",
            "source_icons_sha256",
        )
    }
    if (
        candidate_source != evidence["source"]
        or provenance.get("base_gallery") != evidence["base_gallery"]
        or provenance.get("catalog") != evidence["catalog"]
        or provenance.get("official_rendered_source")
        != evidence["official_rendered_source"]
    ):
        raise StaticReferenceHandoffError(
            "P-item candidate provenance differs from frozen production/source evidence"
        )


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise StaticReferenceHandoffError(f"{label} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise StaticReferenceHandoffError(f"{label} root must be an object")
    return value


def _contract(component: str) -> Mapping[str, str]:
    try:
        return COMPONENT_CONTRACTS[component]
    except KeyError as error:
        raise StaticReferenceHandoffError(f"unsupported static reference component: {component}") from error


def _component_asset(
    component_root: Path,
    manifest: Mapping[str, Any],
    component: str,
) -> tuple[Path, str]:
    contract = _contract(component)
    gallery = manifest.get(contract["gallery_key"])
    if not isinstance(gallery, Mapping):
        raise StaticReferenceHandoffError(f"{component} gallery metadata is invalid")
    relative = gallery.get("path")
    declared_sha256 = gallery.get("sha256")
    if not isinstance(relative, str) or relative != contract["gallery_name"] or not _is_sha256(declared_sha256):
        raise StaticReferenceHandoffError(f"{component} gallery binding is invalid")
    root = component_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise StaticReferenceHandoffError(f"{component} gallery escapes its component root") from error
    if not path.is_file() or sha256_file(path).casefold() != declared_sha256.casefold():
        raise StaticReferenceHandoffError(f"{component} gallery SHA-256 mismatch")
    return path, sha256_file(path)


def _require_file_inventory(
    component: str,
    component_root: Path,
    *,
    promoted: bool,
) -> None:
    contract = _contract(component)
    expected = {"manifest.json", contract["gallery_name"]}
    if promoted:
        expected.add(REPORT_NAME)
    actual = {path.relative_to(component_root).as_posix() for path in component_root.rglob("*") if path.is_file()}
    if actual != expected:
        raise StaticReferenceHandoffError(
            f"{component} component file inventory is incompatible: extra={sorted(actual - expected)}, missing={sorted(expected - actual)}"
        )


def _candidate_manifest(
    component: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _contract(component)
    if "evaluation" in manifest:
        raise StaticReferenceHandoffError("candidate already contains evaluation metadata")
    expected_build = {
        "deterministic_output": "NPZ_AND_CANONICAL_LF_JSON",
        "tool_contract": contract["build_tool_contract"],
    }
    expected_handoff = {
        "reason": contract["candidate_reason"],
        "status": PENDING_STATUS,
    }
    handoff = manifest.get("production_handoff")
    if not isinstance(handoff, Mapping) or dict(handoff) != expected_handoff:
        raise StaticReferenceHandoffError("candidate was not emitted with the required PENDING_VALIDATION handoff")
    build = manifest.get("build")
    if not isinstance(build, Mapping) or dict(build) != expected_build:
        raise StaticReferenceHandoffError("candidate build tool contract is invalid")
    return copy.deepcopy(dict(manifest))


def _strict_array(
    arrays: Mapping[str, Any],
    name: str,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    value = np.asarray(arrays[name])
    if value.dtype != np.dtype(dtype) or value.shape != shape or value.dtype.hasobject:
        raise StaticReferenceHandoffError(f"array {name} has incompatible dtype or shape")
    return value


def _named_file_set_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest().upper()


def _load_cost_catalog_contract(
    catalog_dir: Path,
    gallery: GenericCostReferenceGallery,
    *,
    expected_file_set_sha256: str | None,
) -> dict[str, Any]:
    catalog_dir = catalog_dir.resolve()
    cards_path = catalog_dir / "skill_cards.json"
    customizations_path = catalog_dir / "customizations.json"
    try:
        cards = json.loads(cards_path.read_text(encoding="utf-8"))
        customizations = json.loads(customizations_path.read_text(encoding="utf-8"))
        catalog = ArenaEntityCatalog(cards, customizations)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise StaticReferenceHandoffError("arena cost catalog inputs are unavailable or invalid") from error
    catalog_ids = tuple(catalog.generic_cost_card_ids())
    gallery_ids = tuple(map(int, gallery.generic_card_ids))
    if catalog_ids != gallery_ids:
        raise StaticReferenceHandoffError("arena cost gallery ID order differs from the authoritative catalog")
    rows: list[dict[str, Any]] = []
    for index, card_id in enumerate(catalog_ids):
        descriptor = catalog.card_face_cost_descriptor(card_id)
        hypotheses = catalog.generic_cost_hypotheses(card_id)
        expected_descriptor = (
            gallery.cost_kinds[index],
            int(gallery.base_values[index]),
        )
        if descriptor != expected_descriptor or hypotheses is None:
            raise StaticReferenceHandoffError("arena cost catalog descriptor differs from the gallery")
        if tuple(hypotheses) != (
            expected_descriptor[1],
            max(0, expected_descriptor[1] - 2),
        ):
            raise StaticReferenceHandoffError("arena cost catalog hypotheses are incompatible")
        rows.append(
            {
                "business_id": card_id,
                "descriptor": list(descriptor),
                "hypotheses": list(hypotheses),
            }
        )
    contract_sha256 = hashlib.sha256(canonical_json_bytes({"generic_cost_rows": rows})).hexdigest().upper()
    file_set_sha256 = _named_file_set_sha256((cards_path, customizations_path))
    if (
        expected_file_set_sha256 is not None
        and file_set_sha256 != expected_file_set_sha256.upper()
    ):
        raise StaticReferenceHandoffError("arena cost catalog bytes differ from the builder-bound inputs")
    return {
        "catalog_contract_sha256": contract_sha256,
        "catalog_file_set_sha256": file_set_sha256,
        "generic_card_count": len(rows),
    }


def _descriptor(hex_value: object, *, channels: int) -> np.ndarray:
    if not isinstance(hex_value, str):
        raise StaticReferenceHandoffError("arena cost replay descriptor is invalid")
    try:
        unpacked = np.unpackbits(np.frombuffer(bytes.fromhex(hex_value), dtype=np.uint8))
    except ValueError as error:
        raise StaticReferenceHandoffError("arena cost replay descriptor is invalid") from error
    expected = channels * 24 * 24
    if unpacked.size != expected:
        raise StaticReferenceHandoffError("arena cost replay descriptor shape is invalid")
    return unpacked.reshape(channels, 24, 24) if channels > 1 else unpacked.reshape(24, 24)


def _shift_descriptor(source: np.ndarray, dx: int, dy: int) -> np.ndarray:
    result = np.zeros_like(source)
    source_y = slice(max(0, -dy), min(24, 24 - dy))
    target_y = slice(max(0, dy), min(24, 24 + dy))
    source_x = slice(max(0, -dx), min(24, 24 - dx))
    target_x = slice(max(0, dx), min(24, 24 + dx))
    result[..., target_y, target_x] = source[..., source_y, source_x]
    return result


def _evaluate_cost_external_evidence(
    gallery: GenericCostReferenceGallery,
    *,
    catalog_dir: Path,
    replay_paths: tuple[Path, ...],
    expected_catalog_file_set_sha256: str,
) -> dict[str, Any]:
    expected_names = set(COST_REPLAY_CONTRACT)
    resolved_replays = tuple(path.resolve() for path in replay_paths)
    if (
        len(resolved_replays) != len(expected_names)
        or {path.name for path in resolved_replays} != expected_names
        or any(not path.is_file() for path in resolved_replays)
    ):
        raise StaticReferenceHandoffError("arena cost promotion requires the three fixed descriptor replay families")
    if any(sha256_file(path) != COST_REPLAY_CONTRACT[path.name]["sha256"] for path in resolved_replays):
        raise StaticReferenceHandoffError("arena cost descriptor replay differs from the fixed evidence contract")
    if _named_file_set_sha256(resolved_replays) != COST_REPLAY_SET_SHA256:
        raise StaticReferenceHandoffError("arena cost descriptor replay set hash is incompatible")
    catalog_validation = _load_cost_catalog_contract(
        catalog_dir,
        gallery,
        expected_file_set_sha256=expected_catalog_file_set_sha256,
    )
    zero_capable_ids = tuple(
        int(card_id)
        for card_id, base_value in zip(
            gallery.generic_card_ids,
            gallery.base_values,
            strict=True,
        )
        if int(base_value) == 2
    )
    if zero_capable_ids != (159, 161, 389, 393, 423, 587, 617):
        raise StaticReferenceHandoffError("arena cost 2-to-0 catalog family changed without a new contract")
    replay_frame_count = 0
    expanded_positive_cases = 0
    stable_family_cases = 0
    first_carrier: np.ndarray | None = None
    first_symbol: np.ndarray | None = None
    replay_representatives: list[tuple[np.ndarray, np.ndarray]] = []
    for path in sorted(resolved_replays, key=lambda item: item.name):
        replay = _load_object(path, label="arena cost descriptor replay")
        frames = replay.get("frames")
        if (
            replay.get("schema_version") != 1
            or replay.get("allowed_values") != [2, 0]
            or replay.get("expected_value") != 0
            or replay.get("card_id") != COST_REPLAY_CONTRACT[path.name]["business_id"]
            or replay.get("card_id") not in zero_capable_ids
            or not isinstance(frames, list)
            or len(frames) < 3
        ):
            raise StaticReferenceHandoffError("arena cost descriptor replay contract is invalid")
        parsed_frames: list[tuple[np.ndarray, np.ndarray]] = []
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise StaticReferenceHandoffError("arena cost descriptor replay frame is invalid")
            carrier = _descriptor(frame.get("carrier_descriptor_hex"), channels=1)
            symbol = _descriptor(frame.get("symbol_descriptor_hex"), channels=2)
            parsed_frames.append((carrier, symbol))
            first_carrier = carrier if first_carrier is None else first_carrier
            first_symbol = symbol if first_symbol is None else first_symbol
        replay_representatives.append(parsed_frames[0])
        replay_frame_count += len(parsed_frames)
        for card_id in zero_capable_ids:
            predictions = []
            for carrier, symbol in parsed_frames:
                prediction = gallery.classify_mask(
                    carrier,
                    card_id=card_id,
                    allowed_values=(2, 0),
                    symbol_descriptor=symbol,
                )
                if prediction.status != "MEASURED" or prediction.value != 0 or prediction.evidence_mode != "explicit_zero_hole":
                    raise StaticReferenceHandoffError("arena cost real descriptor replay failed")
                predictions.append(prediction)
                expanded_positive_cases += 1
            if stable_generic_cost_value(tuple(predictions[:3])) != 0:
                raise StaticReferenceHandoffError("arena cost real descriptor stability replay failed")
            stable_family_cases += 1
    assert first_carrier is not None and first_symbol is not None
    if len(replay_representatives) != len(COST_REPLAY_CONTRACT):
        raise StaticReferenceHandoffError("arena cost descriptor replay representatives are incomplete")

    kernel = np.ones((2, 2), dtype=np.uint8)
    zero_perturbation_cases = 0
    zero_perturbation_rejections = 0
    for carrier, symbol in replay_representatives:
        zero_variants = [_shift_descriptor(symbol, dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        zero_variants.extend(np.stack([operation(channel, kernel) for channel in symbol]) for operation in (cv2.erode, cv2.dilate))
        for card_id in zero_capable_ids:
            for variant in zero_variants:
                prediction = gallery.classify_mask(
                    carrier,
                    card_id=card_id,
                    allowed_values=(2, 0),
                    symbol_descriptor=variant,
                )
                if prediction.status == "MEASURED" and prediction.value == 0:
                    zero_perturbation_cases += 1
                elif prediction.status != "MEASURED" and prediction.value is None:
                    zero_perturbation_rejections += 1
                else:
                    raise StaticReferenceHandoffError(
                        "arena cost zero perturbation was misclassified"
                    )

    fixed_two_negative_cases = 0
    for card_id, base_value, carrier, symbol in zip(
        gallery.generic_card_ids,
        gallery.base_values,
        gallery.base_masks,
        gallery.base_symbols,
        strict=True,
    ):
        if int(base_value) != 2:
            continue
        variants = [_shift_descriptor(symbol, dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        variants.extend(np.stack([operation(channel, kernel) for channel in symbol]) for operation in (cv2.erode, cv2.dilate))
        for variant in variants:
            prediction = gallery.classify_mask(
                carrier,
                card_id=int(card_id),
                allowed_values=(2, 0),
                symbol_descriptor=variant,
            )
            if prediction.value == 0:
                raise StaticReferenceHandoffError("arena cost fixed-two perturbation was misclassified as zero")
            fixed_two_negative_cases += 1

    open_ring = np.zeros((24, 24), dtype=np.uint8)
    cv2.circle(open_ring, (12, 12), 8, 1, 4)
    open_ring[:, 11:14] = 0
    rectangle = np.zeros((24, 24), dtype=np.uint8)
    cv2.rectangle(rectangle, (3, 2), (20, 21), 1, -1)
    cv2.rectangle(rectangle, (7, 6), (16, 17), 0, -1)
    double_hole = np.zeros((24, 24), dtype=np.uint8)
    cv2.ellipse(double_hole, (12, 12), (11, 11), 0, 0, 360, 1, -1)
    cv2.rectangle(double_hole, (4, 7), (10, 17), 0, -1)
    cv2.rectangle(double_hole, (12, 7), (18, 17), 0, -1)
    topology_negative_cases = 0
    for card_id in zero_capable_ids:
        for shape in (open_ring, rectangle, double_hole):
            symbol = np.zeros((2, 24, 24), dtype=np.uint8)
            symbol[0] = shape
            prediction = gallery.classify_mask(
                first_carrier,
                card_id=card_id,
                allowed_values=(2, 0),
                symbol_descriptor=symbol,
            )
            if prediction.value == 0:
                raise StaticReferenceHandoffError("arena cost invalid topology was misclassified as zero")
            topology_negative_cases += 1

    return {
        **catalog_validation,
        "descriptor_fixture_set_sha256": COST_REPLAY_SET_SHA256,
        "descriptor_replay": {
            "expanded_positive_cases": expanded_positive_cases,
            "family_count": len(resolved_replays),
            "frame_count": replay_frame_count,
            "stable_family_card_cases": stable_family_cases,
            "zero_capable_business_ids": list(zero_capable_ids),
        },
        "negative_perturbation": {
            "fixed_two_rejection_cases": fixed_two_negative_cases,
            "invalid_topology_rejection_cases": topology_negative_cases,
            "zero_fail_closed_cases": zero_perturbation_rejections,
            "zero_measured_acceptance_cases": zero_perturbation_cases,
            "zero_total_cases": zero_perturbation_cases
            + zero_perturbation_rejections,
        },
    }


def _evaluate_cost(
    component_root: Path,
    manifest: Mapping[str, Any],
    gallery_path: Path,
) -> dict[str, Any]:
    gallery_info = manifest.get("gallery")
    runtime = manifest.get("runtime")
    source = manifest.get("source")
    if not isinstance(gallery_info, Mapping) or not isinstance(runtime, Mapping):
        raise StaticReferenceHandoffError("arena cost manifest metadata is invalid")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("component") != "task410_arena_card_face_cost_reference"
        or not isinstance(source, Mapping)
        or source.get("repository") != "https://github.com/surisuririsu/gakumas-tools"
        or source.get("license") != "BSD-3-Clause"
        or not _is_sha1(source.get("revision"))
        or not _is_sha256(source.get("catalog_file_set_sha256"))
        or not _is_sha256(source.get("source_set_sha256"))
    ):
        raise StaticReferenceHandoffError("arena cost source provenance is invalid")
    if (
        gallery_info.get("normalization_size") != [96, 96]
        or gallery_info.get("roi") != [64, 62, 32, 34]
        or gallery_info.get("carrier_shape") != [24, 24]
        or gallery_info.get("symbol_shape") != [2, 24, 24]
    ):
        raise StaticReferenceHandoffError("arena cost gallery geometry is incompatible")
    if dict(runtime) != COST_RUNTIME_CONTRACT:
        raise StaticReferenceHandoffError("arena cost runtime contract is invalid")
    generic_count = gallery_info.get("generic_card_count")
    template_count = gallery_info.get("template_count")
    if (
        isinstance(generic_count, bool)
        or not isinstance(generic_count, int)
        or generic_count < 1
        or isinstance(template_count, bool)
        or not isinstance(template_count, int)
        or template_count < 1
    ):
        raise StaticReferenceHandoffError("arena cost gallery counts are invalid")
    expected_files = {
        "generic_card_ids",
        "base_values",
        "cost_kinds",
        "base_masks",
        "template_values",
        "template_kinds",
        "template_masks",
        "base_symbols",
        "template_symbols",
    }
    try:
        with np.load(gallery_path, allow_pickle=False) as payload:
            if set(payload.files) != expected_files:
                raise StaticReferenceHandoffError("arena cost gallery array inventory is incompatible")
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
    except StaticReferenceHandoffError:
        raise
    except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
        raise StaticReferenceHandoffError("arena cost gallery is unavailable") from error

    ids = _strict_array(
        arrays,
        "generic_card_ids",
        dtype=np.int16,
        shape=(generic_count,),
    )
    base_values = _strict_array(
        arrays,
        "base_values",
        dtype=np.int8,
        shape=(generic_count,),
    )
    cost_kinds = arrays["cost_kinds"]
    if cost_kinds.shape != (generic_count,) or cost_kinds.dtype.kind != "U":
        raise StaticReferenceHandoffError("arena cost kind array is incompatible")
    _strict_array(
        arrays,
        "base_masks",
        dtype=np.uint8,
        shape=(generic_count, 24, 24),
    )
    _strict_array(
        arrays,
        "base_symbols",
        dtype=np.uint8,
        shape=(generic_count, 2, 24, 24),
    )
    template_values = _strict_array(
        arrays,
        "template_values",
        dtype=np.int8,
        shape=(template_count,),
    )
    template_kinds = arrays["template_kinds"]
    if template_kinds.shape != (template_count,) or template_kinds.dtype.kind != "U":
        raise StaticReferenceHandoffError("arena cost template-kind array is incompatible")
    _strict_array(
        arrays,
        "template_masks",
        dtype=np.uint8,
        shape=(template_count, 24, 24),
    )
    _strict_array(
        arrays,
        "template_symbols",
        dtype=np.uint8,
        shape=(template_count, 2, 24, 24),
    )
    if (
        len(set(map(int, ids))) != generic_count
        or any(int(value) < 1 for value in ids)
        or list(map(int, ids)) != sorted(map(int, ids))
        or any(str(value) not in {"cost", "stamina"} for value in cost_kinds)
        or any(int(value) < 1 for value in base_values)
    ):
        raise StaticReferenceHandoffError("arena cost base identities are invalid")
    template_pairs = tuple((str(kind), int(value)) for kind, value in zip(template_kinds, template_values, strict=True))
    if (
        len(template_pairs) != len(set(template_pairs))
        or any(kind not in {"cost", "stamina"} or value < 1 for kind, value in template_pairs)
        or any(
            value not in {0, 1}
            for name in ("base_masks", "template_masks", "base_symbols", "template_symbols")
            for value in np.unique(arrays[name])
        )
    ):
        raise StaticReferenceHandoffError("arena cost templates or descriptors are invalid")

    try:
        gallery = GenericCostReferenceGallery.load(component_root)
    except (GenericCostReferenceError, OSError, KeyError, TypeError, ValueError) as error:
        raise StaticReferenceHandoffError("arena cost gallery cannot be loaded by the production consumer") from error
    base_pass = 0
    alternate_pass = 0
    blank_rejections = 0
    sparse_rejections = 0
    for index, raw_card_id in enumerate(gallery.generic_card_ids):
        card_id = int(raw_card_id)
        base_value = int(gallery.base_values[index])
        alternate = max(0, base_value - 2)
        allowed = (base_value, alternate)
        prediction = gallery.classify_mask(
            gallery.base_masks[index],
            card_id=card_id,
            allowed_values=allowed,
            symbol_descriptor=gallery.base_symbols[index],
        )
        if prediction.status != "MEASURED" or prediction.value != base_value:
            raise StaticReferenceHandoffError("arena cost base self-replay failed")
        base_pass += 1
        if alternate > 0:
            template_index = gallery._template_indices.get((gallery.cost_kinds[index], alternate))
            if template_index is None:
                raise StaticReferenceHandoffError("arena cost alternate hypothesis lacks a template")
            prediction = gallery.classify_mask(
                gallery.template_masks[template_index],
                card_id=card_id,
                allowed_values=allowed,
                symbol_descriptor=gallery.template_symbols[template_index],
            )
            if prediction.status != "MEASURED" or prediction.value != alternate:
                raise StaticReferenceHandoffError("arena cost alternate self-replay failed")
            alternate_pass += 1
        blank = gallery.classify_mask(
            np.zeros((24, 24), dtype=np.uint8),
            card_id=card_id,
            allowed_values=allowed,
        )
        if blank.status == "MEASURED":
            raise StaticReferenceHandoffError("arena cost blank ROI was accepted")
        blank_rejections += 1
        sparse = np.zeros((24, 24), dtype=np.uint8)
        sparse.flat[:31] = 1
        sparse_prediction = gallery.classify_mask(
            sparse,
            card_id=card_id,
            allowed_values=allowed,
        )
        if sparse_prediction.status == "MEASURED":
            raise StaticReferenceHandoffError("arena cost sparse ROI was accepted")
        sparse_rejections += 1

    return {
        "array_count": len(expected_files),
        "base_self_replay": {"passed": base_pass, "sample_count": generic_count},
        "blank_rejection": {"passed": blank_rejections, "sample_count": generic_count},
        "generic_card_count": generic_count,
        "independent_live_holdout": "PENDING",
        "rule_version": runtime["rule_version"],
        "source_revision": source["revision"],
        "source_set_sha256": str(source["source_set_sha256"]).upper(),
        "sparse_rejection": {"passed": sparse_rejections, "sample_count": generic_count},
        "template_count": template_count,
        "nonzero_alternate_self_replay": {
            "passed": alternate_pass,
            "sample_count": alternate_pass,
        },
    }


def _decode_pngs(
    ids: np.ndarray,
    payload: np.ndarray,
    offsets: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[int, int], ...], dict[int, str]]:
    images: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
    rgb_hashes: dict[int, str] = {}
    for index, raw_business_id in enumerate(ids):
        encoded = bytes(payload[int(offsets[index]) : int(offsets[index + 1])])
        try:
            with Image.open(io.BytesIO(encoded)) as image:
                if image.format != "PNG":
                    raise StaticReferenceHandoffError("P-item reference is not PNG")
                image.load()
                if image.info or image.getexif():
                    raise StaticReferenceHandoffError("P-item reference PNG contains metadata")
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except StaticReferenceHandoffError:
            raise
        except Exception as error:
            raise StaticReferenceHandoffError("P-item reference PNG is not decodable") from error
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 8:
            raise StaticReferenceHandoffError("P-item reference image shape is invalid")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or not np.array_equal(decoded[:, :, ::-1], rgb):
            raise StaticReferenceHandoffError("P-item reference differs between production and provenance decoders")
        business_id = int(raw_business_id)
        images.append(decoded)
        shapes.append((int(rgb.shape[1]), int(rgb.shape[0])))
        rgb_hashes[business_id] = hashlib.sha256(rgb.tobytes()).hexdigest().upper()
    return tuple(images), tuple(shapes), rgb_hashes


def _evaluate_p_item(
    component_root: Path,
    manifest: Mapping[str, Any],
    gallery_path: Path,
    *,
    base_gallery_path: Path | None,
    require_base_bytes: bool,
) -> dict[str, Any]:
    gallery_info = manifest.get("reference_gallery")
    source = manifest.get("source")
    runtime = manifest.get("runtime")
    if not isinstance(gallery_info, Mapping) or not isinstance(source, Mapping) or not isinstance(runtime, Mapping):
        raise StaticReferenceHandoffError("P-item manifest metadata is invalid")
    if manifest.get("schema_version") != 1 or manifest.get("recognizer") != "p_item_rendered_reference_v1":
        raise StaticReferenceHandoffError("P-item recognizer is incompatible")
    count = gallery_info.get("business_id_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise StaticReferenceHandoffError("P-item gallery count is invalid")
    if gallery_info.get("provisional_business_ids") != []:
        raise StaticReferenceHandoffError("P-item promoted fixed references must not be provisional")
    if (
        gallery_info.get("color_order") != "BGR"
        or gallery_info.get("image_shape") != "original_png_dimensions"
        or source.get("business_id_count") != count
        or not _is_sha256(source.get("source_icons_sha256"))
        or not _is_sha256(source.get("sanitized_icons_sha256"))
        or source.get("plan_ambiguous_business_ids") != [406, 407, 408]
        or source.get("plan_to_business_id") != {"anomaly": 408, "logic": 407, "sense": 406}
    ):
        raise StaticReferenceHandoffError("P-item source or gallery contract is invalid")
    if dict(runtime) != P_ITEM_RUNTIME_CONTRACT:
        raise StaticReferenceHandoffError("P-item runtime contract is invalid")
    try:
        with np.load(gallery_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"p_item_ids", "png_bytes", "png_offsets"}:
                raise StaticReferenceHandoffError("P-item gallery array inventory is incompatible")
            ids = np.asarray(arrays["p_item_ids"])
            png_bytes = np.asarray(arrays["png_bytes"])
            offsets = np.asarray(arrays["png_offsets"])
    except StaticReferenceHandoffError:
        raise
    except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
        raise StaticReferenceHandoffError("P-item gallery is unavailable") from error
    if (
        ids.dtype != np.dtype(np.int32)
        or ids.shape != (count,)
        or png_bytes.dtype != np.dtype(np.uint8)
        or png_bytes.ndim != 1
        or offsets.dtype != np.dtype(np.int64)
        or offsets.shape != (count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(png_bytes)
        or np.any(np.diff(offsets) <= 0)
        or len(set(map(int, ids))) != count
        or any(int(value) < 1 for value in ids)
        or list(map(int, ids)) != sorted(map(int, ids))
    ):
        raise StaticReferenceHandoffError("P-item gallery IDs, payload, or offsets are incompatible")
    images, shapes, rgb_hashes = _decode_pngs(ids, png_bytes, offsets)
    if gallery_info.get("image_shape_count") != len(set(shapes)):
        raise StaticReferenceHandoffError("P-item image-shape count is stale")
    encoded_digest = hashlib.sha256()
    for index, business_id in enumerate(ids):
        encoded = bytes(png_bytes[int(offsets[index]) : int(offsets[index + 1])])
        encoded_digest.update(str(int(business_id)).encode("ascii"))
        encoded_digest.update(hashlib.sha256(encoded).digest())
    if encoded_digest.hexdigest().upper() != str(source["sanitized_icons_sha256"]).upper():
        raise StaticReferenceHandoffError("P-item sanitized source-set hash is stale")
    try:
        production_gallery = PItemRenderedReferenceGallery.load(component_root)
    except (PItemReferenceError, OSError, KeyError, TypeError, ValueError) as error:
        raise StaticReferenceHandoffError("P-item gallery cannot be loaded by the production consumer") from error
    if tuple(production_gallery.p_item_ids) != tuple(map(int, ids)):
        raise StaticReferenceHandoffError("P-item production loader changed ID order")

    provenance = source.get("extension_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("schema_version") != 2:
        raise StaticReferenceHandoffError("P-item extension provenance is required and invalid")
    base = provenance.get("base_gallery")
    catalog = provenance.get("catalog")
    official = provenance.get("official_rendered_source")
    validation = provenance.get("validation")
    if not all(isinstance(value, Mapping) for value in (base, catalog, official, validation)):
        raise StaticReferenceHandoffError("P-item extension provenance is incomplete")
    assert isinstance(base, Mapping)
    assert isinstance(catalog, Mapping)
    assert isinstance(official, Mapping)
    assert isinstance(validation, Mapping)
    replaced = base.get("replaced_business_ids")
    included = catalog.get("included_business_ids")
    declared_changed = validation.get("source_replacement_changed_business_ids")
    if (
        not isinstance(replaced, list)
        or not replaced
        or any(isinstance(value, bool) or not isinstance(value, int) for value in replaced)
        or replaced != sorted(set(replaced))
        or any(value not in set(map(int, ids)) for value in replaced)
        or included != replaced
        or declared_changed != replaced
        or base.get("business_id_count") != count
        or base.get("inherited_business_id_count") != count - len(replaced)
        or validation.get("metadata_removed_business_id_count") != count
    ):
        raise StaticReferenceHandoffError("P-item extension replacement lineage is invalid")
    changed_ids = tuple(replaced)
    base_sha256_value = base.get("sha256")
    if not _is_sha256(base_sha256_value):
        raise StaticReferenceHandoffError("P-item base gallery SHA-256 is invalid")
    base_sha256 = str(base_sha256_value).upper()

    catalog_revision = catalog.get("revision")
    production_deployment = official.get("production_deployment")
    if (
        catalog.get("repository") != "https://github.com/surisuririsu/gakumas-tools"
        or catalog.get("path") != "packages/gakumas-data/json/p_items.json"
        or not _is_sha1(catalog_revision)
        or not _is_sha1(catalog.get("p_items_git_blob_sha1"))
        or not _is_sha256(catalog.get("p_items_sha256"))
        or official.get("repository") != "https://github.com/surisuririsu/gakumas-tools"
        or official.get("package_path") != "packages/gakumas-images"
        or official.get("gk_img_gitlink_path") != "gk-img"
        or official.get("gk_img_gitlink_repository") != "https://github.com/surisuririsu/gk-img"
        or not _is_sha1(official.get("gk_img_gitlink_revision"))
        or not _is_sha1(official.get("first_added_revision"))
        or not _is_sha1(official.get("first_successful_production_revision"))
        or not isinstance(production_deployment, Mapping)
        or isinstance(production_deployment.get("deployment_id"), bool)
        or not isinstance(production_deployment.get("deployment_id"), int)
        or int(production_deployment["deployment_id"]) < 1
        or production_deployment.get("status") != "success"
        or production_deployment.get("revision") != catalog_revision
        or source.get("revision") != f"base-gallery-sha256@{base_sha256}+gakumas-tools@{catalog_revision}"
    ):
        raise StaticReferenceHandoffError("P-item source revision or deployment provenance is invalid")

    references = official.get("references")
    if not isinstance(references, Mapping) or set(references) != set(map(str, replaced)):
        raise StaticReferenceHandoffError("P-item official rendered references are incomplete")
    id_indexes = {int(value): index for index, value in enumerate(ids)}
    for business_id in replaced:
        reference = references.get(str(business_id))
        if not isinstance(reference, Mapping):
            raise StaticReferenceHandoffError("P-item official rendered reference is invalid")
        dimensions = reference.get("dimensions")
        if (
            set(reference) != {"dimensions", "git_blob_sha1", "path", "png_sha256", "rgb_pixel_sha256", "size"}
            or dimensions != list(shapes[id_indexes[business_id]])
            or reference.get("path") != f"packages/gakumas-images/images/pItems/icons/{business_id}.png"
            or not _is_sha1(reference.get("git_blob_sha1"))
            or not _is_sha256(reference.get("png_sha256"))
            or not _is_sha256(reference.get("rgb_pixel_sha256"))
            or str(reference["rgb_pixel_sha256"]).upper() != rgb_hashes[business_id]
            or isinstance(reference.get("size"), bool)
            or not isinstance(reference.get("size"), int)
            or int(reference["size"]) < 1
        ):
            raise StaticReferenceHandoffError("P-item official rendered reference pixels are stale")

    live = validation.get("live_jjc_calibration")
    declared_rankings = validation.get("full_gallery_64px")
    if (
        not isinstance(live, Mapping)
        or live.get("status") != "PENDING"
        or live.get("runtime_detail_authority") is not False
        or not isinstance(declared_rankings, Mapping)
        or set(declared_rankings) != set(map(str, replaced)) | {"coarse_fine_guard"}
        or declared_rankings.get("coarse_fine_guard") != "fails_safe_to_full_gallery"
    ):
        raise StaticReferenceHandoffError("P-item validation provenance is invalid")

    if require_base_bytes and base_gallery_path is None:
        raise StaticReferenceHandoffError("P-item extension promotion requires the immutable base gallery")
    if base_gallery_path is not None:
        base_gallery_path = base_gallery_path.resolve()
        if not base_gallery_path.is_file() or sha256_file(base_gallery_path) != base_sha256:
            raise StaticReferenceHandoffError("P-item base gallery SHA-256 mismatch")
        try:
            with np.load(base_gallery_path, allow_pickle=False) as base_arrays:
                if set(base_arrays.files) != {"p_item_ids", "png_bytes", "png_offsets"}:
                    raise StaticReferenceHandoffError("P-item base gallery inventory is invalid")
                base_ids = np.asarray(base_arrays["p_item_ids"])
                base_payload = np.asarray(base_arrays["png_bytes"])
                base_offsets = np.asarray(base_arrays["png_offsets"])
            base_images, _, _ = _decode_pngs(base_ids, base_payload, base_offsets)
        except StaticReferenceHandoffError:
            raise
        except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
            raise StaticReferenceHandoffError("P-item base gallery is invalid") from error
        if tuple(map(int, base_ids)) != tuple(map(int, ids)):
            raise StaticReferenceHandoffError("P-item extension changed the inherited ID order")
        actual_changed = tuple(
            int(ids[index])
            for index, (before, after) in enumerate(zip(base_images, images, strict=True))
            if before.shape != after.shape or not np.array_equal(before, after)
        )
        if actual_changed != changed_ids:
            raise StaticReferenceHandoffError("P-item extension changed undeclared reference pixels")

    sample_ids = changed_ids or tuple(map(int, ids[: min(2, count)]))
    minimum_margin = float("inf")
    for business_id in sample_ids:
        image = images[list(map(int, ids)).index(business_id)]
        query = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        ranking = production_gallery._rank_full(query, (0, 0, 64, 64))
        if len(ranking) < 2 or ranking[0].p_item_id != business_id:
            raise StaticReferenceHandoffError("P-item fixed-reference ranking failed")
        margin = float(ranking[0].similarity - ranking[1].similarity)
        if margin <= production_gallery.runtime.identity_minimum_margin:
            raise StaticReferenceHandoffError("P-item fixed-reference ranking margin is below runtime threshold")
        declared = declared_rankings.get(str(business_id))
        if (
            not isinstance(declared, Mapping)
            or set(declared)
            != {
                "margin",
                "top1_business_id",
                "top1_similarity",
                "top2_business_id",
                "top2_similarity",
            }
            or declared.get("top1_business_id") != business_id
            or declared.get("top2_business_id") != ranking[1].p_item_id
            or any(not _is_number(declared.get(key)) for key in ("margin", "top1_similarity", "top2_similarity"))
            or float(declared["margin"]) <= production_gallery.runtime.identity_minimum_margin
            or abs(float(declared["margin"]) - margin) > 0.00001
            or abs(float(declared["top1_similarity"]) - ranking[0].similarity) > 0.00001
            or abs(float(declared["top2_similarity"]) - ranking[1].similarity) > 0.00001
        ):
            raise StaticReferenceHandoffError("P-item declared full-gallery ranking is stale")
        minimum_margin = min(minimum_margin, margin)

    return {
        "array_count": 3,
        "base_gallery_sha256": base_sha256,
        "business_id_count": count,
        "changed_business_ids": list(changed_ids),
        "decoded_png_count": len(images),
        "independent_live_holdout": "PENDING",
        "metadata_free_png_count": len(images),
        "self_ranking": {
            "minimum_top1_margin": round(minimum_margin, 6),
            "passed": len(sample_ids),
            "sample_count": len(sample_ids),
        },
    }


def _p_item_source_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("ascii"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest().upper()


def _verify_p_item_production_source_files(
    evidence: Mapping[str, Any],
    *,
    base_gallery_path: Path,
    source_dir: Path,
    catalog_path: Path,
) -> tuple[tuple[Path, ...], dict[int, np.ndarray], list[Mapping[str, Any]]]:
    base = evidence["base_gallery"]
    catalog = evidence["catalog"]
    official = evidence["official_rendered_source"]
    source = evidence["source"]
    base_gallery_path = base_gallery_path.resolve()
    source_dir = source_dir.resolve()
    catalog_path = catalog_path.resolve()
    if (
        not base_gallery_path.is_file()
        or sha256_file(base_gallery_path) != str(base["sha256"]).upper()
    ):
        raise StaticReferenceHandoffError(
            "P-item immutable base gallery differs from frozen production/source evidence"
        )
    if not source_dir.is_dir() or not catalog_path.is_file():
        raise StaticReferenceHandoffError("P-item promotion source inputs are unavailable")
    all_source_files = tuple(path for path in source_dir.iterdir() if path.is_file())
    source_paths = tuple(
        sorted(
            (
                path
                for path in all_source_files
                if path.suffix.casefold() == ".png" and path.stem.isdigit()
            ),
            key=lambda path: int(path.stem),
        )
    )
    if (
        len(source_paths) != len(all_source_files)
        or len(source_paths) != source["business_id_count"]
        or _p_item_source_digest(source_paths)
        != str(source["source_icons_sha256"]).upper()
    ):
        raise StaticReferenceHandoffError(
            "P-item promotion source bytes differ from frozen production/source evidence"
        )
    source_images: dict[int, np.ndarray] = {}
    for path in source_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise StaticReferenceHandoffError("P-item promotion source PNG is invalid")
        source_images[int(path.stem)] = image

    catalog_bytes = catalog_path.read_bytes()
    if (
        catalog_path.name != "p_items.json"
        or hashlib.sha256(catalog_bytes).hexdigest().upper()
        != str(catalog["p_items_sha256"]).upper()
        or _git_blob_sha1(catalog_bytes).casefold()
        != str(catalog["p_items_git_blob_sha1"]).casefold()
    ):
        raise StaticReferenceHandoffError(
            "P-item catalog bytes differ from frozen production/source evidence"
        )
    try:
        catalog_rows = json.loads(catalog_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StaticReferenceHandoffError("P-item catalog JSON is invalid") from error
    if not isinstance(catalog_rows, list) or any(
        not isinstance(row, Mapping) for row in catalog_rows
    ):
        raise StaticReferenceHandoffError("P-item catalog rows are invalid")

    references = official["references"]
    for business_id in P_ITEM_PRODUCTION_REFERENCE_IDS:
        path = source_dir / f"{business_id}.png"
        reference = references[str(business_id)]
        if not path.is_file() or business_id not in source_images:
            raise StaticReferenceHandoffError(
                "P-item official rendered source is unavailable"
            )
        raw = path.read_bytes()
        image = source_images[business_id]
        rgb_sha256 = hashlib.sha256(image[:, :, ::-1].tobytes()).hexdigest().upper()
        if (
            len(raw) != reference["size"]
            or hashlib.sha256(raw).hexdigest().upper()
            != str(reference["png_sha256"]).upper()
            or _git_blob_sha1(raw).casefold()
            != str(reference["git_blob_sha1"]).casefold()
            or rgb_sha256 != str(reference["rgb_pixel_sha256"]).upper()
        ):
            raise StaticReferenceHandoffError(
                "P-item official rendered source bytes differ from frozen evidence"
            )
    return source_paths, source_images, catalog_rows


def _evaluate_p_item_external_evidence(
    gallery_path: Path,
    *,
    base_gallery_path: Path,
    source_dir: Path,
    catalog_path: Path,
    production_source_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    source = production_source_evidence["source"]
    base = production_source_evidence["base_gallery"]
    catalog = production_source_evidence["catalog"]
    source_paths, source_images, catalog_rows = (
        _verify_p_item_production_source_files(
            production_source_evidence,
            base_gallery_path=base_gallery_path,
            source_dir=source_dir,
            catalog_path=catalog_path,
        )
    )
    try:
        with np.load(gallery_path, allow_pickle=False) as arrays:
            ids = np.asarray(arrays["p_item_ids"])
            payload = np.asarray(arrays["png_bytes"])
            offsets = np.asarray(arrays["png_offsets"])
        gallery_images, _, _ = _decode_pngs(ids, payload, offsets)
    except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
        raise StaticReferenceHandoffError("P-item promotion gallery is invalid") from error
    if tuple(int(path.stem) for path in source_paths) != tuple(map(int, ids)):
        raise StaticReferenceHandoffError("P-item promotion source IDs differ from the fixed gallery")
    encoded_digest = hashlib.sha256()
    for index, path in enumerate(source_paths):
        encoded = bytes(payload[int(offsets[index]) : int(offsets[index + 1])])
        encoded_digest.update(path.stem.encode("ascii"))
        encoded_digest.update(hashlib.sha256(encoded).digest())
        if not np.array_equal(source_images[int(path.stem)], gallery_images[index]):
            raise StaticReferenceHandoffError("P-item promotion source pixels differ from the fixed gallery")
    if encoded_digest.hexdigest().upper() != str(
        source["sanitized_icons_sha256"]
    ).upper():
        raise StaticReferenceHandoffError(
            "P-item sanitized gallery bytes differ from frozen production/source evidence"
        )
    rows_by_id = {
        int(row["id"]): row
        for row in catalog_rows
        if isinstance(row.get("id"), int)
        and not isinstance(row.get("id"), bool)
    }
    included = tuple(catalog["included_business_ids"])
    if any(business_id not in rows_by_id or rows_by_id[business_id].get("mode") != "stage" for business_id in included):
        raise StaticReferenceHandoffError("P-item included catalog rows do not satisfy the stage contract")
    excluded_rows = catalog.get("excluded_current_rows")
    if not isinstance(excluded_rows, list):
        raise StaticReferenceHandoffError("P-item excluded catalog rows are invalid")
    for declaration in excluded_rows:
        if not isinstance(declaration, Mapping):
            raise StaticReferenceHandoffError("P-item excluded catalog row is invalid")
        business_id = declaration.get("business_id")
        row = rows_by_id.get(business_id)
        if (
            row is None
            or declaration.get("mode") != row.get("mode")
            or declaration.get("name") != row.get("name")
            or declaration.get("source_type") != row.get("sourceType")
            or not isinstance(declaration.get("reason"), str)
            or not declaration["reason"]
        ):
            raise StaticReferenceHandoffError("P-item excluded catalog declaration is stale")

    return {
        "base_gallery_sha256": str(base["sha256"]).upper(),
        "catalog_git_blob_sha1": str(catalog["p_items_git_blob_sha1"]).upper(),
        "catalog_sha256": str(catalog["p_items_sha256"]).upper(),
        "official_reference_count": len(included),
        "production_source_evidence_sha256": P_ITEM_PRODUCTION_SOURCE_EVIDENCE_SHA256,
        "source_icon_count": len(source_paths),
        "source_icons_sha256": str(source["source_icons_sha256"]).upper(),
        "source_pixel_match_count": len(source_paths),
    }


def evaluate_component(
    component: str,
    component_root: Path,
    manifest: Mapping[str, Any],
    *,
    base_gallery_path: Path | None = None,
    require_base_bytes: bool = False,
) -> tuple[str, dict[str, Any]]:
    root = component_root.resolve()
    gallery_path, gallery_sha256 = _component_asset(root, manifest, component)
    if component == "arena_cost_reference":
        validation = _evaluate_cost(root, manifest, gallery_path)
    elif component == "p_item_reference":
        validation = _evaluate_p_item(
            root,
            manifest,
            gallery_path,
            base_gallery_path=base_gallery_path,
            require_base_bytes=require_base_bytes,
        )
    else:  # Defensive; _component_asset has already checked this.
        raise StaticReferenceHandoffError(f"unsupported component: {component}")
    return gallery_sha256, validation


def promote_component(
    component: str,
    candidate_root: Path,
    output_dir: Path,
    *,
    base_gallery_path: Path | None = None,
    cost_catalog_dir: Path | None = None,
    cost_replay_paths: tuple[Path, ...] = (),
    p_item_source_dir: Path | None = None,
    p_item_catalog_path: Path | None = None,
    p_item_production_source_evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate and atomically create its promoted directory."""

    production_source_evidence: dict[str, Any] | None = None
    if component == "p_item_reference":
        if (
            base_gallery_path is None
            or p_item_source_dir is None
            or p_item_catalog_path is None
            or p_item_production_source_evidence_path is None
        ):
            raise StaticReferenceHandoffError(
                "P-item promotion requires immutable base, source, catalog, and frozen production/source evidence inputs"
            )
        production_source_evidence = _load_p_item_production_source_evidence(
            p_item_production_source_evidence_path
        )
    candidate_root = candidate_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise StaticReferenceHandoffError(f"promoted output directory already exists: {output_dir}")
    _require_file_inventory(component, candidate_root, promoted=False)
    manifest_path = candidate_root / "manifest.json"
    manifest = _load_object(manifest_path, label="candidate manifest")
    if manifest_path.read_bytes() != canonical_json_bytes(manifest):
        raise StaticReferenceHandoffError("candidate manifest does not use the builder canonical-LF byte contract")
    candidate = _candidate_manifest(component, manifest)
    if component == "p_item_reference":
        assert production_source_evidence is not None
        _validate_p_item_candidate_source_evidence(
            candidate,
            production_source_evidence,
        )
    gallery_path, gallery_sha256 = _component_asset(candidate_root, candidate, component)
    _, validation = evaluate_component(
        component,
        candidate_root,
        candidate,
        base_gallery_path=base_gallery_path,
        require_base_bytes=True,
    )
    promotion_evidence: dict[str, Any]
    if component == "arena_cost_reference":
        if cost_catalog_dir is None:
            raise StaticReferenceHandoffError("arena cost promotion requires the authoritative catalog directory")
        try:
            production_gallery = GenericCostReferenceGallery.load(candidate_root)
        except (GenericCostReferenceError, OSError, KeyError, TypeError, ValueError) as error:
            raise StaticReferenceHandoffError("arena cost gallery cannot be loaded by the production consumer") from error
        promotion_evidence = _evaluate_cost_external_evidence(
            production_gallery,
            catalog_dir=cost_catalog_dir,
            replay_paths=cost_replay_paths,
            expected_catalog_file_set_sha256=str(candidate["source"]["catalog_file_set_sha256"]),
        )
    elif component == "p_item_reference":
        assert base_gallery_path is not None
        assert p_item_source_dir is not None
        assert p_item_catalog_path is not None
        assert production_source_evidence is not None
        promotion_evidence = _evaluate_p_item_external_evidence(
            gallery_path,
            base_gallery_path=base_gallery_path,
            source_dir=p_item_source_dir,
            catalog_path=p_item_catalog_path,
            production_source_evidence=production_source_evidence,
        )
    else:  # Defensive; evaluate_component already rejected this.
        raise StaticReferenceHandoffError(f"unsupported component: {component}")
    candidate_sha256 = hashlib.sha256(canonical_json_bytes(candidate)).hexdigest().upper()
    report = {
        "artifact": {
            "path": gallery_path.name,
            "sha256": gallery_sha256,
        },
        "candidate_manifest_sha256": candidate_sha256,
        "component": component,
        "privacy": "PUBLIC_SAFE_AGGREGATE_ONLY",
        "promotion_evidence": promotion_evidence,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "tool_contract": PROMOTION_TOOL_CONTRACT,
        "validation": validation,
    }
    contract = _contract(component)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(
        f".{output_dir.name}-{secrets.token_hex(8)}.tmp"
    )
    temporary.mkdir()
    try:
        copied_gallery = temporary / gallery_path.name
        shutil.copyfile(gallery_path, copied_gallery)
        if sha256_file(copied_gallery) != gallery_sha256:
            raise StaticReferenceHandoffError("promoted gallery differs from the evaluated candidate")
        report_path = temporary / REPORT_NAME
        report_path.write_bytes(canonical_json_bytes(report))
        promoted = copy.deepcopy(candidate)
        promoted["evaluation"] = {
            "path": REPORT_NAME,
            "sha256": sha256_file(report_path),
            "status": "PASS",
            "tool_contract": PROMOTION_TOOL_CONTRACT,
        }
        promoted["production_handoff"] = {
            "reason": contract["promoted_reason"],
            "status": PROMOTED_STATUS,
        }
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(promoted))
        os.replace(temporary, output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return promoted


def _cost_catalog_contract_from_gallery(component_root: Path) -> str:
    try:
        gallery = GenericCostReferenceGallery.load(component_root)
    except (GenericCostReferenceError, OSError, KeyError, TypeError, ValueError) as error:
        raise StaticReferenceHandoffError("arena cost catalog contract cannot be reconstructed") from error
    rows = [
        {
            "business_id": int(card_id),
            "descriptor": [str(kind), int(base_value)],
            "hypotheses": [int(base_value), max(0, int(base_value) - 2)],
        }
        for card_id, kind, base_value in zip(
            gallery.generic_card_ids,
            gallery.cost_kinds,
            gallery.base_values,
            strict=True,
        )
    ]
    return hashlib.sha256(canonical_json_bytes({"generic_cost_rows": rows})).hexdigest().upper()


def _expected_promotion_evidence(
    component: str,
    component_root: Path,
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    if component == "arena_cost_reference":
        return {
            "catalog_contract_sha256": _cost_catalog_contract_from_gallery(component_root),
            "catalog_file_set_sha256": str(manifest["source"]["catalog_file_set_sha256"]).upper(),
            "descriptor_fixture_set_sha256": COST_REPLAY_SET_SHA256,
            "descriptor_replay": {
                "expanded_positive_cases": 98,
                "family_count": 3,
                "frame_count": 14,
                "stable_family_card_cases": 21,
                "zero_capable_business_ids": [159, 161, 389, 393, 423, 587, 617],
            },
            "generic_card_count": validation["generic_card_count"],
            "negative_perturbation": {
                "fixed_two_rejection_cases": 77,
                "invalid_topology_rejection_cases": 21,
                "zero_fail_closed_cases": 14,
                "zero_measured_acceptance_cases": 217,
                "zero_total_cases": 231,
            },
        }
    if component == "p_item_reference":
        source = manifest["source"]
        provenance = source["extension_provenance"]
        catalog = provenance["catalog"]
        official = provenance["official_rendered_source"]
        count = validation["business_id_count"]
        return {
            "base_gallery_sha256": str(provenance["base_gallery"]["sha256"]).upper(),
            "catalog_git_blob_sha1": str(catalog["p_items_git_blob_sha1"]).upper(),
            "catalog_sha256": str(catalog["p_items_sha256"]).upper(),
            "official_reference_count": len(official["references"]),
            "production_source_evidence_sha256": (
                P_ITEM_PRODUCTION_SOURCE_EVIDENCE_SHA256
            ),
            "source_icon_count": count,
            "source_icons_sha256": str(source["source_icons_sha256"]).upper(),
            "source_pixel_match_count": count,
        }
    raise StaticReferenceHandoffError(f"unsupported component: {component}")


def validate_promoted_component(
    component: str,
    component_root: Path,
    manifest: Mapping[str, Any],
    *,
    cost_catalog_dir: Path | None = None,
    p_item_production_source_evidence_path: Path | None = None,
) -> None:
    """Recompute every packaged proof required by the release boundary."""

    root = component_root.resolve()
    contract = _contract(component)
    if component == "p_item_reference":
        if p_item_production_source_evidence_path is None:
            raise StaticReferenceHandoffError(
                "P-item release validation requires frozen production/source evidence"
            )
        production_source_evidence = _load_p_item_production_source_evidence(
            p_item_production_source_evidence_path
        )
        _validate_p_item_candidate_source_evidence(
            manifest,
            production_source_evidence,
        )
    _require_file_inventory(component, root, promoted=True)
    manifest_path = root / "manifest.json"
    if manifest_path.read_bytes() != canonical_json_bytes(dict(manifest)):
        raise StaticReferenceHandoffError(f"{component} promoted manifest is not canonical LF JSON")
    build = manifest.get("build")
    handoff = manifest.get("production_handoff")
    evaluation = manifest.get("evaluation")
    if (
        not isinstance(build, Mapping)
        or build
        != {
            "deterministic_output": "NPZ_AND_CANONICAL_LF_JSON",
            "tool_contract": contract["build_tool_contract"],
        }
        or not isinstance(handoff, Mapping)
        or handoff != {"reason": contract["promoted_reason"], "status": PROMOTED_STATUS}
        or not isinstance(evaluation, Mapping)
        or evaluation.get("status") != "PASS"
        or evaluation.get("tool_contract") != PROMOTION_TOOL_CONTRACT
        or evaluation.get("path") != REPORT_NAME
        or set(evaluation) != {"path", "sha256", "status", "tool_contract"}
        or not _is_sha256(evaluation.get("sha256"))
    ):
        raise StaticReferenceHandoffError(f"{component} was not emitted by the static-reference promotion contract")
    report_path = root / REPORT_NAME
    if not report_path.is_file() or sha256_file(report_path).casefold() != str(evaluation["sha256"]).casefold():
        raise StaticReferenceHandoffError(f"{component} evaluation report hash mismatch")
    report = _load_object(report_path, label=f"{component} evaluation report")
    if report_path.read_bytes() != canonical_json_bytes(report):
        raise StaticReferenceHandoffError(f"{component} evaluation report is not canonical LF JSON")
    candidate_manifest_sha256 = report.get("candidate_manifest_sha256")
    if not _is_sha256(candidate_manifest_sha256):
        raise StaticReferenceHandoffError(f"{component} candidate manifest SHA-256 is invalid")
    gallery_sha256, validation = evaluate_component(component, root, manifest)
    if component == "arena_cost_reference":
        if cost_catalog_dir is None:
            raise StaticReferenceHandoffError(
                "arena cost release validation requires the packaged RIS catalog"
            )
        try:
            gallery = GenericCostReferenceGallery.load(root)
        except (
            GenericCostReferenceError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise StaticReferenceHandoffError(
                "arena cost gallery cannot be loaded for catalog validation"
            ) from error
        packaged_catalog = _load_cost_catalog_contract(
            cost_catalog_dir,
            gallery,
            expected_file_set_sha256=None,
        )
        if (
            packaged_catalog["catalog_contract_sha256"]
            != _cost_catalog_contract_from_gallery(root)
            or packaged_catalog["generic_card_count"]
            != validation["generic_card_count"]
        ):
            raise StaticReferenceHandoffError(
                "arena cost packaged RIS catalog contract is incompatible"
            )
    expected_report = {
        "artifact": {
            "path": _component_asset(root, manifest, component)[0].name,
            "sha256": gallery_sha256,
        },
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "component": component,
        "privacy": "PUBLIC_SAFE_AGGREGATE_ONLY",
        "promotion_evidence": _expected_promotion_evidence(
            component,
            root,
            manifest,
            validation,
        ),
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "tool_contract": PROMOTION_TOOL_CONTRACT,
        "validation": validation,
    }
    if report != expected_report:
        raise StaticReferenceHandoffError(f"{component} evaluation report content is stale or incomplete")
    candidate = copy.deepcopy(dict(manifest))
    candidate.pop("evaluation", None)
    candidate["production_handoff"] = {
        "reason": contract["candidate_reason"],
        "status": PENDING_STATUS,
    }
    candidate = _candidate_manifest(component, candidate)
    candidate_sha256 = hashlib.sha256(canonical_json_bytes(candidate)).hexdigest().upper()
    if candidate_sha256.casefold() != str(candidate_manifest_sha256).casefold():
        raise StaticReferenceHandoffError(f"{component} evaluation report is stale for the promoted manifest")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=sorted(COMPONENT_CONTRACTS), required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-gallery",
        type=Path,
        help="Immutable parent P-item NPZ required for an extension lineage",
    )
    parser.add_argument(
        "--cost-catalog-dir",
        type=Path,
        help="Directory containing authoritative skill_cards/customizations JSON",
    )
    parser.add_argument(
        "--cost-replay",
        type=Path,
        action="append",
        default=[],
        help="Fixed public cost descriptor replay; pass all three files",
    )
    parser.add_argument(
        "--p-item-source-dir",
        type=Path,
        help="Builder-bound numeric PNG source directory for P-item promotion",
    )
    parser.add_argument(
        "--p-item-catalog",
        type=Path,
        help="Builder-bound authoritative p_items.json for P-item promotion",
    )
    parser.add_argument(
        "--p-item-production-source-evidence",
        type=Path,
        help="Frozen allowlisted production/source evidence for P-item promotion",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    promoted = promote_component(
        args.component,
        args.candidate_root,
        args.output_dir,
        base_gallery_path=args.base_gallery,
        cost_catalog_dir=args.cost_catalog_dir,
        cost_replay_paths=tuple(args.cost_replay),
        p_item_source_dir=args.p_item_source_dir,
        p_item_catalog_path=args.p_item_catalog,
        p_item_production_source_evidence_path=(
            args.p_item_production_source_evidence
        ),
    )
    print(json.dumps(promoted, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
