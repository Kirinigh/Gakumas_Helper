"""Build verified fixed or private-live arena clean-reference gallery updates."""

from __future__ import annotations

import os
import re
import sys
import json
import uuid
import zlib
import shutil
import hashlib
import argparse
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping

import cv2
import PIL
import numpy as np
from PIL import Image

TOOL_ROOT = Path(__file__).resolve().parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from build_card_embedding_gallery import (
    SOURCE_DOMAIN_OFFICIAL_RAW_ART,
    load_dataset_plan,
)


class BadgeGalleryBuildError(RuntimeError):
    """The incremental clean-reference gallery cannot be built safely."""


RELEASE_READY_HANDOFF_STATUSES = frozenset(
    {"READY", "READY_WITH_REAL_SAMPLES_PENDING"}
)


@dataclass(frozen=True)
class BadgeReference:
    class_name: str
    business_id: int
    visual_group_id: str
    reference_name: str
    top_coarse: np.ndarray
    green_mask: np.ndarray
    file_sha256: str


@dataclass(frozen=True)
class LiveBadgeReference:
    business_id: int
    visual_group_id: str
    top_coarse: np.ndarray
    green_mask: np.ndarray


@dataclass(frozen=True)
class LiveReferenceBatch:
    dataset_id: str
    dataset_revision: str
    source_kind: str
    identity_authority: str
    hard_negative_business_ids: tuple[int, ...]
    references: tuple[LiveBadgeReference, ...]


LIVE_REFERENCE_SOURCE_KIND = "AUTHORIZED_LOCAL_LIVE_CARD_CROP"
LIVE_REFERENCE_AUTHORIZATION_STATUS = "AUTHORIZED_FOR_LOCAL_DERIVATION"
LIVE_REFERENCE_RAW_PUBLICATION = "EXCLUDED"
LIVE_REFERENCE_IDENTITY_AUTHORITY = "INDEPENDENT_DETAIL_TITLE_AND_FULL_EFFECT"
LIVE_REFERENCE_SAME_CARD_BINDING = "VERIFIED"
LIVE_REFERENCE_EVALUATION_TOOL_CONTRACT = (
    "task095-card-badge-hard-negative-evaluation-v1"
)
LIVE_REFERENCE_REQUIRED_EVALUATION_GATES = frozenset(
    {
        "append_only_five_array_prefix",
        "authoritative_hard_negative_blind_test",
        "baseline_wrong_identity_reproduced",
        "candidate_live_target",
        "fixed_geometry_stress",
        "identity_sets_unchanged",
        "inherited_full_replay",
        "new_cross_group_exact_collision",
        "sibling_reference_rows",
    }
)
_PUBLIC_DATASET_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}").fullmatch
_LIVE_REFERENCE_NAME = re.compile(r"L[0-9]{4}").fullmatch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BadgeGalleryBuildError(f"{label} is unavailable or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise BadgeGalleryBuildError(f"{label} must be a JSON object")
    return payload


def _resolve_inside(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BadgeGalleryBuildError(f"{label} path is missing")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise BadgeGalleryBuildError(f"{label} path must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise BadgeGalleryBuildError(f"{label} path escapes its manifest root")
    return resolved


def _declared_sha256(expected_sha256: object, *, label: str) -> str:
    expected = str(expected_sha256).upper()
    if len(expected) != 64 or any(character not in "0123456789ABCDEF" for character in expected):
        raise BadgeGalleryBuildError(f"{label} has an invalid SHA-256 declaration")
    return expected


def _verified_file(path: Path, expected_sha256: object, *, label: str) -> str:
    expected = _declared_sha256(expected_sha256, label=label)
    if not path.is_file() or sha256_file(path) != expected:
        raise BadgeGalleryBuildError(f"{label} SHA-256 mismatch: {path}")
    return expected


def _source_provenance(
    value: object,
    *,
    label: str,
) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        raise BadgeGalleryBuildError(f"extension {label} provenance is missing")
    repository = value.get("repository")
    revision = value.get("commit")
    license_name = value.get("license")
    if (
        not isinstance(repository, str)
        or not repository.startswith("https://")
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in revision)
        or not isinstance(license_name, str)
        or not license_name
    ):
        raise BadgeGalleryBuildError(f"extension {label} provenance is invalid")
    return repository, revision.lower(), license_name


def _class_key(value: str) -> tuple[int, int, str]:
    prefix, separator, suffix = value.partition("_")
    if not prefix.isdigit() or int(prefix) < 1:
        raise BadgeGalleryBuildError(f"invalid reference class name: {value}")
    return int(prefix), int(suffix) if separator and suffix.isdigit() else -1, value


def _arena_identity_sets(
    root: Path,
) -> tuple[dict[str, Any], dict[int, str], frozenset[str], str, str, str]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_object(manifest_path, label="arena component manifest")
    if manifest.get("schema_version") != 1:
        raise BadgeGalleryBuildError("unsupported arena component manifest schema")
    production_handoff = manifest.get("production_handoff")
    if not isinstance(production_handoff, Mapping):
        raise BadgeGalleryBuildError(
            "arena component has no validated production handoff"
        )
    handoff_status = production_handoff.get("status")
    if handoff_status not in RELEASE_READY_HANDOFF_STATUSES:
        raise BadgeGalleryBuildError(
            f"arena component is not release-ready: {handoff_status}"
        )
    gallery = manifest.get("gallery")
    if not isinstance(gallery, Mapping):
        raise BadgeGalleryBuildError("arena component gallery metadata is missing")
    path = _resolve_inside(root, gallery.get("path"), label="arena component gallery")
    gallery_sha256 = _verified_file(
        path,
        gallery.get("sha256"),
        label="arena component gallery",
    )
    try:
        with np.load(path, allow_pickle=False) as arrays:
            card_ids = arrays["card_ids"]
            visual_group_ids = arrays["visual_group_ids"]
            if card_ids.ndim != 1 or visual_group_ids.ndim != 1 or card_ids.shape != visual_group_ids.shape:
                raise BadgeGalleryBuildError("arena gallery identity arrays are inconsistent")
            pairs = tuple(zip(card_ids.tolist(), visual_group_ids.tolist(), strict=True))
    except BadgeGalleryBuildError:
        raise
    except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
        raise BadgeGalleryBuildError("arena gallery is unavailable or invalid") from error

    group_by_id: dict[int, str] = {}
    groups: set[str] = set()
    for raw_id, raw_group in pairs:
        text_id = str(raw_id)
        group = str(raw_group)
        if not text_id.isdigit() or int(text_id) < 1 or not group:
            raise BadgeGalleryBuildError("arena gallery identity mapping is invalid")
        business_id = int(text_id)
        previous = group_by_id.setdefault(business_id, group)
        if previous != group:
            raise BadgeGalleryBuildError(
                f"arena gallery maps business ID {business_id} to multiple visual groups"
            )
        groups.add(group)
    if not group_by_id:
        raise BadgeGalleryBuildError("arena gallery has no business-card identities")
    return (
        manifest,
        group_by_id,
        frozenset(groups),
        sha256_file(manifest_path),
        gallery_sha256,
        str(handoff_status),
    )


def _load_base(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = root / "manifest.json"
    manifest = _load_object(manifest_path, label="base badge manifest")
    if manifest.get("schema_version") != 1:
        raise BadgeGalleryBuildError("unsupported base badge manifest schema")
    gallery = manifest.get("gallery")
    if not isinstance(gallery, Mapping):
        raise BadgeGalleryBuildError("base badge gallery metadata is missing")
    gallery_path = _resolve_inside(root, gallery.get("path"), label="base badge gallery")
    _verified_file(gallery_path, gallery.get("sha256"), label="base badge gallery")
    try:
        with np.load(gallery_path, allow_pickle=False) as arrays:
            required = (
                "business_ids",
                "visual_group_ids",
                "reference_names",
                "top_coarse",
                "green_masks",
            )
            payload = {name: np.asarray(arrays[name]) for name in required}
    except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
        raise BadgeGalleryBuildError("base badge gallery is unavailable or invalid") from error
    count = len(payload["business_ids"])
    if (
        count < 1
        or payload["business_ids"].shape != (count,)
        or payload["visual_group_ids"].shape != (count,)
        or payload["reference_names"].shape != (count,)
        or payload["top_coarse"].shape != (count, 8, 16, 3)
        or payload["green_masks"].shape != (count, 96, 96)
        or payload["business_ids"].dtype.kind not in {"i", "u"}
        or payload["visual_group_ids"].dtype.kind not in {"U", "S"}
        or payload["reference_names"].dtype.kind not in {"U", "S"}
        or payload["top_coarse"].dtype != np.uint8
        or payload["green_masks"].dtype != np.uint8
        or not set(map(int, np.unique(payload["green_masks"]))).issubset({0, 1})
        or len(set(map(int, payload["business_ids"]))) != gallery.get("business_card_id_count")
        or len(set(map(str, payload["visual_group_ids"]))) != gallery.get("visual_group_count")
        or len(set(map(str, payload["reference_names"]))) != count
    ):
        raise BadgeGalleryBuildError("base badge gallery disagrees with its manifest")
    if any(int(value) < 1 for value in payload["business_ids"]):
        raise BadgeGalleryBuildError("base badge gallery contains an invalid business ID")
    return manifest, payload


def _extension_references(
    manifest_path: Path,
    *,
    group_by_id: Mapping[int, str],
) -> tuple[dict[str, Any], tuple[BadgeReference, ...]]:
    manifest = _load_object(manifest_path, label="extension dataset manifest")
    try:
        plan = load_dataset_plan(
            manifest_path,
            source_domain=SOURCE_DOMAIN_OFFICIAL_RAW_ART,
            require_explicit_id_set=True,
        )
    except Exception as error:
        raise BadgeGalleryBuildError(
            "extension dataset identity/source contract is invalid"
        ) from error
    references: list[BadgeReference] = []
    for sample in plan.samples:
        class_name = sample.class_name
        business_id = sample.business_id
        path = sample.marker_path
        try:
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.info:
                    raise BadgeGalleryBuildError(
                        f"extension icon {class_name} is not a metadata-free PNG"
                    )
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except BadgeGalleryBuildError:
            raise
        except (OSError, ValueError) as error:
            raise BadgeGalleryBuildError(f"extension icon {class_name} is undecodable") from error
        normalized = cv2.resize(rgb[:, :, ::-1], (96, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
        blue, green, red = cv2.split(normalized.astype(np.int16))
        hsv_green = cv2.inRange(hsv, (30, 55, 45), (100, 255, 255)) > 0
        dominant = (green >= 65) & ((green - np.maximum(red, blue)) >= 14)
        group = group_by_id.get(business_id)
        if group is None:
            raise BadgeGalleryBuildError(
                f"extension business ID {business_id} is absent from the arena gallery"
            )
        references.append(
            BadgeReference(
                class_name=class_name,
                business_id=business_id,
                visual_group_id=group,
                reference_name=path.name,
                top_coarse=cv2.resize(
                    normalized[:48],
                    (16, 8),
                    interpolation=cv2.INTER_AREA,
                ),
                green_mask=(hsv_green | dominant).astype(np.uint8),
                file_sha256=sample.marker_sha256,
            )
        )
    references.sort(key=lambda item: _class_key(item.class_name))
    return manifest, tuple(references)


def _public_dataset_label(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _PUBLIC_DATASET_LABEL(value) is None:
        raise BadgeGalleryBuildError(f"live reference {label} is invalid")
    return value


def _validated_live_reference_updates(
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if "live_reference_update" in source:
        raise BadgeGalleryBuildError("base badge has an obsolete live reference lineage")
    raw_updates = source.get("live_reference_updates", [])
    if not isinstance(raw_updates, list):
        raise BadgeGalleryBuildError("base badge live reference lineage is invalid")
    required_keys = {
        "dataset_id",
        "dataset_revision",
        "source_kind",
        "identity_authority",
        "authorization_status",
        "raw_source_publication",
        "green_mask_source",
        "hard_negative_business_ids",
        "added_reference_count",
        "business_card_id_count",
        "business_card_ids",
        "visual_group_count",
        "visual_group_ids",
    }
    seen_revisions: set[tuple[str, str]] = set()
    updates: list[dict[str, Any]] = []
    for index, raw_update in enumerate(raw_updates):
        label = f"base badge live reference update {index}"
        if not isinstance(raw_update, Mapping) or set(raw_update) != required_keys:
            raise BadgeGalleryBuildError(f"{label} is invalid")
        dataset_id = _public_dataset_label(
            raw_update.get("dataset_id"),
            label=f"{label} dataset ID",
        )
        dataset_revision = _public_dataset_label(
            raw_update.get("dataset_revision"),
            label=f"{label} dataset revision",
        )
        revision_key = (dataset_id, dataset_revision)
        if revision_key in seen_revisions:
            raise BadgeGalleryBuildError("base badge has duplicate live reference revisions")
        seen_revisions.add(revision_key)
        business_ids = raw_update.get("business_card_ids")
        hard_negative_business_ids = raw_update.get("hard_negative_business_ids")
        added_count = raw_update.get("added_reference_count")
        business_id_count = raw_update.get("business_card_id_count")
        visual_group_count = raw_update.get("visual_group_count")
        visual_group_ids = raw_update.get("visual_group_ids")
        if (
            raw_update.get("source_kind") != LIVE_REFERENCE_SOURCE_KIND
            or raw_update.get("identity_authority")
            != LIVE_REFERENCE_IDENTITY_AUTHORITY
            or raw_update.get("authorization_status")
            != LIVE_REFERENCE_AUTHORIZATION_STATUS
            or raw_update.get("raw_source_publication")
            != LIVE_REFERENCE_RAW_PUBLICATION
            or raw_update.get("green_mask_source")
            != "SAME_VISUAL_GROUP_HISTORICAL_REFERENCE"
            or not isinstance(hard_negative_business_ids, list)
            or not hard_negative_business_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in hard_negative_business_ids
            )
            or hard_negative_business_ids
            != sorted(set(hard_negative_business_ids))
            or isinstance(added_count, bool)
            or not isinstance(added_count, int)
            or added_count < 1
            or isinstance(business_id_count, bool)
            or not isinstance(business_id_count, int)
            or business_id_count < 1
            or isinstance(visual_group_count, bool)
            or not isinstance(visual_group_count, int)
            or visual_group_count < 1
            or not isinstance(visual_group_ids, list)
            or any(not isinstance(value, str) or not value for value in visual_group_ids)
            or visual_group_ids != sorted(set(visual_group_ids))
            or len(visual_group_ids) != visual_group_count
            or not isinstance(business_ids, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in business_ids
            )
            or business_ids != sorted(set(business_ids))
            or len(business_ids) != business_id_count
            or set(business_ids).intersection(hard_negative_business_ids)
            or added_count < business_id_count
            or visual_group_count > added_count
        ):
            raise BadgeGalleryBuildError(f"{label} is invalid")
        updates.append(dict(raw_update))
    return tuple(updates)


def _require_evaluated_live_base(
    root: Path,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], ...]:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise BadgeGalleryBuildError("base badge source metadata is missing")
    updates = _validated_live_reference_updates(source)

    actual_group_by_id: dict[int, str] = {}
    actual_row_counts: dict[int, int] = {}
    live_rows: list[tuple[int, int, str]] = []
    for raw_business_id, raw_group, raw_name in zip(
        arrays["business_ids"],
        arrays["visual_group_ids"],
        arrays["reference_names"],
        strict=True,
    ):
        business_id = int(raw_business_id)
        group = str(raw_group)
        previous = actual_group_by_id.setdefault(business_id, group)
        if previous != group:
            raise BadgeGalleryBuildError(
                f"base badge maps business ID {business_id} to multiple visual groups"
            )
        actual_row_counts[business_id] = actual_row_counts.get(business_id, 0) + 1
        name = str(raw_name)
        if _LIVE_REFERENCE_NAME(name) is not None:
            live_rows.append((int(name[1:]), business_id, group))
    live_rows.sort()
    if [index for index, _, _ in live_rows] != list(range(1, len(live_rows) + 1)):
        raise BadgeGalleryBuildError("base badge live-reference row names are not contiguous")
    if sum(update["added_reference_count"] for update in updates) != len(live_rows):
        raise BadgeGalleryBuildError(
            "base badge live-reference rows and lineage disagree"
        )
    live_offset = 0
    for update in updates:
        next_offset = live_offset + update["added_reference_count"]
        batch_rows = live_rows[live_offset:next_offset]
        batch_business_ids = sorted({business_id for _, business_id, _ in batch_rows})
        batch_groups = sorted({group for _, _, group in batch_rows})
        if (
            update["business_card_id_count"] != 1
            or update["visual_group_count"] != 1
            or batch_business_ids != update["business_card_ids"]
            or len(batch_business_ids) != update["business_card_id_count"]
            or batch_groups != update["visual_group_ids"]
            or len(batch_groups) != update["visual_group_count"]
            or any(
                business_id not in actual_group_by_id
                for business_id in update["hard_negative_business_ids"]
            )
        ):
            raise BadgeGalleryBuildError(
                "base badge live-reference rows disagree with their lineage identities"
            )
        live_offset = next_offset
    handoff = manifest.get("production_handoff")
    if not updates:
        if (
            not isinstance(handoff, Mapping)
            or handoff.get("status") not in RELEASE_READY_HANDOFF_STATUSES
        ):
            raise BadgeGalleryBuildError("base badge is not release-ready")
        return updates

    if (
        not isinstance(handoff, Mapping)
        or handoff.get("status") != "READY_WITH_REAL_SAMPLES_PENDING"
    ):
        raise BadgeGalleryBuildError(
            "base badge live-reference lineage is not release-ready"
        )
    evaluation = manifest.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("status") != "PASS"
        or evaluation.get("tool_contract")
        != LIVE_REFERENCE_EVALUATION_TOOL_CONTRACT
    ):
        raise BadgeGalleryBuildError(
            "base badge live-reference evaluation binding is missing or invalid"
        )
    report_path = _resolve_inside(
        root,
        evaluation.get("path"),
        label="base badge live-reference evaluation",
    )
    _verified_file(
        report_path,
        evaluation.get("sha256"),
        label="base badge live-reference evaluation",
    )
    report = _load_object(
        report_path,
        label="base badge live-reference evaluation report",
    )
    components = report.get("components")
    candidate = (
        components.get("candidate") if isinstance(components, Mapping) else None
    )
    live_batch = report.get("live_reference_batch")
    gates = report.get("gates")
    gallery = manifest.get("gallery")
    last_update = updates[-1]
    if (
        report.get("schema_version") != 1
        or report.get("status") != "PASS"
        or report.get("tool_contract")
        != LIVE_REFERENCE_EVALUATION_TOOL_CONTRACT
        or not isinstance(candidate, Mapping)
        or not isinstance(gallery, Mapping)
        or candidate.get("gallery_sha256") != gallery.get("sha256")
        or candidate.get("reference_row_count")
        != source.get("reference_file_count")
        or not isinstance(live_batch, Mapping)
        or live_batch.get("dataset_id") != last_update["dataset_id"]
        or live_batch.get("dataset_revision")
        != last_update["dataset_revision"]
        or not isinstance(gates, Mapping)
        or set(gates) != LIVE_REFERENCE_REQUIRED_EVALUATION_GATES
        or any(
            not isinstance(gate, Mapping) or gate.get("status") != "PASS"
            for gate in gates.values()
        )
        or report.get("independent_live_holdout") != "PENDING"
    ):
        raise BadgeGalleryBuildError(
            "base badge live-reference evaluation report is incompatible"
        )
    base_component = (
        components.get("base") if isinstance(components, Mapping) else None
    )
    base_reference_count = (
        base_component.get("reference_row_count")
        if isinstance(base_component, Mapping)
        else None
    )
    declared_hashes = (
        candidate.get("manifest_sha256"),
        base_component.get("manifest_sha256")
        if isinstance(base_component, Mapping)
        else None,
        base_component.get("gallery_sha256")
        if isinstance(base_component, Mapping)
        else None,
    )
    authoritative = report.get("authoritative_dataset")
    if (
        isinstance(base_reference_count, bool)
        or not isinstance(base_reference_count, int)
        or base_reference_count < 1
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9A-Fa-f]{64}", value) is None
            for value in declared_hashes
        )
        or not isinstance(source.get("base_manifest_sha256"), str)
        or base_component["manifest_sha256"].casefold()
        != source["base_manifest_sha256"].casefold()
        or not isinstance(authoritative, Mapping)
        or not isinstance(authoritative.get("dataset_id"), str)
        or not authoritative["dataset_id"]
        or not isinstance(authoritative.get("dataset_revision"), str)
        or not authoritative["dataset_revision"]
        or not isinstance(authoritative.get("manifest_sha256"), str)
        or re.fullmatch(
            r"[0-9A-Fa-f]{64}",
            authoritative["manifest_sha256"],
        )
        is None
        or authoritative.get("source_domain")
        not in {"OFFICIAL_RAW_CARD_ART", "FIXED_RENDERED_PNG"}
        or authoritative.get("hard_negative_input")
        != "AUTHORITATIVE_FIXED_RENDERED_ICON"
    ):
        raise BadgeGalleryBuildError(
            "base badge live-reference evaluation provenance is invalid"
        )

    def positive_count(gate_name: str, field: str = "sample_count") -> int | None:
        value = gates[gate_name].get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return None
        return value

    def has_zero_counts(gate_name: str, fields: tuple[str, ...]) -> bool:
        gate = gates[gate_name]
        return all(
            not isinstance(gate.get(field), bool)
            and isinstance(gate.get(field), int)
            and gate.get(field) == 0
            for field in fields
        )

    prototype_count = live_batch.get("prototype_count")
    added_reference_count = last_update["added_reference_count"]
    reference_file_count = source.get("reference_file_count")
    target_count = positive_count("candidate_live_target")
    baseline_count = positive_count("baseline_wrong_identity_reproduced")
    geometry_count = positive_count("fixed_geometry_stress")
    hard_negative_count = positive_count(
        "authoritative_hard_negative_blind_test"
    )
    inherited_count = positive_count("inherited_full_replay")
    append_inherited_count = positive_count(
        "append_only_five_array_prefix",
        "inherited_row_count",
    )
    append_prototype_count = positive_count(
        "append_only_five_array_prefix",
        "appended_prototype_count",
    )
    identity_gate = gates["identity_sets_unchanged"]
    if (
        isinstance(prototype_count, bool)
        or not isinstance(prototype_count, int)
        or prototype_count != added_reference_count
        or live_batch.get("raw_source_publication") != LIVE_REFERENCE_RAW_PUBLICATION
        or isinstance(reference_file_count, bool)
        or not isinstance(reference_file_count, int)
        or reference_file_count < 1
        or target_count != prototype_count
        or baseline_count != prototype_count
        or geometry_count != 41 * prototype_count
        or gates["fixed_geometry_stress"].get("contract")
        != "dx=-3..3;dy=-1..1;size_delta=0,2;exact_excluded"
        or hard_negative_count is None
        or inherited_count is None
        or append_inherited_count is None
        or append_prototype_count != prototype_count
        or append_inherited_count + append_prototype_count
        != reference_file_count
        or inherited_count != append_inherited_count
        or base_reference_count != append_inherited_count
        or identity_gate.get("business_id_count")
        != gallery.get("business_card_id_count")
        or identity_gate.get("visual_group_count")
        != gallery.get("visual_group_count")
        or not has_zero_counts(
            "candidate_live_target",
            (
                "wrong_business_id_count",
                "wrong_visual_group_count",
                "low_confidence_count",
            ),
        )
        or not has_zero_counts(
            "fixed_geometry_stress",
            (
                "wrong_business_id_count",
                "wrong_visual_group_count",
                "low_confidence_count",
            ),
        )
        or not has_zero_counts(
            "authoritative_hard_negative_blind_test",
            (
                "wrong_business_id_count",
                "wrong_visual_group_count",
                "target_flip_count",
                "low_confidence_count",
            ),
        )
        or not has_zero_counts(
            "inherited_full_replay",
            (
                "candidate_changed_baseline_business_id_count",
                "wrong_visual_group_count",
            ),
        )
        or not has_zero_counts(
            "new_cross_group_exact_collision",
            ("collision_count",),
        )
    ):
        raise BadgeGalleryBuildError(
            "base badge live-reference evaluation gate evidence is invalid"
        )

    scope = report.get("scope")
    target_business_id = (
        scope.get("target_business_id") if isinstance(scope, Mapping) else None
    )
    target_visual_group_id = (
        scope.get("target_visual_group_id")
        if isinstance(scope, Mapping)
        else None
    )
    hard_negative_ids = (
        scope.get("hard_negative_business_ids")
        if isinstance(scope, Mapping)
        else None
    )
    sibling_ids = (
        scope.get("sibling_business_ids") if isinstance(scope, Mapping) else None
    )
    baseline_wrong_id = (
        scope.get("baseline_wrong_business_id")
        if isinstance(scope, Mapping)
        else None
    )
    baseline_wrong_group = (
        scope.get("baseline_wrong_visual_group_id")
        if isinstance(scope, Mapping)
        else None
    )
    update_business_ids = last_update["business_card_ids"]
    update_visual_groups = last_update["visual_group_ids"]
    cumulative_hard_negative_ids = {
        business_id
        for update in updates
        for business_id in update["hard_negative_business_ids"]
    }
    declared_scope_ids = [
        target_business_id,
        baseline_wrong_id,
        *(sibling_ids if isinstance(sibling_ids, list) else []),
        *(hard_negative_ids if isinstance(hard_negative_ids, list) else []),
    ]
    affected_visual_groups = (
        scope.get("affected_visual_group_ids")
        if isinstance(scope, Mapping)
        else None
    )
    if (
        isinstance(target_business_id, bool)
        or not isinstance(target_business_id, int)
        or update_business_ids != [target_business_id]
        or last_update["business_card_id_count"] != 1
        or not isinstance(target_visual_group_id, str)
        or update_visual_groups != [target_visual_group_id]
        or last_update["visual_group_count"] != 1
        or not isinstance(hard_negative_ids, list)
        or not hard_negative_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in hard_negative_ids
        )
        or hard_negative_ids != sorted(cumulative_hard_negative_ids)
        or target_business_id in hard_negative_ids
        or not isinstance(sibling_ids, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in sibling_ids
        )
        or sibling_ids != sorted(set(sibling_ids))
        or target_business_id in sibling_ids
        or set(sibling_ids).intersection(hard_negative_ids)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in declared_scope_ids
        )
        or not set(declared_scope_ids).issubset(actual_group_by_id)
        or actual_group_by_id.get(target_business_id) != target_visual_group_id
        or baseline_wrong_id not in hard_negative_ids
        or not isinstance(baseline_wrong_group, str)
        or not baseline_wrong_group
        or actual_group_by_id.get(baseline_wrong_id) != baseline_wrong_group
        or baseline_wrong_group == target_visual_group_id
        or not isinstance(affected_visual_groups, list)
        or affected_visual_groups
        != sorted({actual_group_by_id[value] for value in declared_scope_ids})
    ):
        raise BadgeGalleryBuildError(
            "base badge live-reference evaluation scope is invalid"
        )

    hard_negative_gate = gates["authoritative_hard_negative_blind_test"]
    per_business_count = hard_negative_gate.get("per_business_id_sample_count")
    expected_hard_negative_keys = {str(value) for value in hard_negative_ids}
    sibling_row_counts = gates["sibling_reference_rows"].get("row_counts")
    expected_sibling_keys = {str(value) for value in sibling_ids}
    expected_sibling_row_counts = {
        str(value): actual_row_counts[value] for value in sibling_ids
    }
    if (
        not isinstance(per_business_count, Mapping)
        or set(per_business_count) != expected_hard_negative_keys
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in per_business_count.values()
        )
        or sum(per_business_count.values()) != hard_negative_count
        or not isinstance(sibling_row_counts, Mapping)
        or set(sibling_row_counts) != expected_sibling_keys
        or sibling_row_counts != expected_sibling_row_counts
    ):
        raise BadgeGalleryBuildError(
            "base badge live-reference evaluation identity evidence is invalid"
        )

    confusion = report.get("confusion")
    expected_business_confusion = [
        {
            "expected_business_id": business_id,
            "predicted_business_id": business_id,
            "count": actual_row_counts[business_id]
            + (
                per_business_count[str(business_id)]
                if business_id in cumulative_hard_negative_ids
                else 0
            ),
        }
        for business_id in sorted(set(declared_scope_ids))
    ]
    expected_group_counts: dict[str, int] = {}
    for row in expected_business_confusion:
        group = actual_group_by_id[row["expected_business_id"]]
        expected_group_counts[group] = expected_group_counts.get(group, 0) + row["count"]
    expected_group_confusion = [
        {
            "expected_visual_group_id": group,
            "predicted_visual_group_id": group,
            "count": count,
        }
        for group, count in sorted(expected_group_counts.items())
    ]
    if (
        not isinstance(confusion, Mapping)
        or set(confusion) != {"business_id_sparse", "visual_group_sparse"}
        or confusion.get("business_id_sparse") != expected_business_confusion
        or confusion.get("visual_group_sparse") != expected_group_confusion
    ):
        raise BadgeGalleryBuildError(
            "base badge live-reference evaluation confusion evidence is invalid"
        )
    return updates


def _live_reference_names(
    existing_names: np.ndarray,
    *,
    count: int,
) -> tuple[str, ...]:
    taken = set(map(str, existing_names))
    names: list[str] = []
    for index in range(1, 10_000):
        candidate = f"L{index:04d}"
        if candidate in taken:
            continue
        encoded = np.asarray([candidate], dtype=existing_names.dtype)
        if str(encoded[0]) != candidate:
            raise BadgeGalleryBuildError(
                "live reference names exceed the base gallery dtype"
            )
        names.append(candidate)
        taken.add(candidate)
        if len(names) == count:
            return tuple(names)
    raise BadgeGalleryBuildError("live reference name space is exhausted")


def _live_references(
    manifest_path: Path,
    *,
    base: Mapping[str, np.ndarray],
    group_by_id: Mapping[int, str],
) -> LiveReferenceBatch:
    manifest = _load_object(manifest_path, label="private live reference manifest")
    if manifest.get("schema_version") != 1:
        raise BadgeGalleryBuildError("unsupported private live reference schema")
    dataset_id = _public_dataset_label(manifest.get("dataset_id"), label="dataset ID")
    dataset_revision = _public_dataset_label(
        manifest.get("dataset_revision"),
        label="dataset revision",
    )
    source_kind = manifest.get("source_kind")
    if source_kind != LIVE_REFERENCE_SOURCE_KIND:
        raise BadgeGalleryBuildError("live reference source kind is invalid")
    authorization = manifest.get("authorization")
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("status") != LIVE_REFERENCE_AUTHORIZATION_STATUS
        or authorization.get("raw_source_publication")
        != LIVE_REFERENCE_RAW_PUBLICATION
    ):
        raise BadgeGalleryBuildError("live reference authorization is invalid")
    validation = manifest.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("status") != "CLEAR"
        or validation.get("identity_authority")
        != LIVE_REFERENCE_IDENTITY_AUTHORITY
        or validation.get("same_card_binding")
        != LIVE_REFERENCE_SAME_CARD_BINDING
    ):
        raise BadgeGalleryBuildError("live reference identity validation is invalid")
    evaluation = manifest.get("evaluation")
    hard_negative_business_ids = (
        evaluation.get("hard_negative_business_ids")
        if isinstance(evaluation, Mapping)
        else None
    )
    if (
        not isinstance(hard_negative_business_ids, list)
        or not hard_negative_business_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in hard_negative_business_ids
        )
        or hard_negative_business_ids != sorted(set(hard_negative_business_ids))
    ):
        raise BadgeGalleryBuildError(
            "live reference hard-negative declaration is invalid"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise BadgeGalleryBuildError("live reference samples are missing")

    root = manifest_path.parent
    existing_coarse: dict[bytes, set[str]] = {}
    for group, coarse in zip(
        base["visual_group_ids"],
        base["top_coarse"],
        strict=True,
    ):
        existing_coarse.setdefault(np.asarray(coarse).tobytes(), set()).add(str(group))
    seen_samples: set[tuple[str, tuple[int, int, int, int], int, str]] = set()
    references: list[LiveBadgeReference] = []
    for index, raw_sample in enumerate(samples):
        label = f"live reference sample {index}"
        if not isinstance(raw_sample, Mapping):
            raise BadgeGalleryBuildError(f"{label} must be an object")
        image_relative = raw_sample.get("image_path")
        image_path = _resolve_inside(root, image_relative, label=f"{label} image")
        _verified_file(
            image_path,
            raw_sample.get("image_sha256"),
            label=f"{label} image",
        )
        raw_roi = raw_sample.get("roi")
        if (
            not isinstance(raw_roi, list)
            or len(raw_roi) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_roi)
        ):
            raise BadgeGalleryBuildError(f"{label} ROI is invalid")
        x, y, width, height = raw_roi
        if x < 0 or y < 0 or width < 8 or height < 8:
            raise BadgeGalleryBuildError(f"{label} ROI is invalid")
        business_id = raw_sample.get("expected_business_id")
        if isinstance(business_id, bool) or not isinstance(business_id, int) or business_id < 1:
            raise BadgeGalleryBuildError(f"{label} expected business ID is invalid")
        visual_group_id = raw_sample.get("expected_visual_group_id")
        if not isinstance(visual_group_id, str) or not visual_group_id:
            raise BadgeGalleryBuildError(f"{label} expected visual group is invalid")
        mapped_group = group_by_id.get(business_id)
        if mapped_group is None:
            raise BadgeGalleryBuildError(
                f"{label} business ID {business_id} is absent from the arena gallery"
            )
        if mapped_group != visual_group_id:
            raise BadgeGalleryBuildError(
                f"{label} expected visual group does not match the arena mapping"
            )
        sample_key = (
            str(image_relative),
            (x, y, width, height),
            business_id,
            visual_group_id,
        )
        if sample_key in seen_samples:
            raise BadgeGalleryBuildError("private live reference manifest has duplicate samples")
        seen_samples.add(sample_key)

        try:
            encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except OSError as error:
            raise BadgeGalleryBuildError(f"{label} image is unavailable") from error
        if image is None:
            raise BadgeGalleryBuildError(f"{label} image is undecodable")
        crop = image[y : y + height, x : x + width, :3]
        if crop.shape != (height, width, 3):
            raise BadgeGalleryBuildError(f"{label} ROI exceeds the image bounds")
        normalized = cv2.resize(
            np.ascontiguousarray(crop),
            (96, 96),
            interpolation=cv2.INTER_AREA,
        )
        top_coarse = cv2.resize(
            normalized[:48],
            (16, 8),
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint8)
        coarse_key = top_coarse.tobytes()
        if coarse_key in existing_coarse:
            groups = sorted(existing_coarse[coarse_key])
            raise BadgeGalleryBuildError(
                f"{label} duplicates an existing coarse reference in groups {groups}"
            )

        matching_business_rows = np.flatnonzero(
            (base["business_ids"] == business_id)
            & (base["visual_group_ids"].astype(str) == visual_group_id)
        )
        matching_group_rows = np.flatnonzero(
            base["visual_group_ids"].astype(str) == visual_group_id
        )
        mask_rows = matching_business_rows if len(matching_business_rows) else matching_group_rows
        if not len(mask_rows):
            raise BadgeGalleryBuildError(
                f"{label} has no clean historical row in its visual group"
            )
        references.append(
            LiveBadgeReference(
                business_id=business_id,
                visual_group_id=visual_group_id,
                top_coarse=top_coarse,
                green_mask=np.asarray(base["green_masks"][int(mask_rows[0])]).copy(),
            )
        )
        existing_coarse[coarse_key] = {visual_group_id}

    if {reference.business_id for reference in references}.intersection(
        hard_negative_business_ids
    ):
        raise BadgeGalleryBuildError(
            "live reference target and hard-negative business IDs overlap"
        )
    return LiveReferenceBatch(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        source_kind=source_kind,
        identity_authority=LIVE_REFERENCE_IDENTITY_AUTHORITY,
        hard_negative_business_ids=tuple(hard_negative_business_ids),
        references=tuple(references),
    )


def _build_live_badge_gallery(
    *,
    base_component_root: Path,
    arena_component_root: Path,
    live_reference_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base_manifest, base = _load_base(base_component_root)
    inherited_live_updates = _require_evaluated_live_base(
        base_component_root,
        base_manifest,
        base,
    )
    (
        arena_manifest,
        group_by_id,
        arena_groups,
        arena_manifest_sha256,
        arena_gallery_sha256,
        arena_handoff_status,
    ) = _arena_identity_sets(arena_component_root)
    base_ids = frozenset(map(int, base["business_ids"]))
    if base_ids != frozenset(group_by_id):
        missing = sorted(frozenset(group_by_id) - base_ids)
        surplus = sorted(base_ids - frozenset(group_by_id))
        raise BadgeGalleryBuildError(
            f"badge and arena gallery business-ID sets disagree: missing={missing}, surplus={surplus}"
        )
    for raw_id, raw_group in zip(
        base["business_ids"],
        base["visual_group_ids"],
        strict=True,
    ):
        business_id = int(raw_id)
        if group_by_id.get(business_id) != str(raw_group):
            raise BadgeGalleryBuildError(
                f"base badge and arena gallery disagree for business ID {business_id}"
            )
    if frozenset(map(str, base["visual_group_ids"])) != arena_groups:
        raise BadgeGalleryBuildError("badge and arena gallery visual-group sets disagree")

    batch = _live_references(
        live_reference_manifest,
        base=base,
        group_by_id=group_by_id,
    )
    references = batch.references
    names = tuple(map(str, base["reference_names"])) + _live_reference_names(
        base["reference_names"],
        count=len(references),
    )
    new_business_values = [item.business_id for item in references]
    new_business_ids = np.asarray(new_business_values, dtype=base["business_ids"].dtype)
    if list(map(int, new_business_ids)) != new_business_values:
        raise BadgeGalleryBuildError("live reference business IDs exceed the base gallery dtype")
    business_ids = np.concatenate((base["business_ids"], new_business_ids))
    new_group_values = [item.visual_group_id for item in references]
    new_visual_groups = np.asarray(
        new_group_values,
        dtype=base["visual_group_ids"].dtype,
    )
    if list(map(str, new_visual_groups)) != new_group_values:
        raise BadgeGalleryBuildError(
            "live reference visual-group IDs exceed the base gallery dtype"
        )
    visual_groups = np.concatenate((base["visual_group_ids"], new_visual_groups))
    reference_names = np.asarray(names, dtype=base["reference_names"].dtype)
    if list(map(str, reference_names)) != list(names):
        raise BadgeGalleryBuildError("live reference names exceed the base gallery dtype")
    top_coarse = np.concatenate(
        (
            base["top_coarse"],
            np.stack([item.top_coarse for item in references]).astype(np.uint8),
        )
    )
    green_masks = np.concatenate(
        (
            base["green_masks"],
            np.stack([item.green_mask for item in references]).astype(np.uint8),
        )
    )
    inherited_count = len(base["business_ids"])
    for name, values in (
        ("business_ids", business_ids),
        ("visual_group_ids", visual_groups),
        ("reference_names", reference_names),
        ("top_coarse", top_coarse),
        ("green_masks", green_masks),
    ):
        inherited = base[name]
        prefix = values[:inherited_count]
        if (
            prefix.dtype != inherited.dtype
            or prefix.shape != inherited.shape
            or prefix.tobytes() != inherited.tobytes()
        ):
            raise BadgeGalleryBuildError(
                f"base badge rows changed while appending live references to {name}"
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        gallery_path = temporary / Path(str(base_manifest["gallery"]["path"])).name
        np.savez_compressed(
            gallery_path,
            business_ids=business_ids,
            visual_group_ids=visual_groups,
            reference_names=reference_names,
            top_coarse=top_coarse,
            green_masks=green_masks,
        )
        base_source = base_manifest.get("source")
        if not isinstance(base_source, Mapping):
            raise BadgeGalleryBuildError("base badge source metadata is missing")
        current_revision = (batch.dataset_id, batch.dataset_revision)
        if current_revision in {
            (str(update["dataset_id"]), str(update["dataset_revision"]))
            for update in inherited_live_updates
        }:
            raise BadgeGalleryBuildError(
                "live reference dataset revision is already present in the base badge"
            )
        base_reference_sha256 = _declared_sha256(
            base_source.get("reference_set_sha256"),
            label="base badge reference set",
        )
        digest = hashlib.sha256()
        digest.update(bytes.fromhex(base_reference_sha256))
        digest.update(batch.dataset_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(batch.dataset_revision.encode("ascii"))
        digest.update(b"\0")
        live_names = names[inherited_count:]
        for reference_name, reference in zip(live_names, references, strict=True):
            digest.update(reference_name.encode("ascii"))
            digest.update(b"\0")
            digest.update(str(reference.business_id).encode("ascii"))
            digest.update(b"\0")
            digest.update(reference.visual_group_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(reference.top_coarse.tobytes())
            digest.update(reference.green_mask.tobytes())
        runtime = base_manifest.get("runtime")
        if not isinstance(runtime, Mapping):
            raise BadgeGalleryBuildError("base badge runtime metadata is missing")
        augmented_business_ids = sorted(set(new_business_values))
        augmented_visual_groups = sorted(set(new_group_values))
        current_live_update = {
            "dataset_id": batch.dataset_id,
            "dataset_revision": batch.dataset_revision,
            "source_kind": batch.source_kind,
            "identity_authority": batch.identity_authority,
            "authorization_status": LIVE_REFERENCE_AUTHORIZATION_STATUS,
            "raw_source_publication": LIVE_REFERENCE_RAW_PUBLICATION,
            "green_mask_source": "SAME_VISUAL_GROUP_HISTORICAL_REFERENCE",
            "hard_negative_business_ids": list(batch.hard_negative_business_ids),
            "added_reference_count": len(references),
            "business_card_id_count": len(augmented_business_ids),
            "business_card_ids": augmented_business_ids,
            "visual_group_count": len(augmented_visual_groups),
            "visual_group_ids": augmented_visual_groups,
        }
        source = dict(base_source)
        source.update(
            {
                "base_manifest_sha256": sha256_file(
                    base_component_root / "manifest.json"
                ),
                "arena_component": arena_manifest.get("component"),
                "arena_component_manifest_sha256": arena_manifest_sha256,
                "arena_gallery_sha256": arena_gallery_sha256,
                "base_reference_file_count": len(base["business_ids"]),
                "retained_reference_file_count": len(base["business_ids"]),
                "replaced_reference_file_count": 0,
                "added_reference_file_count": len(references),
                "reference_file_count": len(business_ids),
                "reference_set_sha256": digest.hexdigest().upper(),
                "reference_set_digest_contract": (
                    "sha256(base_reference_set_sha256 || public_dataset_id || NUL || "
                    "public_dataset_revision || NUL || ordered(public_reference_name || NUL || "
                    "business_id || NUL || visual_group_id || NUL || top_coarse_bytes || "
                    "reused_green_mask_bytes))"
                ),
                "live_reference_updates": [
                    *inherited_live_updates,
                    current_live_update,
                ],
            }
        )
        manifest = {
            "schema_version": 1,
            "component": base_manifest.get("component", "arena_card_reference"),
            "build": {
                "mode": "GALLERY_ONLY",
                "tool_contract": "task095-card-badge-live-reference-append-v1",
                "preprocess_contract": (
                    "bgr-image-roi-to-bgr-96-area-top48-to-16x8-area-"
                    "reuse-same-group-green-mask-v1"
                ),
                "unaffected_base_rows": "BITWISE_COPIED",
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
                "pillow_version": PIL.__version__,
                "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            },
            "source": source,
            "gallery": {
                "path": gallery_path.name,
                "sha256": sha256_file(gallery_path),
                "business_card_id_count": len(set(map(int, business_ids))),
                "visual_group_count": len(set(map(str, visual_groups))),
                "normalization_size": [96, 96],
            },
            "runtime": dict(runtime),
            "production_handoff": {
                "status": "PENDING_VALIDATION",
                "reason": (
                    "private-live candidate must pass append-only replay, target, geometry, "
                    "and authoritative hard-negative evaluation before promotion"
                ),
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def build_badge_gallery(
    *,
    base_component_root: Path,
    arena_component_root: Path,
    output_dir: Path,
    extension_manifest: Path | None = None,
    live_reference_manifest: Path | None = None,
) -> dict[str, Any]:
    """Build one atomic badge-gallery update while preserving unaffected rows."""

    base_component_root = base_component_root.resolve()
    arena_component_root = arena_component_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise BadgeGalleryBuildError(f"output directory already exists: {output_dir}")
    if (extension_manifest is None) == (live_reference_manifest is None):
        raise BadgeGalleryBuildError(
            "exactly one extension or private live reference manifest is required"
        )
    if live_reference_manifest is not None:
        return _build_live_badge_gallery(
            base_component_root=base_component_root,
            arena_component_root=arena_component_root,
            live_reference_manifest=live_reference_manifest.resolve(),
            output_dir=output_dir,
        )
    assert extension_manifest is not None
    extension_manifest = extension_manifest.resolve()
    base_manifest, base = _load_base(base_component_root)
    inherited_live_updates = _require_evaluated_live_base(
        base_component_root,
        base_manifest,
        base,
    )
    (
        arena_manifest,
        group_by_id,
        arena_groups,
        arena_manifest_sha256,
        arena_gallery_sha256,
        arena_handoff_status,
    ) = _arena_identity_sets(arena_component_root)
    extension, references = _extension_references(
        extension_manifest,
        group_by_id=group_by_id,
    )
    base_ids = frozenset(map(int, base["business_ids"]))
    extension_ids = frozenset(reference.business_id for reference in references)
    live_business_ids = frozenset(
        business_id
        for update in inherited_live_updates
        for business_id in update["business_card_ids"]
    )
    replaced_live_ids = sorted(extension_ids & live_business_ids)
    if replaced_live_ids:
        raise BadgeGalleryBuildError(
            "official extension would replace live-reference lineage without an "
            f"explicit retirement contract: {replaced_live_ids}"
        )
    retained_mask = np.asarray(
        [int(value) not in extension_ids for value in base["business_ids"]],
        dtype=bool,
    )
    retained = {name: values[retained_mask] for name, values in base.items()}
    retained_ids = base_ids - extension_ids
    if retained_ids.union(extension_ids) != frozenset(group_by_id):
        missing = sorted(frozenset(group_by_id) - retained_ids - extension_ids)
        surplus = sorted((retained_ids | extension_ids) - frozenset(group_by_id))
        raise BadgeGalleryBuildError(
            f"badge and arena gallery business-ID sets disagree: missing={missing}, surplus={surplus}"
        )
    for raw_id, raw_group in zip(
        retained["business_ids"],
        retained["visual_group_ids"],
        strict=True,
    ):
        business_id = int(raw_id)
        if group_by_id.get(business_id) != str(raw_group):
            raise BadgeGalleryBuildError(
                f"base badge and arena gallery disagree for business ID {business_id}"
            )
    names = tuple(map(str, retained["reference_names"])) + tuple(
        reference.reference_name for reference in references
    )
    if len(names) != len(set(names)):
        raise BadgeGalleryBuildError("base and extension reference names overlap")

    new_business_values = [item.business_id for item in references]
    new_business_ids = np.asarray(
        new_business_values,
        dtype=base["business_ids"].dtype,
    )
    if list(map(int, new_business_ids)) != new_business_values:
        raise BadgeGalleryBuildError("extension business IDs exceed the base gallery dtype")
    business_ids = np.concatenate((retained["business_ids"], new_business_ids))
    new_group_values = [item.visual_group_id for item in references]
    new_visual_groups = np.asarray(
        new_group_values,
        dtype=base["visual_group_ids"].dtype,
    )
    if list(map(str, new_visual_groups)) != new_group_values:
        raise BadgeGalleryBuildError(
            "extension visual-group IDs exceed the base gallery dtype"
        )
    visual_groups = np.concatenate(
        (retained["visual_group_ids"], new_visual_groups)
    )
    reference_names = np.asarray(names, dtype=base["reference_names"].dtype)
    if list(map(str, reference_names)) != list(names):
        raise BadgeGalleryBuildError(
            "extension reference names exceed the base gallery dtype"
        )
    top_coarse = np.concatenate(
        (
            retained["top_coarse"],
            np.stack([item.top_coarse for item in references]).astype(np.uint8),
        )
    )
    green_masks = np.concatenate(
        (
            retained["green_masks"],
            np.stack([item.green_mask for item in references]).astype(np.uint8),
        )
    )
    if frozenset(map(str, visual_groups)) != arena_groups:
        raise BadgeGalleryBuildError("badge and arena gallery visual-group sets disagree")
    retained_count = len(retained["business_ids"])
    for name, values in (
        ("business_ids", business_ids),
        ("visual_group_ids", visual_groups),
        ("reference_names", reference_names),
        ("top_coarse", top_coarse),
        ("green_masks", green_masks),
    ):
        inherited = retained[name]
        prefix = values[:retained_count]
        if (
            prefix.dtype != inherited.dtype
            or prefix.shape != inherited.shape
            or prefix.tobytes() != inherited.tobytes()
        ):
            raise BadgeGalleryBuildError(
                f"retained base badge rows changed while building {name}"
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        gallery_path = temporary / Path(str(base_manifest["gallery"]["path"])).name
        np.savez_compressed(
            gallery_path,
            business_ids=business_ids,
            visual_group_ids=visual_groups,
            reference_names=reference_names,
            top_coarse=top_coarse,
            green_masks=green_masks,
        )
        base_source = base_manifest.get("source")
        if not isinstance(base_source, Mapping):
            raise BadgeGalleryBuildError("base badge source metadata is missing")
        base_reference_sha256 = _declared_sha256(
            base_source.get("reference_set_sha256"),
            label="base badge reference set",
        )
        digest = hashlib.sha256()
        digest.update(bytes.fromhex(base_reference_sha256))
        digest.update(
            json.dumps(sorted(extension_ids), separators=(",", ":")).encode("ascii")
        )
        for reference in references:
            digest.update(reference.reference_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(reference.file_sha256))
        runtime = base_manifest.get("runtime")
        if not isinstance(runtime, Mapping):
            raise BadgeGalleryBuildError("base badge runtime metadata is missing")
        extension_sources = extension.get("sources")
        catalog_source = (
            extension_sources.get("catalog")
            if isinstance(extension_sources, Mapping)
            else None
        )
        fixed_icon_source = (
            extension_sources.get("fixed_icons")
            if isinstance(extension_sources, Mapping)
            else None
        )
        catalog_repository, catalog_revision, catalog_license = _source_provenance(
            catalog_source,
            label="catalog",
        )
        repository, revision, license_name = _source_provenance(
            fixed_icon_source,
            label="fixed-icon",
        )
        source = dict(base_source)
        source.update(
            {
                "repository": repository,
                "revision": revision,
                "license": license_name,
                "catalog_repository": catalog_repository,
                "catalog_revision": catalog_revision,
                "catalog_license": catalog_license,
                "base_manifest_sha256": sha256_file(base_component_root / "manifest.json"),
                "arena_component": arena_manifest.get("component"),
                "arena_component_manifest_sha256": arena_manifest_sha256,
                "arena_gallery_sha256": arena_gallery_sha256,
                "extension_dataset_id": extension.get("dataset_id"),
                "extension_dataset_manifest_sha256": sha256_file(extension_manifest),
                "base_reference_file_count": len(base["business_ids"]),
                "retained_reference_file_count": len(retained["business_ids"]),
                "replaced_reference_file_count": int((~retained_mask).sum()),
                "added_reference_file_count": len(references),
                "reference_file_count": len(business_ids),
                "reference_set_sha256": digest.hexdigest().upper(),
                "reference_set_digest_contract": "sha256(base_reference_set_sha256 || replaced_business_ids_json || ordered(name || NUL || png_sha256))",
            }
        )
        manifest = {
            "schema_version": 1,
            "component": base_manifest.get("component", "arena_card_reference"),
            "build": {
                "mode": "GALLERY_ONLY",
                "tool_contract": "task095-card-badge-extension-v1",
                "preprocess_contract": "rgb-png-to-bgr-96-area-top48-to-16x8-area-green-mask-v1",
                "unaffected_base_rows": "BITWISE_COPIED",
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
                "pillow_version": PIL.__version__,
                "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            },
            "source": source,
            "gallery": {
                "path": gallery_path.name,
                "sha256": sha256_file(gallery_path),
                "business_card_id_count": len(set(map(int, business_ids))),
                "visual_group_count": len(set(map(str, visual_groups))),
                "normalization_size": [96, 96],
            },
            "runtime": dict(runtime),
            "production_handoff": {
                "status": (
                    "PENDING_VALIDATION"
                    if inherited_live_updates
                    else arena_handoff_status
                ),
                "reason": (
                    "inherited private-live lineage requires a fresh evaluation bound to "
                    "the extended gallery"
                    if inherited_live_updates
                    else "badge gallery is bound to the exact release-ready arena component"
                ),
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-component-root", type=Path, required=True)
    parser.add_argument("--arena-component-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--extension-manifest", type=Path)
    source.add_argument("--live-reference-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = build_badge_gallery(
        base_component_root=args.base_component_root,
        arena_component_root=args.arena_component_root,
        extension_manifest=args.extension_manifest,
        live_reference_manifest=args.live_reference_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
