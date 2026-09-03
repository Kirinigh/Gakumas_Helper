"""Build a deterministic card-embedding gallery without training a model.

The builder preserves accepted gallery rows when their canonical model input is
unchanged.  Only new or changed inputs are sent through the pinned ONNX encoder.
Raw game art and rendered references remain local-only inputs; the output contains
only the model, numeric gallery, class table, and a portable manifest.
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import zlib
import shutil
import hashlib
import argparse
from copy import deepcopy
from typing import Any
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import PIL
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.local_artifacts import shared_local_artifact
from agent.card_selection.model import sha256_file
from agent.card_selection.embedding import (
    EmbeddingGallery,
    OnnxCardEmbedder,
    focus_card_art,
    extract_upgrade_marker,
)

SOURCE_DOMAIN_OFFICIAL_RAW_ART = "OFFICIAL_RAW_CARD_ART"
SOURCE_DOMAIN_FIXED_RENDERED = "FIXED_RENDERED_PNG"
SOURCE_DOMAINS = (SOURCE_DOMAIN_OFFICIAL_RAW_ART, SOURCE_DOMAIN_FIXED_RENDERED)
INPUT_MODE_FULL = "FULL"
INPUT_MODE_EXTENSION = "EXTENSION"
INPUT_MODES = (INPUT_MODE_FULL, INPUT_MODE_EXTENSION)
PREPROCESS_CONTRACTS = {
    SOURCE_DOMAIN_OFFICIAL_RAW_ART: "task075-raw-art-focus-96-lanczos-64-bilinear-rgb-v1",
    SOURCE_DOMAIN_FIXED_RENDERED: "task095-fixed-render-focus-64-bilinear-rgb-candidate-v1",
}
PRODUCTION_HANDOFF_BLOCKED = "PRODUCTION_HANDOFF_BLOCKED"
PRODUCTION_HANDOFF_PENDING = "PENDING_VALIDATION"


class GalleryBuildError(RuntimeError):
    """Raised when a gallery-only batch cannot be built safely."""


@dataclass(frozen=True)
class SourceSample:
    class_name: str
    business_id: int
    internal_id: str
    upgrade_count: int
    embedding_path: Path
    embedding_sha256: str
    marker_path: Path
    marker_sha256: str


@dataclass(frozen=True)
class DatasetPlan:
    manifest_path: Path
    manifest_sha256: str
    dataset_id: str
    dataset_revision: str
    source_domain: str
    business_ids: tuple[int, ...]
    samples: tuple[SourceSample, ...]
    ambiguity_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BaseGallery:
    manifest: dict[str, Any]
    manifest_path: Path
    model_path: Path
    model_sha256: str
    gallery_path: Path
    gallery_sha256: str
    class_table_path: Path
    class_names: tuple[str, ...]
    embeddings: np.ndarray
    card_ids: tuple[str, ...]
    visual_group_ids: tuple[str, ...]
    upgrade_counts: tuple[int, ...]
    upgrade_markers: np.ndarray


@dataclass(frozen=True)
class GalleryBuildResult:
    output_dir: Path
    manifest: dict[str, Any]
    reused_embedding_rows: int
    regenerated_embedding_rows: int
    cache_hits: int
    encoder_invocations: int


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GalleryBuildError(f"{label} is unavailable or invalid: {path}") from error
    if not isinstance(value, dict):
        raise GalleryBuildError(f"{label} must be a JSON object")
    return value


def _resolve_inside(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise GalleryBuildError(f"{label} path is missing")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise GalleryBuildError(f"{label} path must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise GalleryBuildError(f"{label} path escapes the dataset root")
    return resolved


def _declared_sha256(expected_sha256: object, *, label: str) -> str:
    expected = str(expected_sha256).upper()
    if len(expected) != 64 or any(character not in "0123456789ABCDEF" for character in expected):
        raise GalleryBuildError(f"{label} has an invalid SHA-256 declaration")
    return expected


def _verified_file(path: Path, expected_sha256: object, *, label: str) -> str:
    expected = _declared_sha256(expected_sha256, label=label)
    if not path.is_file():
        raise GalleryBuildError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise GalleryBuildError(f"{label} SHA-256 mismatch: {path}")
    return actual


def _business_id(class_name: str) -> int:
    prefix = class_name.split("_", 1)[0]
    if not prefix.isdigit() or int(prefix) < 1:
        raise GalleryBuildError(f"invalid skill-card class name: {class_name}")
    return int(prefix)


def _natural_class_key(class_name: str) -> tuple[int, int, str]:
    prefix, separator, suffix = class_name.partition("_")
    return int(prefix), int(suffix) if separator and suffix.isdigit() else -1, class_name


def _exact_id_set_sha256(values: tuple[int, ...]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def _validate_scope(
    manifest: dict[str, Any],
    business_ids: tuple[int, ...],
    *,
    require_explicit_id_set: bool,
) -> None:
    scope = manifest.get("scope", {})
    if not isinstance(scope, dict):
        raise GalleryBuildError("dataset scope must be an object")
    declarations: list[set[int]] = []
    for key in ("business_card_ids", "business_ids"):
        if key not in scope:
            continue
        raw_ids = scope[key]
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in raw_ids
            )
            or len(raw_ids) != len(set(raw_ids))
        ):
            raise GalleryBuildError("dataset scope exact business-card ID set is inconsistent")
        declarations.append(set(raw_ids))
    for key in ("business_card_id_ranges", "business_id_ranges"):
        if key not in scope:
            continue
        raw_ranges = scope[key]
        if not isinstance(raw_ranges, list) or not raw_ranges:
            raise GalleryBuildError("dataset scope business-card ID ranges are invalid")
        values: set[int] = set()
        for raw_range in raw_ranges:
            if (
                not isinstance(raw_range, list)
                or len(raw_range) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_range)
                or raw_range[0] < 1
                or raw_range[1] < raw_range[0]
            ):
                raise GalleryBuildError("dataset scope business-card ID ranges are invalid")
            current = set(range(raw_range[0], raw_range[1] + 1))
            if values.intersection(current):
                raise GalleryBuildError("dataset scope business-card ID ranges overlap")
            values.update(current)
        declarations.append(values)
    for minimum_key, maximum_key in (
        ("business_card_id_min", "business_card_id_max"),
        ("business_id_min", "business_id_max"),
    ):
        minimum = scope.get(minimum_key)
        maximum = scope.get(maximum_key)
        if minimum is None and maximum is None:
            continue
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum < 1
            or maximum < minimum
        ):
            raise GalleryBuildError("dataset scope business-card ID range is invalid")
        declarations.append(set(range(minimum, maximum + 1)))
    if require_explicit_id_set and not declarations:
        raise GalleryBuildError("extension dataset scope must declare its exact business-card ID set")
    if any(declared != set(business_ids) for declared in declarations):
        raise GalleryBuildError("dataset scope exact business-card ID declarations disagree")


def _ambiguity_groups(
    manifest: dict[str, Any],
    business_ids: set[int],
    inherited_business_ids: set[int],
) -> tuple[tuple[int, ...], ...]:
    validation = manifest.get("validation", {})
    raw_groups = validation.get("known_visual_ambiguities", [])
    if not isinstance(raw_groups, list):
        raise GalleryBuildError("known visual ambiguities must be a list")
    groups: list[tuple[int, ...]] = []
    for raw in raw_groups:
        if not isinstance(raw, dict) or not isinstance(raw.get("business_ids"), list):
            raise GalleryBuildError("known visual ambiguity entry is invalid")
        values = raw["business_ids"]
        if (
            len(values) < 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
            or len(values) != len(set(values))
            or not set(values).issubset(business_ids | inherited_business_ids)
            or not set(values).intersection(business_ids)
        ):
            raise GalleryBuildError("known visual ambiguity business-card IDs are invalid")
        groups.append(tuple(sorted(values)))
    return tuple(sorted(set(groups)))


def load_dataset_plan(
    manifest_path: Path,
    *,
    source_domain: str,
    verify_source_files: bool = True,
    require_explicit_id_set: bool = False,
    inherited_business_ids: frozenset[int] = frozenset(),
) -> DatasetPlan:
    """Load and verify one immutable gallery source manifest."""

    if source_domain not in SOURCE_DOMAINS:
        raise GalleryBuildError(f"unsupported source domain: {source_domain}")
    manifest_path = manifest_path.resolve()
    manifest = _load_object(manifest_path, label="dataset manifest")
    if manifest.get("schema_version") != 1:
        raise GalleryBuildError("unsupported dataset manifest schema")
    declared_domain = manifest.get("source_domain")
    if declared_domain is not None and declared_domain != source_domain:
        raise GalleryBuildError("dataset manifest source domain disagrees with the explicit build domain")
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "CLEAR":
        state = validation.get("status", "MISSING") if isinstance(validation, dict) else "MISSING"
        raise GalleryBuildError(f"dataset validation is not CLEAR: {state}")
    for field in ("missing_ids", "surplus_classes", "decode_failures", "unexpected_cross_id_duplicate_groups"):
        if validation.get(field, []) not in ([], None):
            raise GalleryBuildError(f"dataset validation still contains {field}")

    dataset_root = manifest_path.parent
    mapping = manifest.get("mapping")
    if not isinstance(mapping, dict):
        raise GalleryBuildError("dataset mapping metadata is missing")
    crosswalk_path = _resolve_inside(dataset_root, mapping.get("crosswalk_path"), label="crosswalk")
    _verified_file(crosswalk_path, mapping.get("crosswalk_sha256"), label="crosswalk")
    try:
        crosswalk = json.loads(crosswalk_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GalleryBuildError("crosswalk is invalid") from error
    if not isinstance(crosswalk, list) or not crosswalk:
        raise GalleryBuildError("crosswalk must be a non-empty list")

    rows_by_id: dict[int, dict[str, Any]] = {}
    for raw in crosswalk:
        if not isinstance(raw, dict):
            raise GalleryBuildError("crosswalk rows must be objects")
        business_id = raw.get("business_id")
        upgrade_count = raw.get("upgrade_count")
        internal_id = raw.get("internal_id")
        variants = raw.get("asset_variants")
        if (
            isinstance(business_id, bool)
            or not isinstance(business_id, int)
            or business_id < 1
            or business_id in rows_by_id
            or isinstance(upgrade_count, bool)
            or upgrade_count not in (0, 1)
            or not isinstance(internal_id, str)
            or not internal_id
            or not isinstance(variants, list)
            or not variants
            or not all(isinstance(value, str) and value for value in variants)
            or len(variants) != len(set(variants))
        ):
            raise GalleryBuildError("crosswalk row identity metadata is invalid")
        rows_by_id[business_id] = raw
    business_ids = tuple(sorted(rows_by_id))
    _validate_scope(
        manifest,
        business_ids,
        require_explicit_id_set=require_explicit_id_set,
    )
    for count_name in ("catalog_count", "decoded_master_count", "one_to_one_count"):
        count = mapping.get(count_name)
        if count is not None and count != len(business_ids):
            raise GalleryBuildError(f"dataset mapping {count_name} is inconsistent")
    expected_count = validation.get("expected_business_id_count")
    if expected_count is not None and expected_count != len(business_ids):
        raise GalleryBuildError("dataset validation business-card count is inconsistent")

    icons = manifest.get("icons")
    if not isinstance(icons, dict) or not isinstance(icons.get("images"), list):
        raise GalleryBuildError("dataset icon inventory is missing")
    icon_records = icons["images"]
    if icons.get("count") != len(icon_records):
        raise GalleryBuildError("dataset icon inventory count is inconsistent")
    icon_root = _resolve_inside(dataset_root, icons.get("root"), label="icon root")

    art_records_by_asset: dict[str, dict[str, Any]] = {}
    art_root: Path | None = None
    if source_domain == SOURCE_DOMAIN_OFFICIAL_RAW_ART:
        art = manifest.get("art")
        if not isinstance(art, dict) or not isinstance(art.get("images"), list):
            raise GalleryBuildError("official raw-art inventory is missing")
        if art.get("count") != len(art["images"]):
            raise GalleryBuildError("official raw-art inventory count is inconsistent")
        art_root = _resolve_inside(dataset_root, art.get("root"), label="art root")
        for raw in art["images"]:
            if not isinstance(raw, dict) or not isinstance(raw.get("asset_name"), str):
                raise GalleryBuildError("official raw-art record is invalid")
            asset_name = raw["asset_name"]
            if asset_name in art_records_by_asset:
                raise GalleryBuildError("official raw-art inventory contains duplicate asset names")
            art_records_by_asset[asset_name] = raw

    samples: list[SourceSample] = []
    class_names: set[str] = set()
    icon_paths: set[str] = set()
    seen_file_hashes: dict[Path, str] = {}

    def verify_once(path: Path, expected: object, *, label: str) -> str:
        expected_text = _declared_sha256(expected, label=label)
        previous = seen_file_hashes.get(path)
        if previous is not None:
            if previous != expected_text:
                raise GalleryBuildError(f"{label} has conflicting SHA-256 declarations")
            return previous
        verified = (
            _verified_file(path, expected_text, label=label)
            if verify_source_files
            else expected_text
        )
        seen_file_hashes[path] = verified
        return verified

    for raw in icon_records:
        if not isinstance(raw, dict):
            raise GalleryBuildError("dataset icon records must be objects")
        class_name = raw.get("class_name")
        business_id = raw.get("business_id")
        asset_variant = raw.get("asset_variant")
        relative_path = raw.get("path")
        if (
            not isinstance(class_name, str)
            or class_name in class_names
            or isinstance(business_id, bool)
            or not isinstance(business_id, int)
            or business_id not in rows_by_id
            or _business_id(class_name) != business_id
            or not isinstance(asset_variant, str)
            or asset_variant not in rows_by_id[business_id]["asset_variants"]
            or not isinstance(relative_path, str)
            or relative_path in icon_paths
        ):
            raise GalleryBuildError("dataset icon class mapping is invalid")
        class_names.add(class_name)
        icon_paths.add(relative_path)
        marker_path = _resolve_inside(icon_root, relative_path, label=f"icon {class_name}")
        marker_sha256 = verify_once(marker_path, raw.get("sha256"), label=f"icon {class_name}")

        if source_domain == SOURCE_DOMAIN_OFFICIAL_RAW_ART:
            art_record = art_records_by_asset.get(asset_variant)
            if art_record is None or art_root is None:
                raise GalleryBuildError(f"official raw art is missing for {class_name}")
            embedding_path = _resolve_inside(
                art_root,
                art_record.get("path"),
                label=f"official raw art {asset_variant}",
            )
            embedding_sha256 = verify_once(
                embedding_path,
                art_record.get("sha256"),
                label=f"official raw art {asset_variant}",
            )
        else:
            embedding_path = marker_path
            embedding_sha256 = marker_sha256

        row = rows_by_id[business_id]
        samples.append(
            SourceSample(
                class_name=class_name,
                business_id=business_id,
                internal_id=str(row["internal_id"]),
                upgrade_count=int(row["upgrade_count"]),
                embedding_path=embedding_path,
                embedding_sha256=embedding_sha256,
                marker_path=marker_path,
                marker_sha256=marker_sha256,
            )
        )

    sample_ids = {sample.business_id for sample in samples}
    if sample_ids != set(business_ids):
        missing = sorted(set(business_ids) - sample_ids)
        surplus = sorted(sample_ids - set(business_ids))
        raise GalleryBuildError(f"dataset icon coverage disagrees: missing={missing}, surplus={surplus}")
    samples.sort(key=lambda sample: _natural_class_key(sample.class_name))
    return DatasetPlan(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        dataset_id=str(manifest.get("dataset_id", "")),
        dataset_revision=str(manifest.get("dataset_revision", "")),
        source_domain=source_domain,
        business_ids=business_ids,
        samples=tuple(samples),
        ambiguity_groups=_ambiguity_groups(
            manifest,
            set(business_ids),
            set(inherited_business_ids),
        ),
    )


def merge_extension_plan(base: DatasetPlan, extension: DatasetPlan) -> DatasetPlan:
    """Overlay an exact ID-level extension without copying inherited source images."""

    if base.source_domain != extension.source_domain:
        raise GalleryBuildError("extension and inherited base datasets must use the same source domain")
    replaced_ids = set(extension.business_ids)
    samples = [sample for sample in base.samples if sample.business_id not in replaced_ids]
    samples.extend(extension.samples)
    samples.sort(key=lambda sample: _natural_class_key(sample.class_name))
    class_names = [sample.class_name for sample in samples]
    if len(class_names) != len(set(class_names)):
        raise GalleryBuildError("extension overlay produces duplicate gallery class names")
    business_ids = tuple(sorted(set(base.business_ids) | replaced_ids))
    sample_ids = {sample.business_id for sample in samples}
    if sample_ids != set(business_ids):
        raise GalleryBuildError("extension overlay does not cover the merged exact business-card ID set")
    inherited_ambiguities = tuple(
        group for group in base.ambiguity_groups if not set(group).intersection(replaced_ids)
    )
    return DatasetPlan(
        manifest_path=extension.manifest_path,
        manifest_sha256=extension.manifest_sha256,
        dataset_id=extension.dataset_id,
        dataset_revision=extension.dataset_revision,
        source_domain=extension.source_domain,
        business_ids=business_ids,
        samples=tuple(samples),
        ambiguity_groups=tuple(sorted(set(inherited_ambiguities + extension.ambiguity_groups))),
    )


class _DisjointSet:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _visual_components(plan: DatasetPlan) -> tuple[tuple[int, ...], ...]:
    groups = _DisjointSet(plan.business_ids)
    by_internal_id: dict[str, list[int]] = defaultdict(list)
    for sample in plan.samples:
        by_internal_id[sample.internal_id].append(sample.business_id)
    for values in by_internal_id.values():
        first = values[0]
        for value in values[1:]:
            groups.union(first, value)
    for values in plan.ambiguity_groups:
        first = values[0]
        for value in values[1:]:
            groups.union(first, value)
    components: dict[int, set[int]] = defaultdict(set)
    for value in plan.business_ids:
        components[groups.find(value)].add(value)
    return tuple(sorted((tuple(sorted(values)) for values in components.values()), key=lambda value: value[0]))


def _groups_by_business_id(base: BaseGallery) -> dict[int, str]:
    groups: dict[int, set[str]] = defaultdict(set)
    for card_id, visual_group in zip(base.card_ids, base.visual_group_ids, strict=True):
        groups[int(card_id)].add(visual_group)
    inconsistent = {card_id: values for card_id, values in groups.items() if len(values) != 1}
    if inconsistent:
        raise GalleryBuildError(f"base gallery maps one business ID to multiple visual groups: {inconsistent}")
    return {card_id: next(iter(values)) for card_id, values in groups.items()}


def assign_visual_groups(plan: DatasetPlan, base: BaseGallery) -> dict[int, str]:
    """Resolve manifest-declared visual components to stable opaque group IDs."""

    base_groups = _groups_by_business_id(base)
    internal_ids: dict[int, str] = {}
    for sample in plan.samples:
        previous = internal_ids.setdefault(sample.business_id, sample.internal_id)
        if previous != sample.internal_id:
            raise GalleryBuildError("one business ID maps to multiple internal card IDs")
    components = _visual_components(plan)
    inherited_by_component = [
        {base_groups[value] for value in component if value in base_groups}
        for component in components
    ]
    inherited_occurrences: dict[str, int] = defaultdict(int)
    for inherited in inherited_by_component:
        for label in inherited:
            inherited_occurrences[label] += 1
    resolved: dict[int, str] = {}
    used_labels: set[str] = set()
    for component, inherited in zip(components, inherited_by_component, strict=True):
        component_internal_ids = {internal_ids[value] for value in component}
        stable_inherited = {
            label for label in inherited if inherited_occurrences[label] == 1
        }
        if len(inherited) == 1 and len(stable_inherited) == 1:
            label = next(iter(stable_inherited))
        elif len(component_internal_ids) == 1:
            label = next(iter(component_internal_ids))
        else:
            digest = hashlib.sha256(
                json.dumps(list(component), separators=(",", ":")).encode("ascii")
            ).hexdigest()[:16]
            label = f"manifest-ambiguity-{digest}"
        if label in used_labels:
            raise GalleryBuildError("two manifest visual components resolved to the same label")
        used_labels.add(label)
        for value in component:
            resolved[value] = label
    return resolved


def _load_base_gallery(component_root: Path, dataset_plan: DatasetPlan) -> BaseGallery:
    component_root = component_root.resolve()
    manifest_path = component_root / "manifest.json"
    manifest = _load_object(manifest_path, label="base component manifest")
    if manifest.get("schema_version") != 1:
        raise GalleryBuildError("unsupported base component manifest schema")
    model = manifest.get("model")
    gallery = manifest.get("gallery")
    if not isinstance(model, dict) or not isinstance(gallery, dict):
        raise GalleryBuildError("base component model/gallery metadata is incomplete")
    model_path = _resolve_inside(component_root, model.get("path"), label="base model")
    model_sha256 = _verified_file(model_path, model.get("sha256"), label="base model")
    gallery_path = _resolve_inside(component_root, gallery.get("path"), label="base gallery")
    gallery_sha256 = _verified_file(gallery_path, gallery.get("sha256"), label="base gallery")
    class_table_path = _resolve_inside(
        component_root,
        gallery.get("class_table_path"),
        label="base class table",
    )
    _verified_file(
        class_table_path,
        gallery.get("class_table_sha256"),
        label="base class table",
    )
    declared_dataset_sha256 = manifest.get("source", {}).get("dataset_manifest_sha256")
    if declared_dataset_sha256 is not None and str(declared_dataset_sha256).upper() != dataset_plan.manifest_sha256:
        raise GalleryBuildError("base component is bound to a different dataset manifest")

    EmbeddingGallery.load(gallery_path, manifest_path)
    try:
        class_table = json.loads(class_table_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GalleryBuildError("base class table is invalid") from error
    if not isinstance(class_table, list) or not all(isinstance(value, str) for value in class_table):
        raise GalleryBuildError("base class table is invalid")
    with np.load(gallery_path, allow_pickle=False) as payload:
        required = {
            "embeddings",
            "class_names",
            "card_ids",
            "visual_group_ids",
            "upgrade_counts",
            "upgrade_markers",
        }
        if set(payload.files) != required:
            raise GalleryBuildError("base gallery array inventory is incompatible")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32).copy()
        class_names = tuple(str(value) for value in payload["class_names"])
        card_ids = tuple(str(value) for value in payload["card_ids"])
        visual_group_ids = tuple(str(value) for value in payload["visual_group_ids"])
        upgrade_counts = tuple(int(value) for value in payload["upgrade_counts"])
        upgrade_markers = np.asarray(payload["upgrade_markers"], dtype=np.float32).copy()
    if tuple(class_table) != class_names:
        raise GalleryBuildError("base class table and gallery order disagree")
    expected_classes = tuple(sample.class_name for sample in dataset_plan.samples)
    if class_names != expected_classes:
        raise GalleryBuildError("base gallery class table differs from the supplied base dataset")
    try:
        actual_business_ids = tuple(sorted({int(value) for value in card_ids}))
    except ValueError as error:
        raise GalleryBuildError("base gallery contains a non-numeric business-card ID") from error
    if actual_business_ids != dataset_plan.business_ids:
        raise GalleryBuildError("base gallery exact business-card ID set differs from the base dataset lineage")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise GalleryBuildError("base component source metadata is missing")
    _validate_scope(
        {"scope": source},
        actual_business_ids,
        require_explicit_id_set=True,
    )
    declared_count = source.get("business_card_id_count")
    if declared_count is not None and (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(actual_business_ids)
    ):
        raise GalleryBuildError("base component business-card ID count is inconsistent")
    declared_visual_count = source.get("visual_identity_count")
    if (
        isinstance(declared_visual_count, bool)
        or not isinstance(declared_visual_count, int)
        or declared_visual_count != len(set(visual_group_ids))
    ):
        raise GalleryBuildError("base component visual-identity count is inconsistent")
    declared_domain = source.get("source_domain")
    if declared_domain is not None and declared_domain != dataset_plan.source_domain:
        raise GalleryBuildError("base component source domain differs from the explicit base domain")
    return BaseGallery(
        manifest=manifest,
        manifest_path=manifest_path,
        model_path=model_path,
        model_sha256=model_sha256,
        gallery_path=gallery_path,
        gallery_sha256=gallery_sha256,
        class_table_path=class_table_path,
        class_names=class_names,
        embeddings=embeddings,
        card_ids=card_ids,
        visual_group_ids=visual_group_ids,
        upgrade_counts=upgrade_counts,
        upgrade_markers=upgrade_markers,
    )


def _canonical_rgb(path: Path, *, source_domain: str) -> np.ndarray:
    try:
        with Image.open(path) as source:
            focused = focus_card_art(source)
            if source_domain == SOURCE_DOMAIN_OFFICIAL_RAW_ART:
                staged = focused.resize((96, 96), Image.Resampling.LANCZOS)
                canonical = staged.resize((64, 64), Image.Resampling.BILINEAR)
            elif source_domain == SOURCE_DOMAIN_FIXED_RENDERED:
                canonical = focused.resize((64, 64), Image.Resampling.BILINEAR)
            else:
                raise GalleryBuildError(f"unsupported source domain: {source_domain}")
            pixels = np.asarray(canonical.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise GalleryBuildError(f"could not decode gallery source image: {path}") from error
    if pixels.shape != (64, 64, 3):
        raise GalleryBuildError(f"canonical gallery source has an invalid shape: {path}")
    return np.ascontiguousarray(pixels)


def _pixel_sha256(pixels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pixels, dtype=np.uint8).tobytes()).hexdigest().upper()


def _cache_segment(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def vector_cache_path(
    cache_root: Path,
    *,
    model_sha256: str,
    preprocess_contract: str,
    pixel_sha256: str,
    encoder_runtime: str,
) -> Path:
    """Return the content-addressed path for one local embedding vector."""

    return (
        cache_root.resolve()
        / model_sha256.upper()
        / _cache_segment(preprocess_contract)
        / _cache_segment(encoder_runtime)
        / f"{pixel_sha256.upper()}.npy"
    )


def _validated_vector(value: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (128,) or not np.isfinite(vector).all():
        raise GalleryBuildError(f"{label} is not one finite float32[128] vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8 or abs(norm - 1.0) > 1e-3:
        raise GalleryBuildError(f"{label} is not L2-normalised")
    return np.ascontiguousarray(vector, dtype=np.float32)


def _load_cached_vector(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        value = np.load(path, allow_pickle=False)
        return _validated_vector(value, label="cached embedding")
    except (OSError, ValueError, GalleryBuildError):
        return None


def _store_cached_vector(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}-{uuid.uuid4().hex}.npy"
    try:
        with temporary.open("xb") as stream:
            np.save(stream, vector, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_inputs(plan: DatasetPlan) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    pixels_by_source: dict[tuple[Path, str], np.ndarray] = {}
    pixels_by_class: dict[str, np.ndarray] = {}
    hashes_by_class: dict[str, str] = {}
    for sample in plan.samples:
        key = sample.embedding_path, sample.embedding_sha256
        pixels = pixels_by_source.get(key)
        if pixels is None:
            pixels = _canonical_rgb(sample.embedding_path, source_domain=plan.source_domain)
            pixels_by_source[key] = pixels
        pixels_by_class[sample.class_name] = pixels
        hashes_by_class[sample.class_name] = _pixel_sha256(pixels)
    return pixels_by_class, hashes_by_class


def _upgrade_markers(plan: DatasetPlan) -> np.ndarray:
    markers_by_source: dict[tuple[Path, str], np.ndarray] = {}
    markers: list[np.ndarray] = []
    for sample in plan.samples:
        key = sample.marker_path, sample.marker_sha256
        marker = markers_by_source.get(key)
        if marker is None:
            try:
                with Image.open(sample.marker_path) as source:
                    marker = np.asarray(extract_upgrade_marker(source), dtype=np.float32)
            except (OSError, ValueError) as error:
                raise GalleryBuildError(f"could not decode upgrade marker: {sample.marker_path}") from error
            if marker.shape != (24, 24, 3) or not np.isfinite(marker).all():
                raise GalleryBuildError(f"upgrade marker is invalid: {sample.marker_path}")
            marker = np.ascontiguousarray(marker, dtype=np.float32)
            markers_by_source[key] = marker
        markers.append(marker)
    return np.stack(markers).astype(np.float32, copy=False)


def _extension_upgrade_markers(
    plan: DatasetPlan,
    extension: DatasetPlan,
    base: BaseGallery,
) -> np.ndarray:
    extension_values = _upgrade_markers(extension)
    extension_by_class = {
        sample.class_name: extension_values[index]
        for index, sample in enumerate(extension.samples)
    }
    base_by_class = {
        class_name: base.upgrade_markers[index]
        for index, class_name in enumerate(base.class_names)
    }
    markers: list[np.ndarray] = []
    for sample in plan.samples:
        marker = extension_by_class.get(sample.class_name)
        if marker is None:
            marker = base_by_class.get(sample.class_name)
        if marker is None:
            raise GalleryBuildError(f"extension marker source is unresolved: {sample.class_name}")
        markers.append(np.ascontiguousarray(marker, dtype=np.float32))
    return np.stack(markers).astype(np.float32, copy=False)


def _default_encoder_runtime() -> str:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise GalleryBuildError("onnxruntime is required for gallery-only encoding") from error
    return f"onnxruntime-{ort.__version__}:CPUExecutionProvider"


def _manifest_payload(
    *,
    plan: DatasetPlan,
    input_plan: DatasetPlan,
    input_mode: str,
    base_plan: DatasetPlan,
    base_dataset_lineage_sha256s: tuple[str, ...],
    base: BaseGallery,
    visual_groups: tuple[str, ...],
    gallery_name: str,
    gallery_sha256: str,
    gallery_bytes: int,
    class_table_name: str,
    class_table_sha256: str,
    class_table_bytes: int,
    model_name: str,
    model_bytes: int,
    reused_rows: int,
    regenerated_rows: int,
    encoder_runtime: str,
) -> dict[str, Any]:
    source = {
        "dataset_id": plan.dataset_id,
        "dataset_revision": plan.dataset_revision,
        "dataset_manifest_sha256": plan.manifest_sha256,
        "dataset_manifest_mode": input_mode,
        "source_domain": plan.source_domain,
        "business_card_ids": list(plan.business_ids),
        "business_card_id_count": len(plan.business_ids),
        "business_card_id_set_sha256": _exact_id_set_sha256(plan.business_ids),
        "image_class_count": len(plan.samples),
        "visual_identity_count": len(set(visual_groups)),
        "training_target": "official raw card art only"
        if plan.source_domain == SOURCE_DOMAIN_OFFICIAL_RAW_ART
        else "fixed rendered PNG candidate only",
    }
    if input_mode == INPUT_MODE_EXTENSION:
        source["extension_business_card_ids"] = list(input_plan.business_ids)
        source["extension_business_card_id_count"] = len(input_plan.business_ids)
    base_model = base.manifest.get("model", {})
    model = {
        "path": model_name,
        "bytes": model_bytes,
        "sha256": base.model_sha256,
        "input": base_model.get("input", "float32[N,3,64,64] RGB range [0,1]"),
        "output": base_model.get("output", "embedding float32[N,128] L2-normalised"),
    }
    base_gallery = base.manifest.get("gallery", {})
    gallery = {
        "path": gallery_name,
        "bytes": gallery_bytes,
        "sha256": gallery_sha256,
        "embedding_dim": 128,
        "distance": base_gallery.get("distance", "cosine_similarity"),
        "class_table_path": class_table_name,
        "class_table_bytes": class_table_bytes,
        "class_table_sha256": class_table_sha256,
    }
    if "top_k" in base_gallery:
        gallery["top_k"] = base_gallery["top_k"]
    handoff_status = (
        PRODUCTION_HANDOFF_BLOCKED
        if plan.source_domain == SOURCE_DOMAIN_FIXED_RENDERED
        else PRODUCTION_HANDOFF_PENDING
    )
    reason = (
        "fixed rendered PNG changes the accepted card-art embedding source domain and requires frozen-set equivalence evidence"
        if handoff_status == PRODUCTION_HANDOFF_BLOCKED
        else "gallery-only candidate requires targeted and full consumer validation before handoff"
    )
    return {
        "schema_version": 1,
        "component": base.manifest.get("component", "card_art_embedding_gallery_only_candidate"),
        "source": source,
        "build": {
            "mode": "GALLERY_ONLY",
            "input_mode": input_mode,
            "tool_contract": "task095-card-embedding-gallery-v1",
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pillow_version": PIL.__version__,
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            "base_dataset_manifest_sha256": base_plan.manifest_sha256,
            "base_dataset_lineage_manifest_sha256s": list(
                base_dataset_lineage_sha256s
            ),
            "base_gallery_sha256": base.gallery_sha256,
            "preprocess_contract": PREPROCESS_CONTRACTS[plan.source_domain],
            "encoder_runtime": encoder_runtime,
            "reused_embedding_row_count": reused_rows,
            "regenerated_embedding_row_count": regenerated_rows,
            "cache_key_fields": [
                "model_sha256",
                "preprocess_contract",
                "canonical_pixel_sha256",
                "encoder_runtime_and_provider",
            ],
        },
        "model": model,
        "gallery": gallery,
        "runtime": deepcopy(base.manifest.get("runtime", {})),
        "validation": {
            "state": "BLOCKED" if handoff_status == PRODUCTION_HANDOFF_BLOCKED else "PENDING",
            "note": reason,
        },
        "production_handoff": {"status": handoff_status, "reason": reason},
    }


def build_gallery(
    *,
    dataset_manifest: Path,
    base_dataset_manifest: Path | tuple[Path, ...] | list[Path],
    base_component_root: Path,
    output_dir: Path,
    input_mode: str,
    source_domain: str,
    base_source_domain: str,
    cache_dir: Path | None = None,
    embedder: Any | None = None,
    encoder_runtime: str | None = None,
) -> GalleryBuildResult:
    """Build one deterministic gallery-only candidate."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise GalleryBuildError(f"output directory already exists: {output_dir}")
    if input_mode not in INPUT_MODES:
        raise GalleryBuildError(f"unsupported input mode: {input_mode}")
    base_manifest_paths = (
        (base_dataset_manifest,)
        if isinstance(base_dataset_manifest, Path)
        else tuple(base_dataset_manifest)
    )
    if not base_manifest_paths or any(not isinstance(path, Path) for path in base_manifest_paths):
        raise GalleryBuildError("at least one ordered base dataset manifest is required")
    base_plan = load_dataset_plan(
        base_manifest_paths[0],
        source_domain=base_source_domain,
        verify_source_files=input_mode == INPUT_MODE_FULL,
    )
    base_dataset_lineage_sha256s = [base_plan.manifest_sha256]
    for base_layer_path in base_manifest_paths[1:]:
        base_layer = load_dataset_plan(
            base_layer_path,
            source_domain=base_source_domain,
            verify_source_files=input_mode == INPUT_MODE_FULL,
            require_explicit_id_set=True,
            inherited_business_ids=frozenset(base_plan.business_ids),
        )
        base_plan = merge_extension_plan(base_plan, base_layer)
        base_dataset_lineage_sha256s.append(base_layer.manifest_sha256)
    input_plan = load_dataset_plan(
        dataset_manifest,
        source_domain=source_domain,
        require_explicit_id_set=input_mode == INPUT_MODE_EXTENSION,
        inherited_business_ids=(
            frozenset(base_plan.business_ids)
            if input_mode == INPUT_MODE_EXTENSION
            else frozenset()
        ),
    )
    plan = (
        merge_extension_plan(base_plan, input_plan)
        if input_mode == INPUT_MODE_EXTENSION
        else input_plan
    )
    base = _load_base_gallery(base_component_root, base_plan)
    visual_group_by_id = assign_visual_groups(plan, base)
    visual_groups = tuple(visual_group_by_id[sample.business_id] for sample in plan.samples)

    pixels_by_class, pixel_hashes = _canonical_inputs(input_plan)
    base_pixel_hashes: dict[str, str] = {}
    if input_mode == INPUT_MODE_FULL:
        base_pixels, base_pixel_hashes = _canonical_inputs(base_plan)
        del base_pixels
    base_index = {class_name: index for index, class_name in enumerate(base.class_names)}
    input_class_names = {sample.class_name for sample in input_plan.samples}
    same_preprocess_contract = (
        PREPROCESS_CONTRACTS[base_plan.source_domain]
        == PREPROCESS_CONTRACTS[plan.source_domain]
    )

    if embedder is None:
        embedder = OnnxCardEmbedder(
            base.model_path,
            expected_model_sha256=base.model_sha256,
            source_color_order="RGB",
        )
    if str(getattr(embedder, "model_sha256", "")).upper() != base.model_sha256:
        raise GalleryBuildError("gallery-only embedder model SHA-256 differs from the accepted base model")
    encoder_runtime = encoder_runtime or _default_encoder_runtime()
    if not encoder_runtime or "provider" not in encoder_runtime.casefold():
        raise GalleryBuildError("encoder runtime fingerprint must include its provider")
    cache_root = (
        cache_dir.resolve()
        if cache_dir is not None
        else shared_local_artifact(PROJECT_ROOT, "task095-card-gallery-cache")
    )
    preprocess_contract = PREPROCESS_CONTRACTS[plan.source_domain]

    vectors: list[np.ndarray] = []
    reused_rows = regenerated_rows = cache_hits = encoder_invocations = 0
    for sample in plan.samples:
        base_row = base_index.get(sample.class_name)
        unchanged = input_mode == INPUT_MODE_EXTENSION and sample.class_name not in input_class_names
        if input_mode == INPUT_MODE_FULL:
            unchanged = (
                same_preprocess_contract
                and base_row is not None
                and base_pixel_hashes[sample.class_name] == pixel_hashes[sample.class_name]
            )
        if unchanged and base_row is None:
            raise GalleryBuildError(
                f"extension inherited class is absent from the accepted base gallery: {sample.class_name}"
            )
        if unchanged:
            vector = np.ascontiguousarray(
                base.embeddings[base_row],
                dtype=np.float32,
            )
            reused_rows += 1
        else:
            if sample.class_name not in pixel_hashes:
                raise GalleryBuildError(
                    f"regenerated class is absent from the input manifest: {sample.class_name}"
                )
            regenerated_rows += 1
            cache_path = vector_cache_path(
                cache_root,
                model_sha256=base.model_sha256,
                preprocess_contract=preprocess_contract,
                pixel_sha256=pixel_hashes[sample.class_name],
                encoder_runtime=encoder_runtime,
            )
            vector = _load_cached_vector(cache_path)
            if vector is None:
                pixels = pixels_by_class[sample.class_name].astype(np.float32) / 255.0
                tensor = np.transpose(pixels, (2, 0, 1))[None, ...]
                vector = _validated_vector(
                    embedder.embed_tensor(np.ascontiguousarray(tensor, dtype=np.float32)),
                    label="ONNX embedding",
                )
                _store_cached_vector(cache_path, vector)
                encoder_invocations += 1
            else:
                cache_hits += 1
        vectors.append(vector)

    embeddings = np.stack(vectors).astype(np.float32, copy=False)
    class_names = tuple(sample.class_name for sample in plan.samples)
    card_ids = tuple(str(sample.business_id) for sample in plan.samples)
    upgrade_counts = tuple(sample.upgrade_count for sample in plan.samples)
    upgrade_markers = (
        _extension_upgrade_markers(plan, input_plan, base)
        if input_mode == INPUT_MODE_EXTENSION
        else _upgrade_markers(plan)
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        base_model_name = Path(str(base.manifest["model"]["path"])).name
        base_gallery_name = Path(str(base.manifest["gallery"]["path"])).name
        base_classes_name = Path(str(base.manifest["gallery"]["class_table_path"])).name
        model_path = temporary / base_model_name
        gallery_path = temporary / base_gallery_name
        classes_path = temporary / base_classes_name
        shutil.copyfile(base.model_path, model_path)
        np.savez_compressed(
            gallery_path,
            embeddings=embeddings,
            class_names=np.asarray(class_names),
            card_ids=np.asarray(card_ids),
            visual_group_ids=np.asarray(visual_groups),
            upgrade_counts=np.asarray(upgrade_counts, dtype=np.int8),
            upgrade_markers=upgrade_markers,
        )
        classes_path.write_text(
            json.dumps(list(class_names), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = _manifest_payload(
            plan=plan,
            input_plan=input_plan,
            input_mode=input_mode,
            base_plan=base_plan,
            base_dataset_lineage_sha256s=tuple(base_dataset_lineage_sha256s),
            base=base,
            visual_groups=visual_groups,
            gallery_name=gallery_path.name,
            gallery_sha256=sha256_file(gallery_path),
            gallery_bytes=gallery_path.stat().st_size,
            class_table_name=classes_path.name,
            class_table_sha256=sha256_file(classes_path),
            class_table_bytes=classes_path.stat().st_size,
            model_name=model_path.name,
            model_bytes=model_path.stat().st_size,
            reused_rows=reused_rows,
            regenerated_rows=regenerated_rows,
            encoder_runtime=encoder_runtime,
        )
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        loaded = EmbeddingGallery.load(gallery_path, manifest_path)
        if loaded.class_names != class_names or loaded.card_ids != card_ids or loaded.visual_group_ids != visual_groups:
            raise GalleryBuildError("written gallery metadata differs from the deterministic build plan")
        os.replace(temporary, output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return GalleryBuildResult(
        output_dir=output_dir,
        manifest=manifest,
        reused_embedding_rows=reused_rows,
        regenerated_embedding_rows=regenerated_rows,
        cache_hits=cache_hits,
        encoder_invocations=encoder_invocations,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--base-dataset-manifest",
        type=Path,
        action="append",
        required=True,
        help="repeat in chronological order when the accepted base already contains extensions",
    )
    parser.add_argument("--base-component-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-mode", choices=INPUT_MODES, required=True)
    parser.add_argument("--source-domain", choices=SOURCE_DOMAINS, required=True)
    parser.add_argument(
        "--base-source-domain",
        choices=SOURCE_DOMAINS,
        required=True,
    )
    parser.add_argument("--cache-dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_gallery(
        dataset_manifest=args.dataset_manifest,
        base_dataset_manifest=args.base_dataset_manifest,
        base_component_root=args.base_component_root,
        output_dir=args.output_dir,
        input_mode=args.input_mode,
        source_domain=args.source_domain,
        base_source_domain=args.base_source_domain,
        cache_dir=args.cache_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "gallery_sha256": result.manifest["gallery"]["sha256"],
                "production_handoff": result.manifest["production_handoff"]["status"],
                "reused_embedding_rows": result.reused_embedding_rows,
                "regenerated_embedding_rows": result.regenerated_embedding_rows,
                "cache_hits": result.cache_hits,
                "encoder_invocations": result.encoder_invocations,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
