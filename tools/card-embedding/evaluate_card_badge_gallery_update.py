"""Evaluate and bind a private-live arena card-reference gallery update.

The evaluator deliberately publishes only aggregate identity evidence.  Raw live
images, regions, per-sample hashes, and the private manifest identity stay local.
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import shutil
import hashlib
import argparse
from copy import deepcopy
from typing import Any
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

TOOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (TOOL_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import build_card_badge_gallery as badge_builder  # noqa: E402
from build_card_embedding_gallery import (  # noqa: E402
    SOURCE_DOMAIN_FIXED_RENDERED,
    SOURCE_DOMAIN_OFFICIAL_RAW_ART,
    load_dataset_plan,
)


class BadgeGalleryUpdateEvaluationError(RuntimeError):
    """The badge-gallery update does not satisfy its fail-closed evidence gates."""


EXPECTED_ARRAY_NAMES = (
    "business_ids",
    "visual_group_ids",
    "reference_names",
    "top_coarse",
    "green_masks",
)
TOOL_CONTRACT = "task095-card-badge-hard-negative-evaluation-v1"
LIVE_APPEND_TOOL_CONTRACT = "task095-card-badge-live-reference-append-v1"
PROMOTED_REPORT_NAME = "hard_negative_evaluation.json"
PROMOTED_HANDOFF_STATUS = "READY_WITH_REAL_SAMPLES_PENDING"


@dataclass(frozen=True)
class Component:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    gallery_path: Path
    gallery_sha256: str
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class EvaluationScope:
    target_business_id: int
    target_visual_group_id: str
    sibling_business_ids: tuple[int, ...]
    hard_negative_business_ids: tuple[int, ...]
    baseline_wrong_business_id: int
    baseline_wrong_visual_group_id: str


@dataclass(frozen=True)
class LiveInput:
    image: np.ndarray
    box: tuple[int, int, int, int]
    business_id: int
    visual_group_id: str


@dataclass(frozen=True)
class Measurement:
    business_id: int
    visual_group_id: str
    top_error: float
    group_margin: float
    low_confidence: bool
    unambiguous: bool


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
        raise BadgeGalleryUpdateEvaluationError(
            f"{label} is unavailable or invalid"
        ) from error
    if not isinstance(payload, dict):
        raise BadgeGalleryUpdateEvaluationError(f"{label} must be a JSON object")
    return payload


def _load_component(root: Path, *, label: str) -> Component:
    root = root.resolve()
    try:
        manifest, arrays = badge_builder._load_base(root)
    except badge_builder.BadgeGalleryBuildError as error:
        raise BadgeGalleryUpdateEvaluationError(
            f"{label} component is invalid: {error}"
        ) from error
    gallery = manifest.get("gallery")
    if not isinstance(gallery, Mapping):
        raise BadgeGalleryUpdateEvaluationError(f"{label} gallery metadata is missing")
    raw_path = gallery.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BadgeGalleryUpdateEvaluationError(f"{label} gallery path is missing")
    gallery_path = (root / raw_path).resolve()
    if not gallery_path.is_relative_to(root):
        raise BadgeGalleryUpdateEvaluationError(f"{label} gallery path escapes its root")
    try:
        with np.load(gallery_path, allow_pickle=False) as payload:
            if tuple(payload.files) != EXPECTED_ARRAY_NAMES:
                raise BadgeGalleryUpdateEvaluationError(
                    f"{label} gallery array inventory is not the frozen five-array contract"
                )
    except BadgeGalleryUpdateEvaluationError:
        raise
    except (OSError, EOFError, TypeError, ValueError) as error:
        raise BadgeGalleryUpdateEvaluationError(
            f"{label} gallery is unavailable or invalid"
        ) from error
    manifest_path = root / "manifest.json"
    return Component(
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        gallery_path=gallery_path,
        gallery_sha256=sha256_file(gallery_path),
        arrays=arrays,
    )


def _positive_id(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BadgeGalleryUpdateEvaluationError(f"{label} must be a positive integer")
    return value


def _positive_ids(value: object, *, label: str, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise BadgeGalleryUpdateEvaluationError(f"{label} must be an ID array")
    values = tuple(_positive_id(item, label=label) for item in value)
    if len(values) != len(set(values)):
        raise BadgeGalleryUpdateEvaluationError(f"{label} contains duplicate IDs")
    return tuple(sorted(values))


def _visual_group(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BadgeGalleryUpdateEvaluationError(f"{label} must be a non-empty string")
    return value


def _declared_value(
    declarations: Mapping[str, Any],
    key: str,
    override: object,
    *,
    required: bool = True,
) -> object:
    declared = declarations.get(key)
    if override is not None and declared is not None and override != declared:
        raise BadgeGalleryUpdateEvaluationError(
            f"private evaluation declaration disagrees with --{key.replace('_', '-')}"
        )
    value = override if override is not None else declared
    if required and value is None:
        raise BadgeGalleryUpdateEvaluationError(
            f"private evaluation declaration {key} is missing"
        )
    return value


def _load_scope(
    private_manifest: Mapping[str, Any],
    *,
    target_business_id: int | None,
    target_visual_group_id: str | None,
    sibling_business_ids: Sequence[int] | None,
    hard_negative_business_ids: Sequence[int] | None,
    baseline_wrong_business_id: int | None,
    baseline_wrong_visual_group_id: str | None,
) -> EvaluationScope:
    declarations = private_manifest.get("evaluation")
    if not isinstance(declarations, Mapping):
        raise BadgeGalleryUpdateEvaluationError(
            "private live manifest evaluation declaration is missing"
        )
    sibling_override = (
        None if sibling_business_ids is None else list(sibling_business_ids)
    )
    hard_negative_override = (
        None
        if hard_negative_business_ids is None
        else list(hard_negative_business_ids)
    )
    scope = EvaluationScope(
        target_business_id=_positive_id(
            _declared_value(
                declarations,
                "target_business_id",
                target_business_id,
            ),
            label="target business ID",
        ),
        target_visual_group_id=_visual_group(
            _declared_value(
                declarations,
                "target_visual_group_id",
                target_visual_group_id,
            ),
            label="target visual group",
        ),
        sibling_business_ids=_positive_ids(
            _declared_value(
                declarations,
                "sibling_business_ids",
                sibling_override,
            ),
            label="sibling business IDs",
            allow_empty=True,
        ),
        hard_negative_business_ids=_positive_ids(
            _declared_value(
                declarations,
                "hard_negative_business_ids",
                hard_negative_override,
            ),
            label="hard-negative business IDs",
        ),
        baseline_wrong_business_id=_positive_id(
            _declared_value(
                declarations,
                "baseline_wrong_business_id",
                baseline_wrong_business_id,
            ),
            label="baseline wrong business ID",
        ),
        baseline_wrong_visual_group_id=_visual_group(
            _declared_value(
                declarations,
                "baseline_wrong_visual_group_id",
                baseline_wrong_visual_group_id,
            ),
            label="baseline wrong visual group",
        ),
    )
    declared_sets = (
        {scope.target_business_id},
        set(scope.sibling_business_ids),
        set(scope.hard_negative_business_ids),
    )
    if any(
        declared_sets[left].intersection(declared_sets[right])
        for left in range(len(declared_sets))
        for right in range(left + 1, len(declared_sets))
    ):
        raise BadgeGalleryUpdateEvaluationError(
            "target, sibling, and hard-negative business-ID sets overlap"
        )
    if scope.baseline_wrong_business_id not in scope.hard_negative_business_ids:
        raise BadgeGalleryUpdateEvaluationError(
            "baseline wrong business ID is not a declared hard negative"
        )
    if scope.baseline_wrong_visual_group_id == scope.target_visual_group_id:
        raise BadgeGalleryUpdateEvaluationError(
            "baseline wrong and target visual groups must differ"
        )
    return scope


def _require_complete_hard_negative_variants(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    business_ids: Sequence[int],
) -> dict[str, int]:
    mapping = manifest.get("mapping")
    icons = manifest.get("icons")
    if not isinstance(mapping, Mapping) or not isinstance(icons, Mapping):
        raise BadgeGalleryUpdateEvaluationError(
            "authoritative hard-negative variant inventory is missing"
        )
    raw_crosswalk_path = mapping.get("crosswalk_path")
    if not isinstance(raw_crosswalk_path, str) or not raw_crosswalk_path:
        raise BadgeGalleryUpdateEvaluationError(
            "authoritative hard-negative crosswalk path is missing"
        )
    root = manifest_path.resolve().parent
    crosswalk_path = (root / raw_crosswalk_path).resolve()
    if not crosswalk_path.is_relative_to(root):
        raise BadgeGalleryUpdateEvaluationError(
            "authoritative hard-negative crosswalk escapes its dataset root"
        )
    try:
        crosswalk = json.loads(crosswalk_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BadgeGalleryUpdateEvaluationError(
            "authoritative hard-negative crosswalk is invalid"
        ) from error
    icon_records = icons.get("images")
    if not isinstance(crosswalk, list) or not isinstance(icon_records, list):
        raise BadgeGalleryUpdateEvaluationError(
            "authoritative hard-negative variant inventory is invalid"
        )
    requested = set(business_ids)
    expected: dict[int, tuple[str, ...]] = {}
    for row in crosswalk:
        if not isinstance(row, Mapping) or row.get("business_id") not in requested:
            continue
        business_id = int(row["business_id"])
        variants = row.get("asset_variants")
        if not isinstance(variants, list) or any(
            not isinstance(value, str) or not value for value in variants
        ):
            raise BadgeGalleryUpdateEvaluationError(
                "authoritative hard-negative variant declaration is invalid"
            )
        expected[business_id] = tuple(variants)

    observed: dict[int, list[str]] = defaultdict(list)
    for record in icon_records:
        if not isinstance(record, Mapping) or record.get("business_id") not in requested:
            continue
        business_id = int(record["business_id"])
        asset_variant = record.get("asset_variant")
        if not isinstance(asset_variant, str) or not asset_variant:
            raise BadgeGalleryUpdateEvaluationError(
                "authoritative hard-negative icon variant is invalid"
            )
        observed[business_id].append(asset_variant)

    counts: dict[str, int] = {}
    for business_id in sorted(requested):
        declared = expected.get(business_id)
        actual = observed.get(business_id, [])
        if declared is None or Counter(actual) != Counter(declared):
            raise BadgeGalleryUpdateEvaluationError(
                "authoritative dataset does not cover every declared asset variant "
                f"for hard-negative business ID {business_id}"
            )
        counts[str(business_id)] = len(actual)
    return counts


def _group_by_business_id(arrays: Mapping[str, np.ndarray]) -> dict[int, str]:
    result: dict[int, str] = {}
    for raw_business_id, raw_group in zip(
        arrays["business_ids"],
        arrays["visual_group_ids"],
        strict=True,
    ):
        business_id = int(raw_business_id)
        group = str(raw_group)
        previous = result.setdefault(business_id, group)
        if previous != group:
            raise BadgeGalleryUpdateEvaluationError(
                f"badge gallery maps business ID {business_id} to multiple groups"
            )
    return result


def _runtime_thresholds(component: Component) -> tuple[float, float, float, float]:
    runtime = component.manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise BadgeGalleryUpdateEvaluationError("candidate runtime metadata is missing")
    try:
        maximum_error = float(runtime["maximum_coarse_error"])
        maximum_zero_error = float(runtime["maximum_zero_coarse_error"])
        minimum_margin = float(runtime["minimum_group_margin"])
        minimum_high_error_margin = float(
            runtime["minimum_high_error_group_margin"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BadgeGalleryUpdateEvaluationError(
            "candidate runtime thresholds are invalid"
        ) from error
    if (
        maximum_error <= 0
        or maximum_zero_error < maximum_error
        or minimum_margin < 0
        or minimum_high_error_margin < 0
    ):
        raise BadgeGalleryUpdateEvaluationError(
            "candidate runtime thresholds are invalid"
        )
    return (
        maximum_error,
        maximum_zero_error,
        minimum_margin,
        minimum_high_error_margin,
    )


def _measure_query(component: Component, query: np.ndarray) -> Measurement:
    query_array = np.asarray(query)
    if query_array.shape != (8, 16, 3):
        raise BadgeGalleryUpdateEvaluationError("coarse identity query has invalid shape")
    errors = np.mean(
        np.abs(
            component.arrays["top_coarse"].astype(np.int16)
            - query_array.astype(np.int16)
        ),
        axis=(1, 2, 3),
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, raw_group in enumerate(component.arrays["visual_group_ids"]):
        grouped[str(raw_group)].append(index)
    ranked_groups = sorted(
        (
            (float(np.min(errors[indices])), group, indices)
            for group, indices in grouped.items()
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not ranked_groups:
        raise BadgeGalleryUpdateEvaluationError("badge gallery has no visual groups")
    top_error, top_group, top_indices = ranked_groups[0]
    top_index = int(top_indices[int(np.argmin(errors[top_indices]))])
    group_margin = (
        float("inf")
        if len(ranked_groups) == 1
        else float(ranked_groups[1][0] - top_error)
    )
    maximum_error, zero_error, minimum_margin, high_error_margin = (
        _runtime_thresholds(component)
    )
    low_confidence = bool(
        top_error > maximum_error
        and (len(ranked_groups) == 1 or group_margin < high_error_margin)
    )
    unambiguous = bool(
        top_error <= zero_error
        and not low_confidence
        and (len(ranked_groups) == 1 or group_margin >= minimum_margin)
    )
    return Measurement(
        business_id=int(component.arrays["business_ids"][top_index]),
        visual_group_id=top_group,
        top_error=top_error,
        group_margin=group_margin,
        low_confidence=low_confidence,
        unambiguous=unambiguous,
    )


def _coarse_from_image(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = box
    crop = image[y : y + height, x : x + width, :3]
    if width < 8 or height < 8 or crop.shape != (height, width, 3):
        raise BadgeGalleryUpdateEvaluationError(
            "private live geometry stress ROI exceeds its source image"
        )
    normalized = cv2.resize(
        np.ascontiguousarray(crop),
        (96, 96),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(
        normalized[:48],
        (16, 8),
        interpolation=cv2.INTER_AREA,
    ).astype(np.uint8)


def _load_live_inputs(
    private_manifest_path: Path,
    private_manifest: Mapping[str, Any],
) -> tuple[LiveInput, ...]:
    raw_samples = private_manifest.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise BadgeGalleryUpdateEvaluationError("private live samples are missing")
    live_inputs: list[LiveInput] = []
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, Mapping):
            raise BadgeGalleryUpdateEvaluationError(
                f"private live sample {index} is invalid"
            )
        try:
            image_path = badge_builder._resolve_inside(
                private_manifest_path.parent,
                raw_sample.get("image_path"),
                label=f"live sample {index} image",
            )
            badge_builder._verified_file(
                image_path,
                raw_sample.get("image_sha256"),
                label=f"live sample {index} image",
            )
        except badge_builder.BadgeGalleryBuildError as error:
            raise BadgeGalleryUpdateEvaluationError(str(error)) from error
        raw_roi = raw_sample.get("roi")
        if (
            not isinstance(raw_roi, list)
            or len(raw_roi) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_roi
            )
        ):
            raise BadgeGalleryUpdateEvaluationError(
                f"private live sample {index} ROI is invalid"
            )
        try:
            image = cv2.imdecode(
                np.frombuffer(image_path.read_bytes(), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        except OSError as error:
            raise BadgeGalleryUpdateEvaluationError(
                f"private live sample {index} is unavailable"
            ) from error
        if image is None:
            raise BadgeGalleryUpdateEvaluationError(
                f"private live sample {index} is undecodable"
            )
        box = tuple(raw_roi)
        _coarse_from_image(image, box)
        live_inputs.append(
            LiveInput(
                image=image,
                box=box,
                business_id=_positive_id(
                    raw_sample.get("expected_business_id"),
                    label=f"private live sample {index} business ID",
                ),
                visual_group_id=_visual_group(
                    raw_sample.get("expected_visual_group_id"),
                    label=f"private live sample {index} visual group",
                ),
            )
        )
    return tuple(live_inputs)


def _require_identity(
    measurement: Measurement,
    *,
    business_id: int,
    visual_group_id: str,
    label: str,
) -> None:
    if (
        measurement.business_id != business_id
        or measurement.visual_group_id != visual_group_id
        or measurement.low_confidence
        or not measurement.unambiguous
    ):
        raise BadgeGalleryUpdateEvaluationError(
            f"{label} did not resolve exactly to business ID {business_id} / "
            f"{visual_group_id}"
        )


def _rounded(value: float) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 6)


def _metric_summary(measurements: Sequence[Measurement]) -> dict[str, float | None]:
    if not measurements:
        raise BadgeGalleryUpdateEvaluationError("cannot summarize zero measurements")
    margins = [item.group_margin for item in measurements]
    errors = [item.top_error for item in measurements]
    return {
        "minimum_group_margin": _rounded(min(margins)),
        "minimum_top_error": _rounded(min(errors)),
        "maximum_top_error": _rounded(max(errors)),
    }


def _sparse_business_confusion(
    records: Sequence[tuple[int, Measurement]],
) -> list[dict[str, int]]:
    counts = Counter((expected, measured.business_id) for expected, measured in records)
    return [
        {"expected_business_id": expected, "predicted_business_id": predicted, "count": count}
        for (expected, predicted), count in sorted(counts.items())
    ]


def _sparse_group_confusion(
    records: Sequence[tuple[str, Measurement]],
) -> list[dict[str, Any]]:
    counts = Counter(
        (expected, measured.visual_group_id) for expected, measured in records
    )
    return [
        {
            "expected_visual_group_id": expected,
            "predicted_visual_group_id": predicted,
            "count": count,
        }
        for (expected, predicted), count in sorted(counts.items())
    ]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise BadgeGalleryUpdateEvaluationError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _promote_candidate(
    *,
    candidate: Component,
    report: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise BadgeGalleryUpdateEvaluationError(
            f"promoted output directory already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        gallery_name = candidate.gallery_path.name
        promoted_gallery = temporary / gallery_name
        shutil.copyfile(candidate.gallery_path, promoted_gallery)
        if sha256_file(promoted_gallery) != candidate.gallery_sha256:
            raise BadgeGalleryUpdateEvaluationError(
                "promoted gallery bytes differ from the evaluated candidate"
            )
        report_path = temporary / PROMOTED_REPORT_NAME
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        promoted_manifest = deepcopy(candidate.manifest)
        gallery = promoted_manifest.get("gallery")
        if not isinstance(gallery, dict):
            raise BadgeGalleryUpdateEvaluationError(
                "candidate gallery metadata is not mutable"
            )
        gallery["path"] = gallery_name
        gallery["sha256"] = candidate.gallery_sha256
        promoted_manifest["evaluation"] = {
            "path": PROMOTED_REPORT_NAME,
            "sha256": sha256_file(report_path),
            "status": "PASS",
            "tool_contract": TOOL_CONTRACT,
        }
        promoted_manifest["production_handoff"] = {
            "status": PROMOTED_HANDOFF_STATUS,
            "reason": (
                "private-live target, inherited rows, geometry stress, and authoritative "
                "hard negatives passed offline; independent live holdout remains PENDING"
            ),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(
                promoted_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return promoted_manifest


def evaluate_badge_gallery_update(
    *,
    base_component_root: Path,
    candidate_component_root: Path,
    live_reference_manifest: Path,
    card_dataset_manifest: Path,
    output_report: Path,
    promoted_output_dir: Path | None = None,
    target_business_id: int | None = None,
    target_visual_group_id: str | None = None,
    sibling_business_ids: Sequence[int] | None = None,
    hard_negative_business_ids: Sequence[int] | None = None,
    baseline_wrong_business_id: int | None = None,
    baseline_wrong_visual_group_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate all gates, then emit a public-safe report and optional promotion."""

    output_report = output_report.resolve()
    if output_report.exists():
        raise BadgeGalleryUpdateEvaluationError(
            f"output report already exists: {output_report}"
        )
    if promoted_output_dir is not None and promoted_output_dir.resolve().exists():
        raise BadgeGalleryUpdateEvaluationError(
            f"promoted output directory already exists: {promoted_output_dir.resolve()}"
        )
    base = _load_component(base_component_root, label="base")
    candidate = _load_component(candidate_component_root, label="candidate")
    try:
        badge_builder._require_evaluated_live_base(
            base.root,
            base.manifest,
            base.arrays,
        )
    except badge_builder.BadgeGalleryBuildError as error:
        raise BadgeGalleryUpdateEvaluationError(
            f"base badge is not an accepted production component: {error}"
        ) from error
    if base.manifest.get("runtime") != candidate.manifest.get("runtime"):
        raise BadgeGalleryUpdateEvaluationError(
            "candidate changed the frozen runtime thresholds or rule version"
        )

    live_reference_manifest = live_reference_manifest.resolve()
    private_manifest = _load_object(
        live_reference_manifest,
        label="private live reference manifest",
    )
    scope = _load_scope(
        private_manifest,
        target_business_id=target_business_id,
        target_visual_group_id=target_visual_group_id,
        sibling_business_ids=sibling_business_ids,
        hard_negative_business_ids=hard_negative_business_ids,
        baseline_wrong_business_id=baseline_wrong_business_id,
        baseline_wrong_visual_group_id=baseline_wrong_visual_group_id,
    )
    base_group_by_id = _group_by_business_id(base.arrays)
    candidate_group_by_id = _group_by_business_id(candidate.arrays)
    if base_group_by_id != candidate_group_by_id:
        raise BadgeGalleryUpdateEvaluationError(
            "candidate changed the badge business-ID or visual-group mapping"
        )
    if base_group_by_id.get(scope.target_business_id) != scope.target_visual_group_id:
        raise BadgeGalleryUpdateEvaluationError(
            "declared target identity disagrees with the badge gallery"
        )
    if (
        base_group_by_id.get(scope.baseline_wrong_business_id)
        != scope.baseline_wrong_visual_group_id
    ):
        raise BadgeGalleryUpdateEvaluationError(
            "declared baseline wrong identity disagrees with the badge gallery"
        )
    declared_ids = {
        scope.target_business_id,
        *scope.sibling_business_ids,
        *scope.hard_negative_business_ids,
    }
    if not declared_ids.issubset(base_group_by_id):
        raise BadgeGalleryUpdateEvaluationError(
            "one or more declared identities are absent from the badge gallery"
        )

    try:
        live_batch = badge_builder._live_references(
            live_reference_manifest,
            base=base.arrays,
            group_by_id=base_group_by_id,
        )
    except badge_builder.BadgeGalleryBuildError as error:
        raise BadgeGalleryUpdateEvaluationError(
            f"private live reference contract is invalid: {error}"
        ) from error
    if not live_batch.references:
        raise BadgeGalleryUpdateEvaluationError("private live batch has no references")
    if any(
        reference.business_id != scope.target_business_id
        or reference.visual_group_id != scope.target_visual_group_id
        for reference in live_batch.references
    ):
        raise BadgeGalleryUpdateEvaluationError(
            "private live batch contains an undeclared target identity"
        )
    live_inputs = _load_live_inputs(live_reference_manifest, private_manifest)
    if len(live_inputs) != len(live_batch.references):
        raise BadgeGalleryUpdateEvaluationError(
            "private live inputs and derived references disagree"
        )

    build = candidate.manifest.get("build")
    source = candidate.manifest.get("source")
    base_source = base.manifest.get("source")
    try:
        inherited_live_updates = (
            badge_builder._validated_live_reference_updates(base_source)
            if isinstance(base_source, Mapping)
            else ()
        )
        candidate_live_updates = (
            badge_builder._validated_live_reference_updates(source)
            if isinstance(source, Mapping)
            else ()
        )
    except badge_builder.BadgeGalleryBuildError as error:
        raise BadgeGalleryUpdateEvaluationError(
            f"candidate live-reference lineage is invalid: {error}"
        ) from error
    current_live_update = (
        candidate_live_updates[-1] if candidate_live_updates else None
    )
    if (
        not isinstance(build, Mapping)
        or build.get("tool_contract") != LIVE_APPEND_TOOL_CONTRACT
        or build.get("unaffected_base_rows") != "BITWISE_COPIED"
        or not isinstance(source, Mapping)
        or source.get("base_manifest_sha256") != base.manifest_sha256
        or source.get("base_reference_file_count") != len(base.arrays["business_ids"])
        or source.get("retained_reference_file_count") != len(base.arrays["business_ids"])
        or source.get("replaced_reference_file_count") != 0
        or source.get("added_reference_file_count") != len(live_batch.references)
        or candidate_live_updates[:-1] != inherited_live_updates
        or not isinstance(current_live_update, Mapping)
        or current_live_update.get("dataset_id") != live_batch.dataset_id
        or current_live_update.get("dataset_revision") != live_batch.dataset_revision
        or current_live_update.get("added_reference_count")
        != len(live_batch.references)
        or current_live_update.get("business_card_ids")
        != [scope.target_business_id]
        or current_live_update.get("visual_group_ids")
        != [scope.target_visual_group_id]
        or current_live_update.get("hard_negative_business_ids")
        != list(scope.hard_negative_business_ids)
    ):
        raise BadgeGalleryUpdateEvaluationError(
            "candidate manifest is not bound to the declared append-only live batch"
        )

    cumulative_hard_negative_ids = tuple(
        sorted(
            {
                business_id
                for update in candidate_live_updates
                for business_id in update["hard_negative_business_ids"]
            }
        )
    )
    if (
        scope.target_business_id in cumulative_hard_negative_ids
        or set(scope.sibling_business_ids).intersection(
            cumulative_hard_negative_ids
        )
    ):
        raise BadgeGalleryUpdateEvaluationError(
            "target or sibling identity overlaps cumulative hard-negative lineage"
        )
    scope = EvaluationScope(
        target_business_id=scope.target_business_id,
        target_visual_group_id=scope.target_visual_group_id,
        sibling_business_ids=scope.sibling_business_ids,
        hard_negative_business_ids=cumulative_hard_negative_ids,
        baseline_wrong_business_id=scope.baseline_wrong_business_id,
        baseline_wrong_visual_group_id=scope.baseline_wrong_visual_group_id,
    )
    declared_ids = {
        scope.target_business_id,
        *scope.sibling_business_ids,
        *scope.hard_negative_business_ids,
    }
    if not declared_ids.issubset(base_group_by_id):
        raise BadgeGalleryUpdateEvaluationError(
            "one or more cumulative hard-negative identities are absent from the badge gallery"
        )

    base_count = len(base.arrays["business_ids"])
    candidate_count = len(candidate.arrays["business_ids"])
    if candidate_count != base_count + len(live_batch.references):
        raise BadgeGalleryUpdateEvaluationError(
            "candidate did not append exactly the declared live prototypes"
        )
    for name in EXPECTED_ARRAY_NAMES:
        inherited = base.arrays[name]
        prefix = candidate.arrays[name][:base_count]
        if (
            prefix.dtype != inherited.dtype
            or prefix.shape != inherited.shape
            or prefix.tobytes() != inherited.tobytes()
        ):
            raise BadgeGalleryUpdateEvaluationError(
                f"candidate changed inherited {name} bytes"
            )
    appended_names = tuple(map(str, candidate.arrays["reference_names"][base_count:]))
    if (
        len(appended_names) != len(set(appended_names))
        or set(appended_names).intersection(map(str, base.arrays["reference_names"]))
    ):
        raise BadgeGalleryUpdateEvaluationError(
            "candidate appended duplicate or overlapping public reference names"
        )
    for offset, reference in enumerate(live_batch.references):
        index = base_count + offset
        if (
            int(candidate.arrays["business_ids"][index]) != reference.business_id
            or str(candidate.arrays["visual_group_ids"][index])
            != reference.visual_group_id
            or candidate.arrays["top_coarse"][index].tobytes()
            != reference.top_coarse.tobytes()
            or candidate.arrays["green_masks"][index].tobytes()
            != reference.green_mask.tobytes()
        ):
            raise BadgeGalleryUpdateEvaluationError(
                "candidate appended rows differ from the verified private derivation"
            )

    base_id_set = set(map(int, base.arrays["business_ids"]))
    candidate_id_set = set(map(int, candidate.arrays["business_ids"]))
    base_group_set = set(map(str, base.arrays["visual_group_ids"]))
    candidate_group_set = set(map(str, candidate.arrays["visual_group_ids"]))
    if base_id_set != candidate_id_set or base_group_set != candidate_group_set:
        raise BadgeGalleryUpdateEvaluationError(
            "candidate changed the business-ID or visual-group set"
        )

    exact_rows: dict[bytes, set[str]] = defaultdict(set)
    for raw_group, coarse in zip(
        candidate.arrays["visual_group_ids"],
        candidate.arrays["top_coarse"],
        strict=True,
    ):
        exact_rows[np.asarray(coarse).tobytes()].add(str(raw_group))
    cross_group_collisions = sum(
        1
        for coarse in candidate.arrays["top_coarse"][base_count:]
        if len(exact_rows[np.asarray(coarse).tobytes()]) > 1
    )
    if cross_group_collisions:
        raise BadgeGalleryUpdateEvaluationError(
            "a new live prototype has an exact cross-group collision"
        )

    historical_measurements: list[Measurement] = []
    historical_same_group_alias_count = 0
    confusion_business: list[tuple[int, Measurement]] = []
    confusion_groups: list[tuple[str, Measurement]] = []
    for business_id, group, coarse in zip(
        base.arrays["business_ids"],
        base.arrays["visual_group_ids"],
        base.arrays["top_coarse"],
        strict=True,
    ):
        baseline_measurement = _measure_query(base, coarse)
        measurement = _measure_query(candidate, coarse)
        expected_business_id = int(business_id)
        expected_group = str(group)
        if (
            baseline_measurement.visual_group_id != expected_group
            or baseline_measurement.low_confidence
            or not baseline_measurement.unambiguous
        ):
            raise BadgeGalleryUpdateEvaluationError(
                "base inherited row does not resolve unambiguously to its own visual group"
            )
        if baseline_measurement.business_id != expected_business_id:
            historical_same_group_alias_count += 1
        if (
            measurement.business_id != baseline_measurement.business_id
            or measurement.visual_group_id != baseline_measurement.visual_group_id
            or measurement.low_confidence != baseline_measurement.low_confidence
            or measurement.unambiguous != baseline_measurement.unambiguous
        ):
            raise BadgeGalleryUpdateEvaluationError(
                "candidate changed an inherited row's baseline identity result"
            )
        historical_measurements.append(measurement)
        if expected_business_id in declared_ids:
            confusion_business.append((expected_business_id, measurement))
            confusion_groups.append((expected_group, measurement))

    sibling_row_counts: dict[str, int] = {}
    for business_id in scope.sibling_business_ids:
        row_count = int(np.sum(base.arrays["business_ids"] == business_id))
        if row_count < 1:
            raise BadgeGalleryUpdateEvaluationError(
                f"sibling business ID {business_id} has no inherited reference row"
            )
        sibling_row_counts[str(business_id)] = row_count

    baseline_measurements: list[Measurement] = []
    target_measurements: list[Measurement] = []
    for live_input in live_inputs:
        query = _coarse_from_image(live_input.image, live_input.box)
        baseline_measurement = _measure_query(base, query)
        if (
            baseline_measurement.business_id != scope.baseline_wrong_business_id
            or baseline_measurement.visual_group_id
            != scope.baseline_wrong_visual_group_id
        ):
            raise BadgeGalleryUpdateEvaluationError(
                "baseline did not reproduce the declared wrong identity"
            )
        baseline_measurements.append(baseline_measurement)
        target_measurement = _measure_query(candidate, query)
        _require_identity(
            target_measurement,
            business_id=scope.target_business_id,
            visual_group_id=scope.target_visual_group_id,
            label="candidate live target",
        )
        target_measurements.append(target_measurement)
        confusion_business.append((scope.target_business_id, target_measurement))
        confusion_groups.append((scope.target_visual_group_id, target_measurement))

    geometry_measurements: list[Measurement] = []
    for live_input in live_inputs:
        x, y, width, height = live_input.box
        for delta_x in range(-3, 4):
            for delta_y in range(-1, 2):
                for size_delta in (0, 2):
                    if delta_x == 0 and delta_y == 0 and size_delta == 0:
                        continue
                    box = (
                        x + delta_x,
                        y + delta_y,
                        width + size_delta,
                        height + size_delta,
                    )
                    measurement = _measure_query(
                        candidate,
                        _coarse_from_image(live_input.image, box),
                    )
                    _require_identity(
                        measurement,
                        business_id=scope.target_business_id,
                        visual_group_id=scope.target_visual_group_id,
                        label="candidate geometry stress",
                    )
                    geometry_measurements.append(measurement)

    dataset_manifest_payload = _load_object(
        card_dataset_manifest.resolve(),
        label="authoritative card dataset manifest",
    )
    dataset_source_domain = dataset_manifest_payload.get(
        "source_domain",
        SOURCE_DOMAIN_OFFICIAL_RAW_ART,
    )
    if dataset_source_domain not in {
        SOURCE_DOMAIN_OFFICIAL_RAW_ART,
        SOURCE_DOMAIN_FIXED_RENDERED,
    }:
        raise BadgeGalleryUpdateEvaluationError(
            "authoritative card dataset source domain is unsupported"
        )
    try:
        dataset_plan = load_dataset_plan(
            card_dataset_manifest.resolve(),
            source_domain=str(dataset_source_domain),
            verify_source_files=True,
        )
    except Exception as error:
        raise BadgeGalleryUpdateEvaluationError(
            "authoritative card dataset is invalid"
        ) from error
    hard_negative_counts = _require_complete_hard_negative_variants(
        card_dataset_manifest.resolve(),
        dataset_manifest_payload,
        scope.hard_negative_business_ids,
    )
    samples_by_id: dict[int, list[Any]] = defaultdict(list)
    dataset_groups: dict[int, set[str]] = defaultdict(set)
    for sample in dataset_plan.samples:
        samples_by_id[sample.business_id].append(sample)
        dataset_groups[sample.business_id].add(sample.internal_id)
    for business_id in declared_ids:
        groups = dataset_groups.get(business_id, set())
        if groups != {base_group_by_id[business_id]}:
            raise BadgeGalleryUpdateEvaluationError(
                f"authoritative dataset identity mapping differs for business ID {business_id}"
            )

    hard_negative_measurements: list[Measurement] = []
    hard_negative_unique_files: set[tuple[int, bytes]] = set()
    for business_id in scope.hard_negative_business_ids:
        samples = samples_by_id.get(business_id, [])
        if not samples:
            raise BadgeGalleryUpdateEvaluationError(
                f"authoritative dataset has no hard-negative images for {business_id}"
            )
        expected_group = base_group_by_id[business_id]
        if hard_negative_counts[str(business_id)] != len(samples):
            raise BadgeGalleryUpdateEvaluationError(
                "authoritative hard-negative sample count disagrees with its variants"
            )
        for sample in samples:
            try:
                encoded = sample.marker_path.read_bytes()
            except OSError as error:
                raise BadgeGalleryUpdateEvaluationError(
                    "an authoritative hard-negative image is unavailable"
                ) from error
            image = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise BadgeGalleryUpdateEvaluationError(
                    "an authoritative hard-negative image is undecodable"
                )
            measurement = _measure_query(
                candidate,
                _coarse_from_image(
                    image,
                    (0, 0, int(image.shape[1]), int(image.shape[0])),
                ),
            )
            _require_identity(
                measurement,
                business_id=business_id,
                visual_group_id=expected_group,
                label="authoritative hard negative",
            )
            if measurement.business_id == scope.target_business_id:
                raise BadgeGalleryUpdateEvaluationError(
                    "an authoritative hard negative flipped to the target identity"
                )
            hard_negative_measurements.append(measurement)
            hard_negative_unique_files.add(
                (business_id, hashlib.sha256(encoded).digest())
            )
            confusion_business.append((business_id, measurement))
            confusion_groups.append((expected_group, measurement))

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "tool_contract": TOOL_CONTRACT,
        "components": {
            "base": {
                "manifest_sha256": base.manifest_sha256,
                "gallery_sha256": base.gallery_sha256,
                "reference_row_count": base_count,
            },
            "candidate": {
                "manifest_sha256": candidate.manifest_sha256,
                "gallery_sha256": candidate.gallery_sha256,
                "reference_row_count": candidate_count,
            },
        },
        "authoritative_dataset": {
            "dataset_id": dataset_plan.dataset_id,
            "dataset_revision": dataset_plan.dataset_revision,
            "manifest_sha256": dataset_plan.manifest_sha256,
            "source_domain": dataset_plan.source_domain,
            "hard_negative_input": "AUTHORITATIVE_FIXED_RENDERED_ICON",
        },
        "live_reference_batch": {
            "dataset_id": live_batch.dataset_id,
            "dataset_revision": live_batch.dataset_revision,
            "prototype_count": len(live_batch.references),
            "raw_source_publication": "EXCLUDED",
        },
        "scope": {
            "target_business_id": scope.target_business_id,
            "target_visual_group_id": scope.target_visual_group_id,
            "sibling_business_ids": list(scope.sibling_business_ids),
            "hard_negative_business_ids": list(scope.hard_negative_business_ids),
            "baseline_wrong_business_id": scope.baseline_wrong_business_id,
            "baseline_wrong_visual_group_id": scope.baseline_wrong_visual_group_id,
            "affected_visual_group_ids": sorted(
                {base_group_by_id[business_id] for business_id in declared_ids}
            ),
        },
        "gates": {
            "append_only_five_array_prefix": {
                "status": "PASS",
                "inherited_row_count": base_count,
                "appended_prototype_count": len(live_batch.references),
            },
            "identity_sets_unchanged": {
                "status": "PASS",
                "business_id_count": len(base_id_set),
                "visual_group_count": len(base_group_set),
            },
            "new_cross_group_exact_collision": {
                "status": "PASS",
                "collision_count": 0,
            },
            "inherited_full_replay": {
                "status": "PASS",
                "sample_count": len(historical_measurements),
                "preexisting_same_group_business_alias_count": (
                    historical_same_group_alias_count
                ),
                "candidate_changed_baseline_business_id_count": 0,
                "wrong_visual_group_count": 0,
                "metrics": _metric_summary(historical_measurements),
            },
            "sibling_reference_rows": {
                "status": "PASS",
                "row_counts": sibling_row_counts,
            },
            "baseline_wrong_identity_reproduced": {
                "status": "PASS",
                "sample_count": len(baseline_measurements),
                "metrics": _metric_summary(baseline_measurements),
            },
            "candidate_live_target": {
                "status": "PASS",
                "sample_count": len(target_measurements),
                "wrong_business_id_count": 0,
                "wrong_visual_group_count": 0,
                "low_confidence_count": 0,
                "metrics": _metric_summary(target_measurements),
            },
            "fixed_geometry_stress": {
                "status": "PASS",
                "contract": "dx=-3..3;dy=-1..1;size_delta=0,2;exact_excluded",
                "sample_count": len(geometry_measurements),
                "wrong_business_id_count": 0,
                "wrong_visual_group_count": 0,
                "low_confidence_count": 0,
                "metrics": _metric_summary(geometry_measurements),
            },
            "authoritative_hard_negative_blind_test": {
                "status": "PASS",
                "sample_count": len(hard_negative_measurements),
                "unique_file_count": len(hard_negative_unique_files),
                "per_business_id_sample_count": hard_negative_counts,
                "wrong_business_id_count": 0,
                "wrong_visual_group_count": 0,
                "target_flip_count": 0,
                "low_confidence_count": 0,
                "metrics": _metric_summary(hard_negative_measurements),
            },
        },
        "confusion": {
            "business_id_sparse": _sparse_business_confusion(confusion_business),
            "visual_group_sparse": _sparse_group_confusion(confusion_groups),
        },
        "independent_live_holdout": "PENDING",
    }
    _write_json_atomic(output_report, report)
    if promoted_output_dir is not None:
        _promote_candidate(
            candidate=candidate,
            report=report,
            output_dir=promoted_output_dir,
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-component-root", type=Path, required=True)
    parser.add_argument("--candidate-component-root", type=Path, required=True)
    parser.add_argument("--live-reference-manifest", type=Path, required=True)
    parser.add_argument("--card-dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--promoted-output-dir", type=Path)
    parser.add_argument("--target-business-id", type=int)
    parser.add_argument("--target-visual-group-id")
    parser.add_argument("--sibling-business-id", type=int, action="append")
    parser.add_argument("--hard-negative-business-id", type=int, action="append")
    parser.add_argument("--baseline-wrong-business-id", type=int)
    parser.add_argument("--baseline-wrong-visual-group-id")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = evaluate_badge_gallery_update(
        base_component_root=args.base_component_root,
        candidate_component_root=args.candidate_component_root,
        live_reference_manifest=args.live_reference_manifest,
        card_dataset_manifest=args.card_dataset_manifest,
        output_report=args.output_report,
        promoted_output_dir=args.promoted_output_dir,
        target_business_id=args.target_business_id,
        target_visual_group_id=args.target_visual_group_id,
        sibling_business_ids=args.sibling_business_id,
        hard_negative_business_ids=args.hard_negative_business_id,
        baseline_wrong_business_id=args.baseline_wrong_business_id,
        baseline_wrong_visual_group_id=args.baseline_wrong_visual_group_id,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
