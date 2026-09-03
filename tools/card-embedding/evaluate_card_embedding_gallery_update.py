from __future__ import annotations

import sys
import json
import hashlib
import argparse
from typing import Any, Sequence
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.card_selection.embedding import (  # noqa: E402
    EmbeddingCardRecognizer,
    resolve_card_identity,
)

EXPECTED_GALLERY_ARRAYS = {
    "embeddings",
    "class_names",
    "card_ids",
    "visual_group_ids",
    "upgrade_counts",
    "upgrade_markers",
}
APRIL_FOOLS_IDS = frozenset({789, 790, 791, 792, 793, 794})
PLAN_BY_BUSINESS_ID = {
    789: "sense",
    790: "sense",
    791: "logic",
    792: "logic",
    793: "anomaly",
    794: "anomaly",
}


class GalleryUpdateEvaluationError(RuntimeError):
    """The frozen gallery-update evidence is missing or inconsistent."""


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
        raise GalleryUpdateEvaluationError(f"{label} is unavailable or invalid") from error
    if not isinstance(payload, dict):
        raise GalleryUpdateEvaluationError(f"{label} must be a JSON object")
    return payload


def _positive_ids(values: object, *, label: str) -> tuple[int, ...]:
    if (
        not isinstance(values, list)
        or not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        )
        or len(values) != len(set(values))
    ):
        raise GalleryUpdateEvaluationError(
            f"{label} must contain unique positive integer business IDs"
        )
    return tuple(sorted(values))


def _manifest_business_ids(manifest: dict[str, Any]) -> tuple[int, ...]:
    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        raise GalleryUpdateEvaluationError("dataset manifest has no business-ID scope")
    declarations: list[tuple[str, tuple[int, ...]]] = []
    for key in ("business_ids", "business_card_ids"):
        if key in scope:
            declarations.append((key, _positive_ids(scope[key], label=f"scope {key}")))
    for minimum_key, maximum_key in (
        ("business_id_min", "business_id_max"),
        ("business_card_id_min", "business_card_id_max"),
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
            raise GalleryUpdateEvaluationError(
                f"scope {minimum_key}/{maximum_key} is invalid"
            )
        declarations.append(
            (f"{minimum_key}/{maximum_key}", tuple(range(minimum, maximum + 1)))
        )
    if not declarations:
        raise GalleryUpdateEvaluationError("dataset manifest has no exact business-ID set")
    first_label, expected = declarations[0]
    for label, declared in declarations[1:]:
        if declared != expected:
            raise GalleryUpdateEvaluationError(
                f"dataset business-ID declarations disagree: {first_label} versus {label}"
            )
    return expected


def _natural_class_key(value: str) -> tuple[int, int, str]:
    prefix, separator, suffix = value.partition("_")
    if not prefix.isdigit():
        raise GalleryUpdateEvaluationError(f"invalid class name: {value}")
    return int(prefix), int(suffix) if separator and suffix.isdigit() else -1, value


@dataclass(frozen=True)
class IconSample:
    class_name: str
    business_id: int
    visual_group_id: str
    upgrade_count: int
    asset_variant: str
    path: Path


@dataclass(frozen=True)
class ArtSample:
    asset_variant: str
    visual_group_id: str
    path: Path


@dataclass(frozen=True)
class DatasetPlan:
    manifest_path: Path
    manifest_sha256: str
    business_ids: tuple[int, ...]
    icon_samples: tuple[IconSample, ...]
    art_samples: tuple[ArtSample, ...]


@dataclass(frozen=True)
class AcceptedDatasetPlan:
    manifest_paths: tuple[Path, ...]
    manifest_sha256s: tuple[str, ...]
    business_ids: tuple[int, ...]
    icon_samples: tuple[IconSample, ...]
    art_samples: tuple[ArtSample, ...]


def _inventory_root(manifest_path: Path, inventory: dict[str, Any]) -> Path:
    raw_root = inventory.get("root")
    if not isinstance(raw_root, str) or not raw_root:
        raise GalleryUpdateEvaluationError("dataset inventory root is missing")
    root = Path(raw_root)
    return root if root.is_absolute() else manifest_path.parent / root


def _verified_inventory_path(
    root: Path,
    record: dict[str, Any],
    *,
    label: str,
) -> Path:
    relative = record.get("path")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or Path(relative).name != relative
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise GalleryUpdateEvaluationError(f"{label} inventory record is invalid")
    path = root / relative
    if not path.is_file() or sha256_file(path) != expected_sha256.upper():
        raise GalleryUpdateEvaluationError(f"{label} file or SHA-256 differs: {relative}")
    return path


def load_dataset_plan(manifest_path: Path) -> DatasetPlan:
    manifest_path = manifest_path.resolve()
    manifest = _load_object(manifest_path, label="dataset manifest")
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "CLEAR":
        raise GalleryUpdateEvaluationError("dataset validation is not CLEAR")
    business_ids = _manifest_business_ids(manifest)

    mapping = manifest.get("mapping")
    if not isinstance(mapping, dict):
        raise GalleryUpdateEvaluationError("dataset crosswalk metadata is missing")
    raw_crosswalk_path = mapping.get("crosswalk_path")
    expected_crosswalk_sha256 = mapping.get("crosswalk_sha256")
    if not isinstance(raw_crosswalk_path, str) or not raw_crosswalk_path:
        raise GalleryUpdateEvaluationError("dataset crosswalk path is missing")
    crosswalk_path = Path(raw_crosswalk_path)
    if not crosswalk_path.is_absolute():
        crosswalk_path = manifest_path.parent / crosswalk_path
    if (
        not isinstance(expected_crosswalk_sha256, str)
        or sha256_file(crosswalk_path) != expected_crosswalk_sha256.upper()
    ):
        raise GalleryUpdateEvaluationError("dataset crosswalk SHA-256 differs")
    try:
        crosswalk = json.loads(crosswalk_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GalleryUpdateEvaluationError("dataset crosswalk is unavailable or invalid") from error
    if not isinstance(crosswalk, list) or not crosswalk:
        raise GalleryUpdateEvaluationError("dataset crosswalk must be a non-empty array")
    rows_by_id: dict[int, dict[str, Any]] = {}
    variant_groups: dict[str, set[str]] = {}
    for row in crosswalk:
        if not isinstance(row, dict):
            raise GalleryUpdateEvaluationError("dataset crosswalk row is invalid")
        business_id = row.get("business_id")
        internal_id = row.get("internal_id")
        upgrade_count = row.get("upgrade_count")
        variants = row.get("asset_variants")
        if (
            isinstance(business_id, bool)
            or not isinstance(business_id, int)
            or business_id < 1
            or business_id in rows_by_id
            or not isinstance(internal_id, str)
            or not internal_id
            or upgrade_count not in {0, 1}
            or not isinstance(variants, list)
            or not variants
            or not all(isinstance(value, str) and value for value in variants)
        ):
            raise GalleryUpdateEvaluationError("dataset crosswalk identity row is invalid")
        rows_by_id[business_id] = row
        group = "april-fools-asari" if business_id in APRIL_FOOLS_IDS else internal_id
        for variant in variants:
            variant_groups.setdefault(variant, set()).add(group)
    if tuple(sorted(rows_by_id)) != business_ids:
        raise GalleryUpdateEvaluationError("dataset crosswalk business-ID set differs")
    if any(len(groups) != 1 for groups in variant_groups.values()):
        raise GalleryUpdateEvaluationError("one raw-art variant maps to multiple visual groups")

    icons = manifest.get("icons")
    if not isinstance(icons, dict) or not isinstance(icons.get("images"), list):
        raise GalleryUpdateEvaluationError("dataset icon inventory is missing")
    icon_records = icons["images"]
    if icons.get("count") != len(icon_records):
        raise GalleryUpdateEvaluationError("dataset icon inventory count differs")
    icon_root = _inventory_root(manifest_path, icons)
    icon_samples: list[IconSample] = []
    for record in icon_records:
        if not isinstance(record, dict):
            raise GalleryUpdateEvaluationError("dataset icon record is invalid")
        business_id = record.get("business_id")
        class_name = record.get("class_name")
        asset_variant = record.get("asset_variant")
        if (
            isinstance(business_id, bool)
            or not isinstance(business_id, int)
            or business_id not in rows_by_id
            or not isinstance(class_name, str)
            or not class_name
            or class_name.split("_", 1)[0] != str(business_id)
            or not isinstance(asset_variant, str)
            or asset_variant not in rows_by_id[business_id]["asset_variants"]
        ):
            raise GalleryUpdateEvaluationError("dataset icon identity record is invalid")
        internal_id = str(rows_by_id[business_id]["internal_id"])
        group = "april-fools-asari" if business_id in APRIL_FOOLS_IDS else internal_id
        icon_samples.append(
            IconSample(
                class_name=class_name,
                business_id=business_id,
                visual_group_id=group,
                upgrade_count=int(rows_by_id[business_id]["upgrade_count"]),
                asset_variant=asset_variant,
                path=_verified_inventory_path(icon_root, record, label="icon"),
            )
        )
    icon_samples.sort(key=lambda sample: _natural_class_key(sample.class_name))
    if tuple(sorted({sample.business_id for sample in icon_samples})) != business_ids:
        raise GalleryUpdateEvaluationError("dataset icon business-ID coverage differs")
    if len({sample.class_name for sample in icon_samples}) != len(icon_samples):
        raise GalleryUpdateEvaluationError("dataset contains duplicate icon class names")

    art = manifest.get("art")
    if not isinstance(art, dict) or not isinstance(art.get("images"), list):
        raise GalleryUpdateEvaluationError("dataset raw-art inventory is missing")
    art_records = art["images"]
    if art.get("count") != len(art_records):
        raise GalleryUpdateEvaluationError("dataset raw-art inventory count differs")
    art_root = _inventory_root(manifest_path, art)
    art_samples: list[ArtSample] = []
    for record in art_records:
        if not isinstance(record, dict):
            raise GalleryUpdateEvaluationError("dataset raw-art record is invalid")
        variant = record.get("asset_name")
        if not isinstance(variant, str) or variant not in variant_groups:
            raise GalleryUpdateEvaluationError("dataset raw-art identity record is invalid")
        art_samples.append(
            ArtSample(
                asset_variant=variant,
                visual_group_id=next(iter(variant_groups[variant])),
                path=_verified_inventory_path(art_root, record, label="raw-art"),
            )
        )
    art_samples.sort(key=lambda sample: sample.asset_variant)
    if {sample.asset_variant for sample in art_samples} != set(variant_groups):
        raise GalleryUpdateEvaluationError("dataset raw-art variants do not cover the crosswalk")
    return DatasetPlan(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        business_ids=business_ids,
        icon_samples=tuple(icon_samples),
        art_samples=tuple(art_samples),
    )


def merge_accepted_dataset_plans(
    plans: Sequence[DatasetPlan],
    *,
    accepted_gallery_arrays: dict[str, np.ndarray],
) -> AcceptedDatasetPlan:
    if not plans:
        raise GalleryUpdateEvaluationError(
            "at least one accepted dataset manifest is required"
        )
    if set(accepted_gallery_arrays) != EXPECTED_GALLERY_ARRAYS:
        raise GalleryUpdateEvaluationError(
            "accepted component gallery array inventory differs"
        )

    row_counts = {
        name: int(np.asarray(values).shape[0])
        for name, values in accepted_gallery_arrays.items()
    }
    if len(set(row_counts.values())) != 1:
        raise GalleryUpdateEvaluationError(
            "accepted component gallery arrays have inconsistent row counts"
        )

    business_ids: set[int] = set()
    icon_samples: list[IconSample] = []
    art_samples_by_variant: dict[str, ArtSample] = {}
    for plan in plans:
        if len(plan.business_ids) != len(set(plan.business_ids)):
            raise GalleryUpdateEvaluationError(
                "accepted dataset manifest contains duplicate business IDs"
            )
        plan_business_ids = set(plan.business_ids)
        plan_sample_ids = {sample.business_id for sample in plan.icon_samples}
        if plan_sample_ids != plan_business_ids:
            raise GalleryUpdateEvaluationError(
                "accepted dataset manifest icon business-ID coverage differs"
            )
        plan_class_names = {sample.class_name for sample in plan.icon_samples}
        if len(plan_class_names) != len(plan.icon_samples):
            raise GalleryUpdateEvaluationError(
                "accepted dataset manifest contains duplicate icon class names"
            )
        plan_art_variants = {sample.asset_variant for sample in plan.art_samples}
        if len(plan_art_variants) != len(plan.art_samples):
            raise GalleryUpdateEvaluationError(
                "accepted dataset manifest contains duplicate raw-art identities"
            )
        plan_icon_variants = {sample.asset_variant for sample in plan.icon_samples}
        if plan_icon_variants != plan_art_variants:
            raise GalleryUpdateEvaluationError(
                "accepted dataset manifest icon/raw-art identity coverage differs"
            )

        icon_samples = [
            sample
            for sample in icon_samples
            if sample.business_id not in plan_business_ids
        ]
        icon_samples.extend(plan.icon_samples)
        icon_samples.sort(key=lambda sample: _natural_class_key(sample.class_name))
        business_ids.difference_update(plan_business_ids)
        business_ids.update(plan_business_ids)

        active_variants = {sample.asset_variant for sample in icon_samples}
        art_samples_by_variant = {
            variant: sample
            for variant, sample in art_samples_by_variant.items()
            if variant in active_variants
        }
        for sample in plan.art_samples:
            art_samples_by_variant[sample.asset_variant] = sample
        if set(art_samples_by_variant) != active_variants:
            raise GalleryUpdateEvaluationError(
                "accepted dataset replacement left incomplete raw-art identity coverage"
            )

    final_class_names = [sample.class_name for sample in icon_samples]
    if len(final_class_names) != len(set(final_class_names)):
        raise GalleryUpdateEvaluationError(
            "accepted dataset replacement produces duplicate icon class names"
        )
    groups_by_variant: dict[str, set[str]] = {}
    for sample in icon_samples:
        groups_by_variant.setdefault(sample.asset_variant, set()).add(
            sample.visual_group_id
        )
    if any(len(groups) != 1 for groups in groups_by_variant.values()):
        raise GalleryUpdateEvaluationError(
            "accepted dataset replacement maps one raw-art identity to multiple visual groups"
        )
    if any(
        sample.visual_group_id
        not in groups_by_variant.get(sample.asset_variant, set())
        for sample in art_samples_by_variant.values()
    ):
        raise GalleryUpdateEvaluationError(
            "accepted dataset replacement raw-art visual group differs"
        )

    expected_class_names = tuple(sample.class_name for sample in icon_samples)
    expected_card_ids = tuple(str(sample.business_id) for sample in icon_samples)
    expected_visual_groups = tuple(sample.visual_group_id for sample in icon_samples)
    expected_upgrade_counts = tuple(sample.upgrade_count for sample in icon_samples)
    actual_class_names = tuple(
        str(value) for value in accepted_gallery_arrays["class_names"]
    )
    actual_card_ids = tuple(str(value) for value in accepted_gallery_arrays["card_ids"])
    actual_visual_groups = tuple(
        str(value) for value in accepted_gallery_arrays["visual_group_ids"]
    )
    actual_upgrade_counts = tuple(
        int(value) for value in accepted_gallery_arrays["upgrade_counts"]
    )
    if actual_class_names != expected_class_names:
        raise GalleryUpdateEvaluationError(
            "accepted dataset manifest order/coverage differs from accepted component class names"
        )
    if actual_card_ids != expected_card_ids:
        raise GalleryUpdateEvaluationError(
            "accepted dataset manifest order/coverage differs from accepted component business IDs"
        )
    if actual_visual_groups != expected_visual_groups:
        raise GalleryUpdateEvaluationError(
            "accepted dataset visual groups differ from accepted component inherited rows"
        )
    if actual_upgrade_counts != expected_upgrade_counts:
        raise GalleryUpdateEvaluationError(
            "accepted dataset upgrade states differ from accepted component inherited rows"
        )

    actual_business_id_order: list[int] = []
    try:
        for raw_card_id in actual_card_ids:
            business_id = int(raw_card_id)
            if business_id not in actual_business_id_order:
                actual_business_id_order.append(business_id)
    except ValueError as error:
        raise GalleryUpdateEvaluationError(
            "accepted component contains a non-numeric business ID"
        ) from error
    merged_business_ids = tuple(sorted(business_ids))
    if tuple(actual_business_id_order) != merged_business_ids:
        raise GalleryUpdateEvaluationError(
            "accepted dataset business-ID order/coverage differs from accepted component"
        )

    return AcceptedDatasetPlan(
        manifest_paths=tuple(plan.manifest_path for plan in plans),
        manifest_sha256s=tuple(plan.manifest_sha256 for plan in plans),
        business_ids=merged_business_ids,
        icon_samples=tuple(icon_samples),
        art_samples=tuple(
            art_samples_by_variant[variant]
            for variant in sorted(art_samples_by_variant)
        ),
    )


@dataclass(frozen=True)
class IconOutcome:
    business_id: int
    expected_group: str
    top1_group: str
    top5_groups: tuple[str, ...]
    visual_top1_correct: bool
    visual_top5_correct: bool
    exact_id_correct: bool
    upgrade_applicable: bool
    upgrade_correct: bool


def summarize_icon_outcomes(outcomes: Sequence[IconOutcome]) -> dict[str, int]:
    if not outcomes:
        raise GalleryUpdateEvaluationError("icon evaluation contains no queries")
    upgrade_queries = [outcome for outcome in outcomes if outcome.upgrade_applicable]
    return {
        "query_count": len(outcomes),
        "visual_top1_correct": sum(outcome.visual_top1_correct for outcome in outcomes),
        "visual_top5_correct": sum(outcome.visual_top5_correct for outcome in outcomes),
        "exact_id_correct": sum(outcome.exact_id_correct for outcome in outcomes),
        "upgrade_query_count": len(upgrade_queries),
        "upgrade_correct": sum(outcome.upgrade_correct for outcome in upgrade_queries),
    }


def _read_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as source:
            return np.asarray(source.convert("RGB"))
    except (OSError, ValueError) as error:
        raise GalleryUpdateEvaluationError(f"image is unavailable or invalid: {path.name}") from error


def _evaluate_icons(
    recognizer: EmbeddingCardRecognizer,
    samples: Sequence[IconSample],
) -> tuple[tuple[IconOutcome, ...], tuple[dict[str, Any], ...]]:
    outcomes: list[IconOutcome] = []
    details: list[dict[str, Any]] = []
    gallery = recognizer.gallery
    pair_states_by_group: dict[str, set[int]] = {}
    for group, upgrade in zip(
        gallery.visual_group_ids,
        gallery.upgrade_counts,
        strict=True,
    ):
        pair_states_by_group.setdefault(group, set()).add(int(upgrade))
    for sample in samples:
        image = _read_rgb(sample.path)
        box = (0, 0, int(image.shape[1]), int(image.shape[0]))
        hits = gallery.search(recognizer.embedder.embed(image, box), top_k=5)
        top1_group = hits[0].visual_group_id
        top5_groups = tuple(hit.visual_group_id for hit in hits)
        upgrade_applicable = pair_states_by_group.get(sample.visual_group_id) == {0, 1}
        predicted_upgrade = (
            gallery.infer_upgrade_state(
                image,
                box,
                hits[0],
                source_color_order="RGB",
            )
            if pair_states_by_group.get(top1_group) == {0, 1}
            else None
        )
        try:
            resolved = resolve_card_identity(
                hits,
                plan=PLAN_BY_BUSINESS_ID.get(sample.business_id, "free"),
                upgraded=predicted_upgrade,
            ).card_id
        except ValueError:
            resolved = None
        outcome = IconOutcome(
            business_id=sample.business_id,
            expected_group=sample.visual_group_id,
            top1_group=top1_group,
            top5_groups=top5_groups,
            visual_top1_correct=top1_group == sample.visual_group_id,
            visual_top5_correct=sample.visual_group_id in top5_groups,
            exact_id_correct=resolved == str(sample.business_id),
            upgrade_applicable=upgrade_applicable,
            upgrade_correct=(
                predicted_upgrade == bool(sample.upgrade_count)
                if upgrade_applicable
                else True
            ),
        )
        outcomes.append(outcome)
        details.append(
            {
                **asdict(outcome),
                "top5_groups": list(top5_groups),
                "predicted_upgraded": predicted_upgrade,
                "resolved_business_id": int(resolved) if resolved is not None else None,
                "top1_similarity": hits[0].similarity,
                "top1_margin": hits[0].similarity - hits[1].similarity,
            }
        )
    return tuple(outcomes), tuple(details)


def _evaluate_raw_art(
    recognizer: EmbeddingCardRecognizer,
    samples: Sequence[ArtSample],
) -> tuple[dict[str, int], tuple[dict[str, Any], ...]]:
    top1_correct = 0
    top5_correct = 0
    details: list[dict[str, Any]] = []
    for sample in samples:
        image = _read_rgb(sample.path)
        box = (0, 0, int(image.shape[1]), int(image.shape[0]))
        hits = recognizer.gallery.search(
            recognizer.embedder.embed(image, box),
            top_k=5,
        )
        top5_groups = [hit.visual_group_id for hit in hits]
        top1_correct += hits[0].visual_group_id == sample.visual_group_id
        top5_correct += sample.visual_group_id in top5_groups
        details.append(
            {
                "asset_variant": sample.asset_variant,
                "expected_group": sample.visual_group_id,
                "top1_group": hits[0].visual_group_id,
                "top5_groups": top5_groups,
                "top1_similarity": hits[0].similarity,
                "top1_margin": hits[0].similarity - hits[1].similarity,
            }
        )
    return (
        {
            "query_count": len(samples),
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
        },
        tuple(details),
    )


def _load_gallery_arrays(root: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    gallery = manifest.get("gallery")
    if not isinstance(gallery, dict):
        raise GalleryUpdateEvaluationError("component gallery metadata is missing")
    path = root / str(gallery.get("path", ""))
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != EXPECTED_GALLERY_ARRAYS:
                raise GalleryUpdateEvaluationError("component gallery array inventory differs")
            return {name: np.asarray(arrays[name]) for name in arrays.files}
    except GalleryUpdateEvaluationError:
        raise
    except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
        raise GalleryUpdateEvaluationError("component gallery is unavailable or invalid") from error


def _verify_bitwise_inheritance(
    candidate_root: Path,
    candidate_manifest: dict[str, Any],
    accepted_root: Path,
    accepted_manifest: dict[str, Any],
    *,
    extension_business_ids: tuple[int, ...],
) -> dict[str, Any]:
    candidate_model = candidate_manifest.get("model")
    accepted_model = accepted_manifest.get("model")
    if (
        not isinstance(candidate_model, dict)
        or not isinstance(accepted_model, dict)
        or candidate_model.get("sha256") != accepted_model.get("sha256")
    ):
        raise GalleryUpdateEvaluationError("gallery-only candidate changed the encoder model")
    candidate = _load_gallery_arrays(candidate_root, candidate_manifest)
    accepted = _load_gallery_arrays(accepted_root, accepted_manifest)
    try:
        candidate_ids = np.asarray(
            [int(value) for value in candidate["card_ids"]],
            dtype=np.int64,
        )
        accepted_ids = np.asarray(
            [int(value) for value in accepted["card_ids"]],
            dtype=np.int64,
        )
    except (TypeError, ValueError) as error:
        raise GalleryUpdateEvaluationError(
            "component gallery contains a non-numeric business ID"
        ) from error
    extension_ids = np.asarray(extension_business_ids, dtype=np.int64)
    candidate_retained = ~np.isin(candidate_ids, extension_ids)
    accepted_retained = ~np.isin(accepted_ids, extension_ids)
    if not np.any(~candidate_retained):
        raise GalleryUpdateEvaluationError(
            "candidate gallery contains no current extension rows"
        )
    inherited_rows = int(np.count_nonzero(accepted_retained))
    for name in sorted(EXPECTED_GALLERY_ARRAYS):
        old = accepted[name][accepted_retained]
        new_retained = candidate[name][candidate_retained]
        if (
            old.dtype != new_retained.dtype
            or old.shape != new_retained.shape
            or old.tobytes() != new_retained.tobytes()
        ):
            raise GalleryUpdateEvaluationError(
                f"candidate gallery does not bitwise inherit array: {name}"
            )
    return {
        "status": "BITWISE_EQUAL",
        "row_count": inherited_rows,
        "array_count": len(EXPECTED_GALLERY_ARRAYS),
        "arrays": sorted(EXPECTED_GALLERY_ARRAYS),
    }


def _new_group_separation(
    recognizer: EmbeddingCardRecognizer,
    extension_groups: set[str],
) -> dict[str, Any]:
    gallery = recognizer.gallery
    groups = np.asarray(gallery.visual_group_ids)
    embeddings = np.asarray(gallery.embeddings, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for group in sorted(extension_groups):
        own_indices = np.flatnonzero(groups == group)
        other_indices = np.flatnonzero(groups != group)
        if own_indices.size == 0 or other_indices.size == 0:
            raise GalleryUpdateEvaluationError("new visual group is absent or not comparable")
        margins = []
        nearest_groups = []
        nearest_scores = []
        for own_index in own_indices:
            scores = embeddings[other_indices] @ embeddings[own_index]
            nearest_position = int(np.argmax(scores))
            nearest_index = int(other_indices[nearest_position])
            nearest_score = float(scores[nearest_position])
            margins.append(1.0 - nearest_score)
            nearest_groups.append(str(groups[nearest_index]))
            nearest_scores.append(nearest_score)
        worst = int(np.argmin(margins))
        rows.append(
            {
                "visual_group_id": group,
                "nearest_other_group": nearest_groups[worst],
                "nearest_other_similarity": nearest_scores[worst],
                "margin": margins[worst],
            }
        )
    return {
        "group_count": len(rows),
        "minimum_margin": min(row["margin"] for row in rows),
        "groups": rows,
    }


def _real_sample_evidence(
    report_path: Path | None,
    *,
    expected_ids: tuple[int, ...],
    model_sha256: str,
    gallery_sha256: str,
) -> dict[str, Any]:
    if report_path is None:
        return {"status": "PENDING", "business_ids": list(expected_ids)}
    report = _load_object(report_path.resolve(), label="real DMM evaluation report")
    if (
        report.get("schema_version") != 2
        or report.get("model_sha256") != model_sha256
        or report.get("gallery_sha256") != gallery_sha256
        or report.get("wrong_id_count") != 0
    ):
        raise GalleryUpdateEvaluationError("real DMM evaluation report failed its identity gate")
    samples = report.get("samples")
    if not isinstance(samples, list):
        raise GalleryUpdateEvaluationError("real DMM evaluation report has no samples")
    covered_ids = {
        int(sample["expected_card_id"])
        for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("expected_card_id"), str)
        and sample["expected_card_id"].isdigit()
        and sample.get("retrieval_correct") is True
    }
    if covered_ids != set(expected_ids):
        raise GalleryUpdateEvaluationError(
            "real DMM evaluation does not cover the exact extension business-ID set"
        )
    return {
        "status": "PASS",
        "business_ids": list(expected_ids),
        "report_sha256": sha256_file(report_path.resolve()),
        "sample_count": len(samples),
    }


def evaluate_gallery_update(
    *,
    candidate_root: Path,
    accepted_component_root: Path,
    extension_manifest_path: Path,
    real_sample_report_path: Path | None,
    accepted_dataset_manifest_path: Path | None = None,
    accepted_dataset_manifest_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    candidate_root = candidate_root.resolve()
    accepted_component_root = accepted_component_root.resolve()
    candidate_manifest_path = candidate_root / "manifest.json"
    accepted_manifest_path = accepted_component_root / "manifest.json"
    candidate_manifest = _load_object(candidate_manifest_path, label="candidate manifest")
    accepted_manifest = _load_object(accepted_manifest_path, label="accepted manifest")
    if candidate_manifest.get("schema_version") != 1:
        raise GalleryUpdateEvaluationError("unsupported candidate manifest schema")
    handoff = candidate_manifest.get("production_handoff")
    validation = candidate_manifest.get("validation")
    if (
        not isinstance(handoff, dict)
        or handoff.get("status") != "PENDING_VALIDATION"
        or not isinstance(validation, dict)
        or validation.get("state") != "PENDING"
    ):
        raise GalleryUpdateEvaluationError("candidate is not a pending gallery-only build")

    extension = load_dataset_plan(extension_manifest_path)
    if accepted_dataset_manifest_paths is not None:
        if accepted_dataset_manifest_path is not None:
            raise GalleryUpdateEvaluationError(
                "accepted dataset manifest inputs must use one calling convention"
            )
        accepted_manifest_paths = tuple(accepted_dataset_manifest_paths)
    elif accepted_dataset_manifest_path is not None:
        accepted_manifest_paths = (accepted_dataset_manifest_path,)
    else:
        raise GalleryUpdateEvaluationError(
            "at least one accepted dataset manifest is required"
        )
    accepted_dataset_plans = tuple(
        load_dataset_plan(path) for path in accepted_manifest_paths
    )
    source = candidate_manifest.get("source")
    if (
        not isinstance(source, dict)
        or source.get("dataset_manifest_sha256") != extension.manifest_sha256
        or tuple(source.get("extension_business_card_ids", ())) != extension.business_ids
    ):
        raise GalleryUpdateEvaluationError("candidate source does not bind the extension dataset")
    recognizer = EmbeddingCardRecognizer.load(candidate_root, source_color_order="RGB")
    inheritance = _verify_bitwise_inheritance(
        candidate_root,
        candidate_manifest,
        accepted_component_root,
        accepted_manifest,
        extension_business_ids=extension.business_ids,
    )
    expected_inherited_rows = candidate_manifest.get("build", {}).get(
        "reused_embedding_row_count"
    )
    if expected_inherited_rows != inheritance["row_count"]:
        raise GalleryUpdateEvaluationError("candidate inherited-row count differs")
    accepted_dataset = merge_accepted_dataset_plans(
        accepted_dataset_plans,
        accepted_gallery_arrays=_load_gallery_arrays(
            accepted_component_root,
            accepted_manifest,
        ),
    )

    raw_summary, raw_details = _evaluate_raw_art(recognizer, extension.art_samples)
    new_outcomes, new_details = _evaluate_icons(recognizer, extension.icon_samples)
    all_old_outcomes, _ = _evaluate_icons(recognizer, accepted_dataset.icon_samples)
    tail_ids = tuple(accepted_dataset.business_ids[-len(extension.business_ids) :])
    tail_samples = tuple(
        sample for sample in accepted_dataset.icon_samples if sample.business_id in tail_ids
    )
    tail_outcomes, _ = _evaluate_icons(recognizer, tail_samples)
    extension_groups = {sample.visual_group_id for sample in extension.icon_samples}
    candidate_gallery = candidate_manifest.get("gallery")
    candidate_model = candidate_manifest.get("model")
    if not isinstance(candidate_gallery, dict) or not isinstance(candidate_model, dict):
        raise GalleryUpdateEvaluationError("candidate asset metadata is missing")
    class_path = candidate_root / str(candidate_gallery.get("class_table_path", ""))
    return {
        "schema_version": 1,
        "candidate": {
            "component": candidate_manifest.get("component"),
            "manifest_sha256": sha256_file(candidate_manifest_path),
            "model_sha256": recognizer.model_sha256,
            "gallery_sha256": recognizer.gallery_sha256,
            "class_table_sha256": sha256_file(class_path),
        },
        "extension": {
            "dataset_id": source.get("dataset_id"),
            "dataset_manifest_sha256": extension.manifest_sha256,
            "business_ids": list(extension.business_ids),
            "query_count": len(extension.icon_samples),
            "visual_group_count": len(extension_groups),
        },
        "inheritance": inheritance,
        "raw_art": {**raw_summary, "queries": list(raw_details)},
        "fixed_rendered_diagnostic": {
            "acceptance_role": "DIAGNOSTIC_ONLY",
            "note": (
                "card-art acceptance uses official raw-art augmentation and frozen "
                "DMM ROI evidence; fixed renders remain diagnostic and upgrade-marker sources"
            ),
            "new": {
                **summarize_icon_outcomes(new_outcomes),
                "queries": list(new_details),
            },
            "historical_full": summarize_icon_outcomes(all_old_outcomes),
            "historical_tail": {
                "business_ids": list(tail_ids),
                **summarize_icon_outcomes(tail_outcomes),
            },
        },
        "new_visual_group_separation": _new_group_separation(
            recognizer,
            extension_groups,
        ),
        "real_dmm_samples": _real_sample_evidence(
            real_sample_report_path,
            expected_ids=extension.business_ids,
            model_sha256=recognizer.model_sha256,
            gallery_sha256=recognizer.gallery_sha256,
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one deterministic skill-card gallery extension"
    )
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--accepted-component-root", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument(
        "--accepted-dataset-manifest",
        type=Path,
        action="append",
        required=True,
        help="repeat in chronological order for every accepted dataset batch",
    )
    parser.add_argument("--real-sample-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = evaluate_gallery_update(
        candidate_root=args.candidate_root,
        accepted_component_root=args.accepted_component_root,
        extension_manifest_path=args.extension_manifest,
        accepted_dataset_manifest_paths=args.accepted_dataset_manifest,
        real_sample_report_path=args.real_sample_report,
    )
    _write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "raw_art": report["raw_art"],
                "new_fixed_rendered": report["fixed_rendered_diagnostic"]["new"],
                "real_dmm_samples": report["real_dmm_samples"]["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
