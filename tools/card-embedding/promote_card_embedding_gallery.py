"""Promote one evaluated card-embedding gallery candidate deterministically.

The input component must still be in ``PENDING_VALIDATION`` state.  Promotion
binds the exact candidate manifest and all three artifacts to an offline
evaluation, verifies the append-only inheritance and authoritative raw-art
gates, and records fixed-rendered results as diagnostics only.
"""

from __future__ import annotations

import os
import sys
import json
import math
import uuid
import shutil
import hashlib
import argparse
from copy import deepcopy
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.card_selection.model import sha256_file

PENDING_STATUS = "PENDING_VALIDATION"
READY_STATUS = "READY"
READY_PENDING_REAL_STATUS = "READY_WITH_REAL_SAMPLES_PENDING"
REAL_SAMPLE_STATES = {"PASS", "PENDING"}
MINIMUM_SEPARATION_MARGIN = 0.01
REQUIRED_GALLERY_ARRAYS = (
    "embeddings",
    "class_names",
    "card_ids",
    "visual_group_ids",
    "upgrade_counts",
    "upgrade_markers",
)


class GalleryPromotionError(RuntimeError):
    """Raised when an evaluated gallery is not safe to promote."""


@dataclass(frozen=True)
class CandidateArtifacts:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    model_path: Path
    model_sha256: str
    gallery_path: Path
    gallery_sha256: str
    class_table_path: Path
    class_table_sha256: str
    class_names: tuple[str, ...]
    card_ids: tuple[int, ...]
    visual_group_ids: tuple[str, ...]
    gallery_array_count: int


@dataclass(frozen=True)
class PromotionResult:
    output_dir: Path
    manifest: dict[str, Any]
    status: str


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GalleryPromotionError(f"{label} is unavailable or invalid: {path}") from error
    if not isinstance(value, dict):
        raise GalleryPromotionError(f"{label} must be a JSON object")
    return value


def _load_hashed_object(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GalleryPromotionError(f"{label} is unavailable or invalid: {path}") from error
    if not isinstance(value, dict):
        raise GalleryPromotionError(f"{label} must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest().upper()


def _declared_sha256(value: object, *, label: str) -> str:
    text = str(value).upper()
    if len(text) != 64 or any(character not in "0123456789ABCDEF" for character in text):
        raise GalleryPromotionError(f"{label} has an invalid SHA-256 declaration")
    return text


def _resolve_inside(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise GalleryPromotionError(f"{label} path is missing")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise GalleryPromotionError(f"{label} path must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise GalleryPromotionError(f"{label} path escapes the component root")
    return resolved


def _verify_artifact(
    root: Path,
    metadata: dict[str, Any],
    *,
    path_key: str,
    hash_key: str,
    bytes_key: str | None,
    label: str,
) -> tuple[Path, str]:
    path = _resolve_inside(root, metadata.get(path_key), label=label)
    if not path.is_file():
        raise GalleryPromotionError(f"{label} is missing: {path}")
    expected = _declared_sha256(metadata.get(hash_key), label=label)
    actual = sha256_file(path)
    if actual != expected:
        raise GalleryPromotionError(f"{label} SHA-256 mismatch: {path}")
    if bytes_key is not None and metadata.get(bytes_key) is not None:
        declared_bytes = metadata[bytes_key]
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes < 1
            or path.stat().st_size != declared_bytes
        ):
            raise GalleryPromotionError(f"{label} byte count is inconsistent")
    if path.stat().st_size < 1:
        raise GalleryPromotionError(f"{label} is empty")
    return path, actual


def _positive_ids(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise GalleryPromotionError(f"{label} must be a non-empty list")
    values: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise GalleryPromotionError(f"{label} must contain positive integers")
        values.append(item)
    result = tuple(values)
    if tuple(sorted(set(result))) != result:
        raise GalleryPromotionError(f"{label} must be unique and sorted")
    return result


def _positive_count(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GalleryPromotionError(f"{label} must be a positive integer")
    return value


def _nonnegative_count(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GalleryPromotionError(f"{label} must be a non-negative integer")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GalleryPromotionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GalleryPromotionError(f"{label} must be finite")
    return result


def _load_candidate(component_root: Path) -> CandidateArtifacts:
    component_root = component_root.resolve()
    manifest_path = component_root / "manifest.json"
    manifest = _load_object(manifest_path, label="pending component manifest")
    if manifest.get("schema_version") != 1:
        raise GalleryPromotionError("unsupported pending component manifest schema")
    handoff = manifest.get("production_handoff")
    validation = manifest.get("validation")
    if not isinstance(handoff, dict) or handoff.get("status") != PENDING_STATUS:
        raise GalleryPromotionError("component production_handoff is not PENDING_VALIDATION")
    if not isinstance(validation, dict) or validation.get("state") != "PENDING":
        raise GalleryPromotionError("component validation state is not PENDING")
    source = manifest.get("source")
    build = manifest.get("build")
    if (
        not isinstance(source, dict)
        or source.get("source_domain") != "OFFICIAL_RAW_CARD_ART"
        or not isinstance(build, dict)
        or build.get("mode") != "GALLERY_ONLY"
        or build.get("input_mode") != "EXTENSION"
    ):
        raise GalleryPromotionError("component is not an official raw-art EXTENSION gallery candidate")

    model = manifest.get("model")
    gallery = manifest.get("gallery")
    if not isinstance(model, dict) or not isinstance(gallery, dict):
        raise GalleryPromotionError("component model/gallery metadata is missing")
    model_path, model_sha256 = _verify_artifact(
        component_root,
        model,
        path_key="path",
        hash_key="sha256",
        bytes_key="bytes",
        label="candidate model",
    )
    gallery_path, gallery_sha256 = _verify_artifact(
        component_root,
        gallery,
        path_key="path",
        hash_key="sha256",
        bytes_key="bytes",
        label="candidate gallery",
    )
    class_table_path, class_table_sha256 = _verify_artifact(
        component_root,
        gallery,
        path_key="class_table_path",
        hash_key="class_table_sha256",
        bytes_key="class_table_bytes",
        label="candidate class table",
    )
    try:
        raw_classes = json.loads(class_table_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GalleryPromotionError("candidate class table is invalid") from error
    if (
        not isinstance(raw_classes, list)
        or not raw_classes
        or not all(isinstance(value, str) and value for value in raw_classes)
        or len(raw_classes) != len(set(raw_classes))
    ):
        raise GalleryPromotionError("candidate class table must contain unique non-empty strings")
    class_names = tuple(raw_classes)

    try:
        with np.load(gallery_path, allow_pickle=False) as payload:
            missing = sorted(set(REQUIRED_GALLERY_ARRAYS) - set(payload.files))
            if missing:
                raise GalleryPromotionError(f"candidate gallery lacks arrays: {missing}")
            arrays = {name: np.asarray(payload[name]) for name in REQUIRED_GALLERY_ARRAYS}
            gallery_array_count = len(payload.files)
    except (OSError, ValueError) as error:
        raise GalleryPromotionError("candidate gallery cannot be loaded safely") from error
    row_count = len(class_names)
    if any(array.shape[0] != row_count for array in arrays.values()):
        raise GalleryPromotionError("candidate gallery arrays have inconsistent row counts")
    if arrays["embeddings"].ndim != 2 or arrays["embeddings"].shape[1] != 128:
        raise GalleryPromotionError("candidate gallery embedding shape is invalid")
    gallery_classes = tuple(str(value) for value in arrays["class_names"].tolist())
    if gallery_classes != class_names:
        raise GalleryPromotionError("candidate class table disagrees with gallery row order")
    try:
        card_ids = tuple(int(str(value)) for value in arrays["card_ids"].tolist())
    except (TypeError, ValueError) as error:
        raise GalleryPromotionError("candidate gallery business IDs are invalid") from error
    if any(value < 1 for value in card_ids):
        raise GalleryPromotionError("candidate gallery business IDs must be positive")
    visual_group_ids = tuple(str(value) for value in arrays["visual_group_ids"].tolist())
    if any(not value for value in visual_group_ids):
        raise GalleryPromotionError("candidate gallery visual-group IDs must be non-empty")

    declared_ids = _positive_ids(source.get("business_card_ids"), label="candidate business IDs")
    if set(card_ids) != set(declared_ids):
        raise GalleryPromotionError("candidate gallery business-ID coverage disagrees with manifest")
    declared_count = source.get("business_card_id_count")
    if declared_count != len(declared_ids):
        raise GalleryPromotionError("candidate manifest business-ID count is inconsistent")
    declared_classes = source.get("image_class_count")
    if declared_classes is not None and declared_classes != row_count:
        raise GalleryPromotionError("candidate manifest image-class count is inconsistent")

    return CandidateArtifacts(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest=manifest,
        model_path=model_path,
        model_sha256=model_sha256,
        gallery_path=gallery_path,
        gallery_sha256=gallery_sha256,
        class_table_path=class_table_path,
        class_table_sha256=class_table_sha256,
        class_names=class_names,
        card_ids=card_ids,
        visual_group_ids=visual_group_ids,
        gallery_array_count=gallery_array_count,
    )


def _validate_candidate_binding(evaluation: dict[str, Any], candidate: CandidateArtifacts) -> dict[str, str]:
    binding = evaluation.get("candidate")
    if not isinstance(binding, dict):
        raise GalleryPromotionError("evaluation candidate binding is missing")
    expected = {
        "manifest_sha256": candidate.manifest_sha256,
        "model_sha256": candidate.model_sha256,
        "gallery_sha256": candidate.gallery_sha256,
        "class_table_sha256": candidate.class_table_sha256,
    }
    normalized: dict[str, str] = {}
    for key, actual in expected.items():
        declared = _declared_sha256(binding.get(key), label=f"evaluation candidate {key}")
        if declared != actual:
            raise GalleryPromotionError(f"evaluation candidate {key} mismatch")
        normalized[key] = declared
    return normalized


def _validate_extension(evaluation: dict[str, Any], candidate: CandidateArtifacts) -> tuple[tuple[int, ...], int, int]:
    extension = evaluation.get("extension")
    if not isinstance(extension, dict):
        raise GalleryPromotionError("evaluation extension metadata is missing")
    business_ids = _positive_ids(extension.get("business_ids"), label="evaluation extension business IDs")
    source = candidate.manifest["source"]
    candidate_ids = _positive_ids(
        source.get("extension_business_card_ids"),
        label="candidate extension business IDs",
    )
    if business_ids != candidate_ids:
        raise GalleryPromotionError("evaluation extension IDs disagree with candidate manifest")
    extension_id_set = set(business_ids)
    row_indexes = [index for index, value in enumerate(candidate.card_ids) if value in extension_id_set]
    query_count = _positive_count(extension.get("query_count"), label="evaluation extension query count")
    if query_count != len(row_indexes):
        raise GalleryPromotionError("evaluation extension query count disagrees with gallery rows")
    visual_group_count = _positive_count(
        extension.get("visual_group_count"),
        label="evaluation extension visual-group count",
    )
    actual_groups = {candidate.visual_group_ids[index] for index in row_indexes}
    if visual_group_count != len(actual_groups):
        raise GalleryPromotionError("evaluation extension visual-group count disagrees with candidate gallery")
    return business_ids, query_count, visual_group_count


def _validate_inheritance(
    evaluation: dict[str, Any],
    candidate: CandidateArtifacts,
    extension_query_count: int,
) -> tuple[int, int]:
    inheritance = evaluation.get("inheritance")
    if not isinstance(inheritance, dict) or inheritance.get("status") != "BITWISE_EQUAL":
        raise GalleryPromotionError("old gallery prefix is not declared BITWISE_EQUAL")
    row_count = _positive_count(inheritance.get("row_count"), label="inheritance row count")
    expected_rows = len(candidate.class_names) - extension_query_count
    if row_count != expected_rows:
        raise GalleryPromotionError("inheritance row count disagrees with candidate gallery")
    build = candidate.manifest.get("build", {})
    if build.get("reused_embedding_row_count") != row_count:
        raise GalleryPromotionError("inheritance row count disagrees with candidate build metadata")
    if build.get("regenerated_embedding_row_count") != extension_query_count:
        raise GalleryPromotionError("extension query count disagrees with regenerated row count")
    array_count = _positive_count(inheritance.get("array_count"), label="inheritance array count")
    if array_count != candidate.gallery_array_count:
        raise GalleryPromotionError("inheritance array count disagrees with candidate gallery")
    return row_count, array_count


def _validate_raw_art(evaluation: dict[str, Any], *, visual_group_count: int) -> tuple[int, int, int]:
    raw_art = evaluation.get("raw_art")
    if not isinstance(raw_art, dict):
        raise GalleryPromotionError("evaluation raw-art gate is missing")
    query_count = _positive_count(raw_art.get("query_count"), label="raw-art query count")
    top1 = _nonnegative_count(raw_art.get("top1_correct"), label="raw-art Top-1 count")
    top5 = _nonnegative_count(raw_art.get("top5_correct"), label="raw-art Top-5 count")
    if query_count < visual_group_count:
        raise GalleryPromotionError("raw-art gate does not cover every new visual group")
    if top1 != query_count or top5 != query_count:
        raise GalleryPromotionError("authoritative raw-art recognition gate did not pass")
    return query_count, top1, top5


def _validate_fixed_diagnostic(evaluation: dict[str, Any]) -> dict[str, Any]:
    diagnostic = evaluation.get("fixed_rendered_diagnostic")
    if not isinstance(diagnostic, dict):
        raise GalleryPromotionError("fixed-rendered diagnostic is missing")
    for section in ("new", "historical_full", "historical_tail"):
        if not isinstance(diagnostic.get(section), dict):
            raise GalleryPromotionError(f"fixed-rendered diagnostic {section} is missing")
    new = diagnostic["new"]
    for key in (
        "query_count",
        "visual_top1_correct",
        "visual_top5_correct",
        "exact_id_correct",
        "upgrade_query_count",
        "upgrade_correct",
    ):
        _nonnegative_count(new.get(key), label=f"fixed-rendered new {key}")
    if (
        new["visual_top1_correct"] > new["query_count"]
        or new["visual_top5_correct"] > new["query_count"]
        or new["exact_id_correct"] > new["query_count"]
        or new["upgrade_correct"] > new["upgrade_query_count"]
    ):
        raise GalleryPromotionError("fixed-rendered diagnostic counts are internally inconsistent")
    return deepcopy(diagnostic)


def _validate_real_samples(
    evaluation: dict[str, Any],
    *,
    extension_business_ids: tuple[int, ...],
    candidate: CandidateArtifacts,
    report_path: Path | None,
) -> tuple[dict[str, Any], str]:
    real_samples = evaluation.get("real_dmm_samples")
    if not isinstance(real_samples, dict) or real_samples.get("status") not in REAL_SAMPLE_STATES:
        raise GalleryPromotionError("real DMM sample status must be explicitly PASS or PENDING")
    status = str(real_samples["status"])
    business_ids = _positive_ids(
        real_samples.get("business_ids"),
        label="real DMM sample business IDs",
    )
    if business_ids != extension_business_ids:
        raise GalleryPromotionError(
            "real DMM sample business IDs disagree with the evaluation extension"
        )

    normalized: dict[str, Any] = {
        "status": status,
        "business_ids": list(business_ids),
    }
    if status == "PASS":
        declared_report_sha256 = _declared_sha256(
            real_samples.get("report_sha256"),
            label="real DMM sample report",
        )
        declared_sample_count = _positive_count(
            real_samples.get("sample_count"),
            label="real DMM sample count",
        )
        if report_path is None:
            raise GalleryPromotionError(
                "real DMM sample PASS requires the actual report file"
            )
        report, actual_report_sha256 = _load_hashed_object(
            report_path.resolve(),
            label="real DMM sample report",
        )
        if declared_report_sha256 != actual_report_sha256:
            raise GalleryPromotionError("real DMM sample report SHA-256 mismatch")
        if (
            report.get("schema_version") != 2
            or report.get("model_sha256") != candidate.model_sha256
            or report.get("gallery_sha256") != candidate.gallery_sha256
        ):
            raise GalleryPromotionError(
                "real DMM sample report disagrees with the candidate artifacts"
            )
        wrong_id_count = report.get("wrong_id_count")
        if (
            isinstance(wrong_id_count, bool)
            or not isinstance(wrong_id_count, int)
            or wrong_id_count != 0
        ):
            raise GalleryPromotionError(
                "real DMM sample report did not pass its wrong-ID gate"
            )
        samples = report.get("samples")
        if not isinstance(samples, list) or not samples:
            raise GalleryPromotionError("real DMM sample report has no samples")
        report_sample_count = _positive_count(
            report.get("sample_count"),
            label="real DMM sample report count",
        )
        covered_ids: set[int] = set()
        for sample in samples:
            if (
                not isinstance(sample, dict)
                or not isinstance(sample.get("expected_card_id"), str)
                or not sample["expected_card_id"].isdigit()
                or sample.get("retrieval_correct") is not True
            ):
                raise GalleryPromotionError(
                    "real DMM sample report contains invalid or failing evidence"
                )
            covered_ids.add(int(sample["expected_card_id"]))
        if covered_ids != set(extension_business_ids):
            raise GalleryPromotionError(
                "real DMM sample report does not cover the exact extension business-ID set"
            )
        if (
            report_sample_count != len(samples)
            or declared_sample_count != report_sample_count
        ):
            raise GalleryPromotionError(
                "real DMM sample count disagrees with the actual report"
            )
        normalized["report_sha256"] = actual_report_sha256
        normalized["sample_count"] = len(samples)
    else:
        if report_path is not None:
            raise GalleryPromotionError(
                "pending real DMM samples must not supply a PASS report file"
            )
        forbidden = sorted({"report_sha256", "sample_count"}.intersection(real_samples))
        if forbidden:
            raise GalleryPromotionError(
                "pending real DMM samples must not declare PASS evidence fields: "
                + ", ".join(forbidden)
            )
    return normalized, status


def _validate_evaluation(
    evaluation: dict[str, Any],
    candidate: CandidateArtifacts,
    *,
    real_sample_report_path: Path | None,
) -> tuple[dict[str, Any], str]:
    if evaluation.get("schema_version") != 1:
        raise GalleryPromotionError("unsupported offline evaluation schema")
    binding = _validate_candidate_binding(evaluation, candidate)
    business_ids, query_count, visual_group_count = _validate_extension(evaluation, candidate)
    inherited_rows, array_count = _validate_inheritance(evaluation, candidate, query_count)
    raw_query_count, raw_top1, raw_top5 = _validate_raw_art(evaluation, visual_group_count=visual_group_count)
    separation = evaluation.get("new_visual_group_separation")
    if not isinstance(separation, dict):
        raise GalleryPromotionError("new visual-group separation gate is missing")
    minimum_margin = _finite_number(separation.get("minimum_margin"), label="new visual-group minimum margin")
    if minimum_margin < MINIMUM_SEPARATION_MARGIN:
        raise GalleryPromotionError("new visual-group separation margin is below the required minimum")
    real_samples, real_status = _validate_real_samples(
        evaluation,
        extension_business_ids=business_ids,
        candidate=candidate,
        report_path=real_sample_report_path,
    )
    diagnostic = _validate_fixed_diagnostic(evaluation)
    summary = {
        "candidate": binding,
        "extension": {
            "business_ids": list(business_ids),
            "query_count": query_count,
            "visual_group_count": visual_group_count,
        },
        "inheritance": {
            "status": "BITWISE_EQUAL",
            "row_count": inherited_rows,
            "array_count": array_count,
        },
        "raw_art": {
            "status": "PASS",
            "query_count": raw_query_count,
            "top1_correct": raw_top1,
            "top5_correct": raw_top5,
        },
        "new_visual_group_separation": {
            "status": "PASS",
            "minimum_margin": minimum_margin,
            "required_minimum_margin": MINIMUM_SEPARATION_MARGIN,
        },
        "real_dmm_samples": real_samples,
        "fixed_rendered_diagnostic": diagnostic,
    }
    return summary, real_status


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def promote_gallery(
    *,
    component_root: Path,
    evaluation_path: Path,
    output_dir: Path,
    real_sample_report_path: Path | None = None,
) -> PromotionResult:
    """Promote one exact PENDING candidate after all authoritative gates pass."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise GalleryPromotionError(f"output directory already exists: {output_dir}")
    candidate = _load_candidate(component_root)
    evaluation = _load_object(evaluation_path, label="offline gallery evaluation")
    evaluation_summary, real_status = _validate_evaluation(
        evaluation,
        candidate,
        real_sample_report_path=real_sample_report_path,
    )
    status = READY_STATUS if real_status == "PASS" else READY_PENDING_REAL_STATUS
    reason = (
        "authoritative offline gates and real DMM sample gate passed"
        if status == READY_STATUS
        else "authoritative offline gates passed; real DMM samples remain explicitly pending"
    )
    evaluation_sha256 = sha256_file(evaluation_path)

    promoted = deepcopy(candidate.manifest)
    promoted["validation"] = {
        "state": status,
        "note": reason,
        "offline_evaluation_sha256": evaluation_sha256,
        "gates": evaluation_summary,
    }
    promoted["production_handoff"] = {"status": status, "reason": reason}
    promoted["promotion"] = {
        "tool_contract": "task095-card-embedding-gallery-promotion-v1",
        "pending_manifest_sha256": candidate.manifest_sha256,
        "offline_evaluation_sha256": evaluation_sha256,
        "artifact_bindings": evaluation_summary["candidate"],
        "fixed_rendered_policy": "DIAGNOSTIC_ONLY",
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for source_path in (
            candidate.model_path,
            candidate.gallery_path,
            candidate.class_table_path,
        ):
            relative = source_path.relative_to(component_root.resolve())
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
        _write_json(temporary / "manifest.json", promoted)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PromotionResult(output_dir=output_dir, manifest=promoted, status=status)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote an offline-evaluated card-embedding gallery candidate")
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--real-sample-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = promote_gallery(
            component_root=args.component_root,
            evaluation_path=args.evaluation,
            output_dir=args.output_dir,
            real_sample_report_path=args.real_sample_report,
        )
    except GalleryPromotionError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "status": result.status,
                "manifest_sha256": sha256_file(result.output_dir / "manifest.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
