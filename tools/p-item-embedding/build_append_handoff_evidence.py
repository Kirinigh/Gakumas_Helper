from __future__ import annotations

import io
import re
import sys
import json
import hashlib
import argparse
from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

try:
    from tools.p_item_derived_source import (
        SOURCE_KEY,
        EVIDENCE_TOOL_CONTRACT,
        SOURCE_REVISION_SUFFIX,
        DerivedSourceError,
        build_derived_source,
    )
    from tools.published_ui_reference import (
        PublishedUiReferenceError,
        validate_published_ui_source_manifest,
    )
    from agent.p_item_recognition.reference import (
        PItemReferenceRuntime,
        PItemRenderedReferenceGallery,
    )
    from tools.deployment.promote_static_reference_handoff import (
        P_ITEM_RUNTIME_CONTRACT,
        P_ITEM_PRODUCTION_SOURCE_EVIDENCE_V2_TOOL_CONTRACT,
        P_ITEM_PRODUCTION_SOURCE_EVIDENCE_V3_TOOL_CONTRACT,
    )
except ModuleNotFoundError:  # Direct script execution from tools/p-item-embedding.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.p_item_derived_source import (
        SOURCE_KEY,
        EVIDENCE_TOOL_CONTRACT,
        SOURCE_REVISION_SUFFIX,
        DerivedSourceError,
        build_derived_source,
    )
    from tools.published_ui_reference import (  # type: ignore[no-redef]
        PublishedUiReferenceError,
        validate_published_ui_source_manifest,
    )
    from agent.p_item_recognition.reference import (  # type: ignore[no-redef]
        PItemReferenceRuntime,
        PItemRenderedReferenceGallery,
    )
    from tools.deployment.promote_static_reference_handoff import (  # type: ignore[no-redef]
        P_ITEM_RUNTIME_CONTRACT,
        P_ITEM_PRODUCTION_SOURCE_EVIDENCE_V2_TOOL_CONTRACT,
        P_ITEM_PRODUCTION_SOURCE_EVIDENCE_V3_TOOL_CONTRACT,
    )


SHA1_PATTERN = re.compile(r"[0-9A-Fa-f]{40}")
TARGET_REFERENCE_SIZE = (130, 130)
STAGE_SOURCE_TYPES = frozenset({"pIdol", "support"})
STAGE_SCOPE_CONTRACT = "EntityBank selectable entities use modes=['stage']; coverage/generate-suite filters p.mode === 'stage'"


class PItemAppendEvidenceError(RuntimeError):
    """Raised when append handoff inputs do not form an exact frozen lineage."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Build deterministic P-item append production/source evidence and candidate provenance"))
    parser.add_argument("--base-gallery", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--gakumas-tools-revision", required=True)
    parser.add_argument("--production-deployment-id", type=int, required=True)
    parser.add_argument("--production-deployment-revision", required=True)
    parser.add_argument("--production-deployment-status", required=True)
    parser.add_argument("--gk-img-revision")
    parser.add_argument("--first-added-revision")
    parser.add_argument("--first-successful-production-revision")
    parser.add_argument("--published-ui-manifest", type=Path)
    parser.add_argument("--published-ui-original-root", type=Path)
    parser.add_argument("--derived-candidate-root", type=Path)
    parser.add_argument("--derived-overlay", type=Path)
    parser.add_argument("--derived-qualification-report", type=Path)
    parser.add_argument(
        "--added-business-id",
        type=int,
        action="append",
        default=[],
        help="New arena-stage business ID; repeat in ascending order",
    )
    parser.add_argument(
        "--replaced-business-id",
        type=int,
        action="append",
        default=[],
        help="Existing business ID with changed pixels; repeat in ascending order",
    )
    return parser.parse_args()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def _validated_revision(value: str | None, *, label: str) -> str:
    if not isinstance(value, str) or SHA1_PATTERN.fullmatch(value) is None:
        raise PItemAppendEvidenceError(f"{label} must be a full 40-character Git revision")
    return value.casefold()


def _ordered_business_ids(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values) or list(values) != sorted(set(values)):
        raise PItemAppendEvidenceError(f"{label} must be unique positive integers in ascending order")
    return tuple(values)


def _decode_png(value: bytes, *, label: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 8 or image.shape[1] < 8:
        raise PItemAppendEvidenceError(f"{label} is not a decodable RGB PNG")
    return np.ascontiguousarray(image)


def _sanitized_png_bytes(path: Path, expected_pixels: np.ndarray) -> bytes:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            buffer = io.BytesIO()
            rgb.save(buffer, format="PNG", optimize=False, compress_level=9)
    except (OSError, ValueError) as error:
        raise PItemAppendEvidenceError(f"source reference {path.name} cannot be sanitized") from error
    encoded = buffer.getvalue()
    decoded = _decode_png(encoded, label=f"sanitized source reference {path.name}")
    if not np.array_equal(expected_pixels, decoded):
        raise PItemAppendEvidenceError(f"source reference {path.name} changed pixels during metadata removal")
    with Image.open(io.BytesIO(encoded)) as image:
        if image.info or image.getexif():
            raise PItemAppendEvidenceError(f"source reference {path.name} retains PNG metadata")
    return encoded


def _read_base_gallery(
    path: Path,
) -> tuple[tuple[int, ...], dict[int, np.ndarray], dict[int, bytes]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise PItemAppendEvidenceError("base gallery is unavailable")
    try:
        with np.load(resolved, allow_pickle=False) as arrays:
            if set(arrays.files) != {"p_item_ids", "png_bytes", "png_offsets"}:
                raise PItemAppendEvidenceError("base gallery inventory is invalid")
            ids = np.asarray(arrays["p_item_ids"])
            payload = np.asarray(arrays["png_bytes"])
            offsets = np.asarray(arrays["png_offsets"])
    except PItemAppendEvidenceError:
        raise
    except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
        raise PItemAppendEvidenceError("base gallery is invalid") from error
    if (
        ids.dtype != np.dtype(np.int32)
        or ids.ndim != 1
        or len(ids) < 1
        or payload.dtype != np.dtype(np.uint8)
        or payload.ndim != 1
        or offsets.dtype != np.dtype(np.int64)
        or offsets.shape != (len(ids) + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(payload)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise PItemAppendEvidenceError("base gallery arrays are invalid")
    business_ids = tuple(map(int, ids))
    if business_ids != tuple(sorted(business_ids)) or len(business_ids) != len(set(business_ids)) or any(value < 1 for value in business_ids):
        raise PItemAppendEvidenceError("base gallery IDs are invalid")
    encoded = {business_id: bytes(payload[int(offsets[index]) : int(offsets[index + 1])]) for index, business_id in enumerate(business_ids)}
    images = {
        business_id: _decode_png(
            encoded[business_id],
            label=f"base reference {business_id}",
        )
        for business_id in business_ids
    }
    return business_ids, images, encoded


def _read_source(
    directory: Path,
) -> tuple[tuple[Path, ...], tuple[int, ...], dict[int, np.ndarray], dict[int, bytes]]:
    resolved = directory.resolve()
    if not resolved.is_dir():
        raise PItemAppendEvidenceError("complete rendered-reference source is unavailable")
    files = tuple(path for path in resolved.iterdir() if path.is_file())
    paths = tuple(
        sorted(
            (path for path in files if path.suffix.casefold() == ".png" and path.stem.isdigit()),
            key=lambda path: int(path.stem),
        )
    )
    if not paths or len(paths) != len(files):
        raise PItemAppendEvidenceError("source must contain only numeric rendered-reference PNG files")
    business_ids = tuple(int(path.stem) for path in paths)
    if len(business_ids) != len(set(business_ids)) or any(
        path.name != f"{business_id}.png" for path, business_id in zip(paths, business_ids, strict=True)
    ):
        raise PItemAppendEvidenceError("source reference filenames must be unique canonical business IDs")
    images: dict[int, np.ndarray] = {}
    sanitized: dict[int, bytes] = {}
    for path, business_id in zip(paths, business_ids, strict=True):
        pixels = _decode_png(path.read_bytes(), label=f"source reference {path.name}")
        images[business_id] = pixels
        sanitized[business_id] = _sanitized_png_bytes(path, pixels)
    return paths, business_ids, images, sanitized


def _source_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("ascii"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest().upper()


def _sanitized_digest(business_ids: Sequence[int], sanitized: Mapping[int, bytes]) -> str:
    digest = hashlib.sha256()
    for business_id in business_ids:
        digest.update(str(business_id).encode("ascii"))
        digest.update(hashlib.sha256(sanitized[business_id]).digest())
    return digest.hexdigest().upper()


def _read_catalog(path: Path) -> tuple[bytes, list[Mapping[str, Any]]]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.name != "p_items.json":
        raise PItemAppendEvidenceError("catalog must be an available p_items.json file")
    value = resolved.read_bytes()
    try:
        rows = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PItemAppendEvidenceError("p_items.json is invalid UTF-8 JSON") from error
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise PItemAppendEvidenceError("p_items.json must contain an array of objects")
    row_ids = tuple(row.get("id") for row in rows)
    if any(isinstance(business_id, bool) or not isinstance(business_id, int) or business_id < 1 for business_id in row_ids) or len(
        row_ids
    ) != len(set(row_ids)):
        raise PItemAppendEvidenceError("p_items.json business IDs are invalid or duplicated")
    return value, rows


def _excluded_current_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_max_business_id: int,
    arena_stage_ids: frozenset[int],
) -> list[dict[str, Any]]:
    excluded: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["id"])):
        business_id = int(row["id"])
        if business_id <= base_max_business_id or business_id in arena_stage_ids:
            continue
        mode = row.get("mode")
        source_type = row.get("sourceType")
        name = row.get("name")
        if not all(isinstance(value, str) and value for value in (mode, source_type, name)):
            raise PItemAppendEvidenceError(f"excluded catalog row {business_id} lacks stable identity fields")
        reason = "non-stage row is outside the arena-stage contract" if mode != "stage" else "source type is outside the arena-stage contract"
        excluded.append(
            {
                "business_id": business_id,
                "mode": mode,
                "name": name,
                "reason": reason,
                "source_type": source_type,
            }
        )
    return excluded


def _full_gallery_rankings(
    business_ids: Sequence[int],
    source_images: Mapping[int, np.ndarray],
    changed_ids: Sequence[int],
) -> dict[str, Any]:
    runtime = PItemReferenceRuntime.from_manifest(P_ITEM_RUNTIME_CONTRACT)
    gallery = PItemRenderedReferenceGallery(
        business_ids,
        tuple(source_images[business_id] for business_id in business_ids),
        runtime=runtime,
    )
    result: dict[str, Any] = {"coarse_fine_guard": "fails_safe_to_full_gallery"}
    for business_id in changed_ids:
        image = source_images[business_id]
        query = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        ranking = gallery._rank_full(query, (0, 0, 64, 64))
        if len(ranking) < 2 or ranking[0].p_item_id != business_id:
            raise PItemAppendEvidenceError(f"source reference {business_id} fails production full-gallery self-ranking")
        margin = float(ranking[0].similarity - ranking[1].similarity)
        if margin <= runtime.identity_minimum_margin:
            raise PItemAppendEvidenceError(f"source reference {business_id} full-gallery margin is below runtime threshold")
        result[str(business_id)] = {
            "margin": margin,
            "top1_business_id": ranking[0].p_item_id,
            "top1_similarity": float(ranking[0].similarity),
            "top2_business_id": ranking[1].p_item_id,
            "top2_similarity": float(ranking[1].similarity),
        }
    return result


def build_handoff_documents(
    *,
    base_gallery_path: Path,
    source_dir: Path,
    catalog_path: Path,
    gakumas_tools_revision: str,
    production_deployment_id: int,
    production_deployment_revision: str,
    production_deployment_status: str,
    added_business_ids: Sequence[int],
    replaced_business_ids: Sequence[int],
    gk_img_revision: str | None = None,
    first_added_revision: str | None = None,
    first_successful_production_revision: str | None = None,
    published_ui_manifest: Path | None = None,
    published_ui_original_root: Path | None = None,
    derived_candidate_root: Path | None = None,
    derived_overlay: Path | None = None,
    derived_qualification_report: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    added = _ordered_business_ids(added_business_ids, label="added business IDs")
    replaced = _ordered_business_ids(replaced_business_ids, label="replaced business IDs")
    if not added and not replaced:
        raise PItemAppendEvidenceError("at least one added or replaced business ID is required")
    if set(added) & set(replaced):
        raise PItemAppendEvidenceError("added and replaced business IDs overlap")

    catalog_revision = _validated_revision(gakumas_tools_revision, label="gakumas-tools revision")
    deployment_revision = _validated_revision(production_deployment_revision, label="production deployment revision")
    if (
        deployment_revision != catalog_revision
        or isinstance(production_deployment_id, bool)
        or production_deployment_id < 1
        or production_deployment_status != "success"
    ):
        raise PItemAppendEvidenceError("production deployment must be a successful deployment of the catalog revision")
    published = published_ui_manifest is not None
    derived = derived_candidate_root is not None
    if any(value is not None for value in (derived_candidate_root, derived_overlay, derived_qualification_report)) and not all(
        value is not None for value in (derived_candidate_root, derived_overlay, derived_qualification_report)
    ):
        raise PItemAppendEvidenceError("derived source requires candidate, overlay and scoped qualification report")
    if derived and (published or published_ui_original_root is not None or added):
        raise PItemAppendEvidenceError("derived full-reference replacement cannot mix published images or added identities")
    if published != (published_ui_original_root is not None):
        raise PItemAppendEvidenceError("published UI manifest and original root must be provided together")
    if published or derived:
        if any(value is not None for value in (gk_img_revision, first_added_revision, first_successful_production_revision)):
            raise PItemAppendEvidenceError("published UI or derived images must not be attributed to image Git revisions")
    else:
        gk_img_revision = _validated_revision(gk_img_revision, label="gk-img revision")
        first_added_revision = _validated_revision(first_added_revision, label="first-added revision")
        first_successful_production_revision = _validated_revision(
            first_successful_production_revision,
            label="first-successful-production revision",
        )

    base_ids, base_images, base_encoded = _read_base_gallery(base_gallery_path)
    source_paths, source_ids, source_images, sanitized = _read_source(source_dir)
    base_set = set(base_ids)
    source_set = set(source_ids)
    actual_added = tuple(sorted(source_set - base_set))
    missing_base = tuple(sorted(base_set - source_set))
    if actual_added != added or missing_base:
        raise PItemAppendEvidenceError("declared additions do not exactly match the base-to-source ID delta")
    if not set(replaced).issubset(base_set):
        raise PItemAppendEvidenceError("replaced business IDs must exist in the base gallery")
    actual_replaced = tuple(
        business_id
        for business_id in base_ids
        if (
            base_images[business_id].shape != source_images[business_id].shape
            or not np.array_equal(base_images[business_id], source_images[business_id])
        )
    )
    if actual_replaced != replaced:
        raise PItemAppendEvidenceError("declared replacements do not exactly match inherited pixel changes")
    for business_id in base_ids:
        if business_id not in set(replaced) and sanitized[business_id] != base_encoded[business_id]:
            raise PItemAppendEvidenceError("non-replaced inherited reference bytes differ from the immutable base")

    changed = tuple(sorted((*added, *replaced)))
    catalog_bytes, catalog_rows = _read_catalog(catalog_path)
    rows_by_id = {int(row["id"]): row for row in catalog_rows}
    arena_stage_ids = tuple(
        sorted(
            business_id for business_id, row in rows_by_id.items() if row.get("mode") == "stage" and row.get("sourceType") in STAGE_SOURCE_TYPES
        )
    )
    arena_stage_set = frozenset(arena_stage_ids)
    if not arena_stage_ids or not arena_stage_set.issubset(source_set):
        raise PItemAppendEvidenceError("complete source does not cover the authoritative arena-stage catalog")
    if not set(changed).issubset(arena_stage_set):
        raise PItemAppendEvidenceError("added and replaced references must be authoritative arena-stage entities")

    references: dict[str, dict[str, Any]] = {}
    derived_contract = None
    if derived:
        assert derived_candidate_root is not None and derived_overlay is not None and derived_qualification_report is not None
        try:
            derived_contract = build_derived_source(
                base_gallery_path=base_gallery_path, catalog_path=catalog_path,
                overlay_path=derived_overlay, source_dir=source_dir,
                candidate_root=derived_candidate_root, qualification_report=derived_qualification_report,
                runtime_contract=P_ITEM_RUNTIME_CONTRACT,
                catalog_production_deployment={"deployment_id": production_deployment_id, "revision": deployment_revision, "status": production_deployment_status},
            )
        except DerivedSourceError as error:
            raise PItemAppendEvidenceError(str(error)) from error
        references = derived_contract["references"]
    elif published:
        assert published_ui_manifest is not None and published_ui_original_root is not None
        try:
            manifest = validate_published_ui_source_manifest(
                published_ui_manifest,
                original_root=published_ui_original_root,
                output_root=source_dir,
                catalog_rows=catalog_rows,
                component="p_item",
                expected_ids=changed,
            )
        except PublishedUiReferenceError as error:
            raise PItemAppendEvidenceError(str(error)) from error
        references = manifest["references"]
    else:
        for business_id in changed:
            path = source_dir.resolve() / f"{business_id}.png"
            raw = path.read_bytes()
            image = source_images[business_id]
            if (image.shape[1], image.shape[0]) != TARGET_REFERENCE_SIZE:
                raise PItemAppendEvidenceError(f"changed source reference {business_id} must be 130x130 pixels")
            references[str(business_id)] = {
                "dimensions": list(TARGET_REFERENCE_SIZE),
                "git_blob_sha1": _git_blob_sha1(raw),
                "path": f"packages/gakumas-images/images/pItems/icons/{business_id}.png",
                "png_sha256": _sha256_bytes(raw),
                "rgb_pixel_sha256": _sha256_bytes(image[:, :, ::-1].tobytes()),
                "size": len(raw),
            }

    base_sha256 = _sha256_file(base_gallery_path.resolve())
    base_contract = {
        "added_business_ids": list(added),
        "base_business_id_count": len(base_ids),
        "inherited_business_id_count": len(base_ids) - len(replaced),
        "replaced_business_ids": list(replaced),
        "sha256": base_sha256,
        "target_business_id_count": len(source_ids),
    }
    catalog_contract = {
        "arena_stage_business_ids": list(arena_stage_ids),
        "excluded_current_rows": _excluded_current_rows(
            catalog_rows,
            base_max_business_id=max(base_ids),
            arena_stage_ids=arena_stage_set,
        ),
        "included_business_ids": list(changed),
        "p_items_git_blob_sha1": _git_blob_sha1(catalog_bytes),
        "p_items_sha256": _sha256_bytes(catalog_bytes),
        "path": "packages/gakumas-data/json/p_items.json",
        "repository": "https://github.com/surisuririsu/gakumas-tools",
        "revision": catalog_revision,
        "stage_scope_contract": STAGE_SCOPE_CONTRACT,
    }
    image_source_contract = {
        "first_added_revision": first_added_revision,
        "first_successful_production_revision": (first_successful_production_revision),
        "gk_img_gitlink_path": "gk-img",
        "gk_img_gitlink_repository": "https://github.com/surisuririsu/gk-img",
        "gk_img_gitlink_revision": gk_img_revision,
        "package_path": "packages/gakumas-images",
        "production_deployment": {
            "deployment_id": production_deployment_id,
            "revision": deployment_revision,
            "status": production_deployment_status,
        },
        "references": references,
        "repository": "https://github.com/surisuririsu/gakumas-tools",
    }
    if published:
        image_source_contract = {
            "source_type": "third_party_published_ui_crops",
            "catalog_production_deployment": {
                "deployment_id": production_deployment_id,
                "revision": deployment_revision,
                "status": production_deployment_status,
            },
            "references": references,
        }
    if derived:
        assert derived_contract is not None
        image_source_contract = derived_contract
    source_key = SOURCE_KEY if derived else "published_ui_crops" if published else "official_rendered_source"
    source_contract = {
        "business_id_count": len(source_ids),
        "business_ids": list(source_ids),
        "revision": (
            f"base-gallery-sha256@{base_sha256}+gakumas-tools@{catalog_revision}"
            + (SOURCE_REVISION_SUFFIX if derived else "+published-ui-crops-v1" if published else "")
        ),
        "sanitized_icons_sha256": _sanitized_digest(source_ids, sanitized),
        "source_icons_sha256": _source_digest(source_paths),
    }
    evidence = {
        "base_gallery": base_contract,
        "catalog": catalog_contract,
        source_key: image_source_contract,
        "schema_version": 4 if derived else 3 if published else 2,
        "source": source_contract,
        "tool_contract": (
            EVIDENCE_TOOL_CONTRACT if derived else P_ITEM_PRODUCTION_SOURCE_EVIDENCE_V3_TOOL_CONTRACT if published else P_ITEM_PRODUCTION_SOURCE_EVIDENCE_V2_TOOL_CONTRACT
        ),
    }
    provenance = {
        "base_gallery": base_contract,
        "catalog": catalog_contract,
        source_key: image_source_contract,
        "schema_version": 5 if derived else 4 if published else 3,
        "validation": {
            "full_gallery_64px": _full_gallery_rankings(source_ids, source_images, changed),
            "live_jjc_calibration": {
                "runtime_detail_authority": False,
                "status": "PENDING",
            },
            "metadata_removed_business_id_count": len(source_ids),
            "source_added_business_ids": list(added),
            "source_replacement_changed_business_ids": list(replaced),
        },
    }
    return evidence, provenance


def main() -> int:
    args = _parse_args()
    evidence_output = args.evidence_output.resolve()
    provenance_output = args.provenance_output.resolve()
    source_dir = args.source.resolve()
    if evidence_output == provenance_output:
        raise PItemAppendEvidenceError("evidence and provenance outputs must be different files")
    if evidence_output.is_relative_to(source_dir) or provenance_output.is_relative_to(source_dir):
        raise PItemAppendEvidenceError("evidence and provenance outputs must be outside the source inventory")
    if evidence_output.exists() or provenance_output.exists():
        raise PItemAppendEvidenceError("evidence and provenance outputs must not exist")
    evidence, provenance = build_handoff_documents(
        base_gallery_path=args.base_gallery,
        source_dir=args.source,
        catalog_path=args.catalog,
        gakumas_tools_revision=args.gakumas_tools_revision,
        production_deployment_id=args.production_deployment_id,
        production_deployment_revision=args.production_deployment_revision,
        production_deployment_status=args.production_deployment_status,
        gk_img_revision=args.gk_img_revision,
        first_added_revision=args.first_added_revision,
        first_successful_production_revision=(args.first_successful_production_revision),
        added_business_ids=args.added_business_id,
        replaced_business_ids=args.replaced_business_id,
        published_ui_manifest=args.published_ui_manifest,
        published_ui_original_root=args.published_ui_original_root,
        derived_candidate_root=getattr(args, "derived_candidate_root", None),
        derived_overlay=getattr(args, "derived_overlay", None),
        derived_qualification_report=getattr(args, "derived_qualification_report", None),
    )
    for path, value in (
        (evidence_output, evidence),
        (provenance_output, provenance),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_json_bytes(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
