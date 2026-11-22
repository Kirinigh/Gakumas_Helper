from __future__ import annotations

import csv
import sys
import json
import hashlib
import argparse
from pathlib import Path
from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.live_decision.high_score import (  # noqa: E402
    EFFECT_IR_VERSION,
    PlanType,
    compile_item_catalog,
)
from agent.p_item_recognition.reference import (  # noqa: E402
    PItemReferenceError,
    PItemRenderedReferenceGallery,
)

P_ITEMS = ROOT / "assets/data/p_items.csv"
P_ITEM_REFERENCE_ROOT = ROOT / "assets/resource/base/model/embedding/p_item_reference"
PINNED_GAKUMAS_TOOLS_REVISION = "3e9e8ebdc929dedd32cdd8d8911e5d8a905f76b6"
PINNED_P_ITEMS_LF_SHA256 = "CD3EF151DCF57ED1F4D5F34B8DBFC2A35C2EA61F24CB0E393C28F4C7D8777430"
LEGACY_SCOPE = "sourceType != produce"
PUBLISHED_SCOPE = (
    "450 immutable inherited PNGs plus six current native-bound published complete UI references; "
    "independent live calibration pending"
)
PUBLISHED_REVISION = (
    "base-gallery-sha256@132D6FA526323DAEDB1BB240E7F97620F6219BDA58FEDEFE63D2FCE791CDD4C9"
    "+gakumas-tools@fc4f18a3f0c391cc69c0048e91537facba1f1f0c+published-ui-crops-v1"
)


class PItemRecognitionAssetAuditError(ValueError):
    """Raised when the tracked JJC P-item asset cannot be audited safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _normalised_lf_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest().upper()


def _display_path(path: Path) -> str:
    try:
        value = path.relative_to(ROOT)
    except ValueError:
        value = path
    return str(value).replace("\\", "/")


def _load_item_rows(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as error:
        raise PItemRecognitionAssetAuditError(f"P-item semantic catalog is unavailable: {path}") from error
    required = {"id", "plan", "mode", "sourceType"}
    if not rows or any(not required <= set(row) for row in rows):
        raise PItemRecognitionAssetAuditError("P-item semantic catalog is missing required identity fields")
    ids = tuple(row["id"] for row in rows)
    if len(ids) != len(set(ids)) or any(not item_id.isdigit() or int(item_id) < 1 for item_id in ids):
        raise PItemRecognitionAssetAuditError("P-item semantic catalog IDs must be unique positive integers")
    return rows


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PItemRecognitionAssetAuditError(f"P-item reference manifest field {field} must be an object")
    return value


def build_recognition_asset_report(
    *,
    p_items_path: Path = P_ITEMS,
    asset_root: Path = P_ITEM_REFERENCE_ROOT,
) -> dict[str, object]:
    """Audit the JJC rendered-reference asset for offline rules analysis.

    This deliberately does not expose a runtime-ready switch.  The tracked
    gallery was calibrated for the arena member page, while the runtime still has
    no validated NIA PRO Live ROI or persistent item-inventory producer.
    """

    manifest_path = asset_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise PItemRecognitionAssetAuditError(f"P-item reference manifest is unavailable or invalid: {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise PItemRecognitionAssetAuditError("P-item reference manifest root must be an object")
    if manifest.get("schema_version") != 1:
        raise PItemRecognitionAssetAuditError("unsupported P-item reference manifest schema")
    if manifest.get("recognizer") != "p_item_rendered_reference_v1":
        raise PItemRecognitionAssetAuditError("unsupported P-item rendered-reference recognizer")

    reference = _mapping(manifest.get("reference_gallery"), field="reference_gallery")
    source = _mapping(manifest.get("source"), field="source")
    validation = _mapping(manifest.get("validation"), field="validation")
    scope = source.get("scope")
    if scope not in (LEGACY_SCOPE, PUBLISHED_SCOPE):
        raise PItemRecognitionAssetAuditError("P-item reference source domain is not the approved non-produce scope")
    source_revision = str(source.get("revision", ""))
    if scope == LEGACY_SCOPE:
        approved_revision = source_revision.split("+", 1)[0] == f"gakumas-tools@{PINNED_GAKUMAS_TOOLS_REVISION}"
    else:
        approved_revision = source_revision == PUBLISHED_REVISION
    if not approved_revision:
        raise PItemRecognitionAssetAuditError("P-item reference source revision does not match the offline audit")

    gallery_relative = Path(str(reference.get("path", "")))
    if gallery_relative.is_absolute() or ".." in gallery_relative.parts:
        raise PItemRecognitionAssetAuditError("P-item reference gallery path must stay inside its asset directory")
    gallery_path = asset_root / gallery_relative
    expected_sha256 = str(reference.get("sha256", "")).upper()
    if len(expected_sha256) != 64 or any(character not in "0123456789ABCDEF" for character in expected_sha256):
        raise PItemRecognitionAssetAuditError("P-item reference gallery SHA-256 is invalid")
    try:
        actual_sha256 = _sha256_file(gallery_path)
    except OSError as error:
        raise PItemRecognitionAssetAuditError(f"P-item reference gallery is unavailable: {gallery_path}") from error
    if actual_sha256 != expected_sha256:
        raise PItemRecognitionAssetAuditError("P-item reference gallery SHA-256 mismatch")

    try:
        gallery = PItemRenderedReferenceGallery.load(asset_root)
    except PItemReferenceError as error:
        raise PItemRecognitionAssetAuditError(str(error)) from error
    gallery_ids = {str(item_id) for item_id in gallery.p_item_ids}
    declared_count = reference.get("business_id_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count != len(gallery_ids):
        raise PItemRecognitionAssetAuditError("P-item reference business-ID count does not match the gallery")

    p_items_normalised_lf_sha256 = _normalised_lf_sha256(p_items_path)
    if p_items_normalised_lf_sha256 != PINNED_P_ITEMS_LF_SHA256:
        raise PItemRecognitionAssetAuditError("P-item semantic catalog LF-normalised SHA-256 mismatch")
    rows = _load_item_rows(p_items_path)
    catalog_ids = {row["id"] for row in rows}
    non_produce_ids = {row["id"] for row in rows if row["sourceType"] != "produce"}
    stage_ids = {row["id"] for row in rows if row["mode"] == "stage"}
    unknown_gallery_ids = gallery_ids - catalog_ids
    missing_non_produce_ids = non_produce_ids - gallery_ids

    plans: list[dict[str, object]] = []
    for plan_type in PlanType:
        plan_name = plan_type.value.lower()
        applicable_ids = {row["id"] for row in rows if row["mode"] == "stage" and row["plan"] in {"free", plan_name}}
        catalog = compile_item_catalog(p_items_path, plan_type=plan_type)
        supported_ids = set(catalog.items)
        covered_supported_ids = supported_ids & gallery_ids
        supported_missing_ids = supported_ids - gallery_ids
        recognised_but_unsupported_ids = (applicable_ids & gallery_ids) - supported_ids
        plans.append(
            {
                "plan_type": plan_type.value,
                "applicable_stage_rows": len(applicable_ids),
                "recognition_asset_applicable_rows": len(applicable_ids & gallery_ids),
                "semantic_supported_rows": len(supported_ids),
                "recognition_asset_supported_rows": len(covered_supported_ids),
                "recognition_asset_supported_ratio": (len(covered_supported_ids) / len(supported_ids) if supported_ids else 0.0),
                "semantic_supported_missing_from_asset_ids": sorted(
                    supported_missing_ids,
                    key=int,
                ),
                "recognised_but_semantically_unsupported_rows": len(recognised_but_unsupported_ids),
            }
        )

    blockers = [
        "nia_pro_live_item_roi_not_calibrated",
        "nia_pro_live_item_inventory_producer_unavailable",
    ]
    if validation.get("upgraded_marker_calibration") != "PASSED":
        blockers.append("normal_plus_real_domain_calibration_incomplete")
    if unknown_gallery_ids or missing_non_produce_ids:
        blockers.append("reference_and_semantic_business_id_sets_disagree")
    if stage_ids - gallery_ids:
        blockers.append("produce_source_stage_items_outside_reference_scope")

    return {
        "schema_version": 1,
        "reuse_status": "OFFLINE_BUSINESS_ID_COMPATIBILITY_ONLY",
        "runtime_ready": False,
        "runtime_blockers": blockers,
        "manifest": _display_path(manifest_path),
        "gallery": _display_path(gallery_path),
        "gallery_sha256": actual_sha256,
        "source_revision": source_revision,
        "source_scope": source["scope"],
        "gallery_business_id_count": len(gallery_ids),
        "semantic_catalog_business_id_count": len(catalog_ids),
        "non_produce_business_id_count": len(non_produce_ids),
        "non_produce_domain_exact_match": (gallery_ids == non_produce_ids and not unknown_gallery_ids),
        "unknown_gallery_item_ids": sorted(unknown_gallery_ids, key=int),
        "missing_non_produce_item_ids": sorted(missing_non_produce_ids, key=int),
        "stage_business_id_count": len(stage_ids),
        "recognition_asset_stage_business_id_count": len(stage_ids & gallery_ids),
        "stage_item_ids_outside_asset_scope": sorted(stage_ids - gallery_ids, key=int),
        "p_items_checkout_sha256": _sha256_file(p_items_path),
        "p_items_normalised_lf_sha256": p_items_normalised_lf_sha256,
        "plans": plans,
        "boundary": (
            "The fixed JJC gallery is accepted only as an offline business-ID and "
            "source-revision compatibility asset. Arena ROI geometry, thresholds, "
            "three-frame timing and acceptance decisions are not NIA PRO Live evidence."
        ),
    }


def build_report() -> dict[str, object]:
    plans: list[dict[str, object]] = []
    for plan_type in PlanType:
        catalog = compile_item_catalog(P_ITEMS, plan_type=plan_type)
        plans.append(
            {
                "plan_type": plan_type.value,
                "total_rows": catalog.total_rows,
                "supported_rows": catalog.supported_rows,
                "unsupported_rows": len(catalog.unsupported),
                "coverage": catalog.coverage,
                "supported_item_ids": sorted(catalog.items, key=int),
                "reason_counts": catalog.reason_counts,
                "unsupported": [
                    {
                        "item_id": item.item_id,
                        "name": item.name,
                        "reasons": list(item.reasons),
                    }
                    for item in catalog.unsupported
                ],
            }
        )
    checkout_source_sha256 = compile_item_catalog(
        P_ITEMS,
        plan_type=PlanType.ANOMALY,
    ).source_sha256
    return {
        "schema_version": 2,
        "compiler": EFFECT_IR_VERSION,
        "source": str(P_ITEMS.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": _normalised_lf_sha256(P_ITEMS),
        "checkout_source_sha256": checkout_source_sha256,
        "selection": "mode=stage and plan in {free, selected plan}",
        "semantic_reference": {
            "repository": "https://github.com/surisuririsu/gakumas-tools.git",
            "commit": "3e9e8ebdc929dedd32cdd8d8911e5d8a905f76b6",
            "license": "BSD-3-Clause",
            "upstream_file": "packages/gakumas-data/csv/p_items.csv",
            "upstream_sha256": "110CEAEDBCD044BFBAF68190CE5FA9DB24F58485377ACE45D3A08EAD2F6E3DBB",
            "supplement": {
                "item_ids": ["473", "474"],
                "commit": "6c3d00648c32ea72a05086a18c63ab62e0d0c7e5",
                "upstream_file": "packages/gakumas-data/json/p_items.json",
                "upstream_sha256": "61C4FB67EDDBBADF4D4C839535A356EDC2B7693532529D02587FEC5E3B2B3BE1",
            },
            "normalization": (
                "source_sha256 uses UTF-8 with LF-normalised line endings; checkout_source_sha256 records the platform checkout bytes."
            ),
        },
        "plans": plans,
        "recognition_asset_compatibility": build_recognition_asset_report(),
        "all_plans_have_atomic_coverage": all(plan["supported_rows"] + plan["unsupported_rows"] == plan["total_rows"] for plan in plans),
        "boundary": (
            "Only stage-mode P-items whose complete persistent effect sequence compiles "
            "are model-supported. Produce-mode rows, runtime recognition, fields, UI input "
            "and real-game differential validation remain outside this coverage claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report persistent P-item Effect IR coverage.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".local/task275-item-coverage.json",
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plans": [
                    {
                        "plan_type": plan["plan_type"],
                        "supported_rows": plan["supported_rows"],
                        "total_rows": plan["total_rows"],
                        "coverage": plan["coverage"],
                    }
                    for plan in report["plans"]
                ],
                "recognition_asset_compatibility": {
                    "reuse_status": report["recognition_asset_compatibility"]["reuse_status"],
                    "runtime_ready": report["recognition_asset_compatibility"]["runtime_ready"],
                    "gallery_business_id_count": report["recognition_asset_compatibility"]["gallery_business_id_count"],
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
