"""Check full-client P-item text against the candidate's actual RIS catalog."""

from __future__ import annotations

import json
import importlib.util
from typing import Any
from pathlib import Path

CATALOG_PATH = "assets/arena-winrate/node_modules/gakumas-data/json/p_items.json"
REFERENCE_PATH = "data/p_item_detail_text.json"
RUNTIME_PATH = "agent/arena_winrate/p_item_text_reference.py"
CLASSIFICATION_PATH = "data/p_item_classification.json"


class PItemTextCoverageError(ValueError):
    """A candidate lacks complete text bound to its arena-reachable items."""


def _candidate_file(candidate: Path, relative: str) -> Path:
    path = (candidate / relative).resolve()
    if not path.is_relative_to(candidate):
        raise PItemTextCoverageError(f"candidate P-item text path escapes root: {relative}")
    if not path.is_file():
        raise PItemTextCoverageError(f"candidate P-item text file is missing: {relative}")
    return path


def _validated_temporary_missing_ids(candidate: Path, catalog_path: Path) -> frozenset[int]:
    classification_path = candidate / CLASSIFICATION_PATH
    if not classification_path.exists():
        return frozenset()
    classification_path = _candidate_file(candidate, CLASSIFICATION_PATH)
    try:
        classification = json.loads(classification_path.read_text(encoding="utf-8"))
        if not isinstance(classification, dict):
            raise ValueError("P-item classification must be an object")
        if classification.get("temporary_missing_references") is None:
            return frozenset()
    except (OSError, UnicodeError, ValueError) as error:
        raise PItemTextCoverageError("candidate temporary missing P-item classification is invalid") from error

    # Import lazily: both packers already import this coverage module. Reuse
    # their exact gallery/catalog/disabled-row/blank-image contract instead of
    # interpreting reference_available=False as a text-coverage exemption.
    try:
        from tools.deployment import build_release_asset as release
    except ModuleNotFoundError:  # Direct script execution from tools/deployment.
        import build_release_asset as release
    try:
        manifest_path = _candidate_file(candidate, release.COMPONENT_MANIFESTS["p_item_reference"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gallery_path = release._resolve_inside(manifest_path.parent, manifest["reference_gallery"]["path"])
        annotation = release._validate_p_item_release_contract(
            manifest, gallery_path=gallery_path, catalog_path=catalog_path,
            classification_path=classification_path, asset_root=candidate,
        )
        if annotation is None:
            raise ValueError("explicit missing-reference declaration was not validated")
        return frozenset(annotation["business_ids"])
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, release.ReleaseBuildError) as error:
        raise PItemTextCoverageError("candidate temporary missing P-item reference contract is invalid") from error


def validate_candidate_p_item_text(candidate: Path) -> dict[str, Any]:
    """Use packaged runtime rules without importing the arena application."""
    candidate = candidate.resolve()
    catalog_path = _candidate_file(candidate, CATALOG_PATH)
    reference_path = _candidate_file(candidate, REFERENCE_PATH)
    runtime_path = _candidate_file(candidate, RUNTIME_PATH)
    try:
        spec = importlib.util.spec_from_file_location("_gkh_candidate_p_item_text", runtime_path)
        if spec is None or spec.loader is None:
            raise ValueError("candidate P-item text runtime cannot be loaded")
        runtime = importlib.util.module_from_spec(spec)
        # Compile the lightweight module directly: validation must not add
        # __pycache__ files to the immutable candidate or execute __init__.py.
        exec(compile(runtime_path.read_text(encoding="utf-8"), str(runtime_path), "exec"), runtime.__dict__)
        reference_bytes = reference_path.read_bytes()
        if len(reference_bytes) > runtime.MAX_REFERENCE_BYTES:
            raise ValueError("candidate P-item text reference exceeds the runtime byte limit")
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        reference = json.loads(reference_bytes.decode("utf-8"))
        if not isinstance(catalog, list) or not isinstance(reference, dict):
            raise ValueError("candidate catalog must be an array and text reference an object")
        runtime.validate_reference_asset(reference)
        report = runtime.reference_coverage(reference, catalog)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError, ImportError, SyntaxError) as error:
        raise PItemTextCoverageError("candidate P-item text reference or catalog is invalid") from error
    temporary_ids = _validated_temporary_missing_ids(candidate, catalog_path)
    if temporary_ids:
        actual_missing = set(report["missing_ids"])
        report = {
            **report,
            "missing_ids": sorted(actual_missing - temporary_ids),
            "temporary_missing_ids": sorted(actual_missing & temporary_ids),
        }
    if report["missing_ids"] or report["binding_mismatch_ids"]:
        raise PItemTextCoverageError(
            "candidate P-item text coverage is incomplete: "
            f"missing_ids={report['missing_ids']}, "
            f"binding_mismatch_ids={report['binding_mismatch_ids']}"
        )
    return report
