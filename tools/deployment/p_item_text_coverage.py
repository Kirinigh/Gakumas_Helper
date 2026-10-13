"""Check full-client P-item text against the candidate's actual RIS catalog."""

from __future__ import annotations

import json
import importlib.util
from typing import Any
from pathlib import Path

CATALOG_PATH = "assets/arena-winrate/node_modules/gakumas-data/json/p_items.json"
REFERENCE_PATH = "data/p_item_detail_text.json"
RUNTIME_PATH = "agent/arena_winrate/p_item_text_reference.py"


class PItemTextCoverageError(ValueError):
    """A candidate lacks complete text bound to its arena-reachable items."""


def _candidate_file(candidate: Path, relative: str) -> Path:
    path = (candidate / relative).resolve()
    if not path.is_relative_to(candidate):
        raise PItemTextCoverageError(f"candidate P-item text path escapes root: {relative}")
    if not path.is_file():
        raise PItemTextCoverageError(f"candidate P-item text file is missing: {relative}")
    return path


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
    if report["missing_ids"] or report["binding_mismatch_ids"]:
        raise PItemTextCoverageError(
            "candidate P-item text coverage is incomplete: "
            f"missing_ids={report['missing_ids']}, "
            f"binding_mismatch_ids={report['binding_mismatch_ids']}"
        )
    return report
