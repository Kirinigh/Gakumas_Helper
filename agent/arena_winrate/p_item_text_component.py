"""Small, versioned text assets paired with an immutable RIS component."""

from __future__ import annotations

import os
import re
import json
import uuid
from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping, Sequence

from .p_item_text_reference import (
    MAX_REFERENCE_BYTES,
    reference_coverage,
    validate_reference_asset,
    reference_revision_number,
)

REFERENCE_FILENAME = "p_item_detail_text.json"
REFERENCE_REPOSITORY = "Kirinigh/Gakumas_Helper"
REFERENCE_SOURCE_PATH = "assets/data/p_item_detail_text.json"


def validate_reference(value: object) -> dict:
    """Reject malformed assets, while permitting explicit catalog coverage gaps."""
    validate_reference_asset(value)
    return deepcopy(dict(value))


def load_reference(path: Path) -> dict:
    if path.stat().st_size > MAX_REFERENCE_BYTES:
        raise ValueError("P-item text reference exceeds its size limit")
    return validate_reference(json.loads(path.read_text(encoding="utf-8")))


def packaged_reference_path(project_root: Path) -> Path:
    installed = project_root / "data" / REFERENCE_FILENAME
    return installed if installed.is_file() else project_root / "assets" / "data" / REFERENCE_FILENAME


def bundle_reference_path(bundle: Path) -> Path | None:
    """A declaration must resolve to the fixed local filename, never an external path."""
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        return None  # Legacy, catalog-only callers do not have component metadata.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declaration = manifest.get("p_item_text_reference")
    if declaration is None:
        return None
    if (not isinstance(declaration, Mapping)
            or declaration.get("path") != REFERENCE_FILENAME):
        raise ValueError("invalid bundled P-item text reference declaration")
    path = bundle / REFERENCE_FILENAME
    reference = load_reference(path)
    if reference["revision"] != declaration.get("revision"):
        raise ValueError("bundled P-item text reference revision mismatch")
    return path


def read_p_item_rows(bundle: Path) -> list:
    path = bundle / "node_modules/gakumas-data/json/p_items.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("RIS P-item catalog must be an array")
    return rows


def _covered_ids(reference: Mapping, catalog: Sequence[Mapping]) -> frozenset[int]:
    coverage = reference_coverage(reference, catalog)
    unavailable = set(coverage["missing_ids"]) | set(coverage["binding_mismatch_ids"])
    return frozenset(row["id"] for row in catalog
                     if row.get("mode") == "stage" and row.get("sourceType") in {"pIdol", "support"}
                     and row["id"] not in unavailable)


def choose_reference(catalog: Sequence[Mapping], *, active: Mapping | None,
                     packaged: Mapping | None, published: Mapping | None) -> dict | None:
    """Never discard previously usable bindings to install a newer text asset.

    Higher revisions can fix the same covered set without adding an ID.
    A stale packaged or published copy cannot replace an activated copy.
    """
    selected = validate_reference(active) if active is not None else None
    covered = _covered_ids(selected, catalog) if selected is not None else frozenset()
    for candidate in (packaged, published):
        if candidate is None:
            continue
        candidate = validate_reference(candidate)
        candidate_ids = _covered_ids(candidate, catalog)
        if selected is not None and not candidate_ids.issuperset(covered):
            continue
        if selected is not None:
            current_number = reference_revision_number(selected["revision"])
            candidate_number = reference_revision_number(candidate["revision"])
            if candidate_number < current_number:
                continue
            if candidate_number == current_number and candidate["revision"] != selected["revision"]:
                raise ValueError("P-item text reference revision number was reused")
        if selected is not None and selected["revision"] == candidate["revision"]:
            if selected != candidate:
                raise ValueError("changed P-item text rows reused an existing revision")
            continue
        selected, covered = candidate, candidate_ids
    return selected


def bind_reference(bundle: Path, reference: Mapping, *, host_revision: str) -> None:
    reference = validate_reference(reference)
    path = bundle / REFERENCE_FILENAME
    # Cloned engine files can be hard links; metadata and text never are.
    path.unlink(missing_ok=True)
    path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component_update = manifest.setdefault("component_update", {"schema_version": 1})
    component_update["host_revision"] = host_revision
    manifest["p_item_text_reference"] = {"revision": reference["revision"], "path": REFERENCE_FILENAME}
    manifest_path.unlink()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def coverage_warnings(bundle: Path, fallback: Path) -> tuple[str, ...]:
    bound = bundle_reference_path(bundle)
    try:
        reference = load_reference(bound or fallback)
    except (OSError, ValueError):
        if bound is not None:
            raise
        return ("P道具正文对照资源不可用；现有图像和唯一标题确认仍可使用，遇到正文缺项将明确提示。",)
    coverage = reference_coverage(reference, read_p_item_rows(bundle))
    gaps = sorted(set(coverage["missing_ids"]) | set(coverage["binding_mismatch_ids"]))
    if not gaps:
        return ()
    return ("P道具正文对照尚未覆盖当前 RIS 的 " + str(len(gaps))
            + " 项（ID：" + ", ".join(map(str, gaps[:16]))
            + ("…" if len(gaps) > 16 else "")
            + "）。已支持的道具仍可使用；遇到缺项将明确提示，重启客户端可检查正文更新。",)


class GitHubPItemTextSource:
    """One latest-path query and a small immutable download, using shared transport."""

    def __init__(self, transport, *, cache_path: Path | None = None, on_cache_error=None) -> None:
        self.transport = transport
        self.cache_path, self.on_cache_error = cache_path, on_cache_error
        self._cached_commit: str | None = None
        self._cached_reference: dict | None = None
        if cache_path is not None:
            try:
                if cache_path.stat().st_size > MAX_REFERENCE_BYTES + 512:
                    raise ValueError("P-item text source cache exceeds size limit")
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                commit = cached["commit"]
                if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
                    raise ValueError("invalid P-item text source cache commit")
                self._cached_reference = validate_reference(cached["reference"])
                self._cached_commit = commit
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError, KeyError) as error:
                self._cache_error(error)

    def _cache_error(self, error: Exception) -> None:
        if self.on_cache_error is not None:
            self.on_cache_error(error)

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        temporary = self.cache_path.with_name(f"{self.cache_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps({"commit": self._cached_commit, "reference": self._cached_reference},
                                            ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, self.cache_path)
        except OSError as error:
            self._cache_error(error)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                self._cache_error(error)

    def fetch_latest_reference(self) -> dict:
        url = (f"https://api.github.com/repos/{REFERENCE_REPOSITORY}/commits"
               "?path=assets%2Fdata%2Fp_item_detail_text.json&per_page=1")
        rows = self.transport._request_json(url, timeout=5.0)
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], Mapping):
            raise ValueError("published P-item text revision is unavailable")
        commit = rows[0].get("sha")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError("invalid published P-item text source commit")
        if commit != self._cached_commit:
            url = f"https://raw.githubusercontent.com/{REFERENCE_REPOSITORY}/{commit}/{REFERENCE_SOURCE_PATH}"
            raw = self.transport._request_bytes(url, timeout=15.0, maximum=MAX_REFERENCE_BYTES)
            self._cached_reference = validate_reference(json.loads(raw.decode("utf-8")))
            self._cached_commit = commit
            self._save_cache()
        return deepcopy(self._cached_reference)
