"""Build-only reuse of the exact Node runtime pinned by a RIS READY manifest."""

from __future__ import annotations

import shutil
import tempfile
from typing import Any
from pathlib import Path

from tools.deployment.build_derived_package import (
    BuildError,
    _load_json,
    sha256_file,
    _is_link_or_reparse,
    _validate_input_tree,
    _validate_engine_bundle,
)

RIS_RUNTIME_FILES = ("node.exe", "THIRD_PARTY_NOTICES/Node.js-LICENSE")


def _pinned_ris_manifest(bundle: Path, expected_sha256: str) -> dict[str, Any]:
    path = bundle / "manifest.json"
    if not path.is_file() or sha256_file(path).casefold() != expected_sha256.casefold():
        raise BuildError("RIS manifest differs from the qualified build input")
    manifest = _load_json(path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not all(name in files for name in RIS_RUNTIME_FILES):
        raise BuildError("RIS manifest must bind the Node executable and its license")
    return manifest


def export_ris_build_input(bundle: Path, output: Path, *, manifest_sha256: str) -> None:
    """Project a qualified RIS bundle into a build input without its reusable runtime."""
    bundle, output = bundle.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(bundle):
        raise BuildError("RIS build input must use a new directory outside the qualified bundle")
    _validate_engine_bundle(bundle)
    _pinned_ris_manifest(bundle, manifest_sha256)

    def omit_runtime(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).relative_to(bundle)
        return {name for name in names if (relative / name).as_posix() in RIS_RUNTIME_FILES}

    with tempfile.TemporaryDirectory(prefix=".ris-export-", dir=output.parent) as temporary:
        staged = Path(temporary) / "component"
        shutil.copytree(bundle, staged, ignore=omit_runtime)
        _pinned_ris_manifest(staged, manifest_sha256)
        staged.rename(output)


def assemble_ris_build_input(
    component: Path, runtime_from: Path, output: Path, *, manifest_sha256: str,
) -> dict[str, Any]:
    """Assemble and verify a split build input before exposing a complete bundle."""
    component, runtime_from, output = component.resolve(), runtime_from.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(component) or output.is_relative_to(runtime_from):
        raise BuildError("RIS assembly must use a new directory outside its inputs")
    _validate_input_tree(component, label="RIS build component", reject_private_names=True)
    manifest = _pinned_ris_manifest(component, manifest_sha256)
    for name in RIS_RUNTIME_FILES:
        if (component / name).exists():
            raise BuildError(f"Split RIS component already contains its reusable runtime: {name}")
        source = runtime_from / name
        if not source.is_file() or source.resolve() != source or _is_link_or_reparse(source):
            raise BuildError(f"Reusable RIS runtime file is missing or linked: {name}")
        expected = manifest["files"][name]
        if not isinstance(expected, str) or sha256_file(source).casefold() != expected.casefold():
            raise BuildError(f"Reusable RIS runtime hash mismatch: {name}")

    with tempfile.TemporaryDirectory(prefix=".ris-assemble-", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        shutil.copytree(component, staged)
        for name in RIS_RUNTIME_FILES:
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runtime_from / name, destination)
        _pinned_ris_manifest(staged, manifest_sha256)
        validated = _validate_engine_bundle(staged)
        staged.rename(output)
        return validated
