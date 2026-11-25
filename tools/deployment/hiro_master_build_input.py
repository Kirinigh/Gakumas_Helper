"""Reuse Hiro's pinned simulator master in an isolated release build."""

from __future__ import annotations

import re
import json
import time
import shutil
import sqlite3
import argparse
import tempfile
from pathlib import Path, PurePosixPath
from contextlib import closing

try:
    from tools.deployment.build_derived_package import BuildError, _load_json, sha256_file, _validate_input_tree
except ModuleNotFoundError:  # Direct script execution from tools/deployment.
    from build_derived_package import BuildError, _load_json, sha256_file, _validate_input_tree

RUNTIME_MANIFEST = "runtime-manifest.json"
MASTER_ROOT = PurePosixPath("vendor/gakumas-player/data/master")


def master_paths(revision: str) -> frozenset[str]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise BuildError("Hiro master revision must be a full lowercase Git SHA")
    snapshot = MASTER_ROOT / revision
    return frozenset({
        (MASTER_ROOT / "current.json").as_posix(),
        *((snapshot / name).as_posix() for name in ("build.json", "catalog.sqlite", "manifest.json", "source-files.json")),
        f"native-master/master-{revision}.json",
    })


def _manifest(runtime: Path, expected_sha256: str) -> tuple[str, dict[str, str], dict[str, str]]:
    path = runtime / RUNTIME_MANIFEST
    if not path.is_file() or sha256_file(path).casefold() != expected_sha256.casefold():
        raise BuildError("Hiro runtime manifest differs from the qualified build input")
    manifest = _load_json(path)
    revisions = manifest.get("source_revisions")
    revision = revisions.get("master") if isinstance(revisions, dict) else None
    if not isinstance(revision, str):
        raise BuildError("Hiro runtime manifest has no pinned master revision")
    files = manifest.get("files")
    if not isinstance(files, dict) or RUNTIME_MANIFEST in files:
        raise BuildError("Hiro runtime manifest has an invalid file inventory")
    selected = master_paths(revision)
    build_report = (MASTER_ROOT / revision / "build.json").as_posix()
    if build_report not in files:
        selected -= {build_report}
    if not selected <= files.keys():
        raise BuildError("Hiro runtime manifest does not bind all simulator master files")
    for name, digest in files.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name
                or ":" in name or "\\" in name or name == "."
                or not isinstance(digest, str) or re.fullmatch(r"[a-fA-F0-9]{64}", digest) is None):
            raise BuildError(f"Invalid Hiro runtime inventory entry: {name}")
    if len({name.casefold() for name in files}) != len(files):
        raise BuildError("Hiro runtime inventory contains duplicate Windows paths")
    master = {name: files[name] for name in selected}
    remaining = {name: digest for name, digest in files.items() if name not in selected}
    remaining[RUNTIME_MANIFEST] = expected_sha256
    return revision, master, remaining


def _check_inventory(root: Path, files: dict[str, str]) -> None:
    _validate_input_tree(root, label="Hiro build input", reject_private_names=False)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != files.keys():
        raise BuildError(f"Hiro build input inventory differs: missing={sorted(files.keys() - actual)}, "
                         f"extra={sorted(actual - files.keys())}")


def _copy_verified(source: Path, destination: Path, files: dict[str, str]) -> None:
    for name, expected in sorted(files.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
        if sha256_file(target).casefold() != expected.casefold():
            raise BuildError(f"Hiro build input hash mismatch: {name}")


def _check_master_pair(root: Path, revision: str, hashes: dict[str, str]) -> None:
    snapshot = MASTER_ROOT / revision
    db_path = snapshot / "catalog.sqlite"
    source_manifest = snapshot / "manifest.json"
    pointer = _load_json(root / MASTER_ROOT / "current.json")
    provenance = _load_json(root / source_manifest)
    native = _load_json(root / f"native-master/master-{revision}.json")
    bindings = [pointer]
    build_report = snapshot / "build.json"
    if build_report.as_posix() in hashes:
        bindings.append(_load_json(root / build_report))
    if any(document.get("commit") != revision for document in (*bindings, provenance, native)):
        raise BuildError("Hiro master files disagree on the snapshot revision")
    if pointer.get("database") != f"{revision}/catalog.sqlite":
        raise BuildError("Hiro master pointer selects a different database")
    expected = {"database_sha256": hashes[db_path.as_posix()].casefold(),
                "manifest_sha256": hashes[source_manifest.as_posix()].casefold()}
    for document in bindings:
        if any(str(document.get(field, "")).casefold() != digest for field, digest in expected.items()):
            raise BuildError("Hiro master pointer or build report disagrees with the pinned database or source")
    if not isinstance(native.get("tables"), dict) or not native["tables"] or not isinstance(native.get("rules"), dict):
        raise BuildError("Hiro native master does not contain its tables and rules")
    try:
        with closing(sqlite3.connect((root / db_path).as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            row = db.execute("SELECT value FROM metadata WHERE key = ?", ("snapshot",)).fetchone()
        identity = json.loads(row[0]) if row else None
    except (sqlite3.Error, ValueError, TypeError) as error:
        raise BuildError("Hiro master database snapshot metadata is unreadable") from error
    if (not isinstance(identity, dict) or identity.get("commit") != revision
            or str(identity.get("manifest_sha256", "")).casefold() != expected["manifest_sha256"]):
        raise BuildError("Hiro master database snapshot metadata disagrees with the manifest")


def _new_output(output: Path, *inputs: Path) -> Path:
    output = output.resolve()
    if output.exists() or any(output.is_relative_to(root) for root in inputs):
        raise BuildError("Hiro build output must be a new directory outside its inputs")
    return output


def _promote(staged: Path, output: Path) -> None:
    # A Windows file scanner can briefly hold a just-validated SQLite file.
    for delay in (0.05, 0.15, 0.3, None):
        if output.exists():
            raise FileExistsError(f"Hiro build output already exists: {output}")
        try:
            staged.rename(output)
            return
        except OSError as error:
            if delay is None or getattr(error, "winerror", None) not in {5, 32, 33}:
                raise
        time.sleep(delay)


def split_hiro_runtime(runtime: Path, output: Path, *, manifest_sha256: str) -> None:
    """Export the simulator master and the rest of its qualified runtime as two trees."""
    runtime = runtime.resolve()
    output = _new_output(output, runtime)
    revision, master, remaining = _manifest(runtime, manifest_sha256)
    _check_inventory(runtime, master | remaining)
    with tempfile.TemporaryDirectory(prefix=".hiro-split-", dir=output.parent) as temporary:
        release_input = Path(temporary) / "build-input"
        _copy_verified(runtime, release_input / "master", master)
        _copy_verified(runtime, release_input / "runtime", remaining)
        _check_master_pair(release_input / "master", revision, master)
        _promote(release_input, output)


def assemble_hiro_runtime(runtime: Path, master_cache: Path, output: Path, *, manifest_sha256: str) -> None:
    """Reassemble the target runtime using only master bytes pinned by its manifest."""
    runtime, master_cache = runtime.resolve(), master_cache.resolve()
    output = _new_output(output, runtime, master_cache)
    revision, master, remaining = _manifest(runtime, manifest_sha256)
    _check_inventory(runtime, remaining)
    _check_inventory(master_cache, master)
    with tempfile.TemporaryDirectory(prefix=".hiro-assemble-", dir=output.parent) as temporary:
        assembled = Path(temporary) / "runtime"
        _copy_verified(master_cache, assembled, master)
        _check_master_pair(assembled, revision, master)
        _copy_verified(runtime, assembled, remaining)
        _promote(assembled, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("split", "assemble"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True, help="Digest from the qualified runtime handoff")
    parser.add_argument("--master-cache", type=Path)
    args = parser.parse_args()
    if args.phase == "split":
        if args.master_cache is not None:
            parser.error("--master-cache is only used for assembly")
        split_hiro_runtime(args.runtime, args.output, manifest_sha256=args.manifest_sha256)
    else:
        if args.master_cache is None:
            parser.error("--master-cache is required for assembly")
        assemble_hiro_runtime(args.runtime, args.master_cache, args.output, manifest_sha256=args.manifest_sha256)
