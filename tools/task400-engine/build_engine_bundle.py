"""Build an offline gakumas-tools engine directory from a pinned source tree."""

from __future__ import annotations

import re
import json
import shutil
import hashlib
import zipfile
import argparse
from pathlib import Path

UPSTREAM_COMMIT = "5658ec13ec9978f1117ccc191dfb83c346d42f6f"
PACKAGE_NAMES = ("gakumas-engine", "gakumas-data")
NODE_VERSION = "24.19.0"
NODE_ARCHIVE_NAME = f"node-v{NODE_VERSION}-win-x64.zip"
NODE_ARCHIVE_ROOT = f"node-v{NODE_VERSION}-win-x64"
ADAPTER_PROTOCOL_VERSION = "2.0"
CALIBRATION_SCHEMA_VERSION = 8
SCORE_AGGREGATION = "raw_sum_plus_stage_wide_first_place_20_percent"
MATCH_RULE = "best_of_three_strict_wins"
RELATIVE_SPECIFIER = re.compile(r'(?P<prefix>\bfrom\s+["\']|\bimport\s*["\'])(?P<path>\.\.?/[^"\']+)(?P<suffix>["\'])')
JSON_IMPORT = re.compile(r'(?P<statement>import\s+[^;]+\s+from\s+["\'][^"\']+\.json["\'])\s*;')
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_deterministic_archive(bundle: Path, archive_path: Path) -> None:
    if archive_path.exists():
        raise ValueError(f"archive output already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(bundle.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(bundle)).replace("\\", "/")
            entry = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def rewrite_module_specifiers(path: Path) -> None:
    source = path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        specifier = match.group("path")
        if Path(specifier).suffix:
            return match.group(0)
        target = (path.parent / specifier).resolve()
        if target.with_suffix(".js").is_file():
            specifier += ".js"
        elif target.is_dir() and (target / "index.js").is_file():
            specifier += "/index.js"
        else:
            raise ValueError(f"cannot resolve module specifier {specifier!r} in {path}")
        return f'{match.group("prefix")}{specifier}{match.group("suffix")}'

    source = RELATIVE_SPECIFIER.sub(replace, source)
    source = JSON_IMPORT.sub(r'\g<statement> with { type: "json" };', source)
    path.write_text(source, encoding="utf-8", newline="\n")


def build(
    source_root: Path,
    output: Path,
    runner: Path,
    source_archive_sha256: str,
    node_runtime_zip: Path | None,
    node_runtime_sha256: str | None,
    archive_output: Path | None,
    upstream_commit: str = UPSTREAM_COMMIT,
) -> None:
    if COMMIT_PATTERN.fullmatch(upstream_commit) is None:
        raise ValueError("upstream commit must be a lowercase 40-character SHA")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    package_root = source_root / "packages"
    for package_name in PACKAGE_NAMES:
        source = package_root / package_name
        if not (source / "package.json").is_file():
            raise ValueError(f"pinned package is missing: {source}")
        destination = output / "node_modules" / package_name
        shutil.copytree(source, destination)
        for javascript in destination.rglob("*.js"):
            rewrite_module_specifiers(javascript)
        package_json_path = destination / "package.json"
        package_json = json.loads(package_json_path.read_text(encoding="utf-8"))
        package_json["main"] = "./index.js"
        package_json["exports"] = "./index.js"
        package_json_path.write_text(json.dumps(package_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    license_file = source_root / "LICENSE"
    if not license_file.is_file():
        raise ValueError("upstream BSD-3-Clause LICENSE is missing")
    notices = output / "THIRD_PARTY_NOTICES"
    notices.mkdir()
    shutil.copy2(license_file, notices / "gakumas-tools-LICENSE")
    runner_source = runner.read_text(encoding="utf-8")
    expected_pin = f'const UPSTREAM_COMMIT = "{UPSTREAM_COMMIT}";'
    if runner_source.count(expected_pin) != 1:
        raise ValueError("runner must contain exactly one default upstream commit pin")
    (output / "runner.mjs").write_text(
        runner_source.replace(
            expected_pin,
            f'const UPSTREAM_COMMIT = "{upstream_commit}";',
        ),
        encoding="utf-8",
        newline="\n",
    )
    scoring = runner.with_name("scoring.mjs")
    if not scoring.is_file():
        raise ValueError(f"scoring module is missing next to runner: {scoring}")
    shutil.copy2(scoring, output / "scoring.mjs")

    runtime: dict[str, object] = {"included": False}
    if (node_runtime_zip is not None or node_runtime_sha256 is not None):
        if node_runtime_zip is None or node_runtime_sha256 is None:
            raise ValueError("node runtime ZIP and SHA-256 must be provided together")
        actual_runtime_sha256 = sha256(node_runtime_zip)
        if actual_runtime_sha256 != node_runtime_sha256.upper():
            raise ValueError(
                f"Node.js archive SHA-256 mismatch: expected {node_runtime_sha256.upper()}, got {actual_runtime_sha256}"
            )
        with zipfile.ZipFile(node_runtime_zip) as archive:
            node_member = f"{NODE_ARCHIVE_ROOT}/node.exe"
            license_member = f"{NODE_ARCHIVE_ROOT}/LICENSE"
            if node_member not in archive.namelist() or license_member not in archive.namelist():
                raise ValueError("Node.js archive is missing node.exe or LICENSE")
            with archive.open(node_member) as source, (output / "node.exe").open("wb") as destination:
                shutil.copyfileobj(source, destination)
            with archive.open(license_member) as source, (notices / "Node.js-LICENSE").open("wb") as destination:
                shutil.copyfileobj(source, destination)
        runtime = {
            "included": True,
            "project": "Node.js",
            "version": NODE_VERSION,
            "source_url": f"https://nodejs.org/download/release/v{NODE_VERSION}/{NODE_ARCHIVE_NAME}",
            "archive_sha256": actual_runtime_sha256,
            "executable_sha256": sha256(output / "node.exe"),
            "license_file": "THIRD_PARTY_NOTICES/Node.js-LICENSE",
        }

    files = {
        str(path.relative_to(output)).replace("\\", "/"): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "project": "gakumas-tools",
        "repository": "https://github.com/surisuririsu/gakumas-tools",
        "branch": "master",
        "commit": upstream_commit,
        "source_archive_sha256": source_archive_sha256.upper(),
        "license": "BSD-3-Clause",
        "runtime_network_required": False,
        "adapter_protocol_version": ADAPTER_PROTOCOL_VERSION,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "score_aggregation": SCORE_AGGREGATION,
        "match_rule": MATCH_RULE,
        "runtime": runtime,
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if archive_output is not None:
        write_deterministic_archive(output, archive_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("runner.mjs"))
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--node-runtime-zip", type=Path)
    parser.add_argument("--node-runtime-sha256")
    parser.add_argument("--archive-output", type=Path)
    parser.add_argument("--upstream-commit", default=UPSTREAM_COMMIT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(
        arguments.source_root.resolve(),
        arguments.output.resolve(),
        arguments.runner.resolve(),
        arguments.source_archive_sha256,
        arguments.node_runtime_zip.resolve() if arguments.node_runtime_zip else None,
        arguments.node_runtime_sha256,
        arguments.archive_output.resolve() if arguments.archive_output else None,
        arguments.upstream_commit,
    )
