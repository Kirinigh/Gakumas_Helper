"""Build a runtime arena engine/data component from one RIS source revision."""

from __future__ import annotations

import io
import os
import re
import csv
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from collections.abc import Mapping, Iterable

from .adapter import PROTOCOL_VERSION

PACKAGE_NAMES = ("gakumas-engine", "gakumas-data")
RELATIVE_SPECIFIER = re.compile(
    r'(?P<prefix>\bfrom\s+["\']|\bimport\s*["\'])(?P<path>\.\.?/[^"\']+)(?P<suffix>["\'])'
)
JSON_IMPORT = re.compile(r'(?P<statement>import\s+[^;]+\s+from\s+["\'][^"\']+\.json["\'])\s*;')
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
RUNNER_COMMIT_PATTERN = re.compile(
    r'^const UPSTREAM_COMMIT = "[0-9a-f]{40}";$',
    flags=re.MULTILINE,
)


class ArenaComponentBuildError(RuntimeError):
    """Raised when a downloaded RIS source tree cannot form a runtime bundle."""


RIS_SOURCE_BASE_PATHS = frozenset({
    "LICENSE", "packages/gakumas-engine/package.json", "packages/gakumas-data/package.json",
})
RIS_CSV_PARSER = "packages/gakumas-data/utils/csv.js"


def missing_ris_source_paths(paths: Iterable[str]) -> set[str]:
    available = set(paths)
    missing = set(RIS_SOURCE_BASE_PATHS - available)
    if "packages/gakumas-data/json/stages.json" not in available:
        missing.update({
            "packages/gakumas-data/csv/stages.csv", RIS_CSV_PARSER,
        } - available)
    return missing


def parse_ris_csv_rows(data: bytes) -> list[dict[str, object]]:
    """Match the official csv.js value conversion for immutable catalog queries."""
    try:
        rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ArenaComponentBuildError("RIS CSV is not valid UTF-8 CSV") from error
    rows = [row for row in rows if row]
    if not rows:
        raise ArenaComponentBuildError("RIS CSV header is missing")
    header, *records = rows
    result = []
    for row in records:
        if len(row) != len(header):
            raise ArenaComponentBuildError("RIS CSV row width differs from its header")
        result.append({
            key: (True if value == "TRUE" else False if value == "FALSE"
                  else int(value) if re.fullmatch(r"[0-9]+", value) else value)
            for key, value in zip(header, row, strict=True)
        })
    return result


def materialize_ris_csv_json(data_root: Path, node: Path) -> dict[str, object] | None:
    """Use this revision's official zero-dependency parser via the bundled Node."""
    if (data_root / "json/stages.json").is_file() or not (data_root / "csv/stages.csv").is_file():
        return None
    parser = data_root / "utils/csv.js"
    if not (data_root / "csv/stages.csv").is_file() or not parser.is_file():
        raise ArenaComponentBuildError("RIS data needs tracked JSON or CSV with its official parser")
    if not node.is_file():
        raise ArenaComponentBuildError("bundled Node is required for the official RIS CSV parser")
    program = (
        "import fs from 'node:fs';import path from 'node:path';"
        "import {pathToFileURL} from 'node:url';const root=process.argv[1];"
        "const {parseCsvRecords}=await import(pathToFileURL(path.join(root,'utils/csv.js')).href);"
        "const result={};for(const name of fs.readdirSync(path.join(root,'csv')).sort()){"
        "if(!/^[a-z_]+\\.csv$/.test(name))continue;"
        "result[name]=parseCsvRecords(fs.readFileSync(path.join(root,'csv',name),'utf8'));}"
        "process.stdout.write(JSON.stringify(result));"
    )
    completed = subprocess.run(
        [str(node), "--input-type=module", "-e", program, str(data_root)],
        cwd=data_root, capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    if completed.returncode:
        raise ArenaComponentBuildError("official RIS CSV parsing failed: " + completed.stderr[:1000])
    try:
        converted = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ArenaComponentBuildError("official RIS CSV parser returned invalid records") from error
    if not isinstance(converted, dict) or "stages.csv" not in converted:
        raise ArenaComponentBuildError("official RIS CSV parser omitted stage records")
    generated = {}
    for name, rows in sorted(converted.items()):
        if not re.fullmatch(r"[a-z_]+\.csv", name) or not isinstance(rows, list):
            raise ArenaComponentBuildError("official RIS CSV parser returned an invalid data entry")
        source = data_root / "csv" / name
        target = data_root / "json" / (Path(name).stem + ".json")
        if target.exists():
            raise ArenaComponentBuildError("derived RIS JSON must not replace tracked source")
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        generated["node_modules/gakumas-data/json/" + target.name] = {
            "source_csv": "packages/gakumas-data/csv/" + name,
            "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest().upper(),
            "json_sha256": hashlib.sha256(target.read_bytes()).hexdigest().upper(),
        }
    return {
        "schema_version": 1, "source_format": "csv",
        "transformer": RIS_CSV_PARSER,
        "transformer_sha256": hashlib.sha256(parser.read_bytes()).hexdigest().upper(),
        "node_sha256": hashlib.sha256(node.read_bytes()).hexdigest().upper(),
        "generation": "official parseCsvRecords(UTF-8 CSV); JSON ensure_ascii=false indent=2 LF",
        "files": generated,
    }


def rewrite_module_specifiers(path: Path) -> None:
    """Make the checked-in RIS JavaScript directly runnable by Node ESM."""

    source = path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        specifier = match.group("path")
        if Path(specifier).suffix == ".csv":
            source_csv = (path.parent / specifier).resolve()
            generated = source_csv.parent.parent / "json" / (source_csv.stem + ".json")
            if source_csv.parent.name != "csv" or not generated.is_file():
                raise ArenaComponentBuildError(f"RIS CSV import has no derived JSON: {specifier}")
            specifier = os.path.relpath(generated, path.parent).replace(os.sep, "/")
            if not specifier.startswith("."):
                specifier = "./" + specifier
            return f'{match.group("prefix")}{specifier}{match.group("suffix")}'
        if Path(specifier).suffix:
            return match.group(0)
        target = (path.parent / specifier).resolve()
        if target.with_suffix(".js").is_file():
            specifier += ".js"
        elif target.is_dir() and (target / "index.js").is_file():
            specifier += "/index.js"
        else:
            raise ArenaComponentBuildError(
                f"cannot resolve module specifier {specifier!r} in {path}"
            )
        return f'{match.group("prefix")}{specifier}{match.group("suffix")}'

    source = RELATIVE_SPECIFIER.sub(replace, source)
    source = JSON_IMPORT.sub(r'\g<statement> with { type: "json" };', source)
    path.write_text(source, encoding="utf-8", newline="\n")


def _read_manifest(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArenaComponentBuildError(f"baseline manifest is unavailable or invalid: {path}") from error
    if not isinstance(value, Mapping):
        raise ArenaComponentBuildError("baseline manifest root must be an object")
    return value


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_runtime_component(
    source_root: Path,
    output: Path,
    baseline_bundle: Path,
    *,
    upstream_commit: str,
    host_revision: str,
) -> None:
    """Materialize matching engine/data packages with the current GKH adapter."""

    if COMMIT_PATTERN.fullmatch(upstream_commit) is None:
        raise ArenaComponentBuildError("upstream commit must be a lowercase 40-character SHA")
    if not host_revision.strip():
        raise ArenaComponentBuildError("host revision must be non-empty")
    if output.exists() and any(output.iterdir()):
        raise ArenaComponentBuildError(f"component output must be absent or empty: {output}")

    baseline_manifest = _read_manifest(baseline_bundle / "manifest.json")
    if (
        baseline_manifest.get("schema_version") != 1
        or baseline_manifest.get("project") != "gakumas-tools"
        or baseline_manifest.get("adapter_protocol_version") != PROTOCOL_VERSION
    ):
        raise ArenaComponentBuildError("baseline engine bundle is incompatible with this adapter")

    try:
        output.mkdir(parents=True, exist_ok=True)
        package_root = source_root / "packages"
        destinations: list[Path] = []
        for package_name in PACKAGE_NAMES:
            source = package_root / package_name
            if not (source / "package.json").is_file():
                raise ArenaComponentBuildError(f"downloaded RIS package is missing: {source}")
            destination = output / "node_modules" / package_name
            shutil.copytree(source, destination)
            destinations.append(destination)

        csv_data = materialize_ris_csv_json(output / "node_modules/gakumas-data", baseline_bundle / "node.exe")
        for destination in destinations:
            for javascript in destination.rglob("*.js"):
                rewrite_module_specifiers(javascript)
            package_json_path = destination / "package.json"
            package_json = json.loads(package_json_path.read_text(encoding="utf-8"))
            if not isinstance(package_json, dict):
                raise ArenaComponentBuildError(f"package metadata is invalid: {package_json_path}")
            package_json["main"] = "./index.js"
            package_json["exports"] = "./index.js"
            package_json_path.write_text(
                json.dumps(package_json, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        license_file = source_root / "LICENSE"
        if not license_file.is_file():
            raise ArenaComponentBuildError("downloaded RIS BSD-3-Clause LICENSE is missing")
        notices = output / "THIRD_PARTY_NOTICES"
        baseline_notices = baseline_bundle / "THIRD_PARTY_NOTICES"
        if baseline_notices.is_dir():
            shutil.copytree(baseline_notices, notices)
        else:
            notices.mkdir()
        shutil.copy2(license_file, notices / "gakumas-tools-LICENSE")

        node = baseline_bundle / "node.exe"
        runner = baseline_bundle / "runner.mjs"
        scoring = baseline_bundle / "scoring.mjs"
        if not all(path.is_file() for path in (node, runner, scoring)):
            raise ArenaComponentBuildError("baseline Node runtime, runner or scoring module is missing")
        _link_or_copy(node, output / "node.exe")
        runner_source = runner.read_text(encoding="utf-8")
        matches = RUNNER_COMMIT_PATTERN.findall(runner_source)
        if len(matches) != 1:
            raise ArenaComponentBuildError("baseline runner must contain exactly one upstream commit pin")
        (output / "runner.mjs").write_text(
            RUNNER_COMMIT_PATTERN.sub(
                f'const UPSTREAM_COMMIT = "{upstream_commit}";',
                runner_source,
            ),
            encoding="utf-8",
            newline="\n",
        )
        shutil.copy2(scoring, output / "scoring.mjs")
        if './calibration-cache.mjs' in runner_source:
            calibration = baseline_bundle / "calibration-cache.mjs"
            if not calibration.is_file():
                raise ArenaComponentBuildError("baseline calibration cache module is missing")
            shutil.copy2(calibration, output / "calibration-cache.mjs")

        manifest = {
            "schema_version": 1,
            "project": "gakumas-tools",
            "repository": "https://github.com/surisuririsu/gakumas-tools",
            "branch": "production-deployment",
            "commit": upstream_commit,
            "license": "BSD-3-Clause",
            "runtime_network_required": False,
            "adapter_protocol_version": PROTOCOL_VERSION,
            "calibration_schema_version": baseline_manifest.get("calibration_schema_version"),
            "score_aggregation": baseline_manifest.get("score_aggregation"),
            "match_rule": baseline_manifest.get("match_rule"),
            "runtime": baseline_manifest.get("runtime"),
            "component_update": {
                "schema_version": 1,
                "source": "github-production-deployment",
                "host_revision": host_revision,
            },
        }
        if csv_data is not None:
            manifest["csv_data"] = csv_data
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
