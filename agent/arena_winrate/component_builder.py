"""Build a runtime arena engine/data component from one RIS source revision."""

from __future__ import annotations

import os
import re
import json
import shutil
from pathlib import Path
from collections.abc import Mapping

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


def rewrite_module_specifiers(path: Path) -> None:
    """Make the checked-in RIS JavaScript directly runnable by Node ESM."""

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
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
