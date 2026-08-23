"""Build a versioned MaaGakumasu package from a clean upstream release.

The builder never edits an installed Maa directory.  It verifies the official
release archive, extracts it into a staging directory, replaces resource code
with this repository's integrated source, validates the pinned arena engine,
and only then publishes the candidate directory atomically.
"""

from __future__ import annotations

import os
import re
import json
import stat
import uuid
import shutil
import hashlib
import zipfile
import argparse
import tempfile
import subprocess
from typing import Any
from pathlib import Path

SCHEMA_VERSION = 1
UPSTREAM_REPOSITORY = "https://github.com/SuperWaterGod/MaaGakumasu"
LOCAL_TRIAL_VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+\+gkh\.[0-9a-f]{7,40}$")
DERIVED_RELEASE_VERSION_PATTERN = re.compile(
    r"^v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-(?:alpha|beta)\.(?:0|[1-9]\d*))?$"
)
DATED_PREVIEW_VERSION_PATTERN = re.compile(r"^v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\+gkh\.\d{6}$")
GITHUB_REPOSITORY_PATTERN = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+/?$")

REPLACED_TREES = {
    "agent": "agent",
    "assets/resource": "resource",
    "assets/tasks": "tasks",
    "assets/lang": "lang",
    "assets/data": "data",
    "THIRD_PARTY_NOTICES": "THIRD_PARTY_NOTICES",
}

ROOT_FILES = ("README.md", "LICENSE", "logo.ico", "requirements.txt", "ASSET_PROVENANCE.md")
MUTABLE_RUNTIME_PATHS = (".local", "config", "appsettings.json")
PRIVATE_INPUT_NAMES = {".env", ".env.local", ".netrc", ".pypirc", "credentials.json", "secrets.json"}
PRIVATE_INPUT_SUFFIXES = {".log", ".dmp", ".dump", ".tmp", ".bak", ".sqlite", ".sqlite3", ".db"}
NON_RUNTIME_SITE_PACKAGE_DIRS = ("bin", "Scripts")

REQUIRED_FILES = (
    "MaaGakumasu.exe",
    "interface.json",
    "ASSET_PROVENANCE.md",
    "agent/main.py",
    "agent/custom/action/arena_reader.py",
    "agent/arena_winrate/challenge_flow.py",
    "agent/arena_winrate/config_migration.py",
    "agent/arena_winrate/customization_badge.py",
    "agent/arena_winrate/own_cache.py",
    "resource/base/model/embedding/arena_card/arena_card_embedding_model.onnx",
    "resource/base/model/embedding/p_item_reference/p_item_rendered_reference_gallery.npz",
    "assets/arena-winrate/manifest.json",
)

REQUIRED_TASKS = {
    "arena_win_rate_manual_challenge",
    "arena_recalculate_own_score",
}


class BuildError(RuntimeError):
    """Raised when a candidate cannot be built without weakening a gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"JSON is unavailable or invalid: {path}") from error
    if not isinstance(value, dict):
        raise BuildError(f"JSON root must be an object: {path}")
    return value


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            relative = Path(member.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise BuildError(f"archive member escapes destination: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise BuildError(f"archive contains a symbolic link: {member.filename}")
            target = (destination / relative).resolve()
            if os.path.commonpath((str(destination_root), str(target))) != str(destination_root):
                raise BuildError(f"archive member escapes destination: {member.filename}")
        zipped.extractall(destination)


def _payload_root(extracted: Path) -> Path:
    executables = tuple(extracted.rglob("MaaGakumasu.exe"))
    if len(executables) != 1:
        raise BuildError(f"expected exactly one MaaGakumasu.exe, found {len(executables)}")
    return executables[0].parent


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo"))}


def _replace_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise BuildError(f"source tree is missing: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=_copy_ignore)


def _remove_non_runtime_python_entrypoints(site_packages: Path) -> None:
    """Drop wheel-generated CLI launchers that are not imported by the Maa agent.

    Windows launchers can embed the installation profile path that created
    them.  The Maa agent invokes the embedded interpreter and imports modules
    directly, so top-level console entrypoint directories are not runtime
    dependencies and must not enter a public package.
    """

    for directory_name in NON_RUNTIME_SITE_PACKAGE_DIRS:
        directory = site_packages / directory_name
        if directory.is_dir():
            shutil.rmtree(directory)
        elif directory.exists():
            raise BuildError(f"Python non-runtime entrypoint path is not a directory: {directory_name}")


def _normalize_python_dependency_sources(site_packages: Path) -> dict[str, str]:
    """Remove nonfunctional user-profile examples from vendored dependency text."""

    relative = Path("maa/define.py")
    path = site_packages / relative
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        source = stream.read()
    example_pattern = re.compile(
        r'(?m)^(\s*# value: string, eg: ")[A-Za-z]:\\\\Users\\\\[^\\\r\n"]+'
        r'\\\\Desktop\\\\log("; val_size: string length)(\r?)$'
    )
    normalized, replacements = example_pattern.subn(
        lambda match: f'{match.group(1)}.\\\\log{match.group(2)}{match.group(3)}',
        source,
    )
    if replacements > 1:
        raise BuildError(f"Python dependency contains multiple profile examples: {relative.as_posix()}")
    if replacements == 0:
        return {}
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(normalized)
    return {relative.as_posix(): "replace_nonfunctional_user_profile_example_with_relative_log_path"}


def _is_link_or_reparse(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except AttributeError:
        attributes = 0
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _validate_input_tree(root: Path, *, label: str, reject_private_names: bool) -> None:
    if not root.is_dir():
        raise BuildError(f"{label} source is missing: {root}")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if _is_link_or_reparse(path):
            raise BuildError(f"{label} source contains a link or reparse point: {relative.as_posix()}")
        if not reject_private_names or not path.is_file():
            continue
        if path.name.casefold() in PRIVATE_INPUT_NAMES or path.suffix.casefold() in PRIVATE_INPUT_SUFFIXES:
            raise BuildError(f"{label} source contains a private runtime file: {relative.as_posix()}")


def _export_source_revision(source_root: Path, source_revision: str, destination: Path) -> None:
    """Export exactly one committed source tree, never the mutable worktree."""
    archive = destination.parent / "source-revision.zip"
    completed = subprocess.run(
        (
            "git",
            "-C",
            str(source_root),
            "archive",
            "--format=zip",
            f"--output={archive}",
            source_revision,
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    if completed.returncode != 0 or not archive.is_file():
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise BuildError(f"cannot export committed source revision: {detail}")
    destination.mkdir()
    _safe_extract(archive, destination)
    archive.unlink()


def _validate_engine_bundle(bundle: Path) -> dict[str, Any]:
    _validate_input_tree(bundle, label="arena engine", reject_private_names=True)
    manifest_path = bundle / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("project") != "gakumas-tools":
        raise BuildError("arena engine manifest identity is invalid")
    commit = manifest.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise BuildError("arena engine commit is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise BuildError("arena engine manifest has no file inventory")
    bundle_root = bundle.resolve()
    for relative_text, expected_hash in files.items():
        if not isinstance(relative_text, str) or not isinstance(expected_hash, str):
            raise BuildError("arena engine file inventory is invalid")
        relative = Path(relative_text.replace("\\", "/"))
        candidate = (bundle / relative).resolve()
        if os.path.commonpath((str(bundle_root), str(candidate))) != str(bundle_root):
            raise BuildError(f"arena engine path escapes bundle: {relative_text}")
        if not candidate.is_file():
            raise BuildError(f"arena engine file is missing: {relative_text}")
        actual_hash = sha256_file(candidate)
        if actual_hash.casefold() != expected_hash.casefold():
            raise BuildError(f"arena engine hash mismatch: {relative_text}")
    actual_files = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    expected_files = {"manifest.json", *(str(relative).replace("\\", "/") for relative in files)}
    if actual_files != expected_files:
        extra = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise BuildError(f"arena engine file inventory mismatch: extra={extra}, missing={missing}")
    return manifest


def _validate_arena_period_selector(root: Path, interface: dict[str, Any]) -> None:
    """Keep the fixed UI choices identical to the bundled complete seasons."""

    stages_path = (
        root
        / "assets"
        / "arena-winrate"
        / "node_modules"
        / "gakumas-data"
        / "json"
        / "stages.json"
    )
    try:
        rows = json.loads(stages_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError("candidate arena stage catalog is unavailable or invalid") from error
    if not isinstance(rows, list):
        raise BuildError("candidate arena stage catalog must be an array")

    contest_stages: dict[int, dict[int, bool]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("type") != "contest":
            continue
        season = row.get("season")
        stage = row.get("stage")
        preview = row.get("preview")
        if (
            isinstance(season, bool)
            or not isinstance(season, int)
            or season < 1
            or isinstance(stage, bool)
            or not isinstance(stage, int)
            or stage not in {1, 2, 3}
            or not isinstance(preview, bool)
        ):
            raise BuildError("candidate contest stage catalog contains an invalid row")
        stages = contest_stages.setdefault(season, {})
        if stage in stages:
            raise BuildError("candidate contest stage catalog contains duplicate season stages")
        stages[stage] = preview
    incomplete = sorted(season for season, stages in contest_stages.items() if set(stages) != {1, 2, 3})
    if incomplete:
        raise BuildError(f"candidate contest stage catalog has incomplete seasons: {incomplete}")
    complete_seasons = sorted(contest_stages)
    if not complete_seasons:
        raise BuildError("candidate contest stage catalog has no complete season")

    options = interface.get("option")
    period = options.get("竞技场期数") if isinstance(options, dict) else None
    cases = period.get("cases") if isinstance(period, dict) else None
    if period is None or period.get("type") != "select" or not isinstance(cases, list):
        raise BuildError("candidate fixed arena period selector is missing")
    expected_values: list[str | int] = ["latest", *complete_seasons]
    actual_values: list[str | int] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise BuildError("candidate arena period selector contains an invalid case")
        try:
            selection = case["pipeline_override"]["ChallengeChoose"]["custom_action_param"]["season"]
        except (KeyError, TypeError) as error:
            raise BuildError("candidate arena period case has no season override") from error
        actual_values.append(selection)
        expected_name = "latest" if index == 0 else f"season-{selection}"
        if case.get("name") != expected_name:
            raise BuildError("candidate arena period case names or order are unstable")
        if isinstance(selection, int):
            if isinstance(selection, bool) or selection not in contest_stages:
                raise BuildError("candidate arena period selector contains an unsupported season")
            expected_description = (
                "$竞技场预览期说明"
                if any(contest_stages[selection].values())
                else None
            )
            if case.get("description") != expected_description:
                raise BuildError("candidate arena period preview labels differ from the engine catalog")
    if actual_values != expected_values or period.get("default_case") != "latest":
        raise BuildError("candidate arena period selector differs from the engine catalog")


def _validate_candidate(root: Path, *, version: str, update_repository: str) -> dict[str, str]:
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise BuildError(f"candidate is missing required files: {', '.join(missing)}")

    interface = _load_json(root / "interface.json")
    if interface.get("version") != version:
        raise BuildError("candidate interface version mismatch")
    if interface.get("github") != update_repository:
        raise BuildError("candidate update repository mismatch")
    agent = interface.get("agent")
    if not isinstance(agent, dict):
        raise BuildError("candidate Agent configuration is invalid")
    if agent.get("child_exec") != "./python/python.exe" or agent.get("child_args") != ["-u", "./agent/main.py"]:
        raise BuildError("candidate Agent does not use the embedded Python entrypoint")
    tasks = interface.get("task")
    if not isinstance(tasks, list):
        raise BuildError("candidate task list is invalid")
    task_names = {entry.get("name") for entry in tasks if isinstance(entry, dict)}
    missing_tasks = sorted(REQUIRED_TASKS - task_names)
    if missing_tasks:
        raise BuildError(f"candidate is missing arena UI tasks: {', '.join(missing_tasks)}")
    _validate_arena_period_selector(root, interface)

    return {relative: sha256_file(root / relative) for relative in REQUIRED_FILES}


def _validate_python_runtime(root: Path) -> dict[str, str]:
    executable = root / "python" / "python.exe"
    if not executable.is_file():
        raise BuildError("candidate embedded Python is missing")
    script = (
        "import json, sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        f"sys.path.insert(0, {str(root / 'agent')!r}); "
        "import cv2, loguru, maa, numpy, onnxruntime, PIL, yaml; "
        "import agent.arena_winrate.customization_badge; "
        "import agent.arena_winrate.own_cache; "
        "import agent.custom.action.arena_reader; "
        "import agent.p_item_recognition.embedding; "
        "print(json.dumps({'numpy': numpy.__version__, 'onnxruntime': onnxruntime.__version__, "
        "'opencv-python-headless': cv2.__version__, 'pillow': PIL.__version__, 'pyyaml': yaml.__version__}))"
    )
    inventory_before = {path.relative_to(root) for path in root.rglob("*")}
    with tempfile.TemporaryDirectory(prefix="gkh-runtime-smoke-", dir=root.parent) as runtime_text:
        runtime = Path(runtime_text)
        smoke_environment = os.environ.copy()
        for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
            smoke_environment[name] = str(runtime)
        completed = subprocess.run(
            (str(executable), "-B", "-c", script),
            cwd=runtime,
            env=smoke_environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
    inventory_after = {path.relative_to(root) for path in root.rglob("*")}
    if inventory_after != inventory_before:
        created = sorted(str(path) for path in inventory_after - inventory_before)
        removed = sorted(str(path) for path in inventory_before - inventory_after)
        raise BuildError(
            "candidate runtime smoke mutated the release tree: "
            f"created={created[:10]}, removed={removed[:10]}"
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise BuildError(f"candidate embedded Python import smoke failed: {detail}")
    try:
        versions = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BuildError("candidate embedded Python returned invalid version inventory") from error
    if not isinstance(versions, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in versions.items()):
        raise BuildError("candidate embedded Python version inventory is invalid")
    return versions


def build_derived_package(
    *,
    source_root: Path,
    upstream_archive: Path,
    upstream_sha256: str,
    upstream_tag: str,
    source_revision: str,
    engine_bundle: Path,
    python_site_packages: Path,
    output: Path,
    update_repository: str,
    allow_upstream_trial_channel: bool,
    derived_version: str | None = None,
    validate_python_runtime: bool = True,
) -> Path:
    source_root = source_root.resolve()
    upstream_archive = upstream_archive.resolve()
    engine_bundle = engine_bundle.resolve()
    python_site_packages = python_site_packages.resolve()
    output = output.resolve()

    if not upstream_archive.is_file():
        raise BuildError(f"upstream archive is missing: {upstream_archive}")
    if re.fullmatch(r"[0-9A-Fa-f]{64}", upstream_sha256) is None:
        raise BuildError("upstream SHA-256 must contain 64 hexadecimal characters")
    actual_upstream_hash = sha256_file(upstream_archive)
    if actual_upstream_hash.casefold() != upstream_sha256.casefold():
        raise BuildError(
            f"upstream archive SHA-256 mismatch: actual={actual_upstream_hash}, expected={upstream_sha256.upper()}"
        )
    if re.fullmatch(r"v\d+\.\d+\.\d+", upstream_tag) is None:
        raise BuildError("upstream tag must use vMAJOR.MINOR.PATCH")
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise BuildError("source revision must be a full lowercase 40-character Git SHA")
    if GITHUB_REPOSITORY_PATTERN.fullmatch(update_repository) is None:
        raise BuildError("update repository must be a GitHub repository URL")
    update_repository = update_repository.rstrip("/")
    if update_repository == UPSTREAM_REPOSITORY and not allow_upstream_trial_channel:
        raise BuildError(
            "the upstream repository is only allowed for an equal-precedence local trial; "
            "a durable build must use the derived release repository"
        )
    if output.exists():
        raise BuildError(f"output already exists: {output}")

    if update_repository == UPSTREAM_REPOSITORY:
        if derived_version is not None:
            raise BuildError("an explicit derived release version cannot use the upstream update repository")
        derived_version = f"{upstream_tag}+gkh.{source_revision[:7]}"
        if LOCAL_TRIAL_VERSION_PATTERN.fullmatch(derived_version) is None:
            raise BuildError("local trial version is invalid")
    else:
        if derived_version is None:
            raise BuildError("a durable derived release requires an explicit monotonic version")
        if (
            DERIVED_RELEASE_VERSION_PATTERN.fullmatch(derived_version) is None
            and DATED_PREVIEW_VERSION_PATTERN.fullmatch(derived_version) is None
        ):
            raise BuildError(
                "derived release version must use vMAJOR.MINOR.PATCH, vMAJOR.MINOR.PATCH-beta.N, "
                "vMAJOR.MINOR.PATCH-alpha.N, or the approved preview form vMAJOR.MINOR.PATCH+gkh.YYMMDD"
            )

    engine_manifest = _validate_engine_bundle(engine_bundle)
    _validate_input_tree(python_site_packages, label="embedded Python site-packages", reject_private_names=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}-staging-{uuid.uuid4().hex}"
    staging.mkdir()
    extracted = staging / "upstream"
    source_snapshot = staging / "source"
    candidate = staging / "candidate"
    try:
        _export_source_revision(source_root, source_revision, source_snapshot)
        extracted.mkdir()
        _safe_extract(upstream_archive, extracted)
        payload = _payload_root(extracted)
        upstream_interface = _load_json(payload / "interface.json")
        if upstream_interface.get("version") != upstream_tag:
            raise BuildError(
                f"upstream package version mismatch: {upstream_interface.get('version')!r} != {upstream_tag!r}"
            )
        shutil.copytree(payload, candidate)

        # A release archive must never inherit logs, caches, instance settings,
        # or other mutable state accidentally packed by an upstream build.
        # The installer migrates the current user's state after validation.
        for relative in MUTABLE_RUNTIME_PATHS:
            mutable = candidate / relative
            if mutable.is_dir():
                shutil.rmtree(mutable)
            elif mutable.exists():
                mutable.unlink()

        for source_relative, target_relative in REPLACED_TREES.items():
            _replace_tree(source_snapshot / source_relative, candidate / target_relative)

        for file_name in ROOT_FILES:
            source_file = source_snapshot / file_name
            if source_file.is_file():
                shutil.copy2(source_file, candidate / file_name)

        shutil.copy2(source_snapshot / "assets" / "interface.json", candidate / "interface.json")
        interface = _load_json(candidate / "interface.json")
        interface["version"] = derived_version
        interface["github"] = update_repository
        agent = interface.get("agent")
        if not isinstance(agent, dict):
            raise BuildError("source interface Agent configuration is invalid")
        agent["child_exec"] = "./python/python.exe"
        agent["child_args"] = ["-u", "./agent/main.py"]
        (candidate / "interface.json").write_text(
            json.dumps(interface, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        engine_target = candidate / "assets" / "arena-winrate"
        engine_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(engine_bundle, engine_target)

        python_target = candidate / "python" / "Lib" / "site-packages"
        _replace_tree(python_site_packages, python_target)
        _remove_non_runtime_python_entrypoints(python_target)
        python_source_normalizations = _normalize_python_dependency_sources(python_target)

        critical_hashes = _validate_candidate(
            candidate,
            version=derived_version,
            update_repository=update_repository,
        )
        python_packages = _validate_python_runtime(candidate) if validate_python_runtime else {}
        update_mode = (
            "upstream_equal_precedence_trial"
            if update_repository == UPSTREAM_REPOSITORY
            else "derived_release_channel"
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "product": "MaaGakumasu",
            "derived_version": derived_version,
            "upstream": {
                "repository": UPSTREAM_REPOSITORY,
                "tag": upstream_tag,
                "archive": upstream_archive.name,
                "sha256": actual_upstream_hash,
            },
            "source": {
                "repository": update_repository,
                "revision": source_revision,
            },
            "arena_engine": {
                "commit": engine_manifest["commit"],
                "manifest_sha256": sha256_file(engine_bundle / "manifest.json"),
            },
            "python_packages": python_packages,
            "python_source_normalizations": python_source_normalizations,
            "update_contract": {
                "mode": update_mode,
                "repository": update_repository,
                "release_channel": (
                    "alpha"
                    if "-alpha." in derived_version
                    else "beta"
                    if "-beta." in derived_version or DATED_PREVIEW_VERSION_PATTERN.fullmatch(derived_version)
                    else "stable"
                ),
                "version_ordering": (
                    "upstream_equal_precedence_build_metadata"
                    if update_mode == "upstream_equal_precedence_trial"
                    else "same_base_build_metadata_manual_bootstrap"
                    if DATED_PREVIEW_VERSION_PATTERN.fullmatch(derived_version)
                    else "monotonic_semver_precedence"
                ),
                "requires_manual_bootstrap": bool(DATED_PREVIEW_VERSION_PATTERN.fullmatch(derived_version)),
                "auto_update_setting": "preserve_user_configuration",
                "safe_upstream_precedence": upstream_tag,
                "requires_derived_release_before_newer_upstream": update_mode
                == "upstream_equal_precedence_trial",
            },
            "critical_files": critical_hashes,
        }
        (candidate / "GAKUMAS_HELPER_BUILD.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        os.replace(candidate, output)
        return output
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--upstream-archive", type=Path, required=True)
    parser.add_argument("--upstream-sha256", required=True)
    parser.add_argument("--upstream-tag", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--engine-bundle", type=Path, required=True)
    parser.add_argument("--python-site-packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-repository", default=UPSTREAM_REPOSITORY)
    parser.add_argument("--allow-upstream-trial-channel", action="store_true")
    parser.add_argument("--derived-version")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_derived_package(
        source_root=args.source_root,
        upstream_archive=args.upstream_archive,
        upstream_sha256=args.upstream_sha256,
        upstream_tag=args.upstream_tag,
        source_revision=args.source_revision,
        engine_bundle=args.engine_bundle,
        python_site_packages=args.python_site_packages,
        output=args.output,
        update_repository=args.update_repository,
        allow_upstream_trial_channel=args.allow_upstream_trial_channel,
        derived_version=args.derived_version,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
