"""Build a versioned MaaGakumasu package from a clean upstream release.

The builder never edits an installed Maa directory.  It verifies the official
release archive, extracts it into a staging directory, replaces resource code
with this repository's integrated source, validates the pinned arena engine,
and only then publishes the candidate directory atomically.
"""

from __future__ import annotations

import os
import re
import csv
import json
import stat
import base64
import shutil
import hashlib
import zipfile
import argparse
import tempfile
import subprocess
from typing import Any
from pathlib import Path
from email.parser import Parser

SCHEMA_VERSION = 1
UPSTREAM_REPOSITORY = "https://github.com/SuperWaterGod/MaaGakumasu"
FRAMEWORK_REPOSITORY = "https://github.com/MaaXYZ/MaaFramework"
FRAMEWORK_NATIVE_PATH = "runtimes/win-x64/native"
FRAMEWORK_PYTHON_NATIVE_PATH = "python/Lib/site-packages/maa/bin"
FRAMEWORK_NOTICE_PATH = "THIRD_PARTY_NOTICES/MaaFramework-LICENSE"
FRAMEWORK_AGENT_PATHS = ("MaaAgentBinary", "libs/MaaAgentBinary", "share/MaaAgentBinary")
FRAMEWORK_REQUIRED_DLLS = ("MaaFramework.dll", "MaaAgentClient.dll", "MaaAgentServer.dll")
MFA_CORE_BUNDLE_SCHEMA_VERSION = 1
MFA_CORE_COMPONENT = "mfaavalonia_core"
MFA_CORE_STATUS = "READY"
MFA_CORE_BUNDLE_PAYLOAD_PATH = "MFAAvalonia.Core.dll"
MFA_CORE_INSTALL_PATH = "libs/MFAAvalonia.Core.dll"
MFA_CORE_NOTICE_PATH = "THIRD_PARTY_NOTICES/MFAAvalonia-LICENSE"
MFA_CORE_PATCH_PATH = (
    "tools/deployment/mfa/MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
)
MFA_CORE_UPSTREAM = {
    "repository": "https://github.com/MaaXYZ/MFAAvalonia",
    "tag": "v2.15.2",
    "commit": "6065fe33798b72906c5079fa6f210646801d9a5c",
    "source_archive": "MFAAvalonia-2.15.2.zip",
    "source_archive_sha256": "76DD02AFE4B1529B1D4F6416B3442E1BD7E64F13B72D8A3B428F955B5E26F67A",
    "version_checker_git_blob_sha1": "6e6d1118fa414ba21c7efa4f15a58ad95dd08bd7",
}
MFA_CORE_SDK_VERSION_COMPONENTS = (10, 0, 400)
MFA_CORE_SDK_VERSION = ".".join(str(part) for part in MFA_CORE_SDK_VERSION_COMPONENTS)
MFA_CORE_BUILD = {
    "sdk_version": MFA_CORE_SDK_VERSION,
    "sdk_archive": f"dotnet-sdk-{MFA_CORE_SDK_VERSION}-win-x64.zip",
    "sdk_archive_sha512": (
        "9B8B88590E4DA131BFD0DA7AA089D0FC04D5418D5F8607EC13D55DC5A17B4399"
        "AFD54D496C12657FA05C6C6546DC5EAB930F26AC6C50F2D3A7712C0FB378C366"
    ),
    "configuration": "Release",
    "runtime": "win-x64",
    "project": "MFAAvalonia/MFAAvalonia.csproj",
    "source_revision_id": "6065fe33798b72906c5079fa6f210646801d9a5c",
    "sentry_project_directory": "/_/MFAAvalonia/",
    "compiler_path_map_target": "/_/MFAAvalonia/",
    "project_directory_override_scope": "WriteSentryAttributes",
    "continuous_integration_build": True,
    "deterministic": True,
    "incremental_build": False,
    "max_cpu_count": 1,
    "sentry_cli": False,
}
# Avoid serializing the SDK's dotted version in public project text, where the
# privacy gate correctly treats dotted numeric tokens as possible private IPv4
# data.  The exact archive remains bound by its SHA-512 and bundle-manifest hash.
MFA_CORE_PACKAGED_BUILD = {
    "sdk_version_components": list(MFA_CORE_SDK_VERSION_COMPONENTS),
    **{
        key: value
        for key, value in MFA_CORE_BUILD.items()
        if key not in {"sdk_version", "sdk_archive"}
    },
}
MFA_CORE_INPUT = {
    "baseline_dll_sha256": "2DF2226CE45FCE8AF0C4DF8022F399B0FA74E8378C9B23C4177BAC5EE6C295C8"
}
MFA_CORE_LICENSE = {
    "spdx": "GPL-3.0-only",
    "upstream_file": "LICENSE",
    "sha256": "3972DC9744F6499F0F9B2DBF76696F2AE7AD8AF9B23DDE66D6AF86C9DFB36986",
}
MFA_CORE_PATCH_SCOPE = [
    "resource_update_check",
    "resource_update_apply",
    "deterministic_build_path",
    "diagnostic_log_export",
    "dynamic_option_cases",
    "startup_template_and_saved_layout",
]
FORBIDDEN_DERIVED_UPDATE_KEYS = frozenset(
    {"mirrorchyan_rid", "mirrorchyan_multiplatform"}
)
LOCAL_TRIAL_VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+\+gkh\.[0-9a-f]{7,40}$")
PROJECT_RELEASE_VERSION_PATTERN = re.compile(
    r"^v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$"
)
LEGACY_DERIVED_VERSION_PATTERN = re.compile(
    r"^v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"\+gkh\.(?:[0-9a-f]{7,40}|\d{6})$"
)
MINIMUM_INDEPENDENT_PROJECT_VERSION = "v0.1.0"
FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION = "v0.1.1"
RESERVED_UNPUBLISHABLE_PROJECT_VERSIONS = frozenset({"v0.1.0"})
LAST_PUBLISHED_LEGACY_CHANNEL_VERSION = "v1.4.9+gkh.260823"
CHANNEL_VERSION_PATTERN = re.compile(
    r"^v(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>alpha|beta)\.(?P<prerelease_number>0|[1-9]\d*)"
    r"|\+gkh\.(?:[0-9a-f]{7,40}|\d{6}))?$"
)
GITHUB_REPOSITORY_PATTERN = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+/?$")

REPLACED_TREES = {
    "agent": "agent",
    "assets/resource": "resource",
    "assets/tasks": "tasks",
    "assets/lang": "lang",
    "assets/data": "data",
    "THIRD_PARTY_NOTICES": "THIRD_PARTY_NOTICES",
}

ROOT_FILE_SOURCES = {
    "config.template.json": "config.template.json",
    "README.md": "README.md",
    "LICENSE": "LICENSE",
    "logo.ico": "logo.ico",
    "requirements.txt": "requirements.txt",
    "ASSET_PROVENANCE.md": "tools/deployment/public/ASSET_PROVENANCE.md",
}
ROOT_FILES = tuple(ROOT_FILE_SOURCES)
INSTALL_PAYLOAD_FILES = {
    "tools/deployment/Start-MaaGakumasu-Admin.cmd": "deployment/Start-MaaGakumasu-Admin.cmd",
    "tools/deployment/.Start-MaaGakumasu-Admin.ps1": "deployment/.Start-MaaGakumasu-Admin.ps1",
}
MUTABLE_RUNTIME_PATHS = (".local", "config", "appsettings.json")
PRIVATE_INPUT_NAMES = {".env", ".env.local", ".netrc", ".pypirc", "credentials.json", "secrets.json"}
PRIVATE_INPUT_SUFFIXES = {".log", ".dmp", ".dump", ".tmp", ".bak", ".sqlite", ".sqlite3", ".db"}
NON_RUNTIME_SITE_PACKAGE_DIRS = ("bin", "Scripts")

REQUIRED_FILES = (
    "config.template.json",
    "MaaGakumasu.exe",
    MFA_CORE_INSTALL_PATH,
    MFA_CORE_NOTICE_PATH,
    "interface.json",
    "ASSET_PROVENANCE.md",
    "agent/main.py",
    "agent/arena_winrate/catalog.py",
    "agent/custom/action/arena_reader.py",
    "agent/arena_winrate/challenge_flow.py",
    "agent/arena_winrate/config_migration.py",
    "agent/arena_winrate/customization_badge.py",
    "agent/arena_winrate/own_cache.py",
    "resource/base/model/embedding/arena_card/arena_card_embedding_model.onnx",
    "resource/base/model/embedding/p_item_reference/p_item_rendered_reference_gallery.npz",
    "assets/arena-winrate/manifest.json",
    "deployment/Start-MaaGakumasu-Admin.cmd",
    "deployment/.Start-MaaGakumasu-Admin.ps1",
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


def _tree_snapshot(root: Path) -> dict[str, tuple[str, int | None, str | None]]:
    """Record path identity and content so runtime smoke must be read-only."""

    snapshot: dict[str, tuple[str, int | None, str | None]] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if _is_link_or_reparse(path):
            snapshot[relative] = ("link_or_reparse", metadata.st_size, None)
        elif stat.S_ISDIR(metadata.st_mode):
            snapshot[relative] = ("directory", None, None)
        elif stat.S_ISREG(metadata.st_mode):
            snapshot[relative] = ("file", metadata.st_size, sha256_file(path))
        else:
            snapshot[relative] = ("other", metadata.st_size, None)
    return snapshot


def _semver_precedence(version: str) -> tuple[int, int, int, int, int]:
    """Return supported SemVer precedence while intentionally ignoring build metadata."""

    match = CHANNEL_VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise BuildError(f"previous channel version is invalid: {version!r}")
    prerelease = match.group("prerelease")
    prerelease_rank = {"alpha": 0, "beta": 1, None: 2}[prerelease]
    prerelease_number = int(match.group("prerelease_number") or 0)
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        prerelease_rank,
        prerelease_number,
    )


def _durable_update_semantics(
    derived_version: str,
    previous_channel_version: str | None,
) -> tuple[str, bool]:
    if previous_channel_version is None:
        raise BuildError("a project release requires an explicit previous channel version")
    if _semver_precedence(derived_version) < _semver_precedence(
        MINIMUM_INDEPENDENT_PROJECT_VERSION
    ):
        raise BuildError(
            "project release version must not precede the minimum independent GKH version "
            f"{MINIMUM_INDEPENDENT_PROJECT_VERSION}"
        )
    if derived_version in RESERVED_UNPUBLISHABLE_PROJECT_VERSIONS:
        raise BuildError(
            f"project release version {derived_version} is permanently reserved after a "
            "failed pre-publication install and must not be rebuilt or published; use at least "
            f"{FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION}"
        )
    if LEGACY_DERIVED_VERSION_PATTERN.fullmatch(previous_channel_version) is not None:
        if previous_channel_version != LAST_PUBLISHED_LEGACY_CHANNEL_VERSION:
            raise BuildError(
                "legacy channel migration must start from the last published legacy release "
                f"{LAST_PUBLISHED_LEGACY_CHANNEL_VERSION}"
            )
        if derived_version != FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION:
            raise BuildError(
                "legacy channel migration is allowed only for the first publishable independent "
                f"GKH version {FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION}"
            )
        return "legacy_combined_to_independent_semver_manual_bootstrap", True
    if PROJECT_RELEASE_VERSION_PATTERN.fullmatch(previous_channel_version) is None:
        raise BuildError(
            "a later project release requires a previous independent GKH SemVer version"
        )
    if previous_channel_version in RESERVED_UNPUBLISHABLE_PROJECT_VERSIONS:
        raise BuildError(
            f"previous channel version {previous_channel_version} is permanently reserved and "
            "must not appear in public release lineage"
        )
    if _semver_precedence(previous_channel_version) < _semver_precedence(
        FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION
    ):
        raise BuildError(
            "previous independent GKH version must not precede the first publishable version "
            f"{FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION}"
        )
    current_precedence = _semver_precedence(derived_version)
    previous_precedence = _semver_precedence(previous_channel_version)
    if current_precedence < previous_precedence:
        raise BuildError("derived release version precedes the previous channel version")
    if current_precedence == previous_precedence:
        raise BuildError("derived release version does not advance the previous channel version")
    return "independent_gkh_semver_precedence", False


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


def _prune_dependency_tests(site_packages: Path) -> None:
    """Keep public testing helpers, headers and metadata; omit only test suites."""
    directories = [site_packages / "colorama/tests"]
    numpy = site_packages / "numpy"
    if numpy.is_dir():
        directories.extend(path for path in numpy.rglob("tests") if path.is_dir())
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            if path.is_symlink() or not path.resolve().is_relative_to(site_packages.resolve()):
                raise BuildError("dependency test directory escaped site-packages")
            shutil.rmtree(path)


def _prune_unused_video_backend(site_packages: Path) -> None:
    """The agent processes still images; retain codecs/models, omit the pinned video plugin."""
    metadata = site_packages / "opencv_python_headless-5.0.0.93.dist-info/METADATA"
    plugin = site_packages / "cv2/opencv_videoio_ffmpeg500_64.dll"
    if metadata.is_file() and plugin.is_file():
        package = Parser().parsestr(metadata.read_text(encoding="utf-8"))
        if package.get("Name", "").replace("_", "-").lower() == "opencv-python-headless" and package.get("Version") == "5.0.0.93":
            plugin.unlink()


def _remove_obsolete_root_agent_tools(candidate: Path, framework: dict[str, Any]) -> None:
    """MFA resolves libs/MaaAgentBinary; preserve Python's separate default path."""
    legacy = candidate / "MaaAgentBinary"
    active = candidate / "libs/MaaAgentBinary"
    for path in legacy.rglob("*"):
        if not path.is_file() or path.name.lower() in {"license", "license.md", "readme", "readme.md"}:
            continue
        counterpart = active / path.relative_to(legacy)
        if counterpart.is_file() and sha256_file(path) == sha256_file(counterpart):
            path.unlink()
            framework["files"].pop(path.relative_to(candidate).as_posix(), None)


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
    records = tuple(
        record for record in site_packages.glob("*.dist-info/RECORD")
        if record.parent.name.casefold().startswith("maafw-")
    )
    if len(records) != 1:
        raise BuildError("Python source normalization requires exactly one maafw RECORD")
    record = records[0]
    record_relative = record.relative_to(site_packages).as_posix()
    with record.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if any(len(row) != 3 for row in rows):
        raise BuildError("Python source normalization found a malformed maafw RECORD")
    source_rows = [row for row in rows if row[0] == relative.as_posix()]
    record_rows = [row for row in rows if row[0] == record_relative]
    original_bytes = path.read_bytes()
    original_hash = base64.urlsafe_b64encode(hashlib.sha256(original_bytes).digest()).rstrip(b"=").decode("ascii")
    if (
        len(source_rows) != 1
        or source_rows[0][1:] != [f"sha256={original_hash}", str(len(original_bytes))]
        or len(record_rows) != 1
        or record_rows[0][1:] != ["", ""]
    ):
        raise BuildError("Python source normalization maafw RECORD does not match the original source")
    normalized_bytes = normalized.encode("utf-8")
    normalized_hash = base64.urlsafe_b64encode(hashlib.sha256(normalized_bytes).digest()).rstrip(b"=").decode("ascii")
    source_rows[0][1:] = [f"sha256={normalized_hash}", str(len(normalized_bytes))]
    path.write_bytes(normalized_bytes)
    with record.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return {
        relative.as_posix(): "replace_nonfunctional_user_profile_example_with_relative_log_path",
        record_relative: "refresh_normalized_source_hash_and_size",
    }


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
    # Keep Windows payload text and component-manifest hashes independent of
    # the Git configuration inherited from the build host.
    completed = subprocess.run(
        (
            "git",
            "-c",
            "core.autocrlf=true",
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


def _require_exact_object(value: object, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise BuildError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _validate_mfa_core_bundle(bundle: Path) -> dict[str, Any]:
    _validate_input_tree(bundle, label="MFA Core", reject_private_names=True)
    expected_entries = {"manifest.json", MFA_CORE_BUNDLE_PAYLOAD_PATH}
    actual_entries = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
    }
    if actual_entries != expected_entries:
        raise BuildError(
            "MFA Core bundle inventory mismatch: "
            f"extra={sorted(actual_entries - expected_entries)}, "
            f"missing={sorted(expected_entries - actual_entries)}"
        )

    manifest_path = bundle / "manifest.json"
    manifest = _load_json(manifest_path)
    _require_exact_object(
        manifest,
        {
            "schema_version",
            "component",
            "status",
            "upstream",
            "patch",
            "build",
            "input",
            "payload",
            "license",
        },
        label="MFA Core manifest",
    )
    if (
        manifest.get("schema_version") != MFA_CORE_BUNDLE_SCHEMA_VERSION
        or manifest.get("component") != MFA_CORE_COMPONENT
        or manifest.get("status") != MFA_CORE_STATUS
    ):
        raise BuildError("MFA Core manifest identity is invalid")

    upstream = _require_exact_object(
        manifest.get("upstream"), set(MFA_CORE_UPSTREAM), label="MFA Core upstream provenance"
    )
    if upstream != MFA_CORE_UPSTREAM:
        raise BuildError("MFA Core upstream provenance is not the fixed official v2.15.2 source")

    patch = _require_exact_object(
        manifest.get("patch"), {"path", "sha256", "scope"}, label="MFA Core patch provenance"
    )
    patch_sha256 = patch.get("sha256")
    if (
        patch.get("path") != MFA_CORE_PATCH_PATH
        or patch.get("scope") != MFA_CORE_PATCH_SCOPE
        or not isinstance(patch_sha256, str)
        or re.fullmatch(r"[0-9A-Fa-f]{64}", patch_sha256) is None
    ):
        raise BuildError("MFA Core patch provenance is invalid")

    build = _require_exact_object(
        manifest.get("build"), set(MFA_CORE_BUILD), label="MFA Core build provenance"
    )
    if build != MFA_CORE_BUILD:
        raise BuildError("MFA Core build provenance is not the fixed v2.15.2 build")

    input_provenance = _require_exact_object(
        manifest.get("input"), set(MFA_CORE_INPUT), label="MFA Core input provenance"
    )
    if input_provenance != MFA_CORE_INPUT:
        raise BuildError("MFA Core input provenance is invalid")

    payload = _require_exact_object(
        manifest.get("payload"), {"path", "sha256", "size"}, label="MFA Core payload"
    )
    payload_sha256 = payload.get("sha256")
    payload_size = payload.get("size")
    if (
        payload.get("path") != MFA_CORE_BUNDLE_PAYLOAD_PATH
        or not isinstance(payload_sha256, str)
        or re.fullmatch(r"[0-9A-Fa-f]{64}", payload_sha256) is None
        or type(payload_size) is not int
        or payload_size <= 0
    ):
        raise BuildError("MFA Core payload declaration is invalid")
    payload_path = bundle / MFA_CORE_BUNDLE_PAYLOAD_PATH
    if payload_path.stat().st_size != payload_size:
        raise BuildError("MFA Core payload size mismatch")
    if sha256_file(payload_path).casefold() != payload_sha256.casefold():
        raise BuildError("MFA Core payload SHA-256 mismatch")

    license_provenance = _require_exact_object(
        manifest.get("license"), set(MFA_CORE_LICENSE), label="MFA Core license provenance"
    )
    if license_provenance != MFA_CORE_LICENSE:
        raise BuildError("MFA Core license provenance is invalid")
    return manifest


def _validate_mfa_core_patch_source(source_snapshot: Path, manifest: dict[str, Any]) -> None:
    patch_path = source_snapshot / MFA_CORE_PATCH_PATH
    if not patch_path.is_file() or _is_link_or_reparse(patch_path):
        raise BuildError(f"MFA Core patch source is missing: {MFA_CORE_PATCH_PATH}")
    expected_hash = manifest["patch"]["sha256"]
    if sha256_file(patch_path).casefold() != expected_hash.casefold():
        raise BuildError("MFA Core patch source SHA-256 mismatch")


def _load_contest_stage_catalog(stages_path: Path) -> dict[int, dict[int, bool]]:
    """Load complete three-stage contest seasons from one fixed engine tree."""

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
    return contest_stages


def _expected_arena_period_cases(
    contest_stages: dict[int, dict[int, bool]],
) -> list[dict[str, Any]]:
    latest_season = max(contest_stages)
    recent_formal = max(
        (season for season, flags in contest_stages.items() if not any(flags.values())),
        default=None,
    ) if any(contest_stages[latest_season].values()) else None

    def make_case(selection: str | int, *, season: int) -> dict[str, Any]:
        preview = any(contest_stages[season].values())
        if selection == "latest":
            label_key = "竞技场最新预览期格式" if preview else "竞技场最新期格式"
        elif preview:
            label_key = "竞技场固定预览期格式"
        elif season == recent_formal:
            label_key = "竞技场最近正式期格式"
        else:
            label_key = "竞技场固定期格式"
        case: dict[str, Any] = {
            "name": "latest" if selection == "latest" else f"season-{selection}",
            "label": f"${label_key}",
            "label_args": {"season": str(season)},
            "pipeline_override": {
                "ChallengeSeasonConfig": {"attach": {"season": selection}}
            },
        }
        if preview:
            case["description"] = "$竞技场预览期说明"
        if selection == "latest":
            case["replacement_case"] = f"season-{latest_season}"
        return case

    return [
        make_case("latest", season=latest_season),
        *(
            make_case(season, season=season)
            for season in sorted(contest_stages, reverse=True)
            if season != latest_season
        ),
        # Keep previous indices; the hidden legacy choice resolves to this case.
        make_case(latest_season, season=latest_season),
    ]


def _synchronize_arena_period_selector(
    root: Path,
    interface: dict[str, Any],
    *,
    stages_path: Path,
) -> int:
    """Generate the release selector and latest label from the fixed catalog."""

    contest_stages = _load_contest_stage_catalog(stages_path)
    options = interface.get("option")
    period = options.get("竞技场期数") if isinstance(options, dict) else None
    if period is None or period.get("type") != "select":
        raise BuildError("candidate fixed arena period selector is missing")
    expected = {case["name"]: case for case in _expected_arena_period_cases(contest_stages)}
    existing = period.get("cases", [])
    names = [case.get("name") for case in existing if isinstance(case, dict)]
    if (len(names) != len(existing) or not names or names[0] != "latest"
            or len(set(names)) != len(names) or any(name not in expected for name in names)):
        raise BuildError("candidate arena period selector cannot preserve existing indices")
    # Runtime RIS updates append missing seasons in ascending order. Preserve that
    # same order in release packages, whose updater does not migrate saved indices.
    cases = [expected.pop(name) for name in names]
    cases.extend(sorted(expected.values(), key=lambda case: int(case["name"].removeprefix("season-"))))
    period["default_case"] = f"season-{max(contest_stages)}"
    period["dynamic_cases"] = True
    period["cases"] = cases

    latest_season = max(contest_stages)
    latest_labels = {
        "zh-CN.json": f"最新（第 {latest_season} 期" + ("，预览）" if any(contest_stages[latest_season].values()) else "）"),
        "zh-Hant.json": f"最新（第 {latest_season} 期" + ("，預覽）" if any(contest_stages[latest_season].values()) else "）"),
    }
    for file_name, label in latest_labels.items():
        language_path = root / "lang" / file_name
        language = _load_json(language_path)
        language["竞技场目录最新"] = label
        language_path.write_text(
            json.dumps(language, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return latest_season


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
    contest_stages = _load_contest_stage_catalog(stages_path)

    options = interface.get("option")
    period = options.get("竞技场期数") if isinstance(options, dict) else None
    cases = period.get("cases") if isinstance(period, dict) else None
    if period is None or period.get("type") != "select" or not isinstance(cases, list):
        raise BuildError("candidate fixed arena period selector is missing")
    expected_cases = _expected_arena_period_cases(contest_stages)
    if (not all(isinstance(case, dict) for case in cases)
            or len(cases) != len(expected_cases)
            or not cases or cases[0].get("name") != "latest"
            or {case.get("name"): case for case in cases}
            != {case["name"]: case for case in expected_cases}
            or period.get("default_case") != f"season-{max(contest_stages)}"
            or period.get("dynamic_cases") is not True):
        raise BuildError("candidate arena period selector differs from the engine catalog")

    latest_season = max(contest_stages)
    expected_labels = {
        "zh-CN.json": f"最新（第 {latest_season} 期" + ("，预览）" if any(contest_stages[latest_season].values()) else "）"),
        "zh-Hant.json": f"最新（第 {latest_season} 期" + ("，預覽）" if any(contest_stages[latest_season].values()) else "）"),
    }
    for file_name, expected_label in expected_labels.items():
        language = _load_json(root / "lang" / file_name)
        if language.get("竞技场目录最新") != expected_label:
            raise BuildError("candidate arena latest-period label differs from the engine catalog")


def _validate_candidate(root: Path, *, version: str, update_repository: str) -> dict[str, str]:
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise BuildError(f"candidate is missing required files: {', '.join(missing)}")

    interface = _load_json(root / "interface.json")
    if interface.get("version") != version:
        raise BuildError("candidate interface version mismatch")
    if interface.get("github") != update_repository:
        raise BuildError("candidate update repository mismatch")
    forbidden_update_keys = sorted(FORBIDDEN_DERIVED_UPDATE_KEYS & interface.keys())
    if forbidden_update_keys:
        raise BuildError(
            "candidate interface retains forbidden upstream update routes: "
            + ", ".join(forbidden_update_keys)
        )
    notice_path = root / MFA_CORE_NOTICE_PATH
    if sha256_file(notice_path).casefold() != MFA_CORE_LICENSE["sha256"].casefold():
        raise BuildError("candidate MFAAvalonia license notice SHA-256 mismatch")
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


def _required_maafw_version(root: Path) -> str:
    try:
        requirements = (root / "requirements.txt").read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise BuildError("candidate requirements.txt is unavailable") from error
    versions = []
    for line in requirements:
        requirement = line.partition("#")[0].strip()
        if re.match(r"maafw(?:\s|[=<>!~;\[]|$)", requirement, flags=re.IGNORECASE):
            match = re.fullmatch(r"maafw\s*==\s*(\d+\.\d+\.\d+)", requirement, flags=re.IGNORECASE)
            if match is None:
                raise BuildError("candidate requirements.txt must pin exactly one maafw version")
            versions.append(match.group(1))
    if len(versions) != 1:
        raise BuildError("candidate requirements.txt must pin exactly one maafw version")
    return versions[0]


def _framework_archive_hash(
    archive: Path | None, expected_hash: str | None, version: str | None,
) -> str | None:
    if archive is None and expected_hash is None and version is None:
        return None
    if archive is None or expected_hash is None or version is None:
        raise BuildError("framework archive, SHA-256 and version must be supplied together")
    if re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        raise BuildError("framework version must use MAJOR.MINOR.PATCH")
    if archive.name != f"MAA-win-x86_64-v{version}.zip":
        raise BuildError("framework archive must name the matching official Windows x64 release")
    if not archive.is_file() or re.fullmatch(r"[0-9A-Fa-f]{64}", expected_hash) is None:
        raise BuildError("framework archive or SHA-256 is invalid")
    actual_hash = sha256_file(archive)
    if actual_hash.casefold() != expected_hash.casefold():
        raise BuildError("framework archive SHA-256 mismatch")
    return actual_hash


def _paired_runtime_files(candidate: Path, *, shared: bool = False) -> dict[str, str]:
    version = _required_maafw_version(candidate)
    site_packages = candidate / "python/Lib/site-packages"
    metadata_files = tuple(
        path for path in site_packages.glob("*.dist-info/METADATA")
        if path.parent.name.casefold().startswith("maafw-")
    )
    if len(metadata_files) != 1:
        raise BuildError("candidate requires exactly one maafw package metadata file")
    package = Parser().parsestr(metadata_files[0].read_text(encoding="utf-8"))
    if package.get("Name", "").casefold() != "maafw" or package.get("Version") != version:
        raise BuildError("candidate maafw package and requirements version mismatch")
    native = candidate / FRAMEWORK_NATIVE_PATH
    python_native = candidate / FRAMEWORK_PYTHON_NATIVE_PATH
    for name in FRAMEWORK_REQUIRED_DLLS:
        if not all((root / name).is_file() for root in ((native,) if shared else (native, python_native))):
            raise BuildError(f"candidate is missing a paired framework DLL: {name}")
        if not shared and sha256_file(native / name) != sha256_file(python_native / name):
            raise BuildError(f"candidate host/Python native payload mismatch: {name}")
    # Existing upstream bundles can use different control-component builds.
    # Bind every file, but compare only the three core IPC libraries here.
    # An explicit SDK overlay separately compares its entire Python payload.
    paths = [candidate / "requirements.txt", metadata_files[0]]
    if shared:
        for path in python_native.rglob("*"):
            if path.is_file() and (native / path.relative_to(python_native)).is_file():
                raise BuildError("shared framework candidate still contains a duplicate Python native file")
        paths.append(metadata_files[0].parent / "RECORD")
    paths.extend(path for root in (native, python_native) for path in root.rglob("*") if path.is_file())
    return {path.relative_to(candidate).as_posix(): sha256_file(path) for path in paths}


def _share_framework_runtime(candidate: Path, framework: dict[str, Any]) -> None:
    """Remove only byte-identical native copies after the SDK overlay validated them."""
    bootstrap = candidate / "agent/main.py"
    if not bootstrap.is_file() or "def configure_maafw_binary_path(" not in bootstrap.read_text(encoding="utf-8-sig"):
        raise BuildError("shared framework layout requires the matching agent bootstrap")
    native = candidate / FRAMEWORK_NATIVE_PATH
    python_native = candidate / FRAMEWORK_PYTHON_NATIVE_PATH
    duplicates = []
    for path in python_native.rglob("*"):
        if not path.is_file():
            continue
        host = native / path.relative_to(python_native)
        if not host.exists():
            host.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, host)
            framework["files"][host.relative_to(candidate).as_posix()] = sha256_file(host)
        if host.is_file():
            if sha256_file(host) != sha256_file(path):
                raise BuildError(f"cannot share differing framework payload: {path.name}")
            duplicates.append(path)
    for path in duplicates:
        path.unlink()
        framework["files"].pop(path.relative_to(candidate).as_posix(), None)
    framework["native_layout"] = "shared"
    framework["files"].update(_paired_runtime_files(candidate, shared=True))


def _overlay_framework(
    candidate: Path, sdk: Path, *, archive: Path, archive_hash: str, version: str,
) -> dict[str, Any]:
    """Update the existing host layout and bind the Python/native pairing."""

    if _required_maafw_version(candidate) != version:
        raise BuildError("framework version differs from the candidate maafw pin")
    site_packages = candidate / "python/Lib/site-packages"
    metadata_files = tuple(
        path for path in site_packages.glob("*.dist-info/METADATA")
        if path.parent.name.casefold().startswith("maafw-")
    )
    if len(metadata_files) != 1:
        raise BuildError("framework update requires exactly one maafw package metadata file")
    metadata_path = metadata_files[0]
    package = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
    if package.get("Name", "").casefold() != "maafw" or package.get("Version") != version:
        raise BuildError("framework update Python package version mismatch")
    native = candidate / FRAMEWORK_NATIVE_PATH
    python_native = candidate / FRAMEWORK_PYTHON_NATIVE_PATH
    sdk_native = sdk / "bin"
    if not native.is_dir() or not python_native.is_dir() or not sdk_native.is_dir():
        raise BuildError("framework update requires the existing host and Python native directories")
    for name in FRAMEWORK_REQUIRED_DLLS:
        if not all((root / name).is_file() for root in (native, python_native, sdk_native)):
            raise BuildError(f"framework update is missing a required DLL: {name}")
    files: dict[str, str] = {}

    def record(path: Path) -> None:
        files[path.relative_to(candidate).as_posix()] = sha256_file(path)

    # Retain the host layout, including its existing plugin and Node entrypoints.
    # New top-level DLLs are runtime dependencies; SDK examples are not added.
    native_targets = {path.relative_to(native) for path in native.rglob("*") if path.is_file()}
    native_targets.update(path.relative_to(sdk_native) for path in sdk_native.glob("*.dll"))
    for relative in sorted(native_targets):
        source = sdk_native / relative
        if not source.is_file():
            raise BuildError(f"framework archive does not cover an existing host native file: {relative}")
        target = native / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        record(target)
    for path in sorted(python_native.rglob("*")):
        if not path.is_file():
            continue
        source = sdk_native / path.relative_to(python_native)
        if not source.is_file() or sha256_file(source) != sha256_file(path):
            raise BuildError(f"framework host/Python native payload mismatch: {path.name}")
        record(path)
    sdk_agents = sdk / "share/MaaAgentBinary"
    for relative_root in FRAMEWORK_AGENT_PATHS:
        target_root = candidate / relative_root
        if not target_root.is_dir():
            continue
        if not sdk_agents.is_dir():
            raise BuildError("framework archive lacks the existing MaaAgentBinary tools")
        # Keep host-only legacy helpers, such as minicap; replace only supplied files.
        for source in sorted(sdk_agents.rglob("*")):
            if source.is_file():
                target = target_root / source.relative_to(sdk_agents)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                record(target)
    license_source = sdk / "LICENSE.md"
    if not license_source.is_file():
        raise BuildError("framework archive is missing its LICENSE.md")
    notice = candidate / FRAMEWORK_NOTICE_PATH
    notice.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(license_source, notice)
    for path in (notice, metadata_path, candidate / "requirements.txt"):
        record(path)
    files.update(_paired_runtime_files(candidate))
    return {
        "repository": FRAMEWORK_REPOSITORY,
        "version": version,
        "archive": archive.name,
        "sha256": archive_hash,
        "files": files,
        "license": {
            "spdx": "LGPL-3.0",
            "upstream_file": "LICENSE.md",
            "install_path": FRAMEWORK_NOTICE_PATH,
            "sha256": files[FRAMEWORK_NOTICE_PATH],
        },
    }


def _validate_python_runtime(root: Path, *, framework_version: str | None = None, shared: bool = False) -> dict[str, str]:
    executable = root / "python" / "python.exe"
    if not executable.is_file():
        raise BuildError("candidate embedded Python is missing")
    required_maafw = _required_maafw_version(root)
    if framework_version is not None and framework_version != required_maafw:
        raise BuildError("framework runtime version differs from the candidate maafw pin")
    import_script = (
        "import json, sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        f"sys.path.insert(0, {str(root / 'agent')!r}); "
        "import cv2, loguru, maa, numpy, onnxruntime, PIL, yaml; "
        "import agent.arena_winrate.customization_badge; "
        "import agent.arena_winrate.own_cache; "
        "import agent.custom.action.arena_reader; "
        "import agent.p_item_recognition.embedding; "
        "print(json.dumps({'numpy': numpy.__version__, 'onnxruntime': onnxruntime.__version__, "
        "'opencv-python-headless': cv2.__version__, 'pillow': PIL.__version__, "
        "'pyyaml': yaml.__version__}))"
    )
    agent_client_script = (
        "import json\n"
        "from importlib import metadata\n"
        f"required_maafw = {required_maafw!r}\n"
        "installed_maafw = metadata.version('maafw')\n"
        "if installed_maafw != required_maafw:\n"
        "    raise RuntimeError(f'maafw version mismatch: installed={installed_maafw}, required={required_maafw}')\n"
        "from maa.agent_client import AgentClient; "
        "agent_client = AgentClient('gkh-build-smoke'); "
        "agent_identifier = agent_client.identifier; "
        "del agent_client; "
        "assert isinstance(agent_identifier, str) and agent_identifier; "
        "print(json.dumps({'agent-client-construction': 'ok'}))"
    )
    native_check = (
        "import ctypes, os\n"
        f"native_directory = {str(root / FRAMEWORK_NATIVE_PATH)!r}\n"
        "native_search = os.add_dll_directory(native_directory)\n"
        f"native_framework = ctypes.WinDLL({str(root / FRAMEWORK_NATIVE_PATH / 'MaaFramework.dll')!r})\n"
        "native_framework.MaaVersion.restype = ctypes.c_char_p\n"
        "native_framework.MaaVersion.argtypes = []\n"
        "native_version = native_framework.MaaVersion().decode().removeprefix('v')\n"
        f"if native_version != {required_maafw!r}:\n"
        "    raise RuntimeError(f'host MaaFramework version mismatch: {native_version}')\n"
    )
    agent_client_script = agent_client_script.replace(
        "from maa.agent_client import AgentClient; ",
        native_check + "from maa.agent_client import AgentClient; ",
        1,
    )
    inventory_before = _tree_snapshot(root)
    short_temp_parent = Path(root.anchor) / "Temp"
    runtime_parent = short_temp_parent if short_temp_parent.is_dir() else root.parent
    with tempfile.TemporaryDirectory(prefix="gkh-runtime-smoke-", dir=runtime_parent) as runtime_text:
        runtime = Path(runtime_text)
        smoke_environment = os.environ.copy()
        if shared:
            smoke_environment["MAAFW_BINARY_PATH"] = str(root / FRAMEWORK_NATIVE_PATH)
        for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
            smoke_environment[name] = str(runtime)
        agent_client_completed = subprocess.run(
            (str(executable), "-B", "-c", agent_client_script),
            cwd=runtime,
            env=smoke_environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        completed_processes = [("AgentClient", agent_client_completed)]
        if agent_client_completed.returncode == 0:
            import_completed = subprocess.run(
                (str(executable), "-B", "-c", import_script),
                cwd=runtime,
                env=smoke_environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            completed_processes.append(("import", import_completed))
    inventory_after = _tree_snapshot(root)
    if inventory_after != inventory_before:
        paths_before = set(inventory_before)
        paths_after = set(inventory_after)
        created = sorted(paths_after - paths_before)
        removed = sorted(paths_before - paths_after)
        modified = sorted(
            path
            for path in paths_before & paths_after
            if inventory_before[path] != inventory_after[path]
        )
        raise BuildError(
            "candidate runtime smoke mutated the release tree: "
            f"created={created[:10]}, removed={removed[:10]}, modified={modified[:10]}"
        )
    versions: dict[str, str] = {}
    for label, completed in completed_processes:
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise BuildError(
                f"candidate embedded Python {label} smoke failed: {detail}"
            )
        try:
            process_versions = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise BuildError(
                f"candidate embedded Python {label} smoke returned invalid version inventory"
            ) from error
        if not isinstance(process_versions, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in process_versions.items()
        ):
            raise BuildError(
                f"candidate embedded Python {label} version inventory is invalid"
            )
        if label == "AgentClient":
            if process_versions != {"agent-client-construction": "ok"}:
                raise BuildError(
                    "candidate embedded Python AgentClient smoke returned an invalid marker"
                )
            continue
        versions.update(process_versions)
    versions["maafw"] = required_maafw
    return versions


def build_derived_package(
    *,
    source_root: Path,
    upstream_archive: Path,
    upstream_sha256: str,
    upstream_tag: str,
    source_revision: str,
    engine_bundle: Path,
    mfa_core_bundle: Path,
    python_site_packages: Path,
    output: Path,
    update_repository: str,
    allow_upstream_trial_channel: bool,
    derived_version: str | None = None,
    previous_channel_version: str | None = None,
    release_channel: str = "beta",
    validate_python_runtime: bool = True,
    framework_archive: Path | None = None,
    framework_sha256: str | None = None,
    framework_version: str | None = None,
) -> Path:
    source_root = source_root.resolve()
    upstream_archive = upstream_archive.resolve()
    engine_bundle = engine_bundle.resolve()
    mfa_core_bundle = mfa_core_bundle.resolve()
    python_site_packages = python_site_packages.resolve()
    output = output.resolve()
    framework_archive = None if framework_archive is None else framework_archive.resolve()
    actual_framework_hash = _framework_archive_hash(framework_archive, framework_sha256, framework_version)

    if release_channel not in {"beta", "stable"}:
        raise BuildError("release channel must be beta or stable")
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
        if release_channel != "beta":
            raise BuildError("a local upstream trial cannot select a durable release channel")
        if derived_version is not None:
            raise BuildError("an explicit derived release version cannot use the upstream update repository")
        if previous_channel_version is not None:
            raise BuildError("a previous channel version cannot use the upstream update repository")
        derived_version = f"{upstream_tag}+gkh.{source_revision[:7]}"
        if LOCAL_TRIAL_VERSION_PATTERN.fullmatch(derived_version) is None:
            raise BuildError("local trial version is invalid")
    else:
        if derived_version is None:
            raise BuildError("a durable derived release requires an explicit project version")
        if PROJECT_RELEASE_VERSION_PATTERN.fullmatch(derived_version) is None:
            raise BuildError(
                "derived release version must use independent GKH SemVer vMAJOR.MINOR.PATCH"
            )
    if update_repository == UPSTREAM_REPOSITORY:
        version_ordering = "upstream_equal_precedence_build_metadata"
        requires_manual_bootstrap = False
    else:
        version_ordering, requires_manual_bootstrap = _durable_update_semantics(
            derived_version,
            previous_channel_version,
        )

    engine_manifest = _validate_engine_bundle(engine_bundle)
    mfa_core_manifest = _validate_mfa_core_bundle(mfa_core_bundle)
    _validate_input_tree(python_site_packages, label="embedded Python site-packages", reject_private_names=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the transient root short: Windows native-extension loading can fail
    # once the staged DLL path grows beyond legacy loader limits.
    staging = Path(tempfile.mkdtemp(prefix=".gkh-", dir=output.parent))
    extracted = staging / "upstream"
    source_snapshot = staging / "source"
    candidate = staging / "candidate"
    try:
        _export_source_revision(source_root, source_revision, source_snapshot)
        _validate_mfa_core_patch_source(source_snapshot, mfa_core_manifest)
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

        for file_name, source_relative in ROOT_FILE_SOURCES.items():
            source_file = source_snapshot / source_relative
            if source_file.is_file():
                shutil.copy2(source_file, candidate / file_name)

        for source_relative, target_relative in INSTALL_PAYLOAD_FILES.items():
            source_file = source_snapshot / source_relative
            if not source_file.is_file():
                raise BuildError(f"source install payload is missing: {source_relative}")
            target_file = candidate / target_relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)

        mfa_core_target = candidate / MFA_CORE_INSTALL_PATH
        if not mfa_core_target.is_file() or _is_link_or_reparse(mfa_core_target):
            raise BuildError(
                f"upstream package is missing the replaceable MFA Core payload: {MFA_CORE_INSTALL_PATH}"
            )
        shutil.copy2(
            mfa_core_bundle / MFA_CORE_BUNDLE_PAYLOAD_PATH,
            mfa_core_target,
        )

        shutil.copy2(source_snapshot / "assets" / "interface.json", candidate / "interface.json")
        interface = _load_json(candidate / "interface.json")
        interface["version"] = derived_version
        interface["github"] = update_repository
        for key in FORBIDDEN_DERIVED_UPDATE_KEYS:
            interface.pop(key, None)
        agent = interface.get("agent")
        if not isinstance(agent, dict):
            raise BuildError("source interface Agent configuration is invalid")
        agent["child_exec"] = "./python/python.exe"
        agent["child_args"] = ["-u", "./agent/main.py"]
        _synchronize_arena_period_selector(
            candidate,
            interface,
            stages_path=(
                engine_bundle
                / "node_modules"
                / "gakumas-data"
                / "json"
                / "stages.json"
            ),
        )
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
        _prune_dependency_tests(python_target)
        _prune_unused_video_backend(python_target)
        python_source_normalizations = _normalize_python_dependency_sources(python_target)

        framework = None
        if framework_archive is not None:
            assert actual_framework_hash is not None and framework_version is not None
            framework_sdk = staging / "framework"
            framework_sdk.mkdir()
            _safe_extract(framework_archive, framework_sdk)
            _validate_input_tree(framework_sdk, label="MaaFramework archive", reject_private_names=False)
            framework = _overlay_framework(
                candidate, framework_sdk, archive=framework_archive,
                archive_hash=actual_framework_hash, version=framework_version,
            )
            _share_framework_runtime(candidate, framework)
            _remove_obsolete_root_agent_tools(candidate, framework)

        critical_hashes = _validate_candidate(
            candidate,
            version=derived_version,
            update_repository=update_repository,
        )
        if framework is not None:
            critical_hashes.update(framework["files"])
        if validate_python_runtime:
            critical_hashes.update(_paired_runtime_files(candidate, shared=framework is not None))
        python_packages = (
            _validate_python_runtime(candidate, **({"framework_version": framework_version, "shared": True} if framework else {}))
            if validate_python_runtime else {}
        )
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
            "mfa_core": {
                "schema_version": MFA_CORE_BUNDLE_SCHEMA_VERSION,
                "component": MFA_CORE_COMPONENT,
                "status": MFA_CORE_STATUS,
                "bundle_manifest_sha256": sha256_file(mfa_core_bundle / "manifest.json"),
                "upstream": mfa_core_manifest["upstream"],
                "patch": mfa_core_manifest["patch"],
                "build": MFA_CORE_PACKAGED_BUILD,
                "input": mfa_core_manifest["input"],
                "payload": {
                    "bundle_path": MFA_CORE_BUNDLE_PAYLOAD_PATH,
                    "install_path": MFA_CORE_INSTALL_PATH,
                    "sha256": sha256_file(candidate / MFA_CORE_INSTALL_PATH),
                    "size": (candidate / MFA_CORE_INSTALL_PATH).stat().st_size,
                },
                "license": mfa_core_manifest["license"],
            },
            "python_packages": python_packages,
            "python_source_normalizations": python_source_normalizations,
            "update_contract": {
                "mode": update_mode,
                "repository": update_repository,
                "previous_channel_version": previous_channel_version,
                "release_channel": release_channel if update_mode == "derived_release_channel" else "local",
                "version_namespace": (
                    "independent_gkh_semver"
                    if update_mode == "derived_release_channel"
                    else "legacy_upstream_based_local_trial"
                ),
                "version_ordering": version_ordering,
                "requires_manual_bootstrap": requires_manual_bootstrap,
                "client_updater": "mfa_builtin_resource_update",
                "payload_scope": "full_derived_package",
                "python_dependency_updater": "existing_agent_pip_update",
                "auto_update_setting": "preserve_user_configuration",
                "safe_upstream_precedence": upstream_tag,
                "requires_derived_release_before_newer_upstream": update_mode
                == "upstream_equal_precedence_trial",
            },
            "critical_files": critical_hashes,
        }
        if framework is not None:
            manifest["framework"] = framework
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
    parser.add_argument("--mfa-core-bundle", type=Path, required=True)
    parser.add_argument("--python-site-packages", type=Path, required=True)
    parser.add_argument("--framework-archive", type=Path)
    parser.add_argument("--framework-sha256")
    parser.add_argument("--framework-version")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-repository", default=UPSTREAM_REPOSITORY)
    parser.add_argument("--allow-upstream-trial-channel", action="store_true")
    parser.add_argument("--derived-version")
    parser.add_argument("--previous-channel-version")
    parser.add_argument("--release-channel", choices=("beta", "stable"), default="beta")
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
        mfa_core_bundle=args.mfa_core_bundle,
        python_site_packages=args.python_site_packages,
        output=args.output,
        update_repository=args.update_repository,
        allow_upstream_trial_channel=args.allow_upstream_trial_channel,
        derived_version=args.derived_version,
        previous_channel_version=args.previous_channel_version,
        release_channel=args.release_channel,
        framework_archive=args.framework_archive,
        framework_sha256=args.framework_sha256,
        framework_version=args.framework_version,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
