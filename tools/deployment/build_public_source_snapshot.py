"""Build a public-only source snapshot from an approved local revision.

The development repository and its history are never copied. Only explicitly
listed product sources are exported and privacy-checked. The first public
snapshot is an explicit root; every later snapshot has the previously verified
public ``main`` as its only parent, preserving a sanitized linear history.
"""

from __future__ import annotations

import os
import re
import json
import shutil
import hashlib
import tomllib
import zipfile
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from collections.abc import Callable

try:
    from tools.deployment.privacy_gate import (
        PrivacyGateError,
        validate_tree,
        validate_relative_path,
        private_machine_markers,
    )
except ModuleNotFoundError:  # Direct script execution from tools/deployment.
    from privacy_gate import (
        PrivacyGateError,
        validate_tree,
        validate_relative_path,
        private_machine_markers,
    )

DIRECTORY_ALLOWLIST = (
    "agent",
    "assets/lang",
    "assets/resource",
    "assets/tasks",
    "guides",
    "THIRD_PARTY_NOTICES",
    "tools/card-embedding",
    "tools/ci",
    "tools/deployment",
    "tools/p-item-embedding",
    "tools/task400-engine",
)

FILE_ALLOWLIST = (
    ".gitattributes",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "SUPPORT.md",
    "logo.ico",
    "maatools.config.mts",
    "package-lock.json",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "assets/interface.json",
    "tools/custom.action.schema.json",
    "tools/custom.recognition.schema.json",
    "tools/interface_config.schema.json",
    "tools/interface_import.schema.json",
    "tools/interface.schema.json",
    "tools/pipeline.schema.json",
    "tools/install.py",
    "tools/calibrated_p_item_reference.py",
    "tools/p_item_derived_source.py",
    "tools/p_item_derived_qualification.py",
    "tools/published_ui_reference.py",
    "tools/sync_cards.py",
    "tools/sync_lang.py",
    "tools/sync_support_cards.py",
    "tools/task080_build_badge_reference.py",
    "tools/task410_build_card_cost_reference.py",
    "tools/update_cards.py",
)

DATA_ALLOWLIST = (
    "arena_own_score_schema.json",
    "arena_winrate_schema.json",
    "card_selection_presets.yaml",
    "idols_cards.json",
    "live_decision_presets.yaml",
    "p_drinks.csv",
    "p_item_detail_text.json",
    "p_items.csv",
    "skill_cards.csv",
    "support_cards.json",
)

PUBLIC_TEST_ALLOWLIST = (
    "tests/test_arena_grade.py",
    "tests/test_arena_stage_catalog.py",
)

GENERATED_FILE_ALLOWLIST = (
    ".gitignore",
    "ASSET_PROVENANCE.md",
    "README.md",
)

FORBIDDEN_PATH_NAMES = {
    ".claude",
    ".local",
    ".vscode",
    "AGENTS.md",
    "PROJECT_CONTEXT.md",
    "ROADMAP.md",
    "TASKS.md",
    "appsettings.json",
    "config",
    "debug",
    "docs",
    "live_provider_assessments.json",
    "tests",
}
FORBIDDEN_PATH_NAMES_CASEFOLD = {name.casefold() for name in FORBIDDEN_PATH_NAMES}

PROJECT_VERSION_PATTERN = re.compile(
    r"^v(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)$"
)
LEGACY_PUBLIC_VERSION_PATTERN = re.compile(
    r"^v(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)\+gkh\.(?P<date>\d{6})$"
)
KNOWN_LEGACY_PUBLIC_VERSIONS = {
    "v1.4.8+gkh.260823",
    "v1.4.9+gkh.260824",
}
MINIMUM_INDEPENDENT_PROJECT_VERSION = "v0.1.0"
FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION = "v0.1.1"
RESERVED_UNPUBLISHABLE_PROJECT_VERSIONS = frozenset({"v0.1.0"})
REPOSITORY_PATTERN = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+$")
PUBLIC_AUTHOR_NAME = "Gakumas Helper Release"
PUBLIC_AUTHOR_EMAIL = "noreply@gakumas-helper.invalid"
INTERNAL_TASK_IDENTIFIER_PATTERN = re.compile(rb"(?i)\bTA" rb"SK-[0-9]{3}\b")
ZIP_CONTAINER_SUFFIXES = {".jar", ".npz", ".whl", ".zip"}
WINDOWS_RESERVED_PATH_STEMS = {
    "aux",
    "clock$",
    "con",
    "conin$",
    "conout$",
    "nul",
    "prn",
    *(f"com{suffix}" for suffix in (*range(1, 10), "¹", "²", "³")),
    *(f"lpt{suffix}" for suffix in (*range(1, 10), "¹", "²", "³")),
}
CURRENT_INTERFACE_FORBIDDEN_FIELDS = (
    "mirrorchyan_rid",
    "mirrorchyan_multiplatform",
)
RELEASE_NOTES_METADATA_HEADING = "### 可复核信息"
RELEASE_NOTES_PLACEHOLDER_PATTERN = re.compile(r"@[A-Z][A-Z0-9_]*@")


class PublicSnapshotError(RuntimeError):
    """Raised when a public source snapshot would violate the release boundary."""


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return

    def make_writable_and_retry(
        function: Callable[[str], object], target: str, _error: BaseException
    ) -> None:
        os.chmod(target, 0o700)
        function(target)

    shutil.rmtree(path, onexc=make_writable_and_retry)


def _reject_internal_task_identifiers(output: Path) -> None:
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        if INTERNAL_TASK_IDENTIFIER_PATTERN.search(relative.encode("utf-8")):
            raise PublicSnapshotError(
                f"public snapshot contains an internal task identifier in a path: {relative}"
            )
        payload = path.read_bytes()
        if INTERNAL_TASK_IDENTIFIER_PATTERN.search(payload):
            raise PublicSnapshotError(
                f"public snapshot contains an internal task identifier: {relative}"
            )
        if path.suffix.casefold() not in ZIP_CONTAINER_SUFFIXES or not zipfile.is_zipfile(path):
            continue
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                if INTERNAL_TASK_IDENTIFIER_PATTERN.search(entry.filename.encode("utf-8")):
                    raise PublicSnapshotError(
                        "public snapshot contains an internal task identifier in "
                        f"{relative}!{entry.filename}"
                    )
                member = archive.read(entry)
                if INTERNAL_TASK_IDENTIFIER_PATTERN.search(member):
                    raise PublicSnapshotError(
                        "public snapshot contains an internal task identifier in "
                        f"{relative}!{entry.filename}"
                    )


def _run(
    arguments: tuple[str, ...],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise PublicSnapshotError(f"command failed ({' '.join(arguments[:3])}): {detail}")
    return completed.stdout.strip()


def _run_bytes(
    arguments: tuple[str, ...],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> bytes:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        capture_output=True,
        timeout=120,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:].decode("utf-8", errors="replace")
        raise PublicSnapshotError(f"command failed ({' '.join(arguments[:3])}): {detail.strip()}")
    return completed.stdout


def _materialize_git_tree(
    repository: Path,
    revision: str,
    destination: Path,
    *,
    env: dict[str, str],
) -> int:
    if destination.exists():
        raise PublicSnapshotError(f"Git tree destination already exists: {destination}")
    destination.mkdir()
    listing = _run_bytes(
        ("git", "ls-tree", "-r", "-t", "-z", "--full-tree", revision),
        cwd=repository,
        env=env,
    )
    records = listing.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    seen_entries: set[str] = set()
    seen_prefix_casefold: dict[str, str] = {}
    tree_paths: set[str] = set()
    blob_paths: set[str] = set()
    path_machine_markers = private_machine_markers()
    for record in records:
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_sha = metadata.decode("ascii").split()
            git_path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise PublicSnapshotError("public Git tree contains an invalid entry") from error
        is_blob = object_type == "blob" and mode in {"100644", "100755"}
        is_tree = object_type == "tree" and mode == "040000"
        if not is_blob and not is_tree:
            raise PublicSnapshotError(
                f"public Git tree contains an unsupported entry: {git_path} ({mode} {object_type})"
            )
        if re.fullmatch(r"[0-9a-f]{40}", object_sha) is None:
            raise PublicSnapshotError(f"public Git tree contains an invalid object id: {git_path}")
        if git_path.startswith("/") or "\\" in git_path:
            raise PublicSnapshotError(f"public Git tree contains an unsafe path: {git_path}")
        parts = git_path.split("/")
        if any(
            not part
            or part in {".", ".."}
            or part.endswith((".", " "))
            or any(character in '<>:"\\|?*' or ord(character) < 32 for character in part)
            for part in parts
        ):
            raise PublicSnapshotError(f"public Git tree contains an unsafe path: {git_path}")
        if any(
            part.split(".", 1)[0].rstrip(" ").casefold() in WINDOWS_RESERVED_PATH_STEMS
            for part in parts
        ):
            raise PublicSnapshotError(
                f"public Git tree contains a reserved Windows device path: {git_path}"
            )
        try:
            validate_relative_path(
                Path(*parts),
                is_directory=is_tree,
                allowed_emails={PUBLIC_AUTHOR_EMAIL},
                allow_arena_engine_config=False,
                machine_markers=path_machine_markers,
            )
        except PrivacyGateError as error:
            raise PublicSnapshotError(str(error)) from error
        if git_path in seen_entries:
            raise PublicSnapshotError(f"public Git tree contains a duplicate path: {git_path}")
        seen_entries.add(git_path)
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            prefix_casefolded = prefix.casefold()
            previous_prefix = seen_prefix_casefold.get(prefix_casefolded)
            if previous_prefix is not None and previous_prefix != prefix:
                raise PublicSnapshotError(
                    "public Git tree contains a case-colliding path prefix: "
                    f"{previous_prefix} and {prefix}"
                )
            seen_prefix_casefold[prefix_casefolded] = prefix
        if is_tree:
            tree_paths.add(git_path)
            continue
        blob_paths.add(git_path)
        target = destination.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            _run_bytes(("git", "cat-file", "blob", object_sha), cwd=repository, env=env)
        )
    nonempty_tree_paths = {
        "/".join(blob_path.split("/")[:length])
        for blob_path in blob_paths
        for length in range(1, len(blob_path.split("/")))
    }
    empty_tree_paths = sorted(tree_paths - nonempty_tree_paths)
    if empty_tree_paths:
        raise PublicSnapshotError(
            f"public Git tree contains an empty directory: {empty_tree_paths[0]}"
        )
    return len(blob_paths)


def _copy_file(source_root: Path, output: Path, relative: str) -> None:
    source = source_root / relative
    if not source.is_file():
        raise PublicSnapshotError(f"allowlisted source file is missing: {relative}")
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source_root: Path, output: Path, relative: str) -> None:
    source = source_root / relative
    if not source.is_dir():
        raise PublicSnapshotError(f"allowlisted source directory is missing: {relative}")
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        ignore=lambda _directory, names: {
            name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
        },
    )


def _replace_required_toml_line(section: str, key: str, replacement: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    updated, replacements = pattern.subn(replacement, section)
    if replacements != 1:
        raise PublicSnapshotError(f"pyproject [project] must define {key} exactly once")
    return updated


def _replace_required_toml_array(section: str, key: str, values: list[str]) -> str:
    pattern = re.compile(rf"(?ms)^{re.escape(key)}\s*=\s*\[.*?^\]\s*$")
    rendered = f"{key} = [\n" + "".join(
        f"    {json.dumps(value, ensure_ascii=False)},\n" for value in values
    ) + "]\n"
    updated, replacements = pattern.subn(rendered, section)
    if replacements != 1:
        raise PublicSnapshotError(f"pyproject [project] must define {key} exactly once")
    return updated


def _rewrite_pyproject(path: Path, *, version: str, repository: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
        metadata = tomllib.loads(text)
        project = metadata["project"]
        classifiers = project["classifiers"]
        dependencies = project["dependencies"]
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise PublicSnapshotError("public pyproject metadata is unavailable or invalid") from error
    if not isinstance(classifiers, list) or not all(isinstance(value, str) for value in classifiers):
        raise PublicSnapshotError("public pyproject classifiers must be an array of strings")
    if not isinstance(dependencies, list) or not all(isinstance(value, str) for value in dependencies):
        raise PublicSnapshotError("public pyproject dependencies must be an array of strings")

    project_header = re.search(r"(?m)^\[project\]\s*$", text)
    if project_header is None:
        raise PublicSnapshotError("pyproject is missing [project]")
    following = re.search(r"(?m)^\[", text[project_header.end() :])
    project_end = (
        project_header.end() + following.start() if following is not None else len(text)
    )
    project_section = text[project_header.end() : project_end]
    project_section = _replace_required_toml_line(
        project_section,
        "version",
        f'version = "{version.removeprefix("v")}"',
    )
    project_section = _replace_required_toml_line(
        project_section,
        "license",
        'license = {text = "AGPL-3.0-only"}',
    )
    public_classifiers = [
        value for value in classifiers if not value.startswith("License ::")
    ]
    license_classifier = "License :: OSI Approved :: GNU Affero General Public License v3"
    audience_index = next(
        (
            index + 1
            for index, value in enumerate(public_classifiers)
            if value.startswith("Intended Audience ::")
        ),
        len(public_classifiers),
    )
    public_classifiers.insert(audience_index, license_classifier)
    project_section = _replace_required_toml_array(
        project_section,
        "classifiers",
        public_classifiers,
    )
    public_dependencies = list(dependencies)
    if not any(value.casefold() == "opencv-python-headless" for value in public_dependencies):
        public_dependencies.append("opencv-python-headless")
    project_section = _replace_required_toml_array(
        project_section,
        "dependencies",
        public_dependencies,
    )
    text = text[: project_header.end()] + project_section + text[project_end:]

    urls_header = re.search(r"(?m)^\[project\.urls\]\s*$", text)
    if urls_header is None:
        raise PublicSnapshotError("pyproject is missing [project.urls]")
    following = re.search(r"(?m)^\[", text[urls_header.end() :])
    urls_end = urls_header.end() + following.start() if following is not None else len(text)
    urls_section = text[urls_header.end() : urls_end]
    urls_section = _replace_required_toml_line(
        urls_section,
        "Homepage",
        f'Homepage = "{repository}"',
    )
    urls_section = _replace_required_toml_line(
        urls_section,
        "Repository",
        f'Repository = "{repository}"',
    )
    urls_section = _replace_required_toml_line(
        urls_section,
        "Issues",
        f'Issues = "{repository}/issues"',
    )
    text = text[: urls_header.end()] + urls_section + text[urls_end:]

    try:
        rewritten = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise PublicSnapshotError("rewritten public pyproject is invalid TOML") from error
    rewritten_project = rewritten.get("project", {})
    rewritten_urls = rewritten_project.get("urls", {})
    if (
        rewritten_project.get("version") != version.removeprefix("v")
        or rewritten_project.get("license") != {"text": "AGPL-3.0-only"}
        or license_classifier not in rewritten_project.get("classifiers", [])
        or any("MIT License" in value for value in rewritten_project.get("classifiers", []))
        or "opencv-python-headless" not in rewritten_project.get("dependencies", [])
        or rewritten_urls
        != {
            "Homepage": repository,
            "Repository": repository,
            "Issues": f"{repository}/issues",
        }
    ):
        raise PublicSnapshotError("rewritten public pyproject metadata is inconsistent")
    path.write_text(text, encoding="utf-8")


def _is_allowlisted_public_file(relative: Path) -> bool:
    normalized = relative.as_posix()
    if normalized in GENERATED_FILE_ALLOWLIST or normalized in FILE_ALLOWLIST:
        return True
    if normalized in PUBLIC_TEST_ALLOWLIST:
        return True
    if normalized in {f"assets/data/{name}" for name in DATA_ALLOWLIST}:
        return True
    return any(
        normalized.startswith(f"{directory}/")
        for directory in DIRECTORY_ALLOWLIST
    )


def _release_commit_date(release_date: str) -> str:
    try:
        calendar_date = datetime.strptime(release_date, "%Y-%m-%d")
    except ValueError as error:
        raise PublicSnapshotError("release date must use a valid YYYY-MM-DD calendar date") from error
    if calendar_date.strftime("%Y-%m-%d") != release_date:
        raise PublicSnapshotError("release date must use zero-padded YYYY-MM-DD")
    return calendar_date.strftime("%Y-%m-%dT00:00:00Z")


def _legacy_version_commit_date(version: str) -> str:
    match = LEGACY_PUBLIC_VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise PublicSnapshotError("legacy public version is invalid")
    try:
        calendar_date = datetime.strptime(f"20{match.group('date')}", "%Y%m%d")
    except ValueError as error:
        raise PublicSnapshotError("legacy public version contains an invalid calendar date") from error
    return calendar_date.strftime("%Y-%m-%dT00:00:00Z")


def _version_core(match: re.Match[str]) -> tuple[int, int, int]:
    return tuple(int(match.group(name)) for name in ("major", "minor", "patch"))


def _legacy_version_lineage_key(version: str) -> tuple[tuple[int, int, int], int, int]:
    match = LEGACY_PUBLIC_VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise PublicSnapshotError("legacy public version is invalid")
    return (
        _version_core(match),
        int(match.group("date")),
        0,
    )


def _project_version_key(version: str) -> tuple[int, int, int]:
    match = PROJECT_VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise PublicSnapshotError(
            "public release version must use independent GKH SemVer vMAJOR.MINOR.PATCH"
        )
    version_key = _version_core(match)
    first_version_match = PROJECT_VERSION_PATTERN.fullmatch(MINIMUM_INDEPENDENT_PROJECT_VERSION)
    assert first_version_match is not None
    if version_key < _version_core(first_version_match):
        raise PublicSnapshotError(
            "public release version must not precede the first independent GKH version "
            f"{MINIMUM_INDEPENDENT_PROJECT_VERSION}"
        )
    return version_key


def _release_commit_message(version: str) -> str:
    return f"chore(release): 发布 {version} 并同步更新文档与公告"


def _release_version_from_message(message: str) -> str | None:
    legacy_prefix = "release: "
    if message.startswith(legacy_prefix):
        version = message.removeprefix(legacy_prefix)
        return version if version in KNOWN_LEGACY_PUBLIC_VERSIONS else None
    prefix = "chore(release): 发布 "
    suffix = " 并同步更新文档与公告"
    if message.startswith(prefix) and message.endswith(suffix):
        version = message[len(prefix) : -len(suffix)]
        if PROJECT_VERSION_PATTERN.fullmatch(version) is not None:
            return version
    return None


def _render_release_changelog(
    template: str,
    *,
    version: str,
    repository: str,
) -> str:
    marker = f"\n{RELEASE_NOTES_METADATA_HEADING}\n"
    if template.count(marker) != 1:
        raise PublicSnapshotError(
            "release notes must contain exactly one verifiable-information heading"
        )
    body, _marker, _metadata = template.partition(marker)
    if "@VERSION@" not in body:
        raise PublicSnapshotError("release notes body must contain @VERSION@")
    rendered = body.replace("@VERSION@", version).replace("@REPOSITORY@", repository)
    unresolved = sorted(set(RELEASE_NOTES_PLACEHOLDER_PATTERN.findall(rendered)))
    if unresolved:
        raise PublicSnapshotError(
            "release notes body contains unresolved placeholders: " + ", ".join(unresolved)
        )
    return rendered.rstrip() + "\n"


def _validate_snapshot(output: Path) -> dict[str, int]:
    for path in output.rglob("*"):
        relative = path.relative_to(output)
        is_public_test_path = relative == Path("tests") or relative.as_posix() in PUBLIC_TEST_ALLOWLIST
        if not is_public_test_path and any(
            part.casefold() in FORBIDDEN_PATH_NAMES_CASEFOLD for part in relative.parts
        ):
            raise PublicSnapshotError(f"public snapshot contains a forbidden path: {relative.as_posix()}")
        if path.is_file() and not _is_allowlisted_public_file(relative):
            raise PublicSnapshotError(
                f"public snapshot contains a file outside the public allowlist: {relative.as_posix()}"
            )
    _reject_internal_task_identifiers(output)
    try:
        return validate_tree(
            output,
            project_path_predicate=lambda _relative: True,
            allowed_emails={PUBLIC_AUTHOR_EMAIL},
            allow_arena_engine_config=False,
        )
    except PrivacyGateError as error:
        raise PublicSnapshotError(str(error)) from error


def _file_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def _release_git_environment(empty_config: Path, *, commit_date: str) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    environment.update(
        {
            "GIT_AUTHOR_NAME": PUBLIC_AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": PUBLIC_AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": PUBLIC_AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": PUBLIC_AUTHOR_EMAIL,
            "GIT_AUTHOR_DATE": commit_date,
            "GIT_COMMITTER_DATE": commit_date,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": str(empty_config),
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return environment


def _git_path(repository: Path, name: str, *, env: dict[str, str]) -> Path:
    value = Path(_run(("git", "rev-parse", "--git-path", name), cwd=repository, env=env))
    if not value.is_absolute():
        value = repository / value
    return value.resolve()


def _git_common_dir(repository: Path, *, env: dict[str, str]) -> Path:
    value = Path(_run(("git", "rev-parse", "--git-common-dir"), cwd=repository, env=env))
    if not value.is_absolute():
        value = repository / value
    return value.resolve()


def _validate_public_history(
    *,
    parent_root: Path,
    parent_revision: str,
    source_root: Path,
    repository: str,
    temporary: Path,
    env: dict[str, str],
) -> tuple[str, ...]:
    parent_root = parent_root.resolve()
    if not parent_root.is_dir():
        raise PublicSnapshotError(f"public parent repository is missing: {parent_root}")
    if _run(("git", "rev-parse", "--is-inside-work-tree"), cwd=parent_root, env=env) != "true":
        raise PublicSnapshotError("public parent must be a non-bare Git working tree")
    top_level = Path(
        _run(("git", "rev-parse", "--show-toplevel"), cwd=parent_root, env=env)
    ).resolve()
    if top_level != parent_root:
        raise PublicSnapshotError("public parent path must be the repository top level")
    if _run(("git", "rev-parse", "--is-shallow-repository"), cwd=parent_root, env=env) != "false":
        raise PublicSnapshotError("public parent repository must contain the complete public history")
    if _run(("git", "rev-parse", "--show-object-format"), cwd=parent_root, env=env) != "sha1":
        raise PublicSnapshotError("public parent repository must use SHA-1 object identifiers")
    parent_git_dir = Path(
        _run(("git", "rev-parse", "--absolute-git-dir"), cwd=parent_root, env=env)
    ).resolve()
    parent_common_dir = _git_common_dir(parent_root, env=env)
    if parent_git_dir != (parent_root / ".git").resolve() or parent_common_dir != parent_git_dir:
        raise PublicSnapshotError("public parent must be an independent Git repository, not a linked worktree")
    if (parent_git_dir / "commondir").exists():
        raise PublicSnapshotError("public parent repository contains a commondir indirection")
    if parent_common_dir == _git_common_dir(source_root, env=env):
        raise PublicSnapshotError("development repository or worktree cannot be used as the public parent")
    if _run(("git", "status", "--short"), cwd=parent_root, env=env):
        raise PublicSnapshotError("public parent working tree must be clean")
    if _run(("git", "symbolic-ref", "--short", "HEAD"), cwd=parent_root, env=env) != "main":
        raise PublicSnapshotError("public parent must have main checked out")
    if _run(("git", "rev-parse", "HEAD"), cwd=parent_root, env=env) != parent_revision:
        raise PublicSnapshotError("public parent HEAD does not match the expected remote main")
    if _run(("git", "rev-parse", "refs/heads/main"), cwd=parent_root, env=env) != parent_revision:
        raise PublicSnapshotError("public parent main does not match the expected remote main")
    if _run(("git", "cat-file", "-t", parent_revision), cwd=parent_root, env=env) != "commit":
        raise PublicSnapshotError("public parent revision is not a commit")

    if _run(("git", "remote"), cwd=parent_root, env=env):
        raise PublicSnapshotError("public parent repository must not retain a Git remote")
    refs = _run(
        ("git", "for-each-ref", "--format=%(refname)"),
        cwd=parent_root,
        env=env,
    )
    if refs != "refs/heads/main":
        raise PublicSnapshotError(f"public parent repository contains unexpected refs: {refs}")

    if _run(("git", "for-each-ref", "--format=%(refname)", "refs/replace"), cwd=parent_root, env=env):
        raise PublicSnapshotError("public parent repository contains replace refs")
    for indirect_path in ("info/grafts", "objects/info/alternates", "shallow"):
        if _git_path(parent_root, indirect_path, env=env).exists():
            raise PublicSnapshotError(
                f"public parent repository contains forbidden Git indirection: {indirect_path}"
            )
    _run(("git", "fsck", "--full", "--strict", "--no-reflogs"), cwd=parent_root, env=env)

    history_lines = _run(
        ("git", "rev-list", "--reverse", "--parents", parent_revision),
        cwd=parent_root,
        env=env,
    ).splitlines()
    commits: list[str] = []
    parents_by_commit: dict[str, tuple[str, ...]] = {}
    roots = 0
    for line in history_lines:
        fields = line.split()
        if len(fields) == 1:
            roots += 1
        elif len(fields) != 2:
            raise PublicSnapshotError("public parent history must be linear and contain no merge commits")
        commits.append(fields[0])
        parents_by_commit[fields[0]] = tuple(fields[1:])
    if roots != 1 or not commits or commits[-1] != parent_revision:
        raise PublicSnapshotError("public parent history must contain exactly one public root")

    expected_identity = "|".join(
        (PUBLIC_AUTHOR_NAME, PUBLIC_AUTHOR_EMAIL, PUBLIC_AUTHOR_NAME, PUBLIC_AUTHOR_EMAIL)
    )
    history_root = temporary / "verified-public-history"
    history_root.mkdir()
    seen_versions: set[str] = set()
    previous_commit_date: str | None = None
    previous_legacy_core: tuple[int, int, int] | None = None
    previous_legacy_date: int | None = None
    previous_legacy_revision: int | None = None
    previous_project_version: tuple[int, int, int] | None = None
    independent_namespace_started = False
    for index, commit in enumerate(commits):
        metadata = _run(
            ("git", "show", "-s", "--format=%an|%ae|%cn|%ce%n%s%n%aI%n%cI", commit),
            cwd=parent_root,
            env=env,
        ).splitlines()
        if len(metadata) != 4 or metadata[0] != expected_identity:
            raise PublicSnapshotError(f"public history commit identity is not fixed: {commit}")
        public_version = _release_version_from_message(metadata[1])
        if public_version is None:
            raise PublicSnapshotError(f"public history commit message is invalid: {commit}")
        if public_version in seen_versions:
            raise PublicSnapshotError(
                f"public history contains a duplicate version: {public_version}"
            )
        if public_version in RESERVED_UNPUBLISHABLE_PROJECT_VERSIONS:
            raise PublicSnapshotError(
                f"public history contains permanently reserved version: {public_version}"
            )
        legacy_match = LEGACY_PUBLIC_VERSION_PATTERN.fullmatch(public_version)
        if legacy_match is not None:
            if independent_namespace_started:
                raise PublicSnapshotError(
                    "legacy upstream-based versions cannot follow independent GKH SemVer"
                )
            expected_date = _legacy_version_commit_date(public_version)
            core_version, legacy_date, local_revision = _legacy_version_lineage_key(
                public_version
            )
            if (
                previous_legacy_core is not None
                and previous_legacy_date is not None
                and previous_legacy_revision is not None
            ):
                if core_version < previous_legacy_core or legacy_date < previous_legacy_date:
                    raise PublicSnapshotError(
                        f"public history legacy versions are not monotonic: {public_version}"
                    )
                if core_version == previous_legacy_core and (
                    legacy_date,
                    local_revision,
                ) <= (
                    previous_legacy_date,
                    previous_legacy_revision,
                ):
                    raise PublicSnapshotError(
                        "public history legacy suffixes must increase within one upstream version: "
                        f"{public_version}"
                    )
            previous_legacy_core = core_version
            previous_legacy_date = legacy_date
            previous_legacy_revision = local_revision
            expected_message = f"release: {public_version}\n"
        else:
            independent_namespace_started = True
            project_version = _project_version_key(public_version)
            if (
                previous_project_version is None
                and public_version != FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION
            ):
                raise PublicSnapshotError(
                    "the first independent GKH public version must be "
                    f"{FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION}"
                )
            if (
                previous_project_version is not None
                and project_version <= previous_project_version
            ):
                raise PublicSnapshotError(
                    f"public history independent GKH versions are not monotonic: {public_version}"
                )
            previous_project_version = project_version
            if metadata[2] != metadata[3]:
                raise PublicSnapshotError(f"public history commit date is inconsistent: {commit}")
            expected_date = _release_commit_date(metadata[2].removesuffix("T00:00:00Z"))
            expected_message = f"{_release_commit_message(public_version)}\n"
        if metadata[2:] != [expected_date, expected_date]:
            raise PublicSnapshotError(f"public history commit date is inconsistent: {commit}")
        if previous_commit_date is not None and expected_date < previous_commit_date:
            raise PublicSnapshotError(
                f"public history commit dates are not monotonic: {commit}"
            )
        seen_versions.add(public_version)
        previous_commit_date = expected_date
        raw_commit = _run_bytes(("git", "cat-file", "commit", commit), cwd=parent_root, env=env)
        try:
            raw_headers, raw_message = raw_commit.split(b"\n\n", 1)
            header_keys = tuple(line.split(b" ", 1)[0] for line in raw_headers.splitlines())
            decoded_message = raw_message.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise PublicSnapshotError(f"public history commit object is invalid: {commit}") from error
        expected_header_keys = (
            b"tree",
            *(b"parent" for _parent in parents_by_commit[commit]),
            b"author",
            b"committer",
        )
        if header_keys != expected_header_keys:
            raise PublicSnapshotError(
                f"public history commit contains unsupported headers or a signature: {commit}"
            )
        if decoded_message != expected_message:
            raise PublicSnapshotError(f"public history commit message must be one line: {commit}")

        tree = history_root / f"{index:04d}-{commit}"
        _materialize_git_tree(parent_root, commit, tree, env=env)
        _validate_snapshot(tree)
        interface_path = tree / "assets" / "interface.json"
        try:
            interface = json.loads(interface_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PublicSnapshotError(
                f"public history interface metadata is invalid: {commit}"
            ) from error
        if interface.get("version") != public_version:
            raise PublicSnapshotError(
                f"public history version does not match its commit message: {commit}"
            )
        if interface.get("github") != repository:
            raise PublicSnapshotError(
                f"public history repository does not match the target repository: {commit}"
            )
        shutil.rmtree(tree)
    history_root.rmdir()
    return tuple(commits)


def build_public_snapshot(
    *,
    source_root: Path,
    source_revision: str,
    output: Path,
    version: str,
    release_date: str,
    repository: str,
    initial_public_root: bool = False,
    public_parent_root: Path | None = None,
    public_parent_revision: str | None = None,
) -> dict[str, object]:
    source_root = source_root.resolve()
    output = output.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise PublicSnapshotError("source revision must be a full lowercase Git SHA")
    _project_version_key(version)
    if version in RESERVED_UNPUBLISHABLE_PROJECT_VERSIONS:
        raise PublicSnapshotError(
            f"public release version {version} is permanently reserved after a failed "
            "pre-publication install and must not be rebuilt or published; use at least "
            f"{FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION}"
        )
    commit_date = _release_commit_date(release_date)
    if REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise PublicSnapshotError("public repository must be a GitHub repository URL without a trailing slash")
    has_parent_root = public_parent_root is not None
    has_parent_revision = public_parent_revision is not None
    if has_parent_root != has_parent_revision:
        raise PublicSnapshotError("public parent root and revision must be provided together")
    if initial_public_root == has_parent_root:
        raise PublicSnapshotError(
            "choose exactly one history mode: initial public root or verified public parent"
        )
    if public_parent_revision is not None and re.fullmatch(
        r"[0-9a-f]{40}", public_parent_revision
    ) is None:
        raise PublicSnapshotError("public parent revision must be a full lowercase Git SHA")
    if public_parent_root is not None:
        public_parent_root = public_parent_root.resolve()
    if output.exists():
        raise PublicSnapshotError(f"public snapshot output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gkh-public-source-", dir=output.parent) as temporary_text:
        temporary = Path(temporary_text)
        exported = temporary / "source"
        empty_template = temporary / "empty-git-template"
        empty_hooks = temporary / "empty-git-hooks"
        empty_attributes = temporary / "empty-git-attributes"
        empty_config = temporary / "empty-git-config"
        empty_template.mkdir()
        empty_hooks.mkdir()
        empty_attributes.write_text("", encoding="utf-8")
        empty_config.write_text("", encoding="utf-8")
        release_git_environment = _release_git_environment(
            empty_config,
            commit_date=commit_date,
        )
        parent_history: tuple[str, ...] = ()
        if public_parent_root is not None and public_parent_revision is not None:
            parent_history = _validate_public_history(
                parent_root=public_parent_root,
                parent_revision=public_parent_revision,
                source_root=source_root,
                repository=repository,
                temporary=temporary,
                env=release_git_environment,
            )
        if _run(
            ("git", "cat-file", "-t", source_revision),
            cwd=source_root,
            env=release_git_environment,
        ) != "commit":
            raise PublicSnapshotError("source revision is not a commit")
        _materialize_git_tree(
            source_root,
            source_revision,
            exported,
            env=release_git_environment,
        )
        output.mkdir()
        try:
            for relative in DIRECTORY_ALLOWLIST:
                _copy_tree(exported, output, relative)
            for relative in FILE_ALLOWLIST:
                _copy_file(exported, output, relative)
            for name in DATA_ALLOWLIST:
                _copy_file(exported, output, f"assets/data/{name}")
            for relative in PUBLIC_TEST_ALLOWLIST:
                _copy_file(exported, output, relative)

            public_readme = output / "tools" / "deployment" / "public" / "README.md"
            readme_text = public_readme.read_text(encoding="utf-8")
            (output / "README.md").write_text(
                readme_text.replace("@VERSION@", version).replace("@REPOSITORY@", repository),
                encoding="utf-8",
            )
            public_gitignore = output / "tools" / "deployment" / "public" / "gitignore"
            shutil.copy2(public_gitignore, output / ".gitignore")
            public_provenance = output / "tools" / "deployment" / "public" / "ASSET_PROVENANCE.md"
            shutil.copy2(public_provenance, output / "ASSET_PROVENANCE.md")
            release_notes_path = output / "tools" / "deployment" / "public" / "RELEASE_NOTES.md"
            release_notes_template = release_notes_path.read_text(encoding="utf-8")
            changelog_path = output / "assets" / "resource" / "Changelog.md"
            changelog_path.write_text(
                _render_release_changelog(
                    release_notes_template,
                    version=version,
                    repository=repository,
                ),
                encoding="utf-8",
            )
            interface_path = output / "assets" / "interface.json"
            interface = json.loads(interface_path.read_text(encoding="utf-8"))
            upstream_version = interface.get("version")
            if not isinstance(upstream_version, str) or PROJECT_VERSION_PATTERN.fullmatch(
                upstream_version
            ) is None:
                raise PublicSnapshotError(
                    "source interface must record a separate upstream Maa vMAJOR.MINOR.PATCH tag"
                )
            forbidden_interface_fields = sorted(
                field for field in CURRENT_INTERFACE_FORBIDDEN_FIELDS if field in interface
            )
            if forbidden_interface_fields:
                raise PublicSnapshotError(
                    "current GKH interface must not declare upstream MirrorChyan fields: "
                    + ", ".join(forbidden_interface_fields)
                )
            interface["version"] = version
            interface["github"] = repository
            interface_path.write_text(json.dumps(interface, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            _rewrite_pyproject(output / "pyproject.toml", version=version, repository=repository)

            _validate_snapshot(output)
            _run(
                ("git", "init", "-q", "-b", "main", f"--template={empty_template}"),
                cwd=output,
                env=release_git_environment,
            )
            top_level = tuple(sorted(path.name for path in output.iterdir() if path.name != ".git"))
            git_isolation = (
                "-c",
                "core.autocrlf=false",
                "-c",
                f"core.attributesFile={empty_attributes}",
                "-c",
                f"core.hooksPath={empty_hooks}",
            )
            if public_parent_root is not None and public_parent_revision is not None:
                _run(
                    (
                        "git",
                        *git_isolation,
                        "-c",
                        "protocol.file.allow=always",
                        "-c",
                        "fetch.fsckObjects=true",
                        "fetch",
                        "--no-tags",
                        "--no-recurse-submodules",
                        "--no-write-fetch-head",
                        str(public_parent_root),
                        public_parent_revision,
                    ),
                    cwd=output,
                    env=release_git_environment,
                )
                _run(
                    ("git", "cat-file", "-e", f"{public_parent_revision}^{{commit}}"),
                    cwd=output,
                    env=release_git_environment,
                )
                _run(
                    (
                        "git",
                        "update-ref",
                        "refs/heads/main",
                        public_parent_revision,
                        "0" * 40,
                    ),
                    cwd=output,
                    env=release_git_environment,
                )
                _run(
                    ("git", "symbolic-ref", "HEAD", "refs/heads/main"),
                    cwd=output,
                    env=release_git_environment,
                )

            _run(("git", *git_isolation, "read-tree", "--empty"), cwd=output, env=release_git_environment)
            _run(
                ("git", *git_isolation, "add", "--", *top_level),
                cwd=output,
                env=release_git_environment,
            )
            tree_revision = _run(
                ("git", *git_isolation, "write-tree"),
                cwd=output,
                env=release_git_environment,
            )
            commit_arguments = ["git", *git_isolation, "commit-tree", tree_revision]
            if public_parent_revision is not None:
                commit_arguments.extend(("-p", public_parent_revision))
            commit_arguments.extend(("-m", _release_commit_message(version)))
            revision = _run(
                tuple(commit_arguments),
                cwd=output,
                env=release_git_environment,
            )
            _run(
                (
                    "git",
                    "update-ref",
                    "refs/heads/main",
                    revision,
                    public_parent_revision or "0" * 40,
                ),
                cwd=output,
                env=release_git_environment,
            )
            _run(
                ("git", "symbolic-ref", "HEAD", "refs/heads/main"),
                cwd=output,
                env=release_git_environment,
            )

            actual_parent = _run(
                ("git", "show", "-s", "--format=%P", revision),
                cwd=output,
                env=release_git_environment,
            )
            expected_parent = public_parent_revision or ""
            if actual_parent != expected_parent:
                raise PublicSnapshotError(
                    f"public snapshot parent is not the verified public main: {actual_parent}"
                )
            if public_parent_revision is not None:
                if _run(
                    ("git", "rev-list", "--count", f"{public_parent_revision}..{revision}"),
                    cwd=output,
                    env=release_git_environment,
                ) != "1":
                    raise PublicSnapshotError("public snapshot must add exactly one public commit")
                _run(
                    ("git", "merge-base", "--is-ancestor", public_parent_revision, revision),
                    cwd=output,
                    env=release_git_environment,
                )
            expected_commit_count = len(parent_history) + 1
            if _run(
                ("git", "rev-list", "--all", "--count"),
                cwd=output,
                env=release_git_environment,
            ) != str(expected_commit_count):
                raise PublicSnapshotError("public snapshot history contains unexpected commits")
            if _run(("git", "remote"), cwd=output, env=release_git_environment):
                raise PublicSnapshotError("public snapshot must not inherit a Git remote")
            refs = _run(
                ("git", "for-each-ref", "--format=%(refname)"),
                cwd=output,
                env=release_git_environment,
            )
            if refs != "refs/heads/main":
                raise PublicSnapshotError(f"public snapshot contains unexpected Git refs: {refs}")
            if _run(("git", "status", "--short"), cwd=output, env=release_git_environment):
                raise PublicSnapshotError("public snapshot working tree changed during commit")
            source_object = subprocess.run(
                ("git", "cat-file", "-e", f"{source_revision}^{{commit}}"),
                cwd=output,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
                check=False,
                env=release_git_environment,
            )
            if source_object.returncode == 0:
                raise PublicSnapshotError("development source commit leaked into public history")
            _run(
                ("git", "fsck", "--full", "--strict", "--no-reflogs"),
                cwd=output,
                env=release_git_environment,
            )
            unreachable = _run(
                ("git", "fsck", "--unreachable", "--full", "--strict", "--no-reflogs"),
                cwd=output,
                env=release_git_environment,
            )
            if unreachable:
                raise PublicSnapshotError(f"public snapshot contains unreachable Git objects: {unreachable}")
            verified_history = _validate_public_history(
                parent_root=output,
                parent_revision=revision,
                source_root=source_root,
                repository=repository,
                temporary=temporary,
                env=release_git_environment,
            )
            if len(verified_history) != expected_commit_count:
                raise PublicSnapshotError("verified public history count is inconsistent")

            committed_tree = temporary / "public-head"
            _materialize_git_tree(
                output,
                "HEAD",
                committed_tree,
                env=release_git_environment,
            )
            privacy = _validate_snapshot(committed_tree)
            if _file_inventory(output) != _file_inventory(committed_tree):
                raise PublicSnapshotError("public snapshot Git tree differs from the privacy-checked working tree")
            return {
                "output": str(output),
                "revision": revision,
                "version": version,
                "release_date": release_date,
                "upstream_version": upstream_version,
                "repository": repository,
                "lineage_mode": "initial_public_root" if initial_public_root else "linear_public_history",
                "parent_revision": public_parent_revision,
                "history_commit_count": len(verified_history),
                "files_scanned": privacy["files_scanned"],
                "bytes_scanned": privacy["bytes_scanned"],
            }
        except Exception:
            _remove_tree(output)
            raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--release-date", required=True)
    parser.add_argument("--repository", required=True)
    history_mode = parser.add_mutually_exclusive_group(required=True)
    history_mode.add_argument("--initial-public-root", action="store_true")
    history_mode.add_argument("--public-parent-root", type=Path)
    parser.add_argument("--public-parent-revision")
    args = parser.parse_args()
    result = build_public_snapshot(
        source_root=args.source_root,
        source_revision=args.source_revision,
        output=args.output,
        version=args.version,
        release_date=args.release_date,
        repository=args.repository,
        initial_public_root=args.initial_public_root,
        public_parent_root=args.public_parent_root,
        public_parent_revision=args.public_parent_revision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
