"""Build a one-commit public source snapshot from an approved local revision.

The development repository and its history are never copied. Only explicitly
listed product sources are exported, privacy-checked, and committed under a
non-personal release identity.
"""

from __future__ import annotations

import os
import re
import json
import stat
import shutil
import hashlib
import zipfile
import tempfile
import subprocess
from pathlib import Path

try:
    from tools.deployment.privacy_gate import PrivacyGateError, validate_tree
except ModuleNotFoundError:  # Direct script execution from tools/deployment.
    from privacy_gate import PrivacyGateError, validate_tree

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
    "p_items.csv",
    "skill_cards.csv",
    "support_cards.json",
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

VERSION_PATTERN = re.compile(r"^v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\+gkh\.\d{6}$")
REPOSITORY_PATTERN = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+$")
PUBLIC_AUTHOR_NAME = "Gakumas Helper Release"
PUBLIC_AUTHOR_EMAIL = "noreply@gakumas-helper.invalid"
PUBLIC_COMMIT_DATE = "2026-08-23T00:00:00Z"
INTERNAL_TASK_IDENTIFIER_PATTERN = re.compile(rb"(?i)\bTA" rb"SK-[0-9]{3}\b")
ZIP_CONTAINER_SUFFIXES = {".jar", ".npz", ".whl", ".zip"}


class PublicSnapshotError(RuntimeError):
    """Raised when a public source snapshot would violate the release boundary."""


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


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename.replace("\\", "/"))
            mode = entry.external_attr >> 16
            target = (destination / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise PublicSnapshotError(f"source archive contains an unsafe member: {entry.filename}")
            if os.path.commonpath((str(destination_root), str(target))) != str(destination_root):
                raise PublicSnapshotError(f"source archive member escapes destination: {entry.filename}")
        archive.extractall(destination)


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


def _validate_snapshot(output: Path) -> dict[str, int]:
    for path in output.rglob("*"):
        relative = path.relative_to(output)
        if any(part.casefold() in FORBIDDEN_PATH_NAMES_CASEFOLD for part in relative.parts):
            raise PublicSnapshotError(f"public snapshot contains a forbidden path: {relative.as_posix()}")
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


def _release_git_environment(empty_config: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    environment.update(
        {
            "GIT_AUTHOR_NAME": PUBLIC_AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": PUBLIC_AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": PUBLIC_AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": PUBLIC_AUTHOR_EMAIL,
            "GIT_AUTHOR_DATE": PUBLIC_COMMIT_DATE,
            "GIT_COMMITTER_DATE": PUBLIC_COMMIT_DATE,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": str(empty_config),
        }
    )
    return environment


def build_public_snapshot(
    *,
    source_root: Path,
    source_revision: str,
    output: Path,
    version: str,
    repository: str,
) -> dict[str, object]:
    source_root = source_root.resolve()
    output = output.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise PublicSnapshotError("source revision must be a full lowercase Git SHA")
    if VERSION_PATTERN.fullmatch(version) is None:
        raise PublicSnapshotError("public preview version must use vMAJOR.MINOR.PATCH+gkh.YYMMDD")
    if REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise PublicSnapshotError("public repository must be a GitHub repository URL without a trailing slash")
    if output.exists():
        raise PublicSnapshotError(f"public snapshot output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gkh-public-source-", dir=output.parent) as temporary_text:
        temporary = Path(temporary_text)
        archive = temporary / "source.zip"
        exported = temporary / "source"
        empty_template = temporary / "empty-git-template"
        empty_hooks = temporary / "empty-git-hooks"
        empty_attributes = temporary / "empty-git-attributes"
        empty_config = temporary / "empty-git-config"
        empty_template.mkdir()
        empty_hooks.mkdir()
        empty_attributes.write_text("", encoding="utf-8")
        empty_config.write_text("", encoding="utf-8")
        release_git_environment = _release_git_environment(empty_config)
        _run(
            (
                "git",
                "-C",
                str(source_root),
                "archive",
                "--format=zip",
                f"--output={archive}",
                source_revision,
            ),
            env=release_git_environment,
        )
        exported.mkdir()
        _safe_extract(archive, exported)
        output.mkdir()
        try:
            for relative in DIRECTORY_ALLOWLIST:
                _copy_tree(exported, output, relative)
            for relative in FILE_ALLOWLIST:
                _copy_file(exported, output, relative)
            for name in DATA_ALLOWLIST:
                _copy_file(exported, output, f"assets/data/{name}")

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
            interface_path = output / "assets" / "interface.json"
            interface = json.loads(interface_path.read_text(encoding="utf-8"))
            interface["version"] = version
            interface["github"] = repository
            interface_path.write_text(json.dumps(interface, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
            _run(("git", *git_isolation, "add", "--", *top_level), cwd=output, env=release_git_environment)
            _run(
                (
                    "git",
                    *git_isolation,
                    "commit",
                    "-q",
                    "-m",
                    f"release: {version}",
                ),
                cwd=output,
                env=release_git_environment,
            )
            revision = _run(("git", "rev-parse", "HEAD"), cwd=output, env=release_git_environment)
            if _run(("git", "rev-list", "--all", "--count"), cwd=output, env=release_git_environment) != "1":
                raise PublicSnapshotError("public snapshot must contain exactly one commit")
            if _run(("git", "remote"), cwd=output, env=release_git_environment):
                raise PublicSnapshotError("public snapshot must not inherit a Git remote")
            identity = _run(
                ("git", "log", "-1", "--format=%an|%ae|%cn|%ce"),
                cwd=output,
                env=release_git_environment,
            )
            expected_identity = "|".join((PUBLIC_AUTHOR_NAME, PUBLIC_AUTHOR_EMAIL, PUBLIC_AUTHOR_NAME, PUBLIC_AUTHOR_EMAIL))
            if identity != expected_identity:
                raise PublicSnapshotError(f"public snapshot commit identity is not fixed: {identity}")
            if _run(("git", "status", "--short"), cwd=output, env=release_git_environment):
                raise PublicSnapshotError("public snapshot working tree changed during commit")
            _run(("git", "fsck", "--full", "--no-reflogs"), cwd=output, env=release_git_environment)

            committed_archive = temporary / "public-head.zip"
            committed_tree = temporary / "public-head"
            _run(
                ("git", "archive", "--format=zip", f"--output={committed_archive}", "HEAD"),
                cwd=output,
                env=release_git_environment,
            )
            committed_tree.mkdir()
            _safe_extract(committed_archive, committed_tree)
            privacy = _validate_snapshot(committed_tree)
            if _file_inventory(output) != _file_inventory(committed_tree):
                raise PublicSnapshotError("public snapshot Git tree differs from the privacy-checked working tree")
            return {
                "output": str(output),
                "revision": revision,
                "version": version,
                "repository": repository,
                "files_scanned": privacy["files_scanned"],
                "bytes_scanned": privacy["bytes_scanned"],
            }
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()
    result = build_public_snapshot(
        source_root=args.source_root,
        source_revision=args.source_revision,
        output=args.output,
        version=args.version,
        repository=args.repository,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
