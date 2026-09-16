"""Generate/verify GitHub notes from a frozen public range using git-cliff.

This command never changes tags, releases, assets, or the startup announcement.
git-cliff is an explicit, version-pinned maintainer tool, not a runtime dependency.
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import argparse
import tempfile
import subprocess
from pathlib import Path

try:
    from tools.deployment.privacy_gate import validate_tree
except ModuleNotFoundError:
    from privacy_gate import validate_tree

TOOL_VERSION = "2.14.1"
REPOSITORY = "https://github.com/Kirinigh/Gakumas_Helper"
CONFIG = Path(__file__).with_name("git-cliff.toml")
VERSION = re.compile(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: list[str], cwd: Path) -> str:
    # Do not let shell/CI overrides silently filter commits or select a template.
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("GIT_CLIFF_")
        and key not in {"GITHUB_REPO", "GITLAB_REPO", "GITEA_REPO", "OUTPUT"}
    }
    return subprocess.check_output(
        args, cwd=cwd, env=environment, encoding="utf-8", timeout=120,
    ).strip()


def generate(
    *, source: Path, previous_tag: str, version: str, revision: str,
    manifest: Path, git_cliff: Path, output: Path, verify: bool = False,
) -> dict:
    source, git_cliff = source.resolve(), git_cliff.resolve()
    require(bool(VERSION.fullmatch(previous_tag)), "Invalid previous version tag")
    require(bool(VERSION.fullmatch(version)), "Invalid target version")
    require(previous_tag != version, "The two versions must differ")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "Full public SHA required")
    git = lambda *args: run(["git", "-C", str(source), *args], source)
    require(not git("status", "--porcelain"), "Public source must be clean")
    require(git("rev-parse", "HEAD") == revision, "HEAD differs from frozen public SHA")
    require(git("rev-parse", "--is-shallow-repository") == "false", "Full public history required")
    base = git("rev-parse", "--verify", f"refs/tags/{previous_tag}^{{commit}}")
    git("merge-base", "--is-ancestor", base, revision)
    target = git("tag", "--list", version)
    if target:
        require(git("rev-parse", f"refs/tags/{version}^{{commit}}") == revision,
                "Existing target tag differs from frozen public SHA")
    rows = git("rev-list", "--reverse", "--parents", f"{base}..{revision}").splitlines()
    commits = [row.split()[0] for row in rows]
    require(bool(commits), "Release range is empty")
    require(rows == [f"{sha} {parent}" for sha, parent in zip(
        commits, [base, *commits[:-1]], strict=True,
    )], "Only the approved linear public history may generate notes")
    value = json.loads(manifest.read_text(encoding="utf-8-sig"))
    release = value["release"]
    require(release["version"] == version and release["repository"] == REPOSITORY,
            "Release manifest identity mismatch")
    require(value["application"]["source_revision"] == revision, "Manifest source mismatch")
    package_hash = release["asset_sha256"].lower()
    require(bool(re.fullmatch(r"[0-9a-f]{64}", package_hash)), "Invalid package digest")
    require(run([str(git_cliff), "--version"], source) == f"git-cliff {TOOL_VERSION}",
            "Unexpected git-cliff version")
    command = [str(git_cliff), "--offline", "--no-exec", "--config", str(CONFIG)]
    # Explicit bounds, not --latest: main may already lead the previous release.
    context = json.loads(run([
        *command, "--repository", str(source), "--ignore-tags", ".*", "--tag", version,
        "--context", f"{base}..{revision}",
    ], source))
    actual = [item["id"] for section in context for item in section["commits"]]
    require(len(actual) == len(commits) and set(actual) == set(commits),
            "git-cliff omitted or added commits")
    require(len(context) == 1 and context[0]["version"] == version,
            "git-cliff generated unexpected version sections")
    for item in context[0]["commits"]:
        upstream = any(
            footer["token"] == "Upstream-Sync" and footer["value"] == "true"
            for footer in item.get("footers", [])
        )
        require((item["group"] == "<!-- -1 -->🔄 上游同步") == upstream,
                "Upstream synchronization classification differs from commit marker")
    with tempfile.TemporaryDirectory(prefix="release-notes-") as temporary:
        root = Path(temporary)
        context_path = root / "context.json"
        context_path.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")
        body = run([*command, "--from-context", str(context_path)], source)
        # Uniform Markdown spacing, without changing commit content.
        body = re.sub(r"\n{3,}", "\n\n", body)
        links = re.findall(re.escape(REPOSITORY) + r"/commit/([0-9a-f]{40})\)", body)
        require(len(links) == len(commits) and set(links) == set(commits),
                "Rendered notes must link every public commit exactly once")
        body += (
            f"\n\n### 完整比较\n\n"
            f"[{previous_tag} → {version}]({REPOSITORY}/compare/{previous_tag}...{version})\n\n"
            f"### 安装与使用\n\n"
            f"[启动公告与使用提醒]({REPOSITORY}/blob/{revision}/assets/resource/announcement/01_更新公告.md)"
            f" · [使用手册]({REPOSITORY}/blob/{revision}/guides/README.md)\n\n"
            f"### 可复核信息\n\n"
            f"- 公开源码提交：`{revision}`\n"
            f"- 完整包 SHA-256：`{package_hash}`\n"
            f"- 校验文件：`GakumasHelper-release-{version}.json`、"
            f"`GakumasHelper-checksums-{version}.txt`。\n"
        )
        metadata = {
            "schema_version": 1, "generator": f"git-cliff {TOOL_VERSION}",
            "repository": REPOSITORY, "version": version, "previous_tag": previous_tag,
            "previous_revision": base, "source_revision": revision, "commits": commits,
            "contributors": sorted({c["author"]["name"] for c in context[0]["commits"]}),
            "config_sha256": digest(CONFIG), "manifest_sha256": digest(manifest),
            "notes_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }
        # Private git-cliff context (including author email) is never an output.
        public = root / "public"
        public.mkdir()
        (public / "notes.md").write_text(body, encoding="utf-8", newline="\n")
        (public / "notes.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        validate_tree(public, project_path_predicate=lambda _: True)
        if verify:
            for name in ("notes.md", "notes.json"):
                require((output / name).read_bytes() == (public / name).read_bytes(),
                        f"Frozen {name} differs from regenerated public history")
        else:
            output.mkdir(parents=True, exist_ok=False)
            for name in ("notes.md", "notes.json"):
                (output / name).write_bytes((public / name).read_bytes())
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "manifest", "git-cliff", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("previous-tag", "version", "revision"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--verify", action="store_true")
    result = generate(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
