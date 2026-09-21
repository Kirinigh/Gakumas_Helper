"""Seven-day retention for arena evidence, matching the framework's debug age."""
from __future__ import annotations

import re
import stat
import time
from pathlib import Path

RETENTION_SECONDS = 7 * 24 * 60 * 60
_GROUP_NAME = re.compile(r"(?:probe-)?[0-9a-f]{32}")
_FRAME_NAME = re.compile(r"[0-9]{2}\.png")


def _plain_stat(path: Path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise OSError("linked evidence path")
    return info


def cleanup_arena_evidence(package_root: Path, *, now: float | None = None) -> dict:
    """Only remove old, flat groups in the installed reader-failures directory.

    The installation entry may itself be a junction (the supported current
    launcher). Links below its resolved root are never traversed. A group's
    newest file or directory timestamp decides its age, including partial saves.
    """
    report = {"deleted_groups": 0, "deleted_files": 0, "deleted_bytes": 0,
              "skipped_groups": 0, "errors": 0}
    cutoff = (time.time() if now is None else now) - RETENTION_SECONDS
    try:
        root = Path(package_root).resolve(strict=True)
        folder = root
        for part in (".local", "arena-win-rate", "reader-failures"):
            folder /= part
            try:
                info = _plain_stat(folder)
            except FileNotFoundError:
                return report
            if not stat.S_ISDIR(info.st_mode):
                raise OSError("evidence root is not a directory")
        groups = list(folder.iterdir())
    except OSError:
        report["errors"] += 1
        return report

    for group in groups:
        if not _GROUP_NAME.fullmatch(group.name):
            continue
        try:
            info = _plain_stat(group)
            if not stat.S_ISDIR(info.st_mode):
                report["skipped_groups"] += 1
                continue
            files = list(group.iterdir())
            snapshots = [(path, _plain_stat(path)) for path in files]
            if any(not stat.S_ISREG(item.st_mode) or
                   (path.name != "evidence.json" and not _FRAME_NAME.fullmatch(path.name))
                   for path, item in snapshots):
                report["skipped_groups"] += 1
                continue
            if max([info.st_mtime] + [item.st_mtime for _, item in snapshots]) >= cutoff:
                continue
            # Validate the complete group before deleting anything. Keep metadata
            # until its pictures are removed; a failed deletion can be retried.
            for path, item in sorted(snapshots, key=lambda entry: entry[0].name == "evidence.json"):
                path.unlink()
                report["deleted_files"] += 1
                report["deleted_bytes"] += item.st_size
            group.rmdir()
            report["deleted_groups"] += 1
        except OSError:
            report["errors"] += 1
    return report


def cleanup_arena_evidence_on_startup(logger) -> None:
    """Cleanup and its diagnostic output must never prevent Agent startup."""
    try:
        report = cleanup_arena_evidence(Path(__file__).resolve().parents[2])
        if report["errors"]:
            logger.warning("部分过期竞技场诊断记录未能清理，下次启动将重试；不影响任务运行")
        logger.debug(f"竞技场诊断记录7天清理：{report}")
    except Exception:
        # Even a logging failure must not turn housekeeping into a task failure.
        pass
