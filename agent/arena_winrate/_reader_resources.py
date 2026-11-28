"""Bounded static reader resources owned by an observed Maa Challenge task.

The existing bundle/asset manifests identify fixed revisions. Their unchanged
contents are compared directly. A split gallery also depends on the observed
state of its files and the existing gallery/table pins. Factories retain their
existing validation. Images, recognition results and controller/context objects must
never be stored here.
"""

from __future__ import annotations

import json
from typing import Any, Callable
from pathlib import Path
from threading import RLock

try:  # Repository imports and the installed agent entrypoint.
    from ..card_selection.model import sha256_file
    from ..card_selection.embedding import SHARED_UPGRADE_MARKER_FILE, SHARED_UPGRADE_MARKER_SCHEMA
except ImportError:
    from card_selection.model import sha256_file
    from card_selection.embedding import SHARED_UPGRADE_MARKER_FILE, SHARED_UPGRADE_MARKER_SCHEMA

RESOURCE_NAMES = frozenset({
    "catalog", "card_reference", "card_cost_reference", "p_item_reference",
    "card_model_base", "card_model_custom",
})
MAX_ACTIVE_RESOURCE_TASKS = 4


def manifest_identity(root: str | Path, *, verify_pins: bool = False) -> tuple[Any, ...] | None:
    """Unknown/unversioned development inputs remain reader-local."""
    path = Path(root).resolve()
    try:
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    identity = str(path), json.dumps(value, sort_keys=True, separators=(",", ":"))
    split = value.get("schema_version") == SHARED_UPGRADE_MARKER_SCHEMA
    if not split and not verify_pins:
        return identity
    try:
        gallery = value["gallery"]
        files = [(path / gallery["path"], gallery["sha256"])]
        if split:
            files.append((path.parent.parent / SHARED_UPGRADE_MARKER_FILE, gallery["upgrade_marker_table_sha256"]))
        if verify_pins and any(sha256_file(file) != pin.upper() for file, pin in files):
            return None
        paths = [file for file, _ in files]
        paths.extend((path / gallery["class_table_path"], path / value["model"]["path"]))
        states = [file.stat() for file in paths]
        return (*identity, tuple(
            (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)
            for state in states
        ))
    except (OSError, AttributeError, KeyError, TypeError):
        return None


class ReaderResources:
    def __init__(self, task_id: int | None = None) -> None:
        self.task_id = task_id
        self.active = True
        self._entries: dict[str, tuple[object, Any]] = {}
        self._lock = RLock()

    def load(self, name: str, identity: object, factory: Callable[[], Any]) -> tuple[Any, bool]:
        if name not in RESOURCE_NAMES:
            raise ValueError(f"unsupported static reader resource: {name}")
        with self._lock:
            previous = self._entries.get(name)
            reusable = self.active and identity is not None
            if reusable and previous is not None and previous[0] == identity:
                return previous[1], True
        # Factories can check the task's cancellation proxy. Do not hold this
        # lock across them: the terminal callback must be able to clear the
        # pool immediately while a native model load is still returning.
        value = factory()
        with self._lock:
            if self.active and identity is not None:
                # One revision per resource kind, at most six entries per task.
                self._entries[name] = (identity, value)
        return value, False

    def close(self) -> None:
        with self._lock:
            self.active = False
            self._entries.clear()


class ReaderResourcePools:
    def __init__(self) -> None:
        self._tasks: dict[int, ReaderResources] = {}
        self._lock = RLock()

    def start(self, task_id: int) -> None:
        if type(task_id) is not int or task_id <= 0:
            return
        with self._lock:
            if len(self._tasks) < MAX_ACTIVE_RESOURCE_TASKS:
                self._tasks.setdefault(task_id, ReaderResources(task_id))

    def finish(self, task_id: int) -> None:
        with self._lock:
            pool = self._tasks.pop(task_id, None)
            if pool is not None:
                pool.close()

    def for_context(self, context: Any) -> ReaderResources:
        try:
            task_id = context.get_task_job().job_id
        except Exception:
            task_id = None
        with self._lock:
            # A task ID alone is insufficient: its lifecycle must be observed.
            pool = self._tasks.get(task_id) if type(task_id) is int else None
            return pool if pool is not None else ReaderResources()


reader_resource_pools = ReaderResourcePools()
