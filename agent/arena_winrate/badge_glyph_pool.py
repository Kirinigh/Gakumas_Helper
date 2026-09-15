"""Trusted, image-free badge examples owned by one Maa Challenge task.

Reader frames never enter this registry. Consumed inferences are kept separately
only to detect later contradictions, never to teach a label. A task end clears
the dictionaries in place, including references held by a departing reader.
"""

from __future__ import annotations

from typing import Any
from threading import RLock
from dataclasses import field, dataclass

# Bounds the existing pairwise fallback comparison, not the number of cards
# that can be read. Saturation keeps exact examples and disables approximate
# transfer: an omitted, differently labelled neighbour must not be ignored.
MAX_BADGE_GLYPH_EXEMPLARS = 96


@dataclass
class BadgeGlyphDomain:
    exemplars: dict[tuple[int, ...], dict[str, Any]] = field(default_factory=dict)
    polluted: set[tuple[int, ...]] = field(default_factory=set)
    # Prior consumed inferences are contradiction evidence only. They never
    # satisfy an exemplar lookup or calibration label coverage.
    consumed_labels: dict[tuple[int, ...], int] = field(default_factory=dict)
    saturated: bool = False
    _consumed_snapshot: tuple = field(default=(), repr=False)
    _consumed_matrix: Any = field(default=None, repr=False)
    _consumed_foreground: Any = field(default=None, repr=False)

    def consumed_comparisons(self, descriptor: tuple[int, ...], count: int):
        """Compare only other labels in one array operation, retaining history.

        Snapshot comparison is linear in the number of immutable entries and
        avoids rebuilding the 216-column matrix while the history is unchanged.
        It also observes direct dictionary changes made by existing callers.
        """

        import numpy as np

        rows = tuple(self.consumed_labels.items())
        invalid = tuple(row for row in rows if type(row[1]) is not int or not 1 <= row[1] <= 9)
        if invalid:
            return invalid, None
        if not rows:
            return (), None
        selected = [index for index, row in enumerate(rows) if row[1] != count]
        if not selected:
            return (), None
        if rows != self._consumed_snapshot or self._consumed_matrix is None:
            self._consumed_matrix = np.asarray([row[0] for row in rows], dtype=np.int16)
            self._consumed_foreground = self._consumed_matrix > 0
            self._consumed_snapshot = rows
        target = np.asarray(descriptor, dtype=np.int16)
        differences = np.abs(self._consumed_matrix[selected] - target)
        target_foreground = target > 0
        foreground = self._consumed_foreground[selected]
        intersection = np.count_nonzero(foreground & target_foreground, axis=1)
        union = np.count_nonzero(foreground | target_foreground, axis=1)
        metrics = {
            "l1_sum": differences.sum(axis=1),
            "mse": np.round((differences * differences).mean(axis=1), 6),
            "q4_changed_cells": np.count_nonzero(differences, axis=1),
            "binary_xor_cells": np.count_nonzero(foreground ^ target_foreground, axis=1),
            "foreground_iou": np.round(intersection / np.maximum(1, union), 6),
        }
        return tuple(rows[index] for index in selected), metrics

    def admit(self, descriptor: tuple[int, ...]) -> bool:
        if descriptor in self.exemplars or descriptor in self.polluted:
            return True
        if len(self.exemplars) + len(self.polluted) >= MAX_BADGE_GLYPH_EXEMPLARS:
            self.saturated = True
            return False
        return True

    def clear(self) -> None:
        self.exemplars.clear()
        self.polluted.clear()
        self.consumed_labels.clear()
        self._consumed_snapshot = ()
        self._consumed_matrix = self._consumed_foreground = None
        self.saturated = False


@dataclass
class BadgeGlyphTaskPool:
    task_id: int | None = None
    active: bool = True
    domains: dict[tuple[int, int], BadgeGlyphDomain] = field(default_factory=dict)

    def domain(self, size: tuple[int, int]) -> BadgeGlyphDomain:
        if not self.active:
            return BadgeGlyphDomain()
        return self.domains.setdefault(size, BadgeGlyphDomain())

    def close(self) -> None:
        self.active = False
        for domain in self.domains.values():
            domain.clear()
        self.domains.clear()


class BadgeGlyphTaskPools:
    def __init__(self) -> None:
        self._tasks: dict[int, BadgeGlyphTaskPool] = {}
        self._lock = RLock()

    def start(self, task_id: int) -> None:
        with self._lock:
            self._tasks.setdefault(task_id, BadgeGlyphTaskPool(task_id))

    def finish(self, task_id: int) -> None:
        with self._lock:
            pool = self._tasks.pop(task_id, None)
            if pool is not None:
                pool.close()

    def for_context(self, context: Any) -> BadgeGlyphTaskPool:
        try:
            task_id = context.get_task_job().job_id
        except Exception:
            task_id = None
        with self._lock:
            # Unknown or standalone entries get a reader-local pool. Never
            # lazily create a global task without its matching end event.
            return self._tasks.get(task_id) or BadgeGlyphTaskPool()


badge_glyph_task_pools = BadgeGlyphTaskPools()
