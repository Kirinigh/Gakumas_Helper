"""Explicit reuse of complete HIF terminal values, never observations."""
from __future__ import annotations

from threading import Lock

from .hif_terminal_cache import TerminalMarginalCache


class HifSearchSession:
    """Only exact terminal subtrees survive between explicitly bound windows.

    Namespace definitions are immutable rule objects, not content hashes. A
    changed registry, utility, planner or kernel cannot reuse old values.
    Memo entries contain distributions of score increments and resources;
    actions with physical source slots are never stored here.
    """

    def __init__(self, deadline):
        self.deadline = deadline
        self.decisions = 0
        self.memo_hits = 0
        self.namespaces_created = 0
        self._namespaces = {}
        self._binding = Lock()

    def _acquire(self, deadline):
        if not self._binding.acquire(blocking=False):
            raise ValueError('HIF search session is already in use')
        self.deadline = deadline
        self.decisions = self.memo_hits = self.namespaces_created = 0

    def _release(self):
        self._binding.release()

    @property
    def entry_count(self):
        return sum(len(memo) + len(marginals.values) + len(marginals.transitions)
                   for memo, marginals in self._namespaces.values())

    def tables(self, planner, request):
        kernel = planner.kernel
        namespace = (
            id(kernel), type(kernel), kernel.ruleset_id,
            type(planner), planner.planner_version, planner.terminal_cache, planner.complete_horizon,
            request.plan_type, request.utility, request.decision_kind, bool(request.drink_slots),
            *(tuple(sorted(getattr(kernel, name).items()))
              for name in ('cards', 'drinks', 'items', 'memory_abilities')),
        )
        if namespace not in self._namespaces:
            self._namespaces[namespace] = ({}, TerminalMarginalCache(kernel))
            self.namespaces_created += 1
        self.decisions += 1
        return self._namespaces[namespace]

    def clear(self):
        self._namespaces.clear()
