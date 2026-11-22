"""One-use planning worlds across an observed deterministic same-turn action.

Reused worlds remain separate from training, but are not fresh independent
validation samples. This in-process helper never stores returns or a policy.
"""
from __future__ import annotations

import time
from dataclasses import replace

from .planner import SearchIncomplete
from .contracts import ActionKind
from .validation_cache import validation_session
from .hif_frozen_policy_continuation import business_action, deterministic_public_successor

VERSION = 'deterministic-validation-cohort-v1'


class DeterministicValidationCohort:
    version = VERSION

    def __init__(self, kernel):
        self.kernel = kernel
        self.reset()

    def reset(self):
        self._link = None
        self._summary = dict(version=VERSION, reused=False, source_plan_index=None,
                             reason='reset', seeds=())

    def prepare(self, request, context):
        """Consume the previous committed link once, including on a mismatch."""
        link, self._link = self._link, None
        reason = 'no_saved_cohort'
        if link is not None:
            if not link['committed']:
                reason = 'not_committed'
            elif link['context'] != context:
                reason = 'context_changed'
            elif (link['after'] != request.state or link['plan_type'] != request.plan_type
                    or link['ruleset_id'] != request.ruleset_id or link['utility'] != request.utility):
                reason = 'public_state_changed'
            elif link['after_slots'] != tuple(request.drink_slots):
                reason = 'drink_slots_changed'
            else:
                reason = 'reused_deterministic_committed_cohort'
        reused = reason == 'reused_deterministic_committed_cohort'
        seeds = link['seeds'] if reused else ()
        self._summary = dict(version=VERSION, reused=reused,
            source_plan_index=link['source_plan_index'] if link is not None else None,
            reason=reason, seeds=seeds)
        return seeds

    def remember(self, request, action, seeds, context, deadline, source_plan_index):
        """Prove one possible next link without replacing this window's summary."""
        self._link = None
        if (not isinstance(seeds, (tuple, list)) or len(seeds) != 4
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
                or len(set(seeds)) != 4):
            raise ValueError('validation_cohort_requires_four_distinct_planning_seeds')
        if type(source_plan_index) is not int or source_plan_index < 0:
            raise ValueError('validation_cohort_invalid_source_plan_index')
        request.validate()
        if request.ruleset_id != self.kernel.ruleset_id:
            return False
        slots = tuple(request.drink_slots)
        if action.kind is ActionKind.USE_DRINK:
            if (action.source_slot, action.drink_id) not in slots:
                raise ValueError('validation_cohort_drink_slot_mismatch')
            after_slots = tuple(pair for pair in slots if pair[0] != action.source_slot)
        else:
            after_slots = slots
        after, _ = deterministic_public_successor(self.kernel, request.state, action,
            plan_type=request.plan_type, deadline=deadline)
        if after is None:
            return False

        # Local import avoids the reanalysis module's MCTS dependency at import
        # time. Its strict kernel rejects sampling before branches are merged.
        from .hif_public_turn_reanalysis import _StrictDeterministicKernel

        def guard():
            if time.perf_counter() >= deadline:
                raise TimeoutError('validation_cohort_deadline')

        guard()
        proof = _StrictDeterministicKernel(self.kernel.cards, self.kernel.drinks,
            self.kernel.items, memory_abilities=self.kernel.memory_abilities)
        with validation_session(guard, reuse_parent=True):
            try:
                branches = proof.transition(request.state, replace(action, source_slot=None),
                                            plan_type=request.plan_type)
            except SearchIncomplete as error:
                if str(error) not in ('public_turn_random_consumption', 'turn_chain_draw_domain'):
                    raise
                guard()
                return False
            guard()
            if (len(branches) != 1 or branches[0].probability != 1.
                    or self.kernel._canonical_probability_state(branches[0].state) != after):
                raise ValueError('validation_cohort_proof_mismatch')
        guard()
        self._link = dict(before=request.state, action=business_action(action), after=after,
            after_slots=after_slots, context=context, seeds=tuple(seeds), committed=False,
            source_plan_index=source_plan_index, plan_type=request.plan_type,
            ruleset_id=request.ruleset_id, utility=request.utility)
        return True

    def committed(self, before, action, after):
        link = self._link
        if link is None:
            return
        if (not link['committed'] and before == link['before']
                and business_action(action) == link['action'] and after == link['after']):
            link['committed'] = True
        else:
            self._link = None

    def summary(self):
        return dict(self._summary)
