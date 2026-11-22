"""Offline visible action-chain search with paired complete continuations.

Prefixes refer only to currently public cards. Each candidate receives the same
independent planning draws; no environment RNG or future observed hand is read.
The full prefix executes before the continuation policy regains control.
"""
from __future__ import annotations

import math
import time
import random
import itertools
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict

from .hif import HifSenseKernel
from .contracts import ActionKind, GameAction
from .hif_environment import HifEnvironment, SampledHifKernel
from .hif_combo_timing import SenseComboTimingPolicy
from .hif_combo_candidate import _key, _resolve

VERSION = 'hif-visible-chain-paired-trial-v2'
CONFIG = dict(version=VERSION, batch_seconds=900., planning_seed_base=603000000,
    policy_kwargs=dict(planning_draws=4, max_chain_cards=3, window_seconds=30.),
    value_scope='COMPLETE_SAMPLED_CONTINUATIONS_AFTER_VISIBLE_PREFIX',
    planning_sampler='common-card-exponential-race-v1',
    future_random_support_modeled=False)
PARALLEL_CONFIG = deepcopy(CONFIG)
PARALLEL_CONFIG['version'] = VERSION + ':exact-terminal-roots-v4'
PARALLEL_CONFIG['policy_kwargs']['parallel_terminal_roots'] = True
ADAPTIVE_PARALLEL_CONFIG = deepcopy(PARALLEL_CONFIG)
ADAPTIVE_PARALLEL_CONFIG['version'] = VERSION + ':exact-terminal-roots-v5'
ADAPTIVE_PARALLEL_CONFIG['policy_kwargs']['adaptive_terminal_roots'] = True


class PrefixUnavailable(ValueError):
    """A proposed visible chain cannot finish in this sampled current turn."""


class PairedPlanningKernel(SampledHifKernel):
    """Common random priorities reduce draw noise between candidate orders.

Independent exponential races select each semantic group with its original
kernel probability. Adding a trouble card does not shift every other group's
categorical interval. This sampler is only for planning, never evaluation.
"""

    def __init__(self, kernel, seed):
        super().__init__(kernel, random.Random(seed))
        self.planning_seed = seed
        self.draw_index = 0

    def _draw_candidates(self, deck):
        candidates = HifSenseKernel._draw_candidates(deck)
        draw_index = self.draw_index
        self.draw_index += 1
        if len(candidates) == 1:
            return candidates
        def priority(candidate):
            card, probability = candidate
            rng = random.Random(repr((self.planning_seed, draw_index, card.semantic_key)))
            return -math.log(1. - rng.random()) / probability
        card, _ = min(candidates, key=priority)
        return ((card, 1.),)


class HifChainCandidatePolicy(SenseComboTimingPolicy):
    """Enumerate visible card orders without identifying any preparation card.

The existing approved timing trial supplies drink/combo boundaries and the
post-prefix continuation. This candidate has no native/production loader.
"""

    version = VERSION

    def __init__(self, kernel, baseline, *, planning_seed, deadline,
                 planning_draws=4, max_chain_cards=3, window_seconds=30.,
                 parallel_terminal_roots=False, adaptive_terminal_roots=False):
        if (type(planning_seed) is not int or not math.isfinite(deadline)
                or type(planning_draws) is not int or planning_draws < 2
                or type(max_chain_cards) is not int or not 2 <= max_chain_cards <= 4
                or not math.isfinite(window_seconds) or window_seconds <= 0
                or type(parallel_terminal_roots) is not bool
                or type(adaptive_terminal_roots) is not bool
                or adaptive_terminal_roots and not parallel_terminal_roots):
            raise ValueError('invalid_visible_chain_budget')
        self.planning_seed, self.deadline = planning_seed, deadline
        self.planning_draws, self.max_chain_cards = planning_draws, max_chain_cards
        self.window_seconds = window_seconds
        self.parallel_terminal_roots = parallel_terminal_roots
        self.adaptive_terminal_roots = adaptive_terminal_roots
        if parallel_terminal_roots:
            from .hif_budget import TieredHifPlanner
            from .hif_hybrid import HybridHifPolicy, VisibleHifCandidatePolicy

            if (type(baseline) is not VisibleHifCandidatePolicy
                    or type(baseline.baseline) is not HybridHifPolicy
                    or type(baseline.baseline.endgame) is not TieredHifPlanner):
                raise ValueError('parallel_terminal_requires_fixed_baseline')
            baseline.baseline.endgame.parallel_terminal_roots = True
            baseline.baseline.endgame.adaptive_terminal_roots = adaptive_terminal_roots
            self.version = PARALLEL_CONFIG['version']
            if adaptive_terminal_roots:
                self.version = ADAPTIVE_PARALLEL_CONFIG['version']
        super().__init__(kernel, baseline)

    def reset(self):
        super().reset()
        self._rng = random.Random(self.planning_seed)
        self.plans = []
        self._pending_prefix = ()
        self._prefix_turn = None

    def _budget(self):
        if time.perf_counter() >= self._window_deadline:
            raise ValueError('visible_chain_planning_incomplete')

    def _terminal_search_session(self):
        """Share complete terminal tables only for the fixed baseline stack."""
        from .hif_budget import TieredHifPlanner
        from .hif_hybrid import HybridHifPolicy, VisibleHifCandidatePolicy
        from .hif_planner import HifActionChainPlanner
        from .hif_teachers import VisibleHifContinuationWrapper

        visible = self.baseline
        if (type(self.kernel) is not HifSenseKernel
                or type(visible) is not VisibleHifCandidatePolicy
                or visible.kernel is not self.kernel
                or visible.wrapper_type is not VisibleHifContinuationWrapper
                or getattr(visible.decide, '__func__', None) is not VisibleHifCandidatePolicy.decide):
            return nullcontext()
        hybrid = visible.baseline
        if (type(hybrid) is not HybridHifPolicy or hybrid.kernel is not self.kernel
                or getattr(hybrid.decide, '__func__', None) is not HybridHifPolicy.decide):
            return nullcontext()
        endgame = hybrid.endgame
        if (type(endgame) is not TieredHifPlanner or endgame.kernel is not self.kernel
                or not endgame.config.require_terminal_leaf or endgame.complete_horizon
                or endgame.advice is not None or endgame.use_learned_value
                or endgame._search_session is not None
                or getattr(endgame.decide, '__func__', None) is not TieredHifPlanner.decide
                or getattr(endgame.search_session, '__func__', None)
                    is not HifActionChainPlanner.search_session):
            return nullcontext()
        return endgame.search_session(deadline=self._window_deadline)

    def sequences(self, request, legal):
        """All visible card permutations, plus existing exact retrieval pairs.

Do not classify preparation using printed EXTRA_ACTION: an item or memory can
grant the action instead. Sampled full-state execution establishes legality.
Drinks/selection do not consume the card-count depth.
"""
        state = request.state
        cards = tuple(GameAction(ActionKind.PLAY_CARD, card.instance_id)
                      for card in sorted(state.deck.hand, key=lambda card: card.instance_id))
        sequences = []
        pairs = self._combo_prefixes(request, legal)
        retrieval_roots = {_key(seq[0]) for seq in pairs}
        roots = tuple(action for action in legal if not (
            action.kind is ActionKind.USE_DRINK and (
                action.drink_id == '22' and _key(action) not in retrieval_roots
                or action.drink_id == '12' and state.turns_remaining > 4)))
        # Retrieval must carry its explicit target through the whole prefix.
        # A naked drink root hands an ambiguous required choice to the old
        # continuation when the complete pair is already held. Keep all full
        # targeted prefixes below, including those retrieving a third card.
        sequences.extend((root,) for root in roots if not (
            root.kind is ActionKind.USE_DRINK and root.drink_id == '22'))
        drinks = tuple(root for root in roots if root.kind is ActionKind.USE_DRINK
                       and root.drink_id != '22')
        for size in range(2, min(self.max_chain_cards, len(cards)) + 1):
            for sequence in itertools.permutations(cards, size):
                if any(_key(sequence[0]) == _key(root) for root in roots):
                    sequences.append(sequence)
        for drink in drinks:
            for size in range(1, min(self.max_chain_cards, len(cards)) + 1):
                sequences.extend((drink, *chain) for chain in itertools.permutations(cards, size))
        sequences.extend(pairs)
        unique = {}
        for sequence in sequences:
            key = tuple(_key(action) for action in sequence)
            unique.setdefault(key, tuple(sequence))
        return tuple(unique.values())

    def _simulate(self, request, sequence, seed, cache, stats):
        self._budget()
        env = HifEnvironment(self.kernel)
        current = env.reset(request.state, seed=seed, drink_slots=request.drink_slots)
        env._sampler = PairedPlanningKernel(self.kernel, seed)
        continuation = SenseComboTimingPolicy(self.kernel, self.baseline)
        turn = request.state.turn
        for planned in sequence:
            self._budget()
            if current.state.terminal or current.state.turn != turn:
                raise PrefixUnavailable('prefix_crossed_current_turn')
            legal = env.legal_actions()
            matches = [action for action in legal if _key(action) == _key(planned)]
            if len(matches) != 1:
                raise PrefixUnavailable('prefix_next_action_unavailable')
            before = current.state
            current = env.step(matches[0]).request
            continuation.committed(before, matches[0], current.state)
            if len(current.state.deck.hand) > min(5, current.state.hand_limit):
                raise PrefixUnavailable('prefix_exceeded_observable_hand')
        # Same public state AND same advanced planning RNG has the same future.
        # The cache is per window; no sampled result is exposed to live binding.
        key = (current.state, env._drink_slots, seed, env._sampler._rng.getstate(),
               env._sampler.draw_index)
        if key in cache:
            stats['continuation_cache_hits'] += 1
            return cache[key]
        for _ in range(256):
            self._budget()
            if current.state.terminal:
                score = float(current.state.score)
                cache[key] = score
                stats['complete_continuations'] += 1
                return score
            legal = env.legal_actions()
            action = continuation.choose(current, legal)
            if action not in legal:
                raise ValueError('visible_chain_continuation_illegal')
            before = current.state
            current = env.step(action).request
            continuation.committed(before, action, current.state)
        raise ValueError('visible_chain_continuation_step_limit')

    def choose(self, request, legal):
        request.validate()
        if not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id:
            raise ValueError('visible_chain_capability_or_ruleset_not_ready')
        state = request.state
        if self._pending_prefix:
            if state.turn == self._prefix_turn:
                matches = [action for action in legal
                           if _key(action) == _key(self._pending_prefix[0])]
                if len(matches) == 1:
                    return matches[0]
            self.interventions.append(dict(turn=state.turn, reason='visible_chain_replan_after_changed_legality'))
            self._pending_prefix = ()
        before_count = len(self.interventions)
        original = super().choose(request, legal)
        reason = (self.interventions[-1]['reason']
                  if len(self.interventions) > before_count else None)
        if (state.turns_remaining == 1 or state.pending_choice or state.resolution_stack
                or state.double_card_effects_remaining
                or reason in ('legal_20_first_turn', 'legal_12_last_four_turns',
                              'deterministic_165_covers_remaining_turns')):
            return original
        if len(state.deck.hand) < 2:
            return original
        started = time.perf_counter()
        self._window_deadline = min(self.deadline, started + self.window_seconds)
        sequences = self.sequences(request, legal)
        seeds = tuple(self._rng.sample(range(2**31), self.planning_draws))
        plan = dict(turn=state.turn, seeds=seeds, rows=[], accepted=False,
                    scope='VISIBLE_CHAIN_PAIRED_SAMPLED_ESTIMATE',
                    original_action=asdict(original), optimality_proven=False,
                    complete_continuations=0, continuation_cache_hits=0)
        self.plans.append(plan)
        cache = {}
        try:
            with self._terminal_search_session():
                for index, sequence in enumerate(sequences):
                    row = dict(sequence_index=index, sequence=[asdict(a) for a in sequence],
                               scores=[], feasible=True)
                    plan['rows'].append(row)
                    for seed in seeds:
                        try:
                            row['scores'].append(self._simulate(request, sequence, seed, cache, plan))
                        except PrefixUnavailable as error:
                            row.update(feasible=False, rejection=str(error))
                            break
                    if row['feasible']:
                        row['mean'] = sum(row['scores']) / len(seeds)
            # Cleanup belongs to this window too; never commit a late result.
            self._budget()
            viable = [row for row in plan['rows'] if row['feasible']]
            if not viable:
                raise ValueError('visible_chain_no_complete_candidate')
            best = max(row['mean'] for row in viable)
            leaders = [row for row in viable if row['mean'] == best]
            # Equal complete sampled values retain the baseline root where
            # possible; within that root keep the shortest complete prefix.
            leaders.sort(key=lambda row: (
                _key(sequences[row['sequence_index']][0]) != _key(original),
                len(sequences[row['sequence_index']]), row['sequence_index']))
            selected = leaders[0]
            sequence = sequences[selected['sequence_index']]
            self._pending_prefix, self._prefix_turn = sequence, state.turn
            plan.update(accepted=True, estimated_score=best,
                        selected_sequence=[asdict(a) for a in sequence],
                        feasible_candidates=len(viable), candidate_count=len(sequences))
            return _resolve(sequence[0], legal)
        finally:
            plan['elapsed_ms'] = (time.perf_counter() - started) * 1000

    def committed(self, before, action, after):
        super().committed(before, action, after)
        if self._pending_prefix:
            if _key(action) != _key(self._pending_prefix[0]):
                raise ValueError('visible_chain_commit_mismatch')
            self._pending_prefix = self._pending_prefix[1:]
            if after.turn != before.turn or after.terminal:
                self._pending_prefix = ()
