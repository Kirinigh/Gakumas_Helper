"""Exact, observation-conditioned historical draws; never a planning kernel.

The original kernel still executes every trigger and every ordering of the
observed cards.  This narrow domain has only uniform draws without replacement
as randomness.  Their identities cannot affect control flow, payment, or draw
count.  Conditional probability one is kept separate from P(observed subset).
"""
import copy
import math
from fractions import Fraction
from collections import Counter
from dataclasses import replace, dataclass

from .hif import HifSenseKernel
from .kernel import EffectKind, CardCostKind, ConditionKind
from .contracts import Stance, ActionKind, DeckKnowledge, SchedulePhase
from .validation_cache import _CHECK_BUDGET, validation_session


class _UnsupportedObservation(Exception):
    pass


_SCALAR_EFFECTS = frozenset({
    EffectKind.SCORE, EffectKind.STAMINA, EffectKind.GENKI, EffectKind.FIXED_GENKI,
    EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION, EffectKind.EXCELLENT_CONDITION,
    EffectKind.EXTRA_ACTION, EffectKind.COST_REDUCTION, EffectKind.SCORE_BUFF,
    EffectKind.CONCENTRATION_EFFECT_BUFF, EffectKind.CONCENTRATION_ADDITION_BUFF,
    EffectKind.NULLIFY_GENKI_TURNS, EffectKind.HALF_COST_TURNS, EffectKind.DOUBLE_COST_TURNS,
    EffectKind.NULLIFY_COST_CARDS, EffectKind.DOUBLE_CARD_EFFECT,
})
_CONDITIONS = frozenset({
    ConditionKind.ALL, ConditionKind.TURNS_REMAINING_MAX, ConditionKind.IS_DIRECT_EFFECT,
    ConditionKind.USED_CARD_EFFECT_TAG, ConditionKind.USED_CARD_BASE_ID,
    ConditionKind.USED_CARD_ID, ConditionKind.USED_CARD_SOURCE_TYPE,
    ConditionKind.USED_CARD_RARITY_NOT, ConditionKind.TURN_ATTRIBUTE,
    ConditionKind.REMOVED_BASE_CARD_COUNT_MIN, ConditionKind.CONCENTRATION_MIN,
    ConditionKind.GOOD_CONDITION_MIN, ConditionKind.GOOD_CONDITION_MAX,
    ConditionKind.STAMINA_RATIO_MIN, ConditionKind.EXCELLENT_CONDITION_MIN,
    ConditionKind.GENKI_MIN, ConditionKind.EFFECT_COUNTER_MIN,
    ConditionKind.EFFECT_COUNTER_MOD_EQ,
})


def _condition(condition):
    if condition is not None:
        if condition.kind not in _CONDITIONS:
            raise _UnsupportedObservation('draw-dependent or unsupported condition')
        for child in condition.all_of:
            _condition(child)


def _effect_conditions(effect):
    _condition(effect.condition)
    _condition(effect.schedule_counter_condition)
    if effect.kind is EffectKind.SCHEDULE and effect.schedule_phase is SchedulePhase.CARD_MOVED_TO_HAND:
        raise _UnsupportedObservation('move-to-hand trigger')
    for child in effect.scheduled_effects:
        _effect_conditions(child)


@dataclass(frozen=True)
class ObservedDrawWitness:
    states: tuple
    pool_count: int
    observed_count: int
    event_probability: Fraction
    conditional_probability_mass: float
    draw_candidate_calls: int
    reshuffle: dict | None = None

    def audit(self):
        result = dict(scope='HISTORICAL_OBSERVED_DRAW_SUBSET', pool_count=self.pool_count,
            observed_count=self.observed_count,
            original_event_probability=dict(numerator=self.event_probability.numerator,
                                            denominator=self.event_probability.denominator),
            conditional_probability_mass=self.conditional_probability_mass,
            complete_state_count=len(self.states), draw_candidate_calls=self.draw_candidate_calls)
        if self.reshuffle is not None:
            result['reshuffle'] = self.reshuffle
        return result


class _ObservedDrawKernel(HifSenseKernel):
    def __init__(self, kernel, targets, *, allow_reshuffle=False):
        # Definitions are read-only for this one reconstruction. No registry or
        # source state is changed, and no input evidence enters the facade.
        self.cards, self.items, self.drinks = kernel.cards, kernel.items, kernel.drinks
        self.memory_abilities, self.ruleset_id = kernel.memory_abilities, kernel.ruleset_id
        self.plan_type = kernel.plan_type
        self.targets = frozenset(targets)
        self.inside_end = False
        self.end_results = []
        self.draw_candidate_calls = 0
        self.allow_reshuffle = allow_reshuffle
        self.refill = None

    def transition(self, state, action, *, plan_type):
        old = self.inside_end
        self.inside_end = action.kind is ActionKind.END_TURN
        try:
            result = super().transition(state, action, plan_type=plan_type)
            if self.inside_end:
                self.end_results.append(result)
            elif len(result) != 1 or result[0].probability != 1.:
                raise _UnsupportedObservation('pre-turn action is not deterministic')
            return result
        finally:
            self.inside_end = old

    def _apply_effect(self, branches, effect, *, source_id, **kwargs):
        _effect_conditions(effect)
        if effect.kind not in _SCALAR_EFFECTS | {EffectKind.SCHEDULE, EffectKind.DRAW, EffectKind.FREE_PLAY_RANDOM}:
            raise _UnsupportedObservation('non-draw random or deck-changing effect')
        if effect.kind is EffectKind.DRAW and not self.inside_end:
            raise _UnsupportedObservation('draw before automatic turn advance')
        return super()._apply_effect(branches, effect, source_id=source_id, **kwargs)

    def _free_play_random(self, state, selector, *, source_id):
        # The same one already-removed limited card is replayed irrespective of
        # which cards have been drawn. Arbitrary random plays remain unsupported.
        candidates = tuple(ref for ref in state.deck.all_cards if self._selector_matches(ref, selector))
        if len(candidates) != 1 or candidates[0] not in state.deck.removed:
            raise _UnsupportedObservation('replay target depends on a random card or zone')
        definition = self.cards[candidates[0].card_id]
        if not definition.limited or definition.required_stance is not None:
            raise _UnsupportedObservation('replay can change the future draw pool')
        _condition(definition.condition)
        for effect in (*definition.effects, *definition.persistent_effects):
            _effect_conditions(effect)
        return super()._free_play_random(state, selector, source_id=source_id)

    def _independent_draw_batch(self, state, count, *, source_id):
        # Keep all orders of the observed subset, including their actual trigger
        # phases. The ordinary kernel merges only complete equivalent states.
        return None

    def _draw_one(self, state, *, source_id):
        if not self.inside_end:
            raise _UnsupportedObservation('draw outside the original turn advance')
        if not state.deck.draw_pile:
            # Let the original kernel move the actual discard pile. No draw
            # order, card movement or trigger is manufactured by this facade.
            if not self.allow_reshuffle or not state.deck.discard:
                raise _UnsupportedObservation('draw outside the original nonempty pool')
            pool = state.deck.discard
            wanted = tuple(ref for ref in pool if ref.instance_id in self.targets)
            outstanding = self.targets - {ref.instance_id for ref in state.deck.hand}
            if not wanted or outstanding != {ref.instance_id for ref in wanted}:
                raise _UnsupportedObservation('reshuffle does not contain the remaining observed cards')
            ways = _observed_multiset_ways(pool, wanted)
            refill = (pool, len(wanted), Fraction(ways, math.comb(len(pool), len(wanted))))
            if self.refill is not None and self.refill != refill:
                raise _UnsupportedObservation('reshuffle depends on prior draw order')
            self.refill = refill
        return super()._draw_one(state, source_id=source_id)

    def _draw_candidates(self, deck):
        if deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION or deck.known_top_count:
            raise _UnsupportedObservation('ordered or partially known draw pool')
        original = super()._draw_candidates(deck)
        # Conditional on the final semantic multiset, the next group has
        # probability remaining target count / remaining target total. The
        # original pool group weight is different when only some copies are
        # wanted. Preserve the original kernel's physical representative.
        remaining = Counter(card.semantic_key for card in deck.draw_pile
                            if card.instance_id in self.targets)
        total = sum(remaining.values())
        if not total:
            raise _UnsupportedObservation('actual draw count exceeds the observed subset')
        candidates = []
        for card, _ in original:
            count = remaining[card.semantic_key]
            if count:
                if card.instance_id not in self.targets:
                    raise _UnsupportedObservation('observed copy is not the group representative')
                candidates.append((card, count / total))
        self.draw_candidate_calls += 1
        return tuple(candidates)


def _observed_multiset_ways(pool, wanted):
    """Keep the same canonical physical copies as full grouped enumeration."""
    groups = {}
    for card in pool:
        groups.setdefault(card.semantic_key, []).append(card)
    targets = {card.instance_id for card in wanted}
    ways = 1
    for cards in groups.values():
        # The initial and later canonicalized groups must choose the same
        # representatives; otherwise retain the original full enumeration.
        if cards != sorted(cards, key=lambda card: card.instance_id):
            raise _UnsupportedObservation('noncanonical equivalent-copy order')
        selected = [card for card in cards if card.instance_id in targets]
        if selected != cards[:len(selected)]:
            raise _UnsupportedObservation('observed copies are not a representative prefix')
        ways *= math.comb(len(cards), len(selected))
    return ways


def _deterministic_normal_preview(policy, definition):
    from .live_policy import status_only_resource_effects

    preview = policy.observed_normal_score_preview
    if preview is None:
        return True
    # These are the two existing strictly bound NORMAL preview domains. The
    # original committed_transition still proves its card-only projection is
    # deterministic and applies its complete correction before automatic END.
    kinds = {effect.kind for effect in definition.effects}
    supported = (status_only_resource_effects(definition.effects)
                 or EffectKind.SCORE in kinds and kinds <= {
                     EffectKind.SCORE, EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION})
    return (definition.cost_kind is CardCostKind.NORMAL and supported
            and isinstance(preview, dict) and set(preview) == {'score', 'stamina', 'genki'}
            and all(type(value) is int and value >= 0 for value in preview.values()))


def observed_core_transition(policy, core):
    """Use only an unambiguous revealed hand as a historical draw constraint.

    No score/resource value is copied from the screen. The caller still checks
    the full calculated successor against its independent fresh observation.
    """
    from types import SimpleNamespace

    from .hif_rule_progress import _core

    try:
        core = _core(core)
        before = policy.bundle.state
        if core['fields'][0] != before.turns_remaining-1:
            return None
        pool = (*before.deck.draw_pile, *before.deck.discard, *before.deck.hand)
        chosen = []
        for family in core['families']:
            matches = [ref for ref in pool if ref not in chosen and
                (policy.bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id) in family]
            if not matches or len({ref.semantic_key for ref in matches}) != 1:
                return None
            chosen.append(min(matches, key=lambda ref: ref.instance_id))
        hint = SimpleNamespace(turn=before.turn+1, turns_remaining=before.turns_remaining-1,
                               deck=SimpleNamespace(hand=tuple(chosen)))
        return observed_committed_transition(policy, hint)
    except (ValueError, KeyError, TypeError):
        return None


def observed_committed_transition(policy, observed):
    """Return an exact historical witness, or None to retain full enumeration.

    The supplied observation only restricts final card membership. It cannot
    override any scalar, schedule, or card field computed by the real rules.
    The caller must still compare the entire result with independent components.
    """
    bundle, state = policy.bundle, policy.bundle.state
    kernel = bundle.kernel
    action = policy.response.action if policy.response else None
    if (type(kernel) is not HifSenseKernel or action is None
            or action.kind not in (ActionKind.PLAY_CARD, ActionKind.END_TURN)
            or policy.selected_skill_support is not None
            or action.kind is ActionKind.END_TURN and policy.observed_normal_score_preview is not None
            or state.pending_choice or state.resolution_stack or state.pending_after_card_ids
            or not state.persistent_effects_initialized or state.turns_remaining < 2
            or state.stance is not Stance.NEUTRAL or state.full_power or state.deck.held
            or state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION or state.deck.known_top_count
            or observed.turn != state.turn + 1 or observed.turns_remaining != state.turns_remaining - 1):
        return None
    pool = state.deck.draw_pile
    wanted = observed.deck.hand
    reshuffle = len(wanted) > len(pool)
    if (not 1 < len(wanted) <= state.hand_limit
            or len({ref.instance_id for ref in wanted}) != len(wanted)
            or (not reshuffle and any(ref not in pool for ref in wanted))
            or (reshuffle and (any(ref not in wanted for ref in pool)
                               or any(ref not in state.deck.all_cards for ref in wanted)))):
        return None
    try:
        ways = _observed_multiset_ways(pool, pool if reshuffle else wanted)
        for ref in state.scheduled_effects:
            if ref.remaining_uses == 0 or ref.remaining_turns == 0:
                continue
            if ref.trigger_phase is SchedulePhase.CARD_MOVED_TO_HAND:
                raise _UnsupportedObservation('existing move-to-hand trigger')
            _effect_conditions(kernel._scheduled_effect_spec(ref))
        if action.kind is ActionKind.PLAY_CARD:
            ref, = (ref for ref in state.deck.hand if ref.instance_id == action.card_instance_id)
            definition = kernel.cards[ref.card_id]
            if not _deterministic_normal_preview(policy, definition):
                return None
            _condition(definition.condition)
            for effect in (*definition.effects, *definition.persistent_effects):
                _effect_conditions(effect)
        facade = _ObservedDrawKernel(kernel, (ref.instance_id for ref in wanted),
                                     allow_reshuffle=reshuffle)
        projection = copy.copy(policy)
        projection.bundle = replace(bundle, kernel=facade)
        # Reuse existing local immutable normalizations without carrying them
        # across epochs or changing full observation validation outside here.
        with validation_session(_CHECK_BUDGET.get()):
            result = projection.committed_transition()
        states = result if isinstance(result, tuple) else (result,)
        if len(facade.end_results) != 1:
            return None
        branches = facade.end_results[0]
        probability = math.fsum(branch.probability for branch in branches)
        targets = {ref.instance_id for ref in wanted}
        if reshuffle and facade.refill is None:
            return None
        refill_pool = facade.refill[0] if facade.refill is not None else ()
        remainder = {ref.instance_id for ref in (*pool, *refill_pool)} - targets
        if (len(states) != 1 or not math.isclose(probability, 1., abs_tol=1e-9)
                or any(value.pending_choice or value.resolution_stack
                       or value.turn != state.turn + 1
                       or {ref.instance_id for ref in value.deck.hand} != targets
                       or {ref.instance_id for ref in value.deck.draw_pile} != remainder
                       for value in states)):
            return None
        event_probability = (facade.refill[2] if reshuffle
                             else Fraction(ways, math.comb(len(pool), len(wanted))))
        refill_audit = (dict(exhausted_pool_count=len(pool), refill_pool_count=len(refill_pool),
                            refill_observed_count=facade.refill[1]) if reshuffle else None)
        return ObservedDrawWitness(states, len(pool) + len(refill_pool), len(wanted),
            event_probability, probability, facade.draw_candidate_calls, refill_audit)
    except _UnsupportedObservation:
        return None
