"""Deterministic offline root continuations, never a native input binding.

Required MOVE targets and opt-in optional post-card END choices preserve all
original roots. Other voluntary plays remain normal public-tree decisions.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from .choice import _valid_branches
from .kernel import EffectKind, KernelError
from .contracts import ActionKind, GameAction, DecisionKind, StrategyState
from .state_copy import replace
from .validation_cache import validation_session

VERSION = 'task275-deterministic-choice-roots-v1'
STOP_ROOT_VERSION = 'task275-deterministic-post-card-end-roots-v1'
_CHOICES = frozenset((EffectKind.MOVE_SELECTED_TO_HAND, EffectKind.HOLD_SELECTED,
                     EffectKind.FREE_PLAY_SELECTED, EffectKind.PAID_PLAY_SELECTED))
_UNOBSERVED_EVENTS = ('DRAW', 'MOVE_RANDOM', 'REMOVE_RANDOM', 'FREE_PLAY_RANDOM', 'EXCHANGE_HAND')


@dataclass(frozen=True)
class RootChoiceOption:
    action: GameAction
    continuation: tuple[GameAction, ...] = ()
    expected_after_first: StrategyState | None = None


def post_card_end_state(kernel, state, action, *, plan_type):
    """Prove a complete public play boundary without enumerating hidden draws."""
    from .planner import SearchIncomplete
    from .hif_turn_chain_continuation import _NoDrawKernel

    if (action.kind is not ActionKind.PLAY_CARD or state.pending_choice is not None
            or state.resolution_stack or state.terminal):
        return None, 'not_free_card_root'
    probe = _NoDrawKernel(kernel.cards, kernel.drinks, kernel.items,
                          memory_abilities=kernel.memory_abilities)
    if probe.ruleset_id != kernel.ruleset_id:
        return None, 'unsupported_ruleset'
    try:
        branches = probe.transition(state, replace(action, source_slot=None), plan_type=plan_type)
    except SearchIncomplete as error:
        if str(error) != 'turn_chain_draw_domain':
            raise
        return None, 'draw_domain'
    except KernelError as error:
        return None, 'unsupported_transition:' + str(error)
    _valid_branches(branches)
    if len(branches) != 1 or branches[0].probability != 1.:
        return None, 'non_deterministic_first_action'
    if any(event.event_type.startswith(_UNOBSERVED_EVENTS) or 'RANDOM' in event.event_type
           for event in branches[0].events):
        return None, 'draw_or_random_event'
    after = kernel._canonical_probability_state(branches[0].state)
    # HifEnvironment auto-ends when the last action is spent. That is not a
    # public decision where a second explicit END can be chosen.
    if (after.terminal or after.turn != state.turn or after.actions_remaining <= 0
            or after.pending_choice is not None or after.resolution_stack):
        return None, 'not_same_turn_free_action_boundary'
    legal = kernel.legal_actions(after, plan_type=plan_type, decision_kind=DecisionKind.LIVE)
    if sum(action.kind is ActionKind.END_TURN for action in legal) != 1:
        return None, 'end_not_uniquely_legal'
    return after, 'deterministic_optional_end'


def build_stop_after_card_root_options(kernel, request, legal, *, base_options=None,
                                      max_options, deadline):
    """Append the entire optional-END group or keep every original option.

    The public allocation cap is decided before sampling. A deadline failure
    propagates; it never selects a partially discovered subset of extra roots.
    """
    if type(max_options) is not int or max_options < 1:
        raise ValueError('stop_roots_positive_max_options_required')
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('stop_roots_finite_deadline_required')
    def check_budget():
        if time.perf_counter() >= deadline:
            raise TimeoutError('stop_roots_deadline')
    check_budget()
    request.validate()
    if request.ruleset_id != kernel.ruleset_id:
        raise ValueError('stop_root_ruleset_mismatch')
    legal = tuple(legal)
    base = tuple(base_options) if base_options is not None else tuple(RootChoiceOption(a) for a in legal)
    if ({option.action for option in base} != set(legal)
            or len({(option.action, option.continuation) for option in base}) != len(base)):
        raise ValueError('stop_root_base_coverage_changed')
    audit = dict(version=STOP_ROOT_VERSION, reason='no_eligible_card', added=0,
                 count=len(base), base_count=len(base), cap=max_options, actions=[])
    with validation_session(check_budget):
        _complete_legal(kernel, request, legal)
        extra = []
        for action in legal:
            check_budget()
            # Existing required-retrieval roots are preserved verbatim.
            if not any(option.action == action and not option.continuation for option in base):
                continue
            after, reason = post_card_end_state(kernel, request.state, action, plan_type=request.plan_type)
            audit['actions'].append(dict(action=asdict(action), eligible=after is not None, reason=reason))
            if after is not None:
                extra.append(RootChoiceOption(action, (GameAction(ActionKind.END_TURN),), after))
        check_budget()
        if len(base) + len(extra) > max_options:
            audit.update(reason='budget_ineligible', attempted_count=len(base) + len(extra))
            return base, audit
        audit.update(reason='expanded_optional_end' if extra else 'no_eligible_card',
                     added=len(extra), count=len(base) + len(extra))
        return (*base, *extra), audit


def _declared_choices(kernel, state, action):
    if action.kind is ActionKind.USE_DRINK:
        definition = kernel.drinks[action.drink_id]
        # For the direct retrieval/payment drink, no earlier effect can free
        # a hand slot. At capacity the HIF rule is a no-op selection, including
        # at the native input boundary. Card plays are deliberately excluded:
        # playing the source can free a slot before its retrieval effect.
        if (getattr(kernel, 'selected_move_respects_hand_limit', False)
                and len(state.deck.hand) >= state.hand_limit
                and all(not effect.scheduled_effects and effect.kind in {
                    EffectKind.MOVE_SELECTED_TO_HAND, EffectKind.NULLIFY_COST_CARDS,
                    EffectKind.STAMINA, EffectKind.DOUBLE_COST_TURNS,
                } for effect in definition.effects)):
            return ()
    elif action.kind is ActionKind.PLAY_CARD:
        card = next(card for card in state.deck.hand if card.instance_id == action.card_instance_id)
        definition = kernel.cards[card.card_id]
    else:
        return ()
    pending = list(definition.effects)
    choices = []
    while pending:
        effect = pending.pop()
        if effect.kind in _CHOICES:
            choices.append(effect)
        pending.extend(effect.scheduled_effects)
    return tuple(choices)


def _complete_legal(kernel, request, legal):
    raw = tuple(replace(action, source_slot=None) for action in legal)
    expected = kernel.legal_actions(request.state, plan_type=request.plan_type,
                                    decision_kind=request.decision_kind)
    if len(set(legal)) != len(legal) or set(raw) != set(expected):
        raise ValueError('choice_roots_require_complete_legal_actions')
    card_slots = dict(request.card_slots)
    for action in legal:
        action.validate()
        if action.kind is ActionKind.USE_DRINK:
            if (action.source_slot, action.drink_id) not in request.drink_slots:
                raise ValueError('choice_root_drink_slot_not_bound')
        elif action.card_instance_id is not None:
            if action.card_instance_id not in card_slots or action.source_slot != card_slots[action.card_instance_id]:
                raise ValueError('choice_root_card_slot_not_bound')


def build_choice_root_options(kernel, request, legal, *, max_options, deadline):
    """Return all options and an audit; unsupported actions keep their plain root.

Only a declared single retrieval is expanded. Unmodeled or nested selection
semantics keep the ordinary candidate; the normal search still resolves them.
Exceeding the allocation cap rolls back the whole expansion, never its targets.
Timeout propagates to the caller's computation budget rather than changing rules.
"""
    if type(max_options) is not int or max_options < 1:
        raise ValueError('choice_roots_positive_max_options_required')
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('choice_roots_finite_deadline_required')

    def check_budget():
        if time.perf_counter() >= deadline:
            raise TimeoutError('choice_roots_deadline')

    check_budget()
    request.validate()
    if request.ruleset_id != kernel.ruleset_id:
        raise ValueError('choice_root_ruleset_mismatch')
    legal = tuple(legal)
    ordinary = tuple(RootChoiceOption(action) for action in legal)
    audit = dict(reason='ordinary_roots', ordinary_root_count=len(ordinary),
                 option_count=len(ordinary), expanded_root_count=0, actions=[])
    with validation_session(check_budget):
        _complete_legal(kernel, request, legal)
        if len(ordinary) > max_options:
            audit['reason'] = 'budget_ineligible'
            return ordinary, audit
        if request.state.pending_choice is not None:
            audit['reason'] = 'already_pending_choice'
            return ordinary, audit
        options = []
        for action in legal:
            check_budget()
            choices = _declared_choices(kernel, request.state, action)
            row = dict(action=asdict(action), expanded=False, reason='no_declared_choice', target_count=0)
            replacement = (RootChoiceOption(action),)
            if choices:
                if (len(choices) != 1 or choices[0].kind is not EffectKind.MOVE_SELECTED_TO_HAND
                        or choices[0].selection_count != 1 or choices[0].selection_optional
                        or choices[0].repeats != 1):
                    row['reason'] = 'unsupported_or_multiple_declared_choices'
                else:
                    try:
                        # suggest() evaluates the canonical public state. Build
                        # its pending candidate order from that same state;
                        # observed hand and drink slots remain on the request.
                        root_state = kernel._canonical_probability_state(request.state)
                        branches = kernel.transition(root_state, replace(action, source_slot=None),
                                                     plan_type=request.plan_type)
                        check_budget()
                        _valid_branches(branches)
                        if len(branches) != 1 or branches[0].probability != 1.:
                            row['reason'] = 'non_deterministic_first_action'
                        elif any(event.event_type.startswith(_UNOBSERVED_EVENTS) for event in branches[0].events):
                            row['reason'] = 'draw_or_random_event_before_choice'
                        else:
                            after = kernel._canonical_probability_state(branches[0].state)
                            pending = after.pending_choice
                            if (pending is None or pending.action_kind is not ActionKind.MOVE_CARD
                                    or pending.selections_remaining != 1 or pending.selection_optional
                                    or pending.remaining_effect_passes != 0
                                    or after.resolution_stack or after.turn != request.state.turn):
                                row['reason'] = 'not_one_required_move_boundary'
                            else:
                                targets = kernel.legal_actions(after, plan_type=request.plan_type,
                                                               decision_kind=DecisionKind.SECONDARY)
                                check_budget()
                                if (not targets or any(target.kind is not ActionKind.MOVE_CARD for target in targets)
                                        or len(targets) != len(pending.candidate_instance_ids)
                                        or {target.card_instance_id for target in targets}
                                        != set(pending.candidate_instance_ids)):
                                    row['reason'] = 'incomplete_or_non_move_targets'
                                else:
                                    expanded = []
                                    for target in targets:
                                        check_budget()
                                        expanded.append(RootChoiceOption(action,
                                            (replace(target, source_slot=None),), after))
                                    replacement = tuple(expanded)
                                    row.update(expanded=True, reason='deterministic_required_move',
                                               target_count=len(targets))
                    except (KernelError, ValueError) as error:
                        row['reason'] = 'unsupported_transition:' + str(error)
            audit['actions'].append(row)
            options.extend(replacement)
            # Unprocessed first actions must remain represented in this bound.
            remaining = len(ordinary) - len(audit['actions'])
            if len(options) + remaining > max_options:
                audit.update(reason='budget_ineligible', attempted_option_count=len(options) + remaining)
                return ordinary, audit
        check_budget()
        audit.update(option_count=len(options),
                     expanded_root_count=sum(row['expanded'] for row in audit['actions']))
        if audit['expanded_root_count']:
            audit['reason'] = 'expanded_required_choices'
        return tuple(options), audit
