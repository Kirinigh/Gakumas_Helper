"""Classify one rule action for a same-turn execution chain, without input.

Only a deterministic result contains a successor. Synchronization results name
the observation scope, never an observed value or an arbitrarily chosen branch.
``random_outcome`` means match the original outcomes using their affected
visible fields; it does not require or pretend to observe hidden deck order.
Every real input still needs its separate fresh target and timing checks.
"""

from dataclasses import dataclass

from .hif import HifSenseKernel
from .choice import _valid_branches
from .planner import SearchIncomplete
from .contracts import ActionKind, StrategyState
from .hif_public_turn_reanalysis import _StrictDeterministicKernel


@dataclass(frozen=True)
class InformationBoundary:
    kind: str
    reason: str
    required_fields: tuple[str, ...]
    successor: StrategyState | None = None


def _synchronize(reason, *fields):
    return InformationBoundary('synchronize', reason, fields)


class _BoundaryKernel(_StrictDeterministicKernel):
    def _draw_one(self, state, *, source_id):
        if not state.deck.draw_pile and not state.deck.discard:
            # The strict planning guard also declines an empty-pile attempt.
            # Its real rule result has no new card and is safe to classify.
            return HifSenseKernel._draw_one(self, state, source_id=source_id)
        return super()._draw_one(state, source_id=source_id)

    def _exchange_hand(self, state, *, source_id):
        raise SearchIncomplete('information_boundary_exchange')


def classify_information_boundary(kernel, state, action, *, plan_type, new_temporary_support=False):
    """Return an exact same-turn successor or a required synchronization scope.

    The caller supplies already bound rules/state and the external indication
    of newly visible temporary enhancement. This is not an execution permit.
    Unknown rule errors and search-budget failures propagate to stop the chain.
    """
    if type(new_temporary_support) is not bool:
        raise ValueError('new temporary support marker must be boolean')
    if state.terminal:
        return _synchronize('terminal', 'terminal_result')
    if new_temporary_support:
        return _synchronize('new_temporary_support', 'hand', 'hand_support')
    if state.pending_choice or state.resolution_stack or state.pending_after_card_ids:
        return _synchronize('pending_resolution', 'choice', 'core')
    if action.kind is ActionKind.END_TURN:
        if state.turns_remaining == 1:
            return _synchronize('terminal_turn_end', 'terminal_result')
        return _synchronize('explicit_turn_end', 'core', 'turns', 'hand', 'hand_support')
    proof = _BoundaryKernel(kernel.cards, kernel.drinks, kernel.items,
        memory_abilities=kernel.memory_abilities, plan_type=kernel.plan_type)
    if proof.ruleset_id != kernel.ruleset_id:
        raise ValueError('information boundary requires the same supported ruleset')
    try:
        branches = proof.transition(state, action, plan_type=plan_type)
    except SearchIncomplete as error:
        reason = str(error)
        if reason == 'turn_chain_draw_domain':
            return _synchronize('draw', 'core', 'hand', 'hand_support')
        if reason == 'information_boundary_exchange':
            return _synchronize('exchange_hand', 'core', 'hand', 'hand_support')
        if reason == 'public_turn_random_consumption':
            return _synchronize('random_before_merge', 'random_outcome')
        raise
    _valid_branches(branches)
    if len(branches) != 1 or branches[0].probability != 1.:
        return _synchronize('non_deterministic_transition', 'random_outcome')
    branch, = branches
    after = branch.state
    if after.pending_choice or after.resolution_stack or after.pending_after_card_ids:
        return _synchronize('pending_resolution', 'choice', 'core')
    if after.terminal:
        return _synchronize('terminal', 'terminal_result')
    if after.turn != state.turn:
        return _synchronize('new_turn', 'core', 'turns', 'hand', 'hand_support')
    events = tuple(event.event_type for event in branch.events)
    if 'EXCHANGE_HAND' in events:
        return _synchronize('exchange_hand', 'core', 'hand', 'hand_support')
    if any(name.startswith('DRAW') and name not in ('DRAW_HAND_FULL', 'DRAW_EMPTY') for name in events):
        return _synchronize('draw', 'core', 'hand', 'hand_support')
    if any('RANDOM' in name for name in events):
        return _synchronize('random_event', 'random_outcome')
    if {card.instance_id for card in after.deck.hand} - {card.instance_id for card in state.deck.hand}:
        return _synchronize('new_hand_instance', 'hand', 'hand_support')
    if set(after.hand_support) - set(state.hand_support):
        return _synchronize('new_temporary_support', 'hand', 'hand_support')
    if after.actions_remaining == 0:
        # The ordinary PLAY_CARD transition stops before the automatic END.
        # Do not return that intermediate state as another playable chain node.
        if after.turns_remaining == 1:
            return _synchronize('automatic_terminal_turn_end', 'terminal_result')
        return _synchronize('automatic_turn_end', 'core', 'turns', 'hand', 'hand_support')
    return InformationBoundary('deterministic_same_turn', 'no_new_information', (), after)
