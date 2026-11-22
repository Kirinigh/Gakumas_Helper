"""Bounded turn advice where an installed score can use the last action."""
from __future__ import annotations

from .contracts import ActionKind, SchedulePhase
from .validation_cache import validation_session, check_search_budget
from .hif_card_use_forecast import _pure_card_score
from .hif_turn_chain_continuation import TurnChainContinuation

VERSION = 'installed-score-last-action-chain-v1'
MULTI_ACTION_VERSION = 'installed-score-multi-action-chain-v1'
FOCUS_VERSION = 'installed-score-focus-projection-chain-v1'
MULTI_ACTION_FOCUS_VERSION = 'installed-score-multi-action-focus-projection-chain-v1'


def helper_version(*, multi_action=False, focus_projection=False):
    if type(multi_action) is not bool or type(focus_projection) is not bool:
        raise ValueError('installed_score_modes_require_bool')
    if focus_projection:
        return MULTI_ACTION_FOCUS_VERSION if multi_action else FOCUS_VERSION
    return MULTI_ACTION_VERSION if multi_action else VERSION


class InstalledScoreTurnChain:
    """Reuse the ordinary complete-turn comparison in one observed domain.

    The caller supplies its full legal action set. No drinks are masked, no
    future sample is introduced and installed-card leaf bonuses stay off.
    Structural chain declines return to the caller's existing timing advice.
    """

    version = VERSION

    def __init__(self, kernel, *, installed_score_multi_action=False,
                 installed_score_focus_projection=False):
        if type(installed_score_multi_action) is not bool:
            raise ValueError('installed_score_multi_action_requires_bool')
        if type(installed_score_focus_projection) is not bool:
            raise ValueError('installed_score_focus_projection_requires_bool')
        self.kernel = kernel
        self.installed_score_multi_action = installed_score_multi_action
        self.installed_score_focus_projection = installed_score_focus_projection
        self.version = helper_version(multi_action=installed_score_multi_action,
                                      focus_projection=installed_score_focus_projection)
        self.turn_chain = TurnChainContinuation(kernel,
            end_turn_focus_projection=installed_score_focus_projection)
        self.reset()

    def reset(self):
        self.calls = self.scope_hits = self.scope_declines = 0
        self.chain_accepted = self.chain_declined = 0
        self.turn_chain.start_window()

    def start_window(self):
        self.reset()

    def summary(self):
        return dict(version=self.version, calls=self.calls, scope_hits=self.scope_hits,
                    scope_declines=self.scope_declines, chain_accepted=self.chain_accepted,
                    chain_declined=self.chain_declined, turn_chain=self.turn_chain.summary())

    def _in_scope(self, request, actions):
        state = request.state
        # Existing actual-window wrappers call this method unbound and may
        # forward unknown attributes to their candidate. Their domain stays AP1.
        multi_action = (isinstance(self, InstalledScoreTurnChain)
                        and getattr(self, 'installed_score_multi_action', False))
        if (state.actions_remaining != 1 and not (multi_action and state.actions_remaining > 1)
                or not actions or any(
                action.kind not in (ActionKind.PLAY_CARD, ActionKind.END_TURN) for action in actions)):
            return False
        if state.actions_remaining > 1 and (state.pending_choice is not None
                or state.resolution_stack or state.pending_after_card_ids):
            return False
        hand = {card.instance_id: card for card in state.deck.hand}
        types = set()
        for action in actions:
            if action.kind is ActionKind.PLAY_CARD:
                types.add(self.kernel.cards[hand[action.card_instance_id].card_id].card_type)
        active = False
        for reference in state.scheduled_effects:
            if (reference.trigger_phase is not SchedulePhase.CARD_USED
                    or reference.turns_until_fire != 0 or reference.remaining_uses == 0
                    or reference.remaining_turns == 0 or reference.direct_on_trigger):
                continue
            effect = self.kernel._scheduled_effect_spec(reference)
            if _pure_card_score(effect) and any(effect.trigger_card_type in (None, kind) for kind in types):
                active = True
        return active

    def choose(self, request, actions, deadline):
        from .hif_mcts import action_key

        with validation_session(lambda: self.turn_chain._check_deadline(deadline), reuse_parent=True):
            check_search_budget()
            request.validate()
            if not self._in_scope(request, actions):
                check_search_budget()
                self.calls += 1
                self.scope_declines += 1
                return None
            selected = self.turn_chain.choose(request, actions, deadline)
            if selected is not None and sum(action_key(action) == selected for action in actions) != 1:
                raise ValueError('installed_score_chain_action_not_uniquely_legal')
            check_search_budget()
            # Complete calls only: errors and hard deadlines never become a
            # successful domain decline or a partial best-action suggestion.
            self.calls += 1
            self.scope_hits += 1
            self.chain_accepted += int(selected is not None)
            self.chain_declined += int(selected is None)
            return selected
