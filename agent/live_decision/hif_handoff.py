"""Run the high-score callback under its existing exclusive game input lock.

The session lease grants ownership only. The Live observer remains responsible
for fresh business evidence, input endpoint elevation and the 1000ms limit.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from route.ledger import restore_ledger
from route.ownership import HifInputSession
from route.battle_policy import FORMAL_PROFILES

from .hif_session import resolve_hif_round_context
from .hif_capture_adapter import adapt_hif_context
from .hif_current_battle_session import CurrentBattleSession


@dataclass(frozen=True)
class HifBattleHandoff:
    run_id: str
    handoff_id: str
    ownership_revision: int
    game_pid: int
    round_number: int
    policy_profile: str
    idol: str | None
    plan: str
    ledger: object | None
    source_frame_ids: tuple[str, ...]
    prior_terminal_source: str | None


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def consume_hif_handoff(session, *, card_catalog, drink_ids):
    """Require a live shared lease plus its complete, settled inventory."""
    from .hif_profiles import supported_hif_identity

    if (not isinstance(session, (HifInputSession, CurrentBattleSession)) or session.owner != 'task275'
            or session.authorise_input() is not True):
        raise ValueError('HIF battle input session is not active')
    record = session.record
    handoff = record['handoff']
    owner = record['ownership']
    if isinstance(session, CurrentBattleSession):
        return HifBattleHandoff(session.run_id, session.handoff_id, owner['revision'], session.game_pid,
            handoff['round_number'], handoff['policy_profile'], None, handoff['plan'], None, (), None)
    required = {'handoff_id', 'to_owner', 'round_number', 'policy_profile', 'idol', 'plan',
        'max_stamina', 'source_frame_ids', 'prior_terminal_source', 'ledger_revision'}
    # The route envelope can supply original entry identity. It cannot grant
    # a Live input or supply the current playable inventory.
    optional = {'live_context', 'nonbattle_context'}
    if (record['run_id'] != session.run_id or owner['phase'] != 'battle'
            or not isinstance(handoff, dict) or not required <= set(handoff)
            or set(handoff) - required - optional
            or ('nonbattle_context' in handoff and not isinstance(handoff['nonbattle_context'], dict))
            or handoff['to_owner'] != 'task275' or handoff['handoff_id'] != session.handoff_id
            or type(handoff['round_number']) is not int or handoff['round_number'] not in (1, 2)
            or handoff['policy_profile'] not in FORMAL_PROFILES
            or not _text(handoff['idol']) or not _text(handoff['plan'])
            or (handoff['prior_terminal_source'] is not None if handoff['round_number'] == 1
                else not _text(handoff['prior_terminal_source']))):
        raise ValueError('HIF battle handoff is incomplete or unsupported')
    supported_hif_identity(handoff['idol'], handoff['plan'])
    frames = handoff['source_frame_ids']
    if (not isinstance(frames, (tuple, list)) or not frames
            or any(not _text(frame) for frame in frames) or len(set(frames)) != len(frames)):
        raise ValueError('HIF battle source frames are incomplete')
    ledger = restore_ledger(record['ledger'], card_catalog=card_catalog, drink_ids=drink_ids)
    state = ledger.state
    if (ledger.pending is not None or state.run_id != session.run_id or not state.cards
            or type(handoff['ledger_revision']) is not int or handoff['ledger_revision'] != state.revision
            or type(handoff['max_stamina']) is not int or handoff['max_stamina'] != state.max_stamina):
        raise ValueError('HIF battle ledger differs from its handoff')
    return HifBattleHandoff(session.run_id, session.handoff_id, owner['revision'], session.game_pid,
        handoff['round_number'], handoff['policy_profile'], handoff['idol'], handoff['plan'], state,
        tuple(frames), handoff['prior_terminal_source'])


def bind_hif_battle_handoff(context, session, *, card_catalog, drink_ids):
    """Bind the consumer before any recovery navigation or battle input.

    Call inside the HifInputSession context manager and keep that manager open
    throughout the Maa invocation. No screen coordinates or input credentials
    are read from the route record. The current Live opening is observed anew.
    """
    if any(getattr(context, name, None) is not None for name in (
            'gkh_hif_battle_handoff', 'gkh_hif_live_context', 'gkh_hif_opening_state',
            'gkh_hif_committed_state', 'gkh_hif_mcts_policy', 'gkh_hif_terminal_event')):
        raise ValueError('HIF battle handoff needs a fresh runtime context')
    handoff = consume_hif_handoff(session, card_catalog=card_catalog, drink_ids=drink_ids)
    guarded = adapt_hif_context(context, input_guard=session.authorise_input)
    guarded.gkh_hif_battle_handoff = handoff
    guarded.gkh_hif_input_session = session
    guarded.gkh_hif_handoff_opening = None
    return guarded


def require_hif_handoff(context, params):
    """Keep the explicit policy while allowing its normal observation components."""
    from .opening_observer import OPENING_MODES

    handoff = getattr(context, 'gkh_hif_battle_handoff', None)
    session = getattr(context, 'gkh_hif_input_session', None)
    if handoff is None and session is None:
        return None
    mode = params.get('live_decision_mode')
    # These components only observe/navigate; opening_component_params removes
    # the policy because none makes a play decision. They still need this same
    # active lease, plan and process, and cannot request a replacement policy.
    component = (mode in ({value for _, value in OPENING_MODES} | {'drink_reset'})
        and 'hif_policy_profile' not in params)
    if (not isinstance(handoff, HifBattleHandoff) or not isinstance(session, (HifInputSession, CurrentBattleSession))
            or session.authorise_input() is not True or session.owner != 'task275'
            or (session.run_id, session.handoff_id, session.game_pid)
            != (handoff.run_id, handoff.handoff_id, handoff.game_pid)
            or session.ownership_snapshot['revision'] != handoff.ownership_revision
            or params.get('expected_plan') != handoff.plan
            or not (component or (mode in ('opening_state_observe', 'opening_execute')
                and params.get('hif_policy_profile') == handoff.policy_profile))):
        raise ValueError('HIF current input owner or explicit battle mode changed')
    return handoff


def verify_hif_handoff_opening(context, event):
    """Compare permanent identity groups and physical drink slots, not old UI IDs."""
    handoff = getattr(context, 'gkh_hif_battle_handoff', None)
    if handoff is None:
        return
    bundle, run = context.gkh_hif_opening_state, context.gkh_hif_live_context
    previous = getattr(context, 'gkh_hif_handoff_opening', None)
    if previous is not None:
        if ((previous['live_run_id'], previous['live_revision']) != (bundle.run_id, bundle.revision)
                or previous['handoff_id'] != handoff.handoff_id):
            raise ValueError('HIF opening handoff belongs to another battle epoch')
        return
    if (event.get('accepted') is not True or event.get('event') != 'live_opening_state'
            or (event.get('run_id'), event.get('revision')) != (bundle.run_id, bundle.revision)
            or (run.run_id, run.revision) != (bundle.run_id, bundle.revision)
            or run.plan != handoff.plan
            or handoff.ledger is not None and (run.idol, run.max_stamina) != (handoff.idol, handoff.ledger.max_stamina)
            or bundle.state.turn != 1 or bundle.state.score != 0
            or bundle.state.deck.discard or bundle.state.deck.removed):
        raise ValueError('HIF handoff requires the complete untouched current opening')
    number, _ = resolve_hif_round_context(getattr(context, 'gkh_hif_round_context', None),
        run_id=bundle.run_id, revision=bundle.revision)
    if number != handoff.round_number or context.gkh_hif_context_game_pid != handoff.game_pid:
        raise ValueError('HIF observed round/process contradicts the handoff')
    if handoff.ledger is None:
        if not bundle.inventory_keys or bundle.drink_slots is None:
            raise ValueError('HIF current opening inventory is incomplete')
        context.gkh_hif_input_session.bind_live_context(run_id=run.run_id, revision=run.revision,
            idol=run.idol, plan=run.plan, max_stamina=run.max_stamina, round_number=number)
        context.gkh_hif_handoff_opening = dict(handoff_id=handoff.handoff_id, route_run_id=handoff.run_id,
            ledger_revision=None, live_run_id=bundle.run_id, live_revision=bundle.revision,
            round_number=number, policy_profile=handoff.policy_profile, opening_inventory_source='current_complete_opening',
            prebattle_stamina=None, observed_opening_stamina=bundle.state.stamina,
            permanent_inventory_verified=None, drink_slots_verified=True)
        return
    expected = Counter((card.card_id, tuple(sorted((int(k), v) for k, v in card.customization)))
                       for card in handoff.ledger.cards)
    actual = Counter((card_id, tuple(customization)) for _, card_id, customization in bundle.inventory_keys)
    generated = []
    if actual != expected or len(bundle.inventory_keys) != handoff.ledger.card_count:
        generated = _observed_stage_start_additions(handoff, bundle, event, expected, actual)
    drinks = tuple(sorted((drink.slot, drink.drink_id) for drink in handoff.ledger.drinks))
    if bundle.drink_slots is None or tuple(sorted(bundle.drink_slots)) != drinks:
        raise ValueError('HIF opening physical drink slots contradict the route ledger')
    context.gkh_hif_input_session.bind_live_context(run_id=run.run_id, revision=run.revision,
        idol=run.idol, plan=run.plan, max_stamina=run.max_stamina, round_number=number)
    # Entry effects may change stamina. The Live assembler proves the current
    # resources; the route value remains explicitly pre-battle, never reused as
    # the new state or changed into an inferred zero-cost transition.
    context.gkh_hif_handoff_opening = dict(handoff_id=handoff.handoff_id, route_run_id=handoff.run_id,
        ledger_revision=handoff.ledger.revision, live_run_id=bundle.run_id, live_revision=bundle.revision,
        round_number=number, policy_profile=handoff.policy_profile,
        prebattle_stamina=handoff.ledger.stamina, observed_opening_stamina=bundle.state.stamina,
        permanent_inventory_verified=True, drink_slots_verified=True)
    if generated:
        context.gkh_hif_handoff_opening['stage_start_generation'] = dict(
            item_id='428', source='current_complete_opening', observed_only=True,
            permanent_card_count=handoff.ledger.card_count, live_card_count=len(bundle.inventory_keys),
            generated_cards=generated)


def _observed_stage_start_additions(handoff, bundle, event, expected, actual):
    """Explain current basic-card additions without changing the route ledger."""
    from .high_score.opening_items import STAGE_START_ITEM_ID, bind_observed_stage_start_item

    additions = actual - expected
    item_component = event.get('components', {}).get('items', {})
    rows = item_component.get('identity_count_observation', {}).get('items', ())
    stage_items = [row for row in rows if row.get('item_id') == STAGE_START_ITEM_ID]
    if (expected - actual or not additions or handoff.ledger.card_count >= 22
            or sum(actual.values()) != 22 or sum(additions.values()) != 22-handoff.ledger.card_count
            or len(stage_items) != 1 or item_component.get('accepted') is not True
            or item_component.get('returned_unchanged') is not True
            or (item_component.get('run_id'), item_component.get('revision')) != (bundle.run_id, bundle.revision)
            or STAGE_START_ITEM_ID not in bundle.state.items):
        raise ValueError('HIF permanent inventory contradicts the route ledger')
    bind_observed_stage_start_item(stage_items[0], definition=bundle.kernel.items.get(STAGE_START_ITEM_ID))
    generated = []
    for (card_id, customization), count in sorted(additions.items()):
        card = bundle.kernel.cards.get(card_id)
        if (customization or card is None or '基本' not in card.name
                or card.base_card_id != card_id or card.source_type != 'default'
                or card.plan is not None and card.plan.value.casefold() != handoff.plan):
            raise ValueError('HIF stage-start addition is not an observed plain basic card')
        generated.append(dict(card_id=card_id, count=count, customizations={}))
    return generated


def return_hif_battle_handoff(context, *, live_return, terminal_source=None, terminal_runtime=None):
    """Return ownership only after the normal transaction/terminal callback."""
    session = getattr(context, 'gkh_hif_input_session', None)
    event = getattr(context, 'gkh_hif_terminal_event', None)
    bound = getattr(context, 'gkh_hif_handoff_opening', None)
    if (live_return is not True or not isinstance(session, (HifInputSession, CurrentBattleSession))
            or session.authorise_input() is not True or not isinstance(bound, dict)
            or not isinstance(event, dict) or event.get('accepted') is not True
            or event.get('terminal') is not True
            or (event.get('run_id'), event.get('revision'), event.get('round_number'))
            != (bound['live_run_id'], bound['live_revision'], bound['round_number'])):
        raise ValueError('HIF return requires this battle completed by its Live observer')
    if terminal_source is None and terminal_runtime is None:
        session.return_to_nonbattle(event)
    else:
        session.return_to_nonbattle(event, terminal_source=terminal_source, terminal_runtime=terminal_runtime)
