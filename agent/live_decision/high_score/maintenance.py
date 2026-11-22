"""Maintain verified HIF components across one known, settled action.

Visible changes still need fresh observations. This module only reuses an
unchanged component, or accounts for cards entering/leaving the visible hand.
Hidden removals, generation, turn changes and ambiguous draws require a scan.
"""

import copy
import json
from collections import Counter
from dataclasses import asdict, fields, replace, dataclass

from route.engine import has_single_frame_id

from .planner import _semantic_deck_key
from .contracts import ActionKind, EffectSourceKind


def complete_action_baseline(bundle, baseline):
    """Accept a checked successor or an actually assembled initial observation."""
    if baseline.get('accepted') is not True:
        return False
    checks = baseline.get('checks')
    if checks is not None:
        return isinstance(checks, dict) and bool(checks) and all(v is True for v in checks.values())
    if (baseline.get('event') != 'live_opening_state' or baseline.get('strategy_state_bound') is not True
            or (baseline.get('run_id'), baseline.get('revision')) != (bundle.run_id, bundle.revision)
            or json.dumps(baseline.get('state'), sort_keys=True) != json.dumps(asdict(bundle.state), sort_keys=True)
            or bundle.state.turn != 1 or bundle.state.deck.discard or bundle.state.deck.removed):
        return False
    components = baseline.get('components', {})
    return all(isinstance(components.get(key), dict)
        and components[key].get('accepted') is True and components[key].get('returned_unchanged') is True
        and (components[key].get('run_id'), components[key].get('revision')) == (bundle.run_id, bundle.revision)
        for key in ('cards', 'status', 'items', 'drinks', 'turns'))

_PREINPUT_SOURCE = 'retained_preinput_no_business_change'
_PREINPUT_METADATA = {'observation_source', 'source_epoch', 'preinput_core_frame_ids'}


def _rule_definitions(kernel):
    return (kernel.ruleset_id, *(dict(getattr(kernel, name))
        for name in ('cards', 'drinks', 'items', 'memory_abilities')))


@dataclass(frozen=True)
class PreinputContinuity:
    """In-process snapshot around the input-free policy calculation only."""
    run_id: str
    revision: int
    state: object
    definitions: tuple
    components: dict
    history_card_epoch: dict | None = None
    source_history: dict | None = None
    current_history: dict | None = None


def _preinput_history_matches_core(history, core, *, empty_hand=False):
    """Use the existing complete latest-history and unchanged-Live contract."""
    if not isinstance(history, dict) or not isinstance(core, dict):
        return False
    observation = history.get('observation', {})
    groups = (history, history.get('before', {}), history.get('after', {}))
    return (history.get('accepted') is True and history.get('returned_unchanged') is True
        and isinstance(observation, dict)
        and isinstance(observation.get('lines'), (tuple, list)) and bool(observation['lines'])
        and all(isinstance(line, str) and line for line in observation['lines'])
        and len(core.get('fields', ())) == 6
        and (core.get('families') in ([], ()) if empty_hand else bool(core.get('families')))
        and observation.get('score') == core['fields'][3]
        and all(type(history.get(key, 0)) is int and history.get(key, 0) == 0
                for key in ('execution_submit_count', 'end_turn_submit_count', 'drinks_consumed'))
        and all(isinstance(group, dict) and has_single_frame_id(group.get('frame_ids')) for group in groups)
        and all(len(group.get('fields', ())) == 6
                and tuple(group['fields'][:5]) == tuple(core['fields'][:5])
                and tuple(group['fields'][5]) == tuple(core['fields'][5])
                and isinstance(group.get('families'), (list, tuple))
                and tuple(map(tuple, group['families'])) == tuple(map(tuple, core['families']))
                for group in groups[1:]))


def _verify_preinput_history(proof, core=None):
    source, current = proof.source_history, proof.current_history
    reference_core = source.get('after') if isinstance(source, dict) else None
    empty_hand = not proof.state.deck.hand
    if (not _preinput_history_matches_core(source, reference_core, empty_hand=empty_hand)
            or not _preinput_history_matches_core(current, reference_core, empty_hand=empty_hand)
            or core is not None and not _preinput_history_matches_core(current, core, empty_hand=empty_hand)
            or tuple(current['observation']['lines']) != tuple(source['observation']['lines'])
            or set(current['frame_ids']) & set(source['frame_ids'])):
        raise ValueError('HIF preinput fresh history is unresolved, reused or changed')


def bind_preinput_history(bundle, expected, *, baseline, proof, current, core):
    """Bind one new post-search observation without changing its source ledger."""
    verify_preinput_continuity(bundle, expected, baseline.get('components'), proof)
    if (proof.current_history is not None
            or baseline.get('history_observation') != proof.source_history):
        raise ValueError('HIF preinput history source changed or was already bound')
    bound = replace(proof, current_history=copy.deepcopy(current))
    _verify_preinput_history(bound, core)
    return bound


def capture_preinput_continuity(bundle, baseline):
    if (baseline.get('accepted') is not True
            or (baseline.get('run_id'), baseline.get('revision')) != (bundle.run_id, bundle.revision)
            or any(baseline.get('components', {}).get(key, {}).get('accepted') is not True
                   for key in ('cards', 'status', 'items', 'drinks', 'turns'))):
        raise ValueError('HIF preinput source is not a complete bound observation')
    history_card_epoch = None
    if baseline.get('observation_source') == 'verified_history_resume':
        source = baseline.get('source_epoch')
        cards = baseline['components']['cards']
        if (not isinstance(source, dict) or set(source) != {'run_id', 'revision'}
                or not isinstance(source['run_id'], str) or not source['run_id']
                or type(source['revision']) is not int or source['revision'] < 1
                or cards.get('observation_source') != 'retained_verified_component'
                or cards.get('source_epoch') != source
                or not baseline.get('checks') or not all(v is True for v in baseline['checks'].values())
                or baseline['checks'].get('permanent_card_sources') is not True):
            raise ValueError('HIF preinput history card provenance is incomplete or contradictory')
        history_card_epoch = copy.deepcopy(source)
    history = baseline.get('history_observation')
    source_history = (copy.deepcopy(history) if baseline.get('checks')
        and all(value is True for value in baseline['checks'].values())
        and _preinput_history_matches_core(history, baseline.get('core'),
                                          empty_hand=not bundle.state.deck.hand) else None)
    return PreinputContinuity(bundle.run_id, bundle.revision, bundle.state,
        _rule_definitions(bundle.kernel), copy.deepcopy(baseline['components']), history_card_epoch,
        source_history)


def verify_preinput_continuity(bundle, expected, baseline_components, proof):
    if (not isinstance(proof, PreinputContinuity)
            or (bundle.run_id, bundle.revision) != (proof.run_id, proof.revision)
            or bundle.state != proof.state or expected != proof.state
            or _rule_definitions(bundle.kernel) != proof.definitions
            or baseline_components != proof.components
            or expected.pending_choice or expected.resolution_stack or expected.pending_after_card_ids):
        raise ValueError('HIF preinput state, definitions, or source ledger changed during search')
    if proof.current_history is not None:
        _verify_preinput_history(proof)


def preinput_maintained_components(bundle, expected, *, core, baseline, proof):
    """Keep business evidence; mutable components need post-search history.

    Original component frames, timings and any earlier real maintenance action
    remain untouched. New core frames are a separate continuity check, never
    input credentials or a relabeling of the older component observation.
    """
    from .opening import match_inventory_hand

    if (baseline.get('accepted') is not True
            or (baseline.get('run_id'), baseline.get('revision')) != (bundle.run_id, bundle.revision)):
        raise ValueError('HIF preinput baseline epoch changed')
    verify_preinput_continuity(bundle, expected, baseline.get('components'), proof)
    if (not has_single_frame_id(core.get('frame_ids')) or tuple(core.get('fields', ()))[:5] != (
            expected.turns_remaining, expected.stamina, expected.genki, expected.score, expected.score_multiplier)):
        raise ValueError('HIF preinput fresh core differs from the planned state')
    match_inventory_hand(core.get('families', ()), tuple(
        bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id for ref in expected.deck.hand))
    retained = {}
    keys = ('cards', 'items', 'turns')
    if proof.current_history is not None:
        _verify_preinput_history(proof, core)
        keys += ('status', 'drinks')
    for key in keys:
        source = baseline['components'][key]
        if (source.get('returned_unchanged') is not True
                or (source.get('run_id'), source.get('revision')) != (bundle.run_id, bundle.revision)):
            raise ValueError('HIF preinput component source is unbound: '+key)
        retained[key] = dict(copy.deepcopy(source), observation_source=_PREINPUT_SOURCE,
            source_epoch=dict(run_id=bundle.run_id, revision=bundle.revision),
            preinput_core_frame_ids=list(core['frame_ids']))
    return retained


def verify_preinput_component(key, component, original, *, bundle, proof=None):
    # Called only after the consumer has verified the in-process proof.
    if key in ('status', 'drinks'):
        if not isinstance(proof, PreinputContinuity):
            raise ValueError('HIF mutable preinput component lacks fresh history')
        _verify_preinput_history(proof)
    if (key not in ('cards', 'items', 'turns', 'status', 'drinks')
            or component.get('source_epoch') != dict(run_id=bundle.run_id, revision=bundle.revision)
            or not has_single_frame_id(component.get('preinput_core_frame_ids'))
            or {k: v for k, v in component.items() if k not in _PREINPUT_METADATA}
            != {k: v for k, v in original.items() if k not in _PREINPUT_METADATA}):
        raise ValueError('HIF preinput retained component differs from its verified source: '+key)


def equivalent_settlements(states):
    """Ignore interchangeable card instance labels, never state or effect data."""
    states = tuple(states)
    if not states:
        return False
    first = states[0]
    return all(
        replace(state, deck=first.deck) == first
        and dict((r.instance_id, r) for r in state.deck.all_cards)
        == dict((r.instance_id, r) for r in first.deck.all_cards)
        and _semantic_deck_key(state.deck) == _semantic_deck_key(first.deck)
        for state in states
    )


def _item_counts(before, state):
    counts = {(r.source_id, r.definition_effect_index): 0
              for r in before.scheduled_effects
              if r.source_kind is EffectSourceKind.ITEM and r.remaining_uses == 0}
    counts.update({(r.source_id, r.definition_effect_index): r.remaining_uses
                   for r in state.scheduled_effects if r.source_kind is EffectSourceKind.ITEM})
    return counts


def unchanged_component(key, before, after):
    if key == 'items':
        return before.items == after.items and _item_counts(before, before) == _item_counts(before, after)
    if key == 'drinks':
        return before.drinks == after.drinks
    if key == 'turns':
        return all(getattr(before, name) == getattr(after, name) for name in (
            'turn', 'turns_remaining', 'turn_attributes', 'score_multiplier', 'future_score_multipliers'))
    if key == 'status':
        # Exclude only other components and non-visible continuation bookkeeping.
        # Any new state field therefore invalidates retention unless unchanged.
        excluded = {'score', 'stamina', 'genki', 'max_stamina', 'hand_limit', 'deck',
                    'turn', 'turns_remaining', 'score_multiplier', 'future_score_multipliers',
                    'turn_attributes', 'items', 'drinks', 'created_card_count',
                    'pending_choice', 'pending_after_card_ids', 'resolution_stack', 'scheduled_effects'}
        schedules = lambda state: tuple(r for r in state.scheduled_effects
                                        if r.source_kind is not EffectSourceKind.ITEM)
        return (schedules(before) == schedules(after)
                and all(getattr(before, f.name) == getattr(after, f.name)
                        for f in fields(before) if f.name not in excluded))
    return False


def known_turn_advance(before, after, core):
    """Advance a previously read remaining-turn table using the new HUD turn."""
    return (after.turn == before.turn + 1 and not after.terminal
            and after.turns_remaining == before.turns_remaining - 1
            and after.turn_attributes == before.turn_attributes[1:]
            and bool(before.future_score_multipliers)
            and after.score_multiplier == before.future_score_multipliers[0]
            and after.future_score_multipliers == before.future_score_multipliers[1:]
            and has_single_frame_id(core.get('frame_ids'))
            and tuple(core.get('fields', ()))[:5] == (
                after.turns_remaining, after.stamina, after.genki, after.score, after.score_multiplier))


def _zone_by_id(state):
    return {card.instance_id: zone for zone in ('hand', 'draw_pile', 'discard', 'removed', 'held')
            for card in getattr(state.deck, zone)}


def card_maintenance_action_valid(bundle, action):
    """The source action must identify an existing hand card or drink slot."""
    if action is None:
        return False
    if action.kind is ActionKind.PLAY_CARD:
        return action.card_instance_id in {card.instance_id for card in bundle.state.deck.hand}
    if action.kind is ActionKind.USE_DRINK:
        return (type(action.source_slot) is int and 0 <= action.source_slot < 4
                and action.drink_id in bundle.state.drinks
                and action.card_instance_id is None and action.target_id is None
                and tuple(row for row in bundle.drink_slots or () if row[0] == action.source_slot)
                == ((action.source_slot, action.drink_id),))
    return False


def visible_hand_accounts_for_cards(bundle, state, core, action):
    from .opening import match_inventory_hand

    before = bundle.state
    if (not card_maintenance_action_valid(bundle, action)
            or before.turn != state.turn or before.created_card_count != state.created_card_count
            or before.pending_choice or state.pending_choice or before.resolution_stack or state.resolution_stack
            or {r.instance_id: r for r in before.deck.all_cards} != {r.instance_id: r for r in state.deck.all_cards}):
        return False
    canonical = tuple(bundle.kernel.cards[r.card_id].canonical_card_id or r.card_id for r in state.deck.hand)
    try:
        match_inventory_hand(core['families'], canonical)
    except (KeyError, ValueError):
        return False
    if action.kind is ActionKind.USE_DRINK:
        # A settled drink can change resources/status without changing any
        # card. Reuse only that exact ledger, checked by the consumer too.
        return (before.deck == state.deck
                and not before.pending_after_card_ids and not state.pending_after_card_ids
                and has_single_frame_id(core.get('frame_ids'))
                and tuple(core.get('fields', ()))[:5] == (
                    state.turns_remaining, state.stamina, state.genki,
                    state.score, state.score_multiplier))
    old, new = _zone_by_id(before), _zone_by_id(state)
    observed_hand = {r.instance_id for r in state.deck.hand}
    if action.card_instance_id in observed_hand:
        return False
    # Every move must be either a visible hand departure or an observed draw.
    # A card removed from the hidden deck cannot be established by this proof.
    return old.keys() == new.keys() and all(
        old[key] == new[key]
        or (old[key] == 'hand' and new[key] in ('discard', 'removed'))
        or (old[key] == 'draw_pile' and key in observed_hand and new[key] == 'hand')
        for key in old
    )


def maintained_components(bundle, candidates, *, core, baseline, action):
    """Return reusable components; absent keys are collected by existing readers."""
    from .opening import match_inventory_hand, expected_permanent_inventory_keys

    if (action is None or baseline.get('accepted') is not True
            or (baseline.get('run_id'), baseline.get('revision')) != (bundle.run_id, bundle.revision)
            or not has_single_frame_id(core.get('frame_ids'))):
        return {}
    states = candidates if isinstance(candidates, tuple) else (candidates,)
    plausible = []
    for state in states:
        if tuple(core.get('fields', ()))[:5] != (
                state.turns_remaining, state.stamina, state.genki, state.score, state.score_multiplier):
            continue
        canonical = tuple(bundle.kernel.cards[r.card_id].canonical_card_id or r.card_id for r in state.deck.hand)
        try:
            match_inventory_hand(core.get('families', ()), canonical)
        except ValueError:
            continue
        plausible.append(state)
    if not plausible:
        return {}
    retained = {}
    advanced_turn = False
    for key in ('items', 'drinks', 'turns', 'status'):
        source = baseline.get('components', {}).get(key, {})
        if (key == 'status' and (bundle.expired_start_scores or bundle.expired_start_drinks)
                and any(state.turn != bundle.state.turn for state in plausible)):
            # The last-fired row is visible only during its firing turn.
            # Its disappearance needs a new status read even if no future
            # kernel trigger remains to change the ordinary status key.
            continue
        if (source.get('accepted') is True and source.get('returned_unchanged') is True
                and (source.get('run_id'), source.get('revision')) == (bundle.run_id, bundle.revision)
                and all(unchanged_component(key, bundle.state, state) for state in plausible)):
            retained[key] = copy.deepcopy(source)
    turn_source = baseline.get('components', {}).get('turns', {})
    if ('turns' not in retained and turn_source.get('accepted') is True
            and turn_source.get('returned_unchanged') is True
            and (turn_source.get('run_id'), turn_source.get('revision')) == (bundle.run_id, bundle.revision)
            and all(known_turn_advance(bundle.state, state, core) for state in plausible)):
        retained['turns'] = copy.deepcopy(turn_source)
        retained['turns']['current_turn'] += 1
        # score_percentages is the fixed vocal/dance/visual mapping, not a
        # per-turn sequence. Only the remaining schedule advances.
        for name in ('turn_attributes', 'score_multipliers', 'future_score_multipliers'):
            if name in retained['turns']:
                retained['turns'][name] = retained['turns'][name][1:]
        advanced_turn = True
    # Only one semantic card ledger may fit the fresh visible hand and core.
    if (len({_semantic_deck_key(s.deck) for s in plausible}) == 1
            and all(visible_hand_accounts_for_cards(bundle, s, core, action) for s in plausible)):
        source = baseline.get('components', {}).get('cards', {})
        if (source.get('accepted') is True and source.get('returned_unchanged') is True
                and (source.get('run_id'), source.get('revision')) == (bundle.run_id, bundle.revision)):
            state = plausible[0]
            keys = {key: (card, custom) for key, card, custom in expected_permanent_inventory_keys(bundle, state)}
            # All permanent card identities must already have real source rows.
            source_keys = Counter((row['card_id'], tuple(sorted((int(k), v) for k, v in row['customizations'].items())))
                                  for row in source['cards'])
            if source_keys == Counter(keys.values()):
                cards, counts = [], {}
                for zone, refs in (('手札', state.deck.hand), ('山札', state.deck.draw_pile),
                                   ('捨札', state.deck.discard), ('除外', state.deck.removed)):
                    counts[zone] = len(refs)
                    for index, ref in enumerate(refs):
                        cid, custom = keys[ref.instance_id]
                        cards.append(dict(zone=zone, index=index, card_id=cid, customizations=dict(custom),
                                          source='retained_verified_card',
                                          retained_from=dict(run_id=bundle.run_id, revision=bundle.revision)))
                retained['cards'] = dict(source, cards=cards, zone_counts=counts)
    for key, component in retained.items():
        # This is now evidence for the actual action. The earlier input-free
        # core reference remains only in its original preinput event.
        component.pop('preinput_core_frame_ids', None)
        component.update(observation_source=('maintained_card_ledger' if key == 'cards' else
                         'maintained_turn_schedule' if key == 'turns' and advanced_turn else
                         'retained_unchanged_component'),
                         source_epoch=dict(run_id=bundle.run_id, revision=bundle.revision),
                         input_click_count=0, elapsed_ms=0,
                         maintenance_action=asdict(action),
                         maintenance_frame_ids=list(core['frame_ids']))
    return retained
