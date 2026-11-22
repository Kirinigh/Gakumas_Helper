"""Bind a submitted action's trusted rule outcomes to fresh core/hand data.

The caller owns candidate provenance: these must be all original outcomes from
the bound policy or formal pending loader, including verified support effects.
This module never replaces them with a base-card simulation. Its receipt is
rule-maintained progress, not a complete independently observed status panel or
permission to send another input. Frame freshness here rejects source reuse;
the separate native reader and input guard own capture timing.
"""

import math
from types import SimpleNamespace
from collections import Counter
from dataclasses import asdict, fields, replace

from .contracts import ActionKind, StrategyState
from ..inventory import match_inventory_hand
from .hif_mcts_cache import json_value, exact_state
from .hif_support_continuity import observed_hand_variant_key
from .hif_information_boundary import classify_information_boundary

VERIFICATION_SCOPE = 'rule_maintained_information_boundary_v1'
PREINPUT_SCOPE = 'rule_maintained_preinput_v1'
_CORE_FIELDS = ('turns_remaining', 'stamina', 'genki', 'score', 'score_multiplier')
_CHECKS = ('submitted', 'baseline_binding', 'action_binding', 'fresh_core',
           'unique_original_candidate', 'hand_binding', 'settled_state', 'action_observed')
_PREINPUT_CHECKS = ('baseline_binding', 'fresh_core', 'same_state', 'hand_binding')
_MAX_BASELINE_DEPTH = 32


def _fail(reason):
    raise ValueError(reason)


def _core(core):
    from ..opening_observer import valid_opening_frame_ids

    if not isinstance(core, dict):
        _fail('core_missing')
    values, frames, families = core.get('fields'), core.get('frame_ids'), core.get('families')
    if (not isinstance(values, (tuple, list)) or len(values) < 5
            or any(type(value) is not int for value in values[:4])
            or type(values[4]) not in (int, float) or not math.isfinite(values[4])
            or not valid_opening_frame_ids(frames)
            or not isinstance(families, (tuple, list)) or len(families) > 5
            or any(not isinstance(family, (tuple, list)) or not family
                   or any(not isinstance(cid, str) or not cid.isdecimal() for cid in family) for family in families)):
        _fail('core_or_frames_invalid')
    return {name: json_value(core[name]) for name in ('fields', 'families', 'frame_ids')}


def _source_frames(value, *, ancestors=frozenset(), depth=0):
    if not isinstance(value, (dict, tuple, list)):
        return set()
    if id(value) in ancestors:
        _fail('baseline_source_cycle')
    if depth > _MAX_BASELINE_DEPTH + 16:
        _fail('baseline_source_too_deep')
    ancestors = ancestors | {id(value)}
    result = set()
    if isinstance(value, dict):
        for name, child in value.items():
            if name.endswith('frame_ids') and isinstance(child, (tuple, list)):
                result.update(frame for frame in child if isinstance(frame, str))
            else:
                result.update(_source_frames(child, ancestors=ancestors, depth=depth+1))
    elif isinstance(value, (tuple, list)):
        for child in value:
            result.update(_source_frames(child, ancestors=ancestors, depth=depth+1))
    return result


def _canonical_hand(bundle, state):
    return tuple(bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id for ref in state.deck.hand)


def _hand_possible(families, ids):
    if len(families) != len(ids):
        return False
    counts = Counter(ids)

    def visit(slot):
        if slot == len(families):
            return True
        for cid in set(families[slot]):
            if counts[cid]:
                counts[cid] -= 1
                matched = visit(slot + 1)
                counts[cid] += 1
                if matched:
                    return True
        return False

    return visit(0)


def _noncustom_hand_families(bundle, state, core, noncustom_slots):
    return [tuple(cid for cid in family if slot not in noncustom_slots or any(
        '@' not in ref.card_id and (bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id)==cid
        for ref in state.deck.hand)) for slot,family in enumerate(core['families'])]


def _ordered_state(bundle, state, core, *, noncustom_slots=()):
    families = _noncustom_hand_families(bundle,state,core,noncustom_slots)
    ids = match_inventory_hand(families, _canonical_hand(bundle, state))
    hand = []
    for cid in ids:
        choices = [ref for ref in state.deck.hand if ref not in hand
            and (bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id) == cid
            and (len(hand) not in noncustom_slots or '@' not in ref.card_id)]
        if not choices or len({observed_hand_variant_key(state, ref) for ref in choices}) != 1:
            _fail('ambiguous_hand_variants')
        hand.append(choices[0])
    return replace(state, deck=replace(state.deck, hand=tuple(hand)))


def _bound_order_matches(bundle, state, core):
    ids = _canonical_hand(bundle, state)
    return len(ids) == len(core['families']) and all(
        cid in family for cid, family in zip(ids, core['families'], strict=True))


def _matches_core(state, core):
    return tuple(core['fields'][:5]) == tuple(getattr(state, name) for name in _CORE_FIELDS)


def _baseline(bundle, event, *, ancestors=frozenset(), depth=0):
    if id(event) in ancestors:
        _fail('baseline_source_cycle')
    if depth >= _MAX_BASELINE_DEPTH:
        _fail('baseline_source_too_deep')
    ancestors = ancestors | {id(event)}
    if isinstance(event, dict) and event.get('current_hand_projection') is not None:
        from .hif_support_planning import replay_visible_hand_projection

        projection = event['current_hand_projection']
        source = {key: value for key, value in event.items() if key != 'current_hand_projection'}
        original = replace(bundle, state=exact_state(projection['bound_state']))
        core = _baseline(original, source, ancestors=ancestors, depth=depth+1)
        projected = replay_visible_hand_projection(original, projection)
        if projected.state != bundle.state:
            _fail('rule_hand_projection_state_conflict')
        return core
    if (not isinstance(event, dict) or event.get('accepted') is not True
            or event.get('strategy_state_bound') is not True
            or (event.get('run_id'), event.get('revision')) != (bundle.run_id, bundle.revision)):
        _fail('baseline_unbound')
    source_core = _core(event.get('core'))
    scope = event.get('verification_scope')
    if scope == PREINPUT_SCOPE:
        if (event.get('event') != 'hif_rule_preinput' or event.get('submitted') is not False
                or event.get('action') is not None
                or set(event.get('checks', ())) != set(_PREINPUT_CHECKS)
                or any(value is not True for value in event['checks'].values())
                or event.get('fresh_observed', {}).get('core') != source_core
                or not isinstance(event.get('source_baseline'), dict)):
            _fail('preinput_baseline_receipt_invalid')
        _baseline(bundle, event['source_baseline'], ancestors=ancestors, depth=depth+1)
        if exact_state(event['expected_state']) != bundle.state:
            _fail('preinput_baseline_state_conflict')
        if set(source_core['frame_ids']) & _source_frames(event['source_baseline']):
            _fail('preinput_baseline_frames_stale')
    elif scope == VERIFICATION_SCOPE:
        projected = event.get('observation_mode') == 'deterministic_hand_removal'
        expected_checks = set(_CHECKS) - {'fresh_core'} | {'fresh_action_confirmation'} if projected else set(_CHECKS)
        if (event.get('event') != 'hif_rule_progress' or event.get('submitted') is not True
                or set(event.get('checks', ())) != expected_checks
                or any(value is not True for value in event['checks'].values())
                or (event.get('fresh_observed', {}).get('confirmation') != event.get('action_confirmation')
                    if projected else event.get('fresh_observed', {}).get('core') != source_core)
                or not isinstance(event.get('source_baseline'), dict)
                or not all(name in event for name in ('before', 'action', 'expected_state'))):
            _fail('rule_baseline_receipt_invalid')
        # A prior accepted flag cannot bind a different state to this action.
        if exact_state(event['expected_state']) != bundle.state:
            _fail('rule_baseline_state_conflict')
    elif scope is not None:
        _fail('unsupported_baseline_scope')
    else:
        for name in ('cards', 'status', 'items', 'drinks', 'turns'):
            component = event.get('components', {}).get(name, {})
            if (component.get('accepted') is not True or component.get('returned_unchanged') is not True
                    or (component.get('run_id'), component.get('revision')) != (bundle.run_id, bundle.revision)):
                _fail('complete_baseline_component_unbound')
        if any(value is not True for value in event.get('checks', {}).values()):
            _fail('complete_baseline_checks_failed')
        raw_state = event.get('expected_state', event.get('state'))
        source_state = exact_state(raw_state)
        if event.get('event') == 'live_opening_state' and event.get('extended_hand_identity') is not None:
            from ..extended_opening import UPSTREAM_VERSION, verify_extended_opening_identity

            ids = _canonical_hand(bundle, source_state)
            catalog = SimpleNamespace(entries={cid: bundle.kernel.cards[cid] for cid in ids})
            proof = event['extended_hand_identity']
            run = bundle
            if isinstance(proof, dict) and proof.get('version') == UPSTREAM_VERSION:
                run = bundle.opening_live_context
                if run is None or (run.run_id, run.revision) != (bundle.run_id, bundle.revision):
                    _fail('complete_opening_live_context_unbound')
            verified = verify_extended_opening_identity(proof,
                run=run, core=source_core, inventory_rows=[dict(zone='手札', card_id=cid) for cid in ids],
                catalog=catalog)
            if verified != ids:
                _fail('complete_opening_physical_order_conflict')
            ordered = source_state
        else:
            ordered = _ordered_state(bundle, source_state, source_core)
        if ordered != bundle.state:
            _fail('complete_baseline_state_conflict')
    if not _matches_core(bundle.state, source_core) or not _bound_order_matches(bundle, bundle.state, source_core):
        _fail('baseline_core_state_conflict')
    return source_core


def with_hand_projection(bundle, source, projection):
    """Attach verified observation-only change without rewriting the old receipt."""
    if projection is None or exact_state(source.get('expected_state', source.get('state'))) == bundle.state:
        return source
    combined = dict(source, current_hand_projection=json_value(projection))
    _baseline(bundle, combined)
    return combined


def resolve_rule_preinput(bundle, baseline_event, core):
    """Refresh only core/hand after search while retaining the exact source state.

    ``baseline_event`` is the inner full/progress/preinput receipt, not a native
    wrapper. No action is submitted and no complete observation is synthesized.
    """
    receipt = dict(event='hif_rule_preinput', verification_scope=PREINPUT_SCOPE,
        accepted=False, strategy_state_bound=False, run_id=bundle.run_id,
        revision=bundle.revision, submitted=False, checks={})
    try:
        _baseline(bundle, baseline_event)
        receipt['checks']['baseline_binding'] = True
        bundle.state.validate()
        if (bundle.state.terminal or bundle.state.pending_choice
                or bundle.state.resolution_stack or bundle.state.pending_after_card_ids):
            _fail('preinput_state_not_settled')
        observed = _core(core)
        if (any(core.get(name, getattr(bundle, name)) != getattr(bundle, name) for name in ('run_id', 'revision'))
                or set(observed['frame_ids']) & _source_frames(baseline_event)):
            _fail('stale_or_foreign_core')
        receipt['checks']['fresh_core'] = True
        receipt['core'] = observed
        if not _matches_core(bundle.state, observed):
            _fail('preinput_core_state_conflict')
        if not _bound_order_matches(bundle, bundle.state, observed):
            _fail('preinput_hand_state_conflict')
        receipt['checks'].update(same_state=True, hand_binding=True)
        receipt.update(accepted=True, strategy_state_bound=True, reason='same_state_fresh_core_bound',
            source_baseline=json_value(baseline_event), expected_state=json_value(asdict(bundle.state)),
            fresh_observed={'core': observed}, observed_state_fields=list(_CORE_FIELDS),
            rule_derived_fields=[field.name for field in fields(bundle.state) if field.name not in _CORE_FIELDS],
            retained_state_source='verified_source_baseline',
            hand_binding_source='fresh_families_and_bound_source_state')
        return bundle.state, receipt
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        receipt.update(reason=str(error), accepted=False, strategy_state_bound=False)
        return None, receipt


def resolve_rule_pending(bundle, candidates, *, baseline_event, action, initial_core, core,
                         single_plus_estimate=None, hand_slot_identity=None):
    """Classify a dispatch from two stable visible projections of original states.

    This reuses existing rule/preinput receipts; it is not a complete panel
    observation. Every original successor participates, even when another
    successor's hidden state cannot be uniquely reconstructed from this view.
    """
    result = dict(accepted=False, pending_outcome='unknown', pending_core_classification={})
    try:
        if action.kind is ActionKind.USE_DRINK:
            _fail('pending_action_requires_effect_or_drink_observation')
        if single_plus_estimate is not None:
            # Current revealed GENKI tiers already have a complete rule
            # projection. Recheck that projection instead of reopening every
            # detail panel merely because the title had more than one plus.
            from .skill_support import project_supported_card_once
            from .support_master import genki_only_support
            from ..hif_single_plus import verify_selected_single_plus_estimate

            verified = verify_selected_single_plus_estimate(single_plus_estimate,
                bundle=bundle, action=action)
            ordinary = bundle.kernel.card_definition(bundle.state, bundle.state.deck.hand[action.source_slot])
            supported = genki_only_support(ordinary, verified['plus_count'])
            _, exact = project_supported_card_once(bundle, action,
                effective_definition=supported, settle_automatic_turn=False)
            supplied = candidates if isinstance(candidates, tuple) else (candidates,)
            canonical = bundle.kernel._canonical_probability_state
            if (len(exact) != 1 or len(supplied) != 1
                    or canonical(exact[0]) != canonical(supplied[0])):
                _fail('pending_support_does_not_match_complete_current_projection')
        _baseline(bundle, baseline_event)
        _action_binding(bundle, action)
        initial, observed = _core(initial_core), _core(core)
        source_frames = _source_frames(baseline_event)
        if (set(initial['frame_ids']) & set(observed['frame_ids'])
                or source_frames & (set(initial['frame_ids']) | set(observed['frame_ids']))
                or any(initial[key] != observed[key] for key in ('fields', 'families'))
                or any(reading.get(name, getattr(bundle, name)) != getattr(bundle, name)
                       for reading in (initial_core, core) for name in ('run_id', 'revision'))):
            _fail('pending_core_not_independently_stable')
        states = candidates if isinstance(candidates, tuple) else (candidates,)
        if not states:
            _fail('pending_original_outcomes_missing')
        for state in states:
            state.validate()

        def visible(state):
            return (_matches_core(state, observed)
                and _hand_possible(observed['families'], _canonical_hand(bundle, state)))

        before_visible = visible(bundle.state)
        after_visible = [i for i, state in enumerate(states) if visible(state)]
        classification = dict(initial_core=initial, confirmation_core=observed,
            before_visible=before_visible, after_visible_indices=after_visible)
        result['pending_core_classification'] = classification
        if before_visible and not after_visible:
            state, receipt = resolve_rule_preinput(bundle, baseline_event, core)
            side, key = 'before', 'rule_preinput'
        elif not before_visible and after_visible:
            state, receipt = resolve_rule_progress(bundle, states, baseline_event=baseline_event,
                action=action, core=core, submitted=True, hand_slot_identity=hand_slot_identity)
            side, key = 'after', 'rule_progress'
        else:
            _fail('pending_visible_outcomes_not_distinguishable')
        if state is None or receipt.get('accepted') is not True:
            _fail(receipt['reason'])
        result.update({name: receipt[name] for name in ('verification_scope', 'accepted',
            'strategy_state_bound', 'run_id', 'revision', 'checks', 'core', 'expected_state')})
        result.update(pending_outcome=side, **{key: receipt}, reason=receipt['reason'],
            observed_hand_instance_ids=[ref.instance_id for ref in state.deck.hand])
        return state, result
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        result.update(reason=str(error), accepted=False, pending_outcome='unknown')
        return None, result


def _action_binding(bundle, action):
    action.validate()
    before = bundle.state
    before.validate()
    if before.terminal or before.pending_choice or before.resolution_stack or before.pending_after_card_ids:
        _fail('before_not_settled')
    if action.kind is ActionKind.PLAY_CARD:
        valid = (type(action.source_slot) is int and 0 <= action.source_slot < len(before.deck.hand)
            and before.deck.hand[action.source_slot].instance_id == action.card_instance_id
            and action.drink_id is None and action.target_id is None)
    elif action.kind is ActionKind.USE_DRINK:
        valid = (type(action.source_slot) is int and 0 <= action.source_slot < 4
            and action.card_instance_id is None and action.target_id is None
            and action.drink_id in before.drinks
            and tuple(pair for pair in bundle.drink_slots or () if pair[0] == action.source_slot)
                == ((action.source_slot, action.drink_id),))
    elif action.kind is ActionKind.END_TURN:
        valid = all(getattr(action, name) is None for name in ('card_instance_id', 'drink_id', 'target_id', 'source_slot'))
    else:
        valid = False
    if not valid:
        _fail('action_not_bound_to_before')


def _surviving_order(before, after, action):
    survivors = [ref.instance_id for ref in before.deck.hand if ref.instance_id != action.card_instance_id]
    by_id = {ref.instance_id:ref for ref in after.deck.hand}
    if len(by_id) != len(survivors) or set(by_id) != set(survivors):
        _fail('deterministic_surviving_hand_changed')
    return replace(after,deck=replace(after.deck,hand=tuple(by_id[key] for key in survivors)))


def _survivors_and_draw_order(bundle, after, action, core, *, retrieved_instance_id=None):
    """Keep the physical order of survivors and bind new cards to their observed tail."""
    before = bundle.state
    survivors = tuple(ref.instance_id for ref in before.deck.hand if ref.instance_id != action.card_instance_id)
    old_ids = {ref.instance_id for ref in before.deck.hand}
    by_id = {ref.instance_id:ref for ref in after.deck.hand}
    added = tuple(ref for ref in after.deck.hand if ref.instance_id not in old_ids)
    if retrieved_instance_id is not None:
        if (action.kind is not ActionKind.USE_DRINK or action.drink_id != '22'
                or tuple(ref.instance_id for ref in added) != (retrieved_instance_id,)
                or any(by_id.get(ref.instance_id) != ref for ref in before.deck.hand)):
            _fail('retrieval_did_not_append_confirmed_instance')
    elif action.kind is not ActionKind.PLAY_CARD:
        _fail('draw_did_not_preserve_original_survivors')
    if (after.turn != before.turn
            or after.turns_remaining != before.turns_remaining or not added
            or not set(survivors).issubset(by_id)
            or set(by_id) != set(survivors) | {ref.instance_id for ref in added}):
        _fail('draw_did_not_preserve_original_survivors')
    tail = _ordered_state(bundle,replace(after,deck=replace(after.deck,hand=added)),
        dict(families=core['families'][len(survivors):]))
    return replace(after,deck=replace(after.deck,
        hand=tuple(by_id[key] for key in survivors)+tail.deck.hand))


def settlement_survivor_families(bundle, candidates, *, baseline_event, action):
    """Known physical copies survive a same-turn play before new draws append.

    All supplied original outcomes must agree. This proves only the old prefix;
    newly drawn identities and the entire resulting state still need observation.
    """
    try:
        core = _baseline(bundle, baseline_event)
        _action_binding(bundle, action)
        if action.kind is not ActionKind.PLAY_CARD:
            return None
        before = bundle.state
        outcomes = candidates if isinstance(candidates, tuple) else (candidates,)
        survivors = tuple(ref for ref in before.deck.hand if ref.instance_id != action.card_instance_id)
        original_ids = {ref.instance_id for ref in before.deck.hand}
        if not outcomes or not survivors:
            return None
        count = len(outcomes[0].deck.hand)
        for after in outcomes:
            after.validate()
            by_id = {ref.instance_id: ref for ref in after.deck.hand}
            if (after.turn != before.turn or after.turns_remaining != before.turns_remaining
                    or after.terminal or after.pending_choice or after.resolution_stack
                    or len(after.deck.hand) != count or len(by_id) != count
                    or any(by_id.get(ref.instance_id) != ref for ref in survivors)
                    or set(by_id) & original_ids != {ref.instance_id for ref in survivors}):
                return None
        return dict(run_id=bundle.run_id, revision=bundle.revision, hand_count=count,
            source_frame_ids=core['frame_ids'],
            instance_ids=[ref.instance_id for ref in survivors],
            families=[family for ref, family in zip(before.deck.hand, core['families'], strict=True)
                if ref.instance_id != action.card_instance_id])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def rule_drink_slots(bundle, action, proof):
    """Maintain bottle identities; only complete occupancy can prove compaction."""
    from .opening import verify_hif_consumed_slots
    from ..opening_observer import valid_opening_frame_ids

    old = tuple(bundle.drink_slots or ())
    if (action.source_slot, action.drink_id) not in old:
        _fail('drink_slot_not_present')
    remaining = tuple(pair for pair in old if pair[0] != action.source_slot)
    if not proof or proof.get('kind') != 'drink_slots_compacted':
        return remaining
    compacted = tuple(enumerate(drink for _,drink in remaining))
    ids = proof.get('frame_ids', ())
    observations = proof.get('observations', ())
    if (proof.get('slot') != action.source_slot or proof.get('drink_id') != action.drink_id
            or proof.get('stable') is not True or not valid_opening_frame_ids(ids)
            or proof.get('identity_source') != 'original_order_after_confirmed_consumption'
            or tuple(map(tuple,proof.get('before_drink_slots', ()))) != old
            or tuple(map(tuple,proof.get('after_drink_slots', ()))) != compacted
            or len(observations) != len(ids)):
        _fail('drink_compaction_confirmation_invalid')
    for row in observations:
        if len(row) != 4:
            _fail('drink_compaction_occupancy_incomplete')
        for slot, item in enumerate(row):
            occupied = slot < len(compacted)
            if item.get('slot') != slot or item.get('state') != ('OCCUPIED' if occupied else 'EMPTY'):
                _fail('drink_compaction_occupancy_conflict')
            if item.get('evidence') == 'retained_unchanged_empty':
                if occupied or slot in dict(old) or item.get('box') is not None or item.get('empty_proof') is not None:
                    _fail('drink_compaction_changed_slot_not_observed')
                continue
            if (len(item.get('box', ())) != 4
                    or any(type(value) is not int for value in item['box'])
                    or any(value <= 0 for value in item['box'][2:])):
                _fail('drink_compaction_occupancy_conflict')
            if not occupied:
                cup = item.get('empty_proof', {})
                if (list(cup.get('box', ())) != list(item['box'])
                        or cup.get('acceptance_threshold') not in (.90,.95)
                        or not cup['acceptance_threshold'] <= cup.get('confidence', -1) <= 1):
                    _fail('drink_compaction_empty_slot_unproven')
    return verify_hif_consumed_slots(old,compacted,action,allow_compaction=True)


def _interchangeable_hidden_removals(states):
    """Reuse settlement equivalence only for unreferenced discard/removal copies."""
    from .maintenance import equivalent_settlements
    from .hif_instance_view import _strings

    if not equivalent_settlements(states):
        return False
    first = states[0]
    if any(getattr(state.deck, zone) != getattr(first.deck, zone)
           for state in states for zone in ('hand', 'draw_pile', 'held')):
        return False
    removed = {ref.instance_id for ref in first.deck.removed}
    changed = set().union(*(removed ^ {ref.instance_id for ref in state.deck.removed} for state in states))
    referenced = {text for field in fields(first) if field.name != 'deck'
                  for text in _strings(getattr(first, field.name))}
    return bool(changed) and not changed & referenced


def needs_hand_inventory(bundle, candidates, receipt):
    """Only exact current-hand identity can distinguish these remaining branches."""
    if (receipt.get('reason') != 'original_outcome_not_unique'
            or receipt.get('information_boundary',{}).get('kind') != 'synchronize'):
        return False
    states = candidates if isinstance(candidates,tuple) else (candidates,)
    indexes = receipt.get('matching_candidate_indices',())
    return (len(indexes) > 1 and all(type(i) is int and 0 <= i < len(states) for i in indexes)
        and len({tuple(sorted(_canonical_hand(bundle,states[i]))) for i in indexes}) > 1)


def _inventory_hand_ids(bundle, core, proof):
    from ..opening_observer import valid_opening_frame_ids

    if (proof.get('version') != 1 or (proof.get('run_id'),proof.get('revision')) != (bundle.run_id,bundle.revision)
            or _core(proof['after_core']) != core):
        _fail('hand_inventory_epoch_or_return_changed')
    before = _core(proof['initial_core'])
    if any(before[k] != core[k] for k in ('fields','families')):
        _fail('hand_inventory_changed_live_state')
    event, = proof['events']
    groups = [event.get(k,()) for k in ('before_frame_ids','inventory_frame_ids','after_frame_ids')]
    if (event.get('phase') != 'inventory_hand_observation' or event.get('accepted') is not True
            or event.get('live_state_unchanged') is not True or event.get('read_failure')
            or event.get('input_click_count') != 2
            or [r.get('phase') for r in event.get('inputs',())] != ['open','close']
            or not all(valid_opening_frame_ids(group) for group in groups)
            or len(set(f for g in groups for f in g)) != sum(map(len, groups))
            or list(groups[-1]) != core['frame_ids']
            or set(before['frame_ids']) & set(groups[1])):
        _fail('hand_inventory_navigation_or_frames_incomplete')
    ids = event['inventory_card_ids']
    rows = event['inventory_observations']
    if (len(ids) != len(core['families']) or len(rows) != len(ids) or not 1 <= len(ids) <= 5
            or any(not isinstance(cid,str) or not cid.isdecimal() for cid in ids)
            or any(row.get('card_id') != cid or len(row.get('observations',())) != len(groups[1])
                or any(r.get('card_id') != cid or not str(r.get('reason','')).startswith('accepted')
                    for r in row['observations']) for cid,row in zip(ids,rows,strict=True))):
        _fail('hand_inventory_exact_identity_incomplete')
    return Counter(ids)


def resolve_rule_progress(bundle, candidates, *, baseline_event, action, core, submitted, action_confirmation=None,
                          legacy_canonical_hand_order=False, inventory_hand=None, hand_slot_identity=None):
    """Return ``(ordered_after, receipt)``; a rejected receipt has no after.

    A synchronization classification can still resolve uniquely after the fresh
    core/hand read. Remaining observation scopes stay recorded in the receipt;
    accepted progress does not authorize bypassing those scopes or input guards.
    """
    receipt = dict(event='hif_rule_progress', verification_scope=VERIFICATION_SCOPE,
        accepted=False, strategy_state_bound=False, run_id=bundle.run_id,
        revision=bundle.revision, submitted=submitted is True, checks={})
    try:
        if submitted is not True:
            _fail('action_not_submitted')
        receipt['checks']['submitted'] = True
        _baseline(bundle, baseline_event)
        receipt['checks']['baseline_binding'] = True
        _action_binding(bundle, action)
        receipt['checks']['action_binding'] = True
        boundary = classify_information_boundary(bundle.kernel, bundle.state, action, plan_type=bundle.kernel.plan_type)
        from ..hif_upstream_support import supported_boundary

        boundary = supported_boundary(bundle, action, boundary, action_confirmation)
        projected = core is None
        if projected:
            proof = action_confirmation or {}
            after = boundary.successor
            removal = (proof.get('kind') == 'hand_removal'
                and proof.get('before_count') == len(bundle.state.deck.hand)
                and after is not None and proof.get('after_count') == len(after.deck.hand)
                and proof.get('selected_slot') is None and proof.get('stable') is True)
            if (boundary.kind != 'deterministic_same_turn' or action.kind is not ActionKind.PLAY_CARD
                    or after is None or len(after.deck.hand) != len(bundle.state.deck.hand)-1
                    or not after.deck.hand or not removal
                    or proof.get('action_instance_id') != action.card_instance_id):
                _fail('invalid_deterministic_hand_confirmation')
            source_core = baseline_event['core']
            if not legacy_canonical_hand_order:
                after = _surviving_order(bundle.state,after,action)
            core = dict(fields=[getattr(after, name) for name in _CORE_FIELDS] + list(source_core['fields'][5:]),
                families=[[cid] for cid in _canonical_hand(bundle, after)], frame_ids=proof.get('frame_ids'))
        observed = _core(core)
        if (any(core.get(name, getattr(bundle, name)) != getattr(bundle, name) for name in ('run_id', 'revision'))
                or set(observed['frame_ids']) & _source_frames(baseline_event)):
            _fail('stale_or_foreign_core')
        receipt['checks']['fresh_action_confirmation' if projected else 'fresh_core'] = True
        receipt['core'] = observed
        receipt['information_boundary'] = dict(kind=boundary.kind, reason=boundary.reason,
            required_fields=list(boundary.required_fields))
        states = candidates if isinstance(candidates, tuple) else (candidates,)
        if not states:
            _fail('original_candidates_missing')
        exact_hand = None
        noncustom_slots = ()
        if hand_slot_identity is not None:
            from ..inventory_customization import verified_live_slot_badge_absence

            if projected or boundary.kind != 'synchronize':
                _fail('hand_slot_identity_requires_information_boundary')
            noncustom_slots = verified_live_slot_badge_absence(hand_slot_identity,observed,bundle)
        if inventory_hand is not None:
            if projected or boundary.kind != 'synchronize':
                _fail('hand_inventory_requires_information_boundary')
            exact_hand = _inventory_hand_ids(bundle,observed,inventory_hand)
        matches = []
        retained_drink_order = False
        retained_draw_order = False
        retrieved_instance_id = (action_confirmation or {}).get('retrieved_instance_id')
        for index, candidate in enumerate(states):
            if not isinstance(candidate, StrategyState):
                _fail('original_candidate_invalid')
            candidate.validate()
            if not _matches_core(candidate, observed):
                continue
            if not _hand_possible(observed['families'], _canonical_hand(bundle, candidate)):
                continue
            if retrieved_instance_id and {ref.instance_id for ref in candidate.deck.hand} - {
                    ref.instance_id for ref in bundle.state.deck.hand} != {retrieved_instance_id}:
                continue
            if exact_hand is not None and Counter(_canonical_hand(bundle,candidate)) != exact_hand:
                continue
            if noncustom_slots and not _hand_possible(
                    _noncustom_hand_families(bundle,candidate,observed,noncustom_slots),_canonical_hand(bundle,candidate)):
                continue
            try:
                ordered = (_surviving_order(bundle.state,candidate,action)
                           if projected and not legacy_canonical_hand_order else _ordered_state(
                               bundle,candidate,observed,noncustom_slots=noncustom_slots))
            except ValueError as error:
                if str(error) != 'inventory hand has no unique Live slot assignment':
                    raise
                if boundary.kind == 'deterministic_same_turn' and action.kind is ActionKind.USE_DRINK:
                    ordered = _surviving_order(bundle.state, candidate, action)
                    retained_drink_order = True
                elif boundary.kind == 'synchronize' and boundary.reason == 'draw':
                    ordered = _survivors_and_draw_order(bundle,candidate,action,observed)
                    retained_draw_order = True
                elif boundary.kind == 'synchronize' and boundary.reason == 'pending_resolution' and retrieved_instance_id:
                    ordered = _survivors_and_draw_order(bundle, candidate, action, observed,
                        retrieved_instance_id=retrieved_instance_id)
                    retained_draw_order = True
                else:
                    raise
            if not _bound_order_matches(bundle, ordered, observed):
                _fail('preserved_hand_order_conflicts_with_fresh_families')
            if ordered.pending_choice or ordered.resolution_stack or ordered.pending_after_card_ids:
                _fail('candidate_has_pending_resolution')
            matches.append((index, ordered))
        unique = tuple(dict.fromkeys(state for _index, state in matches))
        receipt.update(candidate_count=len(states), matching_candidate_indices=[index for index, _state in matches],
            unique_outcome_count=len(unique))
        equivalent = len(unique) > 1 and not projected and _interchangeable_hidden_removals(unique)
        if len(unique) != 1 and not equivalent:
            _fail('original_outcome_not_unique' if unique else 'no_original_outcome_matches')
        after = unique[0]
        if equivalent:
            receipt.update(interchangeable_instance_outcomes=True, selected_original_candidate_index=matches[0][0])
        if projected and bundle.kernel._canonical_probability_state(after) != bundle.kernel._canonical_probability_state(boundary.successor):
            _fail('deterministic_original_candidate_changed')
        receipt['checks'].update(unique_original_candidate=True, hand_binding=True, settled_state=True)
        if action.kind is ActionKind.USE_DRINK and action_confirmation is not None:
            rule_drink_slots(bundle, action, action_confirmation)
            if (action_confirmation.get('kind') == 'drink_slots_compacted'
                    and action_confirmation.get('frame_ids') != observed['frame_ids']):
                _fail('drink_compaction_core_frames_changed')
        if (_matches_core(bundle.state, observed)
                and _canonical_hand(bundle, after) == _canonical_hand(bundle, bundle.state)):
            proof = action_confirmation or {}
            if (action.kind is not ActionKind.USE_DRINK or proof.get('kind') not in ('drink_slot_empty','drink_slots_compacted')
                    or proof.get('slot') != action.source_slot or proof.get('drink_id') != action.drink_id
                    or proof.get('frame_ids') != observed['frame_ids'] or proof.get('stable') is not True
                    or (proof.get('kind') == 'drink_slot_empty'
                        and proof.get('empty_observations') != len(observed['frame_ids']))):
                _fail('needs_action_confirmation')
        receipt['checks']['action_observed'] = True
        receipt.update(accepted=True, strategy_state_bound=True, reason='unique_original_rule_outcome_bound',
            source_baseline=json_value(baseline_event), before=json_value(asdict(bundle.state)),
            action=json_value(asdict(action)), expected_state=json_value(asdict(after)),
            fresh_observed={'core': observed}, observed_state_fields=list(_CORE_FIELDS),
            rule_derived_fields=[field.name for field in fields(after) if field.name not in _CORE_FIELDS],
            hand_binding_source='fresh_families_and_original_candidate',
            base_rule_successor_matches=(boundary.successor == after if boundary.successor is not None else None))
        if action_confirmation is not None:
            receipt['action_confirmation'] = json_value(action_confirmation)
        if inventory_hand is not None:
            receipt['inventory_hand'] = json_value(inventory_hand)
            receipt['hand_binding_source'] = 'fresh_inventory_hand_and_original_candidates'
        if hand_slot_identity is not None:
            receipt['hand_slot_identity'] = json_value(hand_slot_identity)
            receipt['hand_binding_source'] = 'fresh_exposed_badge_absence_and_original_candidates'
        if retained_drink_order:
            receipt.update(hand_binding_source='original_order_after_confirmed_drink', hand_order_revision=2)
        if retained_draw_order:
            receipt.update(hand_binding_source=('original_order_and_confirmed_retrieval' if retrieved_instance_id
                else 'original_survivors_and_fresh_draw'), hand_order_revision=3)
        if projected:
            receipt.update(observation_mode='deterministic_hand_removal',
                fresh_observed={'confirmation': json_value(action_confirmation)}, observed_state_fields=[],
                rule_derived_fields=[field.name for field in fields(after)],
                hand_binding_source='original_order_after_confirmed_hand_removal')
            if not legacy_canonical_hand_order:
                receipt['hand_order_revision'] = 2
        return after, receipt
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        receipt.update(reason=str(error), accepted=False, strategy_state_bound=False)
        return None, receipt
