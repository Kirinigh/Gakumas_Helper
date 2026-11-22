"""Opt-in receipt/source glue. No archived pixels become fresh input authority."""
from copy import deepcopy
from dataclasses import replace

from . import diagnostic_depth_reached

INVENTORY_SOURCE = 'hif_inventory_597_support_v1'
INVENTORY_SOURCES = (INVENTORY_SOURCE, 'hif_inventory_295_support_v1')


def inventory_planning_support(bundle, source):
    """Consume an already loaded complete source, preserving its original epoch."""
    import json
    from dataclasses import asdict

    from live_decision.opening_observer import valid_opening_frame_ids
    from live_decision.inventory_support import verified_inventory_hand_support_fact
    from live_decision.high_score.opening import expected_permanent_inventory_keys
    from live_decision.high_score.contracts import PlanType, _jsonable

    if source is None:
        return bundle, []
    original, state, settlement, source_id = source
    component = settlement.get('components', {}).get('cards', {})
    rows = [row for row in component.get('cards', ()) if row.get('temporary_support') is not None]
    if not rows:
        return bundle, []
    if (settlement.get('accepted') is not True or settlement.get('strategy_state_bound') is not True
            or (settlement.get('run_id'), settlement.get('revision')) != (original.run_id, original.revision)
            or json.dumps(settlement.get('expected_state'), sort_keys=True) != json.dumps(_jsonable(asdict(state)), sort_keys=True)
            or component.get('accepted') is not True or component.get('returned_unchanged') is not True
            or component.get('observation_source') is not None
            or (component.get('run_id'), component.get('revision')) != (original.run_id, original.revision)
            or not isinstance(source_id, str) or not source_id
            or bundle.inventory_keys != expected_permanent_inventory_keys(original, state)
            or any(getattr(bundle.kernel, key) != getattr(original.kernel, key)
                   for key in ('ruleset_id', 'cards', 'items', 'drinks', 'memory_abilities'))):
        raise ValueError('inventory planning support source is not a complete verified inventory')
    before_frames = component.get('before_frame_ids', ())
    after_frames = component.get('after_frame_ids', ())
    if not (valid_opening_frame_ids(before_frames) and valid_opening_frame_ids(after_frames)
            and set(before_frames).isdisjoint(after_frames)):
        raise ValueError('inventory planning support source frames are incomplete')
    current = bundle.state
    if (set(current.deck.hand) != set(state.deck.hand)
            or len(current.deck.hand) != len(state.deck.hand)
            or replace(current, deck=replace(current.deck, hand=state.deck.hand)) != state):
        raise ValueError('inventory planning support current turn or public state changed')
    facts, receipts = list(current.hand_support), []
    for row in rows:
        if row.get('source') != 'observed_detail':
            raise ValueError('inventory support is not an original complete detail observation')
        fact = verified_inventory_hand_support_fact(replace(original, state=state), row)
        if set(row['temporary_support']['observations']['frame_ids']).intersection((*before_frames, *after_frames)):
            raise ValueError('inventory support detail reused its navigation frames')
        prior = [value for value in facts if value.instance_id == fact.instance_id]
        if prior and prior != [fact]:
            raise ValueError('inventory planning support conflicts with a current held fact')
        if not prior:
            facts.append(fact)
        receipts.append(dict(source=row['temporary_support']['source'], source_id=source_id,
            observed_turn=state.turn, detail=deepcopy(row)))
    after = replace(current, hand_support=tuple(sorted(facts, key=lambda fact: fact.instance_id)))
    after.validate()
    bundle.kernel._validate_registered_state(after, plan_type=PlanType.SENSE)
    return replace(bundle, state=after), receipts


def inventory_supported_outcomes(bundle, action, submitted, row, source):
    """Recompute one already dispatched play without changing its old binding."""
    from live_decision.high_score.contracts import ActionKind
    from live_decision.high_score.skill_support import supported_definition, project_supported_card_once

    from .native_single_plus import recorded_single_plus_estimate
    from .native_card_evidence import dispatched_card_refreshes

    if source is None or action.kind is not ActionKind.PLAY_CARD:
        return None
    if any(fact.instance_id == action.card_instance_id for fact in bundle.state.hand_support):
        return None  # Already part of the original planned state, not a repair.
    source_rows = source[2].get('components', {}).get('cards', {}).get('cards', ())
    if not any(item.get('temporary_support', {}).get('card_instance_id') == action.card_instance_id
               for item in source_rows):
        return None
    projected, receipts = inventory_planning_support(bundle, source)
    facts = [fact for fact in projected.state.hand_support if fact.instance_id == action.card_instance_id]
    if not receipts or not facts or facts[0] in bundle.state.hand_support:
        return None
    receipt = recorded_single_plus_estimate(bundle, action, submitted, source_row=row)
    ref = bundle.state.deck.hand[action.source_slot]
    if (receipt is None or receipt.get('raw_title') != bundle.kernel.cards[ref.card_id].name+'+'
            or receipt.get('plus_count') != 2 or row.get('native_resume_source') != source[3]
            or type(submitted.get('execution_submit_count')) is not int
            or submitted['execution_submit_count'] != 1):
        raise ValueError('inventory supported play lacks its one actual current title submission')
    refreshes = dispatched_card_refreshes(submitted, row=row, bundle=bundle)
    if len(refreshes) != 2 or any(refresh.get('accepted') is not True for refresh in refreshes):
        raise ValueError('inventory supported play lacks both original accepted input checks')
    final = refreshes[-1]
    if final.get('phase') != 'authoritative_refresh_b':
        raise ValueError('inventory supported play final evidence is not selected B')
    from .native_card_evidence import input_evidence_is_fresh

    if not input_evidence_is_fresh(final):
        raise ValueError('inventory supported play original input evidence was stale')
    definition = supported_definition(bundle.kernel.cards[ref.card_id], enhancement_count=2)
    transaction = final.get('resource_transaction', {})
    cost = bundle.kernel._effective_stamina_cost(bundle.state, ref, definition)
    # Inventory supplies the full effect body. The actual B supplies payment;
    # neither is fabricated into the selected-detail/printed-cost protocol.
    if (transaction.get('accepted') is not True or transaction.get('selected_slot') != action.source_slot
            or transaction.get('card_id') != definition.canonical_card_id
            or transaction.get('catalog_base_cost') != definition.stamina_cost
            or transaction.get('effective_payment_cost') != cost
            or transaction.get('displayed_effective_cost') != cost
            or type(transaction.get('displayed_effective_cost_confidence')) not in (int, float)
            or not .95 <= transaction['displayed_effective_cost_confidence'] <= 1
            or transaction.get('normalized_stamina') != bundle.state.stamina
            or transaction.get('normalized_shield') != bundle.state.genki
            or not transaction.get('conservation')
            or any(value is not True for value in transaction['conservation'].values())):
        raise ValueError('inventory supported play differs from its actual B payment evidence')
    _, outcomes = project_supported_card_once(bundle, action, effective_definition=definition)
    return outcomes[0] if len(outcomes) == 1 else outcomes


def bind_planning_hand_support(bundle, row, *, runtime=None, module=None, trail=(), inventory_source=None):
    """Recheck only explicit complete receipts against the restored prestate.

    A receipt belongs to this exact epoch and prestate. Future hosts inherit
    facts through the verified source transition/history chain, not by copying
    old receipts into a new epoch. A host must record the same receipts it
    explicitly supplied to BoundHifOpeningPolicy before planning.
    """
    if row.get('planning_inventory_detail_source'):
        rebound, receipt = _source_inventory_detail_support(
            bundle, row, runtime, module, trail, inventory_source=inventory_source)
        if row.get('planning_hand_support_receipts') != [receipt]:
            raise ValueError('native inventory detail original receipt changed')
        return bind_visible_hand_projection(rebound, row)
    if row.get('planning_hand_support_source'):
        rebound, receipt = _source_planning_support(bundle, row, runtime, module, trail)
        if row.get('planning_hand_support_receipts') != [receipt]:
            raise ValueError('native planning support original receipt changed')
        return bind_visible_hand_projection(rebound, row)
    receipts = row.get('planning_hand_support_receipts', ())
    if not isinstance(receipts, (list, tuple)):
        raise ValueError('native planning support receipts must be an explicit sequence')
    if not receipts:
        return bind_visible_hand_projection(bundle, row)
    if any(receipt.get('source') in INVENTORY_SOURCES for receipt in receipts):
        if inventory_source is None or row.get('native_resume_source') != inventory_source[3]:
            raise ValueError('native inventory planning support lacks its verified normal source')
        projected, expected = inventory_planning_support(bundle, inventory_source)
        if receipts != expected:
            raise ValueError('native inventory planning support differs from the original full detail')
        return bind_visible_hand_projection(projected, row)
    from live_decision.high_score.live_policy import BoundHifOpeningPolicy

    policy = BoundHifOpeningPolicy(bundle, action_order=())
    identities = set()
    for receipt in receipts:
        fact = policy.bind_verified_hand_support(receipt)
        if fact.instance_id in identities:
            raise ValueError('native planning support contains duplicate entity receipts')
        identities.add(fact.instance_id)
    return bind_visible_hand_projection(policy.bundle, row)


def bind_visible_hand_projection(bundle, row):
    """Recompute a new-format current-turn estimate from its original identity.

    Old records lacking this explicit event keep their old policy semantics.
    The projected facts are compared as output only, never copied into state.
    Actual dispatch and complete settlement remain the existing callers' gates.
    """
    from dataclasses import asdict

    from live_decision.contracts import LiveState
    from live_decision.high_score.contracts import _jsonable
    from live_decision.high_score.live_policy import BoundHifOpeningPolicy
    from live_decision.high_score.hif_mcts_cache import same_json

    def same(left, right):
        return same_json(_jsonable(left), _jsonable(right))

    evidence = [event for event in row.get('live_events', ())
                if event.get('event') == 'live_hand_projection']
    if not evidence:
        return bundle
    if len(evidence) != 1:
        raise ValueError('native ordinary-plus projection is ambiguous')
    event, = evidence
    context = row.get('observed_live_context', {})
    if (event.get('source') != 'hif_current_turn_ordinary_plus_v1'
            or (event.get('run_id'), event.get('revision')) != (bundle.run_id, bundle.revision)
            or (context.get('run_id'), context.get('revision')) != (bundle.run_id, bundle.revision)
            or not same(event.get('bound_state'), asdict(bundle.state))
            or not same(event.get('inventory_keys'), bundle.inventory_keys)):
        raise ValueError('native ordinary-plus projection public state or inventory changed')
    policy = BoundHifOpeningPolicy(bundle, action_order=())
    live = LiveState.from_dict(event.get('visible_state'))
    restored = policy.bind_visible_hand(live, identity_events=event.get('identity_events', ()))
    if policy.ordinary_plus_projection is None or not same(event, policy.ordinary_plus_projection):
        raise ValueError('native ordinary-plus projection differs from original title evidence')
    return restored


def _source_planning_support(bundle, row, runtime, module, trail=()):
    """Recheck a zero-submit source and the caller's verified history rebound."""
    from .native_selected_resume import load_selected_source

    name = row.get('planning_hand_support_source')
    parent = row.get('native_resume_source')
    if (runtime is None or module is None or not isinstance(name, str) or not name
            or not isinstance(parent, str) or not parent
            or any(row.get(key) for key in ('native_selected_source', 'native_pending_source',
                                            'native_choice_source', 'native_inprocess_source'))
            or row.get('resume_settlement', {}).get('accepted') is not True
            or row.get('resume_history', {}).get('accepted') is not True
            or (row.get('observed_live_context', {}).get('run_id'),
                row.get('observed_live_context', {}).get('revision')) != (bundle.run_id, bundle.revision)):
        raise ValueError('native planning support requires verified normal history resume')
    source, _, _, original = load_selected_source(runtime, name, module, (*trail, row['id']))
    if (original.get('native_resume_source') != parent
            or original['endpoints']['game']['pid'] != row['endpoints']['game']['pid']):
        raise ValueError('native planning support source parent or game changed')
    receipt = original.get('gray_selection_journal', {}).get('selected_skill_support')
    if not isinstance(receipt, dict) or receipt.get('effects_verified') is not True:
        raise ValueError('native planning support source lacks a complete receipt')
    return rebind_support_after_verified_history(source, bundle, receipt), receipt


def _source_inventory_detail_support(bundle, row, runtime, module, trail=(), *, inventory_source=None):
    """Use one failed detail's text after a separately completed END recovery.

    The source remains a failed observation. Only its complete current-hand
    text is consumed; its pixels, slots and partial components are not restored.
    """
    from collections import Counter
    from dataclasses import asdict

    from live_decision.inventory_support import verify_inventory_ordinary_plus, observe_inventory_ordinary_plus
    from live_decision.high_score.opening import expected_permanent_inventory_keys
    from live_decision.high_score.restart import _same
    from live_decision.high_score.contracts import PlanType
    from live_decision.inventory_customization import load_customization_catalog
    from live_decision.high_score.settled_restart import _core

    from .native_opening_root import verify_native_round_context
    from .native_pending_restart import _no_business_submission
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import load_result, _verified_endpoints
    from .native_pending_components import _frames, _navigation_dispatch_matches

    name, parent_name = row.get('planning_inventory_detail_source'), row.get('native_resume_source')
    if (runtime is None or module is None or not isinstance(name, str) or not name
            or name in trail or name == row.get('id') or diagnostic_depth_reached(trail)
            or not isinstance(parent_name, str) or not parent_name or parent_name == name
            or row.get('planning_hand_support_source') or inventory_source is None
            or any(row.get(key) for key in ('native_selected_source', 'native_pending_source',
                'native_choice_source', 'native_inprocess_source', 'native_pending_restart_source'))
            or row.get('resume_settlement', {}).get('accepted') is not True
            or row.get('resume_history', {}).get('accepted') is not True
            or (row.get('observed_live_context', {}).get('run_id'),
                row.get('observed_live_context', {}).get('revision')) != (bundle.run_id, bundle.revision)):
        raise ValueError('native inventory detail requires verified normal history resume')
    original, state, saved, verified_name = inventory_source
    parent, source = load_result(runtime, parent_name), load_result(runtime, name)
    pending = parent.get('native_pending_restart_source')
    source_context = source.get('observed_live_context', {})
    game_pid = row.get('endpoints', {}).get('game', {}).get('pid')
    if (not isinstance(pending, str) or not pending or verified_name != parent_name
            or source.get('native_pending_restart_source') != pending
            or any(value for key, value in source.items() if key.startswith('native_')
                   and key.endswith('_source') and key != 'native_pending_restart_source')
            or parent.get('live_return') is not True
            or not _same(parent.get('pending_restart_settlement'), saved)
            or saved.get('accepted') is not True or saved.get('strategy_state_bound') is not True
            or not _same(saved.get('expected_state'), asdict(state))
            or (saved.get('run_id'), saved.get('revision')) != (original.run_id, original.revision)
            or source.get('endpoints', {}).get('game', {}).get('pid') != game_pid
            or parent.get('endpoints', {}).get('game', {}).get('pid') != game_pid
            or any(source_context.get(key) != row['observed_live_context'].get(key)
                   for key in ('idol', 'plan', 'max_stamina'))
            or not isinstance(source_context.get('run_id'), str) or not source_context['run_id']
            or type(source_context.get('revision')) is not int
            or bundle.inventory_keys != expected_permanent_inventory_keys(original, state)
            or any(getattr(bundle.kernel, key) != getattr(original.kernel, key)
                   for key in ('ruleset_id', 'cards', 'items', 'drinks', 'memory_abilities'))
            or set(bundle.state.deck.hand) != set(state.deck.hand)
            or len(bundle.state.deck.hand) != len(state.deck.hand)
            or replace(bundle.state, deck=replace(bundle.state.deck, hand=state.deck.hand)) != state):
        raise ValueError('native inventory detail parent, process or complete business state changed')
    verify_native_round_context(source, module, parent, required=True)
    events = source.get('live_events', ())
    if ([event.get('event') for event in events] != ['live_inventory_observe', 'live_committed_state']
            or len(events[0].get('inventory_events', ())) != 2
            or not _same(events[1], source.get('settlement'))):
        raise ValueError('native inventory detail lacks its original failed cards transaction')
    report, settlement = events
    inventory, gate = report['inventory_events']
    recovery = source.get('context_recovery', {})
    for value in (source, *events, inventory, gate):
        _no_business_submission(value)
        if value.get('executed') is True or value.get('execution_selection_click_count', 0):
            raise ValueError('native inventory detail contains a business selection')
    if (settlement.get('accepted') is not False
            or settlement.get('reason') != 'HIF opening component failed: cards'
            or report.get('accepted') is not False or inventory.get('accepted') is not False
            or inventory.get('phase') != 'inventory_zone_observation'
            or gate.get('phase') != 'inventory_state_gate' or gate.get('accepted') is not False
            or type(gate.get('input_click_count')) is not int or gate['input_click_count'] != 0
            or gate.get('reason') != inventory.get('reason')
            or inventory.get('live_state_unchanged') is not True
            or recovery.get('accepted') is not True or recovery.get('returned_to_live') is not True
            or recovery.get('context') != source_context or recovery.get('game_pid') != game_pid):
        raise ValueError('native inventory detail did not return unchanged after its failed reader')
    cores = [_core(value) for value in (settlement.get('initial_core'), saved.get('core'),
                                      row['resume_settlement'].get('core'))]
    if any(not _same(core[key], cores[0][key]) for core in cores[1:] for key in ('fields', 'families')):
        raise ValueError('native inventory detail differs from the current complete Live core')
    before, after = inventory.get('before_frame_ids'), inventory.get('after_frame_ids')
    if (not _frames(before) or not _frames(after) or set(before) & set(after)
            or not any(entry.get('accepted') is True and entry.get('frame_ids') == after
                       for entry in inventory.get('return_observations', ()))):
        raise ValueError('native inventory detail lacks independent unchanged return observations')
    recovery_inputs, inputs = recovery.get('inputs', ()), inventory.get('inputs', ())
    dispatches = source.get('native_input_dispatches', ())
    if (not recovery_inputs or [entry.get('phase') for entry in inputs]
            != ['open', 'customization_select', 'close']
            or recovery.get('input_click_count') != len(recovery_inputs)
            or inventory.get('input_click_count') != len(inputs)
            or report.get('total_input_click_count') != len(inputs)
            or settlement.get('total_input_click_count') != len(inputs)
            or type(source.get('input_count')) is not int
            or source['input_count'] != len(dispatches) or len(dispatches) != len(recovery_inputs)+len(inputs)):
        raise ValueError('native inventory detail has unaccounted navigation or business input')
    proofs = [dict(entry, phase='open', box=entry.get('target')) for entry in recovery_inputs] + list(inputs)
    for actual, proof in zip(dispatches, proofs, strict=True):
        _verified_endpoints(actual.get('endpoints'), game_pid)
        _same_input_host(source['endpoints'], actual['endpoints'])
        if not _navigation_dispatch_matches(actual, proof, inventory_customization=True):
            raise ValueError('native inventory detail navigation differs from its dispatches')
    rejections = inventory.get('detail_rejections', ())
    if not rejections:
        raise ValueError('native inventory detail has no original rejected text')
    last = rejections[-1]
    cid = last.get('card_id')
    reason = f'clicked skill-card text disagrees with icon candidates ({cid},)'
    if (not isinstance(cid, str) or not cid.isdigit()
            or any(entry.get('card_id') != cid or entry.get('reason') != reason for entry in rejections)
            or inventory.get('read_failure') != reason or inventory.get('reason') != reason
            or not _frames(last.get('frame_ids')) or set(last['frame_ids']) & set((*before, *after))):
        raise ValueError('native inventory detail failure is not one complete title disagreement')
    # This source failed before page append. Presence entries were produced
    # after complete thumbnail identity consensus and retain its full hand.
    hand = [entry for entry in inventory.get('customization_presence', ()) if entry.get('zone') == '手札']
    current_ids = [bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id for ref in state.deck.hand]
    matches = [entry for entry in hand if entry.get('card_id') == cid]
    if (len(hand) != len(current_ids) or sorted(entry.get('index', -1) for entry in hand) != list(range(len(hand)))
            or Counter(entry.get('card_id') for entry in hand) != Counter(current_ids)
            or len(matches) != 1 or matches[0].get('presence') != 'DETAIL_CANDIDATE'):
        raise ValueError('native inventory detail target is not a unique current hand identity')
    observed = replace(bundle, run_id=source_context['run_id'], revision=source_context['revision'])
    detail = dict(zone='手札', index=matches[0]['index'], card_id=cid,
        rows=deepcopy(last.get('rows')), frame_ids=list(last['frame_ids']), run_id=observed.run_id,
        revision=observed.revision, observed_turn=observed.state.turn)
    catalog = load_customization_catalog(module.CATALOG_PATH, module.CATALOG_PATH.with_name('customizations.json'))
    receipt = observe_inventory_ordinary_plus(observed, detail, catalog=catalog)
    fact = verify_inventory_ordinary_plus(observed, receipt, catalog=catalog)
    old = [value for value in bundle.state.hand_support if value.instance_id == fact.instance_id]
    if old and old != [fact]:
        raise ValueError('native inventory detail conflicts with a current held fact')
    after = replace(bundle.state, hand_support=tuple(sorted(
        set((*bundle.state.hand_support, fact)), key=lambda value: value.instance_id)))
    after.validate()
    bundle.kernel._validate_registered_state(after, plan_type=PlanType.SENSE)
    return replace(bundle, state=after), receipt


def install_planning_hand_support(module, command, *, runtime, current_result, persist):
    """Wrap the native policy factory for one explicit normal-resume transaction.

    Commands name an independently verified source, never supply observations
    or flags. The original receipt keeps its old epoch. No capture, selection,
    search, cancellation or input authorization is performed here.
    """
    if 'planning_hand_support_receipts' in command:
        raise ValueError('native planning support commands must name a verified source')
    name = command.get('planning_hand_support_source')
    detail_name = command.get('planning_inventory_detail_source')
    if detail_name is not None and (name is not None or not isinstance(detail_name, str) or not detail_name):
        raise ValueError('native inventory detail source must be explicit and exclusive')
    automatic = (name is None and detail_name is None and command.get('mode') == 'live_opening_execute'
        and command.get('hif_policy_profile') in ('hif-mcts-v22-trial', 'hif-strong-reference-trial-v1','hif-gkms-fast-trial-v1')
        and bool(command.get('native_resume_source'))
        and not any(command.get(key) for key in ('history_only', 'complete_history_only', 'gray_preview_only')))
    if name is None and detail_name is None and not automatic:
        return lambda: None
    if (not automatic and detail_name is None and (not isinstance(name, str) or not name)
            or command.get('mode') != 'live_opening_execute'
            or command.get('hif_policy_profile') not in ('hif-mcts-v22-trial', 'hif-strong-reference-trial-v1','hif-gkms-fast-trial-v1')
            or not command.get('native_resume_source')
            or any(command.get(key) for key in command if
                   (key.startswith('native_') and key.endswith('_source')
                    and key not in ('native_resume_source', 'native_history_source'))
                   or key in ('history_only', 'complete_history_only', 'settlement_source',
                              'checkpoint_only', 'gray_preview_only'))):
        raise ValueError('native planning support is limited to explicit normal MCTS resume')
    original_bind = module._bind_hif_opening_policy
    applied = False

    def bind(context, event, profile='joint-search'):
        nonlocal applied
        if applied:
            return original_bind(context, event, profile)
        row = current_result()
        if (row['id'] != command['id'] or profile != command['hif_policy_profile']
                or getattr(context, 'gkh_hif_gray_selected_source', None) is not None
                or getattr(context, 'gkh_hif_committed_event', None) != row.get('resume_settlement')):
            raise ValueError('native planning support is not before a new normal intent')
        before = context.gkh_hif_opening_state
        if detail_name is not None:
            source_row = dict(row, planning_inventory_detail_source=detail_name)
            after, receipt = _source_inventory_detail_support(before, source_row, runtime, module,
                inventory_source=getattr(context, 'gkh_hif_inventory_support_source', None))
            row.update(planning_inventory_detail_source=detail_name,
                       planning_hand_support_receipts=[deepcopy(receipt)])
        elif automatic:
            after, receipts = inventory_planning_support(before,
                getattr(context, 'gkh_hif_inventory_support_source', None))
            if not receipts:
                applied = True
                return original_bind(context, event, profile)
            row['planning_hand_support_receipts'] = deepcopy(receipts)
        else:
            source_row = dict(row, planning_hand_support_source=name)
            after, receipt = _source_planning_support(before, source_row, runtime, module)
            row.update(planning_hand_support_source=name,
                       planning_hand_support_receipts=[deepcopy(receipt)])
        # A failed write aborts before creating/reusing a candidate or intent.
        persist()
        policy = original_bind(context, event, profile)
        if policy.bundle != before or policy.response is not None:
            raise ValueError('native planning support policy already has another binding or intent')
        update = getattr(policy.decision_policy, 'bind_hand_support_context', None)
        if not callable(update):
            raise ValueError('native planning support policy cannot invalidate its old context')
        update(run_id=before.run_id, revision=before.revision, before=before.state, after=after.state)
        policy.bundle = after
        context.gkh_hif_opening_state = after
        applied = True
        return policy

    module._bind_hif_opening_policy = bind
    return lambda: setattr(module, '_bind_hif_opening_policy', original_bind)


def supported_transition_candidates(bundle, action, submitted, source_row, observed):
    """Recompute all supported outcomes using the existing submission reader.

    The historical observed-draw path remains exact and may be unavailable;
    unsupported cases fall back to the existing full kernel, never a guessed
    retention mask. The caller still checks all complete observed components.
    """
    from types import SimpleNamespace

    from live_decision.high_score.contracts import ActionKind
    from live_decision.high_score.live_policy import BoundHifOpeningPolicy
    from live_decision.high_score.hif_observed_transition import observed_committed_transition

    policy = BoundHifOpeningPolicy(bundle, action_order=())
    policy.response = SimpleNamespace(action=action)
    if action.kind is ActionKind.PLAY_CARD:
        from .native_card_evidence import dispatched_card_refreshes
        from .native_selected_support import bind_recorded_selected_support

        refreshes = dispatched_card_refreshes(submitted, row=source_row, bundle=bundle)
        policy.bind_generated_card_resource_preview(refreshes[-1]['resource_transaction'],
                                                   refreshes[-1]['visual_fields']['score'])
        bind_recorded_selected_support(policy, submitted, require_submitted=True, source_row=source_row)
    witness = observed_committed_transition(policy, observed)
    if witness is not None:
        return witness.states
    values = policy.committed_transition()
    return values if isinstance(values, tuple) else (values,)


def require_supported_outcome(bundle, action, submitted, source_row, observed):
    """Reject source-free or revived facts before comparing complete outcomes."""
    from live_decision.high_score.hif_support_continuity import hand_support_candidate_continuity

    if not hand_support_candidate_continuity(bundle.state, observed):
        raise ValueError('native settlement hand support has no current prestate source')
    return supported_transition_candidates(bundle, action, submitted, source_row, observed)


def rebind_support_after_verified_history(source_bundle, restored_bundle, receipt):
    """Transfer a newly inspected fact after caller-verified no-business history.

    This pure hook is intentionally not auto-enabled by a profile. Both bundles
    must come from the existing complete/history verifiers. The exact public
    business state and definitions must agree; only physical hand order and
    epoch may differ. The receipt is rechecked in its original epoch. No slot,
    frame, selection or input permission is transferred.
    """
    from live_decision.high_score.hif_support_planning import verified_hand_support_fact, uses_hand_support_for_strategy

    if (source_bundle.kernel.ruleset_id != restored_bundle.kernel.ruleset_id
            or any(getattr(source_bundle.kernel, key) != getattr(restored_bundle.kernel, key)
                   for key in ('cards', 'items', 'drinks', 'memory_abilities'))):
        raise ValueError('native support history rules changed')
    before, current = source_bundle.state, restored_bundle.state
    if (len(before.deck.hand) != len(current.deck.hand)
            or set(before.deck.hand) != set(current.deck.hand)
            or replace(current, deck=replace(current.deck, hand=before.deck.hand)) != before):
        raise ValueError('native support history business state changed')
    fact = verified_hand_support_fact(source_bundle, receipt)
    old = [value for value in current.hand_support if value.instance_id == fact.instance_id]
    if old and old != [fact]:
        raise ValueError('native support history entity tier changed')
    if not uses_hand_support_for_strategy(fact):
        return restored_bundle
    state = replace(current, hand_support=tuple(sorted(
        set((*current.hand_support, fact)), key=lambda value: value.instance_id)))
    return replace(restored_bundle, state=state)
