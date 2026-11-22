"""Complete one dispatched END after a process restart, using only fresh readers.

Preparation only carries rule expectations. No archived input credential or
visual component becomes current, and no computation or search runs here.
"""

import re
import copy
from dataclasses import asdict, replace, dataclass

from . import diagnostic_depth_reached
from .native_settlement_resume import load_result, load_pending_source, decoded_dispatched_action


@dataclass(frozen=True)
class PendingRestartPreparation:
    comparison_bundle: object
    bundle: object
    candidates: tuple
    baseline: dict
    row: dict
    action: object
    old_run: object
    new_run: object
    old_pid: int
    new_pid: int
    round_context: object


def prepare_pending_restart(runtime, source, module, *, new_run, new_pid, new_round_context):
    from live_decision.hif_session import resolve_hif_round_context
    from live_decision.high_score.restart import _epoch, _frame_sets
    from live_decision.high_score.contracts import ActionKind

    from .native_opening_root import load_native_round_context

    bundle, expected, baseline, row = load_pending_source(runtime, source, module)
    action, _ = decoded_dispatched_action(row, bundle)
    old_run = module.HifLiveRunContext(**row['observed_live_context'])
    old_pid = row['endpoints']['game']['pid']
    old_round = load_native_round_context(runtime, row, module, run=old_run)
    if (action.kind is not ActionKind.END_TURN or row.get('choice_journal')
            or any(event.get('event') == 'live_committed_state' and event.get('accepted') is True
                   for event in row.get('live_events', ()))
            or type(new_pid) is not int or new_pid <= 0 or new_pid == old_pid
            or _epoch(new_run) == _epoch(old_run)
            or (bundle.run_id, bundle.revision) != (old_run.run_id, old_run.revision)
            or any(getattr(new_run, key) != getattr(old_run, key) for key in ('idol', 'plan', 'max_stamina'))):
        raise ValueError('pending restart requires one unsettled END and a new matching process/epoch')
    if (new_round_context is None or old_round is None
            or resolve_hif_round_context(new_round_context, run_id=new_run.run_id, revision=new_run.revision)
                != resolve_hif_round_context(old_round, run_id=old_run.run_id, revision=old_run.revision)
            or set(new_round_context.frame_ids) & set(_frame_sets(row))):
        raise ValueError('pending restart round or fresh notebook evidence changed')
    states = expected if isinstance(expected, tuple) else (expected,)
    if not states or len(states) > 10000:
        raise ValueError('pending restart legal outcomes are empty or unbounded')
    for state in (bundle.state, *states):
        state.validate()
        if (state.terminal or state.pending_choice or state.pending_after_card_ids or state.resolution_stack
                or not state.persistent_effects_initialized or state.max_stamina != new_run.max_stamina):
            raise ValueError('pending restart requires complete quiescent states')
    if any(state.turn != bundle.state.turn+1 or state.turns_remaining != bundle.state.turns_remaining-1
           for state in states):
        raise ValueError('pending restart outcomes do not complete exactly one END')
    return PendingRestartPreparation(replace(bundle, **_epoch(new_run)), bundle, states, baseline,
        row, action, old_run, new_run, old_pid, new_pid, new_round_context)


def _complete_turn_tail(history, turn):
    """Require the existing reader's complete turn tail, not a turn label alone."""
    from live_decision.history import _complete_history_draw, normalize_history_text

    lines = history.get('observation', {}).get('lines', ())
    if (not isinstance(lines, (list, tuple)) or len(lines) < 3
            or any(not isinstance(line, str) or not line or normalize_history_text(line) != line for line in lines)
            or lines[0] != f'{turn}ターン目'):
        raise ValueError('pending restart latest history is not the complete current turn start')
    groups = []
    for line in lines[1:]:
        if re.fullmatch(r'[1-9]\d*ターン目', line) or line.startswith(('Pドリンク', 'ドリンク')):
            raise ValueError('pending restart history contains a later turn or business action')
        if line.startswith(('メモリーアビリティ', '持続効果「', 'Pアイテム')):
            groups.append([line])
        elif not groups or (line.startswith('スキルカード「') and line.endswith('」')):
            raise ValueError('pending restart history contains an incomplete or card action group')
        else:
            groups[-1].append(line)
    if not groups or any(len(group) < 2 for group in groups):
        raise ValueError('pending restart turn effect group is incomplete')
    if not any('→' in line for line in lines[1:]):
        for group in groups:
            effects = group[1:]
            if (not group[0].startswith('持続効果「') or not group[0].endswith('」') or len(effects) % 2
                    or any(not _complete_history_draw(name, effect)
                           for name, effect in zip(effects[::2], effects[1::2]))):
                raise ValueError('pending restart turn draw group is incomplete')


def _no_business_submission(value):
    from live_decision.high_score.settled_restart import _no_business_input

    _no_business_input(value)
    if (value.get('choice_journal') or type(value.get('choice_submit_count', 0)) is not int
            or value.get('choice_submit_count', 0) != 0):
        raise ValueError('pending restart contains a secondary choice submission')


def _verify_recorded_input_host(row):
    """Verify actual Maa/input children and bind the existing host summary.

    The launcher summary's `maapicli` is the Python input child, not the
    independently enumerated Maa endpoint. Use the established ancestry
    contract without renaming either recorded process or relaxing integrity.
    """
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints

    dispatches = row.get('native_input_dispatches', ())
    if (not dispatches or type(row.get('input_count')) is not int
            or row['input_count'] != len(dispatches)):
        raise ValueError('pending restart requires its complete actual input chain')
    for entry in dispatches:
        _verified_endpoints(entry.get('endpoints'), row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], entry['endpoints'])


def _verified_definition_rows(prepared, component, *, enabled):
    """Exclude proven immutable definitions, never current inventory evidence.

    The fresh observer supplies canonical composition independently of these
    retained rows. Mixed variants still need their original fresh row proof.
    """
    from collections import Counter

    from live_decision.high_score.opening import expected_permanent_inventory_keys
    from live_decision.high_score.restart import _same, _epoch
    from live_decision.inventory_customization import retained_ambiguous_card_ids, retained_permanent_card_rows

    retained = [row for row in component.get('cards', ()) if row.get('source') == 'retained_verified_card']
    if not retained:
        return component, None
    if enabled is not True:
        raise ValueError('pending restart retained definitions were not explicitly enabled')
    composition = component.get('observed_composition')
    zones = ('手札', '山札', '捨札', '除外')
    if (not isinstance(composition, dict) or set(composition) != set(zones)
            or any(not isinstance(ids, (list, tuple)) or any(not isinstance(cid, str) or not cid for cid in ids)
                   for ids in composition.values())
            or component.get('zone_counts') != {zone: len(composition[zone]) for zone in zones}):
        raise ValueError('pending restart retained definitions lack complete current composition')
    rows = component['cards']
    positions = [(row.get('zone'), row.get('index')) for row in rows]
    wanted = {(zone, index): cid for zone in zones for index, cid in enumerate(composition[zone])}
    if (len(positions) != len(set(positions)) or set(positions) != set(wanted)
            or any(row.get('card_id') != wanted[row['zone'], row['index']] for row in rows)):
        raise ValueError('pending restart definition rows differ from current physical composition')
    before = {ref.instance_id: ref for ref in prepared.bundle.state.deck.all_cards}
    first = prepared.candidates[0]
    ambiguous = retained_ambiguous_card_ids(prepared.bundle, first)
    expected_cards = Counter(cid for _, cid, _ in expected_permanent_inventory_keys(prepared.bundle, first))
    if Counter(cid for ids in composition.values() for cid in ids) != expected_cards:
        raise ValueError('pending restart observed composition changed the complete card inventory')
    for state in prepared.candidates:
        if (state.created_card_count != prepared.bundle.state.created_card_count
                or {ref.instance_id: ref for ref in state.deck.all_cards} != before
                or retained_ambiguous_card_ids(prepared.bundle, state) != ambiguous):
            raise ValueError('pending restart cannot retain changed or newly generated definitions')
    # Rebuild only unique families. The remaining original rows keep all
    # direct-detail/badge evidence and are checked by the unchanged fresh gate.
    mapping = {zone: [index for index, cid in enumerate(composition[zone]) if cid not in ambiguous]
               for zone in zones}
    unique_composition = {zone: [composition[zone][index] for index in mapping[zone]] for zone in zones}
    rebuilt = retained_permanent_card_rows(prepared.comparison_bundle, first, unique_composition)
    proven = {(row['zone'], mapping[row['zone']][row['index']]):
              dict(row, index=mapping[row['zone']][row['index']]) for row in rebuilt}
    current = copy.deepcopy(component)
    for row, checking in zip(rows, current['cards'], strict=True):
        if row.get('source') != 'retained_verified_card':
            continue
        if not _same(row, proven.get((row['zone'], row['index']))):
            raise ValueError('pending restart retained row differs from its verified permanent definition')
        del checking['source']
        del checking['retained_from']
    return current, dict(source_record_id=prepared.row['id'], source_epoch=_epoch(prepared.old_run),
        definition_rows=[dict(zone=row['zone'], index=row['index'], card_id=row['card_id']) for row in retained],
        fresh_definition_read=False, fresh_composition_required=True)


def finalize_pending_restart(prepared, *, settlement, history, resumed=None, continuity_history=None):
    from live_decision.high_score.opening import (
        verified_expired_start_drinks,
        verified_expired_start_scores,
        expected_permanent_inventory_keys,
        resolve_hif_committed_observation,
    )
    from live_decision.high_score.restart import _same, _epoch, _frame_sets, _reject_retained
    from live_decision.high_score.maintenance import _preinput_history_matches_core
    from live_decision.high_score.settled_restart import _core

    from .native_preinput_history import bind_settlement_history
    from .native_settlement_resume import _verified_endpoints

    epoch = _epoch(prepared.new_run)
    if resumed is not None:
        verify_pending_restart_components_continuity(prepared, resumed, continuity_history)
        _verify_resumed_completion(settlement, resumed)
        if (set(_frame_sets(history)) & (set(resumed['audit']['source_reader_frame_ids'])
                                        | set(_frame_sets(continuity_history)))
                or set(_frame_sets(settlement)) & set(_frame_sets(continuity_history))
                or any(not _same(history.get('observation', {}).get(key),
                                 continuity_history.get('observation', {}).get(key)) for key in ('lines', 'score'))):
            raise ValueError('pending restart final history changed or reused its component continuity')
    elif continuity_history is not None or settlement.get('resumed_fresh_components'):
        raise ValueError('pending restart resumed readers lack their verified original source')
    if (settlement.get('event') != 'live_committed_state' or settlement.get('accepted') is not True
            or settlement.get('strategy_state_bound') is not True
            or settlement.get('force_fresh_components') is not True
            or type(settlement.get('execution_submit_count')) is not int or settlement['execution_submit_count'] != 0
            or settlement.get('maintained_components') != []
            or any(settlement.get(key) != value for key, value in epoch.items())):
        raise ValueError('pending restart needs a zero-submit complete fresh settlement')
    _no_business_submission(settlement)
    core, before = _core(settlement.get('core')), _core(settlement.get('initial_core'))
    if (set(core['frame_ids']) & set(before['frame_ids'])
            or any(not _same(core[key], before[key]) for key in ('fields', 'families'))):
        raise ValueError('pending restart complete scan changed its Live core')
    components = settlement.get('components', {})
    if set(components) != {'cards', 'status', 'items', 'drinks', 'turns'}:
        raise ValueError('pending restart needs all five freshly observed components')
    old_frames = set(_frame_sets(prepared.row)) | set(_frame_sets(prepared.baseline))
    if old_frames.intersection(_frame_sets(settlement)):
        raise ValueError('pending restart reuses archived visual evidence')
    cards_for_fresh_check, retained_definitions = _verified_definition_rows(prepared, components['cards'],
        enabled=settlement.get('keep_verified_card_definitions'))
    for key, component in components.items():
        _reject_retained(cards_for_fresh_check if key == 'cards' else component)
        _no_business_submission(component)
        if (component.get('accepted') is not True or component.get('returned_unchanged') is not True
                or any(component.get(name) != value for name, value in epoch.items())):
            raise ValueError('pending restart component is incomplete or from another epoch: '+key)
        frames = _frame_sets(component)
        if not frames or any(not isinstance(frame, str) or not frame for frame in frames):
            raise ValueError('pending restart component lacks independent fresh frames: '+key)
    allowed_card_sources = ('observed_detail', 'observed_badge_absence')
    if retained_definitions is not None:
        allowed_card_sources += ('retained_verified_card',)
    if any(row.get('source') not in allowed_card_sources for row in components['cards'].get('cards', ())):
        raise ValueError('pending restart cards require fresh complete details')
    state, comparison = resolve_hif_committed_observation(prepared.comparison_bundle, prepared.candidates,
        core=core, components=components, baseline_components=prepared.baseline['components'],
        committed_action=prepared.action)
    if state is None or comparison.get('accepted') is not True:
        raise ValueError('pending restart has no unique complete semantic outcome')
    if not _same(settlement.get('expected_state'), asdict(state)):
        raise ValueError('pending restart producer state differs from the complete comparison')
    _no_business_submission(history)
    _complete_turn_tail(history, state.turn)
    if (not _preinput_history_matches_core(history, core, empty_hand=not state.deck.hand)
            or any(key in history and history[key] != value for key, value in epoch.items())
            or (old_frames | set(_frame_sets(settlement))).intersection(_frame_sets(history))):
        raise ValueError('pending restart latest history is not fresh and unchanged')
    if len({group['frame_ids'][0] for group in (history, history['before'], history['after'])}) != 3:
        raise ValueError('pending restart history reuses its own capture groups')
    history_inputs = history.get('inputs', ())
    if (type(history.get('input_click_count')) is not int or history['input_click_count'] != len(history_inputs)
            or len(history_inputs) < 2 or history_inputs[0].get('closing') is not False
            or history_inputs[-1].get('closing') is not True
            or any(item.get('phase') != 'turn_tail_scroll' for item in history_inputs[1:-1])):
        raise ValueError('pending restart history navigation is incomplete')
    for entry in history_inputs:
        _verified_endpoints(entry.get('endpoints'), prepared.new_pid)
    by_id = {ref.instance_id: ref for ref in state.deck.hand}
    state = replace(state, deck=state.deck.replace_zones(hand=tuple(
        by_id[key] for key in comparison['observed_hand_instance_ids'])))
    rebound = replace(prepared.comparison_bundle, state=state,
        expired_start_scores=verified_expired_start_scores(prepared.comparison_bundle, state),
        expired_start_drinks=verified_expired_start_drinks(prepared.comparison_bundle, state),
        hand_instance_ids=tuple(ref.instance_id for ref in state.deck.hand),
        inventory_keys=expected_permanent_inventory_keys(prepared.comparison_bundle, state),
        drink_slots=comparison['drink_slots'])
    event = copy.deepcopy(settlement)
    event.update(comparison, accepted=True, expected_state=asdict(state),
        observation_source='verified_pending_end_restart', source_epoch=_epoch(prepared.old_run),
        source_game_pid=prepared.old_pid, game_pid=prepared.new_pid, pending_source=prepared.row['id'])
    if retained_definitions is not None:
        event['retained_card_definitions'] = retained_definitions
    event = bind_settlement_history(rebound, event, history, run=prepared.new_run, game_pid=prepared.new_pid)
    return rebound, state, event


def _component_navigation(row, reports):
    """Account for recovery and the three completed readers, with no drink input."""
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints
    from .native_pending_components import _navigation_dispatch_matches

    recovery = row.get('context_recovery', {})
    if (recovery.get('accepted') is not True or recovery.get('returned_to_live') is not True
            or recovery.get('context') != row.get('observed_live_context')
            or recovery.get('game_pid') != row['endpoints']['game']['pid']
            or row.get('context_recovery_source')):
        raise ValueError('pending restart component source lacks its original fresh identity')
    readers = [reports[0]['inventory_events'][0], *[r['status_events'][0] for r in reports[1:3]]]
    proofs = []
    for kind, reader in zip(('recovery', 'inventory', 'status', 'items'), (recovery, *readers), strict=True):
        inputs = reader.get('inputs', ())
        if type(reader.get('input_click_count')) is not int or reader['input_click_count'] != len(inputs):
            raise ValueError('pending restart component navigation count is incomplete')
        if kind != 'recovery' and (len(inputs) < 2 or inputs[0].get('phase') != 'open'
                or inputs[-1].get('phase') != 'close' or any(entry.get('phase') not in
                ({'scroll', 'scroll_top'} if kind == 'items' else {'scroll'}) for entry in inputs[1:-1])):
            raise ValueError('pending restart component navigation order changed')
        proofs.extend((kind, entry) for entry in inputs)
    counts = [reader['input_click_count'] for reader in readers]
    if ([r.get('total_input_click_count') for r in reports[:4]] != [*counts, 0]
            or row['settlement'].get('total_input_click_count') != sum(counts)
            or row.get('input_count') != len(proofs) or len(row.get('native_input_dispatches', ())) != len(proofs)):
        raise ValueError('pending restart component source contains unaccounted input')
    for actual, (kind, proof) in zip(row['native_input_dispatches'], proofs, strict=True):
        _verified_endpoints(actual.get('endpoints'), row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], actual['endpoints'])
        if actual['endpoints']['game'].get('observed_client_size') != list(row['endpoints']['game']['client_size']):
            raise ValueError('pending restart component navigation changed its viewport')
        if kind == 'recovery':
            if proof.get('phase') not in ('live', 'menu', 'notebook', 'identity'):
                raise ValueError('pending restart identity navigation phase changed')
            proof = dict(proof, phase='open', box=proof.get('target'))
        elif kind == 'items' and proof['phase'] in ('scroll', 'scroll_top'):
            swipe = (560, 610, 560, 530, 400) if proof['phase'] == 'scroll' else (560, 220, 560, 610, 600)
            if actual.get('duration') != swipe[4]:
                raise ValueError('pending restart item swipe duration changed')
            proof = dict(proof, phase='scroll', begin=list(swipe[:2]), end=list(swipe[2:4]))
        if not _navigation_dispatch_matches(actual, proof, status=kind == 'status',
                                             inventory_customization=kind == 'inventory'):
            raise ValueError('pending restart component input differs from its recorded navigation')


def load_pending_restart_components_source(runtime, name, module, *, pending_source):
    """Retain original same-process readers; this does not accept the failed scan."""
    from live_decision.opening_observer import opening_component
    from live_decision.high_score.restart import _frame_sets, _reject_retained

    from .native_opening_root import PARENT_FIELDS, verify_native_round_context
    from .native_pending_components import _same, _frames, _core_frames

    row = load_result(runtime, name)
    if (row.get('native_pending_restart_source') != pending_source or name == pending_source
            or any(row.get(key) for key in (*PARENT_FIELDS, 'native_support_source',
                'native_pending_restart_components_source') if key != 'native_pending_restart_source')
            or row.get('live_return') is True or row.get('pending_restart_settlement')):
        raise ValueError('pending restart component source differs from the original pending END')
    _no_business_submission(row)
    run = module.HifLiveRunContext(**row['observed_live_context'])
    pid = row['endpoints']['game']['pid']
    _verify_recorded_input_host(row)
    round_context = verify_native_round_context(row, module, required=True)
    prepared = prepare_pending_restart(runtime, pending_source, module, new_run=run, new_pid=pid,
        new_round_context=round_context)
    verify_native_round_context(row, module, parent=prepared.row, required=True)
    reports = row.get('live_events', ())
    if ([r.get('event') for r in reports] != ['live_inventory_observe', *['live_status_observe']*3,
                                            'live_committed_state']
            or any(r.get('executed') is True or r.get('execution_selection_click_count', 0) for r in reports)):
        raise ValueError('pending restart component source contains another operation')
    for report in reports:
        _no_business_submission(report)
    partial = row.get('settlement', {})
    initial = partial.get('initial_core')
    if (not _same(reports[-1], partial) or partial.get('accepted') is not False
            or partial.get('reason') != 'HIF opening component failed: drinks'
            or partial.get('force_fresh_components') is not True or partial.get('execution_submit_count') != 0
            or partial.get('maintained_components') != [] or partial.get('resumed_fresh_components')
            or 'core' in partial or 'expected_state' in partial or not _core_frames(initial)
            or set(partial.get('components', {})) != {'cards', 'status', 'items'}):
        raise ValueError('pending restart source is not the three-reader prefix before drinks')
    failure_events = reports[3].get('status_events', ())
    if len(failure_events) != 1:
        raise ValueError('pending restart source lacks one failed drink reader')
    failure = failure_events[0]
    if (reports[3].get('accepted') is not False or failure.get('phase') != 'complete_drink_inventory'
            or failure.get('accepted') is not False or failure.get('input_click_count') != 0
            or failure.get('drinks_consumed') != 0 or failure.get('full_drink_state_ready') is not False
            or failure.get('strategy_state_bound') is not False
            or failure.get('reason') != 'drink occupancy has unresolved or unstable slots'
            or not failure.get('events') or any(e.get('phase') != 'drink_slot_occupancy'
                or e.get('accepted') is not False or e.get('input_click_count') != 0
                or e.get('drinks_consumed') != 0 or e.get('inputs') for e in failure['events'])):
        raise ValueError('pending restart failed drink reader contains an operation or a different failure')
    archived = set(_frame_sets(prepared.row)) | set(_frame_sets(prepared.baseline))
    if archived.intersection(_frame_sets(partial)):
        raise ValueError('pending restart readers reuse old-process visual evidence')
    components = {}
    for index, key in enumerate(('cards', 'status', 'items')):
        component = opening_component(reports[index], key, run)
        if not _same(component, partial['components'][key]):
            raise ValueError('pending restart component differs from its original reader: '+key)
        checking = component
        if key == 'cards':
            checking, _ = _verified_definition_rows(prepared, component,
                enabled=partial.get('keep_verified_card_definitions'))
            inventory = reports[0]['inventory_events'][0]
            if (inventory.get('phase') != 'inventory_zone_observation'
                    or inventory.get('accepted') is not True or inventory.get('live_state_unchanged') is not True
                    or not _same(inventory.get('composition'), component.get('observed_composition'))
                    or any(inventory.get(k) != component.get(k) for k in ('before_frame_ids', 'after_frame_ids'))):
                raise ValueError('pending restart cards lost their independent inventory observation')
        _reject_retained(checking)
        if (any(component.get(k) != getattr(run, k) for k in ('run_id', 'revision'))
                or not _frames(component.get('before_frame_ids')) or not _frames(component.get('after_frame_ids'))
                or set(component['before_frame_ids']) & set(component['after_frame_ids'])):
            raise ValueError('pending restart reusable component lacks same-epoch return evidence: '+key)
        components[key] = copy.deepcopy(component)
    _component_navigation(row, reports)
    audit = dict(event='native_pending_restart_components_resume', source=name, pending_source=pending_source,
        run_id=run.run_id, revision=run.revision, game_pid=pid, source_client_size=row['endpoints']['game']['client_size'],
        component_names=['cards', 'status', 'items'], source_initial_frame_ids=list(initial['frame_ids']),
        source_core_frame_ids=[], source_reader_frame_ids=sorted(set(_frame_sets(row))),
        source_final_core_available=False, source_settlement_accepted=False, failed_component='drinks',
        navigation_input_count=row['input_count'], business_submit_count=0, submission_authorized=False)
    return dict(prepared=prepared, row=row, components=components, core=copy.deepcopy(initial), audit=audit)


def verify_pending_restart_components_continuity(prepared, resumed, history):
    """A new complete current-turn tail proves no intervening card or drink action."""
    from live_decision.high_score.restart import _epoch, _frame_sets
    from live_decision.high_score.maintenance import _preinput_history_matches_core

    from .native_settlement_resume import _verified_endpoints

    if (resumed['prepared'] != prepared or not isinstance(history, dict)
            or not _preinput_history_matches_core(history, resumed['core'])
            or any(key in history and history[key] != value for key, value in _epoch(prepared.new_run).items())
            or set(_frame_sets(history)) & set(resumed['audit']['source_reader_frame_ids'])):
        raise ValueError('pending restart component continuity is missing, changed, or reuses source frames')
    _no_business_submission(history)
    _complete_turn_tail(history, prepared.bundle.state.turn+1)
    if len({group['frame_ids'][0] for group in (history, history['before'], history['after'])}) != 3:
        raise ValueError('pending restart component history capture groups overlap')
    inputs = history.get('inputs', ())
    if (type(history.get('input_click_count')) is not int or history['input_click_count'] != len(inputs)
            or len(inputs) < 2 or inputs[0].get('closing') is not False or inputs[-1].get('closing') is not True
            or any(entry.get('phase') != 'turn_tail_scroll' for entry in inputs[1:-1])):
        raise ValueError('pending restart component history navigation is incomplete')
    for entry in inputs:
        _verified_endpoints(entry.get('endpoints'), prepared.new_pid)


def _verify_resumed_completion(settlement, resumed):
    from live_decision.high_score.restart import _frame_sets

    from .native_pending_components import verify_pending_component_completion

    audit = resumed['audit']
    # Reuse the existing exact component/core verifier through its metadata
    # adapter. Neither the archived nor newly recorded result is rewritten.
    checking = dict(native_pending_components_audit=audit, native_pending_components_source=audit['source'],
        endpoints=dict(game=dict(pid=audit['game_pid'], client_size=audit['source_client_size'])))
    verify_pending_component_completion(checking, settlement, resumed)
    if any(set(_frame_sets(settlement.get('components', {}).get(key, {}))) & set(audit['source_reader_frame_ids'])
           for key in ('drinks', 'turns')):
        raise ValueError('pending restart completion reused a failed drink or old turn observation')


def verify_pending_restart_component_completion(row, settlement, resumed):
    from live_decision.high_score.restart import _same

    audit = resumed['audit']
    if (row.get('native_pending_restart_components_source') != audit['source']
            or not _same(row.get('pending_restart_components_resume'), audit)
            or row.get('native_pending_restart_source') != audit['pending_source']
            or row['endpoints']['game']['pid'] != audit['game_pid']
            or row['endpoints']['game']['client_size'] != audit['source_client_size']
            or any(row.get('observed_live_context', {}).get(key) != audit[key] for key in ('run_id', 'revision'))):
        raise ValueError('pending restart completion changed its source, process or observation epoch')
    _verify_resumed_completion(settlement, resumed)


def load_verified_pending_restart_source(runtime, name, module, trail=()):
    from live_decision.high_score.restart import _same

    from .native_opening_root import verify_native_round_context
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints
    if name in trail or diagnostic_depth_reached(trail):
        raise ValueError('pending restart provenance is cyclic or unbounded')
    row = load_result(runtime, name)
    source = row.get('native_pending_restart_source')
    if (not isinstance(source, str) or source == name or source in trail or row.get('live_return') is not True
            or any(row.get(key) for key in ('native_pending_source', 'native_resume_source',
                'native_selected_source', 'native_choice_source', 'native_inprocess_source',
                'native_restart_source', 'native_settled_restart_source', 'native_pending_components_source',
                'native_support_source', 'native_manual_choice_source'))):
        raise ValueError('pending restart completion must have one explicit pending source')
    _no_business_submission(row)
    for event in row.get('live_events', ()):
        _no_business_submission(event)
        if event.get('executed') is True or event.get('execution_selection_click_count', 0):
            raise ValueError('pending restart completion contains a business selection')
    game_pid = row['endpoints']['game']['pid']
    _verify_recorded_input_host(row)
    dispatches = row.get('native_input_dispatches', ())
    if type(row.get('input_count')) is not int or row['input_count'] != len(dispatches):
        raise ValueError('pending restart navigation accounting is incomplete')
    for entry in dispatches:
        _verified_endpoints(entry.get('endpoints'), game_pid)
        _same_input_host(row['endpoints'], entry['endpoints'])
    round_context = verify_native_round_context(row, module, required=True)
    reused_source = row.get('native_pending_restart_components_source')
    resumed = None
    if reused_source:
        if reused_source == name or reused_source in trail:
            raise ValueError('pending restart component provenance is cyclic')
        resumed = load_pending_restart_components_source(runtime, reused_source, module, pending_source=source)
        prepared = resumed['prepared']
        if (not _same(row.get('observed_live_context'), asdict(prepared.new_run))
                or row.get('context_recovery_source') != reused_source
                or not _same(row.get('context_recovery'), resumed['row'].get('context_recovery'))):
            raise ValueError('pending restart component completion lost its original identity provenance')
        verify_pending_restart_component_completion(row, row.get('settlement', {}), resumed)
    else:
        prepared = prepare_pending_restart(runtime, source, module,
            new_run=module.HifLiveRunContext(**row['observed_live_context']), new_pid=game_pid,
            new_round_context=round_context)
    verify_native_round_context(row, module, prepared.row, required=True)
    complete = [event for event in row.get('live_events', ())
                if event.get('event') == 'live_committed_state' and event.get('accepted') is True]
    if len(complete) != 1 or not _same(complete[0], row.get('settlement')):
        raise ValueError('pending restart lacks its original complete reader event')
    bundle, state, event = finalize_pending_restart(prepared, settlement=row['settlement'], history=row.get('post_history', {}),
        resumed=resumed, continuity_history=row.get('pending_restart_resume_history'))
    if not _same(event, row.get('pending_restart_settlement')):
        raise ValueError('pending restart saved binding differs from its fresh evidence')
    return bundle, state, event, row
