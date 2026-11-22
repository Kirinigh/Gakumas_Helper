"""Business-source contracts for the route owner's original-save navigation.

No page recognition, input, navigation grant, or cache bypass lives here.
The original native loaders remain responsible for replay and computation.
"""
from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict, replace

from .opening_observer import valid_opening_frame_ids
from .hif_live_recovery import identity, require_budget, load_automatic_chain, automatic_host_active, constrain_parent_deadline
from .hif_native.native_handoff_resume import _binding

_OPENING_PANEL_PHASES = {'status': 'status_inspection', 'items': 'item_description_inspection',
                        'drinks': 'complete_drink_inventory', 'turns': 'turn_information_observation'}


def _bound_request(request, record):
    from .hif_battle_entry import HifBattleRequest

    if not isinstance(request, HifBattleRequest) or not isinstance(record, dict):
        raise ValueError('HIF navigation requires the original battle request')
    live = (record.get('handoff') or {}).get('live_context')
    if live is not None and not isinstance(live, dict):
        raise ValueError('HIF navigation opening metadata is malformed')
    current = replace(request, live_run_id=(live or {}).get('run_id'), live_revision=(live or {}).get('revision'))
    if request.live_run_id is not None and current != request:
        raise ValueError('HIF navigation changed the original Live epoch')
    # Read-only evidence is also needed while the route owner reconciles its
    # persisted navigation pending. Actual battle entry still verifies_record
    # and requires that pending to be completely settled.
    current.verify_identity(record)
    return dict(route_run_id=current.run_id, handoff_id=current.handoff_id,
        ownership_revision=current.ownership_revision, game_pid=current.game_pid,
        ledger_revision=current.ledger_revision, policy_profile=current.policy_profile)


def verify_unbound_opening_source(record, source):
    """Prove a stopped metadata-only opening; never invent a Live epoch/state."""
    from .hif_battle_entry import HifBattleRequest
    from .hif_live_recovery import validate_control

    if not isinstance(source, dict) or not isinstance(source.get('id'), str) or not source['id']:
        raise ValueError('HIF unbound opening requires its original source record')
    if (any(not isinstance(source.get(key), dict) for key in
            ('automatic_battle', 'formal_params', 'endpoints', 'failure_event'))
            or not isinstance(source['endpoints'].get('game'), dict)):
        raise ValueError('HIF unbound opening source metadata is incomplete')
    request = HifBattleRequest.from_identity(record, attempt_id='0000-opening-navigation', client_size=(720, 1280))
    control = source.get('automatic_battle') or {}
    validate_control(control)
    if (record['handoff'].get('live_context') is not None or record['handoff'].get('terminal_receipt') is not None
            or control['identity'] != _bound_request(request, record)
            or source.get('status') != 'STOP' or source.get('information_boundary_execution') is not True
            or source.get('handoff_id') != request.handoff_id
            or source.get('formal_params', {}).get('hif_policy_profile') != request.policy_profile
            or source.get('formal_params', {}).get('expected_plan') != record['handoff'].get('plan')
            or source.get('endpoints', {}).get('game', {}).get('pid') != request.game_pid
            or source.get('live_events') != [] or source.get('transaction_closed') is True
            or any(source.get(key) is not None for key in ('handoff_opening', 'observed_live_context',
                'round_context', 'context_recovery', 'observer_returned', 'native_input_preparation',
                'native_pending_source', 'native_selected_source', 'native_inprocess_source',
                'native_rule_resume_source', 'gray_selection_journal', 'terminal_settlement'))):
        raise ValueError('HIF unbound opening is not its original metadata-only STOP')
    failure = source.get('failure_event') or {}
    inputs, observations = failure.get('inputs'), failure.get('observations')
    dispatches = source.get('native_input_dispatches')
    if (failure.get('phase') != 'hif_context_recovery' or failure.get('accepted') is not False
            or failure.get('returned_to_live') is not False or not isinstance(inputs, list)
            or not isinstance(observations, list) or len(observations) < len(inputs)
            or not isinstance(dispatches, list) or len(dispatches) != len(inputs)
            or type(source.get('input_count')) is not int or source['input_count'] != len(inputs)
            or type(failure.get('input_click_count')) is not int or failure['input_click_count'] != len(inputs)
            or source.get('error') != failure.get('reason')):
        raise ValueError('HIF unbound opening has incomplete original navigation receipts')
    for action, batch, dispatch in zip(inputs, observations, dispatches, strict=False):
        if not isinstance(action, dict) or not isinstance(dispatch, dict):
            raise ValueError('HIF unbound opening contains a malformed navigation receipt')
        endpoints = action.get('endpoints')
        if (not isinstance(endpoints, dict) or not {'game', 'maa', 'agent'} <= set(endpoints)
                or any(not isinstance(endpoint, dict) for endpoint in endpoints.values())):
            raise ValueError('HIF unbound opening lacks original input endpoint proofs')
        target, keys = action.get('target'), action.get('frame_ids')
        times = [action.get(key) for key in ('captured_at', 'post_click_started_at', 'post_click_finished_at', 'frame_age_ms')]
        if (action.get('phase') not in ('live', 'menu', 'notebook', 'identity')
                or action.get('succeeded') is not True or action.get('uncertain') is True
                or not isinstance(target, list) or len(target) != 4 or any(type(v) is not int for v in target)
                or min(target[:2]) < 0 or min(target[2:]) <= 0
                or not isinstance(keys, list) or not keys
                or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys)
                or not isinstance(batch, list) or len(batch) != len(keys)
                or any(not isinstance(row, dict) for row in batch)
                or [row.get('frame_id') for row in batch] != keys
                or any(row.get('phase') != action['phase'] or row.get('target') != target for row in batch)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in times)
                or not times[0] <= times[1] <= times[2] or not 0 <= times[3] <= 1000
                or not 0 <= (times[1] - times[0]) * 1000 <= 1000
                or dispatch.get('kind', 'click') != 'click'
                or (dispatch.get('x'), dispatch.get('y')) != (target[0]+target[2]//2, target[1]+target[3]//2)
                or dispatch.get('endpoints') != action.get('endpoints')
                or endpoints['game'].get('pid') != request.game_pid
                or any(row.get('integrity_rid') != 0x3000 for row in endpoints.values())):
            raise ValueError('HIF unbound opening contains an uncertain or unmatched navigation dispatch')
    preparation = source.get('native_navigation_preparation')
    if preparation is not None and (not isinstance(preparation, dict) or not dispatches
            or preparation.get('dispatch_state') != 'dispatch_attempted'
            or any(preparation.get(key) != dispatches[-1].get(key) for key in ('x', 'y', 'endpoints'))):
        raise ValueError('HIF unbound opening has an unmatched prepared navigation input')
    return dict(boundary_kind='unbound_opening', source_result=source['id'], input_authorized=False)


def verify_unstarted_menu_source(record, source):
    """Recover unfinished menu navigation from the actual acknowledged clicks."""
    proof = verify_unbound_opening_source(record, source)
    failure = source['failure_event']
    inputs, observations = failure['inputs'], failure['observations']
    polls = failure.get('destination_observations', ())
    phases = [row['phase'] for row in inputs]
    if (source['error'] != 'HIF recovery destination page was not confirmed'
            or source.get('error_type') != 'HifRecoveryError'
            or source.get('total_input_count') != len(phases)
            or phases not in (['live'], ['live', 'menu'])
            or len(observations) != len(phases) or len(polls) != len(phases)
            or record['transition']['pending'] is not None or (record['ledger'] or {}).get('pending') is not None):
        raise ValueError('HIF opening resume is not its completed menu-only timeout')
    poll = polls[0]
    counts = poll.get('phase_counts', {})
    baseline = observations[0][0]
    if (any(type(baseline.get(k)) is not int or baseline[k] < 0 for k in ('stamina', 'turns'))
            or any(any(row.get(k) != baseline[k] for k in ('stamina', 'turns')) for row in observations[0])):
        raise ValueError('HIF opening menu timeout lacks unchanged Live observations')
    expected_live = {k: baseline[k] for k in ('stamina', 'turns')}
    if phases == ['live', 'menu']:
        if (poll.get('expected') != ['menu'] or poll.get('confirmed') is not True
                or polls[1].get('expected') != ['notebook', 'identity'] or polls[1].get('confirmed') is not False):
            raise ValueError('HIF notebook navigation does not match the original menu transition')
        return dict(proof, expected_live=expected_live, completed_phases=phases)
    if (poll.get('expected') != ['menu'] or poll.get('confirmed') is not False
            or set(counts) != {'live'} or type(counts['live']) is not int or counts['live'] < 2):
        raise ValueError('HIF opening menu timeout lacks unchanged Live observations')
    first, last = poll.get('first', {}), poll.get('last', {})
    for sample in (first, last):
        live = sample.get('result', {})
        elapsed = sample.get('elapsed_ms')
        if (live.get('phase') != 'live' or any(live.get(k) != baseline[k] for k in ('stamina', 'turns'))
                or not isinstance(sample.get('frame_id'), str) or not sample['frame_id']
                or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError('HIF opening timeout destination evidence is incomplete')
    if first['frame_id'] == last['frame_id'] or first['elapsed_ms'] >= last['elapsed_ms']:
        raise ValueError('HIF opening timeout lacks independent destination observations')
    return dict(proof, expected_live=expected_live)


_PLAYABLE_WAIT_TIMEOUT = 'HIF opening playable wait exceeded its navigation budget'
_OPENING_SAMPLE_FAILURE = 'HIF recovery three-frame observation failed: '


def _stopped_during_opening_observation(source):
    failure = source.get('failure_event')
    return (isinstance(failure, dict) and failure.get('inputs') == [] and failure.get('observations') == []
        and ('readiness_observations' in failure or 'sampling_rejections' in failure))


def _verify_initial_observation(record, source):
    """The first round stopped in its initial read, before any UI dispatch."""
    proof = verify_unbound_opening_source(record, source)
    failure = source['failure_event']
    previous = source['automatic_battle']['previous_record']
    expected_resume = (dict(boundary_kind='unstarted_playable_wait', source_result=previous, input_authorized=False)
        if previous is not None else None)
    current_screen = record['handoff'].get('source') == 'current_battle_page'
    if (not current_screen and (record['handoff']['round_number'] != 1 or record['handoff'].get('prior_terminal_source') is not None)
            or record['transition']['pending'] is not None or record['transition']['stopped'] is not False
            or source.get('error_type') != 'HifRecoveryError' or not _stopped_during_opening_observation(source)
            or type(source.get('total_input_count')) is not int or source['total_input_count'] != 0
            or source.get('published_components') or source.get('live_raw_fields') != []
            or source.get('live_return') is not None or source.get('opening_navigation_resume') != expected_resume
            or any(failure.get(key) for key in ('transitions', 'destination_observations', 'destination_rejections'))):
        raise ValueError('HIF opening lacks a zero-input first-round observation')
    if source['error'] == _PLAYABLE_WAIT_TIMEOUT:
        _verify_playable_wait_poll(failure.get('readiness_observations'))
    else:
        # The reader records its rejected samples before raising the first
        # reason from the final group. Any field can fail this passive read.
        groups = failure.get('sampling_rejections')
        if (not isinstance(groups, list) or not groups
                or any(not isinstance(group, list) or not group for group in groups)
                or any(not isinstance(sample, dict) or not isinstance(sample.get('reason'), str) or not sample['reason']
                    for group in groups for sample in group)
                or source['error'].removeprefix(_OPENING_SAMPLE_FAILURE) != groups[-1][0]['reason']):
            raise ValueError('HIF opening lacks the rejected samples for its recorded read failure')
    # Keep the proof name already persisted by the initial readiness timeout.
    return dict(proof, boundary_kind='unstarted_playable_wait')


def _verify_playable_wait_poll(poll):
    if not isinstance(poll, dict) or type(poll.get('count')) is not int or poll['count'] < 2:
        raise ValueError('HIF opening playable wait lacks its original readiness observations')
    first, last = poll.get('first'), poll.get('last')
    for sample in (first, last):
        if (not isinstance(sample, dict) or sample.get('reason') != 'HIF opening effects have not returned playable Live'
                or not isinstance(sample.get('frame_id'), str) or not sample['frame_id']
                or type(sample.get('elapsed_ms')) not in (float, int) or not math.isfinite(sample['elapsed_ms'])):
            raise ValueError('HIF opening playable wait lacks its original readiness observations')
    if first['elapsed_ms'] != 0 or last['elapsed_ms'] < 8000:
        raise ValueError('HIF opening playable wait did not reach its recorded timeout')


def _unstarted_component_source(rows):
    """Keep the original observation when a reserved resume failed before UI."""
    latest = rows[-1]
    resume = latest.get('opening_navigation_resume')
    if not isinstance(resume, dict) or latest.get('observer_returned') is not None:
        return latest, False
    originals = [r for r in rows[:-1] if r['id'] == resume.get('source_result')]
    if len(originals) != 1:
        raise ValueError('HIF unstarted resume lost its original observation')
    original = originals[0]
    for row in rows[rows.index(original) + 1:]:
        events = row.get('live_events', ())
        if (row.get('opening_navigation_resume') != resume or row.get('status') != 'STOP'
                or row.get('input_count') != 0 or row.get('total_input_count') != 0
                or row.get('frames') != [] or row.get('native_input_dispatches') != []
                or any(row.get(key) is not None for key in ('handoff_opening', 'observer_returned',
                    'native_input_preparation', 'native_navigation_preparation', 'gray_selection_journal'))
                or row.get('published_components') or len(events) != 1
                or events[0].get('event') != 'live_decision_exception_stop'
                or events[0].get('runtime_phase') != 'runtime_prechecks'
                or events[0].get('executed') is not False or events[0].get('execution_submit_count') != 0
                or events[0].get('execution_selection_click_count') != 0
                or events[0].get('input_may_have_been_dispatched') is not False
                or any(row.get(key) != original.get(key)
                    for key in ('observed_live_context', 'round_context', 'context_recovery'))):
            raise ValueError('HIF unstarted resume contains new or uncertain UI work')
    return original, True


def find_unstarted_menu_resume(runtime, session, *, maximum_actions, timeout_seconds,
                               parent_deadline=None, authorise=None):
    """Resume metadata from fresh UI, retaining the unique original child budget."""
    import time

    from .hif_entry_round import UNREAD_INVENTORY_SOURCES
    from .hif_recovery_reports import matching_recovery_reports
    from .hif_opening_item_tail import is_open_item_tail, is_drink_stop_after_item_return
    from .hif_opening_status_resume import is_open_status_prefix, is_closed_items_after_status

    bound = identity(session)
    if not any(json.loads(path.read_text(encoding='utf-8')).get('handoff_id') == bound['handoff_id']
               for path in matching_recovery_reports(runtime, bound['handoff_id'])):
        return None
    rows = load_automatic_chain(Path(runtime), bound)
    # A closed, undispatched observation preserves the last verified component
    # source. Validate that source through its existing reader, then follow its
    # actual successors in order without replaying any historical navigation.
    observations = []
    while len(rows) > 1:
        row = rows[-1]
        resume = row.get('opening_navigation_resume') or {}
        events = row.get('live_events') or []
        if is_open_status_prefix(row) and len(events) == 1:
            observations.append(row)
            rows = rows[:-1]
            continue
        if (not (resume.get('retained_components') or resume.get('pending_inventory_return'))
                or (row.get('input_count') != 0 and not _stopped_before_opening_component(row)
                    and not is_open_status_prefix(row)
                    and not is_closed_items_after_status(row) and not is_open_item_tail(row)
                    and not is_drink_stop_after_item_return(row))
                or not events or events[-1].get('event') != 'live_opening_state'
                or events[-1].get('accepted') is not False):
            break
        observations.append(row)
        rows = rows[:-1]
    latest = rows[-1]
    # A repeated passive wait has real observations of its own. A reserved
    # component retry instead has no fresh UI and keeps its original proof.
    source, reserved = (latest, False) if _stopped_during_opening_observation(latest) else _unstarted_component_source(rows)
    events = source.get('live_events', ())
    resume = source.get('opening_navigation_resume') or {}
    if resume.get('completed_phases'):
        prior = next((row for row in rows if row['id'] == source['automatic_battle']['previous_record']), None)
        if prior is None or verify_unstarted_menu_source(session.record, prior) != resume:
            raise ValueError('HIF continued notebook differs from its original navigation receipts')
    if _stopped_during_opening_observation(source):
        # No earlier opening in this battle may have issued even a menu click.
        for previous in rows[:rows.index(source)]:
            _verify_initial_observation(session.record, previous)
        proof = _verify_initial_observation(session.record, source)
        if len(rows) > 1 and rows[-2].get('error') == source.get('error'):
            raise ValueError('HIF repeated the same zero-input opening failure without progress')
    elif ((source.get('opening_navigation_resume') or {}).get('boundary_kind') in UNREAD_INVENTORY_SOURCES
            and source.get('input_count') == 0 and events and events[-1].get('event') == 'live_opening_state'
            and events[-1].get('components') == {}):
        proof = _verify_unread_initial_inventory(session.record, rows)
        reserved = True
    elif len(events) == 1 and events[0].get('reason') == 'inventory navigation Live visual family is unresolved':
        if source is not latest:
            raise ValueError('HIF initial inventory requires its original observation')
        proof = _first_inventory_identity_source(session.record, rows)
        reserved = True
    elif events and events[-1].get('reason') == 'HIF opening component failed: status':
        if source is latest and (source.get('opening_navigation_resume') or {}).get('recover_closed_inventory') is True:
            from .hif_opening_status_resume import verify_open_status_source

            proof = verify_open_status_source(session.record, rows)
            reserved = True
        else:
            proof = verify_unstarted_status_source(session.record, source)
    elif events and events[-1].get('reason') == 'HIF opening component failed: cards':
        if (source.get('opening_navigation_resume') or {}).get('boundary_kind') in UNREAD_INVENTORY_SOURCES:
            if source is not latest:
                raise ValueError('HIF closed inventory lacks its immediate notebook predecessor')
            proof = _verify_closed_notebook_inventory(session.record, rows)
            reserved = True  # Reconstruct the finished navigation rather than retry its panel.
        else:
            proof = verify_unstarted_inventory_source(session.record, source)
    elif events and events[-1].get('reason') == 'HIF opening component failed: items':
        proof = verify_unstarted_items_source(session.record, source)
    elif events and events[-1].get('reason') == 'HIF opening component failed: drinks':
        proof = verify_unstarted_drinks_source(session.record, source)
    elif events and events[-1].get('reason') == 'HIF opening component failed: turns':
        if source is latest and (source.get('opening_navigation_resume') or {}).get('resume_status_panel') is True:
            from .hif_opening_turn_resume import verify_retained_turn_source

            proof = verify_retained_turn_source(session.record, rows)
            reserved = True
        else:
            proof = verify_unstarted_turns_source(session.record, source)
    else:
        proof = verify_unstarted_menu_source(session.record, latest)
    for observation in reversed(observations):
        counted_return = not reserved and ('pending_inventory_return' in proof or _stopped_before_opening_component(latest))
        proof = _resume_unapplied_opening(session.record, latest, proof, observation, counted_return=counted_return)
        latest, reserved = observation, True
    control = deepcopy(latest['automatic_battle'])
    if (control['maximum_actions'], control['timeout_seconds']) != (maximum_actions, timeout_seconds):
        raise ValueError('HIF opening recovery cannot replace the original battle budget')
    from route.cultivation import _budget_container

    windows = (_budget_container(session.record['handoff']) or {}).get('automatic_cultivation', {}).get('execution_windows', ())
    new_parent = windows[-1].get('battle_resume', {}) if windows else {}
    if not automatic_host_active() and proof['boundary_kind'] == 'unbound_notebook' and new_parent:
        from .hif_joint_time import bind_opening_time

        control = bind_opening_time(control, latest, session.record, parent_deadline=parent_deadline, authorise=authorise)
    if not automatic_host_active() and not observations and not (proof.get('recover_closed_inventory') or proof.get('resume_status_panel') or proof.get('restore_opening_title')) and proof['boundary_kind'] not in ('unstarted_playable_wait', *UNREAD_INVENTORY_SOURCES) and (
            (new_parent.get('source') == latest['id'] and new_parent.get('boundary') == 'new_round_unstarted')
            or time.time() >= control.get('active_parent_deadline', control.get('parent_deadline', control['deadline']))):
        from .hif_battle_engineering_window import append_opening_parent

        if not callable(authorise):
            raise ValueError('HIF opening recovery requires its verified parent window')
        grant = authorise(source=latest, record=session.record)
        control = append_opening_parent(control, latest, grant, now=time.time(), parent_deadline=parent_deadline)
    constrain_parent_deadline(control, parent_deadline)
    require_budget(control)
    if not reserved:
        control['recoveries_used'] += 1
    return control, latest, proof


def _verify_unstarted_component_context(record, source):
    """Validate the original unbound observer before classifying its failure."""
    from .hif_native.native_opening_resume import verify_no_business_since_opening

    if source.get('native_input_preparation') is not None:
        raise ValueError('HIF partial opening is not the original unbound completed observer')
    verify_no_business_since_opening(source)
    return _verify_unbound_component_context(record, source)


def _verify_unbound_component_context(record, source):
    """Verify the observer, handoff and run independent of the particular input receipts."""
    from types import SimpleNamespace

    from .hif_visual import HifLiveRunContext
    from .hif_battle_entry import HifBattleRequest
    from .hif_live_recovery import closed_observer_record
    from .hif_native.native_opening_root import verify_native_round_context

    request = HifBattleRequest.from_identity(record, attempt_id='0000-opening-status', client_size=(720, 1280))
    if (record['handoff'].get('live_context') is not None or record['transition']['pending'] is not None
            or not closed_observer_record(source) or source.get('live_return') is not False
            or source['automatic_battle']['identity'] != _bound_request(request, record)
            or source.get('handoff_opening') is not None
            or source.get('formal_params', {}).get('hif_policy_profile') != request.policy_profile
            or source.get('formal_params', {}).get('expected_plan') != record['handoff']['plan']):
        raise ValueError('HIF partial opening is not the original unbound completed observer')
    metadata = verify_native_round_context(source, SimpleNamespace(HifLiveRunContext=HifLiveRunContext), required=True)
    if (metadata.round_number != request.round_number
            or json.dumps(asdict(metadata), sort_keys=True) != json.dumps(source.get('round_context'), sort_keys=True)):
        raise ValueError('HIF partial opening round differs from its handoff')
    return request, metadata


def _verify_first_unread_inventory(source):
    events = source['live_events']
    if len(events) != 1 or source.get('published_components') != events:
        raise ValueError('HIF initial inventory already published another observation')
    opening = events[0]
    if (opening.get('event') != 'live_opening_state' or opening.get('accepted') is not False
            or opening.get('reason') != 'inventory navigation Live visual family is unresolved'
            or opening.get('initial_core') is not None or opening.get('components') != {}
            or opening.get('total_input_click_count') != 0):
        raise ValueError('HIF opening inventory already progressed past its first read')
    saved = {f.get('frame_id') for f in source.get('frames', ()) if f.get('label') == 'ProduceRecognitionCards'}
    attempts = opening.get('initial_observations', []) + [opening.get('failure_events', [])]
    groups = []
    for attempt in attempts:
        if len(attempt) != 1 or attempt[0].get('phase') != 'inventory_live_identity':
            raise ValueError('HIF initial inventory lacks the original rejected hand observation')
        failed = attempt[0]
        frames = set(failed.get('frame_ids', ()))
        if (not valid_opening_frame_ids(failed.get('frame_ids')) or failed.get('frame_id') not in frames or not frames <= saved
                or not any(p.get('accepted') is False for p in failed.get('predictions', ()))):
            raise ValueError('HIF initial inventory lacks its complete original hand frames')
        groups.append(frames)
    if len(groups) != 2 or groups[0] & groups[1]:
        raise ValueError('HIF initial inventory lacks its two independent observations')
    return opening


def _first_inventory_identity_source(record, rows):
    from .hif_entry_round import SOURCE
    from .hif_native.native_opening_mapping_resume import verify_opening_navigation

    source = rows[-1]
    recovery = source['context_recovery']
    if recovery.get('round_source') != SOURCE:
        if len(rows) < 2 and record['handoff'].get('source') != 'current_battle_page':
            raise ValueError('HIF notebook continuation requires its completed original recovery')
        for previous in rows[:-1]:
            _verify_initial_observation(record, previous)
        return verify_unstarted_notebook_source(record, source)
    request, _ = _verify_unstarted_component_context(record, source)
    if (len(rows) != 1 or record['transition']['stopped'] is not False
            or request.policy_profile != 'hif-strong-reference-trial-v1'
            or source['automatic_battle'].get('previous_record') is not None
            or source.get('opening_navigation_resume') is not None or source.get('total_input_count') != 0
            or any(source.get(k) is not None for k in ('upstream_ui_journal', 'native_upstream_input_preparation',
                'native_navigation_preparation'))):
        raise ValueError('HIF untouched route entry inventory contains prior work or input')
    opening = _verify_first_unread_inventory(source)
    if opening.get('hand_identity_inspection') or opening.get('input_may_have_been_dispatched'):
        raise ValueError('HIF first route inventory already attempted a hand inspection')
    verify_opening_navigation(source, (), opening)
    expected = {k: recovery['observations'][0][0][k] for k in ('turns', 'stamina')}
    return dict(boundary_kind='unbound_route_entry', source_result=source['id'], input_authorized=False,
                expected_live=expected)


def verify_unstarted_notebook_source(record, source):
    """Keep the completed notebook when its first inventory observation failed before any business input."""
    from .hif_native.native_opening_mapping_resume import verify_opening_navigation

    request, metadata = _verify_unstarted_component_context(record, source)
    resume = source.get('opening_navigation_resume')
    current_screen = record['handoff'].get('source') == 'current_battle_page'
    previous = source['automatic_battle']['previous_record']
    expected_resume = (dict(boundary_kind='unstarted_playable_wait', input_authorized=False, source_result=previous)
        if previous is not None else None)
    if (not current_screen and (request.round_number != 1 or not previous) or resume != expected_resume
            or record['transition']['stopped'] is not False or source.get('total_input_count') != 4):
        raise ValueError('HIF initial inventory is not its original first-round observer')
    opening = _verify_first_unread_inventory(source)
    recovery = source['context_recovery']
    actions = recovery['inputs']
    observations = recovery['observations']
    phases = ['live', 'menu', 'notebook', 'identity']
    preparation = source.get('native_navigation_preparation') or {}
    dispatches = source.get('native_input_dispatches') or []
    if (recovery.get('input_click_count') != 4 or [a.get('phase') for a in actions] != phases
            or len(dispatches) != 4 or preparation.get('dispatch_state') != 'dispatch_attempted'
            or any(preparation.get(k) != dispatches[-1].get(k) for k in ('x', 'y', 'endpoints'))
            or recovery.get('game_pid') != request.game_pid
            or any(source['observed_live_context'].get(k) != record['handoff'][k] for k in ('idol', 'plan', 'max_stamina')
                   if not current_screen or record['handoff'].get(k) is not None)
            or len(observations) != 5 or any(not batch or any(r.get('phase') != phase for r in batch)
                for batch, phase in zip(observations, [*phases, 'live'], strict=True))
            or any(a.get('frame_ids') != [r.get('frame_id') for r in batch] for a, batch in zip(actions, observations))
            or any(v.get('stamina') != observations[0][0].get('stamina') or v.get('turns') != metadata.base_turn_count
                for v in [*observations[0], *observations[-1]])):
        raise ValueError('HIF first-round notebook navigation did not return to its original Live')
    verify_opening_navigation(source, (), opening)
    expected_live = {k: observations[-1][0][k] for k in ('turns', 'stamina')}
    return dict(boundary_kind='unbound_notebook', source_result=source['id'], input_authorized=False,
        expected_live=expected_live)


def _verify_unread_initial_inventory(record, rows):
    """Follow closed zero-input inventory reads back to their original identity source."""
    from .hif_entry_round import UNREAD_INVENTORY_SOURCES

    anchor = len(rows) - 1
    while anchor and (rows[anchor].get('opening_navigation_resume') or {}).get('boundary_kind') in UNREAD_INVENTORY_SOURCES:
        anchor -= 1
    notebook = rows[anchor]
    proof = _first_inventory_identity_source(record, rows[:anchor+1])
    previous_control = notebook['automatic_battle']
    for row in rows[anchor+1:]:
        request, _ = _verify_unstarted_component_context(record, row)
        events = row['live_events']
        control = row['automatic_battle']
        if (row.get('opening_navigation_resume') != proof or row.get('published_components') != events[-1:]
                or events[-1].get('event') != 'live_opening_state' or events[-1].get('components') != {}
                or any(e.get('total_input_click_count') != 0 for e in events) or row.get('native_input_dispatches') != []
                or row.get('native_navigation_preparation') is not None
                or any(type(row.get(k)) is not int or row[k] != 0 for k in ('input_count', 'total_input_count'))
                or any(row.get(k) != notebook.get(k) for k in ('context_recovery', 'round_context', 'observed_live_context'))
                or control['recoveries_used'] != previous_control['recoveries_used']
                or previous_control.get('opening_parent_window') is not None
                    and control.get('opening_parent_window') != previous_control['opening_parent_window']):
            raise ValueError('HIF unread inventory has new input, progress, or changed notebook evidence')
        if len(events) == 2:
            _verify_unopened_notebook_inventory(row, request.game_pid, proof['expected_live'])
        elif len(events) != 1 or events[0].get('initial_core') is not None:
            raise ValueError('HIF unread inventory contains an unaccounted opening observation')
        inspection = events[-1].get('hand_identity_inspection')
        if inspection is not None:
            preflight = inspection.get('preflight') or {}
            if (inspection.get('observed_live_context') != row['observed_live_context']
                    or inspection.get('opening') is not True or inspection.get('accepted') is not False
                    or inspection.get('business_submit_count') != 0 or inspection.get('input_click_count') != 0
                    or preflight.get('accepted') is not False or preflight.get('click_dispatched') is not False
                    or preflight.get('input_may_have_been_dispatched') is not False):
                raise ValueError('HIF unread inventory has an uncertain hand inspection')
        proof = dict(proof, source_result=row['id'])
        previous_control = control
    return proof


def _verify_unopened_notebook_inventory(row, game_pid, expected_live):
    """Discard the provisional hand. Only the original notebook carries over."""
    from .hif_native.native_settlement_resume import _verified_endpoints

    _verified_endpoints(row.get('endpoints'), game_pid)
    inventory, opening = row['live_events']
    panels = inventory.get('inventory_events', ())
    if (inventory.get('event') != 'live_inventory_observe' or inventory.get('accepted') is not False
            or inventory.get('foreground_preparation_events') or inventory.get('hand_identity_inspection') is not None
            or [p.get('phase') for p in panels] != ['inventory_zone_observation', 'inventory_state_gate']
            or any(p.get('accepted') is not False or p.get('input_click_count') != 0 or p.get('inputs') for p in panels)
            or any(e.get('input_may_have_been_dispatched') for e in [inventory, opening, *panels])):
        raise ValueError('HIF notebook inventory did not stop before its first panel input')
    fields = (opening.get('initial_core') or {}).get('fields', ())
    for value, expected in zip(fields, (expected_live['turns'], expected_live['stamina'])):
        if value is not None and value != expected:
            raise ValueError('HIF notebook inventory observed a changed turn or stamina')
    if len(fields) > 3 and fields[3] not in (None, 0):
        raise ValueError('HIF notebook inventory already observed battle score progress')


def _stopped_before_opening_component(row):
    """Recognize a reader which published its failure before any input."""
    events = row.get('live_events') or ()
    if len(events) < 2:
        return False
    failed, opening = events[-2:]
    key = (opening.get('reason') or '').removeprefix('HIF opening component failed: ')
    return (key in _OPENING_PANEL_PHASES and key not in opening.get('components', {})
            and bool(opening.get('components')) and _unissued_opening_component(failed, key))


def _unissued_opening_component(failed, key):
    panels = failed.get('status_events') or ()
    return (failed.get('event') == 'live_status_observe' and failed.get('accepted') is False
            and failed.get('total_input_click_count') == 0 and not failed.get('foreground_preparation_events')
            and len(panels) == 1 and panels[0].get('phase') == _OPENING_PANEL_PHASES[key]
            and panels[0].get('accepted') is False and panels[0].get('input_click_count') == 0
            and not panels[0].get('inputs') and not panels[0].get('pages'))


def _resume_unapplied_opening(record, previous, proof, row, *, counted_return=False):
    """Account for completed reader navigation while preserving the verified prefix."""
    from .hif_opening_item_tail import (
        is_open_item_tail,
        resume_open_item_tail,
        is_drink_stop_after_item_return,
        resume_drinks_after_item_return,
    )
    from .hif_opening_status_resume import (
        is_open_status_prefix,
        verify_open_status_source,
        is_closed_items_after_status,
        resume_closed_items_after_status,
    )
    from .hif_native.native_opening_root import PARENT_FIELDS
    from .hif_native.native_opening_resume import verify_no_business_since_opening

    if is_open_status_prefix(row):
        return verify_open_status_source(record, [previous, row], parent=proof, counted_return=counted_return)
    if is_closed_items_after_status(row):
        return resume_closed_items_after_status(record, previous, proof, row, counted_return=counted_return)
    if is_drink_stop_after_item_return(row):
        return resume_drinks_after_item_return(record, previous, proof, row, counted_return=counted_return)
    _verify_unbound_component_context(record, row)
    verify_no_business_since_opening(row)
    _verify_retained_opening_source(row, previous, proof, counted_return=counted_return)
    opening = row['published_components'][-1]
    original = previous['published_components'][-1]
    if is_open_item_tail(row):
        return resume_open_item_tail(previous, proof, row)
    if _stopped_before_opening_component(row):
        return _resume_opening_component_prefix(proof, row, opening, previous)
    if (row.get('native_input_dispatches') != [] or row.get('input_count') != 0 or row.get('total_input_count') != 0
            or any(row.get(k) is not None for k in PARENT_FIELDS)
            or any(event.get('total_input_click_count', 0) for event in row['live_events'])):
        raise ValueError('HIF retained opening contains new or uncertain input')
    for key in ('native_input_preparation', 'native_upstream_input_preparation', 'native_navigation_preparation'):
        guard = row.get(key)
        if guard is not None and (guard.get('dispatch_state') != 'rejected_before_dispatch'
                or guard.get('input_may_have_been_dispatched') is not False):
            raise ValueError('HIF retained opening has an unresolved input preparation')
    probe = opening.get('extended_hand_identity') or {}
    inspection = opening.get('hand_identity_inspection') or {}
    if any(p.get('business_submit_count') or p.get('input_click_count') or p.get('accepted') for p in (probe, inspection)):
        raise ValueError('HIF opening inspection may have selected a card')
    for attempt in (*probe.get('inspections', ()), inspection):
        check = attempt.get('preflight') or {}
        if (attempt.get('selection_accepted') or check.get('click_dispatched', False) is not False
                or check.get('input_may_have_been_dispatched', False) is not False):
            raise ValueError('HIF opening inspection may have selected a card')
    if (any(proof.get('retained_components', {}).get(key) != component for key, component in opening.get('components', {}).items())
            or opening.get('retained_hand_identity_inspection') != (
                original.get('hand_identity_inspection') or original.get('retained_hand_identity_inspection'))):
        raise ValueError('HIF zero-input observation changed a retained component or title')
    for core in (opening.get('initial_core') or {}, probe.get('initial_core') or {}):
        if any(key in core and core[key] != proof['initial_core'][key] for key in ('fields', 'families')):
            raise ValueError('HIF zero-input observation changed the known public hand or resources')
    if (previous.get('input_count') == 0 and original.get('reason') == opening.get('reason')
            and original.get('components') == opening.get('components')):
        raise ValueError('HIF repeated the same zero-input opening failure without progress')
    core = dict(proof['initial_core'], **(opening.get('initial_core') or {}))
    return dict(proof, source_result=row['id'], initial_core=deepcopy(core))


def _resume_opening_component_prefix(proof, row, opening, previous):
    from types import SimpleNamespace

    from .opening_observer import OPENING_MODES, opening_component, opening_component_retryable
    from .hif_native.native_opening_root import PARENT_FIELDS
    from .hif_native.native_opening_mapping_resume import verify_opening_navigation

    original = previous['published_components'][-1]
    same_host = row.get('host_attempt') is not None and row['host_attempt'] == previous.get('host_attempt')
    previous_inputs = previous['total_input_count'] if same_host else 0
    if (row['live_events'][-1] != opening or row.get('total_input_count') != previous_inputs + row['input_count']
            or any(row.get(k) is not None for k in (*PARENT_FIELDS, 'native_input_preparation', 'native_upstream_input_preparation'))
            or opening.get('hand_identity_inspection') or opening.get('extended_hand_identity')
            or opening.get('retained_hand_identity_inspection') != (
                original.get('hand_identity_inspection') or original.get('retained_hand_identity_inspection'))):
        raise ValueError('HIF partial opening has activity outside its component readers')
    continuity = verify_unstarted_core(proof['initial_core'], opening['initial_core'])
    result = dict(proof)
    components = dict(proof.get('retained_components', {}))
    if 'pending_inventory_return' in result:
        components['cards'] = dict(result.pop('pending_inventory_return'), accepted=True,
            returned_unchanged=True, after_frame_ids=continuity['fresh_frame_ids'])
    keys = {key for key, _ in OPENING_MODES}
    if (not set(components) <= keys or opening.get('unstarted_retained_components') != sorted(components)
            or opening.get('unstarted_opening_continuity') != continuity
            or not set(continuity['fresh_frame_ids']) <= {frame['frame_id'] for frame in row['frames']}):
        raise ValueError('HIF partial opening changed its retained components or current hand')
    completed = []
    run = SimpleNamespace(**row['observed_live_context'])
    retried = None
    for report in row['live_events'][:-2]:
        if report.get('event') == 'live_inventory_observe':
            key = 'cards'
        else:
            panels = report.get('status_events') or ()
            phase = panels[-1].get('phase') if panels else None
            key = 'items' if phase == 'item_hud_inventory' else next(
                (key for key, value in _OPENING_PANEL_PHASES.items() if phase == value), None)
        if key not in keys or key in components:
            raise ValueError('HIF partial opening has an unknown or duplicate component')
        if retried != key and opening_component_retryable(report, key) and _unissued_opening_component(report, key):
            retried = key
            continue
        if len(components) >= len(opening['components']):
            raise ValueError('HIF partial opening has an extra component observation')
        components[key] = opening_component(report, key, run)
        completed.append(report)
    if components != opening['components']:
        raise ValueError('HIF partial opening components differ from their native observations')
    verify_opening_navigation(row, completed, opening)
    if 'items' in components:
        result.pop('item_tail_resume', None)
    if same_host and original.get('reason') == opening['reason'] and original.get('components') == components:
        raise ValueError('HIF repeated the same zero-input opening failure without progress')
    return dict(result, source_result=row['id'], initial_core=deepcopy(opening['initial_core']), retained_components=components)


def _verify_retained_opening_source(source, previous, proof, *, prior_reports=(), opening=None, counted_return=False):
    if opening is None:
        opening = source['live_events'][-1]
    if (source.get('opening_navigation_resume') != proof
            or source.get('published_components') != [*prior_reports, opening]
            or any(source.get(k) != previous.get(k) for k in ('observed_live_context', 'round_context', 'context_recovery'))
            or source['automatic_battle']['recoveries_used'] != previous['automatic_battle']['recoveries_used'] + int(counted_return)
            or source['automatic_battle'].get('opening_parent_window') != previous['automatic_battle'].get('opening_parent_window')):
        raise ValueError('HIF opening changed its original notebook or recovery budget')


def _verify_closed_notebook_inventory(record, rows):
    if len(rows) < 2:
        raise ValueError('HIF closed inventory lacks its immediate notebook predecessor')
    source = rows[-1]
    proof = verify_unstarted_inventory_source(record, source)
    notebook = _verify_unread_initial_inventory(record, rows[:-1])
    _verify_retained_opening_source(source, rows[-2], notebook)
    if proof.get('pending_inventory_return') or record['handoff'].get('source') == 'current_battle_page':
        return proof
    # A reconstructed full table needs no individual detail candidates.
    proof.pop('reuse_inventory_details', None)
    return dict(proof, recover_closed_inventory=True)


def verify_unstarted_status_source(record, source):
    """An unbound opening with a closed status reader has no card transaction."""
    from .inventory import initial_checkpoint_retry_evidence
    from .hif_native.native_inprocess_resume import _same_input_host
    from .hif_native.native_settlement_resume import _verified_endpoints
    from .hif_native.native_pending_components import _navigation_dispatch_matches

    request, metadata = _verify_unstarted_component_context(record, source)
    events = source['live_events']
    if [e.get('event') for e in events] != ['live_inventory_observe', 'live_status_observe', 'live_opening_state']:
        raise ValueError('HIF partial opening has an unaccounted observer')
    inventory, status, opening = events
    invs, statuses = inventory.get('inventory_events', ()), status.get('status_events', ())
    if (len(invs) != 2 or len(statuses) != 1 or inventory.get('accepted') is not True
            or status.get('accepted') is not False or opening.get('reason') != 'HIF opening component failed: status'
            or set(opening.get('components', {})) != {'cards'}):
        raise ValueError('HIF partial opening lacks its completed inventory prefix')
    nav, cards, failed = invs[0], invs[1], statuses[0]
    if _unissued_opening_component(status, 'status'):
        from types import SimpleNamespace

        from .opening_observer import opening_component
        from .hif_native.native_opening_mapping_resume import verify_opening_navigation

        run = SimpleNamespace(**source['observed_live_context'])
        complete = opening_component(inventory, 'cards', run)
        if complete != opening['components']['cards']:
            raise ValueError('HIF completed inventory differs from its returned component')
        verify_opening_navigation(source, [inventory], opening)
        core = opening['initial_core']
        proof = dict(boundary_kind='unbound_components', source_result=source['id'], input_authorized=False,
            expected_live=dict(turns=core['fields'][0], stamina=core['fields'][1]), initial_core=deepcopy(core),
            retained_components=dict(cards=deepcopy(complete)))
        if opening.get('hand_identity_inspection'):
            proof['restore_opening_title'] = True
        return proof
    if (nav.get('accepted') is not True or nav.get('live_state_unchanged') is not True
            or cards.get('accepted') is not True
            or json.dumps(dict(cards, returned_unchanged=True), sort_keys=True)
                != json.dumps(opening['components']['cards'], sort_keys=True)
            or failed.get('accepted') is not False or failed.get('read_failure') is not None
            or failed.get('reason') != 'inventory navigation Live visual family is unresolved'):
        raise ValueError('HIF partial opening is not a closed weak-identity status return')
    error = ValueError(failed['reason'])
    error.events = failed.get('failure_events', ())
    if not initial_checkpoint_retry_evidence(error, (None, None, None)):
        raise ValueError('HIF partial opening identity failure is not a supported fresh reobservation')
    core = opening.get('initial_core', {})
    fields, families, frames = (core.get(k, ()) for k in ('fields', 'families', 'frame_ids'))
    if (len(fields) != 6 or fields[0] != metadata.base_turn_count or fields[3] != 0
            or not families or not valid_opening_frame_ids(frames)):
        raise ValueError('HIF partial opening initial public state is incomplete')
    context = source['context_recovery']
    context_inputs, inv_inputs, status_inputs = context.get('inputs', ()), nav.get('inputs', ()), failed.get('inputs', ())
    dispatches = source.get('native_input_dispatches', ())
    if (not 1 <= len(context_inputs) <= 4 or context.get('input_click_count') != len(context_inputs)
            or nav.get('input_click_count') != len(inv_inputs) or failed.get('input_click_count') != len(status_inputs)
            or inventory.get('total_input_click_count') != len(inv_inputs)
            or status.get('total_input_click_count') != len(status_inputs)
            or opening.get('total_input_click_count') != len(inv_inputs)+len(status_inputs)
            or source.get('input_count') != len(dispatches) or source.get('total_input_count') != len(dispatches)
            or len(dispatches) != len(context_inputs)+len(inv_inputs)+len(status_inputs)
            or [v.get('phase') for v in status_inputs[:1]] != ['open']
            or [v.get('phase') for v in status_inputs[-1:]] != ['close']
            or any(v.get('phase') != 'scroll' for v in status_inputs[1:-1])
            or any(v.get('completed') is not True for v in status_inputs)):
        raise ValueError('HIF partial opening navigation accounting is incomplete')
    recorded = [(dict(v, phase='open', box=v.get('target')), {}) for v in context_inputs]
    recorded += [(v, dict(inventory_customization=True)) for v in inv_inputs]
    recorded += [(v, dict(status=True)) for v in status_inputs]
    for sent, (observed, flags) in zip(dispatches, recorded, strict=True):
        _verified_endpoints(sent.get('endpoints'), request.game_pid)
        _same_input_host(source['endpoints'], sent['endpoints'])
        if not _navigation_dispatch_matches(sent, observed, **flags):
            raise ValueError('HIF partial opening includes input outside the recorded readers')
    if any(v.get('succeeded') is not True for v in context_inputs):
        raise ValueError('HIF partial opening metadata navigation is uncertain')
    return dict(boundary_kind='unbound_components', source_result=source['id'], input_authorized=False,
        expected_live={k: fields[i] for k, i in (('stamina', 1), ('turns', 0))}, initial_core=deepcopy(core))


def verify_unstarted_inventory_source(record, source):
    """Keep completed navigation and identify any return observation still due."""
    if source.get('input_count') == 0:
        return verify_unstarted_inventory_preinput_source(record, source)
    from .hif_native.native_opening_mapping_resume import verify_opening_navigation

    _, metadata = _verify_unstarted_component_context(record, source)
    events = source['live_events']
    if ([event.get('event') for event in events] != ['live_inventory_observe', 'live_opening_state']
            or len(events[0].get('inventory_events', ())) != 2):
        raise ValueError('HIF partial inventory has unaccounted observers')
    inventory, opening = events
    failed, gate = inventory['inventory_events']
    inputs = failed.get('inputs', ())
    awaiting_return = failed.get('read_failure') is None and bool(failed.get('return_observations'))
    observations = failed.get('return_observations', ())
    reread_return = (bool(failed.get('read_failure')) and failed.get('live_state_unchanged') is None
        and failed.get('reason') == 'inventory navigation Live visual family is unresolved'
        and bool(observations) and all(row.get('accepted') is False
            and row.get('reason') == failed['reason'] for row in observations))
    if (opening.get('reason') != 'HIF opening component failed: cards'
            or opening.get('components') != {} or inventory.get('accepted') is not False
            or failed.get('accepted') is not False or not (awaiting_return or reread_return) and (
                failed.get('live_state_unchanged') is not True or not isinstance(failed.get('read_failure'), str)
                or not failed['read_failure'] or failed.get('reason') != failed['read_failure'])
            or gate.get('phase') != 'inventory_state_gate' or gate.get('accepted') is not False
            or gate.get('reason') != failed.get('reason') or gate.get('input_click_count') != 0
            or len(inputs) < 2 or inputs[0].get('phase') != 'open' or inputs[-1].get('phase') != 'close'
            or any(item.get('phase') not in ('scroll', 'customization_select') for item in inputs[1:-1])
            or source.get('total_input_count') != source.get('input_count')):
        raise ValueError('HIF partial inventory did not close its unchanged navigation')
    core = opening.get('initial_core', {})
    fields = core.get('fields', ())
    groups = [core.get('frame_ids', ()), failed.get('before_frame_ids', ())]
    if reread_return:
        groups.append(observations[-1].get('frame_ids', ()))
    elif not awaiting_return:
        groups.append(failed.get('after_frame_ids', ()))
    saved = {frame.get('frame_id') for frame in source.get('frames', ())}
    if (len(fields) != 6 or fields[0] != metadata.base_turn_count or fields[3] != 0
            or not core.get('families') or not all(map(valid_opening_frame_ids, groups))
            or len(set().union(*map(set, groups))) != sum(map(len, groups))
            or not set().union(*map(set, groups)) <= saved):
        raise ValueError('HIF partial inventory lacks independent unchanged opening frames')
    verify_opening_navigation(source, [inventory], opening, closed_inventory_failure=True,
                              pending_inventory_return=awaiting_return or reread_return)
    proof = dict(boundary_kind='unbound_components', source_result=source['id'], input_authorized=False,
        expected_live=dict(stamina=fields[1], turns=fields[0]), initial_core=deepcopy(core))
    if reread_return:
        proof['reread_inventory_return'] = True
    if failed.get('detail_observations') and (failed.get('live_state_unchanged') is True or reread_return):
        from .inventory_customization import partial_opening_details

        if partial_opening_details(source):
            proof['reuse_inventory_details'] = True
    if awaiting_return:
        from .hif_closed_inventory import inventory_awaiting_return
        ledger = record['ledger']
        proof['pending_inventory_return'] = inventory_awaiting_return(source, None if ledger is None else ledger['state'])
    return proof


def verify_unstarted_inventory_preinput_source(record, source):
    """Only a closed, zero-input weak pre-navigation frame may be reobserved."""
    from .inventory import fresh_identity_reobserve_allowed
    from .hif_native.native_settlement_resume import _verified_endpoints

    request, metadata = _verify_unstarted_component_context(record, source)
    _verified_endpoints(source.get('endpoints'), request.game_pid)
    events = source['live_events']
    if ([e.get('event') for e in events] != ['live_inventory_observe', 'live_opening_state']
            or any(source.get(k) != 0 for k in ('input_count', 'total_input_count'))
            or source.get('native_input_dispatches') != []):
        raise ValueError('HIF opening preinput source includes navigation or other observers')
    inventory, opening = events
    failures = inventory.get('inventory_events', ())
    if (inventory.get('accepted') is not False or opening.get('accepted') is not False
            or opening.get('reason') != 'HIF opening component failed: cards'
            or opening.get('components') != {} or len(failures) != 2
            or [e.get('phase') for e in failures] != ['inventory_zone_observation', 'inventory_state_gate']
            or any(e.get('accepted') is not False or e.get('input_click_count') != 0
                or e.get('inputs') for e in failures)
            or any(e.get('total_input_click_count') != 0 for e in events)):
        raise ValueError('HIF opening preinput source is not its unchanged zero-input inventory stop')
    failure = failures[0]
    if (failure.get('reason') != 'inventory navigation Live visual family is unresolved'
            or failures[1].get('reason') != failure['reason']):
        raise ValueError('HIF opening preinput source has another failure')
    core = opening.get('initial_core', {})
    error = ValueError(failure['reason'])
    error.events = failure.get('checkpoint_failure_events', ())
    if (not fresh_identity_reobserve_allowed(error, core)
            or core['fields'][0] != metadata.base_turn_count or core['fields'][3] != 0):
        raise ValueError('HIF opening preinput lacks an unchanged complete core and weak frame')
    before_frames = failure.get('before_frame_ids', ())
    saved_frames = {f.get('frame_id') for f in source.get('frames', ())}
    if (not valid_opening_frame_ids(before_frames)
            or not set((*before_frames, *core['frame_ids'], error.events[0]['frame_id'])) <= saved_frames
            or set(before_frames) & set(core['frame_ids'])
            or error.events[0]['frame_id'] in before_frames):
        raise ValueError('HIF opening preinput frame provenance is incomplete')
    resume = source.get('opening_navigation_resume', {})
    if (resume.get('boundary_kind') != 'unbound_components'
            or resume.get('source_result') != source['automatic_battle']['previous_record']
            or resume.get('input_authorized') is not False
            or opening.get('unstarted_opening_continuity') != verify_unstarted_core(resume.get('initial_core', {}), core)):
        raise ValueError('HIF opening preinput did not preserve its original public state')
    return dict(boundary_kind='unbound_components', source_result=source['id'], input_authorized=False,
        expected_live=dict(stamina=core['fields'][1], turns=core['fields'][0]), initial_core=deepcopy(core))


def verify_unstarted_items_source(record, source):
    """A closed partial item scan may resume only after every navigation is accounted."""
    from .hif_native.native_inprocess_resume import _same_input_host
    from .hif_native.native_settlement_resume import _verified_endpoints
    from .hif_native.native_pending_components import _navigation_dispatch_matches

    request, metadata = _verify_unstarted_component_context(record, source)
    events = source['live_events']
    if [e.get('event') for e in events] != [
            'live_inventory_observe', 'live_status_observe', 'live_status_observe', 'live_opening_state']:
        raise ValueError('HIF partial items opening has unaccounted observers')
    inventory, status, items, opening = events
    invs, statuses, item_events = inventory.get('inventory_events', ()), status.get('status_events', ()), items.get('status_events', ())
    if (len(invs) != 2 or len(statuses) != 1 or len(item_events) != 1
            or any(e.get('accepted') is not True for e in (inventory, status))
            or items.get('accepted') is not False or opening.get('accepted') is not False
            or opening.get('reason') != 'HIF opening component failed: items'
            or set(opening.get('components', {})) != {'cards', 'status'}):
        raise ValueError('HIF partial items opening lacks its completed component prefix')
    nav, cards, complete_status, failed = invs[0], invs[1], statuses[0], item_events[0]
    run = source['observed_live_context']
    expected = dict(cards=dict(cards, returned_unchanged=True),
        status=dict(complete_status, run_id=run['run_id'], revision=run['revision']))
    if (json.dumps(expected, sort_keys=True) != json.dumps(opening['components'], sort_keys=True)
            or nav.get('accepted') is not True or nav.get('live_state_unchanged') is not True
            or cards.get('accepted') is not True or complete_status.get('returned_unchanged') is not True
            or failed.get('phase') != 'item_description_inspection' or failed.get('accepted') is not False
            or failed.get('complete_list') is not False or failed.get('returned_unchanged') is not True
            or failed.get('popup_confirmed') is not True
            or failed.get('read_failure') != 'status panel endpoint rows are missing'
            or failed.get('reason') != failed['read_failure']
            or not failed.get('pages') or failed['pages'][-1]['observations'][-1].get('at_bottom') is not False):
        raise ValueError('HIF partial items source is not its closed incomplete endpoint scan')
    core = opening.get('initial_core', {})
    fields = core.get('fields', ())
    if (len(fields) != 6 or fields[0] != metadata.base_turn_count or fields[3] != 0
            or not core.get('families')):
        raise ValueError('HIF partial items opening core is incomplete')
    resume = source.get('opening_navigation_resume', {})
    if (resume.get('source_result') != source['automatic_battle']['previous_record']
            or resume.get('boundary_kind') != 'unbound_components' or resume.get('input_authorized') is not False
            or opening.get('unstarted_opening_continuity') != verify_unstarted_core(resume.get('initial_core', {}), core)):
        raise ValueError('HIF partial items opening lost original state continuity')
    frames = {f.get('frame_id') for f in source.get('frames', ())}
    for group in [core.get('frame_ids', ()), *(v.get(key, ()) for v in (nav, complete_status, failed)
            for key in ('before_frame_ids', 'after_frame_ids'))]:
        if not valid_opening_frame_ids(group) or not set(group) <= frames:
            raise ValueError('HIF partial items navigation lacks saved independent frame evidence')
    groups = [nav.get('inputs', ()), complete_status.get('inputs', ()), failed.get('inputs', ())]
    if (any(e.get('total_input_click_count') != len(inputs) for e, inputs in zip(events[:3], groups, strict=True))
            or any(e.get('input_click_count') != len(inputs) for e, inputs in zip((nav, complete_status, failed), groups, strict=True))
            or any([v.get('phase') for v in group[:1]] != ['open']
                or [v.get('phase') for v in group[-1:]] != ['close'] for group in groups)
            or any(v.get('completed') is not True for v in groups[1])
            or any(v.get('phase') != 'scroll' for v in groups[1][1:-1])
            or any(v.get('phase') not in ('scroll', 'scroll_top') for v in groups[2][1:-1])):
        raise ValueError('HIF partial items navigation sequence is incomplete')
    dispatches = source.get('native_input_dispatches', ())
    count = sum(map(len, groups))
    if (len(dispatches) != count or any(source.get(k) != count for k in ('input_count', 'total_input_count'))
            or opening.get('total_input_click_count') != count or failed.get('scroll_drag_pixels') not in (80, 120)):
        raise ValueError('HIF partial items navigation totals differ')
    index = 0
    for kind, group in enumerate(groups):
        for observed in group:
            sent = dispatches[index]; index += 1
            _verified_endpoints(sent.get('endpoints'), request.game_pid)
            _same_input_host(source['endpoints'], sent['endpoints'])
            if kind == 2 and observed['phase'] == 'scroll_top':
                matched = (sent.get('endpoints') == observed.get('endpoints') and sent.get('kind') == 'swipe'
                    and sent.get('start') == [560, 220] and sent.get('end') == [560, 610] and sent.get('duration') == 600)
            else:
                proof = dict(observed, scroll_distance=failed['scroll_drag_pixels']) if kind == 2 else observed
                matched = _navigation_dispatch_matches(sent, proof, status=kind > 0, inventory_customization=kind == 0)
                if kind == 2 and observed['phase'] == 'scroll': matched = matched and sent.get('duration') == 400
            if not matched:
                raise ValueError('HIF partial items includes input outside the original reader')
    return dict(boundary_kind='unbound_components', source_result=source['id'], input_authorized=False,
        expected_live=dict(stamina=fields[1], turns=fields[0]), initial_core=deepcopy(core))


def verify_unstarted_drinks_source(record, source):
    """Resume an untouched opening whose final detail cancel was never sent."""
    from types import SimpleNamespace

    from .opening_observer import opening_component
    from .hif_native.native_inprocess_resume import _same_input_host
    from .hif_native.native_settlement_resume import _verified_endpoints
    from .hif_native.native_pending_components import _navigation_dispatch_matches

    request, metadata = _verify_unstarted_component_context(record, source)
    events = source['live_events']
    if [e.get('event') for e in events] != [
            'live_inventory_observe', 'live_status_observe', 'live_status_observe',
            'live_status_observe', 'live_opening_state']:
        raise ValueError('HIF partial drinks opening has unaccounted observers')
    inventory, status, items, drinks, opening = events
    run = SimpleNamespace(**source['observed_live_context'])
    prefix = {key: opening_component(event, key, run)
        for key, event in zip(('cards', 'status', 'items'), events[:3], strict=True)}
    scans = drinks.get('status_events', ())
    if (opening.get('reason') != 'HIF opening component failed: drinks'
            or opening.get('components') != prefix or drinks.get('accepted') is not False
            or len(scans) != 1 or scans[0].get('phase') != 'complete_drink_inventory'
            or scans[0].get('accepted') is not False):
        raise ValueError('HIF partial drinks opening lacks its completed component prefix')
    readers = scans[0].get('events', ())
    if (len(readers) != 2 or readers[0].get('phase') != 'drink_slot_occupancy'
            or readers[0].get('accepted') is not True or readers[0].get('input_click_count') != 0
            or readers[1].get('phase') != 'occupied_drink_slots_inspection'):
        raise ValueError('HIF partial drinks opening lacks its slot observation')
    transactions = readers[1].get('transactions', ())
    if not transactions:
        raise ValueError('HIF partial drinks opening lacks its cancelled inspection')
    for index, transaction in enumerate(transactions):
        if (transaction.get('phase') != 'drink_description_inspection'
                or transaction.get('drinks_consumed') != 0
                or transaction.get('input_click_count') != 2
                or [v.get('phase') for v in transaction.get('inputs', ())] != ['open', 'cancel']
                or transaction.get('accepted') is not (index < len(transactions) - 1)
                or (index < len(transactions) - 1 and transaction.get('returned_unchanged') is not True)):
            raise ValueError('HIF partial drinks includes use or uncertain earlier navigation')
    preparation = source.get('native_navigation_preparation', {})
    cancel = transactions[-1]['inputs'][-1]
    if (preparation.get('dispatch_state') != 'rejected_before_dispatch'
            or preparation.get('input_may_have_been_dispatched') is not False
            or preparation.get('reason') != transactions[-1].get('reason')
            or preparation.get('endpoints') != cancel.get('endpoints')
            or not _navigation_dispatch_matches(preparation, dict(cancel, phase='close'))):
        raise ValueError('HIF partial drinks cancel may have been dispatched')

    recovery = source['context_recovery']
    groups = [inventory['inventory_events'][0]['inputs'], status['status_events'][0]['inputs'],
        items['status_events'][0]['inputs'], [v for t in transactions for v in t['inputs']]]
    count = sum(map(len, groups))
    from .hif_entry_round import verified_entry_receipt
    if (recovery.get('accepted') is not True
            or not (recovery.get('returned_to_live') is True or verified_entry_receipt(source, recovery))
            or opening.get('total_input_click_count') != count
            or any(e.get('total_input_click_count') != len(group)
                for e, group in zip(events[:4], groups, strict=True))):
        raise ValueError('HIF partial drinks navigation totals differ')
    recorded = []
    for item in recovery.get('inputs', ()):
        if (item.get('succeeded') is not True or item.get('uncertain') is True
                or item.get('phase') not in ('live', 'menu', 'notebook', 'identity')):
            raise ValueError('HIF partial drinks has unresolved metadata navigation')
        recorded.append((dict(item, phase='open', box=item.get('target')), {}))
    for index, group in enumerate(groups):
        for item in group:
            if index == 2:
                item = dict(item, scroll_distance=items['status_events'][0].get('scroll_drag_pixels', 0))
            recorded.append((dict(item, phase='close') if item['phase'] == 'cancel' else item,
                dict(status=index in (1, 2), inventory_customization=index == 0)))
    dispatches = source.get('native_input_dispatches', ())
    if (len(dispatches) != len(recorded) - 1
            or any(source.get(key) != len(dispatches) for key in ('input_count', 'total_input_count'))):
        raise ValueError('HIF partial drinks has unmatched actual inputs')
    for sent, (observed, options) in zip(dispatches, recorded[:-1], strict=True):
        _verified_endpoints(sent.get('endpoints'), request.game_pid)
        _same_input_host(source['endpoints'], sent['endpoints'])
        if sent.get('endpoints') != observed.get('endpoints') or not _navigation_dispatch_matches(sent, observed, **options):
            raise ValueError('HIF partial drinks includes input outside its original readers')
    core = opening.get('initial_core', {})
    fields, frames = core.get('fields', ()), core.get('frame_ids', ())
    if (len(fields) != 6 or fields[0] != metadata.base_turn_count or fields[3] != 0
            or not core.get('families') or not valid_opening_frame_ids(frames)
            or not set(frames) <= {f.get('frame_id') for f in source.get('frames', ())}):
        raise ValueError('HIF partial drinks opening core is incomplete')
    return dict(boundary_kind='unbound_components', source_result=source['id'], input_authorized=False,
        expected_live=dict(stamina=fields[1], turns=fields[0]), initial_core=deepcopy(core),
        restore_drink_panel=True)


def verify_unstarted_turns_source(record, source):
    """Reuse completed opening components after a closed turn-table failure."""
    from types import SimpleNamespace

    from .opening_observer import opening_component
    from .hif_native.native_opening_mapping_resume import verify_opening_navigation

    _, metadata = _verify_unstarted_component_context(record, source)
    events = source['live_events']
    if [event.get('event') for event in events] != [
            'live_inventory_observe', *['live_status_observe']*4, 'live_opening_state']:
        raise ValueError('partial turns opening has unaccounted observers')
    opening, failed = events[-1], events[-2]['status_events'][0]
    run = SimpleNamespace(**source['observed_live_context'])
    components = {key: opening_component(event, key, run)
        for key, event in zip(('cards', 'status', 'items', 'drinks'), events[:4], strict=True)}
    if (opening.get('reason') != 'HIF opening component failed: turns'
            or opening.get('components') != components or failed.get('accepted') is not False
            or not failed.get('read_failure') or failed.get('returned_unchanged') is not True):
        raise ValueError('partial turns opening lacks its unchanged completed prefix')
    verify_opening_navigation(source, events[:-1], opening, closed_turn_failure=True)
    core = opening.get('initial_core', {})
    fields, frames = core.get('fields', ()), core.get('frame_ids', ())
    if (len(fields) != 6 or fields[0] != metadata.base_turn_count or fields[3] != 0
            or not core.get('families') or not valid_opening_frame_ids(frames)):
        raise ValueError('partial turns opening lacks its original complete public state')
    return dict(boundary_kind='unbound_components', source_result=source['id'], input_authorized=False,
        expected_live=dict(stamina=fields[1], turns=fields[0]), initial_core=deepcopy(core),
        retained_components=deepcopy(components))


def verify_unstarted_core(original, current):
    """Only a new complete unchanged checkpoint can continue an unbound opening."""
    frames = current.get('frame_ids', ())
    if (not valid_opening_frame_ids(frames)
            or any(json.dumps(current.get(key), sort_keys=True) != json.dumps(original.get(key), sort_keys=True)
                   for key in ('fields', 'families'))
            or set(frames) & set(original.get('frame_ids', ()))):
        raise ValueError('HIF resumed opening differs from its original untouched public state')
    return dict(accepted=True, source_frame_ids=original['frame_ids'], fresh_frame_ids=frames)


def _business_boundary(row):
    if row.get('transaction_closed') is True:
        if row.get('status') == 'RULE_MAINTAINED_NATIVE_TRANSACTION' and not row.get('terminal_settlement'):
            return 'settled'
        raise ValueError('HIF navigation cannot reopen a terminal or unverified closure')
    counts = [event[key] for event in row.get('live_events', ())
        for key in ('execution_submit_count', 'end_turn_submit_count') if key in event]
    if any(type(value) is not int or value not in (0, 1) for value in counts) or sum(counts) > 1:
        raise ValueError('HIF navigation business dispatch count is ambiguous')
    if sum(counts) == 1:
        if row.get('native_pending_source'):
            raise ValueError('HIF navigation has conflicting original and new business dispatches')
        return 'submitted_pending'
    if any(event.get('drinks_consumed', 0) for event in row.get('live_events', ())):
        raise ValueError('HIF navigation cannot skip an unresolved drink transaction')
    if row.get('gray_selection_journal') or any(
            event.get('execution_selection_click_count') == 1 for event in row.get('live_events', ())):
        return 'selected'
    return None


def _business_source(rows, record):
    """Follow only an explicit zero-business-input link in this unique chain."""
    by_id = {row['id']: row for row in rows}
    current = rows[-1]
    if record['handoff'].get('live_context') is None:
        for row in rows:
            verify_unbound_opening_source(record, row)
        return current, 'unbound_opening'
    live = record['handoff']['live_context']
    handoff = dict(record['handoff'], route_run_id=record['run_id'])
    seen = set()
    while current['id'] not in seen:
        seen.add(current['id'])
        _binding(current, live, handoff)
        if current.get('endpoints', {}).get('game', {}).get('pid') != record['ownership']['game_pid']:
            raise ValueError('HIF navigation source game changed')
        boundary = _business_boundary(current)
        if boundary is not None:
            return current, boundary
        # A route-navigation click is separate from a possible business click.
        # An unconfirmed business dispatch can never be skipped to an ancestor.
        preparation = current.get('native_input_preparation')
        if preparation is not None and (not isinstance(preparation, dict)
                or preparation.get('dispatch_state') != 'rejected_before_dispatch'
                or preparation.get('input_may_have_been_dispatched') is not False):
            raise ValueError('HIF navigation source has an uncertain business dispatch')
        parents = {current[key] for key in ('native_pending_source', 'native_selected_source',
            'native_inprocess_source', 'native_rule_resume_source') if current.get(key)}
        if len(parents) != 1:
            raise ValueError('HIF navigation source lacks one explicit business predecessor')
        parent = by_id.get(parents.pop())
        if parent is None or parent['automatic_battle']['sequence'] >= current['automatic_battle']['sequence']:
            raise ValueError('HIF navigation source leaves its original automatic chain')
        for skipped in rows[parent['automatic_battle']['sequence']:current['automatic_battle']['sequence'] - 1]:
            preparation = skipped.get('native_input_preparation')
            if (_business_boundary(skipped) is not None or preparation is not None and (
                    not isinstance(preparation, dict)
                    or preparation.get('dispatch_state') != 'rejected_before_dispatch'
                    or preparation.get('input_may_have_been_dispatched') is not False)):
                raise ValueError('HIF navigation source skips a later business transaction')
        current = parent
    raise ValueError('HIF navigation source chain is cyclic')


def load_battle_navigation_source(runtime, request, shared_record, battle_result=None):
    """Return read-only original business evidence, never permission to retry.

    The route scheduler must separately observe a supported interruption and
    use its existing bounded navigation producer. Merely loading this source
    does not make an ordinary STOP recoverable.
    """
    bound = _bound_request(request, shared_record)
    rows = load_automatic_chain(Path(runtime), bound)
    latest = rows[-1]
    if latest.get('status') != 'STOP' or latest.get('transaction_closed') is True:
        raise ValueError('HIF navigation requires the current formal battle STOP')
    if battle_result is not None and (not isinstance(battle_result, dict) or battle_result.get('status') != 'STOP'
            or battle_result.get('returned_to_nonbattle') is not False
            or battle_result.get('last_record') != latest['id']
            or battle_result.get('handoff_id') != request.handoff_id
            or battle_result.get('policy_profile') != request.policy_profile
            or (battle_result.get('request') != json.loads(json.dumps(asdict(request)))
                and battle_result.get('request') != asdict(request))):
        raise ValueError('HIF navigation requires the current formal battle STOP')
    control = deepcopy(latest['automatic_battle'])
    require_budget(control)
    source, boundary = _business_source(rows, shared_record)
    if battle_result is None:
        saved = (shared_record['handoff'].get('nonbattle_context') or {}).get('date_recovery')
        if saved is None and boundary == 'unbound_opening':
            # No business state was bound before this STOP. The route must
            # still observe a real supported interruption before any input.
            pass
        elif (not isinstance(saved, dict) or saved.get('source_result') != source['id']
                or saved.get('handoff_id') != request.handoff_id
                or type(saved.get('episode')) is not int or saved['episode'] < 1):
            raise ValueError('HIF cold navigation lacks its original persisted source')
    return dict(stopped_record=latest['id'], source_report=deepcopy(source), boundary_kind=boundary,
        automatic_battle=control, input_authorized=False)


def find_completed_navigation_resume(runtime, session, *, maximum_actions, timeout_seconds,
                                     parent_deadline=None, engineering_resume=None):
    """Consume one persisted completion while retaining the original limits."""
    record = session.record
    saved = (record.get('handoff') or {}).get('nonbattle_context', {}).get('date_recovery')
    if saved is None:
        return None
    if not isinstance(saved, dict):
        raise ValueError('HIF saved navigation state is malformed')
    if saved.get('completed') is not True:
        return None
    rows = load_automatic_chain(Path(runtime), identity(session))
    latest = rows[-1]
    if latest.get('status') != 'STOP' or latest.get('transaction_closed') is True:
        return None
    # A previous successful navigation stays in history, but is not another
    # recovery ticket for a later transaction or for the same episode.
    if any(row.get('saved_navigation_resume', {}).get('source_result') == saved.get('source_result')
            and row['saved_navigation_resume'].get('episode') == saved.get('episode') for row in rows):
        return None
    source, boundary = _business_source(rows, record)
    if saved.get('source_result') != source['id']:
        return None
    from route.date_recovery import verify_completed_date_return

    receipt = verify_completed_date_return(record, source)
    control = deepcopy(latest['automatic_battle'])
    if (control['maximum_actions'], control['timeout_seconds']) != (maximum_actions, timeout_seconds):
        raise ValueError('HIF navigation cannot replace the original battle budget')
    constrain_parent_deadline(control, parent_deadline)
    from .hif_battle_engineering_window import (
        OPENING_RENEWAL_MIN_SECONDS,
        window_limits,
        can_renew_native_window,
    )
    deadline, _ = window_limits(control)
    now = time.time()
    # Navigation completion is not a new grant. A bound, already classified
    # business source may use the same fixed issuer/checkpoint callback as the
    # ordinary restart path, before checking its otherwise expired old limits.
    if (callable(engineering_resume)
            and now >= control.get('last_checked_at', control['started_at'])
            and now + OPENING_RENEWAL_MIN_SECONDS >= deadline
            and can_renew_native_window(control, latest)):
        control = engineering_resume(latest, control)
    require_budget(control)
    control['recoveries_used'] += 1
    proof = dict(receipt, stopped_record=latest['id'], boundary_kind=boundary)
    return control, latest, source['id'], proof


def verify_navigation_checkpoint(proof, checkpoint):
    """Navigation completion cannot turn an unresolved dispatch into a play."""
    kind = proof['boundary_kind']
    receipt = checkpoint.get('receipt') or {}
    valid = (receipt.get('source') == proof['source_result']
        and receipt.get('handoff_id') == proof['handoff_id']
        and receipt.get('boundary_kind') == kind)
    if kind == 'submitted_pending':
        valid = valid and checkpoint.get('pending') is True and checkpoint.get('expected') is not None
    elif kind == 'selected':
        valid = valid and checkpoint.get('selected') is True and not checkpoint.get('pending')
    elif kind == 'settled':
        valid = valid and not checkpoint.get('selected') and not checkpoint.get('pending')
    else:
        valid = False
    if not valid:
        raise ValueError('HIF native checkpoint differs from the original navigation boundary')
