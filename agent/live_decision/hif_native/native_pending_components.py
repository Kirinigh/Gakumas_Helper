"""Reuse completed component observations without accepting a failed settlement."""
import copy
import json
import math

from route.engine import has_single_frame_id as _frames

from ..opening_observer import direct_item_observation

COMPONENT_NAMES = ('cards', 'status', 'items')


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _core_frames(core):
    return isinstance(core, dict) and _frames(core.get('frame_ids'))


def _current_panel_pages(component):
    return all(_frames(page.get('frame_ids')) and len(page.get('observations', ())) == 1
               for page in component.get('pages', ()))


def _observed_target(inspection, slot):
    boxes = inspection.get('candidate_boxes', ())
    if (not _frames(inspection.get('frame_ids')) or len(boxes) != 1
            or not 0 <= slot < len(boxes[0])):
        raise ValueError('pending component source lacks a current single-frame target')
    return boxes[0][slot]


def load_pending_components_source(runtime, name, module, pending_name, bundle, expected, baseline, *, _trail=()):
    """Bind a completed reader batch to an independently loaded pending action.

    The host supplies bundle/expected/baseline from load_pending_source. These
    observations never authorize input or accept the old failed settlement.
    """
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import load_result, _verified_endpoints, decoded_dispatched_action

    if (not isinstance(name, str) or not name or name in _trail or len(_trail) >= 8):
        raise ValueError('pending component source cycle or chain limit exceeded')
    row = load_result(runtime, name)
    pending = load_result(runtime, pending_name)
    run = module.HifLiveRunContext(**pending['observed_live_context'])
    if name == pending_name:
        from live_decision.high_score.contracts import ActionKind
        action, _ = decoded_dispatched_action(row, bundle)
        if action.kind is ActionKind.PLAY_CARD:
            events = [event for event in row.get('live_events', ())
                      if event.get('event') == 'live_committed_state']
            if (len(events) == 1 and events[0].get('reason') == 'HIF opening component failed: status'
                    and 'core' not in events[0]):
                return _partial_submitted_card_source(row, run, bundle, expected, baseline, module)
            return _complete_submitted_card_source(row, run, bundle, expected, baseline, module)
        return _partial_submitted_drink_source(row, run, bundle, expected, baseline, module)
    if (row.get('native_pending_source') != pending_name
            or row.get('native_resume_source') or row.get('native_inprocess_source')
            or row.get('native_selected_source')
            or row.get('live_return') is not False
            or row.get('execution_live_context') != pending['observed_live_context']
            or any(row.get('observed_live_context', {}).get(key) != getattr(run, key)
                   for key in ('idol', 'plan', 'max_stamina'))
            or row['endpoints']['game']['pid'] != pending['endpoints']['game']['pid']
            or row['endpoints']['game']['client_size'] != pending['endpoints']['game']['client_size']
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
            or baseline.get('run_id') != run.run_id or baseline.get('revision') != run.revision
            or baseline.get('accepted') is not True or not baseline.get('checks')
            or not all(value is True for value in baseline['checks'].values())):
        raise ValueError('pending components source differs from the verified action or execution epoch')
    action, _ = decoded_dispatched_action(pending, bundle)
    possibilities = expected if isinstance(expected, tuple) else (expected,)
    if not possibilities or any(state.terminal for state in possibilities):
        raise ValueError('pending components require non-terminal expected outcomes')
    for state in possibilities:
        state.validate()
    if any(event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
           or event.get('drinks_consumed', 0) or event.get('executed') is True
           for event in row.get('live_events', ())):
        raise ValueError('pending components source contains a new business submission')
    dispatches = row.get('native_input_dispatches', ())
    if not dispatches or row.get('input_count') != len(dispatches):
        raise ValueError('pending components source has incomplete navigation accounting')
    for entry in dispatches:
        if entry.get('kind') not in (None, 'swipe'):
            raise ValueError('pending components source contains unsupported input')
        _verified_endpoints(entry.get('endpoints'), row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], entry['endpoints'])
    events = [event for event in row.get('live_events', ())
              if event.get('event') == 'live_committed_state']
    if len(events) != 1 or not _same(events[0], row.get('settlement')):
        raise ValueError('pending components source lacks one unchanged complete reader event')
    event = events[0]
    if row.get('native_pending_components_source'):
        return _complete_resumed_status_source(runtime, row, module, pending_name,
            bundle, expected, baseline, run, action, (*_trail, name))
    if event.get('reason') == 'HIF opening component failed: status' and 'core' not in event:
        return _partial_cards_source(row, event, run, possibilities, module)
    if event.get('reason') == 'HIF opening component failed: items' and 'core' not in event:
        return _partial_status_source(row, event, run, bundle, expected, baseline, action, module)
    if (event.get('accepted') is not False or event.get('strategy_state_bound') is not False
            or event.get('reason') != 'hif_settlement_outcome_not_unique'
            or event.get('matched_outcome_count') != 0
            or event.get('run_id') != run.run_id or event.get('revision') != run.revision
            or not any(checks.get(missing) is False
                       and all(value is True for key, value in checks.items() if key != missing)
                       for checks in event.get('candidate_checks', ())
                       for missing in ('named_card_schedules', 'named_drink_schedules'))):
        raise ValueError('pending components source is not the complete unresolved schedule observation')
    before, core = event.get('initial_core'), event.get('core')
    if (not _core_frames(before) or not _core_frames(core)
            or set(before['frame_ids']) & set(core['frame_ids'])
            or any(not _same(before.get(key), core.get(key)) for key in ('fields', 'families'))):
        raise ValueError('pending components source changed across its complete reader batch')
    all_components = event.get('components', {})
    if set(all_components) != {'cards', 'status', 'items', 'drinks', 'turns'}:
        raise ValueError('pending components source lacks the complete five-component batch')
    reusable_names = COMPONENT_NAMES
    if 'items' in event.get('maintained_components', ()):
        from live_decision.high_score.maintenance import maintained_components

        maintained = maintained_components(bundle, expected, core=before, baseline=baseline, action=action)
        if (set(maintained) != {'items', 'drinks', 'turns'}
                or set(event['maintained_components']) != set(maintained)
                or any(not _same(all_components[key], value) for key, value in maintained.items())):
            raise ValueError('pending maintained items differ from the original action proof')
        reusable_names = ('cards', 'status')
    for key, component in all_components.items():
        if (component.get('accepted') is not True or component.get('returned_unchanged') is not True
                or component.get('run_id') != run.run_id or component.get('revision') != run.revision):
            raise ValueError('pending component is incomplete or belongs to another epoch: ' + key)
        if key in reusable_names:
            same_frame_item = key == 'items' and direct_item_observation(component)
            if (not _frames(component.get('before_frame_ids')) or not _frames(component.get('after_frame_ids'))
                    or key in ('status', 'items') and not _current_panel_pages(component)
                    or not same_frame_item and set(component['before_frame_ids']) & set(component['after_frame_ids'])
                    or component.get('observation_source') in ('retained_verified_component', 'retained_unchanged_component')):
                raise ValueError('pending reusable component lacks independent fresh reader frames: ' + key)
    if any(checks.get('named_drink_schedules') is False
           and all(value is True for key, value in checks.items() if key != 'named_drink_schedules')
           for checks in event.get('candidate_checks', ())):
        # 1692's final drink activation leaves a one-turn UI row. Reuse this
        # complete batch only after the corrected observation lifecycle proves
        # one full outcome; admitting its old failed label is not sufficient.
        from live_decision.high_score.opening import resolve_hif_committed_observation

        resolved, proof = resolve_hif_committed_observation(bundle, possibilities,
            core=core, components=all_components, baseline_components=baseline['components'],
            committed_action=action)
        if resolved is None or proof.get('accepted') is not True:
            raise ValueError('pending drink schedule components do not verify one complete outcome')
    components = {key: copy.deepcopy(all_components[key]) for key in reusable_names}
    audit = dict(event='native_pending_components_resume', source=name, pending_source=pending_name,
        run_id=run.run_id, revision=run.revision, game_pid=row['endpoints']['game']['pid'],
        component_names=list(reusable_names), source_initial_frame_ids=list(before['frame_ids']),
        source_core_frame_ids=list(core['frame_ids']), navigation_input_count=len(dispatches),
        business_submit_count=0, expected_candidate_count=len(possibilities),
        components_reusable=True, source_settlement_accepted=False, submission_authorized=False)
    return dict(components=components, core=copy.deepcopy(core), audit=audit)


def _navigation_dispatch_matches(actual, recorded, *, status=False, inventory_customization=False,
                                 history=False):
    if actual.get('endpoints') != recorded.get('endpoints'):
        return False
    phase = recorded.get('phase')
    if phase == 'turn_tail_scroll' and history and not status and not inventory_customization:
        swipe = recorded.get('swipe', ())
        return (len(swipe) == 5 and all(type(value) is int for value in swipe)
            and actual.get('kind') == 'swipe' and actual.get('start') == list(swipe[:2])
            and actual.get('end') == list(swipe[2:4])
            and type(actual.get('duration')) is int and actual['duration'] == swipe[4])
    if phase == 'scroll':
        start = [560, 610] if status else recorded.get('begin')
        end = [560, 610-recorded.get('scroll_distance', 0)] if status else recorded.get('end')
        return actual.get('kind') == 'swipe' and actual.get('start') == start and actual.get('end') == end
    phases = ('open', 'close', 'customization_select') if inventory_customization and not status else ('open', 'close')
    if phase not in phases or actual.get('kind') == 'swipe':
        return False
    box = recorded.get('box', ())
    return (len(box) == 4 and all(type(value) is int for value in box) and min(box[2:]) > 0
            and (actual.get('x'), actual.get('y')) == (box[0]+box[2]//2, box[1]+box[3]//2))


def _complete_resumed_status_source(runtime, row, module, pending_name, bundle, expected,
                                    baseline, run, action, trail):
    """Extend verified cards with one complete new status reader after a rule fix."""
    from live_decision.high_score.opening import resolve_hif_committed_observation
    from live_decision.high_score.maintenance import maintained_components

    from .native_single_plus import recorded_single_plus_estimate
    from .native_opening_root import verify_native_round_context
    from .native_settlement_resume import load_result, decoded_dispatched_action

    previous = load_pending_components_source(runtime, row['native_pending_components_source'],
        module, pending_name, bundle, expected, baseline, _trail=trail)
    reports = row.get('live_events', ())
    event = row.get('settlement', {})
    if (set(previous['components']) != {'cards'}
            or [report.get('event') for report in reports] != ['live_status_observe', 'live_committed_state']
            or not _same(reports[-1], event) or event.get('accepted') is not False
            or event.get('strategy_state_bound') is not False
            or event.get('reason') != 'hif_settlement_outcome_not_unique'
            or event.get('matched_outcome_count') != 0
            or (event.get('run_id'), event.get('revision')) != (run.run_id, run.revision)
            or set(event.get('components', {})) != {'cards', 'status', 'items', 'drinks', 'turns'}):
        raise ValueError('resumed status source is not one complete unresolved reader extension')
    verify_native_round_context(row, module, required=True)
    verify_pending_component_completion(row, event, previous)
    initial, core, components = event['initial_core'], event['core'], event['components']
    if any(component.get('accepted') is not True or component.get('returned_unchanged') is not True
           or (component.get('run_id'), component.get('revision')) != (run.run_id, run.revision)
           for component in components.values()):
        raise ValueError('resumed status source has an incomplete or foreign component')
    maintained = maintained_components(bundle, expected, core=initial, baseline=baseline, action=action)
    if (set(maintained) != {'items', 'drinks', 'turns'}
            or event.get('maintained_components') != sorted(maintained)
            or any(not _same(components[key], value) for key, value in maintained.items())):
        raise ValueError('resumed status retained components differ from their original action proof')
    status_report, status = reports[0], components['status']
    status_events = status_report.get('status_events', ())
    if (status_report.get('accepted') is not True or len(status_events) != 1
            or status.get('phase') != 'status_inspection' or status.get('read_failure')
            or status.get('observation_source') is not None
            or not _same(dict(status_events[0], run_id=run.run_id, revision=run.revision), status)):
        raise ValueError('resumed status lacks one independent complete status report')
    old = previous['audit']
    frame_groups = [old['source_initial_frame_ids'], old['source_core_frame_ids'],
                    old.get('source_reader_frame_ids', []), initial['frame_ids'], core['frame_ids']]
    for frames in (status.get('before_frame_ids'), status.get('after_frame_ids')):
        if not _frames(frames) or any(set(frames) & set(group) for group in frame_groups):
            raise ValueError('resumed status reader endpoints are incomplete or reused')
        frame_groups.append(frames)
    _verify_complete_status_pages(status, frame_groups)
    context = row.get('context_recovery', {})
    if (context.get('accepted') is not True or context.get('returned_to_live') is not True
            or context.get('context') != row.get('observed_live_context')
            or context.get('game_pid') != row['endpoints']['game']['pid']):
        raise ValueError('resumed status lacks complete same-process context recovery')
    context_inputs, status_inputs = context.get('inputs', ()), status.get('inputs', ())
    counts = ((context, 'input_click_count', len(context_inputs)),
              (status, 'input_click_count', len(status_inputs)),
              (status_report, 'total_input_click_count', len(status_inputs)),
              (event, 'total_input_click_count', len(status_inputs)))
    if (not context_inputs or any(type(source.get(key)) is not int or source[key] != count
            for source, key, count in counts)
            or [entry.get('phase') for entry in status_inputs]
                != ['open', *(['scroll']*(len(status['pages'])-1)), 'close']
            or row['input_count'] != len(context_inputs)+len(status_inputs)):
        raise ValueError('resumed status navigation accounting or operation order differs')
    recorded = [(dict(item, phase='open', box=item.get('target')), False) for item in context_inputs]
    recorded += [(item, True) for item in status_inputs]
    if not all(_navigation_dispatch_matches(actual, proof, status=is_status)
               for actual, (proof, is_status) in zip(row['native_input_dispatches'], recorded, strict=True)):
        raise ValueError('resumed status inputs differ from its exact recorded readers')
    possibilities = expected if isinstance(expected, tuple) else (expected,)
    pending = load_result(runtime, pending_name)
    original_action, submitted = decoded_dispatched_action(pending, bundle)
    if original_action != action:
        raise ValueError('resumed status differs from its independently dispatched action')
    single_plus_estimate = recorded_single_plus_estimate(bundle, action, submitted, source_row=pending)
    resolved, proof = resolve_hif_committed_observation(bundle, possibilities, core=core,
        components=components, baseline_components=baseline['components'], committed_action=action,
        single_plus_estimate=single_plus_estimate)
    if resolved is None or proof.get('accepted') is not True or proof.get('matched_outcome_count') != 1:
        raise ValueError('resumed status does not verify one complete outcome under current rules')
    audit = dict(event='native_pending_components_resume', source=row['id'], pending_source=pending_name,
        run_id=run.run_id, revision=run.revision, game_pid=row['endpoints']['game']['pid'],
        source_client_size=row['endpoints']['game']['client_size'], component_names=['cards', 'status'],
        source_initial_frame_ids=list(initial['frame_ids']), source_core_frame_ids=list(core['frame_ids']),
        source_final_core_available=True,
        source_reader_frame_ids=sorted({frame for group in frame_groups for frame in group}),
        source_return_frame_ids=list(status['after_frame_ids']),
        navigation_input_count=row['input_count'], business_submit_count=0,
        expected_candidate_count=len(possibilities), components_reusable=True,
        source_settlement_accepted=False, submission_authorized=False)
    return dict(components={key:copy.deepcopy(components[key]) for key in ('cards','status')},
                core=copy.deepcopy(core), audit=audit)


def _verify_complete_status_pages(status, frame_groups):
    """Rebuild the complete status list using the observations recorded for each page."""
    from dataclasses import asdict

    from live_decision.status_panel import (
        StatusPanelRow,
        StatusPanelPage,
        status_panel_consensus,
        join_status_panel_pages,
        status_row_confidence_valid,
    )

    pages = status.get('pages', ())
    if not isinstance(pages, (list, tuple)) or not 1 <= len(pages) <= 12:
        raise ValueError('partial status source lacks the complete bounded page list')
    joined_pages = []
    for group in pages:
        frames = group.get('frame_ids')
        if not _frames(frames) or any(set(frames) & set(previous) for previous in frame_groups):
            raise ValueError('partial status pages reuse or omit observation frames')
        frame_groups.append(frames)
        try:
            observations = tuple(StatusPanelPage(
                rows=tuple(StatusPanelRow(**dict(item, box=tuple(item['box']))) for item in page['rows']),
                thumb=tuple(page['thumb']), at_top=page['at_top'], at_bottom=page['at_bottom'],
                partial_row_count=page['partial_row_count']) for page in group['observations'])
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('partial status page observations are malformed') from error
        if len(observations) != len(frames) or any(not status_row_confidence_valid(item)
                for page in observations for item in page.rows):
            raise ValueError('partial status page confidence or frame provenance is unresolved')
        joined_pages.append(status_panel_consensus(observations))
    rows = join_status_panel_pages(joined_pages)
    if not _same([asdict(item) for item in rows], status.get('rows')):
        raise ValueError('partial status rows differ from independently joined complete pages')


def _partial_status_source(row, event, run, bundle, expected, baseline, action, module):
    """Keep a complete status reader before an input-free initial item failure.

    Maintained cards prove the prefix's dependency, but are rebuilt from the
    next fresh core. Only status is reused; the old initial core and reader
    return certificate remain subject to the existing completion verifier.
    """
    from live_decision.high_score.contracts import ActionKind
    from live_decision.high_score.maintenance import maintained_components

    from .native_opening_root import verify_native_round_context

    verify_native_round_context(row, module, required=True)
    reports = row.get('live_events', ())
    initial, components = event.get('initial_core'), event.get('components', {})
    if (action.kind is not ActionKind.PLAY_CARD
            or [report.get('event') for report in reports] != [
                'live_status_observe', 'live_status_observe', 'live_committed_state']
            or reports[-1] is not event or event.get('accepted') is not False
            or event.get('executed') is not False or not _core_frames(initial)
            or set(components) != {'cards', 'status'}):
        raise ValueError('partial status source is not the cards-status-before-items prefix')
    maintained = maintained_components(bundle, expected, core=initial, baseline=baseline, action=action)
    if (set(maintained) != {'cards', 'drinks', 'turns'}
            or event.get('maintained_components') != sorted(maintained)
            or not _same(components['cards'], maintained['cards'])
            or components['cards'].get('observation_source') != 'maintained_card_ledger'):
        raise ValueError('partial status maintained cards differ from their original proof')
    status_report, failed_report = reports[:2]
    status_events, failed_events = status_report.get('status_events', ()), failed_report.get('status_events', ())
    status = components['status']
    if (status_report.get('accepted') is not True or len(status_events) != 1
            or status.get('phase') != 'status_inspection' or status.get('accepted') is not True
            or status.get('returned_unchanged') is not True or status.get('read_failure')
            or status.get('observation_source') is not None
            or (status.get('run_id'), status.get('revision')) != (run.run_id, run.revision)
            or not _same(dict(status_events[0], run_id=run.run_id, revision=run.revision), status)
            or failed_report.get('accepted') is not False or len(failed_events) != 1):
        raise ValueError('partial status source lacks its complete independent status reader')
    failure = failed_events[0]
    if (failure.get('phase') != 'item_description_inspection' or failure.get('accepted') is not False
            or failure.get('reason') != 'HIF ordinary card row is unstable'
            or failure.get('strategy_state_bound') is not False or failure.get('complete_list') is not False
            or failure.get('calibration_only') is not False
            or failure.get('inputs') != [] or failure.get('pages') != []
            or 'after_frame_ids' in failure or failure.get('returned_unchanged')
            or failure.get('failure_events')):
        raise ValueError('partial status follow-up was not an input-free initial item failure')
    frame_groups = [initial['frame_ids']]
    for frames in (status.get('before_frame_ids'), status.get('after_frame_ids'), failure.get('before_frame_ids')):
        if not _frames(frames) or any(set(frames) & set(group) for group in frame_groups):
            raise ValueError('partial status reader endpoints are incomplete or reused')
        frame_groups.append(frames)
    _verify_complete_status_pages(status, frame_groups)
    pages = status['pages']
    context = row.get('context_recovery', {})
    if (context.get('accepted') is not True or context.get('returned_to_live') is not True
            or context.get('context') != row.get('observed_live_context')
            or context.get('game_pid') != row['endpoints']['game']['pid']):
        raise ValueError('partial status lacks its fresh identity and process recovery')
    context_inputs, status_inputs = context.get('inputs', ()), status.get('inputs', ())
    counts = ((context, 'input_click_count', len(context_inputs)),
        (status, 'input_click_count', len(status_inputs)),
        (status_report, 'total_input_click_count', len(status_inputs)),
        (event, 'total_input_click_count', len(status_inputs)),
        (failure, 'input_click_count', 0), (failed_report, 'total_input_click_count', 0))
    if (not context_inputs or any(type(source.get(key)) is not int or source[key] != count
            for source, key, count in counts)
            or [item.get('phase') for item in status_inputs] != ['open', *(['scroll']*(len(pages)-1)), 'close']
            or row['input_count'] != len(context_inputs)+len(status_inputs)):
        raise ValueError('partial status navigation accounting or operation order differs')
    recorded = [(dict(item, phase='open', box=item.get('target')), False) for item in context_inputs]
    recorded += [(item, True) for item in status_inputs]
    if not all(_navigation_dispatch_matches(actual, proof, status=is_status)
            for actual, (proof, is_status) in zip(row['native_input_dispatches'], recorded, strict=True)):
        raise ValueError('partial status inputs differ from its exact context and status readers')
    possibilities = expected if isinstance(expected, tuple) else (expected,)
    audit = dict(event='native_pending_components_resume', source=row['id'], pending_source=row['native_pending_source'],
        run_id=run.run_id, revision=run.revision, game_pid=row['endpoints']['game']['pid'],
        source_client_size=row['endpoints']['game']['client_size'], component_names=['status'],
        maintained_source_component_names=['cards'], source_initial_frame_ids=list(initial['frame_ids']),
        source_core_frame_ids=list(initial['frame_ids']), source_final_core_available=False,
        continuity_reference='original_initial_core_with_reader_return_certificates',
        source_reader_frame_ids=[frame for group in frame_groups[1:] for frame in group],
        source_return_frame_ids=list(status['after_frame_ids']), source_items_preinput_accepted=False,
        source_items_before_frame_ids=list(failure['before_frame_ids']),
        navigation_input_count=row['input_count'], business_submit_count=0,
        expected_candidate_count=len(possibilities), components_reusable=True,
        source_settlement_accepted=False, submission_authorized=False)
    return dict(components={'status': copy.deepcopy(status)}, core=copy.deepcopy(initial), audit=audit)


def _complete_submitted_card_source(row, run, bundle, expected, baseline, module):
    """Reuse complete readers after one independently reconstructed card."""
    from live_decision.high_score.opening import verify_hif_committed_observation
    from live_decision.high_score.contracts import ActionKind
    from live_decision.high_score.maintenance import maintained_components

    from .native_opening_root import verify_native_round_context
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints, decoded_dispatched_action

    verify_native_round_context(row, module, required=True)
    action, submitted = decoded_dispatched_action(row, bundle)
    possibilities = expected if isinstance(expected, tuple) else (expected,)
    inprocess = bool(row.get('native_inprocess_source'))
    selection_count = int(inprocess)
    business_inputs = 1 + selection_count
    if (action.kind is not ActionKind.PLAY_CARD or row.get('live_return') is not False
            or bool(row.get('native_selected_source')) == inprocess
            or (inprocess and row.get('native_resume_source')) or row.get('native_pending_source')
            or row.get('native_pending_components_source')
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
            or baseline.get('accepted') is not True
            or (baseline.get('run_id'), baseline.get('revision')) != (run.run_id, run.revision)
            or not possibilities or any(state.terminal for state in possibilities)):
        raise ValueError('submitted card component source has no verified pending selection')
    reports = row.get('live_events', ())
    observed_items = len(reports) == 5
    if [event.get('event') for event in reports] != [
            'live_inventory_observe', 'live_status_observe',
            *(['live_status_observe'] if observed_items else []), 'live_committed_state', 'live_decision_observe']:
        raise ValueError('submitted card components contain an unaccounted operation')
    inventory, status_report = reports[:2]
    event, report = reports[-2:]
    reusable_names = ('cards', 'status', 'items') if observed_items else ('cards', 'status')
    maintained_names = {'drinks', 'turns'} if observed_items else {'items', 'drinks', 'turns'}
    before, core = event.get('initial_core'), event.get('core')
    if (report is not submitted or report.get('executed') is not False
            or report.get('execution_submit_count') != 1
            or report.get('execution_selection_click_count') != selection_count
            or event.get('accepted') is not False or event.get('strategy_state_bound') is not False
            or event.get('reason') != 'hif_settlement_outcome_not_unique'
            or event.get('matched_outcome_count') != 0
            or (event.get('run_id'), event.get('revision')) != (run.run_id, run.revision)
            or not any((checks.get('named_card_schedules') is False
                and all(value is True for key, value in checks.items() if key != 'named_card_schedules'))
                or (checks.get('scheduled_draws') is False and checks.get('additional_status_semantics') is False
                    and all(value is True for key, value in checks.items()
                        if key not in {'scheduled_draws', 'additional_status_semantics'}))
                for checks in event.get('candidate_checks', ()))
            or not _core_frames(before) or not _core_frames(core)
            or set(before['frame_ids']) & set(core['frame_ids'])
            or any(not _same(before.get(key), core.get(key)) for key in ('fields', 'families'))):
        raise ValueError('submitted card components lack unchanged complete schedule-mismatch evidence')
    all_components = event.get('components', {})
    if set(all_components) != {'cards', 'status', 'items', 'drinks', 'turns'}:
        raise ValueError('submitted card source lacks all five completed components')
    maintained = maintained_components(bundle, expected, core=before, baseline=baseline, action=action)
    if (set(maintained) != maintained_names
            or sorted(event.get('maintained_components', ())) != sorted(maintained)
            or any(not _same(all_components[key], maintained[key]) for key in maintained)):
        raise ValueError('submitted card maintained components differ from their original proof')
    inventory_events, status_events = inventory.get('inventory_events', ()), status_report.get('status_events', ())
    if (inventory.get('accepted') is not True or len(inventory_events) != 2
            or status_report.get('accepted') is not True or len(status_events) != 1):
        raise ValueError('submitted card source lacks its two complete native readers')
    navigation, permanent = inventory_events
    status = status_events[0]
    if (navigation.get('accepted') is not True or navigation.get('live_state_unchanged') is not True
            or navigation.get('phase') != 'inventory_zone_observation'
            or permanent.get('phase') != 'inventory_permanent_card_state'
            or not _same(dict(permanent, returned_unchanged=True), all_components['cards'])
            or status.get('phase') != 'status_inspection'
            or not _same(dict(status, run_id=run.run_id, revision=run.revision), all_components['status'])):
        raise ValueError('submitted card reader data differs from saved components')
    item_inputs = ()
    if observed_items:
        item_report = reports[2]
        item_events = item_report.get('status_events', ())
        if item_report.get('accepted') is not True or len(item_events) != 1:
            raise ValueError('submitted card lacks its complete item reader')
        items = item_events[0]
        item_inputs = items.get('inputs', ())
        if (not _same(dict(items, run_id=run.run_id, revision=run.revision), all_components['items'])
                or items.get('input_click_count') != len(item_inputs)
                or item_report.get('total_input_click_count') != len(item_inputs)):
            raise ValueError('submitted card item reader evidence does not match')
        if (not direct_item_observation(items) and (items.get('phase') != 'item_description_inspection'
                or len(item_inputs) < 2 or item_inputs[0].get('phase') != 'open'
                or item_inputs[-1].get('phase') != 'close'
                or any(proof.get('phase') not in {'scroll', 'scroll_top'} for proof in item_inputs[1:-1]))):
            raise ValueError('submitted card item reader or navigation is incomplete')
    groups = [before['frame_ids'], core['frame_ids']]
    for key in reusable_names:
        component = all_components[key]
        if (component.get('accepted') is not True or component.get('returned_unchanged') is not True
                or key in ('status', 'items') and not _current_panel_pages(component)
                or (component.get('run_id'), component.get('revision')) != (run.run_id, run.revision)):
            raise ValueError('submitted card reader is incomplete or unbound')
        fields = ('before_frame_ids',) if key == 'items' and direct_item_observation(component) else (
            'before_frame_ids', 'after_frame_ids')
        for field in fields:
            frames = component.get(field)
            if not _frames(frames) or any(set(frames) & set(group) for group in groups):
                raise ValueError('submitted card reader frame provenance is reused or incomplete')
            groups.append(frames)
    inventory_inputs, status_inputs = navigation.get('inputs', ()), status.get('inputs', ())
    if (navigation.get('input_click_count') != len(inventory_inputs)
            or inventory.get('total_input_click_count') != len(inventory_inputs)
            or status.get('input_click_count') != len(status_inputs)
            or status_report.get('total_input_click_count') != len(status_inputs)
            or event.get('total_input_click_count') != len(inventory_inputs)+len(status_inputs)+len(item_inputs)
            or [entry.get('phase') for entry in status_inputs[:1]] != ['open']
            or [entry.get('phase') for entry in status_inputs[-1:]] != ['close']
            or any(entry.get('phase') != 'scroll' for entry in status_inputs[1:-1])):
        raise ValueError('submitted card component navigation accounting differs')
    actual = row.get('native_input_dispatches', ())
    if (row.get('input_count') != len(actual)
            or len(actual) != business_inputs+len(inventory_inputs)+len(status_inputs)+len(item_inputs)):
        raise ValueError('submitted card source includes an unaccounted actual input')
    for entry in actual:
        _verified_endpoints(entry.get('endpoints'), row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], entry['endpoints'])
    if inprocess:
        from .native_card_evidence import _fresh, dispatched_card_refreshes
        refreshes = dispatched_card_refreshes(report, row=row, bundle=bundle)
        stages = [entry for entry in report.get('execution_selection_events', ())
                  if entry.get('click_dispatched') is True]
        if ([entry.get('phase') for entry in stages] != ['execution_selection', 'execution_submit']
                or any(entry.get('attempted_slot') != action.source_slot for entry in stages)):
            raise ValueError('inprocess component source lacks one selection followed by one submit')
        for refresh, entry in zip(refreshes, actual[:2], strict=True):
            _fresh(refresh, action.source_slot)
            box = _observed_target(refresh, action.source_slot)
            if (len(box) != 4 or any(type(value) is not int for value in box) or min(box[2:]) <= 0
                    or entry.get('kind') == 'swipe'
                    or (entry.get('x'), entry.get('y')) != (box[0]+box[2]//2, box[1]+box[3]//2)):
                raise ValueError('inprocess component inputs differ from original A/B targets')
    if (not all(_navigation_dispatch_matches(entry, proof) for entry, proof in
                zip(actual[business_inputs:business_inputs+len(inventory_inputs)], inventory_inputs, strict=True))
            or not all(_navigation_dispatch_matches(entry, proof, status=True) for entry, proof in
                zip(actual[business_inputs+len(inventory_inputs):business_inputs+len(inventory_inputs)+len(status_inputs)],
                    status_inputs, strict=True))):
        raise ValueError('submitted card input order differs from the exact completed readers')
    for entry, original in zip(actual[business_inputs+len(inventory_inputs)+len(status_inputs):], item_inputs, strict=True):
        proof = original
        if proof['phase'] in ('scroll', 'scroll_top'):
            # Existing item_panel gestures; its box denotes Close, not the swipe target.
            swipe = (560, 610, 560, 530, 400) if proof['phase'] == 'scroll' else (560, 220, 560, 610, 600)
            if entry.get('duration') != swipe[4]:
                raise ValueError('submitted card item swipe duration changed')
            proof = dict(proof, phase='scroll', begin=list(swipe[:2]), end=list(swipe[2:4]))
        if not _navigation_dispatch_matches(entry, proof):
            raise ValueError('submitted card item input differs from its recorded navigation')
    matches = [state for state in possibilities if verify_hif_committed_observation(bundle, state,
        core=core, components=all_components, baseline_components=baseline['components'],
        committed_action=action)['accepted']]
    if len(matches) != 1:
        raise ValueError('submitted card source has no unique independently recomputed settlement')
    audit = dict(event='native_pending_components_resume', source=row['id'], pending_source=row['id'],
        run_id=run.run_id, revision=run.revision, game_pid=row['endpoints']['game']['pid'],
        source_client_size=row['endpoints']['game']['client_size'], component_names=list(reusable_names),
        maintained_source_component_names=sorted(maintained_names),
        source_initial_frame_ids=list(before['frame_ids']), source_core_frame_ids=list(core['frame_ids']),
        source_final_core_available=True, source_reader_frame_ids=[frame for group in groups[2:] for frame in group],
        business_submit_count=1, navigation_input_count=len(actual)-business_inputs,
        expected_candidate_count=len(possibilities), components_reusable=True,
        source_settlement_accepted=False, submission_authorized=False)
    if inprocess:
        audit['selection_input_count'] = selection_count
    return dict(components={key: copy.deepcopy(all_components[key]) for key in reusable_names},
        core=copy.deepcopy(core), audit=audit)


def _partial_submitted_drink_source(row, run, bundle, expected, baseline, module):
    """Resume a fresh status reader after one drink, recomputing maintained data.

    The cards/items in this prefix were maintained, not newly scanned. Their
    exact production derivation is checked here; the next observer derives
    them again from its fresh core. Only the genuine status observation is
    returned as a reusable fresh component. No final core is invented.
    """
    from live_decision.high_score.contracts import ActionKind
    from live_decision.high_score.maintenance import maintained_components

    from .native_opening_root import verify_native_round_context
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints, decoded_dispatched_action

    verify_native_round_context(row, module, required=True)
    action, joint = decoded_dispatched_action(row, bundle)
    possibilities = expected if isinstance(expected, tuple) else (expected,)
    if (action.kind is not ActionKind.USE_DRINK or row.get('live_return') is not False
            or row.get('native_pending_source') or row.get('native_pending_components_source')
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
            or len(possibilities) != 1 or possibilities[0].terminal
            or baseline.get('accepted') is not True
            or (baseline.get('run_id'), baseline.get('revision')) != (run.run_id, run.revision)):
        raise ValueError('partial drink components lack the same independently reconstructed action')
    possibilities[0].validate()
    reports = row.get('live_events', ())
    if [event.get('event') for event in reports] != [
            'live_status_observe', 'live_status_observe', 'live_status_observe',
            'live_committed_state', 'live_joint_action']:
        raise ValueError('partial drink components contain an unaccounted operation')
    status_report, *_, event, recorded_joint = reports
    initial, components = event.get('initial_core', {}), event.get('components', {})
    if (recorded_joint is not joint or joint.get('accepted') is not False
            or not _same(joint.get('settlement'), event) or 'core' in event
            or event.get('accepted') is not False or not _core_frames(initial)
            or event.get('reason') != 'HIF opening component failed: drinks'
            or set(components) != {'cards', 'status', 'items'}
            or event.get('maintained_components') != ['cards', 'items', 'turns']
            or any(event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
                   or event.get('executed') is True for event in reports[:-1])):
        raise ValueError('partial drink components are not the saved status-before-drinks prefix')
    maintained = maintained_components(bundle, expected, core=initial, baseline=baseline, action=action)
    if (set(maintained) != {'cards', 'items', 'turns'}
            or any(not _same(components[key], maintained[key]) for key in ('cards', 'items'))):
        raise ValueError('partial drink maintained components differ from their original proof')
    status = components['status']
    status_events = status_report.get('status_events', ())
    if (status_report.get('accepted') is not True or len(status_events) != 1
            or not _same(dict(status_events[0], run_id=run.run_id, revision=run.revision), status)
            or status.get('phase') != 'status_inspection' or status.get('accepted') is not True
            or not _current_panel_pages(status)
            or status.get('returned_unchanged') is not True):
        raise ValueError('partial drink status is not its completed fresh reader')
    groups = [initial['frame_ids']]
    for key in ('before_frame_ids', 'after_frame_ids'):
        frames = status.get(key)
        if not _frames(frames) or any(set(frames) & set(group) for group in groups):
            raise ValueError('partial drink status endpoints are incomplete or reused')
        groups.append(frames)
    for failure in reports[1:3]:
        failed_events = failure.get('status_events', ())
        if (failure.get('accepted') is not False or failure.get('total_input_click_count') != 0
                or len(failed_events) != 1 or failed_events[0].get('phase') != 'complete_drink_inventory'
                or failed_events[0].get('accepted') is not False):
            raise ValueError('partial drink follow-up was not an input-free failed inventory')
        stack = list(failed_events)
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if (item.get('input_click_count', 0) or item.get('drinks_consumed', 0)
                        or item.get('execution_submit_count', 0) or item.get('inputs')
                        or item.get('use_submitted') is True):
                    raise ValueError('partial drink follow-up includes an additional input')
                stack.extend(item.values())
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
    context, history = row.get('context_recovery', {}), row.get('resume_history', {})
    if (context.get('accepted') is not True or context.get('returned_to_live') is not True
            or context.get('context') != row.get('observed_live_context')
            or context.get('game_pid') != row['endpoints']['game']['pid']
            or history.get('accepted') is not True or history.get('returned_unchanged') is not True):
        raise ValueError('partial drink lacks its verified process and pre-action history')
    drink = joint['transactions'][0]
    context_inputs, history_inputs = context.get('inputs', ()), history.get('inputs', ())
    drink_inputs, status_inputs = drink.get('inputs', ()), status.get('inputs', ())
    if (context.get('input_click_count') != len(context_inputs)
            or history.get('input_click_count') != len(history_inputs)
            or drink.get('input_click_count') != len(drink_inputs)
            or [item.get('phase') for item in drink_inputs] != ['open', 'use']
            or status.get('input_click_count') != len(status_inputs)
            or status_report.get('total_input_click_count') != len(status_inputs)
            or event.get('total_input_click_count') != len(status_inputs)
            or [item.get('phase') for item in status_inputs[:1]] != ['open']
            or [item.get('phase') for item in status_inputs[-1:]] != ['close']
            or any(item.get('phase') != 'scroll' for item in status_inputs[1:-1])):
        raise ValueError('partial drink reader accounting differs')
    recorded = [(dict(item, phase='open', box=item.get('target')), False, False) for item in context_inputs]
    from .native_history_navigation import history_navigation_proofs
    recorded += [(item, False, True) for item in history_navigation_proofs(history)]
    recorded += [(dict(item, phase='open'), False, False) for item in drink_inputs]
    recorded += [(item, True, False) for item in status_inputs]
    actual = row.get('native_input_dispatches', ())
    if not actual or len(actual) != len(recorded) or row.get('input_count') != len(actual):
        raise ValueError('partial drink has an unaccounted input after the submitted action')
    for entry, (proof, is_status, is_history) in zip(actual, recorded, strict=True):
        _verified_endpoints(entry.get('endpoints'), row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], entry['endpoints'])
        if not _navigation_dispatch_matches(entry, proof, status=is_status, history=is_history):
            raise ValueError('partial drink actual inputs differ from its exact source readers')
    audit = dict(event='native_pending_components_resume', source=row['id'], pending_source=row['id'],
        run_id=run.run_id, revision=run.revision, game_pid=row['endpoints']['game']['pid'],
        source_client_size=row['endpoints']['game']['client_size'], component_names=['status'],
        maintained_source_component_names=['cards', 'items'],
        source_initial_frame_ids=list(initial['frame_ids']), source_core_frame_ids=list(initial['frame_ids']),
        source_final_core_available=False, continuity_reference='original_initial_core_with_reader_return_certificates',
        source_reader_frame_ids=[frame for group in groups[1:] for frame in group],
        source_return_frame_ids=list(status['after_frame_ids']), navigation_input_count=len(actual)-1,
        business_submit_count=1, expected_candidate_count=1,
        components_reusable=True, source_settlement_accepted=False, submission_authorized=False)
    return dict(components={'status': copy.deepcopy(status)}, core=copy.deepcopy(initial), audit=audit)


def _partial_submitted_card_source(row, run, bundle, expected, baseline, module):
    """Reuse cards after one audited submission and an unchanged status failure."""
    from live_decision.high_score.contracts import ActionKind

    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints, decoded_dispatched_action

    action, submitted = decoded_dispatched_action(row, bundle)
    possibilities = expected if isinstance(expected, tuple) else (expected,)
    selected_resume = bool(row.get('native_selected_source'))
    normal_resume = bool(row.get('native_resume_source'))
    selection_count = int(normal_resume)
    if (action.kind is not ActionKind.PLAY_CARD or row.get('live_return') is not False
            or selected_resume == normal_resume or row.get('native_inprocess_source')
            or row.get('native_pending_source')
            or row.get('native_pending_components_source')
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
            or baseline.get('accepted') is not True
            or (baseline.get('run_id'), baseline.get('revision')) != (run.run_id, run.revision)
            or not possibilities or any(state.terminal for state in possibilities)
            or submitted.get('executed') is not False
            or type(submitted.get('execution_submit_count')) is not int
            or submitted['execution_submit_count'] != 1
            or type(submitted.get('execution_selection_click_count')) is not int
            or submitted['execution_selection_click_count'] != selection_count
            or type(submitted.get('identity_probe_click_count')) is not int
            or submitted['identity_probe_click_count'] != 0
            or submitted.get('total_input_click_count') != 1+selection_count
            or action.card_instance_id != bundle.state.deck.hand[action.source_slot].instance_id):
        raise ValueError('partial submitted card lacks one bound native selection submission')
    actual = row.get('native_input_dispatches', ())
    guard = row.get('native_input_preparation', {})
    floor, checked, deadline = (guard.get(key) for key in
        ('authorization_floor', 'authorization_checked_at', 'input_deadline'))
    refreshes = submitted.get('authoritative_refresh_events', ())
    if (not actual or row.get('input_count') != len(actual) or not refreshes
            or refreshes[-1].get('accepted') is not True
            or guard.get('dispatch_state') != 'dispatch_attempted'
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in (floor, checked, deadline))
            or not floor <= checked < deadline or not 0 < deadline-floor <= 1):
        raise ValueError('partial submitted card lacks its original audited dispatch guard')
    submit_index = 0
    if normal_resume:
        context, history = row.get('context_recovery', {}), row.get('resume_history', {})
        context_inputs, history_inputs = context.get('inputs', ()), history.get('inputs', ())
        if (context.get('accepted') is not True or context.get('returned_to_live') is not True
                or context.get('context') != row.get('observed_live_context')
                or context.get('game_pid') != row['endpoints']['game']['pid']
                or history.get('accepted') is not True or history.get('returned_unchanged') is not True
                or not context_inputs or not history_inputs
                or context.get('input_click_count') != len(context_inputs)
                or history.get('input_click_count') != len(history_inputs)):
            raise ValueError('partial submitted card lacks its verified pre-action navigation')
        recorded = [dict(item, phase='open', box=item.get('target')) for item in context_inputs]
        from .native_history_navigation import history_navigation_proofs
        recorded += history_navigation_proofs(history)
        selection_index = len(recorded)
        if len(actual) <= selection_index+1 or not all(
                _navigation_dispatch_matches(entry, proof, history=True)
                for entry, proof in zip(actual[:selection_index], recorded, strict=True)):
            raise ValueError('partial submitted card pre-action inputs differ from their source readers')
        selections = [entry for entry in submitted.get('execution_selection_events', ())
                      if entry.get('phase') == 'execution_selection' and entry.get('click_dispatched') is True]
        first = refreshes[0]
        if (len(selections) != 1 or selections[0].get('accepted') is not True
                or selections[0].get('attempted_slot') != action.source_slot
                or first.get('phase') != 'authoritative_refresh_a' or first.get('accepted') is not True
                or first.get('select_consensus', {}).get('accepted') is not True
                or first['select_consensus'].get('selected_slot') is not None):
            raise ValueError('partial submitted card lacks its original unselected selection proof')
        selection_box = _observed_target(first, action.source_slot)
        selection_target = (selection_box[0]+selection_box[2]//2, selection_box[1]+selection_box[3]//2)
        selection = actual[selection_index]
        if (selection.get('kind') == 'swipe'
                or (selection.get('x'), selection.get('y')) != selection_target
                or sum(entry.get('kind') != 'swipe' and (entry.get('x'), entry.get('y')) == selection_target
                       for entry in actual) != 1):
            raise ValueError('partial submitted card lacks one exact audited selection')
        submit_index = selection_index+1
    box = _observed_target(refreshes[-1], action.source_slot)
    target = (box[0]+box[2]//2, box[1]+box[3]//2)
    if ((guard.get('x'), guard.get('y')) != target or actual[submit_index].get('kind') == 'swipe'
            or (actual[submit_index].get('x'), actual[submit_index].get('y')) != target
            or not _same(actual[submit_index].get('endpoints'), guard.get('endpoints'))
            or sum(entry.get('kind') != 'swipe' and (entry.get('x'), entry.get('y')) == target
                   for entry in actual) != 1):
        raise ValueError('partial submitted card lacks one unique ordered audited submission')
    for entry in actual:
        _verified_endpoints(entry.get('endpoints'), row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], entry['endpoints'])
    for state in possibilities:
        state.validate()
    events = [event for event in row.get('live_events', ()) if event.get('event') == 'live_committed_state']
    if len(events) != 1 or events[0].get('reason') != 'HIF opening component failed: status' or 'core' in events[0]:
        raise ValueError('partial submitted card has no original cards-before-status prefix')
    return _partial_cards_source(row, events[0], run, possibilities, module,
        submitted=submitted, submitted_prefix_count=submit_index+1)


def _partial_cards_source(row, event, run, possibilities, module, *, submitted=None, submitted_prefix_count=None):
    """Keep only a finished cards reader before a closed, unchanged status failure.

    No final core was saved in this producer failure. The original initial
    core remains the continuity reference; reader return certificates prove
    navigation only. The next complete batch must supply new initial/final
    cores and match this reference before it can bind any state.
    """
    from .native_opening_root import verify_native_round_context

    verify_native_round_context(row, module, required=True)
    initial = event.get('initial_core', {})
    cards = event.get('components', {}).get('cards', {})
    if (event.get('accepted') is not False or not _core_frames(initial)
            or set(event.get('components', {})) != {'cards'}
            or cards.get('accepted') is not True or cards.get('returned_unchanged') is not True
            or (cards.get('run_id'), cards.get('revision')) != (run.run_id, run.revision)
            or cards.get('phase') != 'inventory_permanent_card_state'
            or cards.get('temporary_effects_bound') is not False
            or cards.get('strategy_state_bound') is not False):
        raise ValueError('partial pending source lacks one complete same-epoch cards observation')
    reports = row.get('live_events', ())
    report_names = ['live_inventory_observe', 'live_status_observe', 'live_committed_state']
    if submitted is not None:
        report_names.append('live_decision_observe')
    if ([item.get('event') for item in reports] != report_names
            or (submitted is not None and reports[-1] is not submitted)):
        raise ValueError('partial pending source contains an unaccounted observation or operation')
    inventory, status = reports[:2]
    inventory_events = inventory.get('inventory_events', ())
    status_events = status.get('status_events', ())
    if (inventory.get('accepted') is not True or len(inventory_events) != 2
            or status.get('accepted') is not False or len(status_events) != 1):
        raise ValueError('partial pending source does not have a finished inventory then failed status reader')
    navigation, permanent = inventory_events
    failure = status_events[0]
    status_preinput_stop = (failure.get('reason') == 'inventory navigation Live visual family is unresolved'
        and failure.get('input_click_count') == 0 and failure.get('inputs') == []
        and failure.get('pages') == [] and bool(failure.get('failure_events'))
        and all(item.get('phase') == 'inventory_live_identity' for item in failure['failure_events']))
    if (navigation.get('phase') != 'inventory_zone_observation' or navigation.get('accepted') is not True
            or navigation.get('live_state_unchanged') is not True
            or not _same(dict(permanent, returned_unchanged=True), cards)
            or failure.get('phase') != 'status_inspection' or failure.get('accepted') is not False
            or (not status_preinput_stop
                and (failure.get('returned_unchanged') is not True or not failure.get('read_failure')))):
        raise ValueError('partial pending source navigation did not return unchanged')
    frame_groups = [initial['frame_ids']]
    failed_status_frames = None
    for proof in (navigation, failure):
        keys = ('before_frame_ids',) if proof is failure and status_preinput_stop else ('before_frame_ids', 'after_frame_ids')
        for key in keys:
            frames = proof.get(key)
            if proof is failure and status_preinput_stop and frames is None:
                failures = failure['failure_events']
                groups = [item.get('frame_ids') for item in failures]
                if (not all(_frames(group) for group in groups)
                        or any(not _same(group, groups[0]) for group in groups)
                        or any(item.get('frame_id') not in group for item, group in zip(failures, groups))):
                    raise ValueError('partial status failure lacks one complete failed observation batch')
                frames = failed_status_frames = groups[0]
            if not _frames(frames) or any(set(frames) & set(previous) for previous in frame_groups):
                raise ValueError('partial pending source lacks independent reader endpoint frames')
            frame_groups.append(frames)
    returns = navigation.get('return_observations', [])
    returned = returns[-1] if returns else {}
    complete_return = (returned.get('accepted') is True
        and _same(returned.get('frame_ids'), cards.get('after_frame_ids')))
    if 'observation_attempt' in returned or 'elapsed_ms' in returned:
        attempt, elapsed = returned.get('observation_attempt'), returned.get('elapsed_ms')
        complete_return = (complete_return and type(attempt) is int and 1 <= attempt <= 8
            and type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed < 10000)
    else:
        complete_return = complete_return and _same(returned, dict(accepted=True, frame_ids=cards.get('after_frame_ids')))
    if (cards.get('before_frame_ids') != navigation['before_frame_ids']
            or cards.get('after_frame_ids') != navigation['after_frame_ids']
            or not complete_return
            or navigation.get('complete_zone_counts') != cards.get('zone_counts')):
        raise ValueError('partial cards observation differs from its completed navigation evidence')
    composition = navigation.get('composition', {})
    rows = cards.get('cards', ())
    if set(composition) != set(cards['zone_counts']) or any(
            [card['card_id'] for card in rows if card['zone'] == zone] != composition[zone]
            or [card['index'] for card in rows if card['zone'] == zone] != list(range(len(composition[zone])))
            or len(composition[zone]) != count for zone, count in cards['zone_counts'].items()):
        raise ValueError('partial cards observation lost a zone, card, or slot')
    context = row.get('context_recovery', {})
    if (context.get('accepted') is not True or context.get('returned_to_live') is not True
            or context.get('context') != row.get('observed_live_context')
            or context.get('game_pid') != row['endpoints']['game']['pid']):
        raise ValueError('partial pending source lacks its fresh identity and process recovery')
    # Selected restoration carries historical identity provenance. Its old
    # navigation count is not input performed by this submitted-card process.
    prefix_count = submitted_prefix_count if submitted is not None else context.get('input_click_count')
    context_prefix_count = prefix_count
    if submitted is None and row.get('inspected_art_reference'):
        from live_decision.catalog import SkillCardFeatureCatalog
        from live_decision.hif_inspected_art import restore_reference

        receipt = restore_reference(row, run, SkillCardFeatureCatalog.load(module.CATALOG_PATH))
        preflight = row.get('pending_inspection_preflight', {})
        confirmation = row.get('pending_inspection_confirmation', {})
        # restore_reference verified the complete bounded reset/retry list.
        # Only its first event dispatched the single cancellation.
        reset = row['pending_inspection_reset'][0]
        actual_inspection = row['native_input_dispatches'][prefix_count:prefix_count+2]
        slot = receipt['slot']
        if (preflight.get('accepted') is not True or preflight.get('attempted_slot') != slot
                or preflight.get('select_consensus', {}).get('selected_slot') is not None
                or not 0 <= preflight.get('capture_age_ms', -1) <= 1000
                or confirmation.get('accepted') is not True or len(actual_inspection) != 2):
            raise ValueError('partial pending inspection lacks its bounded selection evidence')
        x, y, w, h = _observed_target(preflight, slot)
        targets = [(x+w//2, y+h//2), (360, 320)]
        cancel_endpoints = reset.get('endpoints')
        if reset.get('endpoint_evidence_source') is not None:
            from .native_inprocess_resume import _same_input_host
            from .native_settlement_resume import _verified_endpoints

            guard = row.get('native_input_preparation') or {}
            floor, checked, deadline = (guard.get(key) for key in
                ('authorization_floor', 'authorization_checked_at', 'input_deadline'))
            if (reset['endpoint_evidence_source'] != 'native_input_dispatches'
                    or guard.get('dispatch_state') != 'dispatch_attempted'
                    or (guard.get('x'), guard.get('y')) != targets[1]
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in (floor, checked, deadline))
                    or not floor < checked < deadline or not deadline-floor <= 1):
                raise ValueError('partial pending inspection lacks its original cancellation guard')
            guard_endpoints = guard.get('endpoints')
            _verified_endpoints(guard_endpoints, row['endpoints']['game']['pid'])
            _same_input_host(row['endpoints'], guard_endpoints)
            if cancel_endpoints is not None and not _same(cancel_endpoints, guard_endpoints):
                raise ValueError('partial pending inspection cancellation endpoints disagree')
            cancel_endpoints = guard_endpoints
        if (any(item.get('kind') == 'swipe' or (item.get('x'), item.get('y')) != target
                for item, target in zip(actual_inspection, targets, strict=True))
                or not _same(actual_inspection[1]['endpoints'], cancel_endpoints)
                or not 0 <= reset.get('capture_age_ms', -1) <= 1000):
            raise ValueError('partial pending inspection contains unaccounted input')
        prefix_count += 2
    inventory_inputs, status_inputs = navigation.get('inputs', ()), failure.get('inputs', ())
    if (type(prefix_count) is not int
            or (submitted is None and context_prefix_count != len(context.get('inputs', ())))
            or navigation.get('input_click_count') != len(inventory_inputs)
            or failure.get('input_click_count') != len(status_inputs)
            or inventory.get('total_input_click_count') != len(inventory_inputs)
            or status.get('total_input_click_count') != len(status_inputs)
            or event.get('total_input_click_count') != len(inventory_inputs)+len(status_inputs)
            or row['input_count'] != prefix_count+len(inventory_inputs)+len(status_inputs)
            or (not status_preinput_stop and [item.get('phase') for item in status_inputs[:1]] != ['open'])
            or (not status_preinput_stop and [item.get('phase') for item in status_inputs[-1:]] != ['close'])
            or any(item.get('phase') != 'scroll' for item in status_inputs[1:-1])):
        raise ValueError('partial pending source navigation accounting or operation order differs')
    actual = row['native_input_dispatches'][prefix_count:]
    if (not all(_navigation_dispatch_matches(item, proof, inventory_customization=True) for item, proof in
                zip(actual[:len(inventory_inputs)], inventory_inputs, strict=True))
            or not all(_navigation_dispatch_matches(item, proof, status=True) for item, proof in
                       zip(actual[len(inventory_inputs):], status_inputs, strict=True))):
        raise ValueError('partial pending source includes an input outside its recorded readers')
    audit = dict(event='native_pending_components_resume', source=row['id'],
        pending_source=row['id'] if submitted is not None else row['native_pending_source'],
        run_id=run.run_id, revision=run.revision,
        game_pid=row['endpoints']['game']['pid'], source_client_size=row['endpoints']['game']['client_size'],
        component_names=['cards'], source_initial_frame_ids=list(initial['frame_ids']),
        source_core_frame_ids=list(initial['frame_ids']), source_final_core_available=False,
        continuity_reference='original_initial_core_with_reader_return_certificates',
        source_reader_frame_ids=[frame for group in frame_groups[1:] for frame in group],
        source_return_frame_ids=list(navigation['after_frame_ids'] if status_preinput_stop else failure['after_frame_ids']),
        navigation_input_count=row['input_count']-int(submitted is not None),
        business_submit_count=int(submitted is not None), expected_candidate_count=len(possibilities),
        components_reusable=True, source_settlement_accepted=False, submission_authorized=False)
    if failed_status_frames is not None:
        audit.update(source_status_failed_frame_ids=list(failed_status_frames), source_status_checkpoint_accepted=False)
    return dict(components={'cards': copy.deepcopy(cards)}, core=copy.deepcopy(initial), audit=audit)


def verify_pending_component_completion(row, settlement, resumed):
    """Recheck original observations plus the new batch's fresh continuity."""
    if not _same(row.get('native_pending_components_audit'), resumed['audit']):
        raise ValueError('pending component completion audit differs from its original source')
    names = resumed['audit']['component_names']
    if (not names or not set(names).issubset(COMPONENT_NAMES) or sorted(names) != sorted(resumed['components'])
            or row.get('native_pending_components_source') != resumed['audit']['source']
            or row['endpoints']['game']['pid'] != resumed['audit']['game_pid']
            or ('source_client_size' in resumed['audit'] and row['endpoints']['game']['client_size']
                != resumed['audit']['source_client_size'])
            or settlement.get('resumed_fresh_components') != sorted(names)):
        raise ValueError('pending component completion source or retained keys differ')
    for key in names:
        if not _same(settlement.get('components', {}).get(key), resumed['components'][key]):
            raise ValueError('pending component completion changed the original observation: ' + key)
    before, after = settlement.get('initial_core'), settlement.get('core')
    old_frames = set(resumed['audit']['source_initial_frame_ids'] + resumed['audit']['source_core_frame_ids']
                     + resumed['audit'].get('source_reader_frame_ids', []))
    if (not _core_frames(before) or not _core_frames(after)
            or set(before['frame_ids']) & set(after['frame_ids'])
            or (set(before['frame_ids']) | set(after['frame_ids'])) & old_frames
            or any(not _same(core.get(key), resumed['core'].get(key))
                   for core in (before, after) for key in ('fields', 'families'))):
        raise ValueError('pending component completion lacks fresh unchanged initial/final core evidence')
