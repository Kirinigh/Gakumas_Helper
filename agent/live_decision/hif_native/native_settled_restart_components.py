"""Resume three finished readers from a navigation-only restart observation."""

import copy
from dataclasses import asdict

from ..opening_observer import direct_item_observation
from .native_inprocess_resume import _same_input_host
from .native_settlement_resume import load_result, _verified_endpoints
from .native_pending_components import _same, _frames, _core_frames, _navigation_dispatch_matches

COMPONENT_NAMES = ('cards', 'status', 'items')


def _navigation(row, reports):
    """Account for this attempt's latest history and five reader navigations.

    context_recovery belongs to the archived history attempt. Its inputs were
    not repeated by this process and must not be added to the current prefix.
    """
    latest = row['settled_restart_resume_history']
    inventory = reports[0]['inventory_events'][0]
    status, items, drinks, turns = [report['status_events'][0] for report in reports[1:5]]
    groups = [(latest, 'latest'), (inventory, 'inventory'), (status, 'status'),
              (items, 'items'), (turns, 'turns')]
    hud_items = direct_item_observation(items)
    if (inventory.get('phase') != 'inventory_zone_observation' or status.get('phase') != 'status_inspection'
            or not hud_items and items.get('phase') != 'item_description_inspection'
            or drinks.get('phase') != 'complete_drink_inventory'
            or drinks.get('input_click_count') != 0 or drinks.get('drinks_consumed') != 0
            or reports[3].get('total_input_click_count') != 0
            or turns.get('phase') != 'turn_information_observation'
            or turns.get('accepted') is not False):
        raise ValueError('settled component source contains an unexpected drink or turn operation')
    proofs = []
    for event, kind in groups:
        if kind == 'items' and hud_items:
            continue
        inputs = event.get('inputs', ())
        if (type(event.get('input_click_count')) is not int or event['input_click_count'] != len(inputs)
                or len(inputs) < 2):
            raise ValueError('settled component reader navigation count is incomplete')
        if kind == 'latest':
            if [item.get('closing') for item in inputs] != [False, True]:
                raise ValueError('settled component latest history is not one open/close observation')
            inputs = [dict(item, phase='close' if item['closing'] else 'open') for item in inputs]
        if (inputs[0].get('phase') != 'open' or inputs[-1].get('phase') != 'close'
                or any(item.get('phase') not in
                       ({'scroll', 'customization_select'} if kind == 'inventory' else
                        {'scroll', 'scroll_top'} if kind == 'items' else {'scroll'})
                       for item in inputs[1:-1])):
            raise ValueError('settled component reader navigation order changed')
        proofs.extend((kind, item) for item in inputs)
    reader_count = sum(report.get('total_input_click_count', -1) for report in reports[:5])
    if (row['settlement'].get('total_input_click_count') != reader_count
            or [report.get('total_input_click_count') for report in reports[:5]]
            != [inventory['input_click_count'], status['input_click_count'], items['input_click_count'],
                0, turns['input_click_count']]
            or row.get('input_count') != reader_count + latest['input_click_count']
            or len(row.get('native_input_dispatches', ())) != len(proofs)
            or row['input_count'] != len(proofs)):
        raise ValueError('settled component source has unaccounted input')
    game = row['endpoints']['game']
    for actual, (kind, original) in zip(row['native_input_dispatches'], proofs, strict=True):
        proof = original
        _verified_endpoints(actual.get('endpoints'), game['pid'])
        _same_input_host(row['endpoints'], actual['endpoints'])
        if actual['endpoints']['game'].get('observed_client_size') != list(game['client_size']):
            raise ValueError('settled component navigation changed client size')
        if kind == 'items' and proof['phase'] in ('scroll', 'scroll_top'):
            # These are the two existing item_panel.dispatch gestures. Its
            # recorded box is the popup close control, not a click target.
            swipe = (560, 610, 560, 530, 400) if proof['phase'] == 'scroll' else (560, 220, 560, 610, 600)
            proof = dict(proof, phase='scroll', begin=list(swipe[:2]), end=list(swipe[2:4]))
            if actual.get('duration') != swipe[4]:
                raise ValueError('settled component item swipe duration changed')
        elif kind == 'turns' and proof['phase'] == 'scroll':
            top = proof.get('toward_top')
            if type(top) is not bool:
                raise ValueError('settled component turn swipe direction is missing')
            swipe = [560, 700, 560, 940 if top else 460, 500]
            if proof.get('swipe') != swipe or actual.get('duration') != swipe[4]:
                raise ValueError('settled component turn swipe changed')
            proof = dict(proof, begin=swipe[:2], end=swipe[2:4])
        if not _navigation_dispatch_matches(actual, proof, status=kind == 'status',
                                             inventory_customization=kind == 'inventory'):
            raise ValueError('settled component source input differs from recorded navigation')


def load_settled_restart_components_source(runtime, name, module, *, prepared, history_source):
    """Use already verified full history and this attempt's fresh continuity.

    No full source replay occurs here. The caller supplies the preparation
    returned by the ordinary settled-history loader and new latest-history
    gate. A failed turns reader is retained as failure, never a return proof.
    """
    from live_decision.opening_observer import opening_component
    from live_decision.high_score.restart import _frame_sets, _reject_retained
    from live_decision.high_score.settled_restart import continue_settled_restart_history

    from .native_opening_root import verify_native_round_context
    from .native_settled_restart import _PARENTS, _OTHER_ROOTS, _submissions

    row = load_result(runtime, name)
    run = prepared.new_run
    epoch = dict(run_id=run.run_id, revision=run.revision)
    if (row.get('native_settled_restart_source') != history_source.get('native_settled_restart_source')
            or row.get('native_settled_restart_history_source') != history_source.get('id')
            or row.get('native_settled_restart_components_source')
            or any(row.get(key) for key in (*_PARENTS, *_OTHER_ROOTS[:-1]))
            or not _same(row.get('observed_live_context'), asdict(run))
            or not _same(history_source.get('observed_live_context'), asdict(run))
            or row['endpoints']['game']['pid'] != prepared.new_pid
            or row['endpoints']['game']['pid'] != history_source['endpoints']['game']['pid']
            or row['endpoints']['game']['client_size'] != history_source['endpoints']['game']['client_size']
            or not _same(row.get('settled_restart_history'), prepared.history)
            or not _same(history_source.get('settled_restart_history'), prepared.history)
            or row.get('settled_restart_settlement') is not None or _submissions(row)):
        raise ValueError('settled component source changed its verified history, process or epoch')
    verify_native_round_context(row, module, history_source, required=True)
    reports = row.get('live_events', ())
    if ([event.get('event') for event in reports] != ['live_inventory_observe', *['live_status_observe']*4,
                                                    'live_committed_state']
            or any(event.get('executed') is True or event.get('drinks_consumed', 0)
                   or event.get('execution_selection_click_count', 0) for event in reports)):
        raise ValueError('settled component source contains another operation')
    partial = row.get('settlement', {})
    if (not _same(reports[-1], partial) or partial.get('accepted') is not False
            or partial.get('reason') != 'HIF opening component failed: turns'
            or partial.get('force_fresh_components') is not True
            or partial.get('execution_submit_count') != 0 or partial.get('maintained_components') != []
            or partial.get('resumed_fresh_components') or 'core' in partial or 'expected_state' in partial
            or set(partial.get('components', {})) != {'cards', 'status', 'items', 'drinks'}):
        raise ValueError('settled component source is not the original unfinished turns observation')
    latest = row.get('settled_restart_resume_history', {})
    continue_settled_restart_history(prepared, current_history=latest, run=run, game_pid=prepared.new_pid)
    current = prepared.continuity_history
    if current is None:
        raise ValueError('settled component reuse needs new latest-history continuity')
    continue_settled_restart_history(prepared, current_history=current, run=run, game_pid=prepared.new_pid)
    initial = partial.get('initial_core')
    if (not _core_frames(initial) or any(not _same(initial.get(key), latest['after'].get(key))
                                       for key in ('fields', 'families'))):
        raise ValueError('settled component initial core differs from its verified latest history')
    archived = (set(_frame_sets(prepared.settlement)) | set(_frame_sets(prepared.reference_history))
                | set(_frame_sets(prepared.history)) | set(_frame_sets(prepared.opening)) | set(_frame_sets(latest)))
    if archived.intersection(_frame_sets(initial)):
        raise ValueError('settled component initial core reuses archived frames')
    components = {}
    for index, key in enumerate(('cards', 'status', 'items', 'drinks')):
        component = opening_component(reports[index], key, run)
        if not _same(component, partial['components'][key]):
            raise ValueError('settled component differs from its original reader: ' + key)
        _reject_retained(component)
        frames = _frame_sets(component)
        if (not frames or archived.intersection(frames)
                or any(component.get(name) != value for name, value in epoch.items())):
            raise ValueError('settled component lacks independent original frames: ' + key)
        if key in COMPONENT_NAMES:
            if (not _frames(component.get('before_frame_ids')) or not _frames(component.get('after_frame_ids'))
                    or not (key == 'items' and direct_item_observation(component))
                    and set(component['before_frame_ids']) & set(component['after_frame_ids'])):
                raise ValueError('settled reusable component lacks independent reader endpoints: ' + key)
            components[key] = copy.deepcopy(component)
    if any(card.get('source') not in ('observed_detail', 'observed_badge_absence')
           for card in components['cards'].get('cards', ())):
        raise ValueError('settled reusable cards lack fresh permanent observations')
    _navigation(row, reports)
    source_frames = set(_frame_sets(partial)) | set(_frame_sets(latest)) | set(_frame_sets(reports))
    if source_frames.intersection(_frame_sets(current)):
        raise ValueError('settled component current continuity reuses source frames')
    audit = dict(event='native_settled_restart_components_resume', source=name,
        settled_source=row['native_settled_restart_source'], history_source=history_source['id'], **epoch,
        game_pid=prepared.new_pid, source_client_size=list(row['endpoints']['game']['client_size']),
        component_names=list(COMPONENT_NAMES), source_initial_frame_ids=list(initial['frame_ids']),
        source_reader_frame_ids=sorted(source_frames), source_final_core_available=False,
        source_settlement_accepted=False, failed_component='turns', navigation_input_count=row['input_count'],
        business_submit_count=0, submission_authorized=False)
    return dict(components=components, core=copy.deepcopy(initial), audit=audit)


def verify_settled_restart_component_completion(row, settlement, resumed):
    """Keep exact original readers and require new unchanged initial/final cores."""
    from live_decision.high_score.restart import _frame_sets

    audit = resumed['audit']
    if (row.get('native_settled_restart_components_source') != audit['source']
            or not _same(row.get('settled_restart_components_resume'), audit)
            or row.get('native_settled_restart_source') != audit['settled_source']
            or row.get('native_settled_restart_history_source') != audit['history_source']
            or row['endpoints']['game']['pid'] != audit['game_pid']
            or row['endpoints']['game']['client_size'] != audit['source_client_size']
            or any(row.get('observed_live_context', {}).get(key) != audit[key] for key in ('run_id', 'revision'))
            or settlement.get('resumed_fresh_components') != sorted(COMPONENT_NAMES)
            or set(resumed['components']) != set(COMPONENT_NAMES)
            or any(not _same(settlement.get('components', {}).get(key), resumed['components'][key])
                   for key in COMPONENT_NAMES)):
        raise ValueError('settled component completion changed its source or original observations')
    before, after = settlement.get('initial_core'), settlement.get('core')
    old_frames = set(audit['source_reader_frame_ids'])
    if any(old_frames.intersection(_frame_sets(settlement.get('components', {}).get(key, {})))
           for key in ('drinks', 'turns')):
        raise ValueError('settled component completion reused an old drink or turn observation')
    if (not _core_frames(before) or not _core_frames(after)
            or set(before['frame_ids']) & set(after['frame_ids'])
            or (set(before['frame_ids']) | set(after['frame_ids'])) & old_frames
            or any(not _same(core.get(key), resumed['core'].get(key))
                   for core in (before, after) for key in ('fields', 'families'))):
        raise ValueError('settled component completion lacks new unchanged complete cores')
