"""Zero-input continuation of a verified opening, with current artwork proof."""
from types import SimpleNamespace


def _geometry_failure(row):
    """Closed pre-input four-card rows with an isolated unresolved frame."""
    events = row.get('live_events', ())
    if (len(events) != 2 or events[0].get('event') != 'live_opening_state'
            or events[0].get('accepted') is not True
            or len(events[0].get('state', {}).get('deck', {}).get('hand', ())) != 4):
        return None
    stop = events[-1]
    if (stop.get('event') != 'live_selection_stop' or stop.get('runtime_phase') != 'authoritative_refresh_a'
            or stop.get('reason') != 'authoritative_live_hand_count_changed'
            or stop.get('executed') is not False or stop.get('input_may_have_been_dispatched') is not False
            or stop.get('would_submit') is not False
            or any(stop.get(k) != 0 for k in ('execution_submit_count', 'execution_selection_click_count', 'identity_probe_click_count'))
            or any(stop.get(k) for k in ('execution_selection_events', 'shadow_transaction_events',
                'authoritative_refresh_events', 'foreground_preparation_events'))):
        return None
    probes = stop.get('identity_probe_events', ())
    if ([p.get('phase') for p in probes] != ['initial_observation', 'retained_complete_hand_identity', 'hif_candidate_precompute']
            or probes[0].get('click_dispatched') is not False or probes[0].get('geometry_stable') is not True
            or probes[0].get('identity_source_accepted') is not True
            or probes[0].get('select_consensus', {}).get('reason') != 'unselected_unanimous'
            or probes[1].get('accepted') is not True or probes[2].get('click_dispatched') is not False
            or probes[2].get('proposal', {}).get('status') != 'PROPOSE'):
        return None
    failures = stop.get('failure_events', ())
    if len(failures) != 2:
        return None
    for index, failure in enumerate(failures):
        rows = failure.get('candidate_boxes', ())
        complete = [r for r in rows if len(r) == 4]
        if (failure.get('phase') != 'authoritative_refresh_a' or failure.get('retry_index') != index
                or failure.get('reason') != ('authoritative_live_hand_count_transient' if index == 0 else 'authoritative_live_hand_count_changed')
                or failure.get('accepted') is not False or failure.get('click_dispatched') is not False
                or failure.get('input_may_have_been_dispatched') is not False
                or failure.get('expected_selected_slot') is not None
                or failure.get('health_votes') != 3 or failure.get('playable_votes') != 3
                or failure.get('popup_votes') != 0 or failure.get('secondary_votes') != 0
                or not 0 <= failure.get('evidence_window_age_ms', -1) <= row['formal_params']['max_snapshot_age_ms']
                or len(rows) != 3 or len(complete) != 2 or sum(not r for r in rows) != 1
                or complete[0] != complete[1] or complete[0] != probes[0]['identity_source_candidate_boxes']
                or len(failure.get('frame_ids', ())) != 3):
            return None
    ids = [fid for f in failures for fid in f['frame_ids']]
    if len(set(ids)) != 6 or not set(ids) <= {f['frame_id'] for f in row['frames']}:
        return None
    return failures


def unsubmitted_proposal(row):
    if _geometry_failure(row) is not None:
        return row['live_events'][-1]['identity_probe_events'][-1]['proposal']
    return row['live_events'][-1]['proposal']


def _require_repaired_geometry(checkpoint, failures):
    import cv2
    from card_selection.model import frame_identifier

    row = checkpoint['row']
    ids = [fid for f in failures for fid in f['frame_ids']]
    frames = row['frames'][-6:]
    raw = [r['raw']['all'] for r in row['live_raw_fields'] if r['node'] == 'ProduceRecognitionCards'][-6:]
    if [f['frame_id'] for f in frames] != ids or len(raw) != 6:
        raise ValueError('HIF opening geometry lacks its six saved detector frames')
    module = checkpoint['observer_module']
    from ..hif_visual import HifLiveRunContext

    context = SimpleNamespace(gkh_hif_live_context=HifLiveRunContext(**row['observed_live_context']))
    expected = row['live_events'][-1]['identity_probe_events'][0]['identity_source_candidate_boxes']
    for frame, detected in zip(frames, raw, strict=True):
        image = cv2.imread(frame['path'])
        if image is None or frame_identifier(image) != frame['frame_id']:
            raise ValueError('HIF opening geometry failure pixels are missing or changed')
        cards = module._normalise_observed_live_candidates(context, image, detected)
        if [list(c.box) for c in cards] != expected:
            raise ValueError('HIF opening geometry remains unresolved on the actual failure frames')


def _safe_failure(row):
    events = row.get('live_events', ())
    if len(events) != 2 or events[0].get('event') != 'live_opening_state' or events[0].get('accepted') is not True:
        return None
    event = events[-1]
    refreshes = event.get('authoritative_refresh_events', ())
    if (event.get('event') != 'live_decision_observe' or event.get('executed') is not False
            or event.get('safe_decision', {}).get('reason') != 'card_identity_below_threshold'
            or event.get('safe_decision', {}).get('status') != 'STOP'
            or event.get('current_selected_slot') is not None or event.get('unknown_fields') != []
            or event.get('contradictory_fields') != [] or len(refreshes) != 1
            or any(event.get(k) != 0 for k in ('execution_submit_count', 'execution_selection_click_count',
                'execution_click_count', 'identity_probe_click_count', 'selection_click_count', 'total_input_click_count'))):
        return None
    refresh = refreshes[0]
    if (refresh.get('phase') != 'authoritative_refresh_a' or refresh.get('accepted') is not False
            or refresh.get('reason') != 'card_identity_below_threshold' or refresh.get('click_dispatched') is not False
            or refresh.get('input_may_have_been_dispatched') is not False
            or refresh.get('expected_selected_slot') is not None):
        return None
    proofs = {p['slot']: p for p in refresh.get('opening_exact_art_families', ())}
    observations = refresh.get('identity_revalidation', ())
    if not proofs or not observations or any(p.get('accepted') is not True for p in observations):
        return None
    raw = [proofs[p['slot']]['original_prediction'] if p['slot'] in proofs else p['fresh_prediction']
        for p in observations]
    return dict(phase='inventory_live_identity', frame_ids=refresh['frame_ids'],
        frame_id=event['snapshot_id'], predictions=raw)


def failure_evidence(row):
    safe = _safe_failure(row)
    return safe if safe is not None else row['live_events'][0]['failure_events'][0]


def is_unsubmitted_opening_identity_stop(row):
    from ..inventory import _weak_identity_retry_evidence
    from ..hif_live_recovery import closed_observer_record

    try:
        receipt = row.get('handoff_resume') or {}
        events = row.get('live_events', ())
        if (not closed_observer_record(row) or row.get('status') != 'STOP' or len(events) not in (1, 2)
                or row.get('input_count') != 0 or row.get('native_input_dispatches') != []
                or receipt.get('version') != 1 or receipt.get('boundary_kind') != 'unsubmitted_opening'
                or not receipt.get('source')
                or receipt['source'] != row.get('native_rule_resume_source')
                or receipt['source'] != row['automatic_battle']['previous_record']
                or any(row.get(k) for k in ('transaction_closed', 'choice_journal', 'gray_selection_journal',
                    'native_pending_source', 'native_selected_source', 'native_navigation_preparation', 'native_input_preparation', 'settlement'))):
            return False
        if _geometry_failure(row) is not None:
            return True
        safe = _safe_failure(row)
        if safe is not None:
            error = ValueError('inventory navigation Live visual family is unresolved')
            error.events = [safe]
            return bool(_weak_identity_retry_evidence(error, frame_count=3))
        if len(events) != 1:
            return False
        stop = events[0]
        if (stop.get('event') != 'live_selection_stop' or stop.get('runtime_phase') != 'runtime_prechecks'
                or stop.get('executed') is not False or stop.get('input_may_have_been_dispatched') is not False
                or any(stop.get(k) != 0 for k in ('execution_submit_count', 'execution_selection_click_count',
                    'identity_probe_click_count'))
                or any(stop.get(k) for k in ('end_turn_submit_count', 'execution_selection_events',
                    'identity_probe_events', 'shadow_transaction_events', 'authoritative_refresh_events'))):
            return False
        error = ValueError(stop.get('reason'))
        error.events = stop.get('failure_events', ())
        return bool(_weak_identity_retry_evidence(error, frame_count=3))
    except (ValueError, KeyError, TypeError):
        return False


def opening_source_chain(runtime, row):
    from .native_opening_root import has_native_opening_root
    from .native_settlement_resume import load_result
    from .native_opening_intent_resume import is_saved_opening_intent_stop

    current, chain = row, []
    while not has_native_opening_root(current):
        if not is_saved_opening_intent_stop(current) or current['id'] in chain:
            raise ValueError('HIF opening continuation lacks a closed zero-input identity boundary')
        chain.append(current['id'])
        parent = load_result(runtime, current['handoff_resume']['source'], allow_closed_transaction=True)
        for key in ('handoff_id', 'handoff_opening', 'observed_live_context'):
            if current.get(key) != parent.get(key):
                raise ValueError('HIF opening continuation changed its verified business binding')
        if (current['endpoints']['game'] != parent['endpoints']['game']
                or current['formal_params']['hif_policy_profile'] != parent['formal_params']['hif_policy_profile']):
            raise ValueError('HIF opening continuation changed its process or policy')
        current = parent
    return current, {*chain, current['id']}


def require_repaired_identity(checkpoint):
    import cv2
    from card_selection.model import frame_identifier
    from card_selection.types import CandidateBox, CardPrediction

    from ..hif_opening_art import corroborate

    row, reference = checkpoint['row'], checkpoint.get('opening_art_reference')
    if (not is_unsubmitted_opening_identity_stop(row) or not reference
            or checkpoint.get('untouched_opening') is not True or checkpoint.get('pending') or checkpoint.get('selected')):
        raise ValueError('HIF opening identity recovery lacks its verified untouched reference')
    geometry = _geometry_failure(row)
    if geometry is not None:
        _require_repaired_geometry(checkpoint, geometry)
        return
    failed = failure_evidence(row)
    paths = {f['frame_id']: f['path'] for f in row['frames']}
    context = SimpleNamespace(gkh_hif_opening_art_reference=reference, gkh_hif_opening_state=checkpoint['bundle'],
        gkh_hif_live_context=SimpleNamespace(**checkpoint['live']))
    for fid in failed['frame_ids']:
        image = cv2.imread(paths.get(fid, ''))
        if image is None or frame_identifier(image) != fid:
            raise ValueError('HIF opening identity failure pixels are missing or changed')
        candidates = tuple(CandidateBox(p['slot'], tuple(p['box'])) for p in failed['predictions'])
        predictions = tuple(CardPrediction(slot=p['slot'], box=tuple(p['box']), frame_id=fid,
            card_id=None, predicted_card_id=None, class_name='', confidence=p['confidence'], margin=p['margin'],
            raw_score=p['confidence'], accepted=p['accepted'], reason=p['reason'], model_sha256='', classes_sha256='',
            recognizer=p['recognizer'], top_k_card_ids=tuple(tuple(f) for f in p['top_k_card_ids']))
            for p in failed['predictions'])
        evidence = []
        repaired = corroborate(context, image, candidates, predictions, selected_slot=None, evidence=evidence)
        if not evidence or not all(p.accepted for p in repaired):
            raise ValueError('HIF opening identity remains unresolved with current fixed artwork proof')
    if _safe_failure(row) is not None:
        from ..contracts import LiveCard, LiveState
        from ..hif_opening_art import safety_identity_verified

        event = row['live_events'][-1]
        generation = event['authoritative_refresh_events'][0]['generation']
        state = LiveState(snapshot_id=event['snapshot_id'], turn=generation['turn'],
            turns_remaining=generation['turns_remaining'], stamina=generation['stamina'],
            max_stamina=checkpoint['state'].max_stamina, score=generation['score'],
            score_multiplier=generation['score_multiplier'], action_order=tuple(generation['action_order']),
            items=(), field_effects=(), statuses=(), capture_age_ms=event['capture_age_ms'],
            hand=tuple(LiveCard(**dict(c, box=tuple(c['box']), tags=tuple(c['tags']))) for c in event['hand']))
        threshold = row['formal_params']['minimum_identity_confidence']
        if any(c.identity_confidence < threshold and not safety_identity_verified(context, state, c) for c in state.hand):
            raise ValueError('HIF opening current identity proof does not reach the safety gate')
