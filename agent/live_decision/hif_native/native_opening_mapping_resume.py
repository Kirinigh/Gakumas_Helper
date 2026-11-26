"""Cold recovery of a mapped first decision; never retry a model STOP in-place."""


def is_unsubmitted_mapping_stop(row):
    """Account for every metadata input, excluding any selection or submission."""
    try:
        _verify(row)
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    return True


def _verify(row):
    from .native_opening_root import PARENT_FIELDS

    def require(value):
        if not value:
            raise ValueError('opening mapping STOP has an incomplete zero-input boundary')

    events = row['live_events']
    opening, decision = events[-2:]
    require(row['status'] == 'STOP' and row.get('live_return') is False
        and row['formal_params']['hif_policy_profile'] == 'hif-gkms-fast-trial-v1'
        and not any(row.get(k) for k in (*PARENT_FIELDS, 'native_input_preparation',
            'choice_journal', 'gray_selection_journal', 'transaction_closed'))
        and opening.get('event') == 'live_opening_state' and opening.get('accepted') is True
        and decision.get('event') == 'live_decision_observe'
        and decision.get('current_selected_slot') is None and decision.get('would_submit') is False)
    require(all(e.get('executed') is False and e.get('execution_submit_count') == 0
        and not e.get('end_turn_submit_count', 0) for e in events))
    require(all(type(decision.get(k)) is int and decision[k] == 0 for k in
        ('execution_click_count', 'execution_selection_click_count', 'selection_click_count',
         'identity_probe_click_count', 'total_input_click_count')))
    require(not decision.get('execution_selection_events') and not decision.get('shadow_transaction_events'))
    require(all(step.get('phase') in ('initial_observation', 'retained_complete_hand_identity',
        'hif_candidate_precompute') and step.get('click_dispatched', False) is False
        and step.get('attempted_slot') is None for step in decision.get('selection_events', ())))
    proposal, safe = decision['proposal'], decision['safe_decision']
    detail = proposal['candidate_traces']['high_score']
    trace = detail['trace']['decision']
    reason = detail['reason']
    require(isinstance(reason, str) and reason.startswith('ordinary_state_mapping_incomplete:')
        and detail.get('action') is None and detail.get('has_action_intent') is False
        and detail.get('outcome') == 'STOP_TASK' and detail.get('execution_authorized') is False
        and trace.get('status') == 'failed' and trace.get('mapping_status') == 'unsupported'
        and trace.get('index') == 0 and trace.get('model_output') is None
        and proposal.get('status') == 'STOP' and proposal.get('action') is None
        and proposal.get('reason') == 'hif_policy_stop:' + reason
        and safe.get('proposal') == proposal and safe.get('status') == 'STOP'
        and safe.get('action') is None and safe.get('has_action_intent') is False
        and safe.get('executed') is False)

    verify_opening_navigation(row, events[:-2], opening)


def verify_opening_navigation(row, panels, opening, *, closed_turn_failure=False,
                              closed_inventory_failure=False, rejected_turn_open=False,
                              pending_inventory_return=False,
                              closed_item_failure=False, open_item_tail=False, rejected_drink_open=False):
    """Verify the original complete reader inputs, including a closed table."""
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints
    from .native_pending_components import _navigation_dispatch_matches

    def require(value):
        if not value:
            raise ValueError('opening reader has incomplete navigation accounting')

    recovery = row['context_recovery']
    from ..hif_entry_round import UNREAD_INVENTORY_SOURCES, verified_entry_receipt
    require(recovery.get('accepted') is True and (
        recovery.get('returned_to_live') is True or verified_entry_receipt(row, recovery)))
    proofs = []
    resumed = row.get('opening_navigation_resume')
    notebook_suffix = bool(resumed and resumed.get('boundary_kind') == 'unbound_opening'
        and resumed.get('completed_phases') == ['live', 'menu'])
    fresh_notebook = resumed is None or resumed.get('boundary_kind') == 'unstarted_playable_wait' or notebook_suffix
    if notebook_suffix:
        actions = [step['phase'] for step in recovery['inputs']]
        require(resumed.get('source_result') == row['automatic_battle']['previous_record']
            and resumed.get('input_authorized') is False and recovery.get('returned_to_live') is True
            and actions in (['notebook', 'identity'], ['identity'])
            and recovery.get('input_click_count') == len(actions))
    elif resumed is not None and fresh_notebook:
        require(resumed == dict(boundary_kind='unstarted_playable_wait', input_authorized=False,
            source_result=row['automatic_battle']['previous_record']))
    inspected = 0
    if resumed is not None and resumed.get('boundary_kind') in UNREAD_INVENTORY_SOURCES:
        core = opening.get('initial_core') or {}
        live = resumed.get('expected_live') or {}
        entry = resumed['boundary_kind'] == 'unbound_route_entry'
        require((not entry or verified_entry_receipt(row, recovery))
            and (entry or closed_inventory_failure)
            and resumed.get('input_authorized') is False
            and resumed.get('source_result') == row['automatic_battle']['previous_record']
            and core.get('fields', [])[:2] == [live.get('turns'), live.get('stamina')])
    elif not fresh_notebook:
        from ..hif_navigation_recovery import verify_unstarted_core

        require(resumed.get('boundary_kind') == 'unbound_components'
            and resumed.get('input_authorized') is False
            and resumed.get('source_result') == row['automatic_battle']['previous_record']
            and opening.get('unstarted_opening_continuity') == verify_unstarted_core(
                resumed.get('initial_core', {}), opening.get('initial_core', {})))
    for item in (recovery['inputs'] if fresh_notebook else ()):
        require(item.get('phase') in ('live', 'menu', 'notebook', 'identity')
            and item.get('succeeded') is True and item.get('uncertain') is not True)
        proofs.append((dict(item, phase='open', box=item['target']), {}))
    context_count = len(proofs)
    extended = opening.get('extended_hand_identity') or {}
    tail_inspections = extended.get('inspections', ()) if opening.get('accepted') is True else ()
    if (resumed and resumed.get('boundary_kind') == 'unbound_notebook'
            or (opening.get('hand_identity_inspection') or {}).get('accepted') is True
            or opening.get('hand_identity_inspection') and (
                opening.get('accepted') is True or resumed and resumed.get('boundary_kind') == 'unbound_route_entry')):
        inspected = _verified_opening_title_inputs(row, opening, offset=context_count,
            final=not tail_inspections)
    for report in panels:
        failed_inventory = (closed_inventory_failure and len(panels) == 1
            and report.get('event') == 'live_inventory_observe'
            and report.get('accepted') is False)
        closed_phase = ('turn_information_observation' if closed_turn_failure else
            'item_description_inspection' if closed_item_failure or open_item_tail else
            'complete_drink_inventory' if rejected_drink_open else None)
        failed_panel = (closed_phase is not None and report is panels[-1]
            and report.get('accepted') is False
            and len(report.get('status_events', ())) == 1
            and report['status_events'][0].get('phase') == closed_phase
            and (report['status_events'][0].get('returned_unchanged') is True or open_item_tail or rejected_drink_open))
        require(report.get('accepted') is True or failed_panel or failed_inventory)
        if report['event'] == 'live_inventory_observe':
            panel, permanent = report['inventory_events']
            require(panel.get('phase') == 'inventory_zone_observation'
                and (panel.get('live_state_unchanged') is True or failed_inventory and pending_inventory_return
                     and panel.get('live_state_unchanged') is None)
                and permanent.get('accepted') is (not failed_inventory)
                and permanent.get('input_click_count') == 0)
            options = dict(inventory_customization=True)
        else:
            require(report['event'] == 'live_status_observe')
            panel, = report['status_events']
            options = dict(status=panel.get('phase') == 'status_inspection')
        require(panel.get('accepted') is True or failed_panel or failed_inventory)
        before = len(proofs)
        phase = panel['phase']
        if phase == 'complete_drink_inventory':
            require((panel.get('returned_unchanged') is True or rejected_drink_open and failed_panel)
                and panel.get('drinks_consumed') == 0)
            for group in panel['events']:
                interrupted = rejected_drink_open and failed_panel and group is panel['events'][-1]
                require(group.get('accepted') is (not interrupted) and group.get('drinks_consumed') == 0)
                if group['phase'] == 'drink_slot_occupancy':
                    require(group['input_click_count'] == 0)
                    continue
                require(group['phase'] == 'occupied_drink_slots_inspection')
                for transaction in group['transactions']:
                    if interrupted and transaction is group['transactions'][-1]:
                        prepared = row.get('native_navigation_preparation') or {}
                        attempted, = transaction['inputs']
                        require(attempted['phase'] == 'open' and transaction.get('input_click_count') == 1
                            and transaction.get('accepted') is False and transaction.get('drinks_consumed') == 0
                            and prepared.get('dispatch_state') == 'rejected_before_dispatch'
                            and prepared.get('input_may_have_been_dispatched') is False
                            and prepared.get('reason') == transaction.get('reason')
                            and _navigation_dispatch_matches(prepared, attempted))
                        continue
                    require(transaction.get('accepted') is True and transaction.get('returned_unchanged') is True
                        and transaction.get('drinks_consumed') == 0 and transaction.get('input_click_count') == 2)
                    selected, cancel = transaction['inputs']
                    require(selected['phase'] == 'open' and cancel['phase'] == 'cancel'
                        and transaction.get('use_submitted', False) is False
                        and transaction.get('execution_submit_count', 0) == 0)
                    proofs.extend(((selected, {}), (dict(cancel, phase='close'), {})))
        elif phase == 'item_hud_inventory':
            from ..opening_observer import direct_item_observation

            require(direct_item_observation(panel))
        else:
            require(phase in ('inventory_zone_observation', 'status_inspection',
                'item_description_inspection', 'turn_information_observation'))
            require(phase == 'inventory_zone_observation' or panel.get('returned_unchanged') is True or open_item_tail and failed_panel)
            for item in panel['inputs']:
                if phase == 'item_description_inspection' and item['phase'] in ('scroll', 'scroll_top'):
                    distance = panel.get('scroll_drag_pixels', 80)
                    require(distance in (80, 120))
                    swipe = (560, 610, 560, 610-distance, 400) if item['phase'] == 'scroll' else (560, 220, 560, 610, 600)
                    item = dict(item, phase='scroll', begin=list(swipe[:2]), end=list(swipe[2:4]))
                    proofs.append((item, dict(options, expected_duration=swipe[4])))
                elif phase == 'turn_information_observation' and item['phase'] == 'scroll':
                    proofs.append((dict(item, phase='turn_tail_scroll'), dict(history=True)))
                elif rejected_turn_open and phase == 'status_inspection' and item['phase'] == 'scroll_retry':
                    require(item.get('completed') is True and (panel.get('scroll_retry') or {}).get('dispatched') is True)
                    proofs.append((dict(item, phase='scroll'), dict(options, expected_duration=400)))
                else:
                    proofs.append((item, options))
        require(panel['input_click_count'] == report['total_input_click_count'] ==
            len(proofs)-before + int(rejected_drink_open and failed_panel))
    panel_end = inspected + len(proofs)
    tail_count = 0
    for index, audit in enumerate(tail_inspections):
        tail_count += _verified_opening_title_inputs(row, opening, audit=audit, offset=panel_end+tail_count,
            final=index+1 == len(tail_inspections))
    if tail_inspections:
        require(extended.get('business_submit_count') == 0 and extended.get('input_click_count') == tail_count)
    opening_count = panel_end + tail_count - context_count
    if panels and resumed and resumed.get('resume_status_panel') and panels[0].get('accepted') is True:
        # A resumed status panel closes in its own observer before the opening reader starts.
        require(panels[0]['status_events'][0].get('resumed_status_source') == resumed.get(
            'status_panel_source', resumed['source_result']))
        opening_count -= panels[0]['total_input_click_count']
    if rejected_turn_open:
        opening_count += 1  # The opening reader counted the rejected turn-table attempt.
    if panels and resumed and (resumed.get('item_tail_resume') or {}).get('open_before'):
        require(panels[0]['status_events'][0].get('resumed_open_item_source') == resumed['source_result'])
        opening_count -= panels[0]['total_input_click_count']
    opening_count += int(rejected_drink_open)
    require(opening['total_input_click_count'] == opening_count
        and row['input_count'] == panel_end+tail_count == len(row['native_input_dispatches']))
    dispatches = row['native_input_dispatches']
    panel_dispatches = dispatches[:context_count] + dispatches[context_count+inspected:panel_end]
    for sent, (proof, options) in zip(panel_dispatches, proofs, strict=True):
        _verified_endpoints(sent.get('endpoints'), row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], sent['endpoints'])
        observed = proof.get('endpoints', {})
        if sent.get('endpoints') != observed:
            # The final Root dispatch adds process-instance metadata to the
            # observer's same endpoint fields. Compare every original field;
            # this projection is receipt matching, not an input credential.
            require(set(sent['endpoints']) == set(observed))
            for name, endpoint in observed.items():
                actual = sent['endpoints'][name]
                require(all(actual.get(key) == value for key, value in endpoint.items())
                    and set(actual) - set(endpoint) <= {'created_at_100ns', 'executable'})
            sent = dict(sent, endpoints=observed)
        if 'expected_duration' in options:
            require(sent.get('duration') == options['expected_duration'])
            options = {k: v for k, v in options.items() if k != 'expected_duration'}
        require(_navigation_dispatch_matches(sent, proof, **options))
    prepared = row.get('native_navigation_preparation')
    require(rejected_turn_open or rejected_drink_open or prepared is None or prepared.get('dispatch_state') == 'dispatch_attempted'
        and panel_dispatches and all(prepared.get(k) == panel_dispatches[-1].get(k) for k in ('x', 'y', 'endpoints')))


def _verified_opening_title_inputs(row, opening, *, audit=None, offset=0, final=True):
    """Match one observed title selection and neutral return at its own SDK dispatch indices."""
    from ..hif_upstream_ui import UPSTREAM_REVISION
    from ..hif_hand_inspection import HAND_VISITS_VERSION, TITLE_VISIT_VERSION, verify_upstream_clear
    from ..upstream_hif_card_ui import BattleRecognitionMixin
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints

    if audit is None:
        audit = opening.get('hand_identity_inspection') or {}
        if audit.get('version') == HAND_VISITS_VERSION:
            if audit.get('opening') is not True or audit.get('accepted') is not True:
                raise ValueError('opening title visits were not completed')
            count = 0
            for index, visit in enumerate(audit['visits']):
                count += _verified_opening_title_inputs(row, {'hand_identity_inspection': visit},
                    offset=offset+count, final=final and index == len(audit['visits'])-1)
            if count != audit['input_click_count']:
                raise ValueError('opening title visits differ from the original input count')
            return count
        bound_slot = audit.get('opening') is True
    else:
        bound_slot = audit.get('selection_accepted') is True and audit.get('slot') in audit.get('physical_slots', ())
    selection = audit.get('upstream_selection') or {}
    selected, resets = selection.get('inputs', ()), audit.get('pending_inspection_reset', ())
    if (audit.get('version') != TITLE_VISIT_VERSION or not bound_slot or audit.get('accepted') is not True
            or audit.get('business_submit_count') != 0 or audit.get('observed_live_context') != row['observed_live_context']
            or selection.get('revision') != UPSTREAM_REVISION
            or not 1 <= len(selected) <= BattleRecognitionMixin.HIF_CARD_CLICK_ATTEMPTS
            or any(click.get('phase') != 'select' for click in selected) or len(resets) != 1):
        raise ValueError('opening inventory lacks its completed title selection')
    inputs = [*selected, *verify_upstream_clear(resets[0])]
    actual = row['native_input_dispatches'][offset:offset+len(inputs)]
    if (not len(inputs) == audit.get('input_click_count') == len(actual)
            or any(click.get('dispatched') is not True or click.get('dispatch_index') != i
                or not click.get('job_id') or sent.get('kind', 'click') != 'click'
                or any(click.get(k) != sent.get(k) for k in ('x', 'y'))
                for i, (click, sent) in enumerate(zip(inputs, actual), start=offset))):
        raise ValueError('opening title inputs differ from the original dispatch receipts')
    for sent in actual:
        _verified_endpoints(sent.get('endpoints'), row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], sent['endpoints'])
    prepared = row.get('native_upstream_input_preparation') or {}
    if final and (prepared.get('dispatch_state') != 'dispatch_attempted'
            or any(prepared.get(k) != actual[-1].get(k) for k in ('x', 'y', 'endpoints'))
            or prepared.get('authorization_floor') != inputs[-1].get('captured_at')
            or prepared.get('input_deadline') != inputs[-1].get('input_deadline')):
        raise ValueError('opening title has an unaccounted upstream input preparation')
    return len(inputs)


def verify_opening_metadata_source(runtime, row):
    """Reused opening metadata must belong to the immediate unbound source."""
    resumed = row.get('opening_navigation_resume')
    if resumed is None:
        return
    from .native_settlement_resume import load_result

    parent = load_result(runtime, resumed['source_result'])
    openings = [event for event in parent.get('live_events', ()) if event.get('event') == 'live_opening_state']
    if resumed.get('boundary_kind') == 'unbound_route_entry':
        from ..hif_entry_round import verified_entry_receipt

        recovery = parent.get('context_recovery', {})
        if (not verified_entry_receipt(parent, recovery) or len(openings) != 1 or openings[0].get('components') != {}
                or parent.get('input_count') != 0 or parent.get('total_input_count') != 0
                or parent.get('native_input_dispatches') != [] or resumed.get('input_authorized') is not False
                or resumed['source_result'] != row['automatic_battle']['previous_record']
                or resumed.get('expected_live') != {k:recovery['observations'][0][0][k] for k in ('turns','stamina')}):
            raise ValueError('opening metadata differs from the original unread route entry')
    elif len(openings) != 1 or openings[0].get('initial_core') != resumed['initial_core']:
        raise ValueError('opening metadata does not match its original unbound core')
    if (len(openings) != 1 or openings[0].get('accepted') is not False
            or row.get('context_recovery') != parent.get('context_recovery')
            or row.get('observed_live_context') != parent.get('observed_live_context')
            or row.get('handoff_id') != parent.get('handoff_id')
            or parent.get('handoff_opening') is not None
            or any(e.get('executed') is True or e.get('execution_submit_count', 0)
                or e.get('end_turn_submit_count', 0) for e in parent.get('live_events', ()))):
        raise ValueError('opening metadata does not match its original unbound source')
    return openings[0]


def require_repaired_mapping(checkpoint):
    """Confirm the actual full opening maps now, before any fresh UI operation."""
    from agent.live_decision.high_score.contracts import PlanType, DecisionKind, DecisionRequest, CapabilityReport
    from agent.live_decision.high_score.gkms_shadow_adapter import build_shadow_observation

    policy = checkpoint['policy'].policy
    bundle = checkpoint['bundle']
    request = DecisionRequest('cold-opening-mapping-check', PlanType.SENSE, DecisionKind.LIVE,
        policy.history.state, CapabilityReport(True, True, True, True, True),
        card_slots=tuple((iid, slot) for slot, iid in enumerate(bundle.hand_instance_ids)),
        drink_slots=bundle.drink_slots, ruleset_id=bundle.kernel.ruleset_id)
    projection = build_shadow_observation(request, policy.kernel, policy.context, policy.history)
    if projection.status != 'supported':
        raise ValueError('opening mapping is still unsupported:' + '|'.join(projection.reason_codes))
