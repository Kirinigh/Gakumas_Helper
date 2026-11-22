"""Resume a completed strong intent after one audited, unsubmitted SELECT."""
from dataclasses import asdict


def selected_plan(row):
    from ..hif_upstream_ui import UPSTREAM_REVISION
    from ..hif_live_recovery import closed_observer_record
    from .native_inprocess_resume import _same_input_host
    from .native_settlement_resume import _verified_endpoints

    ui = row.get('upstream_ui_journal') or {}
    events = row.get('live_events', ())
    cards = [event for event in events if event.get('event') == 'live_upstream_card']
    cache = row.get('mcts_search_cache') or {}
    inputs = ui.get('inputs', ())
    actual = row.get('native_input_dispatches', ())
    if (not closed_observer_record(row) or row.get('status') != 'STOP'
            or row.get('live_return') is not False or row.get('transaction_closed') or row.get('error')
            or len(cards) != 1 or row.get('input_count') != 1 or len(inputs) != 1 or len(actual) != 1
            or ui.get('kind') != 'upstream_hif_card_ui' or ui.get('revision') != UPSTREAM_REVISION
            or ui.get('response_confirmed') is not False or ui.get('playable') is not False
            or cache.get('record_id') != row.get('id') or cache.get('phase') != 'pending_intent'
            or row.get('actual_policy_profile') != 'hif-strong-reference-trial-v1'
            or not row.get('native_inprocess_source')
            or row['automatic_battle']['previous_record'] != row['native_inprocess_source']
            or any(event.get('executed') is not False or event.get('execution_submit_count', 0)
                or event.get('end_turn_submit_count', 0) or event.get('drinks_consumed', 0)
                or event.get('event') not in ('live_upstream_card', 'live_decision_exception_stop')
                for event in events)):
        raise ValueError('upstream selection is not a closed unsubmitted intent')
    event = cards[0]
    proposal = event['proposal']
    plan = proposal['trace']['planning']
    click, dispatched = inputs[0], actual[0]
    guard = row['native_upstream_input_preparation']
    if (event.get('accepted') is not False or proposal.get('has_action_intent') is not True
            or proposal.get('execution_authorized') is not False or plan.get('accepted') is not True
            or any(action != ui.get('action') for action in (event.get('action'), proposal.get('action'), plan.get('action')))
            or ui['action'].get('kind') != 'PLAY_CARD'
            or click.get('phase') != 'select' or click.get('dispatched') is not True
            or click.get('dispatch_index') != 0 or not click.get('job_id')
            or dispatched.get('kind', 'click') != 'click'
            or guard.get('dispatch_state') != 'dispatch_attempted'
            or any(click.get(key) != dispatched.get(key) or guard.get(key) != dispatched.get(key)
                for key in ('x', 'y'))
            or guard.get('authorization_floor') != click.get('captured_at')
            or guard.get('input_deadline') != click.get('input_deadline')
            or not click['captured_at'] <= guard['authorization_checked_at'] < click['input_deadline']):
        raise ValueError('upstream selection has unknown or contradictory input')
    for endpoints in (guard['endpoints'], dispatched['endpoints']):
        _verified_endpoints(endpoints, row['endpoints']['game']['pid'])
        _same_input_host(row['endpoints'], endpoints)
    return plan


def selected_read_without_input(row):
    from ..hif_live_recovery import closed_observer_record

    receipt = row.get('handoff_resume') or {}
    ui = row.get('upstream_ui_journal') or {}
    return (receipt.get('boundary_kind') == 'unsubmitted_selected' and closed_observer_record(row)
            and row.get('status') == 'STOP' and row.get('transaction_closed') is not True
            and row.get('input_count') == 0 and row.get('native_input_dispatches') == []
            and ui.get('inputs') == [] and not row.get('native_upstream_input_preparation')
            and (ui.get('action') or {}).get('kind') == 'PLAY_CARD'
            and not any(e.get('executed') or e.get('execution_submit_count') or e.get('input_may_have_been_dispatched')
                        for e in row.get('live_events', ())))


def original_selection(runtime, row):
    """Read past input-free observations to the SELECT which is still on screen."""
    from .native_settlement_resume import load_result

    current = row
    seen = set()
    while selected_read_without_input(current):
        if current['id'] in seen:
            raise ValueError('cyclic selected-card observation')
        seen.add(current['id'])
        previous = load_result(runtime, current['automatic_battle']['previous_record'])
        if (previous['handoff_id'] != current['handoff_id']
                or previous['upstream_ui_journal']['action'] != current['upstream_ui_journal']['action']
                or current['handoff_resume']['selection_source'] !=
                    (previous.get('handoff_resume') or {}).get('selection_source', previous['id'])):
            raise ValueError('selected-card observation lost its original action')
        current = previous
    selected_plan(current)
    return current


def is_unsubmitted_upstream_selection(row):
    try:
        selected_plan(row)
        return True
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return False


def load_selected_checkpoint(runtime, row, session, module, live, handoff):
    from .native_opening_root import load_native_round_context
    from .native_handoff_policy import phase, pending_action, restore_policy
    from .native_handoff_resume import _binding
    from .native_inprocess_resume import promote_verified_settlement
    from .native_settlement_resume import load_verified_source

    plan = selected_plan(row)
    bundle, state, event, parent = load_verified_source(runtime, row['native_inprocess_source'], module)
    _binding(parent, live, handoff)
    if (parent['endpoints']['game'] != row['endpoints']['game']
            or state.terminal or state.pending_choice or state.resolution_stack):
        raise ValueError('upstream selection parent is not the same complete business state')
    rebound = promote_verified_settlement(bundle, state, event)
    round_context = load_native_round_context(runtime, parent, module)
    policy, report = restore_policy(runtime, row, rebound, before=state, action=None,
        submitted=False, round_context=round_context, unsubmitted_plan=plan)
    action = pending_action(policy, state)
    if phase(policy) != 'pending_intent' or asdict(action) != row['upstream_ui_journal']['action']:
        raise ValueError('upstream selection changed its original computed action')
    # No selected coordinates or old timestamps are installed in the runtime.
    # The pinned UI checks SELECT/detail afresh and uses its already-selected path.
    return dict(bundle=rebound, state=state, event=event, row=row, settled_source=parent, policy=policy,
        cache_report=report, round_context=round_context, live=live, source_runtime=runtime,
        selected=False, pending=False, upstream_selected=True, action=action, expected=None,
        receipt=dict(version=1, source=parent['id'], selection_source=row['id'], handoff_id=handoff['handoff_id'],
            route_run_id=session.record['run_id'], ownership_revision=session.record['ownership']['revision'],
            ledger_revision=handoff['ledger_revision'], live_context=live,
            boundary_kind='unsubmitted_selected', fresh_support_required_before_submit=True))
