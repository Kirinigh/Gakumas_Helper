"""Time-only continuation for the already-paid November 18 opening."""
from copy import deepcopy

RUN_ID = 'hif-018299da474c5b5b940d44e8242e448b'
HANDOFF = f'{RUN_ID}:r1:ledger-36720076'
BOUNDARY = 'joint_opening_time'


def verify_opening(record, source):
    from route.new_run import JOINT_VALIDATION_AUTHORIZATION, start_record, validate_additional_start

    from .hif_navigation_recovery import verify_unstarted_notebook_source

    handoff = record['handoff']
    start = start_record(handoff)
    owner = record['ownership']
    if (record['run_id'] != RUN_ID or handoff.get('handoff_id') != HANDOFF or handoff.get('round_number') != 1
            or owner.get('owner') != 'task275' or owner.get('phase') != 'battle'
            or record['transition'].get('pending') is not None or record['transition'].get('stopped') is not False
            or record['ledger'].get('pending') is not None or record['ledger']['state'].get('revision') != handoff['ledger_revision']
            or not isinstance(start, dict) or start.get('status') != 'complete' or start.get('run_id') != RUN_ID
            or (start.get('ap_receipt') or {}).get('submitted_ap_cost') != 20
            or (start.get('additional_start') or {}).get('authorization_id') != JOINT_VALIDATION_AUTHORIZATION):
        raise ValueError('time-only continuation requires the original paid first-round opening')
    validate_additional_start(start['additional_start'], run_id=RUN_ID, previous_run_id=start['previous_run_id'],
        status='consumed')
    verify_unstarted_notebook_source(record, source)
    control = source['automatic_battle']
    if control.get('opening_parent_window') is not None or control['windows_used'] >= control['maximum_actions']:
        raise ValueError('time-only opening requires remaining original actions')
    return dict(version=1, source=source['id'], boundary=BOUNDARY, identity=deepcopy(source['automatic_battle']['identity']),
        source_control=deepcopy(source['automatic_battle']), live_context=None)


def bind_opening_time(control, latest, record, *, parent_deadline, authorise):
    import time

    from route.cultivation import _budget_container

    from .hif_battle_engineering_window import append_opening_parent

    if record.get('run_id') != RUN_ID or control.get('opening_parent_window') is not None:
        return control
    budget = (_budget_container(record['handoff']) or {}).get('automatic_cultivation') or {}
    windows = budget.get('execution_windows', ())
    proof = (windows[-1].get('battle_resume') or {}) if windows else {}
    if proof.get('boundary') != BOUNDARY:
        return control
    if proof != verify_opening(record, latest) or not callable(authorise):
        raise ValueError('time-only opening differs from its retained parent source')
    grant = authorise(source=latest, record=record)
    return append_opening_parent(control, latest, grant, now=time.time(), parent_deadline=parent_deadline)


def invocation_timeout(control, original_seconds, *, now):
    """Carry the signed-off absolute cutoff through the same observer invocation."""
    if not isinstance(control, dict):
        return original_seconds
    receipt = control.get('opening_parent_window') or {}
    grant = receipt.get('grant') or {}
    proof = (grant.get('parent_execution_window') or {}).get('battle_resume') or {}
    if proof.get('boundary') != BOUNDARY:
        return original_seconds
    from .hif_live_recovery import require_deadline
    from .hif_battle_engineering_window import window_limits

    require_deadline(control, now=now)
    return min(window_limits(control)[0], control['active_parent_deadline']) - now
