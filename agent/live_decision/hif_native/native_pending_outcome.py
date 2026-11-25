"""A complete unchanged observation can retire a failed dispatch attempt once.

The attempt remains in history. This is not a committed card, a new search, or
permission to reuse old UI coordinates; the ordinary input path runs again.
"""
from dataclasses import asdict

from ..high_score.hif_mcts_cache import same_json, json_value


def _same_value(left, right):
    return same_json(json_value(left), json_value(right))


def is_unconsumed_observation(row):
    event = row.get('settlement', {})
    return (row.get('pending_dispatch_outcome') == 'not_consumed'
        and event.get('accepted') is True and event.get('pending_outcome') == 'before')


def verify_unconsumed(bundle, expected, baseline, action, event, single_plus_estimate=None):
    from ..high_score.opening import resolve_hif_pending_observation

    core_only = event.get('observation_mode') == 'pending_core_comparison'
    if core_only:
        from .native_rule_progress_resume import source_receipt
        from ..high_score.hif_rule_progress import resolve_rule_pending

        state, proof = resolve_rule_pending(bundle, expected, baseline_event=source_receipt(baseline),
            action=action, initial_core=event['initial_core'], core=event['core'],
            single_plus_estimate=single_plus_estimate)
        if event.get('force_fresh_components') is not False or event.get('components'):
            raise ValueError('pending core comparison cannot claim complete fresh components')
    else:
        state, proof = resolve_hif_pending_observation(bundle, expected, core=event['core'],
            components=event['components'], baseline_components=baseline['components'],
            committed_action=action, single_plus_estimate=single_plus_estimate)
    if (state is None or proof.get('pending_outcome') != 'before'
            or event.get('accepted') is not True or (not core_only and event.get('force_fresh_components') is not True)
            or (event.get('run_id'), event.get('revision')) != (bundle.run_id, bundle.revision)
            or any(not _same_value(event.get('initial_core', {}).get(k), event['core'].get(k))
                   for k in ('fields', 'families'))
            or not _same_value(event.get('expected_state'), asdict(state))
            or any(not _same_value(event.get(k), v) for k, v in proof.items())):
        raise ValueError('pending unchanged observation does not independently prove non-consumption')
    return state


def retry_index(dispatch_source):
    receipt = dispatch_source.get('native_unconsumed_retry')
    if receipt is None:
        return 0
    if (receipt.get('version') != 1 or receipt.get('retry_index') not in (1, 2, 3)
            or not receipt.get('observation_source') or not receipt.get('dispatch_source')):
        raise ValueError('pending retry accounting is incomplete')
    return receipt['retry_index']


def prepare_unconsumed_retry(checkpoint, result, runtime, *, recovery=None):
    """Rebind after proof, or choose Hiro's SKIP on the third observed miss."""
    from .native_handoff_policy import restore_policy
    from .native_inprocess_resume import promote_verified_settlement

    if checkpoint.get('pending') is not True or not is_unconsumed_observation(result):
        raise ValueError('pending retry requires a freshly reconciled dispatched checkpoint')
    source = checkpoint['dispatch_source']
    index = retry_index(source)
    if index >= (3 if recovery is not None else 1):
        return None
    event, bundle = result['settlement'], checkpoint['bundle']
    state = verify_unconsumed(bundle, checkpoint['expected'], checkpoint['event'], checkpoint['action'],
        event, checkpoint['row'].get('verified_single_plus_estimate'))
    rebound = promote_verified_settlement(bundle, state, event)
    policy = checkpoint['policy']
    report = checkpoint['cache_report']
    if policy is None:
        policy, report = restore_policy(runtime, checkpoint['row'], rebound,
            unconsumed_source=source, round_context=checkpoint['round_context'])
    # Fast already owns the exact cached pending intent. A strong deferred
    # cache is restored above only after the complete unchanged observation.
    from .native_handoff_policy import phase
    if phase(policy) != 'pending_intent' or policy._pending is None:
        raise ValueError('unconsumed retry lost its completed original intention')
    forced = recovery is not None and recovery.record(no_effect=True)
    retry = dict(version=1, retry_index=index + 1, observation_source=result['id'], dispatch_source=source['id'])
    return dict(checkpoint, bundle=rebound, state=rebound.state, event=event, row=result,
        policy=policy, cache_report=report, pending=False, reconciled_unconsumed=True,
        unsubmitted_intent=True, unconsumed_retry=retry, forced_skip=forced,
        receipt=dict(checkpoint['receipt'], source=result['id'], boundary_kind='not_consumed'))
