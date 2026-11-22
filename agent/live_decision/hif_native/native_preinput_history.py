"""Rebuild mutable preinput retention from an independently restored baseline."""

import copy
import json
from dataclasses import asdict

_SOURCE = 'retained_preinput_no_business_change'
_KEYS = {'cards', 'status', 'items', 'drinks', 'turns'}


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def sync_published_settlement_metadata(events, complete, published):
    """Keep the actual post-publication MCTS report without changing evidence."""
    saved = [index for index, event in enumerate(events)
             if event.get('event') == 'live_committed_state' and event.get('accepted') is True]
    if len(saved) != 1:
        raise ValueError('native continuation history lacks its original complete event')
    index = saved[0]
    original = events[index]
    if _same(original, complete):
        return
    # The component sink keeps only this actual object reference. All original
    # fields must still match the existing published JSON snapshot exactly;
    # no business field, removed key, or other added metadata is ignored.
    if (len(published) != 1 or published[0] is not complete
            or 'mcts_commit' in original or not isinstance(complete.get('mcts_commit'), dict)):
        raise ValueError('native continuation settlement publication differs')
    updated = dict(original, mcts_commit=copy.deepcopy(complete['mcts_commit']))
    if not _same(updated, complete):
        raise ValueError('native continuation settlement changed after publication')
    events[index] = updated


def bind_settlement_history(bundle, settlement, history, *, run, game_pid):
    """Attach one observed post-action history without relabeling its ledger."""
    from live_decision.history import restore_verified_history
    from live_decision.high_score.maintenance import capture_preinput_continuity

    from .native_settlement_resume import exact_state

    if (settlement.get('event') != 'live_committed_state'
            or settlement.get('strategy_state_bound') is not True
            or 'history_observation' in settlement):
        raise ValueError('native continuation history requires one original complete settlement')
    # The live event still contains str-enum instances from asdict(); the
    # existing native parser consumes persisted JSON values, not Enum.__str__.
    expected = exact_state(json.loads(json.dumps(settlement['expected_state'])))
    rebound, _, _ = restore_verified_history(bundle, expected, settlement=settlement,
        reference=history.get('observation', {}), current=history,
        old_run=run, new_run=run, old_pid=game_pid, new_pid=game_pid)
    attached = dict(settlement, history_observation=copy.deepcopy(history))
    # The existing preinput source check additionally requires complete
    # three-frame history/before/after groups and zero business input.
    if capture_preinput_continuity(rebound, attached).source_history is None:
        raise ValueError('native continuation history lacks complete unchanged source evidence')
    return attached


def verify_settlement_history(row, bundle, settlement, *, run):
    """Rebuild attached history only after the caller rechecked the settlement."""
    if 'history_observation' not in settlement:
        return settlement
    history = row.get('post_history')
    if not isinstance(history, dict) or not _same(history, settlement['history_observation']):
        raise ValueError('native continuation history differs from its original post-action record')
    original = {key: value for key, value in settlement.items() if key != 'history_observation'}
    verified = bind_settlement_history(bundle, original, history, run=run,
        game_pid=row['endpoints']['game']['pid'])
    if not _same(verified, settlement):
        raise ValueError('native continuation history differs from its independently verified binding')
    return verified


def verify_preinput_history(row, bundle, baseline):
    """Check saved five-component reuse; leave the legacy three-component path alone.

    Old frames prove the recorded transaction only. This never makes an
    archived history or its core frames into fresh input credentials.
    """
    from live_decision.high_score.opening import verify_hif_committed_observation
    from live_decision.high_score.maintenance import (
        bind_preinput_history,
        capture_preinput_continuity,
        preinput_maintained_components,
    )

    events = [event for event in row.get('live_events', ())
              if event.get('event') == 'live_preinput_state']
    relevant = [event for event in events
        if 'preinput_history_observation' in event
        or {'status', 'drinks'} & set(event.get('preinput_retained_components', ()))
        or any(event.get('components', {}).get(key, {}).get('observation_source') == _SOURCE
               for key in ('status', 'drinks'))]
    if not relevant:
        return False
    if len(events) != 1 or len(relevant) != 1:
        raise ValueError('native preinput history has no unique complete revalidation')
    event = relevant[0]
    initial, core = event.get('initial_core', {}), event.get('core', {})
    if (event.get('accepted') is not True or event.get('strategy_state_bound') is not True
            or (event.get('run_id'), event.get('revision')) != (bundle.run_id, bundle.revision)
            or not _same(event.get('expected_state'), asdict(bundle.state))
            or event.get('preinput_retained_components') != sorted(_KEYS)
            or not event.get('checks') or not all(value is True for value in event['checks'].values())
            or any(type(event.get(key, 0)) is not int or event.get(key, 0) != 0
                   for key in ('execution_submit_count', 'end_turn_submit_count', 'drinks_consumed'))
            or any(not _same(initial.get(key), core.get(key)) for key in ('fields', 'families'))
            or any(not view.get('frame_ids')
                   or any(not isinstance(fid, str) or not fid for fid in view['frame_ids'])
                   or len(set(view['frame_ids'])) != len(view['frame_ids']) for view in (initial, core))):
        raise ValueError('native preinput history is incomplete or differs from its restored state')
    proof = capture_preinput_continuity(bundle, baseline)
    proof = bind_preinput_history(bundle, bundle.state, baseline=baseline, proof=proof,
        current=event.get('preinput_history_observation'), core=initial)
    retained = preinput_maintained_components(bundle, bundle.state,
        baseline=baseline, proof=proof, core=initial)
    if set(retained) != _KEYS or not _same(event.get('components'), retained):
        raise ValueError('native preinput history components differ from their original ledger')
    comparison = verify_hif_committed_observation(bundle, bundle.state,
        core=core, components=retained, baseline_components=baseline['components'],
        preinput_continuity=proof)
    if (comparison.get('accepted') is not True
            or not _same(event.get('observed_hand_instance_ids'), comparison['observed_hand_instance_ids'])):
        raise ValueError('native preinput history failed the complete business comparison')
    return True
