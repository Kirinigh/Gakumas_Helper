"""Keep the verified original handback after post-battle ledger recovery.

The origin below is historical evidence in the existing handoff, never the
current ledger. A prepared report remains unchanged on disk and in memory.
"""
from __future__ import annotations

import re
import json
from copy import deepcopy
from pathlib import Path

SCOPE = ('handoff_id', 'to_owner', 'round_number', 'policy_profile', 'idol', 'plan',
    'max_stamina', 'source_frame_ids', 'prior_terminal_source', 'ledger_revision',
    'live_context', 'terminal_receipt', 'terminal_source')


def verify_bound_terminal_return(*, run_id, ownership, handoff, report):
    from live_decision.hif_battle_entry import verify_prepared_terminal_return
    binding = (handoff or {}).get('terminal_return_origin')
    if not isinstance(binding, dict) or binding.get('schema_version') != 1:
        raise ValueError('prepared terminal return has no verified original handback binding')
    origin = binding.get('record')
    if not isinstance(origin, dict) or not isinstance(ownership, dict) or not isinstance(handoff, dict):
        raise ValueError('terminal handback origin is incomplete')
    original_owner = origin.get('ownership') or {}
    if (origin.get('run_id') != run_id or ownership.get('owner') != 'task600'
            or ownership.get('phase') not in ('recovery', 'nonbattle')
            or ownership.get('game_pid') != original_owner.get('game_pid')
            or type(ownership.get('revision')) is not int
            or ownership['revision'] not in (original_owner.get('revision'), original_owner.get('revision', -2) + 1)
            or any(handoff.get(key) != origin.get('handoff', {}).get(key) for key in SCOPE)):
        raise ValueError('current terminal scope differs from its original verified handback')
    proof = verify_prepared_terminal_return(None, origin, report)
    if proof != binding.get('proof'):
        raise ValueError('original terminal handback proof has changed')
    return proof


def bind_terminal_return(record, runtime):
    """Validate a published source, then return one atomic checkpoint candidate."""
    from live_decision.hif_battle_entry import load_prepared_terminal_return
    handoff = record.get('handoff') or {}
    source = handoff.get('terminal_source')
    if source is None:
        return record, None
    if not isinstance(source, str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', source) is None:
        raise ValueError('current terminal source identity is invalid')
    candidate = deepcopy(record)
    if handoff.get('terminal_return_origin') is None:
        report, proof = load_prepared_terminal_return(runtime, record)
        origin = {key: deepcopy(record[key]) for key in ('schema_version', 'run_id', 'ownership', 'transition', 'ledger')}
        origin['handoff'] = {key: deepcopy(handoff[key]) for key in SCOPE if key in handoff}
        candidate['handoff']['terminal_return_origin'] = dict(schema_version=1, record=origin, proof=proof)
    else:
        report = json.loads((Path(runtime) / f'result-{source}.json').read_text(encoding='utf-8'))
    verify_bound_terminal_return(run_id=record['run_id'], ownership=record['ownership'],
        handoff=candidate['handoff'], report=report)
    origin_revision = candidate['handoff']['terminal_return_origin']['record']['ledger']['state']['revision']
    if (record['ledger']['state']['run_id'] != record['run_id']
            or record['ledger']['state']['revision'] < origin_revision):
        raise ValueError('current ledger regressed before its verified terminal origin')
    return candidate, report
