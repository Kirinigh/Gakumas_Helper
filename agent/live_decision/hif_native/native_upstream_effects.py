"""Reconcile a missed selected effect before an already dispatched turn end.

The original card receipt and END remain unchanged. A supplemental extraction
from that card's actual SELECT frames is replayed through the existing rules.
It supplies candidate states, never a successful settlement or input permit.
"""
import json
from types import SimpleNamespace
from dataclasses import asdict


def reconcile_pending_end(runtime, row, module, bundle, expected):
    from ..hif_upstream_support import bind_recorded_support
    from ..high_score.contracts import PlanType, ActionKind
    from ..high_score.live_policy import BoundHifOpeningPolicy
    from .native_settlement_resume import load_pending_source, load_verified_source, decoded_dispatched_action

    end, _ = decoded_dispatched_action(row, bundle)
    source = row.get('native_rule_resume_source')
    if end.kind is not ActionKind.END_TURN or not source:
        return expected, None
    path = runtime / ('upstream-support-' + source + '.json')
    if not path.is_file():
        return expected, None
    extraction = json.loads(path.read_text(encoding='utf8'))
    if extraction.get('version') != 1 or extraction.get('source') != source:
        raise ValueError('upstream supplementary effects source changed')
    before, ordinary, baseline, card_row = load_pending_source(runtime, source, module,
        allow_closed_transaction=True)
    _, original_after, _, _ = load_verified_source(runtime, source, module)
    canonical = bundle.kernel._canonical_probability_state
    if (canonical(original_after) != canonical(bundle.state)
            or before.run_id != bundle.run_id or before.revision != bundle.revision
            or card_row['endpoints']['game']['pid'] != row['endpoints']['game']['pid']):
        raise ValueError('upstream supplementary effects are not the immediate maintained predecessor')
    action, _ = decoded_dispatched_action(card_row, before)
    journal = card_row.get('upstream_ui_journal', {})
    receipt = extraction['receipt']
    if journal.get('selected_skill_support') is not None:
        raise ValueError('upstream supplementary effects cannot replace a previously bound effect')
    confirmation, = [entry for entry in journal['inputs'] if entry['phase'] == 'confirm']
    frames = card_row['frames']
    indices = [index for index, frame in enumerate(frames) if frame['frame_id'] == confirmation['frame_id']]
    if (len(indices) != 1 or indices[0] < 1
            or [frame['frame_id'] for frame in frames[indices[0]-1:indices[0]+1]] != receipt['frame_ids']
            or any(frame['label'] != 'ProduceRecognitionCards'
                   for frame in frames[indices[0]-1:indices[0]+1])
            or journal.get('response_confirmed') is not True or journal.get('playable') is not True):
        raise ValueError('upstream supplementary effects lack the actual selected detail pair')
    policy = BoundHifOpeningPolicy(before, action_order=baseline['core']['fields'][5])
    policy.response = SimpleNamespace(action=action)
    bind_recorded_support(policy, dict(journal, selected_skill_support=receipt,
        selected_detail=receipt['selection_evidence']))
    corrected = policy.committed_transition()
    if isinstance(corrected, tuple) or corrected.turn != bundle.state.turn:
        raise ValueError('upstream supplementary effect has an unresolved card successor')
    branches = bundle.kernel.transition(corrected, end, plan_type=PlanType.SENSE)
    if not branches or abs(sum(branch.probability for branch in branches)-1) > 1e-9:
        raise ValueError('upstream supplementary end outcomes are incomplete')
    return tuple(branch.state for branch in branches), dict(extraction,
        original_card_successor=asdict(original_after), corrected_card_successor=asdict(corrected),
        original_turn_candidates=[asdict(s) for s in (expected if isinstance(expected, tuple) else (expected,))],
        reason='original_selected_detail_replayed_before_dispatched_end', input_authorized=False)
