"""Read-only first-turn restart expectations from a verified native action chain."""

import re
import copy
import math
from types import SimpleNamespace
from dataclasses import asdict

from .native_card_evidence import input_window_limit, input_evidence_is_fresh, dispatched_card_refreshes
from .native_settlement_resume import exact_state, load_result, decoded_dispatched_action


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _one(values, reason):
    values = tuple(values)
    if len(values) != 1:
        raise ValueError(reason)
    return values[0]


def _first_turn(state):
    state.validate()
    if (state.turn != 1 or state.turns_remaining != 9 or state.pending_choice
            or state.pending_after_card_ids or state.resolution_stack
            or not state.persistent_effects_initialized):
        raise ValueError('restart source is not a settled first-turn card chain')


def _bound_title(observed, expected):
    from live_decision.history import normalize_history_text
    from live_decision.upgrade import normalize_upgrade_title

    observed = normalize_upgrade_title(normalize_history_text(observed))
    expected = normalize_upgrade_title(normalize_history_text(expected))
    # The accepted archive reads a leading kanji one as a long-vowel mark.
    # Permit only that position against an already independently bound card;
    # never rewrite effect text or arbitrary card names in the saved evidence.
    return observed == expected or (expected.startswith('一') and observed == 'ー' + expected[1:])


def _verified_card(row, bundle, baseline):
    from live_decision.high_score.contracts import ActionKind
    from live_decision.high_score.live_policy import BoundHifOpeningPolicy

    action, event = decoded_dispatched_action(row, bundle)
    if (action.kind is not ActionKind.PLAY_CARD or type(action.source_slot) is not int
            or not 0 <= action.source_slot < len(bundle.state.deck.hand)
            or bundle.state.deck.hand[action.source_slot].instance_id != action.card_instance_id):
        raise ValueError('restart source action is not a bound first-turn card')
    refreshes = dispatched_card_refreshes(event, row=row, bundle=bundle)
    for refresh, phase, input_phase in zip(refreshes, ('authoritative_refresh_a', 'authoritative_refresh_b'),
                                         ('execution_selection', 'execution_submit'), strict=True):
        if (refresh.get('accepted') is not True or refresh.get('phase') != phase
                or len(set(refresh.get('frame_ids', ()))) != 3
                or refresh.get('safe_decision', {}).get('status') != 'ALLOW'
                or refresh.get('proposal', {}).get('candidate_traces', {}).get('high_score', {}).get('action') != asdict(action)):
            raise ValueError('restart card original input authorization is incomplete')
        if not input_evidence_is_fresh(refresh) or refresh['capture_age_ms'] >= input_window_limit(refresh):
            raise ValueError('restart card original input evidence exceeded its timing limits')
        confirmation = _one((entry for entry in event.get('execution_selection_events', ())
                             if entry.get('phase') == input_phase and entry.get('click_dispatched') is True),
                            'restart card original dispatch is not unique')
        started, authorized, dispatched, deadline = (refresh.get('authoritative_capture_started_at'),
            refresh.get('authorization_floor'), confirmation.get('authorization_floor'), refresh.get('input_deadline'))
        if (not all(_number(value) for value in (started, authorized, dispatched, deadline))
                or not started <= authorized <= dispatched < deadline
                or not 0 < deadline - started <= input_window_limit(refresh)/1000. + 1e-9
                or confirmation.get('attempted_slot') != action.source_slot):
            raise ValueError('restart card dispatch was outside its original deadline')
        boxes = refresh.get('candidate_boxes', ())
        if len(boxes) != 3 or any(len(frame) <= action.source_slot for frame in boxes):
            raise ValueError('restart card original target geometry is absent')
        x, y, width, height = boxes[-1][action.source_slot]
        _one((entry for entry in row['native_input_dispatches']
              if entry.get('kind') != 'swipe' and (entry.get('x'), entry.get('y')) ==
              (x + width // 2, y + height // 2)), 'restart card has no unique audited target input')
    policy = BoundHifOpeningPolicy(bundle, action_order=baseline['core']['fields'][5])
    policy.response = SimpleNamespace(action=action)
    policy.bind_generated_card_resource_preview(refreshes[-1]['resource_transaction'],
                                                refreshes[-1]['visual_fields']['score'])
    outcomes = policy.committed_transition()
    outcomes = outcomes if isinstance(outcomes, tuple) else (outcomes,)
    for state in (bundle.state, *outcomes):
        _first_turn(state)
    title = bundle.kernel.cards[bundle.state.deck.hand[action.source_slot].card_id].name
    return action, outcomes, title


def _pending_numeric(before, candidates):
    # Payment may be absent from game history. The fresh core independently
    # verifies stamina/shield; this map covers visible Sense effect counters.
    names = {'concentration': '集中', 'good_condition_turns': '好調',
             'excellent_condition_turns': '絶好調', 'motivation': 'やる気'}
    result = []
    for field, label in names.items():
        old = getattr(before, field, None)
        values = {getattr(state, field, None) for state in candidates}
        if len(values) != 1:
            raise ValueError('restart pending visible effects differ across outcomes')
        new, = values
        if old != new:
            if type(old) is not int or type(new) is not int or min(old, new) < 0:
                raise ValueError('restart pending effect has no supported numeric log representation')
            result.append(f'{label}{old}→{new}')
    if not result:
        raise ValueError('restart pending card has no supported visible numeric effects')
    return result


def collect_restart_source(runtime, pending_name, module):
    """Verify every archived transition, then return comparison-only inputs.

    Result keys are ``bundle``, ``candidates``, ``baseline``, ``row``, ``action``,
    ``old_run``, ``old_pid`` and ``expected_history``. No current-process binding
    or input permission is produced. The final dispatch must still be pending.
    """
    from live_decision.history import normalize_history_text, restore_verified_history
    from live_decision.high_score.opening import assemble_hif_opening, verify_hif_committed_observation

    seen = set()
    sources, titles, groups = [], [], []
    initial_groups = []

    def read(name):
        if name in seen or len(seen) >= 120:
            raise ValueError('restart native source chain is cyclic, duplicated or unbounded')
        seen.add(name)
        return load_result(runtime, name)

    def run_for(row):
        return module.HifLiveRunContext(**row['observed_live_context'])

    def dispatch(row):
        nonlocal initial_groups
        if row.get('native_pending_source'):
            raise ValueError('restart dispatch points to another pending completion')
        opening = [event for event in row['live_events']
                   if event.get('event') == 'live_opening_state' and event.get('accepted') is True]
        parent_name = row.get('native_resume_source')
        if opening:
            if parent_name or sources:
                raise ValueError('restart chain contains an unexpected second opening')
            baseline = _one(opening, 'restart source requires one complete opening')
            bundle = assemble_hif_opening(run_for(row), core=baseline['core'], components=baseline['components'],
                hand_titles=baseline['hand_titles'], data_root=module.CATALOG_PATH.parent)
            initial_groups = [dict(kind='system', title='メモリーアビリティ',
                lines=['メモリーアビリティ', 'メモリーアビリティが付与']) for _ in bundle.state.memory_abilities]
        else:
            if not parent_name:
                raise ValueError('restart native source chain has no complete opening')
            bundle, state, previous, parent = complete(read(parent_name))
            bundle, _, baseline = restore_verified_history(bundle, state, settlement=previous,
                reference=row['resume_reference'], current=row['resume_history'],
                old_run=run_for(parent), new_run=run_for(row), old_pid=parent['endpoints']['game']['pid'],
                new_pid=row['endpoints']['game']['pid'])
            lines = row['resume_history']['observation']['lines']
            match = re.fullmatch(r'スキルカード「(.+)」', normalize_history_text(lines[0]))
            if len(lines) < 2 or not match or not _bound_title(match[1], titles[-1]):
                raise ValueError('restart native history does not identify its preceding dispatch')
            titles[-1] = match[1]
            groups.append(copy.deepcopy(lines))
        action, outcomes, title = _verified_card(row, bundle, baseline)
        if row['id'] in sources:
            raise ValueError('restart native dispatch was replayed in the source chain')
        sources.append(row['id'])
        titles.append(title)
        return bundle, outcomes, baseline, action

    def complete(row):
        if row.get('live_return') is not True:
            raise ValueError('restart source completion did not return successfully')
        pending = row.get('native_pending_source')
        if pending:
            if row.get('native_resume_source'):
                raise ValueError('restart completion has conflicting source links')
            original = read(pending)
            bundle, outcomes, baseline, action = dispatch(original)
            if (row.get('execution_live_context') != original['observed_live_context']
                    or row['endpoints']['game']['pid'] != original['endpoints']['game']['pid']
                    or any(row['observed_live_context'][key] != original['observed_live_context'][key]
                           for key in ('idol', 'plan', 'max_stamina'))
                    or any(event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
                           for event in row['live_events'])):
                raise ValueError('restart pending completion is not a zero-submit continuation')
            row = dict(row, observed_live_context=row['execution_live_context'])
        else:
            bundle, outcomes, baseline, action = dispatch(row)
        settlement = _one((event for event in row['live_events']
            if event.get('event') == 'live_committed_state' and event.get('accepted') is True),
            'restart source has no unique accepted complete settlement')
        run = run_for(row)
        if (settlement.get('run_id') != run.run_id or settlement.get('revision') != run.revision
                or not settlement.get('checks') or not all(value is True for value in settlement['checks'].values())):
            raise ValueError('restart source settlement binding or checks are incomplete')
        state = exact_state(settlement['expected_state'])
        if state not in outcomes:
            raise ValueError('restart recorded state is not an outcome of its actual dispatched action')
        comparison = verify_hif_committed_observation(bundle, state, core=settlement['core'],
            components=settlement['components'], baseline_components=baseline['components'])
        if not comparison['accepted']:
            raise ValueError('restart recorded components do not prove the dispatched outcome')
        return bundle, state, settlement, row

    row = read(pending_name)
    if any(event.get('event') == 'live_committed_state' and event.get('accepted') is True
           for event in row['live_events']):
        raise ValueError('restart target action is already completely settled')
    bundle, candidates, baseline, action = dispatch(row)
    if len(groups) + 1 != len(sources):
        raise ValueError('restart history prefix omits a dispatched action')
    expected = dict(initial_groups=initial_groups, prefix_groups=groups, action_titles=titles,
                    source_records=sources, pending_numeric_lines=_pending_numeric(bundle.state, candidates))
    return dict(bundle=bundle, candidates=candidates, baseline=baseline, row=row, action=action,
                old_run=run_for(row), old_pid=row['endpoints']['game']['pid'], expected_history=expected)
