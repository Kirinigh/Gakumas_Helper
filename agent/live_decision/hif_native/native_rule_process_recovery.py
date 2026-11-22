"""Rebind an already submitted card after restart using fresh history and core.

The logical game epoch stays unchanged. Current process identity and the new
observations are recorded separately; this never creates input credentials.
"""
import re
from dataclasses import asdict


def process_recovery(row, source, bundle, action, settlement):
    from route.engine import has_single_frame_id
    from live_decision.history import normalize_history_text
    from live_decision.upgrade import normalize_upgrade_title
    from live_decision.high_score.restart import _title, _frame_sets
    from live_decision.high_score.contracts import ActionKind

    from .native_settlement_resume import exact_state, _verified_endpoints
    from .native_rule_progress_resume import _same, is_rule_progress

    old_pid = source['endpoints']['game']['pid']
    new_pid = row['endpoints']['game']['pid']
    identity = row.get('context_recovery', {})
    actual = identity.get('context', {})
    original = source.get('observed_live_context', {})
    _verified_endpoints({key: row['endpoints'][role] for key, role in
        (('game', 'game'), ('agent', 'agent'), ('maa', 'maapicli'))}, new_pid)
    if (old_pid == new_pid or row.get('information_boundary_execution') is not True
            or identity.get('accepted') is not True or identity.get('returned_to_live') is not True
            or identity.get('game_pid') != new_pid or actual == original
            or any(actual.get(key) != original.get(key) for key in ('idol', 'plan', 'max_stamina'))
            or action.kind is not ActionKind.PLAY_CARD or not is_rule_progress(settlement)
            or settlement.get('accepted') is not True
            or settlement.get('execution_submit_count') != 0
            or settlement['rule_progress'].get('observation_mode') == 'deterministic_hand_removal'):
        raise ValueError('rule process recovery lacks fresh identity and original pending card')
    after = exact_state(settlement['expected_state'])
    changed_turn = (after.turn == bundle.state.turn + 1
                    and after.turns_remaining == bundle.state.turns_remaining - 1)
    if not changed_turn and (after.turn != bundle.state.turn
                            or after.turns_remaining != bundle.state.turns_remaining):
        raise ValueError('rule process recovery card changed more than one turn')
    history = row.get('pending_restart_history', {})
    if history.get('accepted') is not True or history.get('returned_unchanged') is not True:
        raise ValueError('rule process recovery lacks complete latest history')
    history_frames = history.get('frame_ids')
    if not has_single_frame_id(history_frames):
        raise ValueError('rule process recovery requires one current history frame')
    frames = set(_frame_sets(source)) | set(settlement['core']['frame_ids'])
    if history_frames[0] in frames:
        raise ValueError('rule process recovery history frames are reused')
    frames.add(history_frames[0])
    for checkpoint in ('before', 'after'):
        core = history.get(checkpoint, {})
        core_frames = core.get('frame_ids')
        if (tuple(core.get('fields', ())[:5]) != tuple(settlement['core']['fields'][:5])
                or not _same(core.get('families'), settlement['core']['families'])
                or not has_single_frame_id(core_frames) or core_frames[0] in frames):
            raise ValueError('rule process recovery Live changed during history')
        frames.add(core_frames[0])
    observation = history.get('observation', {})
    lines = [normalize_history_text(line) for line in observation.get('lines', ())]
    refs = [ref for ref in bundle.state.deck.hand if ref.instance_id == action.card_instance_id]
    if len(refs) != 1 or len(lines) < 2 or observation.get('score') != after.score:
        raise ValueError('rule process recovery history has no bound pending action')
    if changed_turn:
        from .native_pending_restart import _complete_turn_tail

        # A last action advances the turn through the same kernel transition
        # as END. The newest history group is then the turn-start effects,
        # not the previous card. Reuse the existing complete-turn witness;
        # original candidate/state validation and fresh unchanged core remain.
        _complete_turn_tail(history, after.turn)
        return dict(scope='pending_card_restart_turn_history_v1', source=source['id'],
            old_pid=old_pid, new_pid=new_pid, logical_context=original, current_context=actual,
            action=asdict(action), history_frame_ids=list(history['frame_ids']),
            core_frame_ids=list(settlement['core']['frame_ids']),
            turn_start_witness=lines, zero_new_business_inputs=True)
    title = normalize_upgrade_title(normalize_history_text(bundle.kernel.cards[refs[0].card_id].name))
    if _title(lines[0], {title}) != title:
        raise ValueError('rule process recovery latest card differs')
    # Game history lists effect changes after paying the card's declared
    # resource cost. Reuse the actual bound card's payment rules, not the
    # displayed line's starting value or a guessed duration adjustment.
    definition = bundle.kernel.card_definition(bundle.state, refs[0])
    paid, costs = bundle.kernel._pay_declared_non_full_power_cost(bundle.state, refs[0], definition)
    numeric = []
    for field, label in (('concentration', '集中'), ('good_condition_turns', '好調'),
                         ('excellent_condition_turns', '絶好調'), ('motivation', 'やる気')):
        before_value, after_value = getattr(paid, field), getattr(after, field)
        if before_value != after_value:
            transitions = [tuple(map(int, match.groups())) for line in lines
                if (match := re.fullmatch(label+r'(\d+)→(\d+)', line))]
            value = before_value
            for old, new in transitions:
                if old != value:
                    raise ValueError('rule process recovery history effect chain differs: '+label)
                value = new
            if not transitions or value != after_value:
                raise ValueError('rule process recovery history effect differs: '+label)
            numeric.extend(f'{label}{old}→{new}' for old, new in transitions)
    if not numeric:
        raise ValueError('rule process recovery has no supported independent effect witness')
    return dict(scope='pending_card_restart_rule_history_v1', source=source['id'],
        old_pid=old_pid, new_pid=new_pid, logical_context=original, current_context=actual,
        action=asdict(action), history_frame_ids=list(history['frame_ids']),
        core_frame_ids=list(settlement['core']['frame_ids']), numeric_witnesses=numeric,
        cost_events=[asdict(cost) for cost in costs],
        zero_new_business_inputs=True)


def verify_process_recovery(row, source, bundle, action, settlement):
    from .native_rule_progress_resume import _same
    rebuilt = process_recovery(row, source, bundle, action, settlement)
    if not _same(row.get('pending_process_recovery'), rebuilt):
        raise ValueError('rule process recovery differs from original evidence')
    return rebuilt
