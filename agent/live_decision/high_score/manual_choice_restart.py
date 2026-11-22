"""Recover this first-turn, user-completed black-vinegar choice without input.

The user report constrains possible outcomes; only a complete fresh history and
five fresh components certify the result. A manual choice never becomes a Maa
dispatch or a synthetic observed physical slot.
"""

import re
import copy
from collections import Counter
from dataclasses import asdict, replace, dataclass

from .choice import _valid_branches, verify_hif_pending_choice
from .opening import expected_permanent_inventory_keys, resolve_hif_committed_observation
from .restart import _core, _same, _epoch, _lines, _frames, _frame_sets, _reject_retained, _bound_history_line
from ..history import normalize_history_text
from ..upgrade import normalize_upgrade_title
from .contracts import PlanType, ActionKind, DecisionKind
from ..status_panel import _normalize_status_text


@dataclass(frozen=True)
class HifManualChoiceRestart:
    comparison_bundle: object
    candidates: tuple
    action: object
    old_run: object
    new_run: object
    old_pid: int
    new_pid: int
    history: dict
    expected_history: dict
    manual_action: dict


def manual_choice_outcomes(projected, manual_action):
    """Enumerate every legal matching instance; do not invent a clicked slot."""
    if manual_action != dict(source='external_manual', reported_by='user',
                             drink_id='22', card_id='23', reported_completed=True):
        raise ValueError('manual retrieval requires the explicit completed user report')
    verified = verify_hif_pending_choice(projected.source_bundle, projected.action, projected.pending_state)
    if verified.action.drink_id != '22' or verified.source_bundle.state.deck.hand:
        raise ValueError('manual retrieval is outside the verified empty-hand black-vinegar case')
    refs = {ref.instance_id: ref for ref in verified.pending_state.deck.all_cards}
    kernel = verified.source_bundle.kernel
    actions = tuple(action for action in kernel.legal_actions(verified.pending_state,
        plan_type=PlanType.SENSE, decision_kind=DecisionKind.SECONDARY)
        if action.kind is ActionKind.MOVE_CARD
        and (kernel.cards[refs[action.card_instance_id].card_id].canonical_card_id
             or refs[action.card_instance_id].card_id) == manual_action['card_id'])
    if not actions:
        raise ValueError('reported manual card is not a legal pending choice')
    states = []
    for action in actions:
        branches = kernel.transition(verified.pending_state, action, plan_type=PlanType.SENSE)
        _valid_branches(branches)
        for branch in branches:
            state = branch.state
            if (state.pending_choice or state.resolution_stack or state.pending_after_card_ids
                    or state.turn != 1 or state.turns_remaining != 9 or state.actions_remaining != 1
                    or len(state.deck.hand) != 1 or state.drinks != ('2', '21')):
                raise ValueError('manual retrieval outcome is outside the supported first-turn boundary')
            states.append(state)
    return tuple(dict.fromkeys(states)), actions[0]


def _verify_manual_history(bundle, candidates, history, expected, new_run):
    if (history.get('accepted') is not True or history.get('returned_unchanged') is not True
            or history.get('complete_from_start') is not True or history.get('complete_to_end') is not True
            or any(history.get(key) != value for key, value in _epoch(new_run).items())):
        raise ValueError('manual retrieval requires fresh complete first-turn history')
    _frames(history.get('frame_ids'))
    before, after = _core(history.get('before')), _core(history.get('after'))
    if any(not _same(before[key], after[key]) for key in ('fields', 'families')):
        raise ValueError('manual retrieval Live changed during history inspection')
    observation = history.get('observation', {})
    if (observation.get('at_top') is not True or observation.get('at_bottom') is not True
            or observation.get('turn') != 1 or tuple(observation.get('turns', ())) != (1,)
            or observation.get('first_turn_observed') is not True
            or observation.get('score') != after['fields'][3]):
        raise ValueError('manual retrieval history boundaries or score disagree')
    groups = tuple(observation.get('groups', ()))
    # 878 reached the actually loaded beginning, earlier than the archived
    # memory-grant suffix. Preserve that archive; prove the newly visible exact
    # prelude against the independently assembled original opening counters.
    opening = expected.get('opening_fields', {})
    if opening != dict(concentration=6, good_condition_turns=6, score_multiplier=36.23):
        raise ValueError('manual retrieval initialization has no verified original opening')
    prelude = (
        ('system', ('Pアイテム「花ロボ(黄)」','集中0→3','好調0→3','集中3→6','好調3→6')),
        ('turn-start', ('1ターン目','3623%','手札情報')),
    )
    if (len(groups) < 2 or any((group.get('kind'), _lines(group)) != exact
                               for group,exact in zip(groups[:2],prelude,strict=True))):
        raise ValueError('manual retrieval newly observed initialization or first-turn prelude differs')
    groups = groups[2:]
    prefix = tuple(expected.get('prefix_groups', ()))
    sources = tuple(expected.get('source_records', ()))
    if (len(groups) != len(prefix)+1 or len(sources) != 7 or len(set(sources)) != 7
            or any(not isinstance(name, str) or not name for name in sources)):
        raise ValueError('manual retrieval history does not contain exactly five cards and two drinks')
    grants = tuple(group for group in prefix if group.get('kind') == 'system')
    if (len(grants) != len(bundle.state.memory_abilities)
            or any(_lines(group) != ('メモリーアビリティ', 'メモリーアビリティが付与') for group in grants)
            or tuple(group.get('kind') for group in prefix) != ('system',)*len(grants)+('card',)*4+('drink','card')):
        raise ValueError('manual retrieval prefix has no verified initialization or action order')
    known = {normalize_upgrade_title(normalize_history_text(bundle.kernel.cards[ref.card_id].name))
             for ref in bundle.state.deck.all_cards}
    def bound(group):
        return group.get('kind'), tuple(_bound_history_line(line, known) for line in _lines(group))
    if any(bound(actual) != bound(saved) for actual, saved in zip(groups[:-1], prefix, strict=True)):
        raise ValueError('manual retrieval complete history differs from the native prefix')
    if _lines(prefix[-2])[0] != 'Pドリンク「初星ホエイプロテイン」':
        raise ValueError('manual retrieval preceding drink is not the verified protein')
    last = groups[-1]
    lines = _lines(last)
    if last.get('kind') != 'drink' or lines[0] != 'Pドリンク「初星黒酢」':
        raise ValueError('manual retrieval latest group is not the consumed black vinegar')
    joined = ''.join(lines[1:])
    moves = re.findall(r'スキルカード「([^「」]+)」を(山札|捨札)から(\d+)枚手札に移動', joined)
    if moves != [('眠気', '山札', '1')]:
        raise ValueError('manual retrieval history does not prove one reported card moved from draw to hand')
    numeric_fields = {'体力':'stamina', '消費体力変化':'nullify_cost_cards',
                      '消費体力増加':'double_cost_turns'}
    for line in lines[1:]:
        match = re.fullmatch(r'([^0-9]+)(\d+)→(\d+)', line)
        if match:
            # Existing OCR alias is applied only while binding this numeric
            # label; retain the original observed log text unchanged.
            field = numeric_fields.get(_normalize_status_text(match[1]))
            if (field is None or int(match[2]) != getattr(bundle.state, field)
                    or {getattr(state, field) for state in candidates} != {int(match[3])}):
                raise ValueError('manual retrieval history contains a contradictory numeric effect')
    if not any(tuple(after['fields'][:5]) == (state.turns_remaining, state.stamina, state.genki,
                                            state.score, state.score_multiplier) for state in candidates):
        raise ValueError('manual retrieval core differs from every legal reported outcome')


def prepare_manual_choice_restart(projected, *, old_run, new_run, old_pid, new_pid,
                                  manual_action, history, expected_history):
    """Create comparison-only state; fresh history alone does not publish a run."""
    bundle = projected.source_bundle
    if (type(old_pid) is not int or type(new_pid) is not int or min(old_pid, new_pid) <= 0
            or old_pid == new_pid or _epoch(old_run) == _epoch(new_run)
            or any(getattr(old_run, key) != getattr(new_run, key) for key in ('idol','plan','max_stamina'))
            or (bundle.run_id, bundle.revision) != (old_run.run_id, old_run.revision)):
        raise ValueError('manual retrieval source process, epoch or identity is invalid')
    candidates, action = manual_choice_outcomes(projected, manual_action)
    _verify_manual_history(bundle, candidates, history, expected_history, new_run)
    return HifManualChoiceRestart(replace(bundle, **_epoch(new_run)), candidates, action,
        old_run, new_run, old_pid, new_pid, copy.deepcopy(history), copy.deepcopy(expected_history),
        copy.deepcopy(manual_action))


def manual_choice_history_tail(history):
    """Match the existing latest-history reader: last card through every later group."""
    groups = tuple(history.get('observation', {}).get('groups', ()))
    indices = [index for index,group in enumerate(groups) if group.get('kind') == 'card']
    if not indices:
        raise ValueError('manual retrieval full history has no latest card group')
    lines = [line for group in groups[indices[-1]:] for line in group['lines']]
    if len(lines) < 4:
        raise ValueError('manual retrieval latest history tail is incomplete')
    return dict(lines=lines,score=history['observation']['score'])


def resume_manual_choice_history(prepared, *, current_run, current_game_pid,
                                 current_client_size, source_client_size, latest_history):
    """Continue only observation in the same pending epoch; do not publish a binding."""
    _verify_manual_history(prepared.comparison_bundle,prepared.candidates,prepared.history,
                           prepared.expected_history,prepared.new_run)
    current_size,source_size = tuple(current_client_size),tuple(source_client_size)
    if (type(current_game_pid) is not int or current_game_pid != prepared.new_pid
            or len(current_size) != 2 or any(type(value) is not int or value <= 0 for value in current_size)
            or current_size != source_size
            or any(getattr(current_run,key) != getattr(prepared.new_run,key) for key in ('idol','plan','max_stamina'))
            or _epoch(current_run) == _epoch(prepared.new_run)):
        raise ValueError('pending manual history process, dimensions or freshly observed identity changed')
    if latest_history.get('accepted') is not True or latest_history.get('returned_unchanged') is not True:
        raise ValueError('pending manual history lacks a fresh returned latest-log observation')
    _frames(latest_history.get('frame_ids'))
    before,after = _core(latest_history.get('before')),_core(latest_history.get('after'))
    if any(not _same(core[key],prepared.history['after'][key])
           for core in (before,after) for key in ('fields','families')):
        raise ValueError('pending manual history current core differs from the complete scan')
    reference = manual_choice_history_tail(prepared.history)
    observed = latest_history.get('observation', {})
    if any(not _same(observed.get(key),reference[key]) for key in ('lines','score')):
        raise ValueError('pending manual history latest complete card-and-drink tail differs')
    if set(_frame_sets(prepared.history)).intersection(_frame_sets(latest_history)):
        raise ValueError('pending manual history continuity reuses the original frames')
    return dict(accepted=True,observation_source='verified_pending_manual_history_continuity',
        logical_run=asdict(prepared.new_run),current_observed_run=asdict(current_run),
        game_pid=current_game_pid,client_size=list(current_size),latest_history=copy.deepcopy(latest_history))


def resume_manual_choice_components(prepared, *, partial_settlement, baseline_components, continuity):
    """Keep exactly the three already fresh observations after new continuity.

    A failed producer has no complete top-level core or state. Its independently
    returned component receipts retain their original epoch and provenance;
    the new continuity observation supplies the current core for continuation.
    """
    if (partial_settlement.get('accepted') is not False
            or partial_settlement.get('force_fresh_components') is not True
            or partial_settlement.get('execution_submit_count') != 0
            or partial_settlement.get('end_turn_submit_count',0) != 0
            or partial_settlement.get('choice_submit_count',0) != 0
            or partial_settlement.get('maintained_components') != []
            or partial_settlement.get('resumed_fresh_components')):
        raise ValueError('manual partial source is not a zero-submit failed fresh observation')
    verified = resume_manual_choice_history(prepared,
        current_run=type(prepared.new_run)(**continuity['current_observed_run']),
        current_game_pid=continuity['game_pid'],current_client_size=continuity['client_size'],
        source_client_size=continuity['client_size'],latest_history=continuity['latest_history'])
    if not _same(verified,continuity):
        raise ValueError('manual partial continuation has no verified fresh continuity')
    components = partial_settlement.get('components',{})
    if set(components) != {'cards','status','items'}:
        raise ValueError('manual partial source must contain exactly the three completed components')
    old_frames = set(_frame_sets(baseline_components))
    for key,component in components.items():
        _reject_retained(component)
        if (component.get('accepted') is not True or component.get('returned_unchanged') is not True
                or any(component.get(name) != value for name,value in _epoch(prepared.new_run).items())):
            raise ValueError('manual partial component completion or epoch differs: '+key)
        _frames(component.get('before_frame_ids'))
        _frames(component.get('after_frame_ids'))
        if old_frames.intersection(_frame_sets(component)):
            raise ValueError('manual partial component reuses pre-drink baseline frames: '+key)
    if any(row.get('source') not in ('observed_detail', 'observed_badge_absence')
           for row in components['cards'].get('cards',())):
        raise ValueError('manual partial permanent cards require fresh observed details')
    return dict(components=copy.deepcopy(components),core=copy.deepcopy(continuity['latest_history']['after']))


def finalize_manual_choice_restart(prepared, *, settlement, baseline_components):
    """Require five genuinely new components; bind bottle positions from that scan."""
    bundle = prepared.comparison_bundle
    _verify_manual_history(bundle, prepared.candidates, prepared.history, prepared.expected_history, prepared.new_run)
    if (settlement.get('accepted') is not True or settlement.get('force_fresh_components') is not True
            or settlement.get('execution_submit_count') != 0 or settlement.get('end_turn_submit_count', 0) != 0
            or settlement.get('choice_submit_count', 0) != 0 or settlement.get('maintained_components') != []
            or any(settlement.get(key) != value for key, value in _epoch(prepared.new_run).items())):
        raise ValueError('manual retrieval needs a zero-submit complete fresh settlement')
    core = _core(settlement.get('core'))
    if any(not _same(core[key], prepared.history['after'][key]) for key in ('fields','families')):
        raise ValueError('manual retrieval Live changed after its complete history')
    components = settlement.get('components', {})
    if set(components) != {'cards','status','items','drinks','turns'}:
        raise ValueError('manual retrieval is missing fresh components')
    old_frames = set(_frame_sets(baseline_components))
    for key, component in components.items():
        _reject_retained(component)
        if (component.get('accepted') is not True or component.get('returned_unchanged') is not True
                or any(component.get(name) != value for name, value in _epoch(prepared.new_run).items())):
            raise ValueError('manual retrieval component completion or epoch differs: '+key)
        frames = _frame_sets(component)
        if not frames or old_frames.intersection(frames):
            raise ValueError('manual retrieval component lacks independent fresh frames: '+key)
    if any(row.get('source') not in ('observed_detail', 'observed_badge_absence')
           for row in components['cards'].get('cards', ())):
        raise ValueError('manual retrieval permanent cards require fresh details')
    # Cross-process reacquisition uses new observed slot positions, including
    # UI compaction. The action is MOVE_CARD, never a second USE_DRINK receipt.
    state, comparison = resolve_hif_committed_observation(bundle, prepared.candidates, core=core,
        components=components, baseline_components=baseline_components, committed_action=prepared.action)
    if state is None or comparison.get('accepted') is not True:
        raise ValueError('manual retrieval complete state has no unique semantic outcome')
    slots = comparison.get('drink_slots')
    if (slots is None or len(components['drinks'].get('slots', ())) != 4
            or Counter(drink for _, drink in slots) != Counter(('2','21'))
            or Counter(bundle.state.drinks) - Counter(state.drinks) != Counter(('22',))):
        raise ValueError('manual retrieval did not observe exactly black vinegar consumed and two remaining bottles')
    if not _same(settlement.get('expected_state'), asdict(state)):
        raise ValueError('manual retrieval producer state differs from fresh complete observations')
    refs = {ref.instance_id: ref for ref in state.deck.hand}
    state = replace(state, deck=state.deck.replace_zones(hand=tuple(refs[key]
        for key in comparison['observed_hand_instance_ids'])))
    rebound = replace(bundle, state=state, hand_instance_ids=tuple(ref.instance_id for ref in state.deck.hand),
        inventory_keys=expected_permanent_inventory_keys(bundle, state), drink_slots=slots)
    event = copy.deepcopy(settlement)
    event.update(comparison, accepted=True, expected_state=asdict(state),
        observation_source='verified_external_manual_choice_restart', external_manual_action=prepared.manual_action,
        source_epoch=_epoch(prepared.old_run), source_game_pid=prepared.old_pid, game_pid=prepared.new_pid,
        native_source_records=list(prepared.expected_history['source_records']), history_observation=prepared.history)
    return rebound, state, event
