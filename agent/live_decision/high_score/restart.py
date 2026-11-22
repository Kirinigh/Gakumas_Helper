"""Two-stage first-turn recovery from a verified native chain and fresh UI data.

The host verifies the original files, dispatched actions and rule transitions.
This module never reads files or sends input. Preparation binds expectations for
comparison only; complete fresh observations are required before finalization.
"""

import re
import copy
import json
from dataclasses import asdict, replace, dataclass

from .opening import expected_permanent_inventory_keys, resolve_hif_committed_observation
from ..history import normalize_history_text
from ..upgrade import normalize_upgrade_title
from .contracts import ActionKind


@dataclass(frozen=True)
class HifRestartPreparation:
    comparison_bundle: object
    candidates: tuple
    action: object
    old_run: object
    new_run: object
    old_pid: int
    new_pid: int
    history: dict
    expected_history: dict


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _epoch(run):
    return dict(run_id=run.run_id, revision=run.revision)


def _frames(values):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)):
        raise ValueError('restart observation needs distinct fresh frames')


def _core(core):
    if (not isinstance(core, dict) or len(core.get('fields', ())) != 6
            or type(core['fields'][0]) is not int or core['fields'][0] != 9
            or not isinstance(core.get('families'), (list, tuple))):
        raise ValueError('restart requires a complete first-turn core')
    _frames(core.get('frame_ids'))
    return core


def _lines(group):
    values = group.get('lines') if isinstance(group, dict) else group
    if (not isinstance(values, (list, tuple)) or len(values) < 2
            or any(not isinstance(value, str) or not value for value in values)):
        raise ValueError('restart history group is incomplete')
    return tuple(normalize_history_text(value) for value in values)


def _bound_card_name(name, known_names):
    name = normalize_upgrade_title(normalize_history_text(name))
    matches = {known for known in known_names if name == known or (
        name[:1] in ('一', 'ー') and known[:1] in ('一', 'ー') and name[1:] == known[1:])}
    if len(matches) != 1:
        raise ValueError('restart card reference is not uniquely bound to this run')
    return matches.pop()


def _bound_history_line(line, known_names):
    # Only complete card references permit the observed first-glyph OCR alias.
    # Numeric/effect lines and the archived text are never rewritten.
    match = re.fullmatch(r'スキルカード「([^「」]+)」(を?)', line)
    if match is None:
        return line
    return f'スキルカード「{_bound_card_name(match[1], known_names)}」{match[2]}'


def _title(line, known_names):
    match = re.fullmatch(r'スキルカード「(.+)」', line)
    if not match:
        raise ValueError('restart history action heading is incomplete')
    return _bound_card_name(match[1], known_names)


def _verify_history(bundle, candidates, action, history, expected, new_run):
    if (history.get('accepted') is not True or history.get('returned_unchanged') is not True
            or history.get('complete_from_start') is not True or history.get('complete_to_end') is not True
            or any(history.get(key) != value for key, value in _epoch(new_run).items())):
        raise ValueError('restart history lacks fresh complete-run evidence')
    _frames(history.get('frame_ids'))
    before, after = _core(history.get('before')), _core(history.get('after'))
    if any(not _same(before[key], after[key]) for key in ('fields', 'families')):
        raise ValueError('restart Live changed during full history observation')
    observation = history.get('observation', {})
    turn_metadata = observation.get('turn'), tuple(observation.get('turns', ()))
    if (observation.get('at_top') is not True or observation.get('at_bottom') is not True
            or turn_metadata not in ((None, ()), (1, (1,)))
            or observation.get('score') != after['fields'][3]):
        raise ValueError('restart history boundaries, turn or score disagree')
    groups = tuple(observation.get('groups', ()))
    # The real first-turn log begins with memory grants, without a turn title.
    # A title, when explicitly observed, cannot stand in for that initialization.
    if groups and groups[0].get('kind') == 'turn-start':
        if turn_metadata != (1, (1,)) or normalize_history_text(groups[0].get('title', '')) != '1ターン目':
            raise ValueError('restart explicit first-turn heading is contradictory')
        groups = groups[1:]
    elif turn_metadata != (None, ()):
        raise ValueError('restart turn title was not actually observed')
    initial = tuple(_lines(group) for group in expected.get('initial_groups', ()))
    grant = ('メモリーアビリティ', 'メモリーアビリティが付与')
    if (not initial or len(initial) != len(bundle.state.memory_abilities)
            or any(lines != grant for lines in initial)
            or len(groups) <= len(initial)
            or any(group.get('kind') != 'system'
                   or normalize_history_text(group.get('title', '')) != grant[0]
                   for group in groups[:len(initial)])
            or tuple(_lines(group) for group in groups[:len(initial)]) != initial):
        raise ValueError('restart memory initialization does not match the verified opening')
    groups = groups[len(initial):]
    if any(group.get('kind') != 'card' for group in groups):
        raise ValueError('restart only supports first-turn card history')
    known_names = {normalize_upgrade_title(normalize_history_text(bundle.kernel.cards[ref.card_id].name))
                   for state in (bundle.state, *candidates) for ref in state.deck.all_cards}
    cards = tuple(_lines(group) for group in groups)
    prefix = tuple(_lines(group) for group in expected.get('prefix_groups', ()))
    bound_cards = tuple(tuple(_bound_history_line(line, known_names) for line in lines) for lines in cards)
    bound_prefix = tuple(tuple(_bound_history_line(line, known_names) for line in lines) for lines in prefix)
    titles = tuple(expected.get('action_titles', ()))
    sources = tuple(expected.get('source_records', ()))
    if (not 1 <= len(cards) <= 60 or len(cards) != len(prefix) + 1
            or len(titles) != len(cards) or len(sources) != len(cards)
            or any(not isinstance(name, str) or not name for name in sources)
            or len(set(sources)) != len(sources) or bound_cards[:-1] != bound_prefix):
        raise ValueError('restart full history does not match the verified native prefix')
    if any(_title(lines[0], known_names) != _bound_card_name(title, known_names)
           for lines, title in zip(cards, titles, strict=True)):
        raise ValueError('restart action order differs from the native chain')
    selected = next((ref for ref in bundle.state.deck.hand if ref.instance_id == action.card_instance_id), None)
    if selected is None or _title(cards[-1][0], known_names) != _bound_card_name(bundle.kernel.cards[selected.card_id].name, known_names):
        raise ValueError('restart latest group does not identify the dispatched card')
    numeric = tuple(normalize_history_text(line) for line in expected.get('pending_numeric_lines', ()))
    if (not numeric or len(set(numeric)) != len(numeric)
            or any(re.fullmatch(r'[^0-9]+\d+→\d+', line) is None for line in numeric)
            or any(cards[-1].count(line) != 1 for line in numeric)):
        raise ValueError('restart latest numeric effects do not match the dispatched outcome')
    labels = {re.match(r'[^0-9]+', line)[0] for line in numeric}
    if any(match and match[1] in labels and line not in numeric
           for line in cards[-1][1:] if (match := re.fullmatch(r'([^0-9]+)\d+→\d+', line))):
        raise ValueError('restart latest numeric effects contain a contradictory transition')
    if not any(tuple(after['fields'][:5]) == (state.turns_remaining, state.stamina, state.genki,
                                              state.score, state.score_multiplier) for state in candidates):
        raise ValueError('restart core is not a possible dispatched outcome')


def prepare_hif_restart(bundle, candidates, *, old_run, new_run, old_pid, new_pid,
                        history, expected_history, action):
    """Bind a comparison bundle, without authorizing or certifying a new action.

    ``expected_history`` comes from the host's verified native file chain:
    ``initial_groups`` contains one exact memory-grant group per original memory
    ability. ``prefix_groups`` contains complete accepted card groups, ``action_titles`` and
    ``source_records`` identify all dispatched actions including the pending one,
    and ``pending_numeric_lines`` contains expected visible effect transitions.
    Payment need not appear in the log; fresh core checks cover its result.
    """
    if (type(old_pid) is not int or type(new_pid) is not int or min(old_pid, new_pid) <= 0
            or old_pid == new_pid or _epoch(old_run) == _epoch(new_run)
            or any(getattr(old_run, key) != getattr(new_run, key) for key in ('idol', 'plan', 'max_stamina'))
            or bundle.run_id != old_run.run_id or bundle.revision != old_run.revision):
        raise ValueError('restart source process, epoch or identity is invalid')
    states = candidates if isinstance(candidates, tuple) else (candidates,)
    if not states or len(states) > 10000:
        raise ValueError('restart dispatched outcome support is empty or unbounded')
    action.validate()
    if action.kind is not ActionKind.PLAY_CARD:
        raise ValueError('restart currently supports dispatched cards only')
    for state in (bundle.state, *states):
        state.validate()
        if (state.turn != 1 or state.turns_remaining != 9 or state.max_stamina != new_run.max_stamina or state.pending_choice
                or state.pending_after_card_ids or state.resolution_stack or not state.persistent_effects_initialized):
            raise ValueError('restart requires a quiescent first-turn action chain')
    _verify_history(bundle, states, action, history, expected_history, new_run)
    comparison = replace(bundle, **_epoch(new_run))
    return HifRestartPreparation(comparison, states, action, old_run, new_run, old_pid, new_pid,
                                 copy.deepcopy(history), copy.deepcopy(expected_history))


def _frame_sets(value):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith('frame_ids') and isinstance(child, (list, tuple)):
                result.extend(child)
            else:
                result.extend(_frame_sets(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.extend(_frame_sets(child))
    return result


def _reject_retained(value):
    if isinstance(value, dict):
        if any(key in value for key in ('retained_from', 'source_epoch', 'maintenance_action', 'maintenance_frame_ids')):
            raise ValueError('restart component contains retained provenance')
        for key, child in value.items():
            if key in ('source', 'observation_source') and isinstance(child, str) and child.startswith(('retained', 'maintained')):
                raise ValueError('restart component was not freshly observed')
            _reject_retained(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_retained(child)


def finalize_hif_restart(prepared, *, settlement, baseline_components):
    """Accept only a full new scan matching the pending rule outcomes and log."""
    bundle = prepared.comparison_bundle
    _verify_history(bundle, prepared.candidates, prepared.action, prepared.history,
                    prepared.expected_history, prepared.new_run)
    if (settlement.get('accepted') is not True or settlement.get('force_fresh_components') is not True
            or settlement.get('execution_submit_count') != 0 or settlement.get('end_turn_submit_count', 0) != 0
            or settlement.get('maintained_components') != []
            or any(settlement.get(key) != value for key, value in _epoch(prepared.new_run).items())):
        raise ValueError('restart needs a zero-submit complete fresh settlement')
    core = _core(settlement.get('core'))
    if any(not _same(core[key], prepared.history['after'][key]) for key in ('fields', 'families')):
        raise ValueError('restart Live changed between history and complete settlement')
    components = settlement.get('components', {})
    if set(components) != {'cards', 'status', 'items', 'drinks', 'turns'}:
        raise ValueError('restart complete fresh components are missing')
    old_frames = set(_frame_sets(baseline_components))
    for key, component in components.items():
        _reject_retained(component)
        if (component.get('accepted') is not True or component.get('returned_unchanged') is not True
                or any(component.get(name) != value for name, value in _epoch(prepared.new_run).items())):
            raise ValueError('restart fresh component epoch or completion is invalid: ' + key)
        frames = _frame_sets(component)
        if not frames:
            raise ValueError('restart fresh component has no frame evidence: ' + key)
        if old_frames.intersection(frames):
            raise ValueError('restart component reuses archived frame evidence: ' + key)
    if any(row.get('source') not in ('observed_detail', 'observed_badge_absence')
           for row in components['cards'].get('cards', ())):
        raise ValueError('restart permanent cards require freshly observed details')
    state, comparison = resolve_hif_committed_observation(bundle, prepared.candidates, core=core,
        components=components, baseline_components=baseline_components, committed_action=prepared.action)
    if not comparison['accepted'] or state is None:
        raise ValueError('restart complete state has no unique semantic outcome')
    if not _same(settlement.get('expected_state'), asdict(state)):
        raise ValueError('restart producer state differs from complete comparison')
    by_id = {ref.instance_id: ref for ref in state.deck.hand}
    state = replace(state, deck=state.deck.replace_zones(hand=tuple(
        by_id[key] for key in comparison['observed_hand_instance_ids'])))
    rebound = replace(bundle, state=state, hand_instance_ids=tuple(ref.instance_id for ref in state.deck.hand),
        inventory_keys=expected_permanent_inventory_keys(bundle, state), drink_slots=comparison['drink_slots'])
    event = copy.deepcopy(settlement)
    event.update(comparison, accepted=True, expected_state=asdict(state), observation_source='verified_first_turn_restart',
        source_epoch=_epoch(prepared.old_run), source_game_pid=prepared.old_pid, game_pid=prepared.new_pid,
        native_source_records=list(prepared.expected_history['source_records']), history_observation=copy.deepcopy(prepared.history))
    return rebound, state, event
