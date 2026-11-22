"""Rebind a settled ledger after a process restart using fresh full observations.

The native host verifies the archived action chain and current round. These
pure gates neither send input nor reuse an old process's visual certificates.
"""

import re
import copy
import math
from dataclasses import asdict, replace, dataclass

from .opening import (
    observed_hif_drink_slots,
    verified_expired_start_drinks,
    verified_expired_start_scores,
    expected_permanent_inventory_keys,
    resolve_hif_committed_observation,
)
from .restart import _same, _epoch, _frames, _frame_sets, _bound_card_name, _reject_retained
from ..history import normalize_history_text
from ..upgrade import normalize_upgrade_title


@dataclass(frozen=True)
class HifSettledRestartPreparation:
    comparison_bundle: object
    expected: object
    settlement: dict
    reference_history: dict
    chronology: tuple
    old_run: object
    new_run: object
    old_pid: int
    new_pid: int
    history: dict
    opening: dict | None
    continuity_history: dict | None = None


def _core(value):
    if (not isinstance(value, dict) or len(value.get('fields', ())) != 6
            or type(value['fields'][0]) is not int or value['fields'][0] <= 0
            or not isinstance(value.get('families'), (list, tuple))):
        raise ValueError('settled restart requires a complete nonterminal core')
    _frames(value.get('frame_ids'))
    return value


def _no_business_input(value):
    for key in ('execution_submit_count', 'end_turn_submit_count', 'drinks_consumed'):
        if key in value and (type(value[key]) is not int or value[key] != 0):
            raise ValueError('settled restart observation contains business input')


def _action(turn, kind, title):
    if type(turn) is not int or turn < 1 or kind not in ('card', 'drink') or not isinstance(title, str):
        raise ValueError('settled restart action chronology is malformed')
    title = normalize_history_text(title)
    if not title or any(mark in title for mark in '「」'):
        raise ValueError('settled restart action title is incomplete')
    return turn, kind, normalize_upgrade_title(title) if kind == 'card' else title


def verified_replay_action(kernel, before, after, deck, turn):
    """Describe one automatic use from an already verified schedule transition."""
    from .kernel import CardSelector

    definition = kernel.cards[before['card_id']]
    index = before['definition_effect_index']
    if (type(index) is not int or not 0 <= index < len(definition.persistent_effects)
            or before.get('source_kind') != 'CARD' or before.get('persistent') is not True
            or before.get('trigger_phase') != 'EVERY_TURN'
            or type(before.get('remaining_uses')) is not int or before['remaining_uses'] <= 0
            or type(before.get('effect_counter')) is not int or before['effect_counter'] < 0
            or not _same(after, dict(before, remaining_uses=before['remaining_uses']-1,
                effect_counter=before['effect_counter']+1, last_fired_turn=turn))):
        raise ValueError('settled restart automatic use lacks a verified schedule decrement')
    effect = definition.persistent_effects[index]
    if (effect.kind.value != 'SCHEDULE' or effect.schedule_phase.value != 'EVERY_TURN'
            or type(effect.schedule_limit) is not int
            or not before['remaining_uses'] <= effect.schedule_limit
            or len(effect.scheduled_effects) != 1):
        raise ValueError('settled restart automatic schedule is unsupported')
    child = effect.scheduled_effects[0]
    condition = child.condition
    conditions = {part.kind.value: part for part in condition.all_of} if condition and condition.kind.value == 'ALL' else {}
    if (child.kind.value != 'FREE_PLAY_RANDOM' or child.repeats != 1
            or len(condition.all_of if condition else ()) != 3
            or set(conditions) != {'REMOVED_BASE_CARD_COUNT_MIN', 'EFFECT_COUNTER_MOD_EQ', 'STAMINA_RATIO_MIN'}):
        raise ValueError('settled restart automatic play rule is unsupported')
    removed, period = conditions['REMOVED_BASE_CARD_COUNT_MIN'], conditions['EFFECT_COUNTER_MOD_EQ']
    if (child.card_selector != CardSelector(base_card_id=removed.value)
            or effect.schedule_counter_condition != removed or not period.modulus
            or before['effect_counter'] % period.modulus != period.remainder):
        raise ValueError('settled restart automatic selector or timing differs')
    targets = [card for card in deck['all_cards'] if kernel.cards[card['card_id']].base_card_id == removed.value]
    if (len(targets) != 1 or targets[0] not in deck['removed']
            or targets[0]['instance_id'] != before['source_id']
            or targets[0]['card_id'] != before['card_id'] or removed.threshold != 1):
        raise ValueError('settled restart automatic target is not uniquely bound in removal')
    return dict(turn=turn, kind='card', title=definition.name, automatic=True,
                replay_evidence=dict(before=copy.deepcopy(before), after=copy.deepcopy(after)))


def _history_card_name(title, lines, known_names):
    """Keep an OCR alias local to an independently named effect in this group."""
    try:
        return _bound_card_name(title, known_names)
    except ValueError:
        name = normalize_upgrade_title(title)
        matches = {known for known in known_names if len(name) == len(known)
            and [(a, b) for a, b in zip(name, known, strict=True) if a != b] in ([('カ', '力')], [('力', 'カ')])
            and lines.count(known+'が付与') == 1}
        if len(matches) != 1:
            raise ValueError('settled restart card title lacks independent name evidence') from None
        return matches.pop()


def _initial_prefix(bundle, expected, parsed, observation, opening):
    if parsed[0][0] != 'system' or not parsed[0][1][0].startswith('Pアイテム'):
        explicit = parsed[0] == ('turn-start', ('1ターン目',))
        return int(explicit), explicit
    # 878 already proved this exact prelude against opening 6/6. It is an
    # observed initialization result, not an invented kernel item or trigger.
    from .opening_items import STAGE_START_ITEM_ID, bind_observed_stage_start_item

    state = opening.get('state', {}) if isinstance(opening, dict) else {}
    if (not isinstance(opening, dict) or opening.get('accepted') is not True
            or opening.get('strategy_state_bound') is not True or state.get('turn') != 1
            or state.get('concentration') != 6 or state.get('good_condition_turns') != 6
            or tuple(state.get('items', ())) != expected.items
            or tuple(state.get('memory_abilities', ())) != expected.memory_abilities
            or len(state.get('deck', {}).get('all_cards', ())) != 22):
        raise ValueError('settled restart startup prelude lacks its verified original opening')
    rows = opening.get('components', {}).get('items', {}).get('identity_count_observation', {}).get('items', ())
    matches = [row for row in rows if row.get('item_id') == STAGE_START_ITEM_ID]
    if len(matches) != 1 or STAGE_START_ITEM_ID not in expected.items:
        raise ValueError('settled restart startup item is not independently bound')
    definition = bind_observed_stage_start_item(matches[0], definition=bundle.kernel.items.get(STAGE_START_ITEM_ID))
    multiplier = state.get('score_multiplier')
    if type(multiplier) not in (int, float) or not 0 < multiplier < 1000:
        raise ValueError('settled restart opening multiplier is invalid')
    prefix = [
        ('system', ('Pアイテム「花ロボ(黄)」', '集中0→3', '好調0→3', '集中3→6', '好調3→6')),
        ('system', (f'Pアイテム「{normalize_history_text(definition.name)}」', '変化なし')),
        ('turn-start', ('1ターン目', f'{round(multiplier*100)}%', '手札情報')),
    ]
    if parsed[:3] != prefix or observation.get('first_turn_observed') is not True:
        raise ValueError('settled restart observed startup prelude differs from the original opening')
    return 3, True


def _verify_history(bundle, expected, source, reference, chronology, history, new_run, opening):
    if (history.get('accepted') is not True or history.get('returned_unchanged') is not True
            or history.get('complete_from_start') is not True or history.get('complete_to_end') is not True
            or any(history.get(key) != value for key, value in _epoch(new_run).items())):
        raise ValueError('settled restart history lacks fresh complete-run evidence')
    _no_business_input(history)
    _frames(history.get('frame_ids'))
    old_core = _core(source.get('core'))
    for core in (history.get('before'), history.get('after')):
        core = _core(core)
        if any(not _same(core[key], old_core[key]) for key in ('fields', 'families')):
            raise ValueError('settled restart current Live differs from the settled ledger')
    old_frames = set(_frame_sets(source)) | set(_frame_sets(reference)) | set(_frame_sets(opening))
    if old_frames.intersection(_frame_sets(history)):
        raise ValueError('settled restart history reuses archived frames')
    observation = history.get('observation', {})
    if (observation.get('at_top') is not True or observation.get('at_bottom') is not True
            or observation.get('score') != expected.score):
        raise ValueError('settled restart history boundaries or score disagree')
    groups = observation.get('groups')
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError('settled restart history groups are missing')
    parsed = []
    for group in groups:
        if not isinstance(group, dict) or group.get('kind') not in ('system', 'turn-start', 'card', 'drink'):
            raise ValueError('settled restart history has an unknown group')
        lines = group.get('lines')
        if (not isinstance(lines, (list, tuple)) or not lines
                or any(not isinstance(line, str) or not normalize_history_text(line) for line in lines)):
            raise ValueError('settled restart history group is incomplete')
        lines = tuple(normalize_history_text(line) for line in lines)
        if normalize_history_text(group.get('title', '')) != lines[0] or (group['kind'] != 'turn-start' and len(lines) < 2):
            raise ValueError('settled restart history heading or body is incomplete')
        parsed.append((group['kind'], lines))
    offset, explicit_first = _initial_prefix(bundle, expected, parsed, observation, opening)
    grant = ('system', ('メモリーアビリティ', 'メモリーアビリティが付与'))
    count = len(expected.memory_abilities)
    if not count or parsed[offset:offset+count] != [grant] * count:
        raise ValueError('settled restart history lacks the original memory initialization')
    known_names = {normalize_upgrade_title(normalize_history_text(bundle.kernel.cards[card.card_id].name))
                   for card in expected.deck.all_cards}
    # Native chronology also carries independently verified temporary upgrades;
    # a later permanent deck snapshot deliberately does not retain those marks.
    known_names.update(_action(row['turn'], row['kind'], row['title'])[2]
                       for row in chronology if row['kind'] == 'card')
    turn, turns, actual, latest, replay = 1, ([1] if explicit_first else []), [], None, None
    for index in range(offset+count, len(parsed)):
        kind, lines = parsed[index]
        if replay is not None and kind != 'card':
            raise ValueError('settled restart automatic preparation lacks its immediate card')
        if kind == 'turn-start':
            match = re.fullmatch(r'([1-9]\d*)ターン目', lines[0])
            if match is None or int(match[1]) != turn + 1:
                raise ValueError('settled restart history turns are not continuous')
            turn += 1
            turns.append(turn)
            latest = index
        elif kind in ('card', 'drink'):
            pattern = r'スキルカード「([^「」]+)」' if kind == 'card' else r'(?:Pドリンク|ドリンク)「([^「」]+)」'
            match = re.fullmatch(pattern, lines[0])
            if match is None:
                raise ValueError('settled restart action heading is incomplete')
            title = _history_card_name(match[1], lines, known_names) if kind == 'card' else match[1]
            if replay is not None and title != replay:
                raise ValueError('settled restart automatic preparation names another card')
            actual.append((*_action(turn, kind, title), replay is not None))
            replay = None
            latest = index
        elif lines[0] == grant[1][0]:
            raise ValueError('settled restart contains an extra memory initialization')
        else:
            repeat = re.fullmatch(r'持続効果「([^「」]+)\(再演\)」', lines[0])
            if repeat is not None:
                replay = _bound_card_name(repeat[1], known_names)
                if ''.join(lines[1:]) != f'スキルカード「{repeat[1]}」を使用準備':
                    raise ValueError('settled restart automatic preparation is incomplete')
    wanted = []
    for row in chronology:
        automatic = row.get('automatic', False)
        if type(automatic) is not bool:
            raise ValueError('settled restart automatic action flag is invalid')
        if automatic:
            proof = row.get('replay_evidence', {})
            if (set(proof) != {'before', 'after'} or not _same(row, verified_replay_action(
                    bundle.kernel, proof['before'], proof['after'], asdict(expected.deck), row['turn']))):
                raise ValueError('settled restart automatic action lacks its bound rule evidence')
        wanted.append((*_action(row['turn'], row['kind'], row['title']), automatic))
    if replay is not None or not wanted or actual != wanted:
        raise ValueError('settled restart card or drink chronology changed')
    if (turn != expected.turn or observation.get('turn') != (turns[-1] if turns else None)
            or tuple(observation.get('turns', ())) != tuple(turns)):
        raise ValueError('settled restart latest turn disagrees')
    reference_lines = tuple(normalize_history_text(line) for line in reference['observation']['lines'])
    tail_groups = parsed[latest:] if latest is not None else []
    if tail_groups and tail_groups[0][0] == 'turn-start' and len(tail_groups[0][1]) != 1:
        lines = tail_groups[0][1]
        if lines[1:] != (f'{round(expected.score_multiplier*100)}%', '手札情報'):
            raise ValueError('settled restart latest turn layout disagrees with its bound multiplier')
        tail_groups = [('turn-start', lines[:1]), *tail_groups[1:]]
    tail = tuple(line for _, lines in tail_groups for line in lines)
    if tail != reference_lines:
        raise ValueError('settled restart latest history differs from the verified settlement')


def prepare_settled_restart(bundle, expected, *, settlement, reference_history, chronology,
                            old_run, new_run, old_pid, new_pid, history, opening=None):
    """Prepare comparison only; the host supplies its verified settled source."""
    if (type(old_pid) is not int or type(new_pid) is not int or min(old_pid, new_pid) <= 0
            or old_pid == new_pid or _epoch(old_run) == _epoch(new_run)
            or any(getattr(old_run, key) != getattr(new_run, key) for key in ('idol', 'plan', 'max_stamina'))
            or bundle.run_id != old_run.run_id or bundle.revision != old_run.revision):
        raise ValueError('settled restart process, epoch or identity is invalid')
    expected.validate()
    if (expected.max_stamina != new_run.max_stamina or expected.turns_remaining <= 0
            or expected.pending_choice or expected.pending_after_card_ids or expected.resolution_stack
            or not expected.persistent_effects_initialized):
        raise ValueError('settled restart requires a quiescent complete state')
    if (settlement.get('accepted') is not True or not settlement.get('checks')
            or not all(value is True for value in settlement['checks'].values())
            or any(settlement.get(key) != value for key, value in _epoch(old_run).items())
            or not _same(settlement.get('expected_state'), asdict(expected))):
        raise ValueError('settled restart lacks a matching verified settlement')
    if (reference_history.get('accepted') is not True or reference_history.get('returned_unchanged') is not True
            or not reference_history.get('observation', {}).get('lines')
            or reference_history['observation'].get('score') != expected.score):
        raise ValueError('settled restart lacks the verified latest history')
    _frames(reference_history.get('frame_ids'))
    for key in ('before', 'after'):
        if any(not _same(_core(reference_history.get(key))[field], settlement['core'][field])
               for field in ('fields', 'families')):
            raise ValueError('settled restart reference history differs from the settled core')
    _verify_history(bundle, expected, settlement, reference_history, chronology, history, new_run, opening)
    comparison = replace(bundle, **_epoch(new_run), state=expected,
        hand_instance_ids=tuple(ref.instance_id for ref in expected.deck.hand),
        inventory_keys=expected_permanent_inventory_keys(bundle, expected),
        expired_start_scores=verified_expired_start_scores(bundle, expected),
        expired_start_drinks=verified_expired_start_drinks(bundle, expected),
        drink_slots=observed_hif_drink_slots(settlement['components']['drinks']))
    return HifSettledRestartPreparation(comparison, expected, copy.deepcopy(settlement),
        copy.deepcopy(reference_history), tuple(copy.deepcopy(chronology)), old_run, new_run,
        old_pid, new_pid, copy.deepcopy(history), copy.deepcopy(opening))


def continue_settled_restart_history(prepared, *, current_history, run, game_pid):
    """Retain the proved full history; require a new unchanged latest-history read."""
    current = current_history
    if (game_pid != prepared.new_pid or _epoch(run) != _epoch(prepared.new_run)
            or any(getattr(run, key) != getattr(prepared.new_run, key) for key in ('idol', 'plan', 'max_stamina'))
            or any(current.get(key) != value for key, value in _epoch(run).items())
            or current.get('accepted') is not True or current.get('returned_unchanged') is not True):
        raise ValueError('settled restart history continuation changed its process or epoch')
    _no_business_input(current)
    _frames(current.get('frame_ids'))
    archived = (set(_frame_sets(prepared.history)) | set(_frame_sets(prepared.reference_history))
                | set(_frame_sets(prepared.settlement)) | set(_frame_sets(prepared.opening)))
    if archived.intersection(_frame_sets(current)):
        raise ValueError('settled restart latest history reused archived frames')
    reference = prepared.reference_history['observation']
    observed = current.get('observation', {})
    if (not reference.get('lines') or not _same(observed.get('lines'), reference['lines'])
            or observed.get('score') != reference['score'] or observed.get('score') != prepared.expected.score):
        raise ValueError('settled restart latest history changed')
    for value in (current.get('before'), current.get('after')):
        core = _core(value)
        if any(not _same(core[key], prepared.history['after'][key]) for key in ('fields', 'families')):
            raise ValueError('settled restart history continuation changed Live state')
    return replace(prepared, continuity_history=copy.deepcopy(current))


def finalize_settled_restart(prepared, *, settlement):
    """Require five new-process components and one unique unchanged rule state."""
    bundle = prepared.comparison_bundle
    _verify_history(bundle, prepared.expected, prepared.settlement, prepared.reference_history,
                    prepared.chronology, prepared.history, prepared.new_run, prepared.opening)
    if prepared.continuity_history is not None:
        continue_settled_restart_history(prepared, current_history=prepared.continuity_history,
                                        run=prepared.new_run, game_pid=prepared.new_pid)
    if (settlement.get('accepted') is not True or settlement.get('force_fresh_components') is not True
            or type(settlement.get('execution_submit_count')) is not int or settlement['execution_submit_count'] != 0
            or settlement.get('maintained_components') != []
            or any(settlement.get(key) != value for key, value in _epoch(prepared.new_run).items())):
        raise ValueError('settled restart needs a zero-submit complete fresh settlement')
    _no_business_input(settlement)
    core = _core(settlement.get('core'))
    if any(not _same(core[key], prepared.history['after'][key]) for key in ('fields', 'families')):
        raise ValueError('settled restart Live changed after complete history')
    components = settlement.get('components', {})
    if set(components) != {'cards', 'status', 'items', 'drinks', 'turns'}:
        raise ValueError('settled restart complete fresh components are missing')
    old_frames = (set(_frame_sets(prepared.settlement)) | set(_frame_sets(prepared.reference_history))
                  | set(_frame_sets(prepared.opening)))
    if prepared.continuity_history is not None:
        old_frames |= set(_frame_sets(prepared.history)) | set(_frame_sets(prepared.continuity_history))
    if old_frames.intersection(_frame_sets(core)):
        raise ValueError('settled restart core reuses archived frames')
    for key, component in components.items():
        _reject_retained(component)
        _no_business_input(component)
        if (component.get('accepted') is not True or component.get('returned_unchanged') is not True
                or any(component.get(name) != value for name, value in _epoch(prepared.new_run).items())):
            raise ValueError('settled restart component epoch or completion is invalid: ' + key)
        frames = _frame_sets(component)
        if not frames or any(not isinstance(frame, str) or not frame for frame in frames):
            raise ValueError('settled restart component lacks fresh frames: ' + key)
        if old_frames.intersection(frames):
            raise ValueError('settled restart component reuses archived frames: ' + key)
    if any(row.get('source') not in ('observed_detail', 'observed_badge_absence')
           for row in components['cards'].get('cards', ())):
        raise ValueError('settled restart permanent cards require fresh observations')
    state, comparison = resolve_hif_committed_observation(bundle, (prepared.expected,), core=core,
        components=components, baseline_components=prepared.settlement['components'])
    if state is None or comparison.get('accepted') is not True or not _same(asdict(state), asdict(prepared.expected)):
        raise ValueError('settled restart complete state differs from the unique settled outcome')
    if not _same(settlement.get('expected_state'), asdict(state)):
        raise ValueError('settled restart producer state differs from complete comparison')
    by_id = {ref.instance_id: ref for ref in state.deck.hand}
    state = replace(state, deck=state.deck.replace_zones(hand=tuple(
        by_id[key] for key in comparison['observed_hand_instance_ids'])))
    rebound = replace(bundle, state=state, hand_instance_ids=tuple(ref.instance_id for ref in state.deck.hand),
        inventory_keys=expected_permanent_inventory_keys(bundle, state), drink_slots=comparison['drink_slots'])
    event = copy.deepcopy(settlement)
    event.update(comparison, accepted=True, expected_state=asdict(state), observation_source='verified_settled_restart',
        source_epoch=_epoch(prepared.old_run), source_game_pid=prepared.old_pid, game_pid=prepared.new_pid,
        expected_action_chronology=copy.deepcopy(list(prepared.chronology)), history_observation=copy.deepcopy(prepared.history))
    if prepared.continuity_history is not None:
        event['history_continuity_observation'] = copy.deepcopy(prepared.continuity_history)
    return rebound, state, event
