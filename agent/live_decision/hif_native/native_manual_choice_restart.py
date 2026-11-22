"""Archived native prefix plus explicitly external manual retrieval recovery."""

import re
import copy

from .native_choice_resume import _same, load_choice_source
from .native_settlement_resume import load_result


def _latest_groups(lines):
    from live_decision.history import normalize_history_text

    result = []
    for raw in lines:
        line = normalize_history_text(raw)
        card = re.fullmatch(r'スキルカード「[^「」]+」', line)
        drink = re.fullmatch(r'Pドリンク「[^「」]+」', line)
        if card or drink:
            result.append(dict(kind='card' if card else 'drink', title=line, lines=[line]))
        elif result:
            result[-1]['lines'].append(line)
        else:
            raise ValueError('manual restart archived latest history begins without an action')
    if not result or any(len(group['lines']) < 2 for group in result):
        raise ValueError('manual restart archived latest history lacks complete action effects')
    return result


def collect_manual_choice_restart_source(runtime, name, module):
    """Reverify the real drink ancestry, then collect its already observed prefix."""
    from live_decision.history import normalize_history_text
    from live_decision.upgrade import normalize_upgrade_title
    from live_decision.high_score.opening import assemble_hif_opening
    from live_decision.high_score.restart import _bound_history_line

    projected, baseline, row = load_choice_source(runtime, name, module)
    if row['choice_journal'].get('choice', {}).get('confirmation_submitted') is True:
        raise ValueError('manual restart source already has a Maa selection confirmation')
    rows, visited, cursor = [], set(), row
    while not cursor.get('native_restart_source'):
        if cursor['id'] in visited or len(visited) >= 30:
            raise ValueError('manual restart ancestry is cyclic or unbounded')
        visited.add(cursor['id'])
        rows.append(cursor)
        parent = cursor.get('native_resume_source') or cursor.get('native_pending_source')
        if not parent:
            raise ValueError('manual restart has no verified complete-history anchor')
        cursor = load_result(runtime, parent)
    anchor = cursor
    if (anchor.get('live_return') is not True or anchor.get('restart_settlement', {}).get('accepted') is not True
            or anchor.get('restart_history', {}).get('complete_from_start') is not True
            or anchor['restart_history'].get('complete_to_end') is not True):
        raise ValueError('manual restart anchor lacks verified complete history')
    groups = copy.deepcopy(anchor['restart_history']['observation']['groups'])
    sources = list(anchor['restart_settlement']['native_source_records'])
    known = {normalize_upgrade_title(normalize_history_text(projected.source_bundle.kernel.cards[ref.card_id].name))
             for ref in projected.source_bundle.state.deck.all_cards}
    def key(group):
        return group['kind'], tuple(_bound_history_line(normalize_history_text(line), known)
                                    for line in group['lines'])
    for saved in reversed(rows):
        if saved.get('resume_history'):
            latest = _latest_groups(saved['resume_history']['observation']['lines'])
            overlap = max((count for count in range(1, min(len(groups), len(latest))+1)
                           if [key(group) for group in groups[-count:]] == [key(group) for group in latest[:count]]), default=0)
            if not overlap and any(key(group) in {key(old) for old in groups} for group in latest):
                raise ValueError('manual restart history repeats an out-of-order native action')
            groups.extend(copy.deepcopy(latest[overlap:]))
        dispatched = [event for event in saved.get('live_events', ())
                      if event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)]
        if dispatched:
            if len(dispatched) != 1 or dispatched[0].get('execution_submit_count') != 1:
                raise ValueError('manual restart source contains an unsupported business action')
            sources.append(saved['id'])
    if (len(sources) != 7 or len(set(sources)) != 7
            or sum(group['kind'] == 'card' for group in groups) != 5
            or sum(group['kind'] == 'drink' for group in groups) != 1):
        raise ValueError('manual restart does not reconstruct five cards, protein and the pending vinegar')
    original = load_result(runtime, sources[0])
    openings = [event for event in original['live_events']
                if event.get('event') == 'live_opening_state' and event.get('accepted') is True]
    if len(openings) != 1:
        raise ValueError('manual restart original opening is not unique')
    opening = openings[0]
    initial = assemble_hif_opening(module.HifLiveRunContext(**original['observed_live_context']),
        core=opening['core'],components=opening['components'],hand_titles=opening['hand_titles'],
        data_root=module.CATALOG_PATH.parent).state
    return dict(projected=projected, baseline=baseline, row=row,
        old_run=module.HifLiveRunContext(**row['observed_live_context']), old_pid=row['endpoints']['game']['pid'],
        expected_history=dict(prefix_groups=groups, source_records=sources,
            opening_fields=dict(concentration=initial.concentration,good_condition_turns=initial.good_condition_turns,
                                score_multiplier=initial.score_multiplier)))


def collect_manual_history_source(runtime, name, module):
    """Keep an already complete history in its original, still-pending epoch."""
    from live_decision.high_score.manual_choice_restart import prepare_manual_choice_restart

    row = load_result(runtime,name)
    if (row.get('manual_history_source') or row.get('manual_choice_settlement',{}).get('accepted') is True
            or any(event.get('execution_submit_count',0) or event.get('end_turn_submit_count',0)
                   or event.get('choice_submit_count',0) for event in row.get('live_events',()))):
        raise ValueError('manual history source is not an uncommitted zero-business-input full scan')
    saved = collect_manual_choice_restart_source(runtime,row['native_manual_choice_source'],module)
    prepared = prepare_manual_choice_restart(saved['projected'],old_run=saved['old_run'],
        new_run=module.HifLiveRunContext(**row['observed_live_context']),old_pid=saved['old_pid'],
        new_pid=row['endpoints']['game']['pid'],manual_action=row['external_manual_action'],
        history=row['manual_choice_history'],expected_history=saved['expected_history'])
    return dict(saved=saved,prepared=prepared,row=row)


def _prepare_record(runtime, row, module):
    from live_decision.high_score.manual_choice_restart import (
        resume_manual_choice_history,
        prepare_manual_choice_restart,
    )
    if row.get('manual_history_source'):
        source = collect_manual_history_source(runtime,row['manual_history_source'],module)
        saved,prepared,parent = source['saved'],source['prepared'],source['row']
        continuity = row['manual_history_continuity']
        metadata = row.get('context_recovery',{})
        if (row['native_manual_choice_source'] != parent['native_manual_choice_source']
                or not _same(row['external_manual_action'],parent['external_manual_action'])
                or not _same(row['observed_live_context'],parent['observed_live_context'])
                or not _same(row['manual_choice_history'],parent['manual_choice_history'])
                or metadata.get('accepted') is not True or metadata.get('returned_to_live') is not True
                or not _same(metadata.get('context'),continuity.get('current_observed_run'))):
            raise ValueError('manual history continuation changed source evidence or lacks fresh observed metadata')
        verified = resume_manual_choice_history(prepared,
            current_run=module.HifLiveRunContext(**metadata['context']),
            current_game_pid=row['endpoints']['game']['pid'],current_client_size=row['endpoints']['game']['client_size'],
            source_client_size=parent['endpoints']['game']['client_size'],latest_history=continuity['latest_history'])
        if not _same(verified,continuity):
            raise ValueError('manual history saved continuity differs from its verified current evidence')
    else:
        saved = collect_manual_choice_restart_source(runtime, row['native_manual_choice_source'], module)
        prepared = prepare_manual_choice_restart(saved['projected'], old_run=saved['old_run'],
            new_run=module.HifLiveRunContext(**row['observed_live_context']), old_pid=saved['old_pid'],
            new_pid=row['endpoints']['game']['pid'], manual_action=row['external_manual_action'],
            history=row['manual_choice_history'], expected_history=saved['expected_history'])
    return saved,prepared


def collect_manual_component_source(runtime, name, module):
    """Reverify the pending source and retain only its three complete fresh scans."""
    from live_decision.high_score.manual_choice_restart import resume_manual_choice_components

    row = load_result(runtime,name)
    if (not row.get('manual_history_source') or row.get('manual_component_source')
            or row.get('manual_choice_settlement',{}).get('accepted') is True
            or any(event.get('execution_submit_count',0) or event.get('end_turn_submit_count',0)
                   or event.get('choice_submit_count',0) for event in row.get('live_events',()))):
        raise ValueError('manual component source is not an original zero-submit partial continuation')
    saved,prepared = _prepare_record(runtime,row,module)
    partial = resume_manual_choice_components(prepared,partial_settlement=row['settlement'],
        baseline_components=saved['baseline']['components'],continuity=row['manual_history_continuity'])
    return dict(saved=saved,prepared=prepared,row=row,**partial)


def load_verified_manual_choice_source(runtime, name, module):
    """Revalidate saved fresh observations; an external action never needs fake dispatch."""
    from live_decision.high_score.restart import _frame_sets
    from live_decision.high_score.manual_choice_restart import finalize_manual_choice_restart

    row = load_result(runtime,name)
    if (row.get('live_return') is not True
            or any(event.get('execution_submit_count',0) or event.get('end_turn_submit_count',0)
                   or event.get('choice_submit_count',0) for event in row.get('live_events',()))):
        raise ValueError('manual restart completion is not a zero-business-input observation')
    saved,prepared = _prepare_record(runtime,row,module)
    resumed = row['settlement'].get('resumed_fresh_components',[])
    if row.get('manual_component_source'):
        source = collect_manual_component_source(runtime,row['manual_component_source'],module)
        parent = source['row']
        if (set(resumed) != set(source['components']) or len(resumed) != len(source['components'])
                or row.get('manual_history_source') != parent['manual_history_source']
                or any(not _same(row[key],parent[key]) for key in (
                    'native_manual_choice_source','external_manual_action','observed_live_context','manual_choice_history'))
                or row['endpoints']['game']['pid'] != parent['endpoints']['game']['pid']
                or row['endpoints']['game']['client_size'] != parent['endpoints']['game']['client_size']
                or _same(row['context_recovery']['context'],parent['context_recovery']['context'])
                or set(_frame_sets(row['manual_history_continuity'])).intersection(
                    _frame_sets(parent['manual_history_continuity']))
                or any(not _same(row['settlement']['components'].get(key),component)
                       for key,component in source['components'].items())):
            raise ValueError('manual resumed components or continuity differ from their actual partial source')
    elif resumed:
        raise ValueError('manual resumed components have no independently verified partial source')
    bundle, state, event = finalize_manual_choice_restart(prepared, settlement=row['settlement'],
        baseline_components=saved['baseline']['components'])
    if not _same(event, row.get('manual_choice_settlement')):
        raise ValueError('manual restart saved binding differs from its verified observations')
    return bundle, state, event, row
