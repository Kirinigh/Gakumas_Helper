"""Restore a new Live's complete opening and its observed round metadata."""
import re
import json
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
from dataclasses import asdict

from . import diagnostic_depth_reached

PARENT_FIELDS = ('native_resume_source', 'native_selected_source', 'native_inprocess_source',
    'native_pending_source', 'native_pending_components_source', 'native_restart_source',
    'native_manual_choice_source', 'native_choice_source', 'native_settled_restart_source',
    'native_pending_restart_source', 'native_rule_resume_source')


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def has_native_opening_root(row):
    # A verified cold handoff republishes the original opening for the live
    # observer. Its parent still owns that evidence; it is not another root.
    receipt = row.get('handoff_resume') or {}
    if (receipt.get('version') == 1 and receipt.get('boundary_kind') == 'unsubmitted_opening'
            and row.get('native_rule_resume_source')
            and receipt.get('source') == row['native_rule_resume_source']
            and [key for key in PARENT_FIELDS if row.get(key)] == ['native_rule_resume_source']):
        return False
    return any(event.get('event') == 'live_opening_state' and event.get('accepted') is True
               for event in row.get('live_events', ()))


def confirmed_deck_baseline_from_source(row, *, complete_component_only=False, game_pid=None):
    """Extract fixed definitions only from an accepted complete opening.

    The caller selects the user-confirmed source. Its old physical arrangement
    describes the complete deck, never the new run's slots or dynamic state.
    """
    source_id = row.get('id')
    openings = [event for event in row.get('live_events', ())
                if event.get('event') == 'live_opening_state'
                and (event.get('accepted') is True or complete_component_only)]
    run = row.get('observed_live_context', {})
    if (not isinstance(source_id, str) or not re.fullmatch(r'[0-9]+-[a-z0-9-]+', source_id)
            or len(openings) != 1 or any(row.get(key) for key in PARENT_FIELDS)
            or not isinstance(run, dict) or not isinstance(run.get('run_id'), str)
            or not run['run_id'] or type(run.get('revision')) is not int or run['revision'] < 1):
        raise ValueError('confirmed deck source is not one complete native opening')
    opening = openings[0]
    if complete_component_only:
        if (game_pid is None or game_pid != row.get('endpoints',{}).get('game',{}).get('pid')
                or opening.get('accepted') is not False
                or opening.get('reason') != 'observed_opening_effect_counters_are_not_initial'
                or any(event.get('execution_submit_count',0) or event.get('end_turn_submit_count',0)
                    for event in row.get('live_events',()))):
            raise ValueError('observed deck source is not the same untouched complete opening')
        # This is a replay of static card evidence after an assembler repair,
        # not an accepted native opening or renewed observation authority.
        from utils.runtime_paths import resolve_runtime_path
        from live_decision.hif_visual import HifLiveRunContext
        from live_decision.high_score.opening import assemble_hif_opening
        root=resolve_runtime_path(Path('assets/data'), Path('data'))
        if not (root/'skill_cards.csv').is_file():
            raise ValueError('observed deck source catalog is unavailable')
        assemble_hif_opening(HifLiveRunContext(**run),core=opening['extended_hand_identity']['core'],
            components=opening['components'],hand_titles=opening['hand_titles'],data_root=root,
            extended_hand_identity=opening['extended_hand_identity'])
    epoch = (run['run_id'], run['revision'])
    components = opening.get('components', {})
    if ((not complete_component_only and (opening.get('strategy_state_bound') is not True
            or (opening.get('run_id'), opening.get('revision')) != epoch))
            or not isinstance(components, dict)
            or set(components) != {'cards', 'status', 'items', 'drinks', 'turns'}):
        raise ValueError('confirmed deck opening is incomplete or unbound')
    for event in components.values():
        if (not isinstance(event, dict) or event.get('accepted') is not True
                or event.get('returned_unchanged') is not True
                or (event.get('run_id'), event.get('revision')) != epoch):
            raise ValueError('confirmed deck component is incomplete or from another epoch')
    cards = components['cards'].get('cards')
    if not isinstance(cards, list) or not cards:
        raise ValueError('confirmed deck source has no complete card definitions')
    counts = dict.fromkeys(('手札', '山札', '捨札', '除外'), 0)
    for card in cards:
        if (not isinstance(card, dict)
                or set(card) != {'zone', 'index', 'card_id', 'customizations', 'source'}
                or card['zone'] not in counts or type(card['index']) is not int
                or card['index'] != counts[card['zone']]
                or not isinstance(card['card_id'], str) or not card['card_id'].isdecimal()
                or str(int(card['card_id'])) != card['card_id']):
            raise ValueError('confirmed deck physical definitions are incomplete or unordered')
        group = card['customizations']
        if (not isinstance(group, dict)
                or any(not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key
                       or type(value) is not int or value <= 0 for key, value in group.items())
                or card['source'] not in ('observed_detail', 'observed_badge_absence')
                or (card['source'] == 'observed_detail') != bool(group)):
            raise ValueError('confirmed deck definitions lack their complete original source')
        counts[card['zone']] += 1
    if (counts['手札'] not in (3,4,5) or not counts['山札'] or counts['捨札'] or counts['除外']
            or components['cards'].get('zone_counts') != counts):
        raise ValueError('confirmed deck source does not cover the complete opening inventory')
    return dict(source_id=source_id, cards=deepcopy(cards))


def _round_evidence(row, module, *, required=False):
    from live_decision.hif_session import bind_hif_round_context

    event = row.get('context_recovery') or {}
    saved = event.get('round_context')
    if saved is None:
        if required:
            raise ValueError('native round-two source lacks notebook round evidence')
        return None
    raw_run = row.get('observed_live_context')
    evidence_run = event.get('context')
    if raw_run != evidence_run:
        # Pending completion returns the dispatch's logical epoch. Its fresh
        # notebook proof keeps its own epoch, never rewritten to the dispatch.
        rule_parent = row.get('information_boundary_execution') is True and (
            row.get('native_inprocess_source') or row.get('native_rule_resume_source')
            or row.get('native_selected_source') or row.get('native_choice_source'))
        if (raw_run != row.get('execution_live_context') or not (row.get('native_pending_source') or rule_parent)
                or not isinstance(raw_run, dict) or not isinstance(evidence_run, dict)
                or any(raw_run.get(key) != evidence_run.get(key)
                       for key in ('idol', 'plan', 'max_stamina'))):
            raise ValueError('native round evidence belongs to another observation epoch')
    from ..hif_entry_round import SOURCE, verified_entry_receipt
    if event.get('round_source') == SOURCE:
        from ..hif_session import HifRoundContext
        if not verified_entry_receipt(row, event):
            raise ValueError('HIF opening round differs from its handoff or current observations')
        return HifRoundContext(**saved)
    if (not isinstance(evidence_run, dict)
            or type(event.get('game_pid')) is not int or event['game_pid'] <= 0
            or event.get('accepted') is not True or event.get('returned_to_live') is not True
            or event.get('game_pid') != row.get('endpoints', {}).get('game', {}).get('pid')):
        raise ValueError('native round notebook recovery is incomplete or from another process')
    groups = event.get('observations', ())
    identities = [group for group in groups if isinstance(group, (list, tuple))
                  and group and all(frame.get('phase') == 'identity' for frame in group)]
    if (len(identities) != 1 or not groups or not groups[-1]
            or any(frame.get('phase') != 'live' for frame in groups[-1])
            or any(not frame.get('frame_id') for frame in groups[-1])
            or len({frame.get('frame_id') for frame in groups[-1]}) != len(groups[-1])):
        raise ValueError('native round lacks complete notebook and returned Live frames')
    run = module.HifLiveRunContext(**evidence_run)
    metadata = bind_hif_round_context(identities[0], run=run,
        round_number=saved.get('round_number'))
    if not _same(asdict(metadata), saved):
        raise ValueError('native round metadata differs from its notebook observations')
    return metadata


def restore_selected_execution_epoch(row, parent):
    """Old selected-resume hosts omitted this metadata; prove its source first."""
    if (row.get('execution_live_context') is None and row.get('information_boundary_execution') is True
            and row.get('native_selected_source')):
        if (not isinstance(parent,dict) or row['native_selected_source'] != parent.get('id')
                or row.get('observed_live_context') != parent.get('observed_live_context')
                or row.get('context_recovery') != parent.get('context_recovery')
                or row['endpoints']['game']['pid'] != parent['endpoints']['game']['pid']):
            raise ValueError('selected execution epoch lacks its unchanged verified source')
        return dict(row,execution_live_context=parent['observed_live_context'])
    return row


def verify_native_round_context(row, module, parent=None, *, required=False):
    """Verify original evidence and round continuity; never relabel its epoch."""
    if parent is not None:
        row = restore_selected_execution_epoch(row,parent)
    previous = _round_evidence(parent, module) if parent is not None else None
    current = _round_evidence(row, module,
        required=required or (previous is not None and previous.round_number == 2))
    if parent is not None:
        old = (previous.round_number, previous.base_turn_count) if previous else (1, 9)
        new = (current.round_number, current.base_turn_count) if current else (1, 9)
        if new != old:
            raise ValueError('native continuation changed its Live round')
    return current


def load_native_round_context(runtime, row, module, *, run=None, trail=()):
    """Return original proof for the requested bound epoch, following a pending dispatch.

    A completed pending record may retain fresh notebook metadata in a newer
    epoch than its logical dispatch. Follow its verified source instead of
    changing either metadata epoch or its original frame identifiers.
    """
    from .native_settlement_resume import load_result, load_pending_source

    name = row.get('id')
    if not isinstance(name, str) or name in trail or diagnostic_depth_reached(trail):
        raise ValueError('native round provenance is cyclic or unbounded')
    target = run or module.HifLiveRunContext(**row['observed_live_context'])
    if (row.get('information_boundary_execution') is True and row.get('native_selected_source')
            and row.get('execution_live_context') is None):
        parent = load_result(runtime,row['native_selected_source'],allow_closed_transaction=True)
        row = restore_selected_execution_epoch(row,parent)
    metadata = verify_native_round_context(row, module)
    if metadata is None:
        return None
    if (metadata.run_id, metadata.revision) == (target.run_id, target.revision):
        return metadata
    source_name = row.get('native_pending_source')
    rule_parent = None
    if row.get('information_boundary_execution') is True:
        rule_parent = (row.get('native_inprocess_source') or row.get('native_rule_resume_source')
                       or row.get('native_selected_source') or row.get('native_choice_source'))
    if not source_name and rule_parent and row.get('execution_live_context') == asdict(target):
        source = load_result(runtime, rule_parent, allow_closed_transaction=True)
        if (source.get('information_boundary_execution') is True and source.get('native_selected_source')
                and source.get('execution_live_context') is None):
            selection_parent = load_result(runtime, source['native_selected_source'], allow_closed_transaction=True)
            source = restore_selected_execution_epoch(source, selection_parent)
        verify_native_round_context(row, module, source)
        return load_native_round_context(runtime, source, module, run=target, trail=(*trail, name))
    if not source_name or row.get('execution_live_context') != asdict(target):
        raise ValueError('native round has no original evidence for the bound epoch')
    bundle, _, _, source = load_pending_source(runtime, source_name, module)
    if ((bundle.run_id, bundle.revision) != (target.run_id, target.revision)
            or source.get('observed_live_context') != asdict(target)):
        raise ValueError('native round pending dispatch belongs to another epoch or process')
    if source['endpoints']['game']['pid'] != row['endpoints']['game']['pid']:
        from .native_settlement_resume import decoded_dispatched_action
        from .native_rule_process_recovery import verify_process_recovery
        action, _ = decoded_dispatched_action(source, bundle)
        verify_process_recovery(row, source, bundle, action, row.get('settlement', {}))
    verify_native_round_context(row, module, source)
    return load_native_round_context(runtime, source, module, run=target, trail=(*trail, name))


def load_native_opening_root(row, module=None):
    """Rebuild one complete opening; dispatch/selection verification stays upstream."""
    from live_decision.hif_visual import HifLiveRunContext
    from live_decision.high_score.opening import assemble_hif_opening

    if module is None:
        from utils.runtime_paths import resolve_runtime_path
        module = SimpleNamespace(HifLiveRunContext=HifLiveRunContext,
            CATALOG_PATH=resolve_runtime_path(Path('assets/data/skill_cards.csv'), Path('data/skill_cards.csv')))
    openings = [event for event in row.get('live_events', ())
                if event.get('event') == 'live_opening_state' and event.get('accepted') is True]
    if len(openings) != 1 or any(row.get(key) for key in PARENT_FIELDS):
        raise ValueError('native opening root is missing, duplicated, or has a parent source')
    opening = openings[0]
    run = module.HifLiveRunContext(**row['observed_live_context'])
    if (opening.get('run_id') != run.run_id or opening.get('revision') != run.revision
            or opening.get('strategy_state_bound') is not True):
        raise ValueError('native opening root is not completely bound to its epoch')
    turns = opening.get('components', {}).get('turns', {})
    round_two = turns.get('round_number') == 2 or turns.get('base_turn_count') == 12
    metadata = verify_native_round_context(row, module, required=round_two)
    bundle = assemble_hif_opening(run, core=opening['core'], components=opening['components'],
        hand_titles=opening['hand_titles'], data_root=module.CATALOG_PATH.parent,
        round_context=metadata, opening_skill_support=opening.get('opening_skill_support', ()),
        extended_hand_identity=opening.get('extended_hand_identity'),
        confirmed_deck_baseline=opening.get('confirmed_deck_baseline'))
    if not _same(asdict(bundle.state), opening.get('state')):
        raise ValueError('native opening state differs from its complete observations')
    return bundle, opening
