"""Bind the fixed user selection to the current run and exact owned instances.

Goals are configuration, never observed counts, prices, coordinates or input
permission. Menu acquisition remains the current pixel producer's obligation.
"""
from __future__ import annotations

import re
import json
from copy import deepcopy
from pathlib import Path
from dataclasses import replace, dataclass

from .hif import active_hif_role
from .engine import has_single_frame_id, source_frames_match, validate_observation
from .ownership import write_hif_record
from .customization import CustomizationMenu
from .customization_selection import CustomizationSelection, load_customization_selection, parse_customization_selection


@dataclass(frozen=True)
class CustomizationQueueEntry:
    card_id: str
    goals: tuple[tuple[str, int], ...]
    status: str
    instance_id: str | None = None


def queue_for_ledger(selection, ledger):
    if not isinstance(selection, CustomizationSelection) or ledger.pending is not None:
        raise ValueError('customization queue requires explicit goals and a settled ledger')
    result = []
    for goal in selection.cards if selection.enabled else ():
        owned = tuple(card for card in ledger.state.cards if card.card_id == goal.card_id and card.upgraded)
        if not owned:
            status, chosen = 'not_owned', None
        elif any(all(dict(card.customization).get(key, 0) >= count for key, count in goal.goals) for card in owned):
            status, chosen = 'complete', None
        elif len(owned) != 1:
            status, chosen = 'ambiguous_instances', None
        else:
            status, chosen = 'pending', owned[0].instance_id
        result.append(CustomizationQueueEntry(goal.card_id, goal.goals, status, chosen))
    return tuple(result)


def menu_for_selection(selection, ledger, menu):
    if not isinstance(menu, CustomizationMenu):
        raise ValueError('customization menu producer is not bound')
    queue = queue_for_ledger(selection, ledger)
    active = next((row for row in queue if row.status not in ('not_owned', 'complete')), None)
    if active is None:
        raise ValueError('customization queue has no unfinished owned card')
    if active.status == 'ambiguous_instances':
        raise ValueError('customization queue has ambiguous physical instances')
    if (menu.card_id, menu.card_instance_id) != (active.card_id, active.instance_id):
        raise ValueError('customization current menu is not the next configured physical instance')
    return replace(menu, goals=active.goals)


def _payload(selection):
    return dict(enabled=selection.enabled, profession=selection.profession, source=selection.source,
        source_revision=selection.source_revision,
        cards=[dict(card_id=card.card_id, goals=[list(row) for row in card.goals]) for card in selection.cards])


def _restore(value, *, host, catalog):
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'run_id', 'source_revision', 'frame_ids', 'selection'}
            or value['schema_version'] != 1 or type(value['schema_version']) is not int
            or value['run_id'] != host.run_id or type(value['source_revision']) is not int
            or value['source_revision'] > host.ledger.state.revision
            or not has_single_frame_id(value['frame_ids'])):
        raise ValueError('customization selection source is incomplete')
    return restore_selection(value['selection'], catalog=catalog)


def restore_selection(data, *, catalog):
    """Restore the same canonical goals from start or interval configuration."""
    if (not isinstance(data, dict) or set(data) != {'enabled', 'profession', 'source', 'source_revision', 'cards'}
            or data['profession'] != '集中' or type(data['enabled']) is not bool
            or data['source'] not in ('disabled', 'missing', 'profiles', 'legacy_cards')
            or not isinstance(data['cards'], list)):
        raise ValueError('customization retained configuration is invalid')
    selected = {}
    for row in data['cards']:
        if (not isinstance(row, dict) or set(row) != {'card_id', 'goals'}
                or not isinstance(row['card_id'], str) or row['card_id'] in selected
                or not isinstance(row['goals'], list) or any(not isinstance(pair, list) or len(pair) != 2 for pair in row['goals'])):
            raise ValueError('customization retained queue is incomplete')
        goals = dict(row['goals'])
        if len(goals) != len(row['goals']):
            raise ValueError('customization retained goals are ambiguous')
        selected[row['card_id']] = goals
    result = parse_customization_selection(dict(sourceRevision=data['source_revision'], profiles={'集中': selected}),
        profession='集中', enabled=data['enabled'], catalog=catalog)
    result = replace(result, source=data['source'], source_revision=data['source_revision'])
    if _payload(result) != data:
        raise ValueError('customization retained goals are not canonical')
    return result


def configured_selection(config, *, catalog=None):
    """Read the actual fixed configuration before allocating a cultivation."""
    enabled = config.customization_enabled is True
    if not enabled:
        return parse_customization_selection(None, profession='集中', enabled=False, catalog=catalog)
    from utils.runtime_paths import resolve_runtime_path
    if catalog is None:
        from .customization import CustomizationCatalog
        catalog = CustomizationCatalog.load(
            resolve_runtime_path(Path('assets/data/skill_cards.csv'), Path('data/skill_cards.csv')),
            resolve_runtime_path(Path('assets/data/customizations.json'), Path('data/customizations.json')))
    path = resolve_runtime_path(Path('config/hif/hif_custom_card_selection.json'),
        Path('config/hif/hif_custom_card_selection.json'))
    return load_customization_selection(path, profession='集中', enabled=True, catalog=catalog)


def selection_for_run(host, config, page, *, catalog):
    """The actual supply/menu caller consumes this under the existing host lock."""
    retained = (host.handoff or {}).get('nonbattle_context', {}).get('customization_selection')
    explicit = config.customization_enabled
    if retained is not None:
        selection = _restore(retained, host=host, catalog=catalog)
        if explicit is not None and explicit != selection.enabled:
            raise ValueError('customization switch differs from the retained cultivation')
        return selection
    from .new_run import start_record
    start = start_record(host.handoff) or {}
    configured = start.get('customization_selection')
    selection = (restore_selection(configured, catalog=catalog) if configured is not None
        else configured_selection(config, catalog=catalog))
    if explicit is not None and explicit != selection.enabled:
        raise ValueError('customization switch differs from the retained cultivation')
    if not selection.enabled or config.mode != 'advance':
        return selection
    observation, owner = page.observation, host.ownership or {}
    if (not host._active or page is not host._page or host.ledger.pending is not None
            or host.hooks.transition_gate.pending is not None or host.hooks.transition_gate.stopped
            or owner.get('owner') != 'task600' or owner.get('phase') != 'nonbattle'
            or owner.get('game_pid') != host.game_pid or page.baseline != host.ledger.state
            or observation.scene not in ('interval', 'card_customize')
            or (observation.idol, observation.plan, observation.scenario) != (*active_hif_role(host), 'HIF')
            or validate_observation(observation) is not None
            or not 0 <= observation.capture_age_ms + host.clock.now() - observation.revision <= 1000
            or not isinstance(page.frame_ids, tuple) or not source_frames_match(observation, page.frame_ids)
            or any(not isinstance(key, str) or re.fullmatch(r'route-frame-' + str(host.game_pid) + r'-[0-9]+', key) is None
                for key in page.frame_ids)):
        raise ValueError('customization selection needs the fresh settled current interval')
    captures = [int(key.rsplit('-', 1)[1]) for key in page.frame_ids]
    if (captures != sorted(captures) or captures[-1] > observation.revision
            or not 0 <= host.clock.now() - captures[0] <= 1000):
        raise ValueError('customization selection frames are stale or unordered')
    original = host._prepared_business_record()
    if json.loads(host.path.read_bytes()) != original:
        raise ValueError('customization selection source record changed')
    handoff = deepcopy(host.handoff or {})
    handoff.setdefault('nonbattle_context', {})['customization_selection'] = dict(schema_version=1,
        run_id=host.run_id, source_revision=host.ledger.state.revision, frame_ids=list(page.frame_ids),
        selection=_payload(selection))
    write_hif_record(host.path, dict(original, handoff=handoff))
    host.handoff = handoff
    return selection
