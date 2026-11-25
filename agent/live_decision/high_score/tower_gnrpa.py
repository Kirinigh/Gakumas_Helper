"""Bridge to the shared Hiro Rust runtime for Tower search and live rules."""

import os
import re
import sys
from types import SimpleNamespace
from pathlib import Path
from importlib import import_module

from agent.utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path

from .contracts import DecisionKind
from ..hiro_runtime import HIRO_REVISION
from .gkms_shadow_adapter import TurnPlayHistory, _card, build_shadow_observation
from .gkms_continuous_policy import _tower_legal_projection


class RustTowerStageBindingMissing(ValueError):
    """The selected floor has no usable binding for native search."""


def load_native_tower():
    if os.environ.get('GAKUMAS_EXAM_NATIVE') == '0':
        raise ValueError('idol_road_rust_backend_disabled')
    runtime = resolve_runtime_path(Path('.local/task275-rust/runtime') / HIRO_REVISION,
                                   Path('data/idol_road/rust_gnrpa') / HIRO_REVISION)
    if not runtime.is_dir():
        raise ValueError('idol_road_rust_runtime_missing')
    source = str(runtime / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)
    os.environ['GAKUMAS_PLAYER'] = str(runtime / 'vendor/gakumas-player')
    os.environ['HIRO_TOWER_BINDINGS'] = str(runtime / 'tower_bindings.json')
    os.environ['GAKUMAS_EXAM_RS_CACHE'] = str(runtime / 'native-master')
    try:
        rust = import_module('gakumas_exam_rs')
        import_module('hiro.sim')
        tables = import_module('hiro.bench_bridge.tower')
    except (ImportError, OSError) as error:
        raise ValueError(f'idol_road_rust_backend_unavailable:{error}') from error
    if not callable(getattr(rust, 'Exam', None)):
        raise ValueError('idol_road_rust_backend_incompatible')
    return rust, tables


def load_rust_tower():
    rust, tables = load_native_tower()
    advisor = import_module('hiro.search.tower_algo')
    if (not callable(getattr(rust, 'search', None)) or not callable(getattr(rust, 'plan_many', None))
            or not callable(getattr(advisor, 'TowerGnrpa', None))):
        raise ValueError('idol_road_rust_gnrpa_backend_incompatible')
    return advisor.TowerGnrpa, tables


def lookup_hiro_floor(tables, idol_card_id, layer):
    match = re.fullmatch(r'i_card-([a-z0-9]+)-\d-\d+', idol_card_id or '')
    if not match or type(layer) is not int or layer <= 0:
        return None, None
    tower_id = 'tower_001-' + match[1]
    return tower_id, tables.layer_exam(tower_id, layer)


def resolve_rust_tower_stage(tables, loadout):
    layer = loadout.overview.layer
    tower_id, stage = lookup_hiro_floor(tables, loadout.idol.idol_card_id, layer)
    if tower_id is None:
        raise ValueError('idol_road_gnrpa_stage_identity_unread')
    exam = (stage or {}).get('exam', {})
    if (stage is None or not exam.get('produce_exam_battle_config_id')
            or not exam.get('produce_exam_gimmick_effect_group_id')
            or not stage.get('mechanic') or not stage.get('rivals')
            or stage.get('turns') != loadout.overview.turns):
        raise RustTowerStageBindingMissing(f'idol_road_gnrpa_stage_binding_missing:{tower_id}:{layer}')
    return stage


def _deck_entry(card, kernel, *, native=False):
    form = _card(card, kernel=kernel, native=native)
    if form.get('g'):
        raise ValueError('idol_road_gnrpa_opening_growth_unmapped')
    card_id = form['id'] + (f"+{form['u']}" if form['u'] else '')
    return '/'.join([card_id, *(f'{key}*{count}' for key, count in form.get('c', ()))])


def _item_ids(tables, kernel, state, stage_items):
    if not state.items:
        return list(stage_items)
    items = tables.master().rows('ProduceItem')
    stage_ids = set(stage_items)
    stage_names = {row['name'] for row in items if row['id'] in stage_ids}
    result = list(stage_items)
    for key in state.items:
        name = kernel.items[key].name
        if name in stage_names:
            continue
        matches = {row['id'] for row in items if row.get('name') == name and not row['id'].startswith('pitem_tower_')}
        if len(matches) != 1:
            raise ValueError(f'idol_road_gnrpa_item_identity_unmapped:{key}:{name}')
        result.append(matches.pop())
    return result


def open_native_tower_search(rules, formation, initial, stage, *, provider=None):
    """Use Hiro's own search on a copy of the already observed Rust position."""
    factory = load_rust_tower()[0] if provider is None else provider
    engine = factory()
    loadout = dict(produce_id='tower', tower_id=stage['tower_id'], layer=stage['layer'],
        mechanic=stage['mechanic'], exam=stage['exam'], idol_card_id=formation.idol.idol_card_id,
        max_stamina=initial.max_stamina, starting_stamina=initial.stamina,
        score_bonus_permils=list(formation.overview.multiplier_permils),
        produce_cards=[_deck_entry(card, rules.kernel, native=True) for card in initial.deck.all_cards],
        produce_item_ids=[wire for _, wire in rules.items], rivals=[hi for _, hi in stage['rivals']])
    begun = False

    def search(native, observation):
        nonlocal begun
        args = SimpleNamespace(obs=observation, loadout=loadout, known_draws=())
        if not begun:
            loadout['drink_ids'] = list(observation['drinks'])
            engine.begin(args)
            begun = True
        engine.rx = native.clone()
        engine.prev = None
        return tuple(engine.decide(args).action)

    return search


class RustTowerPolicy:
    """Read observed states from the same host and dispatch only bound actions."""

    def __init__(self, kernel, context, initial, *, source, provider=None):
        from ..idol_road_runtime import TowerChoicePolicy

        factory, tables = load_rust_tower() if provider is None else provider
        stage = getattr(context, 'rust_stage', None) or resolve_rust_tower_stage(tables, context.formation)
        stage_items = tuple(stage['exam'].get('produce_item_ids') or ())
        self.engine = factory()
        self.kernel, self.context = kernel, context
        self.history = TurnPlayHistory(initial, (), source)
        self.secondary = TowerChoicePolicy(kernel)
        self.loadout = dict(produce_id='tower', tower_id=stage['tower_id'], layer=stage['layer'],
            mechanic=stage['mechanic'], exam=stage['exam'], idol_card_id=context.idol,
            max_stamina=initial.max_stamina, starting_stamina=initial.stamina, score_bonus_permils=list(context.permil),
            produce_cards=[_deck_entry(card, kernel) for card in initial.deck.all_cards],
            produce_item_ids=_item_ids(tables, kernel, initial, stage_items),
            rivals=[hi for _, hi in stage['rivals']])
        self.began = False

    def choose(self, request, legal):
        if request.decision_kind is DecisionKind.SECONDARY:
            return self.secondary.choose(request, legal)
        # Hiro uses the native engine's enchant state from the selected floor and items.
        projected = build_shadow_observation(request, self.kernel, self.context, self.history, include_enchants=False)
        if projected.status != 'supported':
            raise ValueError('idol_road_gnrpa_observation:' + '|'.join(projected.reason_codes))
        projected = _tower_legal_projection(projected, request, legal)
        args = SimpleNamespace(obs=projected.observation, loadout=self.loadout, known_draws=())
        if not self.began:
            self.loadout['drink_ids'] = list(projected.observation['drinks'])
            self.engine.begin(args)
            self.began = True
        output = self.engine.decide(args)
        return projected.bind_action(output.action, request=request, kernel=self.kernel)

    def observe_transition(self, before, action, step):
        self.history = self.history.advance_step(action, step)
