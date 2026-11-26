"""Offline fixtures for the shared cultivation model observation and receipt tests."""
import csv
import sys
import json
from types import ModuleType, SimpleNamespace
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from importlib.util import module_from_spec, spec_from_file_location

import pytest

from agent.live_decision.high_score.hif import HifSenseKernel
from agent.live_decision.high_score.contracts import (
    CardRef,
    PlanType,
    DeckBelief,
    DecisionKind,
    StrategyState,
    DecisionRequest,
    CapabilityReport,
)
from agent.live_decision.high_score.catalog_v2 import compile_item_catalog, compile_drink_catalog, compile_skill_card_catalog
from agent.live_decision.high_score.customizations import compile_observed_customizations
from agent.live_decision.high_score.gkms_shadow_adapter import ExamContext, TurnPlayHistory, build_shadow_observation
from agent.live_decision.high_score.gkms_shadow_mappings import CARD_FORMS

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ExamContext(
    'produce-008', 'AUDITION_MID1', 2, (3, 2, 1, 1, 3, 3, 2, 1, 3),
    (26200, 35380, 38150), (1392, 1088, 1871), 'i_card-jsna-3-020',
    'synthetic cultivation fixture',
)


@pytest.fixture(scope='module')
def kernel():
    data = ROOT / 'assets/data'
    cards = dict(compile_skill_card_catalog(data / 'skill_cards.csv', plan_type=PlanType.SENSE,
                                          include_free_card_ids=('23', '49')).cards)
    with (data / 'skill_cards.csv').open(encoding='utf-8-sig', newline='') as stream:
        metadata = {row['id']: row for row in csv.DictReader(stream)}
    customs = json.loads((data / 'customizations.json').read_text(encoding='utf-8'))
    for base, group in (('117', {'31': 1}), ('87', {'32': 1}), ('25', {'68': 3}),
                        ('597', {'14': 1, '54': 1}), ('597', {'54': 1}), ('295', {'3': 1, '52': 1})):
        variant = compile_observed_customizations(cards[base], metadata[base], customs, group)
        cards[variant.card_id] = variant
    drinks = compile_drink_catalog(data / 'p_drinks.csv', plan_type=PlanType.SENSE).drinks
    items = compile_item_catalog(data / 'p_items.csv', plan_type=PlanType.SENSE).items
    return HifSenseKernel(cards, drinks, items)


def initial():
    hand = tuple(CardRef(f'hand-{index}', card_id, True) for index, card_id in enumerate(('343', '163', '862')))
    deck = tuple(CardRef(f'deck-{index}', card_id, card_id not in ('23', '596', '704', '706', '708'))
                 for index, card_id in enumerate(('589', '363', '109', '762', '23', '117@31=1',
                                                  '87@32=1', '25@68=3')))
    return StrategyState(1, 9, 0, 36, 36, 0, 1, 5, 38.15,
        DeckBelief(hand + deck, hand, deck), concentration=6, good_condition_turns=6,
        future_score_multipliers=(35.38, 26.2, 26.2, 38.15, 38.15, 35.38, 26.2, 38.15),
        turn_attributes=('visual', 'dance', 'vocal', 'vocal', 'visual', 'visual', 'dance', 'vocal', 'visual'),
        persistent_effects_initialized=True, drinks=('22', '22', '12', '18'))


def request(state, *, card_slots=None, drink_slots=None):
    return DecisionRequest('fixture-root', PlanType.SENSE, DecisionKind.LIVE, state,
        CapabilityReport(True, True, True, True, False),
        card_slots=tuple((card.instance_id, i) for i, card in enumerate(state.deck.hand)) if card_slots is None else card_slots,
        drink_slots=tuple(enumerate(state.drinks)) if drink_slots is None else drink_slots)


def project(kernel, state=None, *, req=None, history=None):
    state = initial() if state is None else state
    req = request(state) if req is None else req
    history = TurnPlayHistory(state, (), 'complete fixture opening') if history is None else history
    return build_shadow_observation(req, kernel, CONTEXT, history)


def isolated(card_id, *, turn=1, removed=False, **changes):
    """One-step rule fixture, not a sampled game or score evaluation."""
    card = CardRef('source', card_id, bool(CARD_FORMS[card_id][1]))
    fillers = tuple(CardRef(f'fill-{i}', '23') for i in range(4))
    deck = DeckBelief((card, *fillers), () if removed else (card,), fillers,
                      removed=(card,) if removed else ())
    state = initial()
    multipliers = (state.score_multiplier, *state.future_score_multipliers)
    return replace(state, turn=turn, turns_remaining=10-turn, score_multiplier=multipliers[turn-1],
        future_score_multipliers=multipliers[turn:], turn_attributes=state.turn_attributes[turn-1:],
        concentration=100, good_condition_turns=20, actions_remaining=3, deck=deck, drinks=(), **changes)


class Network:
    def __init__(self, *tokens):
        self.tokens = iter(tokens)
        self.calls = []

    def decide(self, observation):
        self.calls.append(observation)
        token = next(self.tokens)
        return dict(action=token, probs=[float(action == token) for action in observation['legal']])


class Job:
    job_id = 7
    succeeded = True

    def __init__(self, image=None):
        self.image = image

    def wait(self):
        return self

    def get(self):
        return self.image


def load_action_module():
    from agent import live_decision, card_selection
    from agent.card_selection import model as card_selection_model

    maa = ModuleType("maa")
    maa_agent = ModuleType("maa.agent")
    maa_context = ModuleType("maa.context")
    maa_custom_action = ModuleType("maa.custom_action")
    maa_agent_server = ModuleType("maa.agent.agent_server")
    maa_context.Context = object

    class CustomAction:
        class RunArg:
            pass

    class AgentServer:
        @staticmethod
        def custom_action(_name):
            return lambda value: value

    maa_custom_action.CustomAction = CustomAction
    maa_agent_server.AgentServer = AgentServer
    utils = ModuleType("utils")
    utils.logger = SimpleNamespace(
        warning=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        exception=lambda *_args, **_kwargs: None,
    )
    aliases = {
        "maa": maa,
        "maa.agent": maa_agent,
        "maa.context": maa_context,
        "maa.custom_action": maa_custom_action,
        "maa.agent.agent_server": maa_agent_server,
        "utils": utils,
        "live_decision": live_decision,
        "card_selection": card_selection,
        "card_selection.model": card_selection_model,
    }
    path = ROOT / "agent/custom/action/live_decision.py"
    spec = spec_from_file_location("cultivation_live_action_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load cultivation action module")
    module = module_from_spec(spec)
    with patch.dict(sys.modules, aliases):
        spec.loader.exec_module(module)
    return module


def continuous_initial():
    cards = tuple(CardRef(str(i), card, True) for i, card in enumerate(('343', '163', '862', '589')))
    return StrategyState(1, 9, 0, 36, 36, 0, 1, 5, 38.15,
        DeckBelief(cards, cards[:3], cards[3:]), concentration=6, good_condition_turns=6,
        future_score_multipliers=(35.38, 26.2, 26.2, 38.15, 38.15, 35.38, 26.2, 38.15),
        turn_attributes=('visual', 'dance', 'vocal', 'vocal', 'visual', 'visual', 'dance', 'vocal', 'visual'),
        persistent_effects_initialized=True, drinks=('22', '22'))
