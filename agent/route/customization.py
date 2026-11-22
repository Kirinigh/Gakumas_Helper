"""HIF cumulative customization, adapted to local identity/ledger transactions.

The host supplies observed menu facts and explicit goals. No target, price,
remaining count, input authority or configuration is inferred here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from dataclasses import replace, dataclass

from .engine import Outcome, Decision, propose_unique, validate_observation
from .ledger import HifLedger, LedgerEffect
from .nonbattle_observation import parse_customization_success

try:
    from agent.arena_winrate.catalog import ArenaEntityCatalog
except ModuleNotFoundError:
    from arena_winrate.catalog import ArenaEntityCatalog


@dataclass(frozen=True)
class CustomizationMenu:
    card_instance_id: str
    card_id: str
    goals: tuple[tuple[str, int], ...]
    remaining: tuple[tuple[str, int], ...]
    prices: tuple[tuple[str, int | None], ...]
    total_remaining: int
    selected_option_id: str | None
    option_targets: tuple[tuple[str, str], ...]
    execute_target_key: str


@dataclass(frozen=True)
class CustomizationPlan:
    decision: Decision
    effect: LedgerEffect | None = None


@dataclass(frozen=True)
class CustomizationResult:
    card_instance_id: str
    card_id: str
    frame_ids: tuple[str, ...]


def observe_customization_result(card, *, frame_ids, rows_by_frame, card_catalog):
    from .engine import has_single_frame_id
    if not has_single_frame_id(frame_ids) or len(rows_by_frame) != 1:
        raise ValueError('customization result needs one current frame')
    entry = card_catalog.entries.get(card.card_id)
    if entry is None or not parse_customization_success(rows_by_frame[0], card_name=entry.name):
        raise ValueError('current card customization success was not confirmed')
    return CustomizationResult(card.instance_id, card.card_id, frame_ids)


class CustomizationCatalog:
    """Use existing local card/option IDs, maxima and level-dependent prices."""
    def __init__(self, cards, options):
        self.cards = {str(row['id']): dict(row) for row in cards}
        self.options = {str(row['id']): dict(row) for row in options}
        if len(self.cards) != len(cards) or len(self.options) != len(options):
            raise ValueError('duplicate HIF customization catalog ID')
        # Reuse the existing read-only legality index. Do not create a second
        # interpretation of which permanent options belong to a business ID.
        indexed_cards = [dict(row, id=int(row['id']),
                             upgraded=str(row.get('upgraded')).upper() == 'TRUE') for row in cards]
        self.identity_catalog = ArenaEntityCatalog(indexed_cards, options)

    @classmethod
    def load(cls, card_path: str | Path, option_path: str | Path):
        with Path(card_path).open(encoding='utf-8-sig', newline='') as stream:
            cards = tuple(csv.DictReader(stream))
        options = json.loads(Path(option_path).read_text(encoding='utf-8'))
        if not isinstance(options, list):
            raise ValueError('invalid HIF customization catalog')
        return cls(cards, options)

    def rules(self, card_id):
        card = self.cards.get(card_id)
        if card is None or str(card.get('upgraded')).upper() != 'TRUE':
            raise ValueError('customization requires one known upgraded card')
        ids = tuple(str(value) for value in self.identity_catalog.available_customization_ids(int(card_id)))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('HIF card has no complete customization options')
        rules = {}
        for key in ids:
            row = self.options.get(key)
            if row is None or type(row.get('max')) is not int or row['max'] <= 0:
                raise ValueError('HIF customization option/max missing')
            raw_price = row.get('price')
            if type(raw_price) is int:
                prices = (raw_price,) * row['max']
            elif isinstance(raw_price, str):
                prices = tuple(int(value) for value in raw_price.split(','))
            else:
                raise ValueError('HIF customization price missing')
            if len(prices) != row['max'] or any(value < 0 for value in prices):
                raise ValueError('HIF customization price levels incomplete')
            rules[key] = (row['max'], prices)
        return rules


def _pairs(values, *, numbers=False):
    if not isinstance(values, tuple) or any(not isinstance(row, tuple) or len(row) != 2
            or not isinstance(row[0], str) or not row[0] for row in values):
        raise ValueError('incomplete HIF customization menu')
    result = dict(values)
    if len(result) != len(values):
        raise ValueError('duplicate HIF customization menu identity')
    if numbers and any(type(value) is not int or value < 0 for value in result.values()):
        raise ValueError('unknown HIF customization count')
    return result


def observe_customization_capacity(observation, ledger, menu, catalog):
    """Bind a first observed capacity; never refill an existing counter.

    The caller binds the menu to its current capture and persists the candidate
    atomically. All menu facts pass the same planner used before real input.
    """
    if not isinstance(menu, CustomizationMenu) or ledger.pending is not None or not menu.goals:
        raise ValueError('customization capacity needs an explicit settled menu')
    key = 'custom_remaining:' + menu.card_instance_id
    counters = dict(ledger.state.counters)
    if key in counters:
        if type(menu.total_remaining) is not int or counters[key] != menu.total_remaining:
            raise ValueError('customization capacity cannot replace an observed counter')
        candidate = ledger
    else:
        if type(menu.total_remaining) is not int or menu.total_remaining < 0:
            raise ValueError('customization capacity was not independently observed')
        counters[key] = menu.total_remaining
        state = replace(ledger.state, revision=ledger.state.revision + 1, counters=tuple(sorted(counters.items())))
        candidate = HifLedger(state, card_catalog=ledger.card_catalog, drink_ids=ledger.drink_ids,
            settled=ledger.settled)
    plan = plan_customization(observation, candidate, menu, catalog)
    if plan.decision.outcome == Outcome.STOP:
        raise ValueError(plan.decision.reason)
    return candidate


def plan_customization(observation, ledger, menu, catalog):
    """Select one known option or execute one already selected option.

    Complete goals pause rather than leaving the page by a generic back click.
    The host prepares/saves the returned effect before the ordinary Maa bridge
    dispatches, and confirms all resulting resources before another operation.
    """
    def stop(reason):
        return CustomizationPlan(Decision(Outcome.STOP, reason))

    error = validate_observation(observation)
    if error:
        return stop(error)
    if (observation.scene != 'card_customize' or not isinstance(menu, CustomizationMenu)
            or ledger.pending is not None or observation.run_id != ledger.state.run_id
            or observation.day != ledger.state.day):
        return stop('customization_page_or_pending_mismatch')
    matches = [card for card in ledger.state.cards if card.instance_id == menu.card_instance_id]
    if len(matches) != 1 or matches[0].card_id != menu.card_id or matches[0].upgraded is not True:
        return stop('customization_card_instance_changed')
    card = matches[0]
    try:
        rules = catalog.rules(menu.card_id)
        remaining, goals = _pairs(menu.remaining, numbers=True), _pairs(menu.goals, numbers=True)
        prices, targets = _pairs(menu.prices), _pairs(menu.option_targets)
        current = _pairs(card.customization, numbers=True)
    except (ValueError, TypeError, KeyError):
        return stop('customization_menu_or_catalog_incomplete')
    if set(remaining) != set(rules) or set(prices) != set(rules) or set(targets) != set(rules):
        return stop('customization_options_incomplete')
    if (set(goals) - set(rules) or set(current) - set(rules)
            or any(value > rules[key][0] for key, value in goals.items())
            or any(not isinstance(value, str) or not value for value in targets.values())
            or len(set(targets.values())) != len(targets)):
        return stop('customization_target_configuration_invalid')
    for key, (maximum, _) in rules.items():
        if remaining[key] > maximum or maximum - remaining[key] != current.get(key, 0):
            return stop('customization_count_contradiction')
    counter_key = 'custom_remaining:' + card.instance_id
    counters = dict(ledger.state.counters)
    if (type(menu.total_remaining) is not int or menu.total_remaining < 0
            or counters.get(counter_key) != menu.total_remaining
            or menu.total_remaining > sum(remaining.values())):
        return stop('customization_total_contradiction')
    if menu.selected_option_id is not None and menu.selected_option_id not in rules:
        return stop('customization_selected_option_unknown')
    eligible = tuple(key for key, goal in menu.goals if current.get(key, 0) < goal)
    if not eligible:
        return CustomizationPlan(Decision(Outcome.PAUSE, 'customization_goals_reached'))
    if menu.total_remaining == 0:
        return CustomizationPlan(Decision(Outcome.PAUSE, 'customization_capacity_exhausted'))
    option = eligible[0]
    price = prices[option]
    if type(price) is not int or price != rules[option][1][current.get(option, 0)]:
        return stop('customization_price_unconfirmed_or_changed')
    if price > ledger.state.currency:
        return CustomizationPlan(Decision(Outcome.PAUSE, 'customization_insufficient_currency'))
    selecting = menu.selected_option_id != option
    key = targets[option] if selecting else menu.execute_target_key
    kind = 'customization_option' if selecting else 'execute_customization'
    candidates = tuple(target for target in observation.targets if target.key == key and target.kind == kind)
    # Rule identity includes the complete authorized cumulative goal and exact
    # current price/count. A refresh cannot silently change the consumption.
    rule = f'hif-customize-v1:{ledger.state.revision}:{card.instance_id}:{option}:{current.get(option, 0)}:{price}:{menu.goals}'
    decision = propose_unique(observation, candidates, rule,
        ('card_customize',) if selecting else ('card_customize_result', 'card_customize'))
    if decision.outcome != Outcome.ACTION:
        return CustomizationPlan(decision)
    effect = LedgerEffect(f'{ledger.state.run_id}:{ledger.state.revision}:{kind}:{option}', key)
    if not selecting:
        updated = dict(current)
        updated[option] = updated.get(option, 0) + 1
        effect = replace(effect, remove_cards=(card,),
            add_cards=(replace(card, customization=tuple(sorted((k, v) for k, v in updated.items() if v))),),
            currency_delta=-price, counter_deltas=((counter_key, -1),))
    return CustomizationPlan(decision, effect)
