"""HIF non-battle transactions over observed resources and card instances."""
from __future__ import annotations

from dataclasses import asdict, replace, dataclass

from .engine import has_single_frame_id


def _name(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


@dataclass(frozen=True)
class CardInstance:
    instance_id: str
    card_id: str
    upgraded: bool
    customization: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class DrinkInstance:
    slot: int
    instance_id: str
    drink_id: str


@dataclass(frozen=True)
class HifLedgerState:
    run_id: str
    revision: int
    day: int
    phase: str
    cards: tuple[CardInstance, ...]
    card_count: int
    drinks: tuple[DrinkInstance, ...]
    drink_count: int
    currency: int
    stamina: int
    max_stamina: int
    counters: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PartialDeckLedgerState(HifLedgerState):
    """Only resolved card instances are listed; the observed total can be unknown."""
    card_count: int | None
    cards_complete: bool = False


@dataclass(frozen=True)
class LedgerEffect:
    """One observed, policy-selected result, including all known resource costs."""
    transaction_id: str
    target_key: str
    remove_cards: tuple[CardInstance, ...] = ()
    add_cards: tuple[CardInstance, ...] = ()
    remove_drinks: tuple[DrinkInstance, ...] = ()
    add_drinks: tuple[DrinkInstance, ...] = ()
    currency_delta: int = 0
    stamina_delta: int = 0
    max_stamina_delta: int = 0
    counter_deltas: tuple[tuple[str, int], ...] = ()
    day: int | None = None
    phase: str | None = None


@dataclass(frozen=True)
class CardObservationEffect(LedgerEffect):
    """Previously unidentified cards observed during normal nonbattle navigation."""
    observed_cards: tuple[CardInstance, ...] = ()


@dataclass(frozen=True)
class LedgerPending:
    effect: LedgerEffect
    before: HifLedgerState
    expected: HifLedgerState
    source_frame_ids: tuple[str, ...] = ()


def _counts(values, *, signed=False):
    if not isinstance(values, tuple) or any(
            not isinstance(row, tuple) or len(row) != 2 or not _name(row[0])
            or type(row[1]) is not int or (not signed and row[1] < 0) for row in values):
        raise ValueError('invalid HIF ledger counters')
    if len({row[0] for row in values}) != len(values):
        raise ValueError('duplicate HIF ledger counter')


def validate_ledger(state, *, card_catalog, drink_ids):
    if (not isinstance(state, HifLedgerState) or not _name(state.run_id)
            or not _integer(state.revision) or type(state.day) is not int or not 1 <= state.day <= 7
            or not _name(state.phase) or not isinstance(state.cards, tuple)
            or not isinstance(state.drinks, tuple) or not _integer(state.drink_count)
            or state.drink_count != len(state.drinks) or state.drink_count > 4
            or not _integer(state.currency) or not _integer(state.stamina)
            or not _integer(state.max_stamina, 1) or state.stamina > state.max_stamina):
        raise ValueError('incomplete or contradictory HIF ledger')
    if isinstance(state, PartialDeckLedgerState):
        if (state.cards_complete is not False or state.card_count is not None
                and not _integer(state.card_count, len(state.cards))):
            raise ValueError('partial HIF deck contradicts its observed card count')
    elif not _integer(state.card_count) or state.card_count != len(state.cards):
        raise ValueError('incomplete or contradictory HIF ledger')
    _counts(state.counters)
    for card in state.cards:
        if not isinstance(card, CardInstance) or not _name(card.instance_id) or type(card.upgraded) is not bool:
            raise ValueError('unresolved HIF card instance')
        entry = card_catalog.entries.get(card.card_id)
        if entry is None or entry.upgraded is not card.upgraded or entry.plan not in ('free', 'sense'):
            raise ValueError('HIF card identity/upgrade disagrees with existing catalog')
        _counts(card.customization)
    for drink in state.drinks:
        if (not isinstance(drink, DrinkInstance) or type(drink.slot) is not int
                or not 0 <= drink.slot < 4 or not _name(drink.instance_id) or drink.drink_id not in drink_ids):
            raise ValueError('unresolved HIF drink slot')
    keys = [row.instance_id for row in (*state.cards, *state.drinks)]
    if len(set(keys)) != len(keys) or len({row.slot for row in state.drinks}) != len(state.drinks):
        raise ValueError('ambiguous HIF inventory instances or slots')
    return state


def _apply_instances(current, removed, added, cls):
    if not isinstance(removed, tuple) or not isinstance(added, tuple):
        raise ValueError('invalid HIF inventory mutation')
    if any(not isinstance(row, cls) for row in (*removed, *added)):
        raise ValueError('invalid HIF mutation instance')
    if len({row.instance_id for row in removed}) != len(removed):
        raise ValueError('duplicate HIF removed instance')
    if any(row not in current for row in removed):
        raise ValueError('HIF target instance changed before transaction')
    result = tuple(row for row in current if row not in removed) + added
    return tuple(sorted(result, key=lambda row: row.instance_id if cls is CardInstance else row.slot))


def project_effect(state, effect, *, card_catalog, drink_ids):
    validate_ledger(state, card_catalog=card_catalog, drink_ids=drink_ids)
    if (not isinstance(effect, LedgerEffect) or not _name(effect.transaction_id) or not _name(effect.target_key)
            or any(type(value) is not int for value in
                   (effect.currency_delta, effect.stamina_delta, effect.max_stamina_delta))):
        raise ValueError('unverified HIF effect identity or resource delta')
    _counts(effect.counter_deltas, signed=True)
    counters = dict(state.counters)
    for key, delta in effect.counter_deltas:
        if key not in counters:
            raise ValueError('HIF resource counter was not observed')
        counters[key] += delta
    day = state.day if effect.day is None else effect.day
    if type(day) is not int or day not in (state.day, min(7, state.day + 1)):
        raise ValueError('unconfirmed HIF day change')
    known = state.cards
    if isinstance(effect, CardObservationEffect):
        if not isinstance(state, PartialDeckLedgerState):
            raise ValueError('a complete deck cannot acquire additional observed instances')
        known = _apply_instances(known, (), effect.observed_cards, CardInstance)
    cards = _apply_instances(known, effect.remove_cards, effect.add_cards, CardInstance)
    drinks = _apply_instances(state.drinks, effect.remove_drinks, effect.add_drinks, DrinkInstance)
    expected = replace(state, revision=state.revision + 1, day=day,
        phase=state.phase if effect.phase is None else effect.phase,
        cards=cards, card_count=(state.card_count + len(effect.add_cards) - len(effect.remove_cards)
            if state.card_count is not None else None), drinks=drinks, drink_count=len(drinks),
        currency=state.currency + effect.currency_delta,
        stamina=state.stamina + effect.stamina_delta,
        max_stamina=state.max_stamina + effect.max_stamina_delta,
        counters=tuple(sorted(counters.items())))
    return validate_ledger(expected, card_catalog=card_catalog, drink_ids=drink_ids)


class HifLedger:
    """In-memory candidate; host must save the candidate before publishing it."""
    def __init__(self, state, *, card_catalog, drink_ids, pending=None, settled=()):
        self.card_catalog, self.drink_ids = card_catalog, frozenset(drink_ids)
        self.state = validate_ledger(state, card_catalog=card_catalog, drink_ids=self.drink_ids)
        self.pending = pending
        self.settled = tuple(settled)
        if (len(set(self.settled)) != len(self.settled)
                or any(not _name(value) for value in self.settled)):
            raise ValueError('invalid HIF settled transaction identities')
        if pending is not None:
            expected = project_effect(state, pending.effect, card_catalog=card_catalog, drink_ids=self.drink_ids)
            if (pending.before != state or expected != pending.expected
                    or pending.effect.transaction_id in self.settled):
                raise ValueError('HIF pending transaction contradicts its baseline')
            source_ids = pending.source_frame_ids
            if (not isinstance(source_ids, tuple)
                    or source_ids and not has_single_frame_id(source_ids)):
                raise ValueError('invalid HIF pending source frame identities')

    def prepare(self, effect, *, frame_ids=()):
        if self.pending is not None or effect.transaction_id in self.settled:
            raise ValueError('HIF pending/settled transaction cannot be replayed')
        expected = project_effect(self.state, effect, card_catalog=self.card_catalog, drink_ids=self.drink_ids)
        return HifLedger(self.state, card_catalog=self.card_catalog, drink_ids=self.drink_ids,
            pending=LedgerPending(effect, self.state, expected, frame_ids), settled=self.settled)

    def confirm(self, receipt, *, transaction_id, frame_ids, observation=None, battle_census_frame_id=None):
        from .engine import ledger_frames_match
        # The native opening census supplies its own unchanged-return proof and
        # a single frame ID, independently of the route observation producer.
        opening_census = (battle_census_frame_id is not None and observation is None and self.pending is not None
            and self.pending.effect.target_key == 'finish_consult' and frame_ids == (battle_census_frame_id,))
        if (self.pending is None or self.pending.effect.transaction_id != transaction_id
                or not isinstance(frame_ids, tuple)
                or any(not _name(value) for value in frame_ids)
                or not (opening_census if battle_census_frame_id is not None else ledger_frames_match(observation, frame_ids))
                or set(frame_ids) & set(self.pending.source_frame_ids)):
            raise ValueError('HIF settlement lacks matching transaction or independent evidence')
        validate_ledger(receipt, card_catalog=self.card_catalog, drink_ids=self.drink_ids)
        # All values are exact. A smaller counter is never masked by max().
        if receipt != self.pending.expected:
            raise ValueError('HIF receipt contradicts projected inventory/resources')
        return HifLedger(receipt, card_catalog=self.card_catalog, drink_ids=self.drink_ids,
            settled=(*self.settled, transaction_id))

    def confirm_card_census(self, observed, *, frame_ids, variable_reward=False, observation=None):
        """Resolve a reward from its complete post-action census, keeping known instances."""
        pending = self.pending
        cards = tuple(sorted(observed, key=lambda card: card.instance_id))
        args = dict(transaction_id=pending.effect.transaction_id, frame_ids=frame_ids, observation=observation)
        if not isinstance(pending.before, PartialDeckLedgerState):
            return self.confirm(replace(pending.expected, cards=cards), **args)
        expected = pending.expected
        if not set(expected.cards).issubset(cards) or (expected.card_count is not None
                and (len(cards) < expected.card_count or not variable_reward and len(cards) != expected.card_count)):
            raise ValueError('complete post-reward census contradicts known permanent cards')
        settled = self.confirm(expected, **args)
        fields = dict(vars(expected), cards=cards, card_count=len(cards))
        del fields['cards_complete']
        return HifLedger(HifLedgerState(**fields), card_catalog=self.card_catalog, drink_ids=self.drink_ids,
            settled=settled.settled)

    def record(self):
        return dict(schema_version=1, state=asdict(self.state),
            pending=asdict(self.pending) if self.pending is not None else None,
            settled=list(self.settled))


def _fields(raw, cls):
    from dataclasses import fields
    if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(cls)}:
        raise ValueError('incomplete HIF ledger record')
    return dict(raw)


def _pairs(raw):
    if not isinstance(raw, (list, tuple)) or any(not isinstance(row, (list, tuple)) for row in raw):
        raise ValueError('invalid HIF ledger count record')
    return tuple(tuple(row) for row in raw)


def _cards(raw):
    if not isinstance(raw, (list, tuple)):
        raise ValueError('invalid HIF ledger cards')
    result = []
    for value in raw:
        row = _fields(value, CardInstance)
        row['customization'] = _pairs(row['customization'])
        result.append(CardInstance(**row))
    return tuple(result)


def _drinks(raw):
    if not isinstance(raw, (list, tuple)):
        raise ValueError('invalid HIF ledger drinks')
    return tuple(DrinkInstance(**_fields(row, DrinkInstance)) for row in raw)


def _state(raw):
    cls = PartialDeckLedgerState if isinstance(raw, dict) and raw.get('cards_complete') is False else HifLedgerState
    row = _fields(raw, cls)
    row['cards'], row['drinks'], row['counters'] = _cards(row['cards']), _drinks(row['drinks']), _pairs(row['counters'])
    return cls(**row)


def restore_ledger(raw, *, card_catalog, drink_ids):
    if (not isinstance(raw, dict) or set(raw) != {'schema_version', 'state', 'pending', 'settled'}
            or type(raw['schema_version']) is not int or raw['schema_version'] != 1
            or not isinstance(raw['settled'], (list, tuple))):
        raise ValueError('invalid HIF ledger checkpoint')
    state = _state(raw['state'])
    pending = None
    if raw['pending'] is not None:
        row = _fields(raw['pending'], LedgerPending)
        effect_cls = CardObservationEffect if isinstance(row['effect'], dict) and 'observed_cards' in row['effect'] else LedgerEffect
        effect = _fields(row['effect'], effect_cls)
        if effect_cls is CardObservationEffect:
            effect['observed_cards'] = _cards(effect['observed_cards'])
        for key in ('remove_cards', 'add_cards'):
            effect[key] = _cards(effect[key])
        for key in ('remove_drinks', 'add_drinks'):
            effect[key] = _drinks(effect[key])
        effect['counter_deltas'] = _pairs(effect['counter_deltas'])
        if not isinstance(row['source_frame_ids'], (list, tuple)):
            raise ValueError('invalid HIF pending source frames')
        pending = LedgerPending(effect_cls(**effect), _state(row['before']), _state(row['expected']),
                                tuple(row['source_frame_ids']))
    return HifLedger(state, card_catalog=card_catalog, drink_ids=drink_ids,
                     pending=pending, settled=raw['settled'])
