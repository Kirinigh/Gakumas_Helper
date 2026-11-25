"""Keep model-specific effect provenance from verified transitions."""
from dataclasses import dataclass

from .contracts import ActionKind
from .gkms_shadow_mappings import DRINK_IDS, CARD_FORMS

VINEGAR = '22'
COST_SEARCH = 'p_card_search-deck_all'
REPLAY_SEARCH = 'p_card_search-n-r-sr-ssr-playing'
_COST = (COST_SEARCH, 0, 2)
_REPLAY = (REPLAY_SEARCH, 1, 1, -1)


def _matching_row(row, kind, fields):
    return (row.get('effectType') == 'ProduceExamEffectType_' + kind
        and all(row.get(key) == value for key, value in fields.items()))


def effect_sources(tables):
    """Verify the producer links and exact scope, value, uses and lifetime."""
    effects = {row['id']: row for row in tables['ProduceExamEffect']}
    searches = {row['id'] for row in tables['ProduceCardSearch']}
    drinks = {row['id']: row for row in tables['ProduceDrink']}
    drink_effects = {row['id']: row for row in tables['ProduceDrinkEffect']}
    vinegar = drinks[DRINK_IDS[VINEGAR]]
    matches = []
    for drink_effect in vinegar['produceDrinkEffectIds']:
        exam_id = drink_effects[drink_effect]['produceExamEffectId']
        if _matching_row(effects[exam_id], 'ExamSearchPlayCardStaminaConsumptionChange',
                dict(produceCardSearchId=COST_SEARCH, effectValue1=0, effectCount=2, effectTurn=-1)):
            matches.append(exam_id)

    replay_forms = set()
    local_forms = {CARD_FORMS[cid][:2]: cid for cid in ('162', '163')}
    for card in tables['ProduceCard']:
        form = (card['id'], card['upgradeCount'])
        if form not in local_forms:
            continue
        verified = [entry for entry in card.get('playEffects', ())
            if _matching_row(effects[entry['produceExamEffectId']], 'ExamCardSearchEffectPlayCountBuff',
                dict(produceCardSearchId=REPLAY_SEARCH, effectValue1=1, effectCount=1, effectTurn=-1))]
        if len(verified) == 1:
            replay_forms.add(local_forms[form])
    return (len(matches) == 1 and COST_SEARCH in searches,
        frozenset(replay_forms) if REPLAY_SEARCH in searches else frozenset())


@dataclass(frozen=True)
class V2EffectState:
    # Empty tuple means verified absent. None means aggregate count without provenance.
    cost: tuple | None
    replay: tuple | None

    @classmethod
    def initial(cls, state):
        return cls(None if state.nullify_cost_cards else (),
                   None if state.double_card_effects_remaining else ())

    @classmethod
    def from_cache(cls, rows, state):
        if (not isinstance(rows, dict) or set(rows) != {'cost', 'replay'}
                or any(row is not None and not isinstance(row, (tuple, list)) for row in rows.values())):
            raise ValueError('v2_effect_cache_invalid')
        value = cls(*(None if rows[key] is None else tuple(rows[key]) for key in ('cost', 'replay')))
        if not value.matches(state):
            raise ValueError('v2_effect_cache_disagrees_with_state')
        return value

    def matches(self, state):
        count = state.nullify_cost_cards
        double = state.double_card_effects_remaining
        return ((self.cost is None and count > 0 or self.cost == () and count == 0
                 or 0 < count <= _COST[2] and self.cost == (*_COST[:2], count))
            and (self.replay is None and double > 0 or self.replay == () and double == 0
                 or double == _REPLAY[2] and self.replay == _REPLAY))

    def advance(self, before, action, after, events, sources):
        vinegar, replay_forms = sources
        cards = {card.instance_id: card.card_id for card in before.deck.all_cards}
        from_vinegar = (
            action.kind is ActionKind.USE_DRINK and action.drink_id == VINEGAR
            and any(event.event_type == 'CONSUME_DRINK' and event.source_id == 'drink:22' for event in events)
            or action.kind is ActionKind.MOVE_CARD and before.pending_choice is not None
            and before.pending_choice.source_id == 'drink:22')
        cost, replay = self.cost, self.replay
        for event in events:
            if event.event_type == 'NULLIFY_COST_CARDS':
                cost = (_COST if cost == () and vinegar and from_vinegar
                        and event.source_id == 'drink:22' and event.amount == _COST[2] else None)
            elif event.event_type == 'NULLIFY_CARD_COST':
                if cost:
                    cost = (*cost[:2], cost[2] - 1) if cost[2] > 1 and event.amount == 1 else ()
            elif event.event_type == 'DOUBLE_CARD_EFFECT':
                replay = (_REPLAY if replay == () and event.amount == _REPLAY[2]
                          and cards.get(event.source_id) in replay_forms else None)
            elif event.event_type == 'CONSUME_DOUBLE_CARD_EFFECT':
                if replay:
                    replay = ()
        # The kernel carries totals. Provenance is usable only when its events
        # reconcile with that confirmed state. Stacked layers need separate evidence.
        count, double = after.nullify_cost_cards, after.double_card_effects_remaining
        return V2EffectState(
            () if count == 0 else cost if cost and cost[2] == count else None,
            () if double == 0 else replay if replay and replay[2] == double else None)


class CatalogV2RuleSources:
    """Resolve an executed producer to raw 2.1 rules in the pinned master."""

    def __init__(self, kernel, tables):
        self.kernel = kernel
        self.effects = {row['id']: row for row in tables['ProduceExamEffect']}
        self.card_effects = {(row['id'], row['upgradeCount']): tuple(
            link['produceExamEffectId'] for link in row.get('playEffects', ())) for row in tables['ProduceCard']}
        drink_effects = {row['id']: row['produceExamEffectId'] for row in tables['ProduceDrinkEffect']}
        self.drink_effects = {row['id']: tuple(drink_effects[key] for key in row['produceDrinkEffectIds'])
                              for row in tables['ProduceDrink']}
        self.search_ids = {row['id'] for row in tables['ProduceCardSearch']}

    def _producer_effects(self, cards, source):
        from .gkms_master_forms import drink_id, card_form

        if source.startswith('drink:'):
            local = source.removeprefix('drink:')
            wire = DRINK_IDS.get(local) or drink_id(self.kernel.drinks.get(local))
            return self.drink_effects.get(wire, ())
        card = cards.get(source)
        if card is None:
            return ()
        form = CARD_FORMS.get(card.card_id) or card_form(self.kernel.cards.get(card.card_id), card.upgraded)
        return self.card_effects.get(form[:2], ()) if form else ()

    def direct_rule(self, cards, event, kind):
        rows = (self.effects.get(key, {}) for key in self._producer_effects(cards, event.source_id))
        rules = {(row['produceCardSearchId'], row['effectValue1'], row['effectCount'], row['effectTurn'])
                 for row in rows if row.get('effectType') == 'ProduceExamEffectType_' + kind
                 and row.get('produceCardSearchId') in self.search_ids}
        # Multiple co-installed effects require producer evidence per layer.
        return next(iter(rules)) if len(rules) == 1 else None


@dataclass(frozen=True)
class CatalogV2EffectState(V2EffectState):
    """Keep the target, set value, count and lifetime observed from the producer."""

    def matches(self, state):
        cost, replay = self.cost, self.replay
        return ((cost is None and state.nullify_cost_cards > 0
                 or cost == () and state.nullify_cost_cards == 0
                 or bool(cost) and cost[2] == state.nullify_cost_cards > 0)
                and (replay is None and state.double_card_effects_remaining > 0
                     or replay == () and state.double_card_effects_remaining == 0
                     or bool(replay) and replay[2] == state.double_card_effects_remaining > 0))

    def advance(self, before, action, after, events, sources):
        cards = {card.instance_id: card for card in before.deck.all_cards + after.deck.all_cards}
        cost, replay = self.cost, self.replay
        for event in events:
            name = event.event_type
            if name == 'NULLIFY_COST_CARDS':
                rule = sources.direct_rule(cards, event, 'ExamSearchPlayCardStaminaConsumptionChange')
                cost = rule[:3] if cost == () and rule and rule[1:] == (0, event.amount, -1) else None
            elif name == 'NULLIFY_CARD_COST' and cost:
                cost = (*cost[:2], cost[2] - 1) if cost[2] > 1 else ()
            elif name == 'DOUBLE_CARD_EFFECT':
                rule = sources.direct_rule(cards, event, 'ExamCardSearchEffectPlayCountBuff')
                replay = rule if replay == () and rule and rule[1] == 1 and rule[2] == event.amount else None
            elif name == 'CONSUME_DOUBLE_CARD_EFFECT' and replay:
                replay = (*replay[:2], replay[2] - 1, replay[3]) if replay[2] > 1 else ()
            elif name == 'END_TURN' and replay and replay[3] >= 0:
                replay = (*replay[:3], replay[3] - 1) if replay[3] > 1 else ()
        count, double = after.nullify_cost_cards, after.double_card_effects_remaining
        return CatalogV2EffectState(
            () if count == 0 else cost if cost and cost[2] == count else None,
            () if double == 0 else replay if replay and replay[2] == double else None)
