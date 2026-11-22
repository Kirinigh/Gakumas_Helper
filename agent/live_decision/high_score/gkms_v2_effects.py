"""Keep model-specific effect provenance from verified transitions."""
from dataclasses import dataclass

from .gkms_master_forms import drink_id, card_form


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
        if source.startswith('drink:'):
            local = source.removeprefix('drink:')
            wire = drink_id(self.kernel.drinks.get(local))
            return self.drink_effects.get(wire, ())
        card = cards.get(source)
        if card is None:
            return ()
        form = card_form(self.kernel.cards.get(card.card_id), card.upgraded)
        return self.card_effects.get(form[:2], ()) if form else ()

    def direct_rule(self, cards, event, kind):
        rows = (self.effects.get(key, {}) for key in self._producer_effects(cards, event.source_id))
        rules = {(row['produceCardSearchId'], row['effectValue1'], row['effectCount'], row['effectTurn'])
                 for row in rows if row.get('effectType') == 'ProduceExamEffectType_' + kind
                 and row.get('produceCardSearchId') in self.search_ids}
        # Multiple co-installed effects require producer evidence per layer.
        return next(iter(rules)) if len(rules) == 1 else None


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
        if value.cost or value.replay:
            from ..hif_gkms_runtime import uinet21_master

            searches = {row['id'] for row in uinet21_master()['ProduceCardSearch']}
            if any(effect and effect[0] not in searches for effect in (value.cost, value.replay)):
                raise ValueError('v2_effect_cache_search_missing_from_master')
        return value

    def matches(self, state):
        cost, replay = self.cost, self.replay
        def counted(row, total, size, value):
            if row is None:
                return total > 0
            if not row:
                return total == 0
            return (len(row) == size and isinstance(row[0], str) and bool(row[0])
                    and type(row[1]) is int and row[1] == value
                    and type(row[2]) is int and row[2] == total > 0)

        return (counted(cost, state.nullify_cost_cards, 3, 0)
                and counted(replay, state.double_card_effects_remaining, 4, 1)
                and (not replay or (type(replay[3]) is int and replay[3] >= -1)))

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
        return V2EffectState(
            () if count == 0 else cost if cost and cost[2] == count else None,
            () if double == 0 else replay if replay and replay[2] == double else None)
