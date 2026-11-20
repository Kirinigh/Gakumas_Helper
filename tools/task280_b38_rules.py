"""Isolated rule definitions for the B38 referee adapter, not a product registry."""

from __future__ import annotations

from dataclasses import replace

from tools.task280_shared_rules import SUMMER, summer_memory
from agent.live_decision.high_score.hif import HifSenseKernel
from agent.live_decision.high_score.kernel import EffectKind, EffectSpec, ItemDefinition
from agent.live_decision.high_score.contracts import PlanType, SchedulePhase
from agent.live_decision.high_score.support_schedule import ordinary_counted_score_definition

SPOTLIGHT = '762@b38'
MICROPHONE = '285'


def spotlight():
    ordinary = ordinary_counted_score_definition()
    concentration, *remaining = ordinary.effects
    return replace(ordinary, card_id=SPOTLIGHT, canonical_card_id='762',
                   effects=(replace(concentration, amount=concentration.amount + 2), *remaining))


def microphone():
    return ItemDefinition(MICROPHONE, '世界を救ったマイク+', plan=PlanType.SENSE, persistent_effects=(
        EffectSpec(EffectKind.SCHEDULE, schedule_phase=SchedulePhase.CARD_USED, delay_turns=0,
                   schedule_limit=3, scheduled_effects=(EffectSpec(EffectKind.HALF_COST_TURNS, amount=2),)),
    ))


class B38RuleKernel(HifSenseKernel):
    def __init__(self, cards, drinks, items, *, focus_card_ids, memory_abilities=None):
        self.focus_card_ids = frozenset(focus_card_ids)
        if not self.focus_card_ids <= (set(cards) | {SPOTLIGHT, SUMMER}):
            raise ValueError('b38_focus_card_filter_has_unbound_ids')
        super().__init__({**cards, SPOTLIGHT: spotlight(), SUMMER: summer_memory()},
                         drinks, {**items, MICROPHONE: microphone()}, memory_abilities=memory_abilities)

    def _scheduled_card_filter_matches(self, scheduled, trigger_context):
        if scheduled.card_id == MICROPHONE and scheduled.source_id == 'item:' + MICROPHONE:
            return trigger_context is not None and trigger_context.used_card_id in self.focus_card_ids
        return super()._scheduled_card_filter_matches(scheduled, trigger_context)

    def _apply_non_draw_effect_with_freshness(self, state, effect, **kwargs):
        updated, event = super()._apply_non_draw_effect_with_freshness(state, effect, **kwargs)
        if (effect.kind is EffectKind.HALF_COST_TURNS and kwargs.get('source_id') == 'item:' + MICROPHONE
                and not state.half_cost_turns):
            return replace(updated, half_cost_turns=updated.half_cost_turns + 1,
                           half_cost_fresh=False), event
        return updated, event
