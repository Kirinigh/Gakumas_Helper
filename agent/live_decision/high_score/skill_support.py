"""One-action support effects; no temporary enhancement is stored in the deck."""
from copy import copy
from dataclasses import replace

from .kernel import EffectKind, EffectSpec, CardCostKind, ConditionKind, CardDefinition, EffectCondition, EffectValueSource
from .contracts import PlanType, ActionKind, GameAction, SchedulePhase, VerifiedHandSupport

SUPPORT_SOURCE = 'gakumas-core@88d73149d926c5a5d1cc3f520e73d0efe4b80a68:shinkokyu:++'
SPOTLIGHT_CARD_ID = '295@3=1,52=1'


def ordinary_spotlight_definition():
    """The verified permanent variant, without its observed current support."""
    return CardDefinition(SPOTLIGHT_CARD_ID, 'スポットライト+', PlanType.SENSE,
        stamina_cost=3, cost_kind=CardCostKind.STAMINA,
        effects=(EffectSpec(EffectKind.GENKI, amount=9),
            EffectSpec(EffectKind.GOOD_CONDITION, amount=7),
            EffectSpec(EffectKind.SCHEDULE, delay_turns=1,
                schedule_phase=SchedulePhase.TURN, schedule_limit=1,
                scheduled_effects=(EffectSpec(EffectKind.DRAW, amount=2),)),
            EffectSpec(EffectKind.SCHEDULE, delay_turns=2,
                schedule_phase=SchedulePhase.TURN, schedule_limit=1,
                scheduled_effects=(EffectSpec(EffectKind.DRAW, amount=1),)),
            EffectSpec(EffectKind.SCORE, amount=6)),
        card_type='mental', rarity='SR', base_card_id='294',
        source_type='produce', canonical_card_id='295')


def supported_definition(ordinary, *, enhancement_count=2):
    """Return a pinned support tier after checking the full ordinary entry.

    This is a catalog projection, not evidence that a selected card has support.
    The caller still needs its exact instance, raw title and complete effects.
    Only the explicitly verified permanent variant has a support projection.
    """
    from .support_schedule import SUPPORTED_CARD_IDS, supported_schedule_card

    if ordinary.card_id == '816':
        from .support_master import cost_only_support
        return cost_only_support(ordinary, enhancement_count)[0]
    if ordinary.card_id in SUPPORTED_CARD_IDS:
        if type(enhancement_count) is not int or enhancement_count != 2:
            raise ValueError('support projection requires the exact ordinary definition and verified support tier')
        return supported_schedule_card(ordinary)
    if ordinary.card_id == SPOTLIGHT_CARD_ID:
        if (type(enhancement_count) is not int or enhancement_count != 2
                or ordinary != ordinary_spotlight_definition()):
            raise ValueError('support projection requires the exact ordinary definition and verified support tier')
        return replace(ordinary, effects=(ordinary.effects[0],
            replace(ordinary.effects[1], amount=8), *ordinary.effects[2:]))
    definitions = {
        '29': CardDefinition('29', '準備運動+', PlanType.SENSE, stamina_cost=4,
            effects=(EffectSpec(EffectKind.SCORE, amount=9),
                     EffectSpec(EffectKind.CONCENTRATION, amount=3)),
            card_type='active', rarity='R', base_card_id='28', source_type='produce'),
        '35': CardDefinition('35', 'ハイタッチ+', PlanType.SENSE, stamina_cost=4,
            effects=(EffectSpec(EffectKind.SCORE, amount=23, score_concentration_multiplier=2),),
            limited=True, card_type='active', rarity='R', base_card_id='34', source_type='produce'),
        '49': CardDefinition('49', '気合十分！+', None, stamina_cost=0,
            effects=(EffectSpec(EffectKind.GENKI, amount=2),
                     EffectSpec(EffectKind.HALF_COST_TURNS, amount=3)),
            card_type='mental', rarity='R', base_card_id='48', source_type='produce'),
        '55': CardDefinition('55', '楽観的+', PlanType.SENSE, stamina_cost=2,
            effects=(EffectSpec(EffectKind.GOOD_CONDITION, amount=4),
                     EffectSpec(EffectKind.CONCENTRATION, amount=1,
                         condition=EffectCondition(ConditionKind.GOOD_CONDITION_MIN, threshold=1))),
            card_type='mental', rarity='R', base_card_id='54', source_type='produce'),
        '57': CardDefinition('57', '深呼吸+', PlanType.SENSE, stamina_cost=3,
            effects=(EffectSpec(EffectKind.CONCENTRATION, amount=3),
                     EffectSpec(EffectKind.GOOD_CONDITION, amount=3,
                         condition=EffectCondition(ConditionKind.CONCENTRATION_MIN, threshold=3))),
            card_type='mental', rarity='R', base_card_id='56', source_type='produce'),
        '87': CardDefinition('87', 'シュプレヒコール+', PlanType.SENSE, stamina_cost=2,
            cost_kind=CardCostKind.CONCENTRATION,
            effects=(EffectSpec(EffectKind.SCORE, amount=6),
                     EffectSpec(EffectKind.GOOD_CONDITION, amount=3),
                     EffectSpec(EffectKind.EXTRA_ACTION, amount=1)),
            card_type='active', rarity='SR', base_card_id='86', source_type='produce'),
        '116': CardDefinition('116', '存在感', PlanType.SENSE, stamina_cost=2,
            cost_kind=CardCostKind.GOOD_CONDITION,
            effects=(EffectSpec(EffectKind.CONCENTRATION, amount=4),
                     EffectSpec(EffectKind.EXTRA_ACTION, amount=1)),
            card_type='mental', rarity='SR', base_card_id='116', upgrade_card_id='117',
            source_type='produce'),
        '117': CardDefinition('117', '存在感+', PlanType.SENSE, stamina_cost=1,
            cost_kind=CardCostKind.GOOD_CONDITION,
            effects=(EffectSpec(EffectKind.CONCENTRATION, amount=5),
                     EffectSpec(EffectKind.EXTRA_ACTION, amount=1)),
            card_type='mental', rarity='SR', base_card_id='116', source_type='produce'),
        '288': CardDefinition('288', 'ひと呼吸', PlanType.SENSE, stamina_cost=7,
            effects=(EffectSpec(EffectKind.GOOD_CONDITION, amount=3),
                     EffectSpec(EffectKind.CONCENTRATION, amount=4)),
            limited=True, card_type='mental', rarity='R', base_card_id='288',
            upgrade_card_id='289', source_type='produce'),
        '707': CardDefinition('707', 'スリリング+', PlanType.SENSE, stamina_cost=2,
            effects=(EffectSpec(EffectKind.GOOD_CONDITION, amount=7),
                     EffectSpec(EffectKind.EXTRA_ACTION, amount=1),
                     EffectSpec(EffectKind.DRAW, amount=1),
                     EffectSpec(EffectKind.ADD_CARD_TO_DECK, card_id='23')),
            limited=True, card_type='mental', rarity='SR', base_card_id='706', source_type='produce'),
        '709': CardDefinition('709', '大胆不敵+', PlanType.SENSE, stamina_cost=3,
            effects=(EffectSpec(EffectKind.GOOD_CONDITION, amount=3),
                     EffectSpec(EffectKind.CONCENTRATION, amount=5),
                     EffectSpec(EffectKind.EXTRA_ACTION, amount=1),
                     EffectSpec(EffectKind.ADD_CARD_TO_DECK, card_id='23')),
            card_type='mental', rarity='SR', base_card_id='708', source_type='produce'),
        '818': CardDefinition('818', 'あなたがくれた夢+', PlanType.SENSE, stamina_cost=4,
            cost_kind=CardCostKind.STAMINA,
            effects=(EffectSpec(EffectKind.STAMINA, value_source=EffectValueSource.MAX_STAMINA, value_scale=.2),
                     EffectSpec(EffectKind.CONCENTRATION_ADDITION_BUFF, amount=2, duration=2),
                     EffectSpec(EffectKind.EXTRA_ACTION, amount=1,
                         condition=EffectCondition(ConditionKind.STAMINA_RATIO_MIN, threshold=.8))),
            card_type='mental', rarity='SSR', base_card_id='817', source_type='pIdol'),
    }
    expected = definitions.get(ordinary.card_id)
    if expected is None:
        from .support_master import cost_only_support
        return cost_only_support(ordinary, enhancement_count)[0]
    tier = 1 if ordinary.card_id in ('116', '288') else 2
    if (type(enhancement_count) is not int or enhancement_count != tier
            or expected is None or ordinary != expected):
        raise ValueError('support projection requires the exact ordinary definition and verified support tier')
    if ordinary.card_id == '288':
        # Fixed CSV 288 -> 289 effects, without upgrading the permanent instance.
        return replace(ordinary, effects=(replace(ordinary.effects[0], amount=4),
                                          replace(ordinary.effects[1], amount=5)))
    if ordinary.card_id == '116':
        # Fixed CSV 116 -> 117: pay good-condition once, retain one action.
        # The permanent ID, discard identity and nonlimited status stay 116.
        return replace(ordinary, stamina_cost=1,
            effects=(replace(ordinary.effects[0], amount=5), ordinary.effects[1]))
    if ordinary.card_id == '117':
        return replace(ordinary, effects=(replace(ordinary.effects[0], amount=6), ordinary.effects[1]))
    if ordinary.card_id == '55':
        return replace(ordinary, stamina_cost=1)
    if ordinary.card_id == '709':
        # Native 1117/1118 complete ++ text retains every effect and costs 2.
        return replace(ordinary, stamina_cost=2)
    if ordinary.card_id == '707':
        # Native 1984 complete ++ body retains all four effects and costs 1.
        return replace(ordinary, stamina_cost=1)
    if ordinary.card_id == '818':
        # Native 1229 complete ++ body keeps recovery/buff/condition unchanged.
        return replace(ordinary, stamina_cost=3)
    if ordinary.card_id in ('29', '35'):
        return replace(ordinary, stamina_cost=3,
            effects=(replace(ordinary.effects[0], amount=ordinary.effects[0].amount+1), *ordinary.effects[1:]))
    if ordinary.card_id == '87':
        return replace(ordinary, effects=(replace(ordinary.effects[0], amount=12), *ordinary.effects[1:]))
    return replace(ordinary, effects=(replace(ordinary.effects[0], amount=4), ordinary.effects[1]))


def supported_deep_breath_definition(ordinary):
    """Compatibility name for the already dispatched historical 57-only proof."""
    if ordinary.card_id != '57':
        raise ValueError('support historical projection requires deep breath')
    return supported_definition(ordinary)


def project_supported_card_once(bundle, action, *, effective_definition, settle_automatic_turn=True):
    """Apply an independently proved definition to one resolved PLAY only.

    The caller proves the actual support tier. Other plays, pending choices,
    arbitrary persistent support effects and enhancements carried in CardRef
    are outside this projection. Verified 597 schedules keep a derived effect
    reference; automatic next-turn effects otherwise use the original kernel.
    """
    state, kernel = bundle.state, bundle.kernel
    if (action.kind is not ActionKind.PLAY_CARD or type(action.source_slot) is not int
            or not 0 <= action.source_slot < len(state.deck.hand)):
        raise ValueError('support projection requires one bound card slot')
    ref = state.deck.hand[action.source_slot]
    ordinary = kernel.cards[ref.card_id]
    memory_support = ref.card_id == '816' and effective_definition == supported_definition(ordinary)
    if (ref.instance_id != action.card_instance_id
            or effective_definition.card_id != ref.card_id
            or effective_definition.persistent_effects != ordinary.persistent_effects
            or not memory_support and (effective_definition.persistent_effects
                or any(item.card_id == ref.card_id for item in state.scheduled_effects))):
        raise ValueError('support projection differs from the target instance or is persistent')
    effective_definition.validate()
    if (ref.card_id == SPOTLIGHT_CARD_ID
            and effective_definition != supported_definition(ordinary)):
        raise ValueError('spotlight support differs from the verified full definition')
    if memory_support:
        # Keep the permanent card and existing replay schedules on the original kernel.
        fact = VerifiedHandSupport(ref.instance_id, ref.card_id, state.turn, 2)
        existing = tuple(item for item in state.hand_support if item.instance_id == ref.instance_id)
        if existing and existing != (fact,):
            raise ValueError('816 support conflicts with the already bound hand instance')
        held_state = state if existing else replace(state, hand_support=(*state.hand_support, fact))
        branches = kernel.transition(held_state, action, plan_type=PlanType.SENSE)
    else:
        local = copy(kernel)
        local.cards = {**kernel.cards, ref.card_id: effective_definition}
        branches = local.transition(state, action, plan_type=PlanType.SENSE)
    if not branches or abs(sum(branch.probability for branch in branches) - 1) > 1e-9:
        raise ValueError('support projection has incomplete outcome probabilities')
    from .support_schedule import (
        CARD_ID,
        FOUR_TURN_CARD_ID,
        SUPPORTED_CARD_IDS,
        ordinary_schedule_index,
        supported_schedule_card,
        supported_schedule_index,
        schedule_definition_effects,
    )

    if ref.card_id in SUPPORTED_CARD_IDS:
        if effective_definition != supported_schedule_card(ordinary):
            raise ValueError('support scheduled effects differ from the verified full definition')
        rewritten = []
        for branch in branches:
            created = [item for item in branch.state.scheduled_effects
                if item.source_id == ref.instance_id and item not in state.scheduled_effects]
            if ref.card_id in (CARD_ID, FOUR_TURN_CARD_ID):
                if (len(created) != 1 or created[0].definition_effect_index != 1
                        or created[0].remaining_uses != 4 or created[0].persistent):
                    raise ValueError('support play did not create exactly one four-use schedule')
            elif (len(created) != 1 + int(state.double_card_effects_remaining > 0)
                    or any(item.definition_effect_index != ordinary_schedule_index(ref.card_id)
                        or item.persistent for item in created)):
                raise ValueError('support play did not retain its counted schedule instances')
            for item in created:
                schedule_definition_effects(kernel.cards, replace(item,
                    definition_effect_index=supported_schedule_index(ref.card_id)))
            rewritten.append(replace(branch, state=replace(branch.state, scheduled_effects=tuple(
                replace(item, definition_effect_index=supported_schedule_index(ref.card_id)) if item in created else item
                for item in branch.state.scheduled_effects))))
        branches = tuple(rewritten)
    elif not memory_support and any(effect.kind is EffectKind.SCHEDULE
            and (ref.card_id != SPOTLIGHT_CARD_ID or effect != ordinary.effects[index])
            for index, effect in enumerate(effective_definition.effects)):
        raise ValueError('support scheduled effect is outside the verified slice')
    outcomes = []
    for branch in branches:
        played = [event.source_id for event in branch.events
                  if event.event_type == 'CONSUME_ACTION']
        if (played != [ref.instance_id]
                or any(event.event_type.startswith(('FREE_PLAY', 'PAID_PLAY')) for event in branch.events)
                or branch.state.pending_choice or branch.state.resolution_stack
                or branch.state.turn != state.turn):
            raise ValueError('support projection did not resolve exactly one direct play')
        settled = branch.state
        if settle_automatic_turn and not settled.terminal and settled.actions_remaining == 0:
            next_branches = kernel.transition(settled, GameAction(ActionKind.END_TURN), plan_type=PlanType.SENSE)
            if not next_branches or abs(sum(item.probability for item in next_branches) - 1) > 1e-9:
                raise ValueError('support projection automatic turn is incomplete')
            outcomes.extend(item.state for item in next_branches)
        else:
            outcomes.append(settled)
    if any(item.pending_choice or item.resolution_stack for item in outcomes):
        raise ValueError('support projection has unresolved continuation')
    return branches, tuple(dict.fromkeys(outcomes))
