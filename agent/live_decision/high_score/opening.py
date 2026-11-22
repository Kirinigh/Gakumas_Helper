"""Assemble a quiescent HIF opening from complete, same-batch observations.

This boundary is deliberately opening-only. Later turns need the committed
effect history and cannot reset schedules through this constructor.
"""

import re
import csv
import json
import math
from pathlib import Path
from collections import Counter
from dataclasses import replace, dataclass

from route.engine import has_single_frame_id

from .hif import HifSenseKernel, compile_observed_hif_memories
from .kernel import EffectKind, EffectSpec, ConditionKind, DrinkDefinition, EffectCondition
from ..catalog import SkillCardFeatureCatalog
from ..upgrade import normalize_upgrade_title
from .contracts import CardRef, PlanType, ActionKind, DeckBelief, SchedulePhase, StrategyState, EffectSourceKind
from ..inventory import match_inventory_hand
from .catalog_v2 import compile_item_catalog, compile_drink_catalog, compile_skill_card_catalog
from ..hif_visual import HifLiveRunContext
from ..hif_session import resolve_hif_round_context
from ..status_panel import (
    StatusPanelRow,
    StatusPanelPage,
    status_panel_consensus,
    join_status_panel_pages,
    status_row_confidence_valid,
)
from .customizations import compile_observed_customizations
from .support_schedule import is_support_schedule, schedule_status_name


@dataclass(frozen=True)
class HifOpeningState:
    # Both native simulation profiles model the complete future deck, schedules
    # and resource choices. These are inputs to this model, not UI prerequisites.
    required_observations = ('cards', 'status', 'items', 'drinks', 'turns')

    run_id: str
    revision: int
    state: StrategyState
    kernel: HifSenseKernel
    hand_instance_ids: tuple[str, ...]
    inventory_keys: tuple[tuple[str, str, tuple], ...] = ()
    drink_slots: tuple[tuple[int, str], ...] | None = None
    expired_start_scores: tuple = ()
    opening_skill_support: tuple = ()
    opening_support_origin: tuple[str, int] | None = None
    expired_start_drinks: tuple = ()
    opening_live_context: HifLiveRunContext | None = None

    @staticmethod
    def observation_limitation(name, observation):
        if name == 'drinks' and observation.get('full_drink_state_ready') is False:
            if observation.get('unsupported_slots'):
                return 'HIF opening model cannot use recognized event drinks in slots: '+str(observation['unsupported_slots'])
            if observation.get('unknown_slots'):
                return 'HIF opening model cannot represent unknown occupied drink slots: '+str(observation['unknown_slots'])
        return None


def verified_expired_start_scores(bundle, expected):
    """Keep a last-fired UI row for its firing turn, never a kernel trigger.

    Native 945 proves that the final start-of-turn score leaves a visible
    one-turn row until that turn ends. Preserve its original one-use ref as
    observation provenance; the strategy state remains fully exhausted.
    """
    present = {(ref.source_id, ref.definition_effect_index) for ref in expected.scheduled_effects}
    candidates = list(bundle.expired_start_scores)
    if (expected.turn == bundle.state.turn + 1
            and expected.turns_remaining == bundle.state.turns_remaining - 1):
        candidates.extend((expected.turn, ref) for ref in bundle.state.scheduled_effects)
    result = []
    for turn, ref in candidates:
        if (turn != expected.turn or ref.source_kind is not EffectSourceKind.CARD
                or ref.remaining_uses != 1 or ref.last_fired_turn != turn - 1
                or ref.turns_until_fire != 1 or ref.remaining_turns is not None
                or ref.trigger_phase is not SchedulePhase.START_OF_TURN
                or ref.direct_on_trigger or ref.persistent
                or (ref.source_id, ref.definition_effect_index) in present
                or not any(card.instance_id == ref.source_id and card.card_id == ref.card_id
                           for card in expected.deck.all_cards)):
            continue
        definition = bundle.kernel.cards[ref.card_id]
        effect = bundle.kernel._scheduled_effect_spec(ref)
        if ((definition.canonical_card_id or definition.card_id) != '597'
                or len(effect.scheduled_effects) != 1
                or effect != EffectSpec(EffectKind.SCHEDULE, delay_turns=1,
                    schedule_phase=SchedulePhase.START_OF_TURN, schedule_limit=4,
                    scheduled_effects=(EffectSpec(EffectKind.SCORE,
                        amount=effect.scheduled_effects[0].amount),))):
            continue
        # A double play registers two equal refs; both leave a visible row.
        # Same-turn rebinding uses the retained proof only. Advancing a turn
        # excludes the old proof above, so this does not derive it twice.
        result.append((turn, ref))
    return tuple(result)


def _finite_drink_draw_definition():
    return DrinkDefinition('14', 'センブリソーダ', effects=(
        EffectSpec(EffectKind.SCORE_BUFF, amount=.1, duration=5),
        EffectSpec(EffectKind.DRAW, amount=2), EffectSpec(EffectKind.STAMINA, amount=-2),
        EffectSpec(EffectKind.SCHEDULE, delay_turns=1,
            schedule_phase=SchedulePhase.AFTER_START_OF_TURN, schedule_limit=5,
            scheduled_effects=(EffectSpec(EffectKind.DRAW, amount=1),))))


def verified_expired_start_drinks(bundle, expected):
    """1692's final soda draw keeps one UI turn, never another kernel draw.

    As for the already verified final start-score row, retain only the exact
    exhausted predecessor for its firing turn. The complete observed status
    must still match this row; no schedule is added to the strategy state.
    """
    candidates = list(bundle.expired_start_drinks)
    if (expected.turn == bundle.state.turn + 1
            and expected.turns_remaining == bundle.state.turns_remaining - 1):
        candidates.extend((expected.turn, ref) for ref in bundle.state.scheduled_effects)
    present = {(ref.source_kind, ref.source_id, ref.definition_effect_index)
               for ref in expected.scheduled_effects}
    return tuple((turn, ref) for turn, ref in candidates
        if turn == expected.turn and ref.source_kind is EffectSourceKind.DRINK
        and ref.source_id == 'drink:14' and ref.card_id == '14' and ref.definition_effect_index == 3
        and ref.trigger_phase is SchedulePhase.AFTER_START_OF_TURN and ref.turns_until_fire == 1
        and type(ref.remaining_uses) is int and ref.remaining_uses == 1
        and ref.last_fired_turn == turn - 1 and ref.remaining_turns is None
        and not ref.persistent and not ref.direct_on_trigger and not ref.duration_fresh and ref.effect_counter == 0
        and (ref.source_kind, ref.source_id, ref.definition_effect_index) not in present
        and bundle.kernel.drinks.get(ref.card_id) == _finite_drink_draw_definition())


def observed_hif_drink_slots(component):
    """Keep physical holes; an identity-only legacy log cannot bind an input."""
    if (component.get('accepted') is not True or component.get('full_drink_state_ready') is not True
            or component.get('returned_unchanged') is not True):
        raise ValueError('HIF physical drink observation is incomplete')
    rows = component.get('slots')
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 4:
        raise ValueError('HIF physical drink slots are missing')
    occupied = []
    for expected_slot, row in enumerate(rows):
        if type(row.get('slot')) is not int or row['slot'] != expected_slot:
            raise ValueError('HIF physical drink slots are duplicated or unordered')
        if row.get('state') == 'EMPTY':
            if row.get('drink_id') is not None or row.get('available') is not False:
                raise ValueError('HIF empty drink slot is contradictory')
        elif row.get('state') == 'OCCUPIED':
            if not isinstance(row.get('drink_id'), str) or not row['drink_id'] or row.get('available') is not True:
                raise ValueError('HIF occupied drink slot is unresolved')
            occupied.append((row['slot'], row['drink_id']))
        else:
            raise ValueError('HIF drink slot occupancy is unknown')
    if tuple(drink for _, drink in occupied) != tuple(component.get('drinks', ())):
        raise ValueError('HIF physical drink slots disagree with inventory')
    return tuple(occupied)


def verify_hif_drink_consumption(before, after, action):
    """Verify exactly one bottle was consumed, preserving remaining bottle order.

    Complete four-slot observations may show the remaining bottles compacted
    to the left. Older partial-slot records retain their fixed-position rule.
    """
    old_slots = observed_hif_drink_slots(before)
    new_slots = observed_hif_drink_slots(after)
    if (before.get('run_id') != after.get('run_id') or not before.get('run_id')
            or before.get('revision') != after.get('revision')
            or len(before['slots']) != len(after['slots'])):
        raise ValueError('HIF drink consumption observation epoch or capacity changed')
    return verify_hif_consumed_slots(old_slots,new_slots,action,allow_compaction=len(before['slots']) == 4)


def verify_hif_consumed_slots(old_slots, new_slots, action, *, allow_compaction):
    """Check bottle order after the caller proves the physical observations."""
    if (action.kind is not ActionKind.USE_DRINK or type(action.source_slot) is not int
            or (action.source_slot, action.drink_id) not in old_slots):
        raise ValueError('HIF drink consumption has no verified physical target')
    expected = tuple(row for row in old_slots if row[0] != action.source_slot)
    compacted = tuple(enumerate(drink for _, drink in expected))
    if new_slots != expected and not (allow_compaction and new_slots == compacted):
        raise ValueError('HIF drink consumption did not match the selected bottle')
    return new_slots


def assemble_hif_opening(run, *, core, components, hand_titles, data_root, round_context=None,
                         opening_skill_support=(), confirmed_deck_baseline=None,
                         extended_hand_identity=None):
    """Consume native component results; no unknown value becomes a default.

    The host owns one uninterrupted observation batch and supplies the final
    observed core/title checkpoint. Every component must belong to that
    batch and have completed its return check. Upgrade suffixes follow D-047's
    single-plus approximation; complete card names must still match.
    """
    from ..inventory_customization import verified_zero_customization_detail

    round_number, base_turn_count = resolve_hif_round_context(
        round_context, run_id=run.run_id, revision=run.revision)
    for name in HifOpeningState.required_observations:
        if name not in components:
            raise ValueError('HIF opening model lacks the observed '+name+' component')
        event = components[name]
        limitation = HifOpeningState.observation_limitation(name, event)
        if limitation:
            raise ValueError(limitation)
        if (event.get('accepted') is not True or event.get('run_id') != run.run_id
                or event.get('revision') != run.revision or event.get('returned_unchanged') is not True):
            raise ValueError('HIF opening component is stale, incomplete or from another run')
    from ..opening_observer import valid_opening_frame_ids

    if not valid_opening_frame_ids(core.get('frame_ids')) or len(core.get('fields', ())) != 6:
        raise ValueError('HIF opening core certificate is incomplete')
    plan_type = PlanType(run.plan.upper())
    remaining, stamina, genki, score, multiplier, _ = core['fields']
    turns = components['turns']
    stamina_valid = (stamina == run.max_stamina if round_number == 1 else
                     type(stamina) is int and 0 <= stamina <= run.max_stamina)
    if (round_number == 2 and (turns.get('round_number') != 2
                               or turns.get('base_turn_count') != 12)):
        raise ValueError('HIF opening turn table lacks its explicit second-round context')
    if (turns['current_turn'] != 1 or remaining != base_turn_count or score != 0
            or type(genki) is not int or genki < 0
            or (genki != 0 and not (genki == 2 and turns['turn_attributes'][0] == 'vocal'
                and any(row['item_id'] == '75' and row['remaining_uses'] == 2
                    for row in components['items']['identity_count_observation']['items'])))
            or not stamina_valid or len(turns['turn_attributes']) != remaining
            or len(turns['score_multipliers']) != remaining
            or turns['score_multipliers'][0] != multiplier):
        raise ValueError('HIF opening resource/turn invariant is not observed')
    card_rows = components['cards']['cards']
    confirmed = any(row.get('source') == 'confirmed_deck_baseline' for row in card_rows)
    zones = {'手札':[], '山札':[], '捨札':[], '除外':[]}
    for row in card_rows:
        if row['zone'] not in zones or row['index'] != len(zones[row['zone']]):
            raise ValueError('HIF card physical slots are incomplete or out of order')
        if row['source'] == 'confirmed_deck_baseline':
            if confirmed_deck_baseline is None:
                raise ValueError('HIF confirmed deck rows lack their explicit baseline')
        elif row['source'] not in ('observed_detail', 'observed_badge_absence'):
            raise ValueError('HIF permanent effects lack an observation source')
        elif ((row['source'] == 'observed_detail') != bool(row['customizations'])
              and not (row['source'] == 'observed_detail' and verified_zero_customization_detail(row))):
            raise ValueError('HIF permanent effect presence is contradictory')
        zones[row['zone']].append(row)
    if len(zones['手札']) not in (3, 4, 5) or zones['捨札'] or zones['除外'] or not zones['山札']:
        raise ValueError('HIF opening card-zone invariant is not observed')
    if components['cards'].get('zone_counts') != {zone:len(rows) for zone,rows in zones.items()}:
        raise ValueError('HIF opening card counts do not cover the observed inventory')
    if confirmed:
        from ..inventory_customization import confirmed_deck_card_rows
        rebuilt = confirmed_deck_card_rows(confirmed_deck_baseline,
            {zone: [row['card_id'] for row in rows] for zone, rows in zones.items()},
            presence=components['cards'].get('confirmed_deck_presence', ()))
        if card_rows != rebuilt:
            raise ValueError('HIF confirmed deck rows differ from the unique baseline mapping')
    root = Path(data_root)
    with (root/'skill_cards.csv').open(encoding='utf-8-sig', newline='') as stream:
        metadata = {row['id']:row for row in csv.DictReader(stream)}
    customization_rows = json.loads((root/'customizations.json').read_text(encoding='utf-8'))
    free = tuple(sorted({row['card_id'] for row in card_rows
                         if metadata[row['card_id']]['plan']=='free'}, key=int))
    catalog = compile_skill_card_catalog(root/'skill_cards.csv', plan_type=plan_type,
                                         include_free_card_ids=free)
    definitions = dict(catalog.cards)
    refs, refs_by_zone = [], {zone:[] for zone in zones}
    inventory_keys = []
    for zone, rows in zones.items():
        for row in rows:
            canonical = row['card_id']
            if canonical not in definitions:
                raise ValueError('HIF deck contains an unsupported card: '+canonical)
            definition = definitions[canonical]
            if row['customizations']:
                definition = compile_observed_customizations(definition, metadata[canonical],
                    customization_rows, row['customizations'])
                definitions[definition.card_id] = definition
            ref = CardRef(f'{run.run_id}:{zone}:{row["index"]}', definition.card_id,
                          upgraded=metadata[canonical]['upgraded'].upper()=='TRUE')
            refs.append(ref)
            inventory_keys.append((ref.instance_id, canonical,
                tuple(sorted((int(k), v) for k,v in row['customizations'].items()))))
            refs_by_zone[zone].append(ref)
    if len(zones['手札']) == 3:
        hand_ids = match_inventory_hand(core['families'], tuple(row['card_id'] for row in zones['手札']))
    else:
        from ..extended_opening import verify_extended_opening_identity
        hand_ids = verify_extended_opening_identity(extended_hand_identity, run=run,
            core=core, inventory_rows=card_rows,
            catalog=SkillCardFeatureCatalog.load(root/'skill_cards.csv'))
    forced = {row['card_id'] for row in card_rows
              if metadata[row['card_id']]['forceInitialHand'].upper()=='TRUE'}
    if not forced or not forced <= set(hand_ids):
        raise ValueError('HIF opening forced hand is not present')
    # Equal copies can exchange physical labels only while all effect semantics
    # agree. A customized/base pair needs separate direct hand evidence.
    hand = []
    for cid in hand_ids:
        candidates = [ref for ref in refs_by_zone['手札'] if ref not in hand
            and (definitions[ref.card_id].canonical_card_id or ref.card_id)==cid]
        if not candidates or len({ref.semantic_key for ref in candidates}) != 1:
            raise ValueError('HIF duplicate hand effects are not uniquely mapped')
        hand.append(candidates[0])
    names = [metadata[cid]['name'] for cid in hand_ids]
    verified_support, support_slots = [], set()
    for receipt in opening_skill_support:
        from ..opening_support import verify_opening_skill_support_receipt
        slot = receipt.get('slot') if isinstance(receipt, dict) else None
        if (type(slot) is not int or not 0 <= slot < len(hand) or slot in support_slots
                or hand[slot].card_id != '288' or hand[slot].upgraded):
            raise ValueError('HIF opening support target is not a unique base instance')
        observed = verify_opening_skill_support_receipt(receipt, run_id=run.run_id,
            revision=run.revision, card_instance_id=hand[slot].instance_id,
            card_id=hand[slot].card_id, slot=slot)
        if tuple(observed['frame_ids']) != tuple(core['frame_ids']):
            raise ValueError('HIF opening support differs from the final core checkpoint')
        names[slot] = metadata[observed['visible_card_id']]['name']
        support_slots.add(slot)
        verified_support.append(observed)
    if len(hand_titles) != len(core['frame_ids']) or any(tuple(normalize_upgrade_title(title) for title in titles) != tuple(names) for titles in hand_titles):
        raise ValueError('HIF opening hand identity is unresolved')
    groups = tuple(tuple(StatusPanelPage(
        rows=tuple(StatusPanelRow(**dict(row,box=tuple(row['box']))) for row in page['rows']),
        thumb=tuple(page['thumb']),at_top=page['at_top'],at_bottom=page['at_bottom'],
        partial_row_count=page['partial_row_count']) for page in group['observations'])
        for group in components['status']['pages'])
    status_rows = join_status_panel_pages(tuple(status_panel_consensus(group) for group in groups))
    scalar = {}
    allowed_status = {'集中': ('concentration', ''), '好調': ('good_condition_turns', 'ターン'),
                      '好印象': ('good_impression_turns', 'ターン'), 'やる気': ('motivation', ''),
                      '全力値': ('full_power', ''), '熱意': ('passion', '')}
    for row in status_rows:
        if row.title == 'メモリーアビリティ':
            continue
        if row.title not in allowed_status or row.title in scalar:
            raise ValueError('HIF opening has a non-initial or unsupported status')
        name, unit = allowed_status[row.title]
        if row.unit != unit or type(row.count) is not int or row.count < 1:
            raise ValueError('HIF opening scalar status is unresolved')
        scalar[row.title] = (name,row.count)
    memories = compile_observed_hif_memories(groups,groups,
        SkillCardFeatureCatalog.load(root/'skill_cards.csv'),
        known_run_id=run.run_id,current_run_id=run.run_id,plan_type=plan_type)
    item_rows = components['items']['identity_count_observation']['items']
    item_ids = tuple(row['item_id'] for row in item_rows)
    if len(set(item_ids)) != len(item_ids):
        raise ValueError('HIF item source inventory is ambiguous')
    item_catalog = compile_item_catalog(root/'p_items.csv',plan_type=plan_type)
    item_definitions = dict(item_catalog.items)
    stage_start_ids = set()
    if any(row['item_id'] == '428' or row.get('effect_scope') is not None for row in item_rows):
        from .opening_items import bind_observed_stage_start_item
        with (root/'p_items.csv').open(encoding='utf-8-sig', newline='') as stream:
            item_metadata = {row['id']: row for row in csv.DictReader(stream)}
        for row in item_rows:
            if row['item_id'] == '428' or row.get('effect_scope') is not None:
                definition = bind_observed_stage_start_item(row, item_metadata.get(row['item_id']))
                item_definitions[definition.item_id] = definition
                stage_start_ids.add(definition.item_id)
    drink_catalog = compile_drink_catalog(root/'p_drinks.csv',plan_type=plan_type)
    drinks = components['drinks']
    if drinks.get('full_drink_state_ready') is not True:
        raise ValueError('HIF drink inventory is incomplete')
    kernel = HifSenseKernel(definitions, drink_catalog.drinks, item_definitions, plan_type=plan_type,
                           memory_abilities={row.definition.ability_id:row.definition for row in memories})
    state = StrategyState(turn=1, turns_remaining=remaining, score=score, stamina=stamina,
        max_stamina=run.max_stamina, genki=genki, actions_remaining=1, hand_limit=5,
        score_multiplier=multiplier, deck=DeckBelief(tuple(refs),tuple(hand),tuple(refs_by_zone['山札'])),
        items=item_ids, drinks=tuple(drinks['drinks']),
        memory_abilities=tuple(row.definition.ability_id for row in memories),
        future_score_multipliers=tuple(turns['score_multipliers'][1:]),
        turn_attributes=tuple(turns['turn_attributes']),
        **{name:count for name,count in scalar.values()})
    usage = {(EffectSourceKind.ITEM, 'item:'+row['item_id'], row['definition_effect_index']):
             row['remaining_uses'] for row in item_rows if row['item_id'] not in stage_start_ids}
    usage.update({(EffectSourceKind.MEMORY,'memory:'+row.definition.ability_id,index):row.remaining_uses
                  for row in memories for index in range(len(row.definition.persistent_effects))})
    from .opening_items import initialize_observed_opening
    state = initialize_observed_opening(kernel,state,usage)
    return HifOpeningState(run.run_id,run.revision,state,kernel,
        tuple(ref.instance_id for ref in hand),tuple(inventory_keys),
        observed_hif_drink_slots(drinks) if 'slots' in drinks else None,
        opening_live_context=run if extended_hand_identity is not None else None,
        opening_skill_support=tuple(verified_support),
        opening_support_origin=(run.run_id, run.revision) if verified_support else None)


def resolve_hif_committed_observation(bundle, candidates, *, core, components, baseline_components,
                                     committed_action=None, preinput_continuity=None,
                                     single_plus_estimate=None):
    """Resolve the maintained ledger against freshly observed changed state."""
    if preinput_continuity is not None and committed_action is not None:
        raise ValueError('HIF preinput retention cannot verify a submitted action')
    states=candidates if isinstance(candidates,tuple) else (candidates,)
    if not states:
        raise ValueError('HIF settlement has no candidate outcomes')
    drink_slots = None
    if committed_action is not None and committed_action.kind is ActionKind.USE_DRINK:
        drink_slots = verify_hif_drink_consumption(baseline_components['drinks'], components['drinks'],
                                                  committed_action)
    elif 'slots' in components['drinks']:
        drink_slots = observed_hif_drink_slots(components['drinks'])
    observed_durations = _observed_hif_duration_updates(bundle, components)
    observed_scalars = _observed_hif_single_plus_updates(
        bundle, components, committed_action, single_plus_estimate)
    reconciliations = []
    scalar_reconciliations = []
    reconciled = []
    for state in states:
        changes = {}
        played = [ref for ref in bundle.state.deck.hand if ref.instance_id not in
                  {r.instance_id for r in state.deck.hand}]
        kinds = {effect.kind.value for ref in played for effect in bundle.kernel.cards[ref.card_id].effects}
        for name, (value, kind) in observed_durations.items():
            if kind in kinds and getattr(state, name) > getattr(bundle.state, name):
                if value != getattr(state, name):
                    changes[name] = dict(estimated=getattr(state, name), observed=value)
        reconciliations.append(dict(changes))
        scalar_changes = {
            name: dict(estimated=getattr(state, name), observed=value)
            for name, value in observed_scalars.items() if value != getattr(state, name)
        }
        scalar_reconciliations.append(scalar_changes)
        changes.update(scalar_changes)
        reconciled.append(replace(state, **{name: values['observed'] for name, values in changes.items()}))
    states = tuple(reconciled)
    results=tuple(verify_hif_committed_observation(bundle,state,core=core,
        components=components,baseline_components=baseline_components,
        preinput_continuity=preinput_continuity, committed_action=committed_action) for state in states)
    matches=[index for index,result in enumerate(results) if result['accepted']]
    equivalent = False
    if len(matches) > 1 and committed_action is not None:
        from .maintenance import equivalent_settlements
        equivalent = equivalent_settlements(states[index] for index in matches)
    if len(matches)!=1 and not equivalent:
        result = dict(accepted=False,reason='hif_settlement_outcome_not_unique',
            candidate_count=len(states),matched_outcome_count=len(matches),
            candidate_checks=[result['checks'] for result in results])
        if single_plus_estimate is not None:
            result['candidate_single_plus_reconciliations'] = scalar_reconciliations
        return None, result
    index=matches[0]
    state=states[index]
    keys={(ref.source_id,ref.definition_effect_index) for ref in state.scheduled_effects}
    exhausted=tuple(ref for ref in bundle.state.scheduled_effects
        if ref.source_kind is EffectSourceKind.ITEM and ref.remaining_uses==0
        and (ref.source_id,ref.definition_effect_index) not in keys)
    if exhausted:
        state=replace(state,scheduled_effects=(*state.scheduled_effects,*exhausted))
    result = dict(results[index],candidate_count=len(states),matched_outcome_count=len(matches),
                  interchangeable_instance_outcomes=equivalent,
                  observed_duration_reconciliation=reconciliations[index], drink_slots=drink_slots)
    if single_plus_estimate is not None:
        result['observed_single_plus_reconciliation'] = scalar_reconciliations[index]
    return state, result


def resolve_hif_pending_observation(bundle, candidates, *, core, components,
                                    baseline_components, committed_action,
                                    single_plus_estimate=None):
    """Compare a complete actual observation with both sides of a dispatch.

    A dispatch proves an attempt, not consumption. An indistinguishable outcome
    or an incomplete read never permits retrying the action.
    """
    # Rule-maintained predecessors have no old panel batch. The existing full
    # comparator can check a complete current panel against an independently
    # maintained state using its self-baseline path; do not synthesize old rows.
    current_baseline = not baseline_components.get('status')
    if current_baseline:
        if committed_action.kind is not ActionKind.PLAY_CARD:
            raise ValueError('pending non-card comparison requires its original component baseline')
        baseline_components = components
    outcomes = {}
    for side, states, action, estimate in (
            ('before', bundle.state, None, None),
            ('after', candidates, committed_action, single_plus_estimate)):
        try:
            outcomes[side] = resolve_hif_committed_observation(bundle, states,
                core=core, components=components, baseline_components=baseline_components,
                committed_action=action, single_plus_estimate=estimate)
        except ValueError as error:
            outcomes[side] = None, dict(accepted=False, reason=str(error))
    matched = [side for side, (_, result) in outcomes.items() if result['accepted']]
    comparisons = {side: result for side, (_, result) in outcomes.items()}
    if len(matched) != 1:
        return None, dict(accepted=False, pending_outcome='unknown',
            reason='hif_pending_state_not_distinguishable', pending_comparisons=comparisons,
            pending_current_component_baseline=current_baseline)
    side, = matched
    state, result = outcomes[side]
    return state, dict(result, pending_outcome=side, pending_comparisons=comparisons,
        pending_current_component_baseline=current_baseline)


def _observed_hif_single_plus_updates(bundle, components, action, receipt):
    """Read declared scalar totals; this never infers an unknown support tier.

    The caller supplies actual submission evidence. The receipt verifier owns
    same-state title/SELECT and permanent-instance proof. Only the submitted
    card's unconditional direct scalar kinds are eligible here; disappearing
    hand cards, schedules, action counts and unknown effects grant no scope.
    """
    if receipt is None:
        return {}
    from ..hif_single_plus import verify_selected_single_plus_estimate

    verified = verify_selected_single_plus_estimate(receipt, bundle=bundle, action=action)
    if (not isinstance(verified, dict) or verified.get('accepted') is not True
            or type(receipt.get('submit_count')) is not int or receipt['submit_count'] != 1
            or action is None or action.kind is not ActionKind.PLAY_CARD):
        raise ValueError('HIF single-plus settlement lacks one verified submission')
    if verified.get('ordinary_plus_projection') is True:
        # A known single-plus catalog projection grants no unknown-effect delta.
        return {}
    ref = bundle.state.deck.hand[action.source_slot]
    definition = bundle.kernel.cards[ref.card_id]
    supported = {
        EffectKind.CONCENTRATION: ('concentration', '集中', '', 'スコア上昇量を{}増加'),
        EffectKind.GOOD_CONDITION: ('good_condition_turns', '好調', 'ターン', 'スコア上昇量を50%増加{}ターン'),
    }
    declared = {
        effect.kind for effect in definition.effects if effect.kind in supported
        and effect.condition is None and effect.value_source is None
        and type(effect.amount) in (int, float) and math.isfinite(effect.amount)
        and effect.amount > 0 and float(effect.amount).is_integer()
    }
    if not declared:
        return {}
    event = components.get('status', {})
    if (event.get('accepted') is not True or event.get('returned_unchanged') is not True
            or event.get('run_id') != bundle.run_id or event.get('revision') != bundle.revision
            or event.get('observation_source') in {
                'retained_unchanged_component', 'retained_preinput_no_business_change'}):
        raise ValueError('HIF single-plus scalar observation is not a fresh complete batch')
    groups = []
    frame_ids = []
    for group in event.get('pages', ()):
        ids = group.get('frame_ids', ())
        if not has_single_frame_id(ids) or len(group['observations']) != 1:
            raise ValueError('HIF single-plus status page lacks its current observation')
        frame_ids.extend(ids)
        pages = tuple(StatusPanelPage(
            rows=tuple(StatusPanelRow(**dict(row, box=tuple(row['box']))) for row in page['rows']),
            thumb=tuple(page['thumb']), at_top=page['at_top'], at_bottom=page['at_bottom'],
            partial_row_count=page['partial_row_count']) for page in group['observations'])
        if any(not status_row_confidence_valid(row) for page in pages for row in page.rows):
            raise ValueError('HIF single-plus status observation has insufficient confidence')
        groups.append(status_panel_consensus(pages))
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError('HIF single-plus status pages reuse observation frames')
    rows = join_status_panel_pages(groups)
    result = {}
    for kind in declared:
        name, title, unit, description = supported[kind]
        matches = [row for row in rows if row.title == title]
        # Missing rows never supply an invented zero. Header and complete
        # description must independently contain the same positive integer.
        if len(matches) != 1:
            raise ValueError('HIF single-plus scalar row is missing or ambiguous: ' + title)
        row = matches[0]
        if (type(row.count) is not int or row.count <= 0 or row.unit != unit
                or row.remaining_turns is not None or row.per_turn_limit is not None
                or row.description != description.format(row.count)):
            raise ValueError('HIF single-plus scalar body contradicts its count: ' + title)
        result[name] = row.count
    return result


def _observed_hif_duration_updates(bundle, components):
    """D-047 estimates card effects; complete status observations own duration."""
    event = components.get('status', {})
    if (event.get('accepted') is not True or event.get('returned_unchanged') is not True
            or event.get('run_id') != bundle.run_id or event.get('revision') != bundle.revision):
        return {}
    groups = tuple(tuple(StatusPanelPage(
        rows=tuple(StatusPanelRow(**dict(row, box=tuple(row['box']))) for row in page['rows']),
        thumb=tuple(page['thumb']), at_top=page['at_top'], at_bottom=page['at_bottom'],
        partial_row_count=page['partial_row_count']) for page in group['observations'])
        for group in event['pages'])
    rows = join_status_panel_pages(tuple(status_panel_consensus(group) for group in groups))
    definitions = {
        '絶好調': ('excellent_condition_turns', 'EXCELLENT_CONDITION',
                  '好調のスコア上昇量を好調1ターンごとに10%増加'),
        '消費体力減少': ('half_cost_turns', 'HALF_COST_TURNS', '消費体力を50%軽減'),
    }
    result = {}
    for title, (name, kind, description) in definitions.items():
        matches = [row for row in rows if row.title == title]
        if len(matches) == 1:
            row = matches[0]
            if (row.unit == 'ターン' and row.remaining_turns is None and row.per_turn_limit is None
                    and row.description == description + str(row.count) + 'ターン'):
                result[name] = (row.count, kind)
    return result


def expected_permanent_inventory_keys(bundle, expected):
    """Project existing instance identities and kernel-created plain cards.

    This is an expectation, not observed evidence. Complete physical zone and
    customization observations must still verify every row before retention.
    """
    keys = {instance:(canonical,customizations)
            for instance,canonical,customizations in bundle.inventory_keys}
    before = {ref.instance_id:ref for ref in bundle.state.deck.all_cards}
    after = {ref.instance_id:ref for ref in expected.deck.all_cards}
    if len(keys) != len(bundle.inventory_keys) or set(keys) != set(before):
        raise ValueError('HIF permanent instance mapping is incomplete')
    added = set(after)-set(before)
    generated = {f'task275-created-{index}' for index in
                 range(bundle.state.created_card_count,expected.created_card_count)}
    if (not set(before) <= set(after) or expected.created_card_count < bundle.state.created_card_count
            or added != generated):
        raise ValueError('HIF permanent instance change is not a generated-card transition')
    for instance in sorted(added):
        ref = after[instance]
        definition = bundle.kernel.cards.get(ref.card_id)
        if (ref != CardRef(instance_id=instance,card_id=ref.card_id) or definition is None
                or definition.canonical_card_id not in (None,ref.card_id)):
            raise ValueError('HIF generated permanent identity is unresolved')
        keys[instance] = (ref.card_id,())
    return tuple((instance,*value) for instance,value in keys.items())


def _verified_exhausted_item_counts(bundle, baseline_components):
    """Retain observed exhausted counters after runtime trigger refs are pruned."""
    event = baseline_components.get('items', {})
    observation = event.get('identity_count_observation', {})
    rows = observation.get('items', ())
    if (event.get('accepted') is not True or event.get('complete_list') is not True
            or event.get('returned_unchanged') is not True
            or event.get('run_id') != bundle.run_id or event.get('revision') != bundle.revision
            or tuple(row.get('item_id') for row in rows) != bundle.state.items
            or not has_single_frame_id(observation.get('after_frame_ids'))):
        return {}
    rows = _counted_item_rows(bundle, rows)
    if any(type(row.get('definition_effect_index')) is not int
           or row['definition_effect_index'] < 0 for row in rows):
        return {}
    keys = [('item:'+row['item_id'], row['definition_effect_index']) for row in rows]
    if len(set(keys)) != len(keys):
        return {}
    previous = {(ref.source_id, ref.definition_effect_index): ref.remaining_uses
                for ref in bundle.state.scheduled_effects if ref.source_kind is EffectSourceKind.ITEM}
    return {key: 0 for key, row in zip(keys, rows)
            if type(row.get('remaining_uses')) is int and row['remaining_uses'] == 0
            and previous.get(key, 0) == 0}


def _counted_item_rows(bundle, rows):
    """Separate explicitly verified start-only identity from runtime counters."""
    from .opening_items import bind_observed_stage_start_item

    counted = []
    for row in rows:
        if row.get('item_id') == '428' or row.get('effect_scope') is not None:
            bind_observed_stage_start_item(row, definition=bundle.kernel.items.get(row.get('item_id')))
        else:
            counted.append(row)
    return tuple(counted)


def _counted_after_card_score_row(row):
    """Retain the entire observed 1781 line, including its displayed count."""
    if (row.unit != '回' or type(row.count) is not int or row.count <= 0
            or row.remaining_turns is not None or row.per_turn_limit is not None):
        return None
    match = re.fullmatch(r'好調が(\d+)ターン以下の場合、スキルカード使用後(\d+)回ごとに、'
        r'スコア\+(\d+)\(集中効果を(\d+)倍適用\)・好調減少(\d+)', row.description)
    return (row.title, row.count, *map(int, match.groups())) if match else None


def _counted_score_exact_play_matches(bundle, expected, committed_action, self_baseline):
    """Prove draws/generation by complete existing rules, not a hand-size guess."""
    from .kernel import KernelError
    from .contracts import GameAction

    if (expected.turn != bundle.state.turn or expected.pending_choice or expected.resolution_stack
            or committed_action is None and not self_baseline):
        return False
    actions = ((committed_action,) if committed_action is not None else tuple(
        GameAction(ActionKind.PLAY_CARD, card_instance_id=card.instance_id, source_slot=slot)
        for slot, card in enumerate(bundle.state.deck.hand)))
    matches = []
    for action in actions:
        if (action.kind is not ActionKind.PLAY_CARD or type(action.source_slot) is not int
                or not 0 <= action.source_slot < len(bundle.state.deck.hand)
                or action.card_instance_id != bundle.state.deck.hand[action.source_slot].instance_id):
            continue
        try:
            outcomes = bundle.kernel.transition(bundle.state, action, plan_type=bundle.kernel.plan_type)
        except KernelError:
            continue
        if (outcomes and math.isclose(sum(branch.probability for branch in outcomes), 1.0)
                and any(branch.state == expected
                    and [event.source_id for event in branch.events if event.event_type == 'CONSUME_ACTION']
                        == [action.card_instance_id]
                    and not any(event.event_type.startswith(('FREE_PLAY', 'PAID_PLAY')) for event in branch.events)
                    for branch in outcomes)):
            matches.append(action)
    return len(matches) == 1


def _counted_score_automatic_turn_midpoints(bundle, expected, core, committed_action, self_baseline):
    """Recover an exact single-play midpoint before its automatic turn change."""
    from .kernel import KernelError
    from .contracts import GameAction
    from .maintenance import known_turn_advance

    before = bundle.state
    if (before.actions_remaining != 1 or before.pending_choice or before.resolution_stack
            or expected.pending_choice or expected.resolution_stack
            or not known_turn_advance(before, expected, core or {})
            or committed_action is None and not self_baseline):
        return ()
    try:
        match_inventory_hand(core.get('families', ()), tuple(
            bundle.kernel.cards[card.card_id].canonical_card_id or card.card_id for card in expected.deck.hand))
    except (KeyError, TypeError, ValueError):
        return ()
    actions = ((committed_action,) if committed_action is not None else tuple(
        GameAction(ActionKind.PLAY_CARD, card_instance_id=card.instance_id, source_slot=slot)
        for slot, card in enumerate(before.deck.hand)))
    matches = []
    for action in actions:
        if (action.kind is not ActionKind.PLAY_CARD or type(action.source_slot) is not int
                or not 0 <= action.source_slot < len(before.deck.hand)
                or action.card_instance_id != before.deck.hand[action.source_slot].instance_id):
            continue
        try:
            played = bundle.kernel.transition(before, action, plan_type=bundle.kernel.plan_type)
            if not played or not math.isclose(sum(branch.probability for branch in played), 1.0):
                continue
            for branch in played:
                middle = branch.state
                if (middle.turn != before.turn or middle.actions_remaining != 0 or middle.terminal
                        or middle.pending_choice or middle.resolution_stack
                        or [event.source_id for event in branch.events if event.event_type == 'CONSUME_ACTION']
                           != [action.card_instance_id]
                        or any(event.event_type.startswith(('FREE_PLAY', 'PAID_PLAY')) for event in branch.events)):
                    continue
                advanced = bundle.kernel.transition(middle, GameAction(ActionKind.END_TURN), plan_type=bundle.kernel.plan_type)
                if (advanced and math.isclose(sum(item.probability for item in advanced), 1.0)
                        and any(item.state == expected for item in advanced)):
                    matches.append(middle)
        except KernelError:
            continue
    return tuple(dict.fromkeys(matches))


def _initial_counted_card_score(bundle, expected, ref, *, baseline_rows, committed_action,
                                self_baseline=False, core=None):
    """Bind the observed two-card countdown, including its score-trigger reset.

    Frozen upstream checks conditions before incrementing its private counter.
    Thus the registration play leaves counter=1 and does not score. 1781 proves
    that this initial state displays 2; 1785 proves counter=2 displays 1;
    1788 proves the first activation leaves counter=3 and displays 2 again.
    The frozen two-use condition continues that same countdown cycle.
    """
    from .support_schedule import supported_schedule_card, ordinary_counted_score_definition

    ordinary = bundle.kernel.cards[ref.card_id]
    if (ordinary != ordinary_counted_score_definition()
            or ref.definition_effect_index not in (2, 3)
            or ref.trigger_phase is not SchedulePhase.AFTER_CARD_USED
            or ref.persistent or ref.direct_on_trigger or ref.duration_fresh
            or ref.turns_until_fire != 0 or ref.remaining_uses is not None
            or ref.remaining_turns is not None
            or type(ref.effect_counter) is not int or ref.effect_counter < 1
            or (ref.effect_counter <= 2 and ref.last_fired_turn is not None)
            or (ref.effect_counter >= 3 and (type(ref.last_fired_turn) is not int
                or not 1 <= ref.last_fired_turn <= expected.turn))
            or sum(card.instance_id == ref.source_id and card.card_id == ref.card_id
                   for card in expected.deck.removed) != 1):
        return None
    effective = supported_schedule_card(ordinary) if ref.definition_effect_index == 3 else ordinary
    if bundle.kernel._scheduled_effect_spec(ref) != effective.effects[2]:
        return None
    signature = (schedule_status_name(bundle.kernel.cards, ref), 2-(ref.effect_counter-1) % 2, 20, 2,
        int(effective.effects[2].scheduled_effects[0].amount), 4, 3)
    peers = tuple(item for item in expected.scheduled_effects
        if item.source_id == ref.source_id and item.card_id == ref.card_id)
    previous = tuple(item for item in bundle.state.scheduled_effects
        if item.source_id == ref.source_id and item.card_id == ref.card_id)
    observed = Counter(value for row in baseline_rows
        if (value := _counted_after_card_score_row(row)) is not None)
    if previous:
        # A verified turn advance does not itself use a card. Retain this
        # exact row only when all source refs remain unchanged and the new
        # three-frame HUD matches the established next-turn schedule.
        if Counter(peers) == Counter(previous) and observed[signature] == len(peers):
            from .maintenance import known_turn_advance

            if (expected.turn == bundle.state.turn
                    or known_turn_advance(bundle.state, expected, core or {})):
                return signature
        if expected.turn != bundle.state.turn:
            # The final action can fire this counter in the old turn and then
            # discard/refill the remaining hand. Verify those two existing
            # kernel transitions exactly, then reuse the same-turn proof.
            middles = _counted_score_automatic_turn_midpoints(
                bundle, expected, core, committed_action, self_baseline)
            valid = [middle for middle in middles if ref in middle.scheduled_effects
                and _initial_counted_card_score(bundle, middle, ref, baseline_rows=baseline_rows,
                    committed_action=committed_action, self_baseline=self_baseline) == signature]
            return signature if len(valid) == 1 else None
        # One later card advances each private counter. A positive even
        # pre-increment count fires the two-use condition and resets the UI.
        prior_signature = (signature[0], 3-signature[1], *signature[2:])
        advanced = Counter(replace(item, effect_counter=item.effect_counter+1,
            last_fired_turn=(expected.turn if item.effect_counter >= 2
                and item.effect_counter % 2 == 0 else item.last_fired_turn)) for item in previous)
        if (ref.effect_counter < 2
                or any(item.effect_counter != ref.effect_counter-1 for item in previous)
                or Counter(peers) != advanced
                or observed[signature if self_baseline else prior_signature] != len(peers)):
            return None
        if _counted_score_exact_play_matches(bundle, expected, committed_action, self_baseline):
            return signature
        moved = tuple((Counter(bundle.state.deck.hand) - Counter(expected.deck.hand)).elements())
        if (len(moved) != 1 or moved[0].instance_id == ref.source_id
                or Counter(bundle.state.deck.hand) != Counter(expected.deck.hand) + Counter(moved)
                or not _counted_score_card_generation_matches(bundle, expected, moved[0])
                or bundle.state.deck.held != expected.deck.held
                or Counter((*expected.deck.discard, *expected.deck.removed))
                    != Counter((*bundle.state.deck.discard, *bundle.state.deck.removed)) + Counter(moved)):
            return None
        if committed_action is None:
            return signature if self_baseline else None
        if (committed_action.kind is not ActionKind.PLAY_CARD
                or committed_action.card_instance_id != moved[0].instance_id
                or type(committed_action.source_slot) is not int
                or not 0 <= committed_action.source_slot < len(bundle.state.deck.hand)
                or bundle.state.deck.hand[committed_action.source_slot] != moved[0]):
            return None
        return signature
    if ref.effect_counter != 1 or expected.turn != bundle.state.turn:
        return None
    if self_baseline and committed_action is None:
        # History can recheck the complete accepted successor against itself
        # while its bundle still owns the pre-play state. Require exactly that
        # physical move and the full already accepted row multiset.
        cards = tuple(card for card in bundle.state.deck.hand
            if card.instance_id == ref.source_id and card.card_id == ref.card_id)
        if (len(cards) == 1 and observed[signature] == len(peers)
                and len(peers) == 1 + int(bundle.state.double_card_effects_remaining > 0)
                and Counter(bundle.state.deck.all_cards) == Counter(expected.deck.all_cards)
                and Counter(bundle.state.deck.hand) == Counter(expected.deck.hand) + Counter(cards)
                and Counter(expected.deck.removed) == Counter(bundle.state.deck.removed) + Counter(cards)
                and expected.double_card_effects_remaining == max(0, bundle.state.double_card_effects_remaining - 1)):
            return signature
        return None
    if (committed_action is None or committed_action.kind is not ActionKind.PLAY_CARD
            or committed_action.card_instance_id != ref.source_id
            or type(committed_action.source_slot) is not int
            or not 0 <= committed_action.source_slot < len(bundle.state.deck.hand)
            or bundle.state.deck.hand[committed_action.source_slot].instance_id != ref.source_id
            or bundle.state.deck.hand[committed_action.source_slot].card_id != ref.card_id
            or len(peers) != 1 + int(bundle.state.double_card_effects_remaining > 0)):
        return None
    return signature


def _counted_score_card_generation_matches(bundle, expected, played):
    """Keep old cards fixed while accounting for declared plain deck additions."""
    before = bundle.state
    if before.created_card_count == expected.created_card_count:
        return (Counter(before.deck.all_cards) == Counter(expected.deck.all_cards)
            and before.deck.draw_pile == expected.deck.draw_pile)
    # Native 1830: 708's ordinary/+ bodies both add one 23. The counted
    # trigger still advances once; generation must not impersonate a draw,
    # changed old instance, or an undeclared extra card.
    definition = bundle.kernel.cards[played.card_id]
    additions = tuple(effect for effect in definition.effects
        if effect.kind is EffectKind.ADD_CARD_TO_DECK)
    if not additions or any(effect != EffectSpec(EffectKind.ADD_CARD_TO_DECK,
            card_id=effect.card_id) for effect in additions):
        return False
    try:
        expected_permanent_inventory_keys(bundle, expected)
    except ValueError:
        return False
    old_ids = {card.instance_id for card in before.deck.all_cards}
    added = tuple(card for card in expected.deck.all_cards if card.instance_id not in old_ids)
    declared = tuple(effect.card_id for effect in additions) * (
        1 + int(before.double_card_effects_remaining > 0))
    return (Counter(card.card_id for card in added) == Counter(declared)
        and Counter(expected.deck.all_cards) == Counter(before.deck.all_cards) + Counter(added)
        and Counter(expected.deck.draw_pile) == Counter(before.deck.draw_pile) + Counter(added)
        and tuple(card for card in expected.deck.draw_pile if card.instance_id in old_ids)
            == before.deck.draw_pile)


def verify_hif_committed_observation(bundle, expected, *, core, components, baseline_components,
                                   preinput_continuity=None, committed_action=None):
    """Check new native observations against a committed model transition.

    Changed components are freshly observed. Unchanged components retain their
    verified source, and visible hand moves can maintain the card ledger.
    Unknown status semantics leave the complete-state gate closed.
    """
    checks = {}
    if bundle.state.hand_support or expected.hand_support:
        from .hif_support_continuity import hand_support_candidate_continuity
        checks['hand_support_continuity'] = hand_support_candidate_continuity(bundle.state, expected)
    if preinput_continuity is not None:
        from .maintenance import verify_preinput_continuity
        verify_preinput_continuity(bundle, expected, baseline_components, preinput_continuity)
    for key in ('cards', 'status', 'items', 'drinks', 'turns'):
        event = components.get(key, {})
        if (event.get('accepted') is not True or event.get('returned_unchanged') is not True
                or event.get('run_id') != bundle.run_id or event.get('revision') != bundle.revision):
            raise ValueError('HIF settlement component unbound: '+key)
        source = event.get('observation_source')
        if source == 'retained_preinput_no_business_change':
            from .maintenance import verify_preinput_component
            if preinput_continuity is None:
                raise ValueError('HIF preinput retention requires its in-process no-submission proof')
            verify_preinput_component(key, event, baseline_components[key], bundle=bundle,
                proof=preinput_continuity)
            checks['preinput_retained_'+key] = True
        if source in ('retained_unchanged_component', 'maintained_card_ledger', 'maintained_turn_schedule'):
            from .contracts import GameAction
            from .maintenance import known_turn_advance, unchanged_component, card_maintenance_action_valid, visible_hand_accounts_for_cards
            if (event.get('source_epoch') != dict(run_id=bundle.run_id, revision=bundle.revision)
                    or not has_single_frame_id(event.get('maintenance_frame_ids'))):
                raise ValueError('HIF maintained component has no continuity evidence: '+key)
            if source == 'retained_unchanged_component':
                if key == 'cards':
                    raise ValueError('HIF retained component has an invalid source kind: '+key)
                checks['maintained_'+key] = unchanged_component(key, bundle.state, expected)
            elif source == 'maintained_turn_schedule':
                if key != 'turns':
                    raise ValueError('HIF maintained turn schedule has an invalid component key')
                checks['maintained_'+key] = known_turn_advance(bundle.state, expected, core)
            else:
                if key != 'cards':
                    raise ValueError('HIF maintained card ledger has an invalid component key')
                raw = dict(event.get('maintenance_action', {}))
                raw['kind'] = ActionKind(raw.get('kind'))
                action = GameAction(**raw)
                action.validate()
                if not card_maintenance_action_valid(bundle, action):
                    raise ValueError('HIF maintained card ledger has an invalid source action')
                # A valid source can disprove one random outcome without
                # invalidating the other candidates of the same submission.
                checks['maintained_'+key] = visible_hand_accounts_for_cards(bundle, expected, core, action)
    fields = (expected.turns_remaining, expected.stamina, expected.genki,
              expected.score, expected.score_multiplier)
    checks.update(core=tuple(core.get('fields', ()))[:5] == fields,
                  core_frames=has_single_frame_id(core.get('frame_ids')))
    def card_key(row):
        return row['card_id'], tuple(sorted((int(k), v) for k,v in row['customizations'].items()))
    # Physical zone order changes after a play. Permanent identity belongs to
    # the original instance, never the position in a later observation list.
    initial_keys = {instance:(canonical,customizations)
                    for instance,canonical,customizations in expected_permanent_inventory_keys(bundle,expected)}
    zones = dict(手札=expected.deck.hand, 山札=expected.deck.draw_pile,
                 捨札=expected.deck.discard, 除外=expected.deck.removed)
    actual_rows = components['cards']['cards']
    if any(row.get('temporary_support') is not None for row in actual_rows):
        from ..inventory_support import verify_inventory_support_row
        checks['temporary_support_permanent_identity'] = all(
            verify_inventory_support_row(bundle, row) for row in actual_rows)
    expected_counts = {zone: len(refs) for zone,refs in zones.items()}
    checks['zone_counts'] = components['cards']['zone_counts'] == expected_counts
    retained_card_epochs = [dict(run_id=bundle.run_id, revision=bundle.revision)]
    if (checks.get('preinput_retained_cards') is True
            and preinput_continuity.history_card_epoch is not None):
        # History recovery preserves the original row provenance. The explicit
        # input-free proof above requires these rows to equal that verified
        # baseline verbatim; it does not relabel them as newly observed rows.
        retained_card_epochs.append(preinput_continuity.history_card_epoch)
    from ..inventory_customization import verified_zero_customization_detail

    checks['permanent_card_sources'] = all(
        (row.get('source') == 'confirmed_deck_baseline'
         # The preinput proof already checked the complete unchanged source
         # against this exact run, revision, state and rule definitions. A
         # fresh or submitted observation cannot assert this source itself.
         and checks.get('preinput_retained_cards') is True)
        or (row.get('source') in ('observed_detail','observed_badge_absence','retained_verified_detail','retained_verified_card')
        and (row['source']=='retained_verified_card' or
             (row['source'] != 'observed_badge_absence') == bool(row['customizations']) or
             (row['source'] == 'observed_detail' and verified_zero_customization_detail(row)))
        and (not row['source'].startswith('retained_verified_') or row.get('retained_from') in
             retained_card_epochs)) for row in actual_rows)
    if expected.created_card_count != bundle.state.created_card_count:
        # New instances invalidate retention for their entire canonical family.
        # Other unique variants keep the same-run proof only if every old
        # instance is unchanged; all physical zones are still freshly scanned.
        from ..inventory_customization import retained_ambiguous_card_ids

        fresh_ids = retained_ambiguous_card_ids(bundle, expected)
        checks['permanent_card_sources'] &= all(
            row.get('source') in ('observed_detail','observed_badge_absence')
            or (row.get('source') == 'retained_verified_card' and row['card_id'] not in fresh_ids)
            for row in actual_rows)
    for zone,refs in zones.items():
        rows = [row for row in actual_rows if row['zone'] == zone]
        checks['zone_'+zone] = (tuple(row['index'] for row in rows) == tuple(range(len(refs)))
            and Counter(card_key(row) for row in rows) == Counter(initial_keys[ref.instance_id] for ref in refs))
    checks['all_card_zones'] = len(actual_rows) == sum(expected_counts.values())
    hand_canonical = tuple(bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id for ref in expected.deck.hand)
    observed_hand = []
    try:
        # The search kernel sorts equivalent states; its tuple order is not a
        # physical UI slot order. Rebind from the newly observed hand only.
        ids = match_inventory_hand(core.get('families', ()), hand_canonical)
        for cid in ids:
            candidates = [ref for ref in expected.deck.hand if ref not in observed_hand
                and (bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id) == cid]
            from .hif_support_continuity import observed_hand_variant_key
            if not candidates or len({observed_hand_variant_key(expected, ref) for ref in candidates}) != 1:
                raise ValueError('post-action hand variants are ambiguous')
            observed_hand.append(candidates[0])
        checks['hand_identity'] = True
    except ValueError:
        checks['hand_identity'] = False
    actual_items = components['items']['identity_count_observation']['items']
    checks['item_inventory'] = tuple(row['item_id'] for row in actual_items) == expected.items
    counted_items = _counted_item_rows(bundle, actual_items)
    stage_start_ids = {row['item_id'] for row in actual_items if row not in counted_items}
    # The kernel prunes exhausted triggers on later transitions. Their visible
    # zero badges remain; retain only previously verified zero-use records.
    counts = _verified_exhausted_item_counts(bundle, baseline_components)
    counts.update({(ref.source_id, ref.definition_effect_index): 0
              for ref in bundle.state.scheduled_effects
              if ref.source_kind is EffectSourceKind.ITEM and ref.remaining_uses == 0})
    counts.update({(ref.source_id, ref.definition_effect_index): ref.remaining_uses
                   for ref in expected.scheduled_effects if ref.source_kind is EffectSourceKind.ITEM})
    checks['item_counts'] = (not any(ref.source_kind is EffectSourceKind.ITEM
                                   and ref.card_id in stage_start_ids for ref in expected.scheduled_effects)
        and all(type(row.get('remaining_uses')) is int and row['remaining_uses'] >= 0
                and type(row.get('definition_effect_index')) is int and row['definition_effect_index'] >= 0
                and counts.get(('item:'+row['item_id'], row['definition_effect_index'])) == row['remaining_uses']
                for row in counted_items))
    checks['drinks'] = (components['drinks'].get('full_drink_state_ready') is True
                        and tuple(components['drinks']['drinks']) == expected.drinks)
    checks['turns'] = (components['turns']['current_turn'] == expected.turn
        and tuple(components['turns']['turn_attributes']) == expected.turn_attributes
        and tuple(components['turns']['score_multipliers']) == (expected.score_multiplier, *expected.future_score_multipliers))
    def groups_of(event):
        return tuple(tuple(StatusPanelPage(
            rows=tuple(StatusPanelRow(**dict(row, box=tuple(row['box']))) for row in page['rows']),
            thumb=tuple(page['thumb']), at_top=page['at_top'], at_bottom=page['at_bottom'],
            partial_row_count=page['partial_row_count']) for page in group['observations']) for group in event['pages'])
    baseline_groups=groups_of(baseline_components['status'])
    current_groups=groups_of(components['status'])
    baseline_rows = join_status_panel_pages(tuple(status_panel_consensus(group) for group in baseline_groups))
    prior_cost_rows = [row for row in baseline_rows if row.title == '消費体力変化']
    rows = join_status_panel_pages(tuple(status_panel_consensus(group) for group in current_groups))
    from ..status_panel import memory_use_counts
    memory_observations=memory_use_counts(baseline_groups,current_groups,
        previous_run_id=bundle.run_id,current_run_id=bundle.run_id)
    # The kernel removes exhausted runtime effects on a later resolution.
    # Stable memory identities remain in memory_abilities; an absent runtime
    # effect therefore has zero remaining uses, still checked against the
    # complete observed list and its baseline disappearance evidence below.
    expected_memories=dict.fromkeys(expected.memory_abilities,0)
    expected_memories.update({ref.card_id:ref.remaining_uses for ref in expected.scheduled_effects
        if ref.source_kind is EffectSourceKind.MEMORY})
    checks['memory_history']=(set(expected.memory_abilities)==set(bundle.state.memory_abilities)
        and set(expected_memories)==set(expected.memory_abilities))
    checks['memory_rows']=True
    observed_memory_ids=set()
    for observation in memory_observations:
        name=observation['description_key'].split('使用後、',1)[0]+'使用後'
        ids=[key for key,definition in bundle.kernel.memory_abilities.items() if definition.name==name]
        if len(ids)!=1 or expected_memories.get(ids[0])!=observation['remaining_uses']:
            checks['memory_rows']=False
        else:
            observed_memory_ids.add(ids[0])
    checks['memory_rows'] &= all(count==0 for key,count in expected_memories.items() if key not in observed_memory_ids)
    content = lambda row: row.content_key
    expected_scalar = {'集中': (expected.concentration, ''), '好調': (expected.good_condition_turns, 'ターン'),
                       '絶好調': (expected.excellent_condition_turns, 'ターン'),
                       '消費体力減少': (expected.half_cost_turns, 'ターン'),
                       '消費体力変化': (expected.nullify_cost_cards, '回'),
                       '消費体力増加': (expected.double_cost_turns, 'ターン'),
                       '元気増加無効': (expected.nullify_genki_turns, 'ターン'),
                       '消費体力削減': (expected.cost_reduction, ''),
                       'スキルカード追加発動': (expected.double_card_effects_remaining, '回'),
                       # One currently usable action has no additional-use badge.
                       'スキルカード使用数追加': (max(0, expected.actions_remaining - 1), '回')}
    extra = []
    seen = set()
    addition_rows = []
    percentage_rows = []
    score_percentage_rows = []
    timer_rows = []
    start_score_rows = []
    permanent_focus_rows = []
    active_score_rows = []
    typed_focus_rows = []
    counted_card_score_rows = []
    permanent_drink_rows = []
    finite_drink_draw_rows = []
    repeat_rows = []
    card_names={definition.name for definition in bundle.kernel.cards.values()}
    expired_start_scores = verified_expired_start_scores(bundle, expected)
    card_names.update(schedule_status_name(bundle.kernel.cards, ref)
        for ref in (*expected.scheduled_effects, *(ref for _, ref in expired_start_scores))
        if is_support_schedule(ref))
    reservation_rows = []
    score_reservation_rows = []
    for row in rows:
        if row.per_turn_limit is not None and not row.title.endswith('(再演)'):
            extra.append(content(row))
            continue
        if row.title == 'メモリーアビリティ':
            continue
        if row.title in expected_scalar:
            cost_description = {
                '消費体力変化': f'使用したスキルカードの消費体力を0にする({row.count}回)',
                '消費体力増加': f'消費体力が100%増加{row.count}ターン',
                # Native 1178: the remaining next-card repeats are a count,
                # with no turn expiry; this is not an extra playable action.
                'スキルカード追加発動': f'スキルカードの効果をもう1回発動{row.count}回∞ターン',
            }.get(row.title)
            if row.title == '消費体力変化' and len(prior_cost_rows) == 1:
                prior = prior_cost_rows[0]
                original = re.fullmatch(r'使用したスキルカードの消費体力を0にする\(([1-9]\d*)回\)', prior.description)
                # History revalidates the already verified complete successor
                # against itself while retaining the original pre-action bundle.
                baseline_count = (expected.nullify_cost_cards if baseline_components == components
                    else bundle.state.nullify_cost_cards)
                if (original and type(row.count) is int and type(prior.count) is int
                        and row.count > 0 and prior.count == baseline_count > 0
                        and prior.unit == '回' and prior.remaining_turns is None
                        and prior.per_turn_limit is None):
                    amount = int(original[1])
                    known_body = any(effect.kind is EffectKind.NULLIFY_COST_CARDS
                        and effect.condition is None and effect.amount == amount
                        for drink in bundle.kernel.drinks.values() for effect in drink.effects)
                    # Native 899 -> 903 and 1775: the heading holds the live
                    # total, while the body retains the verified source grant.
                    retained = row.count <= prior.count and (prior.count <= amount or known_body)
                    if (row.count > prior.count and committed_action is not None
                            and committed_action.kind is ActionKind.USE_DRINK):
                        drink = bundle.kernel.drinks.get(committed_action.drink_id)
                        grants = tuple(effect.amount for effect in drink.effects
                            if effect.kind is EffectKind.NULLIFY_COST_CARDS and effect.condition is None) if drink else ()
                        retained = (grants == (amount,)
                            and expected.nullify_cost_cards - bundle.state.nullify_cost_cards == amount
                            and Counter(bundle.state.drinks) == Counter(expected.drinks)
                                + Counter({committed_action.drink_id: 1}))
                    # A current body can describe the remaining count (1830),
                    # while 903/1775 retain the source grant. Only select the
                    # latter form when that is what was actually observed.
                    if retained and row.description == prior.description:
                        cost_description = prior.description
            checks['status_'+row.title] = (checks.get('status_'+row.title, True)
                and row.title not in seen
                and row.remaining_turns == (1 if row.title == 'スキルカード使用数追加' else None)
                and (cost_description is None or (type(row.count) is int and row.count > 0
                     and row.description == cost_description))
                and (row.count, row.unit) == expected_scalar[row.title])
            seen.add(row.title)
        elif row.title == '集中増加量追加':
            addition_rows.append((row.count, row.unit, row.remaining_turns))
        elif row.title == '集中強化':
            match=re.fullmatch(r'集中によるスコア上昇量を(\d+)%増加(∞|\d+)ターン',row.description)
            duration=None if match and match[2]=='∞' else int(match[2]) if match else None
            if match and row.unit=='%' and row.count==int(match[1]) and row.remaining_turns==duration:
                percentage_rows.append((row.count,duration))
            else:
                extra.append(content(row))
        elif row.title == 'パラメータ上昇量増加':
            # Native 1615 reads the entire clause, including 好印象. The
            # opening parenthesis beside its pictogram may be absent in OCR;
            # retain the actual text and compare every word and the amount.
            match = re.fullmatch(r'スコア上昇量を(\d+)%増加\(?好印象による上昇も含む\)', row.description)
            if (match and row.unit == '%' and row.count == int(match[1]) and row.count > 0
                    and type(row.remaining_turns) is int and row.remaining_turns > 0):
                score_percentage_rows.append((row.count, row.remaining_turns))
            else:
                extra.append(content(row))
        elif row.title == 'センブリソーダ':
            if (row.unit == 'ターン' and type(row.count) is int and 0 < row.count <= 5
                    and row.remaining_turns is None
                    and row.description == f'ターン開始後、スキルカードを引く{row.count}ターン'):
                finite_drink_draw_rows.append((row.title, row.count))
            else:
                extra.append(content(row))
        elif (row.title == '厳選初星マキアート' and row.count is None and row.unit == ''
                and row.remaining_turns is None and row.per_turn_limit is None
                and row.description == 'ターン終了時、集中+1'):
            permanent_drink_rows.append((row.title, 1))
        elif row.title in card_names and _counted_after_card_score_row(row) is not None:
            counted_card_score_rows.append(_counted_after_card_score_row(row))
        elif (row.title in card_names and row.count is None and row.unit == ''
                and row.remaining_turns is None and row.per_turn_limit is None):
            match = re.fullmatch(r'ターン終了時、集中が(\d+)以上の場合、集中\+(\d+)', row.description)
            if match:
                permanent_focus_rows.append((row.title, int(match[1]), int(match[2])))
            elif (row.title == '至高のエンタメ+'
                    and row.description == 'アクティブスキルカード使用時、スコア+5'):
                active_score_rows.append((row.title, 5))
            elif (match := re.fullmatch(r'(アクティブ|メンタル)スキルカード使用時、集中\+([1-9]\d*)',
                    row.description)):
                typed_focus_rows.append((row.title,
                    {'アクティブ': 'active', 'メンタル': 'mental'}[match[1]], int(match[2])))
            else:
                extra.append(content(row))
        elif row.title in card_names and row.unit=='ターン' and row.remaining_turns is None:
            suffix=str(row.count)+'ターン'
            match=re.fullmatch(r'ターン終了時、集中\+(\d+)',row.description.removesuffix(suffix))
            start_score=re.fullmatch(r'ターン開始時、スコア\+(\d+)',row.description.removesuffix(suffix))
            if row.description.endswith(suffix) and match and row.per_turn_limit is None:
                timer_rows.append((row.title,row.count,int(match[1])))
            elif (row.description.endswith(suffix) and start_score and row.per_turn_limit is None
                    and type(row.count) is int and row.count > 0):
                # Formal 268 reads the recurring score duration as ターン.
                # Keep that observed unit; the fixed rule counts activations.
                start_score_rows.append((row.title,row.count,int(start_score[1])))
            else:
                extra.append(content(row))
        elif row.title.removesuffix('(再演)') in card_names and row.title.endswith('(再演)'):
            match=re.fullmatch(r'(\d+)ターンごとに、体力が(\d+)%以上の場合、自身を再使用(\d+)回',row.description)
            if match and row.unit=='回' and row.count==int(match[3]):
                repeat_rows.append((row.title,row.count,row.remaining_turns,row.per_turn_limit,int(match[1]),int(match[2])))
            else:
                extra.append(content(row))
        elif row.title == '発動予約':
            match=re.fullmatch(r'(\d+)ターン後、スキルカードを(?:(\d+)枚)?引く',row.description)
            score_match = re.fullmatch(r'([1-9]\d*)ターン後、スコア\+([1-9]\d*)'
                r'(?:\(好調効果を([1-9]\d*)倍適用\))?', row.description)
            if match and row.unit=='回' and row.remaining_turns==int(match[1]):
                reservation_rows.append((int(match[1]),row.count,int(match[2]) if match[2] else 1))
            elif (score_match and row.unit == '回' and type(row.count) is int and row.count > 0
                    and row.remaining_turns == int(score_match[1])):
                score_reservation_rows.append((int(score_match[1]), row.count,
                    int(score_match[2]), int(score_match[3]) if score_match[3] else 1))
            else:
                extra.append(content(row))
        else:
            extra.append(content(row))
    checks['scalar_presence'] = seen == {name for name,(value,_) in expected_scalar.items() if value}
    # Fixed addition differs from a percentage modifier. Compare both amount
    # and duration from the actual row; no expected value fills an absent row.
    additions = {}
    for modifier in expected.concentration_addition_buffs:
        additions[modifier.remaining_turns] = additions.get(modifier.remaining_turns, 0) + modifier.amount
    checks['concentration_addition'] = Counter(addition_rows) == Counter(
        (amount, '', turns) for turns, amount in additions.items())
    percentages={}
    for modifier in expected.concentration_effect_buffs:
        percentages[modifier.remaining_turns]=percentages.get(modifier.remaining_turns,0)+modifier.amount*100
    checks['concentration_percentage']=Counter(percentage_rows)==Counter(
        (round(amount,8),turns) for turns,amount in percentages.items())
    # Compare each observed finite row to its actual modifier. Native 1615
    # proves one 10%/5-turn row, not a rule for merging simultaneous modifiers.
    checks['score_percentage'] = Counter(score_percentage_rows) == Counter(
        (round(modifier.amount * 100, 8), modifier.remaining_turns)
        for modifier in expected.score_buffs)
    expected_reservations=[]
    expected_score_reservations=[]
    reservation_semantics=True
    score_reservation_semantics=True
    for ref in expected.scheduled_effects:
        if ref.source_kind is not EffectSourceKind.CARD or not ref.direct_on_trigger:
            continue
        effect=bundle.kernel.cards[ref.card_id].effects[ref.definition_effect_index]
        child = effect.scheduled_effects[0] if len(effect.scheduled_effects) == 1 else None
        if (effect.kind is EffectKind.SCHEDULE and ref.trigger_phase is SchedulePhase.TURN
                and child is not None and child.kind is EffectKind.DRAW and child.condition is None):
            expected_reservations.append((ref.turns_until_fire,ref.remaining_uses,child.amount))
        elif child is not None and child.kind is EffectKind.SCORE:
            # 1681's remaining 147 reservation is a future score, not a draw.
            # Bind the complete plain one-shot declaration and its live source;
            # compare all observed terms without consuming or predicting it.
            if (effect != EffectSpec(EffectKind.SCHEDULE, delay_turns=effect.delay_turns,
                    scheduled_effects=(child,))
                    or child != EffectSpec(EffectKind.SCORE, amount=child.amount,
                        score_good_condition_multiplier=child.score_good_condition_multiplier)
                    or ref.trigger_phase is not SchedulePhase.TURN
                    or type(effect.delay_turns) is not int or type(ref.turns_until_fire) is not int
                    or not 1 <= ref.turns_until_fire <= effect.delay_turns
                    or type(ref.remaining_uses) is not int or ref.remaining_uses != 1
                    or ref.remaining_turns is not None or ref.persistent or ref.duration_fresh
                    or ref.effect_counter != 0 or ref.last_fired_turn is not None
                    or not all(type(value) in (int, float) and math.isfinite(value)
                        and value > 0 and float(value).is_integer()
                        for value in (child.amount, child.score_good_condition_multiplier))
                    or sum(card.instance_id == ref.source_id and card.card_id == ref.card_id
                        for card in expected.deck.all_cards) != 1):
                score_reservation_semantics=False
                continue
            expected_score_reservations.append((ref.turns_until_fire,ref.remaining_uses,
                child.amount,child.score_good_condition_multiplier))
        else:
            reservation_semantics=False
    checks['scheduled_draws']=(reservation_semantics
        and Counter(reservation_rows)==Counter(expected_reservations))
    checks['scheduled_scores']=(score_reservation_semantics
        and Counter(score_reservation_rows)==Counter(expected_score_reservations))
    expected_timers=[]
    expected_start_scores=[]
    expected_permanent_focus=[]
    expected_active_scores=[]
    expected_typed_focus=[]
    expected_repeats=[]
    permanent_focus_effect = EffectSpec(EffectKind.SCHEDULE, delay_turns=0,
        schedule_phase=SchedulePhase.END_OF_TURN, schedule_limit=None,
        scheduled_effects=(EffectSpec(EffectKind.CONCENTRATION, amount=2,
            condition=EffectCondition(ConditionKind.CONCENTRATION_MIN, threshold=3)),))
    active_score_effect = EffectSpec(EffectKind.SCHEDULE, delay_turns=0,
        schedule_phase=SchedulePhase.CARD_USED, schedule_limit=None,
        trigger_card_type='active',
        scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=5),))
    named_semantics=True
    expected_counted_card_scores = []
    for ref in expected.scheduled_effects:
        if ref.source_kind is not EffectSourceKind.CARD or ref.direct_on_trigger:
            continue
        if ref.remaining_turns == 0:
            continue
        definition=bundle.kernel.cards[ref.card_id]
        effect=bundle.kernel._scheduled_effect_spec(ref)
        if definition.card_id == '762':
            value = _initial_counted_card_score(bundle, expected, ref,
                baseline_rows=baseline_rows, committed_action=committed_action,
                self_baseline=baseline_components == components, core=core)
            if value is None:
                named_semantics=False
            else:
                expected_counted_card_scores.append(value)
        elif (ref.trigger_phase.value=='END_OF_TURN' and len(effect.scheduled_effects)==1
                and effect.scheduled_effects[0].kind.value=='CONCENTRATION'
                and effect.scheduled_effects[0].condition is None and ref.remaining_uses is None):
            expected_timers.append((definition.name,ref.remaining_turns,effect.scheduled_effects[0].amount))
        elif (definition.base_card_id == '168' and effect == permanent_focus_effect
                and ref.trigger_phase is SchedulePhase.END_OF_TURN and ref.turns_until_fire == 0
                and ref.remaining_turns is None and ref.remaining_uses is None and not ref.persistent):
            # Native 910 shows no count: this independent conditional effect
            # lasts for the Live. It is not an unconditional finite timer.
            expected_permanent_focus.append((definition.name, 3, 2))
        elif (definition.card_id == '301' and ref.definition_effect_index == 1
                and effect == active_score_effect
                and ref.trigger_phase is SchedulePhase.CARD_USED and ref.turns_until_fire == 0
                and ref.remaining_turns is None and ref.remaining_uses is None and not ref.persistent
                and any(card.instance_id == ref.source_id and card.card_id == ref.card_id
                        for card in expected.deck.all_cards)):
            # Native 1049: no counter is displayed for this unlimited active-
            # card trigger. Its separate one-turn draw remains checked above.
            expected_active_scores.append((definition.name, 5))
        elif (effect.trigger_card_type in ('active', 'mental')
                and len(effect.scheduled_effects) == 1
                and effect.scheduled_effects[0].kind is EffectKind.CONCENTRATION):
            # Native 1184 / 1788: both card-type-triggered focus rows are
            # permanent and uncounted. Match the complete compiled effect,
            # actual card name, and its one physical source instance.
            amount = effect.scheduled_effects[0].amount
            exact = EffectSpec(EffectKind.SCHEDULE, delay_turns=0,
                schedule_phase=SchedulePhase.CARD_USED, schedule_limit=None,
                trigger_card_type=effect.trigger_card_type,
                scheduled_effects=(EffectSpec(EffectKind.CONCENTRATION, amount=amount),))
            if (type(amount) not in (int, float) or not math.isfinite(amount)
                    or amount <= 0 or not float(amount).is_integer() or effect != exact
                    or ref.trigger_phase is not SchedulePhase.CARD_USED or ref.turns_until_fire != 0
                    or ref.remaining_turns is not None or ref.remaining_uses is not None
                    or ref.persistent or ref.duration_fresh or ref.effect_counter != 0
                    or not 0 <= ref.definition_effect_index < len(definition.effects)
                    or definition.effects[ref.definition_effect_index] != effect
                    or sum(card.instance_id == ref.source_id and card.card_id == ref.card_id
                        for card in expected.deck.all_cards) != 1):
                named_semantics=False
            else:
                expected_typed_focus.append((definition.name, effect.trigger_card_type, amount))
        elif ((definition.canonical_card_id or definition.card_id) == '597'
                and len(effect.scheduled_effects) == 1
                and ref.trigger_phase is SchedulePhase.START_OF_TURN
                and ref.turns_until_fire == 1 and ref.remaining_turns is None
                and type(ref.remaining_uses) is int and 0 < ref.remaining_uses <= 4
                and not ref.persistent
                and effect == EffectSpec(EffectKind.SCHEDULE, delay_turns=1,
                    schedule_phase=SchedulePhase.START_OF_TURN, schedule_limit=4,
                    scheduled_effects=(EffectSpec(EffectKind.SCORE,
                        amount=effect.scheduled_effects[0].amount),))):
            # Native 912 -> 913: the start-of-turn score has already fired
            # in the current turn. Its visible duration includes that turn,
            # while the kernel retains only future activations. Final-use
            # residue below requires its exact original predecessor ref.
            visible_turns = ref.remaining_uses + int(ref.last_fired_turn == expected.turn)
            if visible_turns > effect.schedule_limit:
                named_semantics=False
                continue
            expected_start_scores.append((schedule_status_name(bundle.kernel.cards, ref),visible_turns,
                effect.scheduled_effects[0].amount))
        elif (ref.trigger_phase.value=='EVERY_TURN' and ref.persistent and len(effect.scheduled_effects)==1
                and effect.scheduled_effects[0].kind.value=='FREE_PLAY_RANDOM'):
            child=effect.scheduled_effects[0]
            condition=child.condition
            conditions={c.kind.value:c for c in condition.all_of} if condition and condition.kind.value=='ALL' else {}
            if (set(conditions)!={'REMOVED_BASE_CARD_COUNT_MIN','EFFECT_COUNTER_MOD_EQ','STAMINA_RATIO_MIN'}
                    or len(condition.all_of)!=3 or child.card_selector is None
                    or child.card_selector.base_card_id!=conditions['REMOVED_BASE_CARD_COUNT_MIN'].value):
                named_semantics=False
                continue
            removed=conditions['REMOVED_BASE_CARD_COUNT_MIN']
            if sum(bundle.kernel.cards[c.card_id].base_card_id==removed.value for c in expected.deck.removed)<removed.threshold:
                continue
            period=conditions['EFFECT_COUNTER_MOD_EQ']
            ratio=conditions['STAMINA_RATIO_MIN']
            if not period.modulus or period.remainder is None:
                named_semantics=False
                continue
            # Native 1079 step 2: after 816's final replay the lifetime row
            # disappears from a completely observed status list. A zero
            # turn-local activation counter with future uses remains visible.
            if (ref.remaining_uses == 0 and definition.base_card_id == '815'
                    and ref.definition_effect_index == 0 and effect.schedule_limit == 2
                    and ref.last_fired_turn == expected.turn
                    and not any(row[0] == definition.name+'(再演)' for row in repeat_rows)):
                continue
            countdown=(period.remainder-ref.effect_counter)%period.modulus+1
            # The UI's turn-local count is remaining activations, not a fixed
            # limit. A replay fired at this turn's start displays zero.
            turn_remaining = 0 if ref.last_fired_turn == expected.turn else 1
            expected_repeats.append((definition.name+'(再演)',ref.remaining_uses,countdown,turn_remaining,period.modulus,ratio.threshold*100))
        else:
            named_semantics=False
    for _, ref in expired_start_scores:
        effect = bundle.kernel._scheduled_effect_spec(ref)
        expected_start_scores.append((schedule_status_name(bundle.kernel.cards, ref), 1, effect.scheduled_effects[0].amount))
    checks['named_card_schedules']=(named_semantics and Counter(timer_rows)==Counter(expected_timers)
        and Counter(start_score_rows)==Counter(expected_start_scores)
        and Counter(permanent_focus_rows)==Counter(expected_permanent_focus)
        and Counter(active_score_rows)==Counter(expected_active_scores)
        and Counter(typed_focus_rows)==Counter(expected_typed_focus)
        and Counter(counted_card_score_rows)==Counter(expected_counted_card_scores)
        and Counter(repeat_rows)==Counter(expected_repeats))
    # Native 1016: drink 20 has a complete uncounted permanent end-turn row.
    # Its one-point effect is verified against the fixed drink definition and
    # the committed schedule; absent counters never become invented uses.
    drink_effect = EffectSpec(EffectKind.SCHEDULE, delay_turns=0,
        schedule_phase=SchedulePhase.END_OF_TURN, schedule_limit=None,
        scheduled_effects=(EffectSpec(EffectKind.CONCENTRATION, amount=1),))
    drink_definition = DrinkDefinition('20', '厳選初星マキアート',
        effects=(drink_effect,), plan=PlanType.SENSE)
    drink_refs = [ref for ref in expected.scheduled_effects
                  if ref.source_kind is EffectSourceKind.DRINK]
    permanent_drink_refs = [ref for ref in drink_refs if ref.card_id == '20']
    finite_drink_refs = [ref for ref in drink_refs if ref.card_id == '14']
    drink_semantics = all(ref.source_id == 'drink:20'
        and ref.definition_effect_index == 0 and ref.trigger_phase is SchedulePhase.END_OF_TURN
        and ref.turns_until_fire == 0 and ref.remaining_uses is None and ref.remaining_turns is None
        and not ref.persistent and not ref.direct_on_trigger
        and bundle.kernel.drinks.get(ref.card_id) == drink_definition for ref in permanent_drink_refs)
    finite_drink_definition = _finite_drink_draw_definition()
    expected_finite_drinks = []
    for ref in finite_drink_refs:
        if (ref.source_id != 'drink:14' or ref.definition_effect_index != 3
                or ref.trigger_phase is not SchedulePhase.AFTER_START_OF_TURN
                or ref.turns_until_fire != 1 or ref.remaining_turns is not None
                or type(ref.remaining_uses) is not int or not 0 < ref.remaining_uses <= 5
                or ref.persistent or ref.direct_on_trigger or ref.duration_fresh or ref.effect_counter
                or (ref.last_fired_turn is not None and not 1 <= ref.last_fired_turn <= expected.turn)
                or bundle.kernel.drinks.get(ref.card_id) != finite_drink_definition):
            drink_semantics = False
            continue
        visible_turns = ref.remaining_uses + int(ref.last_fired_turn == expected.turn)
        if visible_turns > 5:
            drink_semantics = False
        expected_finite_drinks.append((finite_drink_definition.name, visible_turns))
    expired_drinks = verified_expired_start_drinks(bundle, expected)
    expected_finite_drinks.extend((finite_drink_definition.name, 1) for _ in expired_drinks)
    checks['named_drink_schedules'] = (drink_semantics
        and len(drink_refs) == len(permanent_drink_refs) + len(finite_drink_refs)
        and len(permanent_drink_refs) <= 1 and len(finite_drink_refs) + len(expired_drinks) <= 1
        and Counter(permanent_drink_rows) == Counter(
            (drink_definition.name, 1) for _ in permanent_drink_refs)
        and Counter(finite_drink_draw_rows) == Counter(expected_finite_drinks))
    checks['additional_status_semantics'] = (not extra and checks['concentration_addition']
        and all(checks.get('status_'+name, value == 0) for name, (value, _) in expected_scalar.items()))
    return dict(accepted=all(checks.values()), checks=checks, unresolved_status_rows=extra,
                observed_hand_instance_ids=tuple(ref.instance_id for ref in observed_hand) if all(checks.values()) else (),
                reason='hif_complete_settlement_verified' if all(checks.values()) else 'hif_settlement_observation_mismatch_or_unmapped_status')
