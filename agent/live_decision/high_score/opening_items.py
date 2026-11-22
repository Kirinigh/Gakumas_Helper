"""A known stage-start item observed after opening, with no future trigger.

This is not general support for simulating the stage-start rule. The actual
complete opening inventory already contains any cards it generated.
"""

import math
from dataclasses import replace

from .kernel import ItemDefinition
from .contracts import PlanType
from ..status_panel import _normalize_status_text


def initialize_observed_opening(kernel, state, usage):
    """Keep the already observed first-turn draw/Genki item consumption."""
    from .kernel import EffectKind, EffectSpec, ConditionKind, EffectCondition
    from .contracts import SchedulePhase, EffectSourceKind

    key = (EffectSourceKind.ITEM, 'item:75', 0)
    if key not in usage:
        return kernel.initialize_observed_opening_effects(state,usage,plan_type=kernel.plan_type)
    condition=EffectCondition(ConditionKind.TURN_ATTRIBUTE,value='vocal')
    expected=EffectSpec(EffectKind.SCHEDULE,schedule_phase=SchedulePhase.AFTER_START_OF_TURN,
        schedule_limit=3,scheduled_effects=(EffectSpec(EffectKind.DRAW,amount=1,condition=condition),
            EffectSpec(EffectKind.FIXED_GENKI,amount=2,condition=condition)))
    if kernel.items['75'].persistent_effects != (expected,):
        raise ValueError('opening_after_draw_item_definition_changed')
    fired=state.turn_attributes[0]=='vocal'
    if usage[key] != 3-int(fired) or fired and state.genki < 2:
        raise ValueError('opening_after_draw_item_counter_or_resource_conflict')
    initial=kernel.initialize_observed_opening_effects(state,usage | {key:3},plan_type=kernel.plan_type)
    if not fired:
        return initial
    return replace(initial,scheduled_effects=tuple(
        replace(ref,remaining_uses=2,last_fired_turn=1) if
        (ref.source_kind,ref.source_id,ref.definition_effect_index)==key else ref
        for ref in initial.scheduled_effects))

STAGE_START_ITEM_ID = '428'
STAGE_START_ITEM_NAME = 'H.I.F応援棒（緑）'
STAGE_START_ITEM_EFFECTS = (
    'at:startOfStage { if:countCards[deck]<22 { '
    'addRandomBasicCardToDeckAtRandom(22-countCards[deck]) }}')
STAGE_START_ITEM_DESCRIPTION = (
    '試験開始時、山札にあるスキルカードが22枚になるように、'
    'ランダムな名前に「基本」を含むスキルカードを、山札のランダムな位置に生成')
STAGE_START_ITEM_SCOPE = 'observed_stage_start_only'


def _complete_box(region):
    return (isinstance(region, (tuple, list)) and len(region) == 4
            and all(type(value) is int for value in region)
            and min(region[:2]) >= 0 and min(region[2:]) > 0
            and region[0]+region[2] <= 720 and region[1]+region[3] <= 1280)


def _resolved_visual_evidence(record):
    absence = record.get('counter_absence_evidence') or {}
    visual = record.get('opening_resolved_visual') or {}
    if (not isinstance(absence, dict) or not isinstance(visual, dict)
            or absence.get('profile') != 'opaque_badge_upper_cap'
            or type(absence.get('minimum_channel')) is not int
            or not 250 <= absence['minimum_channel'] <= 255
            or type(absence.get('white_fraction')) not in (int, float)
            or absence['white_fraction'] != 1.0
            or visual.get('profile') != 'faded_name_and_start_trigger'):
        return False
    for key in ('name', 'trigger'):
        box = visual.get(key+'_box')
        ink_min = visual.get(key+'_ink_min')
        pixels = visual.get(key+'_ink_pixels')
        if (not _complete_box(box) or type(ink_min) is not int or not 180 <= ink_min < 230
                or type(pixels) is not int or not 0 < pixels <= box[2]*box[3]):
            return False
    name = visual['name_box']
    trigger = visual['trigger_box']
    return name[1]+name[3] <= trigger[1]


def bind_observed_stage_start_item(record, metadata_row=None, *, definition=None):
    """Validate the precise observed-only slice without inventing use counts."""
    expected = ItemDefinition(STAGE_START_ITEM_ID, STAGE_START_ITEM_NAME, (), PlanType.SENSE)
    if metadata_row is not None:
        if (metadata_row.get('id') != STAGE_START_ITEM_ID
                or metadata_row.get('name') != STAGE_START_ITEM_NAME
                or metadata_row.get('plan') != 'sense' or metadata_row.get('mode') != 'stage'
                or metadata_row.get('upgraded') != 'FALSE'
                or metadata_row.get('effects') != STAGE_START_ITEM_EFFECTS):
            raise ValueError('observed stage-start item differs from its fixed source rule')
    elif definition != expected:
        raise ValueError('observed stage-start item has no bound opening-only definition')
    confidence = record.get('confidence')
    region = record.get('counter_region')
    if (record.get('item_id') != STAGE_START_ITEM_ID
            or record.get('name') != _normalize_status_text(STAGE_START_ITEM_NAME)
            or record.get('upgraded') is not False
            or record.get('effect_scope') != STAGE_START_ITEM_SCOPE
            or any(key not in record or record[key] is not None
                   for key in ('remaining_uses', 'use_limit', 'definition_effect_index'))
            or type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not .95 <= confidence <= 1
            or _normalize_status_text(record.get('description', '')).removesuffix('。')
                != STAGE_START_ITEM_DESCRIPTION
            or not _complete_box(region)):
        raise ValueError('observed stage-start item has incomplete identity, text, or counter evidence')
    # Pixel evidence is supplied by the complete-row reader and retained in
    # its three original observations; a missing field is never an absence.
    if (record.get('counter_absent') is not True or record.get('opening_resolved') is not True
            or not _resolved_visual_evidence(record)):
        raise ValueError('observed stage-start item lacks explicit resolved/no-counter evidence')
    return expected
