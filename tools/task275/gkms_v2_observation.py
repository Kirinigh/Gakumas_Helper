"""Offline UIObs v2 projection for an ordinary, already bound GKH observation.

Contract: GKMSUINet 1fc50378753bd35fb239bebb77cb315c019915ff,
gkms_uinet/data/uiobs.v2.schema.json, encode2.py and policy.py. No model runs here.
"""
from copy import deepcopy
from dataclasses import replace

from agent.live_decision.high_score.gkms_shadow_adapter import ShadowMappingError, build_shadow_observation
from agent.live_decision.high_score.hif_support_planning import uses_hand_support_for_strategy
from agent.live_decision.high_score.gkms_continuous_policy import verify_model_output

_COMMON = ('exam', 'num', 'stance', 'status', 'ench', 'hand', 'deck', 'grave',
           'hold', 'lost', 'played', 'drinks', 'legal')
_EXAM = ('produce', 'step', 'plan', 'turn', 'limit', 'types', 'permil', 'crit', 'idol')
_NUM = ('score', 'stamina', 'max_stamina', 'block', 'playable', 'plays')


def _timers(state, exam_effects, schedules):
    effects = {} if exam_effects is None else exam_effects
    rows = []
    for ref in state.scheduled_effects:
        if ref.remaining_turns == 0 or ref.remaining_uses == 0:
            continue
        key = (ref.source_kind.value, ref.card_id, ref.definition_effect_index, ref.persistent)
        mapping = schedules.get(key)
        if mapping is None:
            raise ShadowMappingError('v2_scheduled_effect_unmapped')
        _, mode, phase, delay, _, _, _, installer_id = mapping
        if mode != 'timer_not_in_protocol':
            continue
        # The ordinary source validated a one-shot start-of-turn timer. The
        # visible payload is its child effect, not the installer effect ID.
        installer = effects.get(installer_id) or {}
        child = installer.get('chainProduceExamEffectId')
        if (phase != 'TURN' or ref.trigger_phase.value != phase or not ref.direct_on_trigger
                or ref.remaining_uses != 1 or not 1 <= ref.turns_until_fire <= delay
                or installer.get('id') != installer_id
                or installer.get('effectType') != 'ProduceExamEffectType_ExamEffectTimer'
                or installer.get('effectValue1') != delay or installer.get('effectValue2') != 0
                or installer.get('effectCount') != 1 or not isinstance(child, str)
                or not child or effects.get(child, {}).get('id') != child):
            raise ShadowMappingError('v2_timer_requires_verified_installer_and_child')
        rows.append([ref.turns_until_fire, child, 0])
    return rows


def project_v2(shadow, *, exam_effects=None):
    """Reuse the verified roots and physical bindings from build_shadow_observation.

    The optional table is an already loaded, ID-indexed ProduceExamEffect table.
    Positive cost/play-count totals alone cannot identify the original target,
    layers or expiry. Such roots receive no mock model input.
    """
    state = shadow.request.state
    if ((shadow.status == 'supported' or any(reason.startswith('hand_support_tier_unmapped:816:')
                                            for reason in shadow.reason_codes))
            and any(not uses_hand_support_for_strategy(fact) for fact in state.hand_support)):
        # Partial correction of hand.u would retain legality computed using ++.
        return replace(shadow, status='unsupported', observation=None, bindings=(),
                       reason_codes=('v2_816_requires_original_plus_root',))
    if shadow.status != 'supported':
        return shadow
    source = shadow.observation
    try:
        if (not isinstance(source, dict) or source.get('v') != 'uiobs.v1'
                or any(not isinstance(source.get(field), list) for field in _COMMON[2:])
                or not isinstance(source.get('exam'), dict) or not source['exam'].get('idol')
                or any(source['exam'].get(field) is None for field in _EXAM)
                or not isinstance(source.get('num'), dict)
                or any(source['num'].get(field) is None for field in _NUM)):
            raise ShadowMappingError('v2_common_fields_incomplete')
        if source['legal'] != [list(binding.token) for binding in shadow.bindings]:
            raise ShadowMappingError('v2_legal_list_differs_from_bindings')
        reasons = tuple(reason for count, reason in (
            (state.nullify_cost_cards, 'v2_costs_source_and_layers_unresolved'),
            (state.double_card_effects_remaining, 'v2_pcount_search_and_duration_unresolved'),
        ) if type(count) is not int or count != 0)
        if reasons:
            return replace(shadow, status='unsupported', observation=None, bindings=(), reason_codes=reasons)
        if shadow.mappings is None:
            raise ShadowMappingError('verified_shadow_mappings_required')
        timers = _timers(state, exam_effects, shadow.mappings.schedules)
    except ShadowMappingError as error:
        return replace(shadow, status='unsupported', observation=None, bindings=(), reason_codes=(str(error),))
    observation = {field: deepcopy(source[field]) for field in _COMMON}
    observation.update(v='uiobs.v2', timers=timers, costs=[], pcount=[])
    residuals = tuple(row for row in shadow.compression_evidence if row['local_field'] != 'scheduled_effects.timer')
    return replace(shadow, observation=observation, compression_evidence=residuals)


def build_v2_observation(request, kernel, context, history, *, exam_effects=None):
    """Rebuild from the ordinary 816+ when a historical input recorded live 816++."""
    state = request.state
    kept = tuple(fact for fact in state.hand_support if uses_hand_support_for_strategy(fact))
    if kept == state.hand_support or history is None or history.state != state:
        source = build_shadow_observation(request, kernel, context, history)
    else:
        request.validate()
        ordinary = replace(state, hand_support=kept)
        source = build_shadow_observation(replace(request, state=ordinary), kernel, context,
                                          replace(history, state=ordinary))
    projected = project_v2(source, exam_effects=exam_effects)
    # Scored hand forms, playable flags and legal tokens come from the ordinary
    # root. The bind-time identity and last legality check use the caller's root.
    return replace(projected, request=request)


def bind_mock_output(shadow, output, *, request, kernel):
    """Resolve the fake response's argmax to its original instance and physical slot."""
    if shadow.status != 'supported' or (shadow.observation or {}).get('v') != 'uiobs.v2':
        raise ShadowMappingError('v2_mock_requires_projected_ordinary_state')
    if shadow.observation['legal'] != [list(binding.token) for binding in shadow.bindings]:
        raise ShadowMappingError('v2_legal_list_differs_from_bindings')
    token = verify_model_output(output, shadow.observation)
    return shadow.bind_action(token, request=request, kernel=kernel)
