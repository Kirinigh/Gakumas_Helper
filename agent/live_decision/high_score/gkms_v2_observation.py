"""UIObs v2 projection for an ordinary, already bound GKH observation.

Contract: GKMSUINet 1fc50378753bd35fb239bebb77cb315c019915ff,
gkms_uinet/data/uiobs.v2.schema.json, encode2.py and policy.py. No model runs here.
"""
from copy import deepcopy
from dataclasses import replace

from .gkms_v2_effects import V2EffectState
from .hif_support_planning import uses_hand_support_for_strategy
from .gkms_native_shadow_adapter import ShadowMappingError, build_shadow_observation

_COMMON = ('exam', 'num', 'stance', 'status', 'ench', 'hand', 'deck', 'grave',
           'hold', 'lost', 'played', 'drinks', 'legal')
_EXAM = ('produce', 'step', 'plan', 'turn', 'limit', 'types', 'permil', 'crit', 'idol')
_NUM = ('score', 'stamina', 'max_stamina', 'block', 'playable', 'plays')


def project_v2(shadow, *, effects=None):
    """Reuse the verified roots and physical bindings from build_shadow_observation.

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
        exam = source.get('exam') if isinstance(source, dict) else None
        if (not isinstance(source, dict) or source.get('v') != 'uiobs.v1'
                or any(not isinstance(source.get(field), list) for field in _COMMON[2:])
                or not isinstance(exam, dict) or not exam.get('idol')
                or any(field not in exam for field in _EXAM)
                or any(exam[field] is None for field in _EXAM if field != 'step')
                or not isinstance(source.get('num'), dict)
                or any(source['num'].get(field) is None for field in _NUM)):
            raise ShadowMappingError('v2_common_fields_incomplete')
        if exam['produce'] == 'tower':
            if exam['step'] is not None or type(exam['plan']) is not int or exam['plan'] != 0 or exam['crit'] != [0, 0, 0]:
                raise ShadowMappingError('v2_tower_exam_fields_invalid')
        elif exam['step'] is None:
            raise ShadowMappingError('v2_common_fields_incomplete')
        if source['legal'] != [list(binding.token) for binding in shadow.bindings]:
            raise ShadowMappingError('v2_legal_list_differs_from_bindings')
        effects = V2EffectState.initial(state) if effects is None else effects
        if not effects.matches(state):
            raise ShadowMappingError('v2_effect_provenance_disagrees_with_state')
        reasons = tuple(reason for row, reason in (
            (effects.cost, 'v2_costs_source_and_layers_unresolved'),
            (effects.replay, 'v2_pcount_search_and_duration_unresolved'),
        ) if row is None)
        if reasons:
            return replace(shadow, status='unsupported', observation=None, bindings=(), reason_codes=reasons)
    except ShadowMappingError as error:
        return replace(shadow, status='unsupported', observation=None, bindings=(), reason_codes=(str(error),))
    observation = {field: deepcopy(source[field]) for field in _COMMON}
    observation.update(v='uiobs.v2', timers=[],
        costs=[list(effects.cost)] if effects.cost else [], pcount=[list(effects.replay)] if effects.replay else [])
    residuals = tuple(row for row in shadow.compression_evidence if row['local_field'] not in
        ('nullify_cost_cards', 'double_card_effects_remaining'))
    return replace(shadow, observation=observation, compression_evidence=residuals)


def build_v2_observation(request, kernel, context, history, *, effects=None):
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
    projected = project_v2(source, effects=effects)
    # Scored hand forms, playable flags and legal tokens come from the ordinary
    # root. The bind-time identity and last legality check use the caller's root.
    return replace(projected, request=request)
