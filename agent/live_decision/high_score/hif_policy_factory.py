"""Build the selected policy for the formal HIF battle callback.

Consumers pass an observed opening and the existing lifecycle hooks. Policy
construction does not capture, click, or change the selected profile.
"""

import time

from route.battle_policy import FAST_PROFILE, STRONG_PROFILE, FORMAL_PROFILES, UINET21_PROFILE

from .live_policy import BoundHifOpeningPolicy


def _network(context, event, profile, _catalog_path):
    from .hif_fast_native import HifFastNativePolicy, observed_exam_context

    bundle = context.gkh_hif_opening_state
    candidate = getattr(context, 'gkh_hif_mcts_policy', None)
    if candidate is None:
        if event.get('event') != 'live_opening_state' or event.get('accepted') is not True:
            raise ValueError('fast_native_requires_complete_opening_or_history_cache')
        facts = observed_exam_context(bundle, context.gkh_hif_live_context,
            round_context=getattr(context, 'gkh_hif_round_context', None))
        candidate = HifFastNativePolicy(bundle, exam_context=facts, profile=profile,
                                        source=f'complete opening {bundle.run_id}:{bundle.revision}')
        context.gkh_hif_mcts_policy = candidate
    elif not isinstance(candidate, HifFastNativePolicy) or not candidate.matches_context(
            bundle.kernel, run_id=bundle.run_id, revision=bundle.revision, profile=profile):
        raise ValueError('fast_native_restored_profile_or_epoch_mismatch')
    candidate.before_secondary = getattr(context, 'gkh_hif_fast_secondary_start', None)
    candidate.after_secondary = getattr(context, 'gkh_hif_mcts_cache_sink', None)
    return candidate


def _search(context, event, profile, _catalog_path):
    from .hif_mcts_native import HifMctsNativePolicy, HifMctsSelectedIntentPolicy, profile_configuration

    _, configuration = profile_configuration(profile)
    bundle = context.gkh_hif_opening_state
    journal = getattr(context, 'gkh_hif_gray_selected_source', None)
    if journal is not None:
        candidate = HifMctsSelectedIntentPolicy(bundle, baseline=event, journal=journal, profile=profile)
        if profile == STRONG_PROFILE:
            verify_cache = getattr(context, 'gkh_hif_mcts_verify_selected_cache', None)
            if not callable(verify_cache):
                raise ValueError('strong_selected_intent_requires_complete_cache_verification')
            verify_cache(bundle, candidate)
        context.gkh_hif_mcts_policy = None
        return candidate

    candidate = getattr(context, 'gkh_hif_mcts_policy', None)
    if candidate is None or not candidate.matches_context(
            bundle.kernel, run_id=bundle.run_id, revision=bundle.revision, profile=profile):
        host_deadline = getattr(context, 'gkh_hif_mcts_session_deadline', None)
        deadline = host_deadline() if profile == STRONG_PROFILE and callable(host_deadline) else time.perf_counter() + 420.
        retrieval_ready = (getattr(context, 'gkh_hif_mcts_retrieval_ready', False) is True
            and callable(getattr(context, 'gkh_hif_choice_transaction', None))
            and callable(getattr(context, 'gkh_hif_choice_journal', None)))
        candidate = HifMctsNativePolicy(bundle.kernel, run_id=bundle.run_id, revision=bundle.revision,
            planning_seed=configuration['planning_seed_base'], profile=profile, deadline=deadline,
            secondary_ready=False, retrieval_ready=retrieval_ready,
            information_boundary_execution=getattr(context, 'gkh_hif_information_boundary_execution', False) is True)
        context.gkh_hif_mcts_policy = candidate
    if profile == STRONG_PROFILE:
        start_window = getattr(context, 'gkh_hif_mcts_window_start', None)
        if not callable(start_window):
            raise ValueError('strong_native_requires_persistent_window_budget')
        candidate.before_planning_window = start_window
    return candidate


def _frozen_hybrid(context, _event, _profile, catalog_path):
    from .hif_hybrid import load_frozen_hif_candidate

    return load_frozen_hif_candidate(context.gkh_hif_opening_state.kernel,
                                    catalog_path.parent / 'hif_policy_candidates' / 'ppo-27502-v1.json')


def _visible_hybrid(context, _event, _profile, catalog_path):
    from .hif_hybrid import load_visible_hif_candidate

    return load_visible_hif_candidate(context.gkh_hif_opening_state.kernel,
                                     catalog_path.parent / 'hif_policy_candidates' / 'ppo-27502-v1.json')


_POLICIES = {
    FAST_PROFILE: _network,
    UINET21_PROFILE: _network,
    STRONG_PROFILE: _search,
    'hif-mcts-v22-trial': _search,
    'hybrid-ppo27502-adaptive-v1': _frozen_hybrid,
    'hybrid-ppo27502-visible-d-trial-v1': _visible_hybrid,
}


def bind_hif_policy(context, event, profile, *, catalog_path):
    """Use one factory for all HIF policies, with the original Maa lifecycle."""
    if (getattr(context, 'gkh_hif_information_boundary_execution', False) is True
            and profile not in FORMAL_PROFILES):
        raise ValueError('information boundary execution requires the strong reference experiment')
    factory = _POLICIES.get(profile)
    if factory is None and profile != 'joint-search':
        raise ValueError('unsupported_hif_policy_profile')
    context.gkh_hif_complete_opening_event = event if event.get('event') == 'live_opening_state' else None
    candidate = factory(context, event, profile, catalog_path) if factory is not None else None
    return BoundHifOpeningPolicy(context.gkh_hif_opening_state, action_order=event['core']['fields'][5],
        decision_policy=candidate, hold_ready=callable(getattr(context, 'gkh_hif_hold_transaction', None)))
