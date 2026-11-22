"""Profile-specific computation recovery behind the same verified handoff."""
from types import SimpleNamespace

from route.battle_policy import STRONG_PROFILE as STRONG
from route.battle_policy import NETWORK_PROFILES


def phase(policy):
    if policy is None:
        return None
    if getattr(policy, 'profile', None) == STRONG:
        if policy._window_incomplete:
            return None
        return ('pending_intent' if policy._pending is not None else
            'observed_successor' if policy._expected_after is not None else None)
    return getattr(policy, '_phase', None)


def pending_action(policy, state):
    """Read the original intent without selecting or advancing its policy."""
    if phase(policy) != 'pending_intent' or policy._pending is None:
        raise ValueError('HIF selected recovery lacks its original pending action')
    if getattr(policy, 'profile', None) == STRONG:
        before, action = policy._pending['before'], policy._pending['action']
        expected = policy.kernel._canonical_probability_state(state)
    else:
        before, action = policy._pending
        expected = state
    if before != expected:
        raise ValueError('HIF selected recovery changed its original pending state')
    return action


def restore_policy(runtime, source, bundle, *, before=None, action=None, submitted=False,
                   round_context=None, selected=False, pending=False, baseline=None,
                   source_drink_slots=None, unconsumed_source=None, unsubmitted_plan=None,
                   verified_successors=None):
    profile = source['formal_params']['hif_policy_profile']
    if profile in NETWORK_PROFILES:
        from .native_fast_cache import restore_fast_cache

        return restore_fast_cache(runtime, source, bundle, before=before, action=action,
            submitted=submitted, accepted=True, round_context=round_context)
    if profile != STRONG:
        raise ValueError('HIF handoff computation profile is unsupported')
    from .native_search_cache import _cache_source, restore_search_cache, verify_selected_cache

    if unsubmitted_plan is not None:
        return restore_search_cache(runtime, source, bundle, accepted=True, submitted=False,
            profile=STRONG, retrieval_ready=True, information_boundary_execution=True,
            unsubmitted_plan=unsubmitted_plan)
    if unconsumed_source is not None:
        from .native_settlement_resume import decoded_dispatched_action

        action, dispatched = decoded_dispatched_action(unconsumed_source)
        refreshes = dispatched.get('authoritative_refresh_events', ())
        if not refreshes:
            raise ValueError('unconsumed strong intention has no original evaluated plan')
        original = refreshes[0]['proposal']['candidate_traces']['high_score']
        plan = original['trace']['planning']
        # restore_search_cache checks the complete tree and exact same-state
        # intention. No prior SELECT coordinates or timestamps are restored.
        return restore_search_cache(runtime, source, bundle, accepted=True, submitted=False,
            profile=STRONG, retrieval_ready=True, information_boundary_execution=True,
            unsubmitted_plan=plan)
    if pending:
        # Validate the full calculation now, but do not invent a settled state
        # or turn a dispatched action into a fresh unsubmitted intention.
        if action is None:
            raise ValueError('strong pending computation requires its verified dispatched action')
        report = verify_selected_cache(runtime, source, bundle, SimpleNamespace(profile=STRONG, action=action))
        return None, dict(verified_complete_pending_computation=report, deferred_until_observed_settlement=True,
            profile=STRONG, source_record=source['id'], input_authorized=False)
    if selected:
        from live_decision.high_score.hif_mcts_native import HifMctsSelectedIntentPolicy

        intent = HifMctsSelectedIntentPolicy(bundle, baseline=baseline,
            journal=source['gray_selection_journal'], profile=STRONG)
        verify_selected_cache(runtime, source, bundle, intent)
        return restore_search_cache(runtime, source, bundle, accepted=True, submitted=False,
            profile=STRONG, retrieval_ready=True, information_boundary_execution=True,
            unsubmitted_plan=intent.historical_decision['trace']['planning'])
    _, payload = _cache_source(runtime, source)
    submission = dict(before=before, action=action, submitted=submitted,
        source_drink_slots=source_drink_slots) if payload and payload['phase'] == 'pending_intent' else {}
    return restore_search_cache(runtime, source, bundle, accepted=True, profile=STRONG,
        retrieval_ready=True, information_boundary_execution=True,
        **(dict(verified_successors=verified_successors) if verified_successors is not None else {}), **submission)
