"""Explicit battle choice, retained across a cultivation's handoffs."""

FAST_PROFILE = 'hif-gkms-fast-trial-v1'
UINET21_PROFILE = 'hif-gkms-uinet21-trial-v1'
STRONG_PROFILE = 'hif-strong-reference-trial-v1'
NETWORK_PROFILES = (FAST_PROFILE, UINET21_PROFILE)
FORMAL_PROFILES = (*NETWORK_PROFILES, STRONG_PROFILE)
BATTLE_PROFILES = {'fast': FAST_PROFILE, 'uinet-2.1': UINET21_PROFILE, 'strong-reference-trial': STRONG_PROFILE}


def selected_profile(selection):
    if selection is None:
        return None
    if not isinstance(selection, str) or selection not in BATTLE_PROFILES:
        raise ValueError('unknown explicit HIF battle policy')
    return BATTLE_PROFILES[selection]


def retained_profile(handoff, *, selection=None):
    """Missing old settings inherit fast; explicit choices cannot relabel a save."""
    requested = selected_profile(selection)
    profiles = []
    for _ in range(20):
        if not isinstance(handoff, dict):
            break
        if 'policy_profile' in handoff:
            profiles.append(handoff['policy_profile'])
        start = handoff.get('cultivation_start')
        if isinstance(start, dict) and 'policy_profile' in start:
            profiles.append(start['policy_profile'])
        handoff = handoff.get('nonbattle_context')
    else:
        raise ValueError('HIF battle policy history nesting is invalid')
    if any(not isinstance(value, str) or value not in BATTLE_PROFILES.values() for value in profiles):
        raise ValueError('HIF cultivation has contradictory or unknown battle policies')
    profiles = set(profiles)
    if len(profiles) > 1:
        raise ValueError('HIF cultivation has contradictory or unknown battle policies')
    retained = next(iter(profiles), FAST_PROFILE)
    if requested is not None and requested != retained:
        raise ValueError('HIF explicit battle policy differs from its retained cultivation')
    return retained


def preflight_strong_provider():
    """Construct the real frozen provider before allocation, without planning.

    Static catalog/model compatibility is checked here. The actual Live state,
    policy quality and every input remain the normal battle host's obligation.
    """
    import time
    from pathlib import Path
    from importlib import import_module

    prefix = 'agent.' if __package__.startswith('agent.') else ''
    modules = prefix + 'live_decision.high_score.'
    paths = import_module(prefix + 'utils.runtime_paths')
    catalogs = import_module(modules + 'catalog_v2')
    contracts = import_module(modules + 'contracts')
    reference = import_module(modules + 'hif_strong_reference')
    native = import_module(modules + 'hif_mcts_native')
    model_path = reference.verified_model_path()
    version, config = native.profile_configuration(STRONG_PROFILE)
    def data(name):
        return paths.resolve_runtime_path(Path('assets/data') / name, Path('data') / name)
    plan = contracts.PlanType.SENSE
    cards = catalogs.compile_skill_card_catalog(data('skill_cards.csv'), plan_type=plan)
    drinks = catalogs.compile_drink_catalog(data('p_drinks.csv'), plan_type=plan)
    items = catalogs.compile_item_catalog(data('p_items.csv'), plan_type=plan)
    kernel = import_module(modules + 'hif').HifSenseKernel(cards.cards, drinks.drinks, items.items)
    policy = native.HifMctsNativePolicy(kernel, run_id='strong-provider-preflight', revision=0,
        planning_seed=config['planning_seed_base'], deadline=time.perf_counter() + 420.,
        secondary_ready=False, retrieval_ready=True, profile=STRONG_PROFILE,
        information_boundary_execution=True)
    candidate, search = policy.candidate, policy.candidate.search
    settings = dict(simulations=search.simulations, min_root_samples=search.min_root_samples,
        window_seconds=candidate.window_seconds, soft_window_seconds=search.soft_timeout_ms / 1000,
        validation_samples=search.validation_samples, scenario_reuse=search.scenario_reuse)
    if (policy.profile != STRONG_PROFILE or policy.version != version
            or settings != {key: config['policy_kwargs'][key] for key in settings}
            or policy._calls != 0 or policy._windows != 0 or policy._pending is not None):
        raise ValueError('strong provider preflight differs from its native configuration')
    advice = candidate.baseline.baseline.advice
    def name(value):
        return type(value).__module__ + '.' + type(value).__qualname__
    return dict(status='static_provider_constructed_no_planning', profile=policy.profile,
        provider_class=name(policy), candidate_class=name(candidate), search_class=name(search),
        baseline_class=name(candidate.baseline), model_advice_class=name(advice),
        model_version=advice.model_version, model_weights_count=int(advice.weights.size),
        model_encoder_class=name(advice.encoder), model_path=str(model_path.resolve()),
        model_sha256=reference.REFERENCE_MODEL_SHA256, reference_commit=reference.REFERENCE_COMMIT,
        configuration_version=version, actual_search_settings=settings, policy_kwargs=config['policy_kwargs'],
        native_transaction_max_seconds=420, implicit_profile_fallback=False,
        independent_quality_passed=reference.INDEPENDENT_QUALITY_PASSED,
        planning_calls=0, input_count=0, ap_consumed=0)
