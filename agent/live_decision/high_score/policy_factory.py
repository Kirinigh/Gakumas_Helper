"""Select battle policies from the observed state and the host's capabilities."""

import time
from dataclasses import dataclass
from collections.abc import Callable

from .contracts import BattleStrategy, DecisionRequest

if __package__.startswith('agent.'):
    from ...route.battle_policy import FAST_PROFILE, STRONG_PROFILE, FORMAL_PROFILES, UINET21_PROFILE
else:
    from route.battle_policy import FAST_PROFILE, STRONG_PROFILE, FORMAL_PROFILES, UINET21_PROFILE

def _hif_network(profile, *, context, event, catalog_path):
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


def _hif_search(profile, *, context, event, catalog_path):
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


def _frozen_hybrid(_profile, *, context, event, catalog_path):
    from .hif_hybrid import load_frozen_hif_candidate

    return load_frozen_hif_candidate(context.gkh_hif_opening_state.kernel,
                                    catalog_path.parent / 'hif_policy_candidates' / 'ppo-27502-v1.json')


def _visible_hybrid(_profile, *, context, event, catalog_path):
    from .hif_hybrid import load_visible_hif_candidate

    return load_visible_hif_candidate(context.gkh_hif_opening_state.kernel,
                                     catalog_path.parent / 'hif_policy_candidates' / 'ppo-27502-v1.json')


def _bind_hif_policy(factory, profile, *, context, event, catalog_path):
    """Bind the shared factory's policy to the HIF hand and receipt contract."""
    from .live_policy import BoundHifOpeningPolicy

    if (getattr(context, 'gkh_hif_information_boundary_execution', False) is True
            and profile not in FORMAL_PROFILES):
        raise ValueError('information boundary execution requires the strong reference experiment')
    context.gkh_hif_complete_opening_event = event if event.get('event') == 'live_opening_state' else None
    candidate = factory(profile, context=context, event=event, catalog_path=catalog_path)
    return BoundHifOpeningPolicy(context.gkh_hif_opening_state, action_order=event['core']['fields'][5],
        decision_policy=candidate, hold_ready=callable(getattr(context, 'gkh_hif_hold_transaction', None)))


def network_policy(profile, kernel, context, initial, *, source, planning_seed, deadline, model=None,
                   opening_hand_order=None, use_observed_hand_order=False, model_path=None,
                   model_sha256=None, exam_effects=None):
    from functools import partial

    from .. import hif_gkms_runtime as runtime
    from .gkms_v2_effects import CatalogV2RuleSources, effect_sources
    from .gkms_v2_observation import build_v2_observation
    from .gkms_continuous_policy import GkmsContinuousPolicy

    observation_builder = None
    v2_sources = None
    options = {}
    if getattr(context, 'use_host_legality', False):
        from ..idol_road_runtime import TowerChoicePolicy

        options['secondary_factory'] = TowerChoicePolicy
    if profile in ('uinet-2.1', UINET21_PROFILE):
        from ..idol_road_card_rules import TowerCardKernel

        if model is None:
            model = runtime.load_uinet21(model_path, model_sha256=(
                runtime.UINET21_SHA256 if model_sha256 is None else model_sha256),
                native_rules=isinstance(kernel, TowerCardKernel))
        observation_builder = (runtime.uinet21_observation_builder() if exam_effects is None
            else partial(build_v2_observation, exam_effects=exam_effects))
        tables = runtime.uinet21_master()
        v2_sources = (CatalogV2RuleSources(kernel, tables) if getattr(context, 'use_host_legality', False)
                      else effect_sources(tables))
    elif profile in ('uinet-1.1', FAST_PROFILE):
        if model is None:
            model = runtime.load_model()
    else:
        raise ValueError('decision_box_unknown_network_profile')
    return GkmsContinuousPolicy(kernel, model, context, initial, source=source,
        planning_seed=planning_seed, deadline=deadline, opening_hand_order=opening_hand_order,
        use_observed_hand_order=use_observed_hand_order, observation_builder=observation_builder,
        v2_sources=v2_sources, **options)


def _python_network(profile, *, kernel, planning_seed, deadline, context=None, state=None,
                    source=None, history=None, opening=None, **options):
    if context is None or state is None or not source:
        raise ValueError('decision_box_network_requires_exam_and_play_history')
    policy = network_policy(profile, kernel, context, state, source=source,
                            planning_seed=planning_seed, deadline=deadline, **options)
    if history is not None:
        policy.history = history
    return policy


def _python_strong(_profile, *, kernel, planning_seed, deadline, **_):
    from .gkms_continuous_policy import strong_policy

    return strong_policy(kernel, planning_seed=planning_seed, deadline=deadline)


def _python_rust(_profile, *, kernel, context=None, state=None, source=None, **_):
    from .tower_gnrpa import RustTowerPolicy

    if getattr(context, 'formation', None) is None or state is None or not source:
        raise ValueError('decision_box_rust_gnrpa_requires_tower_formation')
    return RustTowerPolicy(kernel, context, state, source=source)


@dataclass(frozen=True)
class NativePolicy:
    model: object
    observation_version: int | None
    open_search: Callable | None = None
    open_strategy: Callable[[DecisionRequest], BattleStrategy] | None = None


def _native_network(profile, *, model=None, exam=None):
    from ..hif_gkms_runtime import load_model, load_uinet21

    v2 = profile in ('uinet-2.1', UINET21_PROFILE)
    loader = load_uinet21 if v2 else load_model
    return NativePolicy(loader(native_rules=True) if model is None else model, 2 if v2 else 1)


def _native_rust(_profile, *, model=None, exam=None):
    from .tower_gnrpa import open_native_tower_search

    if exam is not None:
        raise ValueError('decision_box_unknown_native_profile')
    return NativePolicy(model, 1, open_native_tower_search)


def _screen_policy(profile, *, host, opening_drinks):
    from ..hiro_runtime import hiro_module
    from ..hiro_maa_exam import game_recommendation

    brain = hiro_module('brain.battle').BattleBrain(None)
    base_advisor = hiro_module('defaults.advisor').RuleLLMAdvisor
    recommend = game_recommendation(host, opening_drinks=opening_drinks) if profile == 'recommended' else None

    class ScreenAdvisor(base_advisor):
        def __init__(self):
            self.brain, self.profile, self.round_no = brain, None, None

        def decide(self, state, memory, previews, legal):
            if recommend is not None:
                return hiro_module('exec.exam').DecisionContract(legal[0], source='game-recommended')
            return super().decide(state, memory, previews, legal)

    return brain, ScreenAdvisor(), recommend


def _maa_recommended(_profile, *, card_info):
    def select(rows, *, wait_elapsed):
        suggested, unusable, playable, suggestion, best = card_info(rows)
        return (unusable > 0 and not (suggested or playable),
                suggestion if suggested else best if playable == 1 or wait_elapsed else None)

    return select


def _hif_joint_search(_profile, *, context, event, catalog_path):
    # BoundHifOpeningPolicy supplies the original joint-search planner.
    return None


_PROVIDERS = {
    'uinet-1.1': {'python': _python_network, 'native': _native_network},
    'uinet-2.1': {'python': _python_network, 'native': _native_network},
    'strong': {'python': _python_strong},
    'rust-gnrpa': {'python': _python_rust, 'native': _native_rust},
    'recommended': {'screen': _screen_policy, 'maa': _maa_recommended},
    'hiro-screen': {'screen': _screen_policy},
    FAST_PROFILE: {'python': _python_network, 'hif': _hif_network},
    UINET21_PROFILE: {'python': _python_network, 'native': _native_network, 'hif': _hif_network},
    STRONG_PROFILE: {'hif': _hif_search},
    'hif-mcts-v22-trial': {'hif': _hif_search},
    'hybrid-ppo27502-adaptive-v1': {'hif': _frozen_hybrid},
    'hybrid-ppo27502-visible-d-trial-v1': {'hif': _visible_hybrid},
    'joint-search': {'hif': _hif_joint_search},
}


def shared_policy_builder(profile):
    """A single `shared` registration can consume the common battle contract."""
    return _PROVIDERS.get(profile, {}).get('shared') if isinstance(profile, str) else None


def supports_native_policy(profile, *, cultivation=False):
    if not isinstance(profile, str):
        return False
    providers = _PROVIDERS.get(profile, {})
    return ('native' in providers or 'shared' in providers) and (not cultivation or profile != 'rust-gnrpa')


def uses_hif_transaction(profile):
    """Whether the selected finals policy requires the existing HIF receipt contract."""
    providers = _PROVIDERS.get(profile, {})
    return 'hif' in providers and not supports_native_policy(profile, cultivation=True)


def build_battle_policy(profile, *, backend='python', **options):
    """Construct one registered provider with the inputs its host supplies."""
    error = {'native': 'decision_box_unknown_native_profile', 'hif': 'unsupported_hif_policy_profile'}
    factory = _PROVIDERS.get(profile, {}).get(backend)
    if factory is None:
        shared = shared_policy_builder(profile)
        if shared is not None and backend == 'native':
            return NativePolicy(None, None, open_strategy=shared)
        if shared is not None and backend == 'python':
            return shared(options['opening'])
        raise ValueError(error.get(backend, 'decision_box_unknown_profile'))
    if backend == 'hif':
        return _bind_hif_policy(factory, profile, **options)
    return factory(profile, **options)
