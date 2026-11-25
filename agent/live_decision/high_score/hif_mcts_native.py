"""Explicit native intent adapter for isolated research profiles.

Search never authorizes input. Only an accepted, actually submitted native
transaction may confirm an observed successor. Tree caches are explicit
computation sidecars; aggregate planning logs cannot reconstruct a tree.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from dataclasses import asdict, replace, dataclass

from .planner import RobustHighScorePlanner
from .hif_mcts import action_key
from .contracts import PlanType, ActionKind, GameAction, DecisionKind, DecisionTrace, HifActionIntent
from .hif_hybrid import load_visible_hif_candidate
from .hif_choice_roots import _declared_choices
from .hif_mcts_candidate import CONFIG, HifMctsCandidatePolicy
from .hif_strong_reference import PROFILE as STRONG_PROFILE

PROFILE = 'hif-mcts-v22-trial'
_VERSION = 'hif-public-state-mcts-trial-v22'
PROFILES = (PROFILE, STRONG_PROFILE)
_INFORMATION_EXECUTION_VERSION = 'hif-native-information-boundary-v1'


def profile_configuration(profile):
    """Keep v22 as default; stronger research settings require an explicit ID."""
    if profile == PROFILE:
        if CONFIG['version'] != _VERSION:
            raise ValueError('invalid_native_mcts_context_or_version')
        return _VERSION, copy.deepcopy(CONFIG)
    if profile == STRONG_PROFILE:
        from .hif_strong_reference import VERSION, configuration

        return VERSION, configuration()
    raise ValueError('unsupported_native_mcts_profile')


def _definitions(kernel):
    # Complete immutable definitions, not a new hash or credential protocol.
    return (kernel.ruleset_id, *(dict(getattr(kernel, name))
        for name in ('cards', 'drinks', 'items', 'memory_abilities')))


def _has_post_card_end(action, plan):
    return action.kind is ActionKind.PLAY_CARD and any(
        isinstance(value, dict) and value.get('kind') == ActionKind.END_TURN
        for value in (plan.get('selected_continuation') or ()))


@dataclass(frozen=True)
class HifMctsNativeTrace(DecisionTrace):
    policy_profile: str = PROFILE
    run_id: str = ''
    revision: int = 0
    tree_context: str = 'new_native_instance_no_restored_tree'
    planning_seed: int = 0
    planning_window_index: int = 0
    planning: dict | None = None


@dataclass(frozen=True)
class HifMctsSelectedIntentTrace(DecisionTrace):
    policy_profile: str = PROFILE
    run_id: str = ''
    revision: int = 0
    source_record: str = ''
    new_search_performed: bool = False
    tree_context: str = 'recovered_selected_intent_no_restored_tree'
    historical_decision: dict | None = None


@dataclass(frozen=True)
class HifMctsNativeDecision(HifActionIntent):
    """Sampled means are retained in planning, never invented exact quantiles."""
    trace: HifMctsNativeTrace | HifMctsSelectedIntentTrace
    protocol_version = 'task275-hif-mcts-native-intent-v1'


class HifMctsSelectedIntentPolicy:
    """Rebind one verified, unsubmitted selection; never reconstruct its tree."""
    def __init__(self, bundle, *, baseline, journal, profile=PROFILE):
        self.profile = profile
        self.version, _ = profile_configuration(profile)
        refresh = journal.get('refresh_a', {})
        original = refresh.get('proposal', {}).get('candidate_traces', {}).get('high_score', {})
        trace = original.get('trace', {})
        raw = original.get('action') or {}
        try:
            action = GameAction(**{**raw, 'kind': ActionKind(raw['kind'])})
            action.validate()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('native_mcts_selected_original_action_invalid') from error
        dispatches = [row for row in journal.get('selection_events', ())
                      if row.get('phase') == 'execution_selection']
        if (baseline.get('accepted') is not True or not baseline.get('checks')
                or not all(value is True for value in baseline['checks'].values())
                or (baseline.get('run_id'), baseline.get('revision')) != (bundle.run_id, bundle.revision)
                or journal.get('selection_ready') is not True
                or journal.get('selection_count') != 1 or journal.get('business_submit_count') != 0
                or not isinstance(journal.get('source_record'), str) or not journal['source_record']
                or journal.get('stage') != 'selection_dispatched'
                or len(dispatches) != 1 or dispatches[0].get('accepted') is not True
                or dispatches[0].get('click_dispatched') is not True
                or dispatches[0].get('attempted_slot') != action.source_slot
                or original.get('has_action_intent') is not True
                or trace.get('policy_profile') != self.profile
                or trace.get('planner_version') != self.version
                or (trace.get('run_id'), trace.get('revision')) != (bundle.run_id, bundle.revision)
                or (trace.get('planning') or {}).get('accepted') is not True
                or action.kind is not ActionKind.PLAY_CARD
                or type(action.source_slot) is not int or not 0 <= action.source_slot < len(bundle.state.deck.hand)
                or bundle.state.deck.hand[action.source_slot].instance_id != action.card_instance_id):
            raise ValueError('native_mcts_selected_original_intent_not_bound')
        if journal.get('ordinary_safe_preview') is True:
            if (journal.get('provenance') != 'completed_native_selection_report'
                    or refresh.get('accepted') is not True
                    or refresh.get('safe_decision', {}).get('status') != 'ALLOW'):
                raise ValueError('native_mcts_selected_original_intent_not_bound')
        else:
            from ..hif_gray_selection import verify_recorded_gray_selection_reservation

            boxes = refresh.get('candidate_boxes', ())
            fields = baseline.get('core', {}).get('fields', ())
            if (journal.get('provenance') != 'verified_native_gray_selection_reservation'
                    or journal.get('stages') != ['selection_reserved', 'selection_dispatched']
                    or len(fields) != 6 or len(boxes) != 3
                    or any(len(row) != len(bundle.state.deck.hand) for row in boxes)):
                raise ValueError('native_mcts_selected_original_intent_not_bound')
            state = bundle.state
            generation = dict(turn=state.turn, turns_remaining=state.turns_remaining,
                stamina=state.stamina, shield=state.genki, score=state.score,
                score_multiplier=state.score_multiplier, hand_count=len(state.deck.hand),
                plan='sense', action_order=list(fields[5]))
            if refresh.get('generation') != generation:
                raise ValueError('native_mcts_selected_original_intent_not_bound')
            verify_recorded_gray_selection_reservation(bundle, journal,
                slot=action.source_slot, box=boxes[-1][action.source_slot], generation=generation)
        self.kernel, self.state = bundle.kernel, bundle.state
        self.run_id, self.revision = bundle.run_id, bundle.revision
        self.drink_slots = bundle.drink_slots or ()
        self.definitions = _definitions(bundle.kernel)
        self.action = action
        self.source_record = journal['source_record']
        self.historical_decision = copy.deepcopy(original)

    def decide(self, request):
        started = time.perf_counter()
        action = None
        reason = 'native_mcts_recovered_selected_intent'
        try:
            request.validate()
            slots = tuple((ref.instance_id, slot) for slot, ref in enumerate(self.state.deck.hand))
            if (not request.capabilities.all_ready or request.plan_type is not PlanType.SENSE
                    or request.decision_kind is not DecisionKind.LIVE
                    or request.ruleset_id != self.kernel.ruleset_id
                    or request.state != self.state or request.drink_slots != self.drink_slots
                    or dict(request.card_slots) != dict(slots)
                    or _definitions(self.kernel) != self.definitions):
                raise ValueError('native_mcts_selected_fresh_binding_changed')
            legal = self.kernel.legal_actions(request.state, plan_type=request.plan_type,
                                              decision_kind=request.decision_kind)
            rebound = [RobustHighScorePlanner._attach_source_slot(value, request) for value in legal]
            if self.action not in rebound:
                raise ValueError('native_mcts_selected_original_action_not_legal')
            action = self.action
        except (ValueError, RuntimeError) as error:
            reason = str(error)
        trace = HifMctsSelectedIntentTrace(planner=self.profile, planner_version=self.version,
            policy_profile=self.profile,
            ruleset_id=self.kernel.ruleset_id, completed_depth=0, expanded_nodes=0,
            elapsed_ms=(time.perf_counter()-started)*1000, reason_codes=(reason,),
            value_scope='RECOVERED_SELECTED_INTENT', run_id=self.run_id, revision=self.revision,
            source_record=self.source_record, historical_decision=copy.deepcopy(self.historical_decision))
        return HifMctsNativeDecision(action, reason, trace)


class HifMctsNativePolicy:
    def __init__(self, kernel, *, run_id, revision, planning_seed, deadline,
                 baseline=None, secondary_ready=False, retrieval_ready=False, profile=PROFILE,
                 information_boundary_execution=False):
        self.profile = profile
        self.version, configuration = profile_configuration(profile)
        if (not isinstance(run_id, str) or not run_id or type(revision) is not int or revision < 0
                or type(secondary_ready) is not bool or type(retrieval_ready) is not bool
                or type(information_boundary_execution) is not bool):
            raise ValueError('invalid_native_mcts_context_or_version')
        if baseline is None:
            if profile == STRONG_PROFILE:
                from .hif_strong_reference import load_baseline

                baseline = load_baseline(kernel)
            else:
                model = Path(__file__).resolve().parents[3] / 'assets/data/hif_policy_candidates/ppo-27502-v1.json'
                baseline = load_visible_hif_candidate(kernel, model)
        self.kernel, self.run_id, self.revision = kernel, run_id, revision
        self.planning_seed = planning_seed
        self.definitions = _definitions(kernel)
        self.secondary_ready = secondary_ready
        self.retrieval_ready = retrieval_ready
        self.information_boundary_execution = information_boundary_execution
        self._information_boundary = None
        self._exact_segment = []
        self.candidate = HifMctsCandidatePolicy(kernel, baseline, planning_seed=planning_seed,
            deadline=deadline if profile == STRONG_PROFILE else min(deadline, time.perf_counter()+420.),
            **configuration['policy_kwargs'])
        self.candidate.compare_drink_order = profile == STRONG_PROFILE and information_boundary_execution
        self._pending = None
        self._expected_after = None
        self._calls = 0
        self._windows = 0
        self._tree_context = 'new_native_instance_no_restored_tree'
        self._cache_request = None
        self._retrieval_choice = None
        self._logical_drink_slots = None
        self._drink_binding = None
        self.before_planning_window = None
        self._window_incomplete = False
        self._restored_host_calls = None

    def _logical_request(self, request, *, drink_slots=None):
        if self.information_boundary_execution:
            # Calculation order is independent of the newly observed slots.
            request = replace(request, state=self.kernel._canonical_probability_state(request.state))
        if self.profile != STRONG_PROFILE:
            return request
        from .hif_drink_slot_binding import HifDrinkSlotBinding

        logical = self._logical_drink_slots if drink_slots is None else tuple(drink_slots)
        if logical is None:
            logical = request.drink_slots
        binding = HifDrinkSlotBinding.create(request.state, logical, request.drink_slots)
        self._logical_drink_slots, self._drink_binding = binding.logical_slots, binding
        return binding.logical_request(request)

    def _logical_retrieval_projection(self, projected, *, drink_slots=None):
        if self.profile != STRONG_PROFILE:
            return projected
        from .hif_drink_slot_binding import HifDrinkSlotBinding

        slots = self._logical_drink_slots if drink_slots is None else tuple(drink_slots)
        binding = HifDrinkSlotBinding.create(projected.source_bundle.state, slots,
                                             projected.source_bundle.drink_slots)
        action = binding.logical_action(projected.action)
        source = replace(projected.source_bundle, drink_slots=binding.logical_slots)
        remaining = tuple(pair for pair in binding.logical_slots
                          if pair != (action.source_slot, action.drink_id))
        return replace(projected, source_bundle=source, action=action, drink_slots=remaining)

    def bind_retrieval_choice(self, projected, request, *, drink_submitted):
        from .hif_mcts_native_choice import bind_retrieval_choice

        projected = self._logical_retrieval_projection(projected)
        request = self._logical_request(request, drink_slots=projected.drink_slots)
        return bind_retrieval_choice(self, projected, request, drink_submitted=drink_submitted)

    def confirm_retrieval_committed(self, **evidence):
        from .hif_mcts_native_choice import confirm_retrieval_committed

        observed_action = evidence['action']
        if self.profile == STRONG_PROFILE and self._pending is not None:
            evidence = dict(evidence, action=self._pending['drink_binding'].logical_action(observed_action))
        boundary = (self._classify_committed_boundary(evidence['before'], evidence['action'],
                    evidence['after']) if self.information_boundary_execution else None)
        result = confirm_retrieval_committed(self, **evidence)
        self._information_boundary = boundary
        result['original_action'] = asdict(observed_action)
        return result

    def restore_retrieval_cache(self, payload, *, allow_information_boundary_upgrade=False, **evidence):
        from .hif_mcts_native_choice import _restore_retrieval_cache

        try:
            staged, payload, boundary, segment = self._prepare_cache_restore(payload,
                allow_information_boundary_upgrade=allow_information_boundary_upgrade)
            result = _restore_retrieval_cache(staged, payload, **evidence)
            staged._information_boundary = boundary
            self._commit_cache_restore(staged, segment)
            return result
        except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
            raise ValueError('native_mcts_retrieval_cache_malformed') from error

    def _candidate_cache_anchor(self, payload):
        if self.profile == STRONG_PROFILE:
            if (not isinstance(payload, dict) or set(payload) != {
                    'version', 'computation', 'choice', 'frozen_policy_continuation', 'actual_tail'}
                    or payload['version'] != 'hif-complete-candidate-cache-v1'):
                raise ValueError('native_mcts_complete_policy_cache_required')
            return payload['computation']['search']['tree']['anchor']
        return payload['search']['anchor']

    def _restore_candidate_cache(self, payload, anchor):
        if self.profile == STRONG_PROFILE:
            return self.candidate.restore_search_cache(payload, anchor, complete_policy=True)
        return self.candidate.restore_search_cache(payload, anchor)

    def _new_candidate(self, *, retain_drinks):
        kernel, baseline = self.kernel, self.candidate.baseline
        if retain_drinks:
            from .hif_round1_card_only import reserve_drinks_for_planning
            from .hif_strong_reference import load_baseline

            kernel = reserve_drinks_for_planning(kernel)
            baseline = load_baseline(kernel)
        candidate = HifMctsCandidatePolicy(kernel, baseline, deadline=self.candidate.deadline,
            **self.candidate._cache_configuration)
        candidate.compare_drink_order = self.profile == STRONG_PROFILE and self.information_boundary_execution
        candidate.search._order_deadline = self.candidate.search._order_deadline
        return candidate

    def candidate_for_cache(self, payload):
        """Recreate the same planning rules for both validation and full restoration."""
        retain_drinks = False
        if self.profile == STRONG_PROFILE:
            from .hif_mcts_cache import exact_state
            from .hif_round1_card_only import LIVE_RUN_ID, retains_round1_drinks

            if self.run_id == LIVE_RUN_ID:
                tree = payload['candidate']['computation']['search']['tree']
                retain_drinks = tree['definitions']['drinks'].get('4', {}).get('reserved_for_later') is True
                if retain_drinks and not retains_round1_drinks(self.run_id, self.revision,
                        exact_state(tree['anchor']['state'])):
                    raise ValueError('native_mcts_reserved_drink_cache_round_changed')
        return self._new_candidate(retain_drinks=retain_drinks)

    def export_search_cache(self):
        """Export once outside input freshness windows, not into every trace."""
        from .hif_mcts_cache import VERSION, json_value

        if self.profile == STRONG_PROFILE and self._window_incomplete:
            raise ValueError('native_mcts_window_incomplete')
        if (self._cache_request is None
                or self.profile != STRONG_PROFILE and self.candidate.search._namespace is None):
            return None
        pending = self._pending
        if pending is None and self._expected_after is None:
            return None
        action = None if pending is None else pending['action']
        if action is not None and action.kind is not ActionKind.USE_DRINK:
            action = replace(action, source_slot=None)
        payload = dict(version=VERSION, profile=self.profile,
            run_id=self.run_id, revision=self.revision,
            phase='pending_intent' if pending is not None else 'observed_successor',
            intent=None if action is None else asdict(action),
            planning_windows=self._windows, decision_calls=self._calls,
            candidate=self.candidate.export_search_cache(self._cache_request,
                **({'complete_policy': True} if self.profile == STRONG_PROFILE else {})))
        if self.profile == STRONG_PROFILE and pending is not None:
            payload['pending_plan'] = copy.deepcopy(pending['plan'])
        if self.information_boundary_execution:
            payload['information_boundary_execution'] = dict(
                version=_INFORMATION_EXECUTION_VERSION, continuation=self._information_boundary,
                exact_segment=self._exact_segment)
        return json_value(payload)

    def _classify_committed_boundary(self, before, action, after, *, verified_successors=None,
                                     observed_temporary_support=False):
        from .hif_mcts_cache import json_value
        from .hif_information_boundary import classify_information_boundary

        result = classify_information_boundary(self.kernel, before, action, plan_type=PlanType.SENSE)
        actual = self.kernel._canonical_probability_state(after)
        if (result.kind == 'deterministic_same_turn'
                and (self.kernel._canonical_probability_state(result.successor) != actual
                     or observed_temporary_support and verified_successors is not None)):
            if verified_successors is not None:
                outcomes = (verified_successors if isinstance(verified_successors, tuple)
                            else (verified_successors,))
                for state in outcomes:
                    self.kernel._validate_registered_state(state, plan_type=PlanType.SENSE)
                if not any(self.kernel._canonical_probability_state(state) == actual for state in outcomes):
                    raise ValueError('native_mcts_verified_effect_successor_changed')
                # The business observer has already checked the current-play
                # projection. Retain computation and counters, but invalidate
                # the ordinary continuation after these newly revealed effects.
                return json_value(dict(kind='synchronize', reason='new_temporary_support',
                    before=asdict(before), action=asdict(action), state=asdict(actual),
                    verified_successors=[asdict(state) for state in outcomes],
                    source_plan_index=max(0, self.candidate._plan_offset + len(self.candidate.plans) - 1)))
            from types import SimpleNamespace

            from .skill_support import project_supported_card_once
            from .support_master import genki_only_support

            matches = []
            if action.kind is ActionKind.PLAY_CARD:
                refs = [ref for ref in before.deck.hand if ref.instance_id == action.card_instance_id]
                if len(refs) == 1:
                    physical = replace(action, source_slot=before.deck.hand.index(refs[0]))
                    ordinary = self.kernel.card_definition(before, refs[0])
                    for tier in (2, 3):
                        try:
                            supported = genki_only_support(ordinary, tier)
                            _, outcomes = project_supported_card_once(
                                SimpleNamespace(state=before, kernel=self.kernel), physical,
                                effective_definition=supported, settle_automatic_turn=False)
                        except (ValueError, KeyError):
                            continue
                        if (len(outcomes) == 1
                                and self.kernel._canonical_probability_state(outcomes[0]) == actual):
                            matches.append(tier)
            if len(matches) != 1:
                raise ValueError('native_mcts_information_boundary_successor_changed')
            # Business observation was independently accepted by the caller.
            # Its complete successor matches one pinned current-play tier;
            # preserve that reality, but never continue the old ordinary tree
            # as if no new information had been revealed.
            result = classify_information_boundary(self.kernel, before, action,
                plan_type=PlanType.SENSE, new_temporary_support=True)
        return json_value(dict(kind=result.kind, reason=result.reason,
            before=asdict(before), action=asdict(action), state=asdict(actual),
            source_plan_index=max(0, self.candidate._plan_offset + len(self.candidate.plans) - 1)))

    def _decode_information_cache(self, payload, *, allow_upgrade):
        """Decode computation mode only; no execution or observation permit."""
        from .hif_mcts_cache import json_value, exact_state, action_from_json, request_from_public

        if type(allow_upgrade) is not bool:
            raise ValueError('native_mcts_information_boundary_upgrade_flag_invalid')
        if payload is None:
            return payload, None, []
        payload = json_value(payload)
        has_metadata = 'information_boundary_execution' in payload
        metadata = payload.pop('information_boundary_execution', None)
        if not has_metadata:
            if self.information_boundary_execution and not allow_upgrade:
                raise ValueError('native_mcts_information_boundary_upgrade_requires_explicit_flag')
            if self.information_boundary_execution and payload.get('phase') == 'observed_successor':
                anchor = request_from_public(self._candidate_cache_anchor(payload['candidate']), self.kernel)
                return payload, dict(kind='requires_replan', reason='legacy_successor_has_no_boundary_proof',
                                     state=json_value(asdict(anchor.state))), []
            return payload, None, []
        if (not self.information_boundary_execution or not isinstance(metadata, dict)
                or set(metadata) not in ({'version', 'continuation'}, {'version', 'continuation', 'exact_segment'})
                or metadata['version'] != _INFORMATION_EXECUTION_VERSION):
            raise ValueError('native_mcts_information_boundary_cache_mode_changed')
        from .hif_executable_tail import validate
        segment = validate(self.kernel, metadata.get('exact_segment', []))
        saved = metadata['continuation']
        if saved is None:
            if payload.get('phase') != 'pending_intent':
                raise ValueError('native_mcts_information_boundary_cache_proof_missing')
            return payload, None, segment
        if not isinstance(saved, dict):
            raise ValueError('native_mcts_information_boundary_cache_malformed')
        anchor = request_from_public(self._candidate_cache_anchor(payload['candidate']), self.kernel)
        if exact_state(saved['state']) != anchor.state:
            raise ValueError('native_mcts_information_boundary_cache_state_changed')
        if saved.get('kind') == 'requires_replan':
            if (set(saved) != {'kind', 'reason', 'state'}
                    or saved['reason'] != 'legacy_successor_has_no_boundary_proof'):
                raise ValueError('native_mcts_information_boundary_cache_malformed')
            return payload, saved, segment
        projected = saved.get('verified_successors')
        expected_keys = {'kind', 'reason', 'before', 'action', 'state', 'source_plan_index'}
        if projected is not None:
            if not isinstance(projected, list) or not projected:
                raise ValueError('native_mcts_information_boundary_cache_malformed')
            expected_keys.add('verified_successors')
        if (set(saved) != expected_keys
                or type(saved['source_plan_index']) is not int or saved['source_plan_index'] < 0):
            raise ValueError('native_mcts_information_boundary_cache_malformed')
        action = action_from_json(saved['action'])
        # The 0041 observed 818++ had a different printed cost but reached the
        # same scalar state as ordinary 818. Only its verified experiment needs
        # to replan at that otherwise deterministic numerical boundary.
        corrected_818 = (payload['run_id'] == 'hif-recovered-e5fc94bdd66a4338a94b09b86cb2b977'
            and payload['revision'] == 1 and action.card_instance_id == payload['run_id'] + ':手札:2'
            and action.source_slot == 0 and projected is not None and saved['reason'] == 'new_temporary_support')
        checked = self._classify_committed_boundary(exact_state(saved['before']), action, anchor.state,
            verified_successors=None if projected is None else tuple(exact_state(state) for state in projected),
            observed_temporary_support=corrected_818)
        if any(saved[key] != checked[key] for key in ('kind', 'reason', 'state')):
            raise ValueError('native_mcts_information_boundary_cache_proof_changed')
        return payload, saved, segment

    def _prepare_cache_restore(self, payload, *, allow_information_boundary_upgrade):
        """Version gate and decode into isolated owned computation only."""
        from copy import copy

        from .hif_mcts_cache import VERSION, require_candidate_tree_versions

        if type(allow_information_boundary_upgrade) is not bool:
            raise ValueError('native_mcts_information_boundary_upgrade_flag_invalid')
        if payload is not None:
            if not isinstance(payload, dict) or payload.get('version') != VERSION:
                raise ValueError('native_mcts_cache_source_not_verified')
            require_candidate_tree_versions(payload['candidate'], self.candidate.search,
                complete_policy=self.profile == STRONG_PROFILE)
            candidate = self.candidate
            search = candidate.search
            if (self._pending is not None or self._expected_after is not None
                    or self._cache_request is not None or self._windows or self._calls
                    or self._exact_segment or self._information_boundary is not None
                    or self._retrieval_choice is not None or self._window_incomplete
                    or search._tree or search._namespace is not None or search._expected_state is not None
                    or candidate.plans or candidate._plan_offset or candidate._session is not None
                    or candidate._tail_session is not None or candidate._choice_continuation is not None
                    or candidate._frozen_policy_continuation is not None or candidate._actual_tail_link is not None):
                raise ValueError('native_mcts_cache_requires_new_instance')
        # Restore on a fresh candidate: its controllers and callbacks retain
        # their own owner identity. No receiver field changes on rejection.
        staged = copy(self)
        if payload is not None:
            staged.candidate = self.candidate_for_cache(payload)
        payload, boundary, segment = staged._decode_information_cache(payload,
            allow_upgrade=allow_information_boundary_upgrade)
        return staged, payload, boundary, segment

    def _commit_cache_restore(self, staged, segment):
        """All validation is complete; publish only the staged receiver."""
        staged._exact_segment = segment
        staged._restored_host_calls = staged._calls
        self.__dict__.update(staged.__dict__)

    def bind_restored_host_session(self, deadline):
        """Start one runtime session after recovery, within the current parent.

        No saved clock or unfinished search window is renewed. All trees,
        random streams, plans and spent work remain the restored objects.
        """
        import math

        if self._restored_host_calls is None:
            return None
        now = time.perf_counter()
        if (self._window_incomplete or self._calls != self._restored_host_calls
                or deadline is None and self.profile != STRONG_PROFILE
                or deadline is not None and (type(deadline) not in (int, float)
                    or not math.isfinite(deadline) or deadline <= now)):
            raise ValueError('native_mcts_restored_session_cannot_renew_started_work')
        self.candidate.deadline = deadline if self.profile == STRONG_PROFILE else min(deadline, now + 420.)
        self._restored_host_calls = None
        return dict(started_at=now, deadline=self.candidate.deadline,
            parent_deadline=deadline, decision_calls=self._calls, planning_windows=self._windows)

    def restore_search_cache(self, payload, *, request, source_run_id, source_revision,
                             before=None, action=None, submitted=False, accepted=False,
                             source_drink_slots=None, allow_information_boundary_upgrade=False,
                             unsubmitted_plan=None, verified_successors=None):
        """Use the existing recovery caller's verified business-state boundary.

        A history rebound may replace the observation epoch. The caller must
        independently verify that lineage and pass its original source epoch.
        Pending/selected intent alone never confirms a predicted successor.
        An explicitly supplied completed unsubmitted plan can retain its
        calculation; no credential, old deadline or input permission returns.
        """
        try:
            staged, payload, boundary, segment = self._prepare_cache_restore(payload,
                allow_information_boundary_upgrade=allow_information_boundary_upgrade)
            result = staged._restore_search_cache(payload, request=request,
                source_run_id=source_run_id, source_revision=source_revision,
                before=before, action=action, submitted=submitted, accepted=accepted,
                source_drink_slots=source_drink_slots, unsubmitted_plan=unsubmitted_plan,
                verified_successors=verified_successors)
            if staged.information_boundary_execution and result.get('restored'):
                if payload['phase'] == 'observed_successor':
                    staged._information_boundary = boundary
            if result.get('restored'):
                if (staged._information_boundary or {}).get('verified_successors') is not None:
                    segment = []
                self._commit_cache_restore(staged, segment)
            return result
        except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
            raise ValueError('native_mcts_cache_malformed') from error

    def _restore_search_cache(self, payload, *, request, source_run_id, source_revision,
                              before, action, submitted, accepted, source_drink_slots, unsubmitted_plan=None,
                              verified_successors=None):
        from .hif_mcts_cache import (
            VERSION,
            json_value,
            action_from_json,
            same_pending_plan,
            request_from_public,
            verified_drink_slot_map,
        )

        if payload is None:
            if self.profile == STRONG_PROFILE:
                raise ValueError('native_mcts_complete_policy_cache_required')
            return dict(restored=False, reason='no_search_cache')
        payload = json_value(payload)
        cached_plan = payload.pop('pending_plan', None)
        if cached_plan is not None and (self.profile != STRONG_PROFILE
                or payload.get('phase') != 'pending_intent' or not isinstance(cached_plan, dict)
                or cached_plan.get('accepted') is not True):
            raise ValueError('native_mcts_pending_plan_invalid')
        if cached_plan is not None and unsubmitted_plan is not None and not same_pending_plan(cached_plan, json_value(unsubmitted_plan)):
            raise ValueError('native_mcts_pending_plan_changed')
        if (set(payload) != {'version', 'profile', 'run_id', 'revision', 'phase', 'intent',
                'planning_windows', 'decision_calls', 'candidate'}
                or payload['version'] != VERSION or payload['profile'] != self.profile
                or (payload['run_id'], payload['revision']) != (source_run_id, source_revision)
                or not isinstance(source_run_id, str) or not source_run_id
                or type(source_revision) is not int or source_revision < 0
                or accepted is not True):
            raise ValueError('native_mcts_cache_source_not_verified')
        if (self._pending is not None or self._expected_after is not None
                or self._cache_request is not None or self._windows or self._calls):
            raise ValueError('native_mcts_cache_requires_new_instance')
        for name in ('planning_windows', 'decision_calls'):
            if type(payload[name]) is not int or payload[name] < 0:
                raise ValueError('native_mcts_cache_progress_invalid')
        if payload['decision_calls'] < payload['planning_windows']:
            raise ValueError('native_mcts_cache_progress_invalid')
        request.validate()
        self.kernel._validate_registered_state(request.state, plan_type=request.plan_type)
        anchor = request_from_public(self._candidate_cache_anchor(payload['candidate']), self.kernel)
        if (request.plan_type is not anchor.plan_type or request.utility != anchor.utility
                or request.ruleset_id != self.kernel.ruleset_id):
            raise ValueError('native_mcts_cache_request_context_changed')
        canonical = self.kernel._canonical_probability_state(request.state)
        unsubmitted_plan = json_value(unsubmitted_plan)
        unsubmitted = unsubmitted_plan is not None
        if unsubmitted and (self.profile != STRONG_PROFILE or payload['phase'] != 'pending_intent'
                or submitted is not False or before is not None or action is not None
                or canonical != anchor.state or (self.run_id, self.revision) != (source_run_id, source_revision)
                or not isinstance(unsubmitted_plan, dict) or unsubmitted_plan.get('accepted') is not True
                or action_key(action_from_json(unsubmitted_plan.get('action')))
                   != action_key(action_from_json(payload['intent']))):
            raise ValueError('native_mcts_unsubmitted_plan_not_bound')
        if payload['phase'] == 'pending_intent':
            intended = action_from_json(payload['intent'])
            if self.profile == STRONG_PROFILE and action is not None and action.kind is ActionKind.USE_DRINK:
                from .hif_drink_slot_binding import HifDrinkSlotBinding

                if before is None or source_drink_slots is None:
                    raise ValueError('native_mcts_cache_requires_verified_source_drink_slots')
                action = HifDrinkSlotBinding.create(before, anchor.drink_slots,
                    source_drink_slots).logical_action(action)
            if not unsubmitted and (submitted is not True or before is None or action is None
                    or self.kernel._canonical_probability_state(before) != anchor.state
                    or action_key(action) != action_key(intended)):
                raise ValueError('native_mcts_cache_requires_verified_submission')
            slots = anchor.drink_slots
            if not unsubmitted and action.kind is ActionKind.USE_DRINK:
                slots = tuple(pair for pair in slots if pair != (action.source_slot, action.drink_id))
        elif payload['phase'] == 'observed_successor':
            if (payload['intent'] is not None or before is not None or action is not None
                    or canonical != anchor.state):
                raise ValueError('native_mcts_cache_observed_successor_changed')
            slots = anchor.drink_slots
        else:
            raise ValueError('native_mcts_cache_phase_invalid')
        verified_drink_slot_map(canonical, slots, request.drink_slots)
        request = self._logical_request(request, drink_slots=slots)
        boundary = (self._classify_committed_boundary(before, action, request.state,
                        verified_successors=verified_successors)
            if self.information_boundary_execution and payload['phase'] == 'pending_intent' and not unsubmitted else None)
        result = self._restore_candidate_cache(payload['candidate'], anchor)
        if unsubmitted:
            matches = [value for value in self._legal(request) if action_key(value) == action_key(intended)]
            if (intended.kind not in (ActionKind.PLAY_CARD, ActionKind.USE_DRINK, ActionKind.END_TURN) or len(matches) != 1
                    or matches[0] != action_from_json(unsubmitted_plan['action'])):
                raise ValueError('native_mcts_unsubmitted_action_not_legal')
            if intended.kind is ActionKind.USE_DRINK and intended.drink_id == '22':
                from .hif_mcts_native_choice import validate_retrieval_plan

                validate_retrieval_plan(self, canonical, matches[0], unsubmitted_plan,
                    self.candidate._choice_continuation)
            self._pending = dict(before=canonical, action=matches[0], utility=request.utility,
                drink_slots=request.drink_slots, plan=copy.deepcopy(unsubmitted_plan), drink_binding=self._drink_binding)
            self._cache_request = replace(request, state=canonical)
            self._calls, self._windows = payload['decision_calls'], payload['planning_windows']
            self._tree_context = 'restored_complete_unsubmitted_intent'
            return dict(result, restored=True, reason=self._tree_context, new_search_performed=False,
                input_authorized=False, retained_nodes=len(self.candidate.search._tree))
        if payload['phase'] == 'pending_intent':
            saved = self.candidate._choice_continuation
            if (action.kind is ActionKind.PLAY_CARD and saved is not None
                    and saved['target'] == GameAction(ActionKind.END_TURN)
                    and not (boundary or {}).get('verified_successors')
                    and (saved['first'] != replace(action, source_slot=None)
                         or saved['before'] != anchor.state or saved['expected_state'] != canonical)):
                raise ValueError('native_mcts_post_card_end_observation_changed')
            committed_before = (self.kernel._canonical_probability_state(before)
                                if self.information_boundary_execution else before)
            self.candidate.committed(committed_before, action, request.state)
            if self.information_boundary_execution:
                boundary['source_plan_index'] = max(0, self.candidate._plan_offset - 1)
                self._information_boundary = boundary
        if self.profile != STRONG_PROFILE:
            self._rebind_observed_drink_slots(canonical, slots, request.drink_slots)
        if self.profile != STRONG_PROFILE:
            root = self.candidate.search._state_key(canonical, request.drink_slots)
            self.candidate.search._retain(root)
        self._expected_after = canonical
        self._cache_request = replace(request, state=canonical)
        self._calls, self._windows = payload['decision_calls'], payload['planning_windows']
        self._tree_context = 'restored_completed_training_verified_public_state'
        return dict(result, restored=True, retained_nodes=len(self.candidate.search._tree),
                    source_run_id=source_run_id, source_revision=source_revision,
                    reason=self._tree_context)

    def matches_context(self, kernel, *, run_id, revision, profile=None):
        return ((profile is None or profile == self.profile)
                and (run_id, revision) == (self.run_id, self.revision)
                and _definitions(kernel) == self.definitions
                and _definitions(self.kernel) == self.definitions)

    def bind_hand_support_context(self, *, run_id, revision, before, after):
        """Invalidate a same-business-state intent after learning held effects.

        Only typed facts are received here. Full receipt verification belongs to
        the bound policy; this method never grants selection or input authority.
        """
        if not self.matches_context(self.kernel, run_id=run_id, revision=revision):
            raise ValueError('native_mcts_hand_support_context_changed')
        before.validate()
        after.validate()
        self.kernel._validate_registered_state(after, plan_type=PlanType.SENSE)
        if (replace(after, hand_support=before.hand_support) != before
                or not set(before.hand_support).issubset(after.hand_support)):
            raise ValueError('native_mcts_hand_support_requires_same_public_state')
        held = {(card.instance_id, card.card_id) for card in after.deck.hand}
        if any(fact.turn != after.turn or (fact.instance_id, fact.card_id) not in held
               for fact in after.hand_support):
            raise ValueError('native_mcts_hand_support_requires_current_held_entities')
        if after.hand_support == before.hand_support:
            return False
        canonical = self.kernel._canonical_probability_state(before)
        if ((self._pending is not None and self._pending['before'] != canonical)
                or (self._expected_after is not None and self._expected_after != canonical)):
            raise ValueError('native_mcts_hand_support_previous_state_changed')
        self.candidate.reset()
        self._pending = self._expected_after = None
        self._cache_request = None
        self._retrieval_choice = None
        self._information_boundary = None
        self._tree_context = 'replanned_after_verified_current_hand_support'
        return True

    def discard_pending(self):
        """For uncertain/unsupported poststates; never confirm a predicted branch."""
        if self.profile == STRONG_PROFILE:
            # A lost settlement/cache must not turn the strong reference into
            # an apparently equivalent cold search on the next host step.
            self._window_incomplete = True
        self.candidate.reset()
        self._pending = self._expected_after = None
        self._cache_request = None
        self._retrieval_choice = None
        self._information_boundary = None
        self._tree_context = 'discarded_unverified_or_unsupported_successor'

    def _legal(self, request, *, retrieval_target=None):
        request.validate()
        if not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id:
            raise ValueError('native_mcts_request_not_ready')
        if _definitions(self.kernel) != self.definitions:
            raise ValueError('native_mcts_definitions_changed')
        expected = (request.state.pending_choice.candidate_instance_ids if request.state.pending_choice else
                    tuple(ref.instance_id for ref in request.state.deck.hand))
        if retrieval_target is not None:
            if not request.state.pending_choice or retrieval_target not in expected:
                raise ValueError('native_mcts_retrieval_target_outside_current_choice')
            expected = (retrieval_target,)
        if (set(dict(request.card_slots)) != set(expected)
                or any(type(slot) is not int or slot < 0 for _, slot in request.card_slots)):
            raise ValueError('native_mcts_complete_card_slot_binding_required')
        actions = self.candidate.kernel.legal_actions(request.state, plan_type=request.plan_type,
                                            decision_kind=request.decision_kind)
        return tuple(RobustHighScorePlanner._attach_source_slot(action, request) for action in actions
                     if retrieval_target is None or action.card_instance_id == retrieval_target)

    def _response(self, action, reason, started, plan, tree_context):
        planning = copy.deepcopy(plan)
        if self.profile == STRONG_PROFILE:
            from .hif_strong_reference import REFERENCE_COMMIT

            if action is not None:
                action = self._drink_binding.observed_action(action)
                planning['action'] = asdict(action)
            planning.update(reference_commit=REFERENCE_COMMIT, independent_quality_passed=False)
        trace = HifMctsNativeTrace(planner=self.profile, planner_version=self.version,
            policy_profile=self.profile,
            ruleset_id=self.kernel.ruleset_id, completed_depth=plan.get('maximum_depth', 0),
            expanded_nodes=plan.get('expanded_nodes', 0), elapsed_ms=(time.perf_counter()-started)*1000,
            reason_codes=(reason,), value_scope=plan.get('value_scope', plan.get('scope', 'SAMPLING_ESTIMATE')),
            run_id=self.run_id, revision=self.revision, tree_context=tree_context,
            planning_seed=self.planning_seed, planning_window_index=self._windows,
            planning=planning)
        return HifMctsNativeDecision(action, reason, trace)

    def _frozen_lookup_is_read_only(self, request, legal):
        """Do not let a frozen lookup start sparse/timing helper evaluation."""
        search = self.candidate.search
        if search._internal_policy is not None:
            from .hif_internal_reanalysis import lookup

            if lookup(search._internal_policy, request, legal,
                      context=search._context(request), kernel=self.kernel) is not None:
                return True
        _, node = search._physical_node(request, legal)
        if node is None or any(not row[0] for row in node.statistics.values()):
            return False
        original = max(node.order, key=lambda key: node.statistics[key][1] / node.statistics[key][0])
        extra = getattr(search.unknown_node_policy, 'extra_action_policy', None)
        return not (extra is not None and (
            original[0] == ActionKind.END_TURN.value
            or original[0] == ActionKind.PLAY_CARD.value and getattr(extra, 'visited_card_extra_action', False)
            or original[0] == ActionKind.USE_DRINK.value and getattr(extra, 'unused_drink_enabled', False)))

    def _evaluated_continuation(self, request, legal):
        """Choose from the committed state without training or opening a window."""
        from .hif_frozen_policy_continuation import matches_link

        candidate = self.candidate
        context, deadline = candidate.search._context(request), candidate.deadline
        if deadline is None:
            deadline = candidate.window_deadline(time.perf_counter())
        plan = dict(accepted=False, new_search_performed=False,
            information_boundary_execution_version=_INFORMATION_EXECUTION_VERSION,
            continuation_origin=copy.deepcopy(self._information_boundary),
            requested_simulations=0, completed_simulations=0, candidate_statistics=[])
        cohort = ()
        if candidate.deterministic_validation_cohort:
            cohort = candidate._validation_cohort.prepare(request, context)
            plan['validation_cohort_link'] = candidate._validation_cohort.summary()
            plan['validation_cohort_link_created'] = False
        retained = candidate._retained_policy
        if retained is not None:
            from .hif_mcts_cache import public_request

            proof = getattr(retained, 'proof', None)
            if (proof is not None and proof.get('committed') is True
                    and proof['after'] == public_request(request)
                    and set(dict(proof['after_card_slots'])) == set(dict(request.card_slots))):
                # Same complete business state; replace calculation slot labels
                # with this request's bindings, never old input credentials.
                proof['after_card_slots'] = tuple(request.card_slots)
            retained.prepare(request, cohort, plan.get('validation_cohort_link', {}))
            plan['retained_policy_lineage'] = retained.evidence()
            plan['retained_policy_link_created'] = False
        action = candidate._resume_choice(request, legal, plan)
        if action is None and request.state.turns_remaining <= 2:
            from .hif_executable_tail import lookup
            action = lookup(self.kernel, request, legal, self._exact_segment)
            if action is not None:
                plan.update(scope='EVALUATED_EXACT_TAIL_CONTINUATION',
                    stop_reason='existing_completed_terminal_action')
        if action is None and request.state.turns_remaining <= 2:
            # The complete tail memo stores values, not generally winning
            # actions. Only an existing complete ordering is reusable here.
            orderer = candidate.search.complete_orderer
            saved = (orderer._cache.get(orderer._key(request, tuple(legal)))
                     if orderer is not None and orderer._frozen else None)
            order = None if saved is None else saved[0]
            if order is not None and len(order) == len(legal) and set(order) == {action_key(a) for a in legal}:
                action = next(a for a in legal if action_key(a) == order[0])
                plan.update(scope='EVALUATED_EXACT_TAIL_CONTINUATION',
                            stop_reason='existing_complete_terminal_order')
        if action is None and request.state.turns_remaining > 2:
            for controller in (candidate._public_turn_continuation, candidate._resource_continuation):
                if controller is not None:
                    controller.prepare(request, context)
            for controller in (candidate._public_turn_continuation, candidate._resource_continuation):
                if controller is not None:
                    action = controller.choose(request, legal, plan, deadline)
                    if action is not None:
                        break
            if (action is None and matches_link(candidate._frozen_policy_continuation, request, context)
                    and self._frozen_lookup_is_read_only(request, legal)):
                action = candidate._resume_frozen_policy(request, legal, plan, deadline)
            if action is None and retained is not None and retained.active and retained.frozen is not None:
                # This is the same complete frozen policy which won the paired
                # comparison, including its deterministic fallback selector.
                # Executing that selector does not train or sample new returns.
                retained_started = time.perf_counter()
                action = retained.choose(request, legal, plan, deadline,
                    evaluated_complete_policy=self.profile == STRONG_PROFILE)
                plan['retained_policy_evaluation_ms'] = (time.perf_counter() - retained_started) * 1000
                if (self.profile == STRONG_PROFILE and action is not None
                        and plan.get('retained_policy_execution_origin') == 'evaluated_complete_root_policy'):
                    from .hif_sparse_score_setup import select_sparse_score_setup

                    try:
                        setup = select_sparse_score_setup(retained.frozen, request, legal, action)
                    except (KeyError, ValueError, RuntimeError, TimeoutError):
                        setup = None
                    if setup is not None:
                        plan = dict(scope='PUBLIC_CARD_FIRST_SCORE_SETUP', stop_reason='unique_score_setup_before_draw_drink',
                            new_search_performed=False, requested_simulations=0, started_simulations=0, completed_simulations=0,
                            training_started_simulations=0, training_completed_simulations=0,
                            validation_started_simulations=0, validation_completed_simulations=0,
                            complete_root_coverage=False, complete_validation_coverage=False, candidate_statistics=[],
                            information_boundary_execution_version=_INFORMATION_EXECUTION_VERSION,
                            continuation_origin=plan['continuation_origin'], retained_alternative=dict(
                                action=asdict(action), source_plan_index=plan['retained_policy_execution_source_plan_index'],
                                scope=plan['scope'], visits_per_action=1))
                        action = setup
                        # The earlier winner evaluated the drink. Its cohort and
                        # frozen execution link cannot certify a different play.
                        cohort = ()
                        candidate._validation_cohort.reset()
                        candidate._frozen_policy_continuation = None
        if action is None:
            raise ValueError('native_mcts_requires_replan:no_evaluated_continuation')
        matches = [value for value in legal if action_key(value) == action_key(action)]
        if len(matches) != 1:
            raise ValueError('native_mcts_continuation_action_not_uniquely_bound')
        action = matches[0]
        plan.update(accepted=True, action=asdict(action))
        candidate.plans.append(plan)
        index = candidate._plan_offset + len(candidate.plans) - 1
        if cohort:
            plan['validation_cohort_link_created'] = candidate._validation_cohort.remember(
                request, action, cohort, context, deadline, index)
        if candidate.deterministic_policy_continuation:
            candidate._remember_frozen_policy(request, action, plan, deadline)
        if candidate._public_turn_continuation is not None:
            candidate._public_turn_continuation.remember(request, action, plan, None, context, index)
        if candidate._resource_continuation is not None:
            candidate._resource_continuation.remember(request, action, plan, None, context, index, deadline)
        if retained is not None:
            plan['retained_policy_link_created'] = retained.remember(request, plan, index, deadline)
            if candidate.revealed_policy_competition:
                plan['revealed_policy_link_created'] = plan['retained_policy_link_created']
        return action, plan

    def replan_exhausted_continuation(self, request, stopped):
        """Extend missing evaluated work from the committed state and same tree.

        The host invokes this only after a read-only continuation miss. Existing
        plans still run without search; incomplete windows and unknown poststates
        cannot obtain another budget through this path.
        """
        canonical = self.kernel._canonical_probability_state(self._logical_request(request).state)
        if (self.profile != STRONG_PROFILE or not self.information_boundary_execution
                or stopped.action is not None
                or stopped.reason != 'native_mcts_requires_replan:no_evaluated_continuation'
                or self._pending is not None or self._window_incomplete
                or self._expected_after != canonical
                or (self._information_boundary or {}).get('kind') != 'deterministic_same_turn'
                or stopped.trace.planning_window_index != self._windows
                or (stopped.trace.run_id, stopped.trace.revision) != (self.run_id, self.revision)):
            raise ValueError('native_mcts_replan_requires_committed_missing_continuation')
        self._information_boundary = None
        self._tree_context = 'same_tree_extended_after_evaluated_continuation_exhausted'
        return self.decide(request)

    def decide(self, request):
        started = time.perf_counter()
        tree_context = self._tree_context
        self._tree_context = 'same_native_instance_observed_successors_only'
        self._calls += 1
        plan = {}
        try:
            if self.profile == STRONG_PROFILE and self._window_incomplete:
                raise ValueError('native_mcts_previous_window_incomplete')
            request = self._logical_request(request)
            if self.candidate.deadline is not None and time.perf_counter() >= self.candidate.deadline:
                raise TimeoutError('native_mcts_host_session_expired')
            canonical = self.kernel._canonical_probability_state(request.state)
            from .hif_round1_card_only import retains_round1_drinks

            if (self.profile == STRONG_PROFILE and retains_round1_drinks(self.run_id, self.revision, request.state)
                    and not getattr(self.candidate.kernel.drinks.get('4'), 'reserved_for_later', False)):
                if self._pending is not None and self._pending['before'] != canonical:
                    raise ValueError('native_mcts_previous_submission_not_confirmed')
                if self._expected_after is not None and self._expected_after != canonical:
                    raise ValueError('native_mcts_observed_successor_changed')
                # The old post-drink tree priced future drink actions. Keep the
                # actual commit, elapsed work and bottle slots, start new values.
                prior = self.candidate
                self.candidate = self._new_candidate(retain_drinks=True)
                self.candidate._plan_offset = prior._plan_offset + len(prior.plans)
                self.candidate._strong_work_totals = dict(prior._strong_work_totals)
                self.candidate._strong_last_window_work = copy.deepcopy(prior._strong_last_window_work)
                self._pending = self._cache_request = self._retrieval_choice = None
                self._information_boundary, self._exact_segment = None, []
                tree_context = 'replanned_after_round1_drinks_reserved'
            legal = self._legal(request)
            if request.decision_kind is DecisionKind.SECONDARY and not self.secondary_ready:
                raise ValueError('native_mcts_mandatory_choice_not_connected')
            if self._pending is not None:
                pending = self._pending
                if (canonical != pending['before'] or request.utility != pending['utility']
                        or request.drink_slots != pending['drink_slots']):
                    raise ValueError('native_mcts_previous_submission_not_confirmed')
                matches = [action for action in legal if action_key(action) == action_key(pending['action'])]
                if len(matches) != 1:
                    raise ValueError('native_mcts_refresh_action_not_uniquely_bound')
                pending['action'] = matches[0]
                if self.profile == STRONG_PROFILE:
                    pending['drink_binding'] = self._drink_binding
                plan = copy.deepcopy(pending['plan'])
                plan['action'] = asdict(matches[0])
                return self._response(matches[0], 'native_mcts_same_state_rebound', started, plan, tree_context)
            if self._expected_after is not None and canonical != self._expected_after:
                raise ValueError('native_mcts_observed_successor_changed')
            boundary = self._information_boundary if self.information_boundary_execution else None
            if boundary is not None and boundary['kind'] == 'requires_replan':
                raise ValueError('native_mcts_requires_replan:' + boundary['reason'])
            if boundary is not None and boundary['kind'] == 'deterministic_same_turn':
                from .hif_mcts_cache import exact_state

                if exact_state(boundary['state']) != canonical:
                    raise ValueError('native_mcts_information_boundary_successor_changed')
                action, plan = self._evaluated_continuation(request, legal)
            else:
                before_plans = len(self.candidate.plans)
                if self.profile == STRONG_PROFILE:
                    self._window_incomplete = True
                self._windows += 1
                try:
                    if self.before_planning_window is not None:
                        try:
                            self.before_planning_window(self, request)
                        except OSError as error:
                            raise ValueError('native_mcts_window_start_not_recorded') from error
                    from .hif_executable_tail import extract_segment, completed_choices
                    with completed_choices(self.candidate, request, self.information_boundary_execution) as choices:
                        action = self.candidate.choose(request, legal)
                    if self.information_boundary_execution and choices:
                        self._exact_segment = extract_segment(self.kernel, request, action, choices)
                finally:
                    if len(self.candidate.plans) > before_plans:
                        plan = self.candidate.plans[-1]
            if plan.get('accepted') is not True or action not in legal:
                raise ValueError('native_mcts_candidate_not_accepted_or_bound')
            if _has_post_card_end(action, plan):
                from .hif_mcts_native_choice import validate_post_card_end_plan

                validate_post_card_end_plan(self, request, action, plan,
                                           self.candidate._choice_continuation)
            elif not self.secondary_ready and (plan.get('selected_continuation')
                    or _declared_choices(self.kernel, request.state, action)):
                from .hif_mcts_native_choice import validate_retrieval_plan

                # The narrow host connection covers one already evaluated
                # black-vinegar MOVE, never arbitrary card/secondary choices.
                # An unsupported winner still stops instead of falling back.
                validate_retrieval_plan(self, request.state, action, plan,
                                        self.candidate._choice_continuation)
            self._pending = dict(before=canonical, action=action, utility=request.utility,
                drink_slots=request.drink_slots, plan=copy.deepcopy(plan))
            if self.profile == STRONG_PROFILE:
                self._pending['drink_binding'] = self._drink_binding
            self._cache_request = replace(request, state=canonical)
            if self.profile == STRONG_PROFILE:
                self._window_incomplete = False
            return self._response(action, 'native_mcts_complete_candidate_intent', started, plan, tree_context)
        except (ValueError, RuntimeError, TimeoutError) as error:
            return self._response(None, str(error), started, plan, tree_context)

    def recover_end_turn(self, state):
        """Keep the prior decision as evidence and bind the host's legal SKIP."""
        pending = self._pending
        request = self._cache_request
        if (pending is None or request is None or self._window_incomplete
                or self.kernel._canonical_probability_state(state) != pending['before']):
            raise ValueError('native_mcts_recovery_requires_original_pending_state')
        action = next((candidate for candidate in self._legal(request) if candidate.kind is ActionKind.END_TURN), None)
        if action is None:
            raise ValueError('native_mcts_recovery_skip_not_legal')
        plan = dict(accepted=True, action=asdict(action), source='hiro_forced_skip',
                    value_scope='UNSCORED_HOST_RECOVERY', replaced_action=asdict(pending['action']))
        self._pending = dict(pending, action=action, plan=plan)
        self._exact_segment = []
        return self._response(action, 'hiro_forced_skip', time.perf_counter(), plan, 'observed_no_effect')

    def _rebind_observed_drink_slots(self, state, source_slots, observed_slots):
        from .hif_mcts_cache import rebind_retained_drink_slots

        if tuple(source_slots) == tuple(observed_slots):
            return
        if self.profile == STRONG_PROFILE:
            raise ValueError('native_mcts_complete_policy_drink_slots_changed')
        if (self.candidate._choice_continuation is not None
                or self.candidate._frozen_policy_continuation is not None):
            raise ValueError('native_mcts_drink_rebind_active_continuation')
        rebind_retained_drink_slots(self.candidate.search, state, source_slots, observed_slots)

    def confirm_committed(self, *, run_id, revision, before, action, after, submitted, accepted,
                          drink_slots=None, verified_successors=None):
        """The caller supplies actual dispatch and complete observation evidence."""
        if submitted is not True or accepted is not True:
            raise ValueError('native_mcts_requires_submitted_and_observed_commit')
        if not self.matches_context(self.kernel, run_id=run_id, revision=revision):
            raise ValueError('native_mcts_commit_context_changed')
        pending = self._pending
        observed_action = action
        if self.profile == STRONG_PROFILE and pending is not None:
            action = pending['drink_binding'].logical_action(action)
        if (self.retrieval_ready and not self.secondary_ready and pending is not None
                and pending['action'].kind is ActionKind.USE_DRINK
                and pending['action'].drink_id == '22'
                and pending['plan'].get('selected_continuation')):
            raise ValueError('native_mcts_retrieval_requires_complete_pair_commit')
        if (pending is None or action != pending['action']
                or self.kernel._canonical_probability_state(before) != pending['before']):
            raise ValueError('native_mcts_commit_does_not_match_planned_intent')
        after.validate()
        self.kernel._validate_registered_state(after, plan_type=PlanType.SENSE)
        boundary = (self._classify_committed_boundary(before, action, after,
                        verified_successors=verified_successors)
                    if self.information_boundary_execution else None)
        if _has_post_card_end(action, pending['plan']) and not (boundary or {}).get('verified_successors'):
            from .hif_mcts_native_choice import validate_post_card_end_plan

            expected = validate_post_card_end_plan(self, self._cache_request, action,
                pending['plan'], self.candidate._choice_continuation)
            if self.kernel._canonical_probability_state(after) != expected:
                raise ValueError('native_mcts_post_card_end_observation_changed')
        slots = self._cache_request.drink_slots
        if action.kind is ActionKind.USE_DRINK:
            slots = tuple(pair for pair in slots if pair != (action.source_slot, action.drink_id))
        binding = None
        if self.profile == STRONG_PROFILE:
            from .hif_drink_slot_binding import HifDrinkSlotBinding

            observed = drink_slots
            if observed is None:
                observed = pending['drink_binding'].observed_slots
                if action.kind is ActionKind.USE_DRINK:
                    observed = tuple(pair for pair in observed
                        if pair != (observed_action.source_slot, observed_action.drink_id))
            binding = HifDrinkSlotBinding.create(after, slots, observed)
        self.candidate.committed(
            self.kernel._canonical_probability_state(before) if self.information_boundary_execution else before,
            action,
            self.kernel._canonical_probability_state(after) if self.information_boundary_execution else after)
        self._information_boundary = boundary
        if (boundary or {}).get('verified_successors'):
            self._exact_segment = []
        self._expected_after = self.kernel._canonical_probability_state(after)
        if binding is not None:
            self._logical_drink_slots, self._drink_binding = binding.logical_slots, binding
        elif drink_slots is not None:
            from .hif_mcts_cache import json_value, slots_from_json

            observed_slots = slots_from_json(json_value(drink_slots), after)
            self._rebind_observed_drink_slots(after, slots, observed_slots)
            slots = observed_slots
        ids = (after.pending_choice.candidate_instance_ids if after.pending_choice else
               tuple(card.instance_id for card in after.deck.hand))
        self._cache_request = replace(self._cache_request, state=self._expected_after,
            decision_kind=DecisionKind.SECONDARY if after.pending_choice else DecisionKind.LIVE,
            drink_slots=slots, card_slots=tuple((identity, slot) for slot, identity in enumerate(ids)))
        self._pending = None
