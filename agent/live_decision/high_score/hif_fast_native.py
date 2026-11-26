"""Explicit experimental bridge to the frozen offline fast policy.

Only business data crosses the native/offline Python package boundary. The
normal native transaction remains the sole owner of observations and input.
"""
from __future__ import annotations

import math
import time
from enum import Enum
from dataclasses import asdict, fields, replace, dataclass, is_dataclass

from route.battle_policy import FAST_PROFILE, UINET21_PROFILE, NETWORK_PROFILES

from . import kernel as native_types
from . import contracts as native_contracts
from .planner import RobustHighScorePlanner
from .contracts import ActionKind, DecisionKind, DecisionTrace, HifActionIntent
from .hif_mcts_cache import json_value, definitions, exact_state, action_from_json
from .gkms_master_forms import PLAN_CODES

PROFILE = FAST_PROFILE
VERSION = 'hif-native-gkms-continuous-v1'
CACHE_VERSION = 'hif-native-gkms-history-v3'
V2_CACHE_VERSION = 'hif-native-gkms-v2.1-history-v4'


def _decode_expired_start_effects(raw, *, kernel, state):
    """Read history residues without admitting exhausted effects into state."""
    from .hif_strong_policy_cache import decode_value

    if (not isinstance(raw, dict) or set(raw) != {'type', 'items'}
            or raw['type'] != 'tuple' or not isinstance(raw['items'], list)):
        raise ValueError('fast_cache_expired_start_history_invalid')
    result = []
    for item in raw['items']:
        values = item.get('fields') if isinstance(item, dict) else None
        if (not isinstance(values, dict) or item.get('name') != 'ScheduledEffectRef'
                or type(values.get('remaining_uses')) is not int or values['remaining_uses'] != 0):
            raise ValueError('fast_cache_expired_start_history_invalid')
        # TurnPlayHistory.advance records the final activation with uses=0.
        # Validate every other field through the unchanged active-ref codec,
        # then retain the original zero only in this explicit history tuple.
        ref = decode_value(dict(item, fields=dict(values, remaining_uses=1)), kernel=kernel)
        if (ref.persistent or ref.trigger_phase is not native_contracts.SchedulePhase.START_OF_TURN
                or ref.last_fired_turn != state.turn):
            raise ValueError('fast_cache_expired_start_history_invalid')
        result.append(replace(ref, remaining_uses=0))
    return tuple(result)


def persist_projected_events(context,policy,branches=None):
    candidate=getattr(policy,'decision_policy',None)
    if not isinstance(candidate,HifFastNativePolicy):
        return
    if branches is not None:
        policy.committed_event_branches=tuple(branches)
    sink=getattr(context,'gkh_hif_mcts_cache_sink',None)
    if not callable(sink):
        raise ValueError('fast_native_projection_cache_sink_missing')
    candidate.retain_projected_events(policy)
    sink(candidate)


def _master_exam_fields(run, round_number, base_turns, tables):
    from ..hif_profiles import supported_hif_identity
    from ..idol_road_formation import find_idol_card

    profile = supported_hif_identity(run.idol, run.plan)
    card = find_idol_card(profile.idol, profile.song, characters=tables['Character'], idol_cards=tables['IdolCard'])
    plan = PLAN_CODES[profile.plan.upper()]
    if card is None or card['planType'] != f'ProducePlanType_Plan{plan - 1}':
        raise ValueError('fast_exam_idol_master_match_missing')
    phase, step = {1: ('AuditionMid1', 'AUDITION_MID1'), 2: ('AuditionFinal', 'AUDITION_FINAL')}[round_number]
    difficulty = [row for row in tables['ProduceStepAuditionDifficulty']
                  if row['id'] == card['produceStepAuditionDifficultyId']
                  and row['produceId'] == 'produce-008' and row['stepType'] == 'ProduceStepType_' + phase
                  and row['number'] == 1]
    if len(difficulty) != 1:
        raise ValueError('fast_exam_difficulty_master_match_missing')
    key = difficulty[0]['produceExamBattleConfigId']
    configs = [row for row in tables['ProduceExamBattleConfig'] if row['id'] == key and row['turn'] == base_turns]
    if len(configs) != 1:
        raise ValueError('fast_exam_battle_master_match_missing')
    criteria = tuple(configs[0][name] for name in ('vocal', 'dance', 'visual'))
    return step, plan, criteria, card['id'], key


def observed_exam_context(bundle, run, *, round_context=None):
    """Use observed turns and multipliers with the selected role's exam data."""
    from agent.live_decision.high_score.gkms_shadow_adapter import ExamContext

    from ..hif_session import resolve_hif_round_context
    from ..hif_gkms_runtime import battle_identity_master
    state = bundle.state
    if state.turn != 1 or state.score != 0 or state.deck.discard or state.deck.removed:
        raise ValueError('fast_initial_history_requires_complete_zero_play_opening')
    round_number, base_turns = resolve_hif_round_context(
        round_context, run_id=bundle.run_id, revision=bundle.revision)
    if len(state.turn_attributes) != base_turns:
        raise ValueError('fast_current_native_exam_round_not_mapped')
    step, plan, criteria, idol, criteria_source = _master_exam_fields(
        run, round_number, base_turns, battle_identity_master())
    values = {}
    for attribute, multiplier in zip(state.turn_attributes,
            (state.score_multiplier, *state.future_score_multipliers), strict=True):
        value = round(multiplier * 1000)
        if attribute in values and values[attribute] != value:
            raise ValueError('fast_exam_attribute_multiplier_changed')
        values[attribute] = value
    names = ('vocal','dance','visual')
    if set(values) != set(names):
        raise ValueError('fast_exam_complete_attribute_table_missing')
    return ExamContext('produce-008', step, plan,
        tuple(names.index(value)+1 for value in state.turn_attributes),
        tuple(values[name] for name in names), criteria, idol,
        f'complete native opening {bundle.run_id}:{bundle.revision}; '
        f'round {round_number}; criteria: {criteria_source}')


def business_copy(value, *, offline):
    """Copy declared immutable records, not executable objects or module names.

    The offline tools intentionally import agent.live_decision; the native
    application imports live_decision. Equal Enum names are not equal types.
    """
    from agent.live_decision.high_score import kernel as offline_types
    from agent.live_decision.high_score import contracts as offline_contracts

    target = (offline_contracts, offline_types) if offline else (native_contracts, native_types)
    origins = (native_contracts, native_types, offline_contracts, offline_types)

    def copy(current):
        if current is None or type(current) in (bool, str, int):
            return current
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError('fast_business_nonfinite')
            return current
        if isinstance(current, Enum) or is_dataclass(current):
            name = type(current).__name__
            if not any(getattr(module, name, None) is type(current) for module in origins):
                raise ValueError('fast_business_record_not_declared:' + name)
            cls = next((getattr(module, name) for module in target if hasattr(module, name)), None)
            if cls is None:
                raise ValueError('fast_business_record_not_declared:' + name)
            return cls(current.value) if isinstance(current, Enum) else cls(**{
                field.name: copy(getattr(current, field.name)) for field in fields(current)})
        if type(current) in (tuple, list):
            return type(current)(copy(item) for item in current)
        if type(current) is dict:
            return {copy(key): copy(item) for key, item in current.items()}
        raise ValueError('fast_business_object_not_supported:' + type(current).__name__)
    return copy(value)


@dataclass(frozen=True)
class FastTrace(DecisionTrace):
    policy_profile: str = PROFILE
    run_id: str = ''
    revision: int = 0
    decision: dict | None = None


@dataclass(frozen=True)
class FastDecision(HifActionIntent):
    trace: FastTrace
    protocol_version = VERSION


class HifFastNativePolicy:
    profile, version = PROFILE, VERSION

    def __init__(self, bundle, *, exam_context, source, model=None, planning_seed=1731105100,
                 deadline=None, initial_history=None, profile=PROFILE):
        from agent.live_decision.high_score.hif import HifSenseKernel
        from agent.live_decision.high_score.decision_box import build_battle_policy

        if profile not in NETWORK_PROFILES:
            raise ValueError('fast_native_profile_not_supported')
        self.profile = profile

        self.kernel, self.run_id, self.revision = bundle.kernel, bundle.run_id, bundle.revision
        self.definitions = definitions(bundle.kernel)
        offline_kernel = HifSenseKernel(plan_type=business_copy(bundle.kernel.plan_type, offline=True), **{name: business_copy(getattr(bundle.kernel, name), offline=True)
            for name in ('cards', 'drinks', 'items', 'memory_abilities')})
        state = business_copy(bundle.state, offline=True)
        algorithm = 'uinet-2.1' if profile == UINET21_PROFILE else 'uinet-1.1'
        self.policy = build_battle_policy(algorithm, kernel=offline_kernel, context=exam_context, state=state, model=model,
            source=source, planning_seed=planning_seed,
            deadline=deadline if deadline is not None else time.perf_counter() + 300.,
            opening_hand_order=bundle.hand_instance_ids, use_observed_hand_order=True)
        upgrades = {key: card.upgrade_card_id for key, card in offline_kernel.cards.items()
                    if card.upgrade_card_id in offline_kernel.cards}
        self.policy.history = replace(initial_history or self.policy.history, upgrade_ids=upgrades)
        self._pending = None
        self._box = None
        self._retrieval = None
        self._pending_branches = ()
        self._phase = 'observed_successor'
        self._invalid = None
        self.before_secondary = None
        self.after_secondary = None

    def matches_context(self, kernel, *, run_id, revision, profile=None):
        return (self.run_id, self.revision, self.profile, self.definitions) == (
            run_id, revision, profile or self.profile, definitions(kernel))

    def _same(self, left, right):
        return self.kernel._canonical_probability_state(left) == self.kernel._canonical_probability_state(right)

    def _legal(self, request):
        request.validate()
        if (not request.capabilities.all_ready or request.ruleset_id != self.kernel.ruleset_id
                or request.plan_type is not self.kernel.plan_type or definitions(self.kernel) != self.definitions):
            raise ValueError('fast_native_request_not_ready')
        legal = self.kernel.legal_actions(request.state, plan_type=request.plan_type,
            decision_kind=request.decision_kind)
        return tuple(RobustHighScorePlanner._attach_source_slot(action, request) for action in legal)

    def _response(self, action, reason, start, entry=None):
        return FastDecision(action, reason, FastTrace(self.profile, VERSION, self.kernel.ruleset_id,
            0, 0, (time.perf_counter() - start) * 1000, reason_codes=(reason,),
            value_scope='ORIGINAL_NETWORK_OR_SAMPLED_LOCAL_CHOICE', run_id=self.run_id,
            revision=self.revision, decision=json_value(entry) if entry is not None else None))

    def decide(self, request):
        started = time.perf_counter()
        try:
            if self._phase == 'terminal_observed':
                raise ValueError('fast_live_terminal_already_observed')
            if self._invalid:
                raise ValueError(self._invalid)
            legal = self._legal(request)
            if request.decision_kind is not DecisionKind.LIVE:
                raise ValueError('fast_secondary_requires_observed_drink_transaction')
            if self._pending is not None:
                before, selected = self._pending
                if not self._same(before, request.state):
                    raise ValueError('fast_uncommitted_intent_state_changed')
                action = self._rebind(selected, request, legal)
                return self._response(action, 'fast_retained_unsubmitted_intent', started,
                                      self.policy.decisions[-1])
            offline_request = business_copy(request, offline=True)
            native_history_state = business_copy(self.policy.history.state, offline=False)
            if not self._same(native_history_state, request.state):
                raise ValueError('fast_complete_turn_history_not_bound')
            self.policy.history = replace(self.policy.history, state=offline_request.state)
            self.policy.hand_order.ids = tuple(iid for iid, _ in sorted(request.card_slots, key=lambda row: row[1]))
            # A new observed state starts a new decision window. UI time and
            # engineering pauses do not expire already-computed actions.
            self.policy.deadline = started + 300.
            action = self._choose(offline_request, legal)
            native = business_copy(action, offline=False)
            native = self._rebind(native, request, legal)
            self._pending = request.state, native
            self._phase = 'pending_intent'
            return self._response(native, 'gkms_uinet_original_argmax', started, self.policy.decisions[-1])
        except (ValueError, RuntimeError, TimeoutError) as error:
            return self._response(None, str(error), started,
                                  self.policy.decisions[-1] if self.policy.decisions else None)

    def _choose(self, request, legal):
        from agent.live_decision.high_score.decision_box import DecisionBox

        box = DecisionBox(self.policy, request)
        action = box.choose(request, business_copy(legal, offline=True))
        self._box = box
        return action

    def recover_end_turn(self, state):
        """Bind Hiro's host-selected SKIP after verified non-consumption."""
        from agent.live_decision.high_score.decision_box import DecisionBox

        if (self._box is None or self._pending is None or self._invalid
                or not self._same(self._pending[0], state)):
            raise ValueError('fast_recovery_requires_original_pending_state')
        request = self._box.request
        legal = self._legal(business_copy(request, offline=False))
        action = next((candidate for candidate in legal if candidate.kind is ActionKind.END_TURN), None)
        if action is None:
            raise ValueError('fast_recovery_skip_not_legal')
        started = time.perf_counter()
        offline = business_copy(action, offline=True)
        self.policy.decisions[-1]['status'] = 'observed_no_effect'
        entry = dict(index=len(self.policy.decisions), source='hiro_forced_skip',
                     turn=state.turn, status='selected', action=asdict(offline))
        self.policy.decisions.append(entry)
        self.policy._pending = request.state, offline, None
        self._box = DecisionBox(self.policy, request, pending_action=offline)
        self._pending = state, action
        self._pending_branches = ()
        return self._response(action, 'hiro_forced_skip', started, entry)

    @staticmethod
    def _rebind(action, request, legal):
        if action.kind is ActionKind.PLAY_CARD:
            action = replace(action, source_slot=dict(request.card_slots)[action.card_instance_id])
        if action.kind is ActionKind.USE_DRINK:
            if (action.source_slot, action.drink_id) not in request.drink_slots:
                raise ValueError('fast_physical_drink_binding_changed')
            if replace(action, source_slot=None) not in tuple(replace(a, source_slot=None) for a in legal):
                raise ValueError('fast_native_action_not_legal')
        elif action not in legal:
            raise ValueError('fast_native_action_not_legal')
        return action

    def _advance_history(self, before, action, after, branches, *, next_request=None):
        """All rule branches compatible with actual settlement must agree on history."""
        if self._pending is None or not self._same(before, self._pending[0]):
            raise ValueError('fast_observed_commit_has_no_original_intent')
        if replace(action, source_slot=None) != replace(self._pending[1], source_slot=None):
            raise ValueError('fast_observed_commit_action_changed')
        if action.kind is ActionKind.USE_DRINK and action.source_slot != self._pending[1].source_slot:
            raise ValueError('fast_observed_drink_slot_changed')
        matches = [branch for branch in branches if self._same(branch.state, after)]
        if not matches:
            raise ValueError('fast_observed_commit_has_no_complete_rule_events')
        offline_after = business_copy(after, offline=True)
        histories = [self.policy.history.advance(business_copy(action, offline=True), offline_after,
            business_copy(branch.events, offline=True)) for branch in matches]
        if any(history != histories[0] for history in histories[1:]):
            raise ValueError('fast_observed_play_history_ambiguous')
        from agent.live_decision.high_score.hif_environment import HifStep

        previous = self._box.request
        branch_events = [business_copy(branch.events, offline=True) for branch in matches]
        if self.policy.v2_effects is not None:
            effect_states = [self.policy.effect_state_after(previous.state, self._box.pending_action,
                offline_after, events) for events in branch_events]
            if any(effect != effect_states[0] for effect in effect_states[1:]):
                raise ValueError('v2_observed_effect_sources_ambiguous')
        if next_request is None:
            slots = tuple((slot, drink) for slot, drink in previous.drink_slots
                if action.kind is not ActionKind.USE_DRINK or slot != action.source_slot)
            following = replace(previous, snapshot_id=previous.snapshot_id + ':observed',
                state=offline_after,
                decision_kind=business_copy(DecisionKind.LIVE, offline=True),
                card_slots=tuple((ref.instance_id, i) for i, ref in enumerate(offline_after.deck.hand)),
                drink_slots=slots)
        else:
            following = business_copy(next_request, offline=True)
        self._box.observe_transition(previous.state, self._box.pending_action,
            HifStep(following, offline_after.score - before.score, offline_after.terminal,
                    branch_events[0]))
        self._box = None
        self._pending = None
        self._phase = 'observed_successor'

    def confirm_observed(self, policy, after, settlement, *, submitted):
        if submitted is not True or settlement.get('accepted') is not True:
            raise ValueError('fast_commit_requires_actual_submission_and_settlement')
        if self._retrieval is not None:
            before, action, branches = self._retrieval
        else:
            before, action = policy.bundle.state, policy.response.action
            branches = getattr(policy, 'committed_event_branches', ())
        self._advance_history(before, action, after, branches)
        self._retrieval = None
        self._pending_branches = ()

    def confirm_terminal(self, policy, settlement, *, submitted):
        """Close a scored Live without inventing unobserved terminal resources."""
        from ..opening_observer import valid_opening_frame_ids

        before, action = policy.bundle.state, policy.response.action
        if (submitted is not True or settlement.get('accepted') is not True
                or settlement.get('terminal') is not True
                or (settlement.get('run_id'),settlement.get('revision')) != (self.run_id,self.revision)
                or type(settlement.get('round_number')) is not int or settlement['round_number'] not in (1, 2)
                or type(settlement.get('score')) is not int or settlement['score'] < 0
                or self._pending is None or not self._same(self._pending[0],before)
                or replace(self._pending[1],source_slot=None) != replace(action,source_slot=None)
                or not valid_opening_frame_ids(settlement.get('frame_ids'))):
            raise ValueError('fast_terminal_requires_original_submission_and_displayed_score')
        # The result page proves a final score, not a fully observed hand or
        # resource state. Keep the pre-terminal history as evidence and close
        # the action; no resumable successor cache or further decision exists.
        self.policy.decisions[-1]['status'] = 'terminal_observed'
        self.policy.decisions[-1]['registered_score'] = settlement['score']
        self.policy._pending = self._pending = None
        self._box = None
        self._retrieval = None
        self._pending_branches = ()
        self._phase = 'terminal_observed'

    def retain_projected_events(self, policy):
        if self._pending is None or self._pending[1] != policy.response.action:
            raise ValueError('fast_projection_has_no_matching_intent')
        self._pending_branches = tuple(policy.committed_event_branches)

    def bind_retrieval_choice(self, projected, request, *, drink_submitted):
        started = time.perf_counter()
        try:
            if self._pending is not None and self._pending[1].kind is ActionKind.MOVE_CARD:
                if not self._same(self._pending[0],request.state) or not self._pending_branches:
                    raise ValueError('fast_retained_choice_state_or_events_changed')
                action=self._rebind(self._pending[1],request,self._legal(request))
                self._retrieval=request.state,action,self._pending_branches
                return self._response(action,'fast_retained_unsubmitted_choice',started,self.policy.decisions[-1])
            if (drink_submitted is not True or self._pending is None
                    or projected.source_bundle.run_id != self.run_id
                    or projected.source_bundle.revision != self.revision
                    or not self._same(projected.pending_state, request.state)
                    or request.decision_kind is not DecisionKind.SECONDARY):
                raise ValueError('fast_retrieval_requires_submitted_drink_and_fresh_panel')
            before, drink = self._pending
            if drink != projected.action or not self._same(before, projected.source_bundle.state):
                raise ValueError('fast_retrieval_original_drink_changed')
            branches = self.kernel.transition(before, drink, plan_type=request.plan_type)
            self._advance_history(before, drink, request.state, branches, next_request=request)
            legal = self._legal(request)
            if not callable(self.before_secondary) or not callable(self.after_secondary):
                raise ValueError('fast_secondary_requires_persistent_computation_boundary')
            # The host persists this exact deadline before search and refuses
            # an unfinished saved window. Restoring a chosen action never
            # reaches this new-window branch.
            self.policy.deadline = started + 300.
            self.before_secondary(self, request)
            selected = self._choose(business_copy(request, offline=True), legal)
            action = business_copy(selected, offline=False)
            action = self._rebind(action, request, legal)
            self._pending, self._phase = (request.state, action), 'pending_intent'
            branches = self.kernel.transition(request.state, action, plan_type=request.plan_type)
            self._retrieval = request.state, action, branches
            self._pending_branches = tuple(branches)
            self.after_secondary(self)
            return self._response(action, 'local_strong_reference_secondary', started, self.policy.decisions[-1])
        except (ValueError, RuntimeError, TimeoutError) as error:
            self._invalid = 'fast_secondary_not_completed:' + str(error)
            return self._response(None, str(error), started,
                self.policy.decisions[-1] if self.policy.decisions else None)

    def bind_hand_support_context(self, *, run_id, revision, before, after):
        if (run_id, revision) != (self.run_id, self.revision) or not self._same(
                business_copy(self.policy.history.state, offline=False), before):
            raise ValueError('fast_support_history_context_changed')
        if replace(after, hand_support=before.hand_support) != before:
            raise ValueError('fast_support_updated_non_support_state')
        self.policy.history = replace(self.policy.history, state=business_copy(after, offline=True))
        self.policy._pending = self._pending = None
        self._box = None
        self._phase = 'observed_successor'

    def discard_pending(self):
        # Keep the unconfirmed action as evidence; do not cold-start history.
        self._invalid = 'fast_previous_submission_or_settlement_unconfirmed'

    def export_search_cache(self):
        from .hif_strong_policy_cache import encode_value

        if self._phase == 'terminal_observed':
            return None
        history = self.policy.history
        request = self._box.request if self._pending is not None else None
        v2 = self.profile == UINET21_PROFILE
        return dict(version=V2_CACHE_VERSION if v2 else CACHE_VERSION,
            profile=self.profile, run_id=self.run_id, revision=self.revision,
            phase=self._phase, definitions=self.definitions, context=asdict(self.policy.context),
            planning_seed=self.policy.planning_seed, compute_deadline=self.policy.deadline,
            history=dict(state=json_value(asdict(business_copy(history.state, offline=False))),
                played=encode_value(business_copy(history.played, offline=False)), source=history.source,
                played_support=history.played_support,
                expired_start_effects=encode_value(business_copy(history.expired_start_effects, offline=False))),
            hand_order=self.policy.hand_order.ids, secondary_calls=self.policy._secondary_calls,
            decisions=json_value(self.policy.decisions), secondary_plans=json_value(self.policy.secondary_plans),
            timing=self.policy.timing, pending=None if self._pending is None else dict(
                before=json_value(asdict(self._pending[0])), action=json_value(asdict(self._pending[1])),
                snapshot_id=request.snapshot_id, card_slots=request.card_slots, drink_slots=request.drink_slots),
            event_branches=[dict(state=json_value(asdict(b.state)),probability=b.probability,
                events=json_value([asdict(e) for e in b.events])) for b in self._pending_branches],
            invalid=self._invalid, **({'v2_effects': asdict(self.policy.v2_effects)} if v2 else {}))

    @staticmethod
    def require_cache_version(payload, profile=None):
        actual = payload.get('profile') if isinstance(payload, dict) else None
        if (actual not in NETWORK_PROFILES or (profile is not None and actual != profile)
                or payload.get('version') != (V2_CACHE_VERSION if actual == UINET21_PROFILE else CACHE_VERSION)):
            raise ValueError('fast_cache_source_or_rules_not_verified')

    @classmethod
    def from_cache(cls, payload, bundle, *, source_run_id, source_revision, accepted, model=None, profile=None):
        from agent.live_decision.high_score.gkms_shadow_adapter import ExamContext, TurnPlayHistory

        from .hif_strong_policy_cache import decode_value

        cls.require_cache_version(payload, profile)
        payload = json_value(payload)
        if (accepted is not True
                or payload.get('definitions') != definitions(bundle.kernel)
                or (payload.get('run_id'), payload.get('revision')) != (source_run_id, source_revision)
                or payload.get('phase') not in ('pending_intent', 'observed_successor')):
            raise ValueError('fast_cache_source_or_rules_not_verified')
        saved = payload['history']
        state = exact_state(saved['state'])
        canonical = bundle.kernel._canonical_probability_state
        if canonical(state) != canonical(bundle.state):
            raise ValueError('fast_cache_observed_state_changed')
        v2_effects = None
        if payload['profile'] == UINET21_PROFILE:
            from agent.live_decision.high_score.gkms_v2_effects import V2EffectState

            v2_effects = V2EffectState.from_cache(payload.get('v2_effects'), state)
        deadline = payload['compute_deadline']
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
            raise ValueError('fast_cache_compute_budget_invalid')
        history = TurnPlayHistory(business_copy(bundle.state, offline=True),
            business_copy(decode_value(saved['played'], kernel=bundle.kernel), offline=True), saved['source'],
            tuple(saved['played_support']), business_copy(_decode_expired_start_effects(
                saved['expired_start_effects'], kernel=bundle.kernel, state=state), offline=True))
        history.validate(history.state)
        context = dict(payload['context'])
        for key in ('turn_types', 'permil', 'criteria'):
            context[key] = tuple(context[key])
        context = ExamContext(**context)
        from .kernel import RuleEvent, TransitionBranch
        branches = tuple(TransitionBranch(probability=b['probability'],state=exact_state(b['state']),
            events=tuple(RuleEvent(**e) for e in b['events'])) for b in payload['event_branches'])
        pending_intent = None
        if payload['pending'] is not None:
            pending = payload['pending']
            before, action = exact_state(pending['before']), action_from_json(pending['action'])
            if canonical(before) != canonical(bundle.state) or payload['phase'] != 'pending_intent':
                raise ValueError('fast_cache_pending_intent_changed')
            pending_intent = before, action
        elif payload['phase'] != 'observed_successor':
            raise ValueError('fast_cache_missing_pending_intent')
        # Construct and install only after the entire saved payload validates.
        secondary_calls, decisions = payload['secondary_calls'], payload['decisions']
        secondary_plans, timing = payload['secondary_plans'], payload['timing']
        phase, invalid = payload['phase'], payload['invalid']
        offline_pending = (None if pending_intent is None else (
            business_copy(pending_intent[0], offline=True), business_copy(pending_intent[1], offline=True), None))
        request = None
        if offline_pending is not None:
            from agent.live_decision.high_score.contracts import DecisionRequest, CapabilityReport

            saved_request = payload['pending']
            request = DecisionRequest(saved_request['snapshot_id'],
                business_copy(bundle.kernel.plan_type, offline=True),
                business_copy(DecisionKind.SECONDARY if state.pending_choice else DecisionKind.LIVE, offline=True),
                offline_pending[0], CapabilityReport(True, True, True, True, True),
                card_slots=tuple(tuple(pair) for pair in saved_request['card_slots']),
                drink_slots=tuple(tuple(pair) for pair in saved_request['drink_slots']),
                ruleset_id=bundle.kernel.ruleset_id)
            request.validate()
            action = offline_pending[1]
            if action.card_instance_id is not None and (action.card_instance_id, action.source_slot) not in request.card_slots:
                raise ValueError('fast_cache_pending_card_slot_changed')
            if action.drink_id is not None and (action.source_slot, action.drink_id) not in request.drink_slots:
                raise ValueError('fast_cache_pending_drink_slot_changed')
        policy = cls(bundle, exam_context=context, source=history.source,
            planning_seed=payload['planning_seed'], deadline=deadline, model=model, initial_history=history,
            profile=payload['profile'])
        policy.policy._secondary_calls, policy.policy.decisions = secondary_calls, decisions
        if v2_effects is not None:
            policy.policy.v2_effects = v2_effects
        policy.policy.secondary_plans, policy.policy.timing = secondary_plans, timing
        policy._phase, policy._invalid = phase, invalid
        policy._pending_branches, policy._pending = branches, pending_intent
        policy.policy._pending = offline_pending
        if request is not None:
            from agent.live_decision.high_score.decision_box import DecisionBox

            policy._box = DecisionBox(policy.policy, request, pending_action=offline_pending[1])
        return policy
