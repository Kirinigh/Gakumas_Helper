"""Offline UI-network ordinary decisions with an explicitly local choice bridge.

No input controller, game capture, production registration or automatic fallback.
The environment owns randomness and supplies completed transition events.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace

from agent.live_decision.high_score.contracts import ActionKind, DecisionKind

from .gkms_v2_effects import V2EffectState, CatalogV2EffectState, CatalogV2RuleSources
from .gkms_shadow_adapter import TurnPlayHistory, build_shadow_observation

VERSION = 'gkms-uinet-continuous-local-choice-v1'


class ObservedHandOrder:
    """Observed opening order, then only actual draw/move event order.

    The rule environment canonicalizes sets for probability computation; that
    ordering is not the UI network's position feature. Keep this small sidecar
    without changing the rule engine or exposing its random generator.
    """

    def __init__(self, instance_ids):
        self.ids = tuple(instance_ids)

    def request(self, request):
        owned = {card.instance_id for card in request.state.deck.hand}
        if len(self.ids) != len(set(self.ids)) or set(self.ids) != owned:
            raise ValueError('observed_hand_order_state_mismatch')
        return replace(request, card_slots=tuple((iid, i) for i, iid in enumerate(self.ids)))

    def advance(self, before, step):
        order = list(self.ids)
        advanced = step.request.state.turn != before.turn
        ended, entered = False, False
        phases = {'BEFORE_START_OF_TURN', 'START_OF_TURN', 'AFTER_START_OF_TURN', 'TURN', 'EVERY_TURN'}
        for event in step.events:
            if event.event_type == 'END_TURN':
                ended = True
            start = (event.event_type == 'DRAW' and event.source_id == 'system'
                     or event.event_type == 'TRIGGER_SCHEDULED'
                     and (event.detail or '').rsplit(':', 1)[-1] in phases)
            if advanced and ended and not entered and start:
                order.clear()
                entered = True
            if event.event_type == 'EXCHANGE_HAND':
                order.clear()
            elif event.event_type == 'MOVE_CARD':
                source, _, destination = (event.detail or '').partition('->')
                if source == 'HAND' and event.source_id in order:
                    order.remove(event.source_id)
                if destination == 'HAND':
                    order.append(event.source_id)
            elif event.event_type == 'DRAW':
                order.append(event.detail)
            elif event.event_type in ('MOVE_RANDOM_TO_HAND', 'MOVE_SELECTED_TO_HAND'):
                zone, separator, iid = (event.detail or '').partition(':')
                if not separator or not zone.endswith('->HAND'):
                    raise ValueError('observed_hand_move_event_invalid')
                order.append(iid)
        if advanced and not entered:
            order.clear()
        self.ids = tuple(order)
        self.request(step.request)  # Fail on any missing or unexplained arrival.


def strong_policy(kernel, *, planning_seed, deadline):
    """Current explicit strong reference, never v22 or a restored old tree."""
    from agent.live_decision.high_score.hif_mcts_candidate import HifMctsCandidatePolicy
    from agent.live_decision.high_score.hif_strong_reference import configuration, load_baseline

    from .hif_audited_policy import _AuditedMctsPolicy

    settings = configuration()['policy_kwargs']
    if (settings['simulations'], settings['min_root_samples'], settings['window_seconds']) != (256, 4, 300.):
        raise ValueError('strong_reference_budget_changed')
    baseline = load_baseline(kernel)
    return _AuditedMctsPolicy(HifMctsCandidatePolicy(
        kernel, baseline, planning_seed=planning_seed, deadline=deadline, **settings))


def verify_model_output(output, observation):
    """Keep the original deterministic argmax; reject malformed model output."""
    if not isinstance(output, dict):
        raise ValueError('network_output_not_mapping')
    probabilities = output.get('probs')
    legal = observation['legal']
    if (not isinstance(probabilities, (tuple, list)) or len(probabilities) != len(legal)
            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities)
            or not math.isclose(sum(probabilities), 1., rel_tol=0., abs_tol=1e-5)):
        raise ValueError('network_probabilities_invalid')
    chosen = max(range(len(probabilities)), key=probabilities.__getitem__)
    if output.get('action') != legal[chosen]:
        raise ValueError('network_action_differs_from_original_argmax')
    return output['action']


def _tower_legal_projection(projection, request, legal):
    """Give the model the same physical options the live host can execute."""
    positions = dict(request.card_slots)
    valid = set(legal)
    bindings = tuple(binding for binding in projection.bindings
        if (replace(binding.action, source_slot=positions[binding.action.card_instance_id])
            if binding.action.kind is ActionKind.PLAY_CARD else binding.action) in valid)
    if not bindings:
        raise ValueError('tower_no_model_action_matches_visible_legal_actions')
    playable = {binding.token[1] for binding in bindings if binding.token[0] == 'play'}
    observation = dict(projection.observation, legal=[list(binding.token) for binding in bindings],
        hand=[dict(card, p=int(index in playable)) for index, card in enumerate(projection.observation['hand'])])
    return replace(projection, bindings=bindings, observation=observation)


class GkmsContinuousPolicy:
    """One episode; ordinary decisions never use the strong-reference chooser."""

    def __init__(self, kernel, model, context, initial, *, source, planning_seed,
                 deadline, secondary_factory=strong_policy, opening_hand_order=None,
                 observation_builder=None, use_observed_hand_order=False, v2_sources=None):
        self.kernel, self.model, self.context = kernel, model, context
        self.observation_builder = observation_builder
        self.use_observed_hand_order = use_observed_hand_order
        self.v2_sources = v2_sources
        effect_state = CatalogV2EffectState if isinstance(v2_sources, CatalogV2RuleSources) else V2EffectState
        self.v2_effects = effect_state.initial(initial) if v2_sources is not None else None
        self.history = TurnPlayHistory(initial, (), source)
        self.hand_order = ObservedHandOrder(opening_hand_order if opening_hand_order is not None
                                           else (c.instance_id for c in initial.deck.hand))
        self.planning_seed, self.deadline = planning_seed, deadline
        self.secondary_factory = secondary_factory
        self.decisions, self.secondary_plans = [], []
        self._pending = None
        self._secondary_calls = 0
        self.timing = dict(mapping_ms=0., inference_ms=0., local_choice_ms=0.,
                           binding_ms=0., history_ms=0., complete_decision_ms=0.)

    def choose(self, request, legal):
        started = time.perf_counter()
        if self._pending is not None:
            raise ValueError('network_previous_action_not_committed')
        if started >= self.deadline:
            raise TimeoutError('continuous_policy_batch_timeout')
        self.history.validate(request.state)
        entry = dict(index=len(self.decisions), turn=request.state.turn,
                     turns_remaining=request.state.turns_remaining,
                     decision_kind=request.decision_kind.value, status='started')
        self.decisions.append(entry)
        bridge = None
        try:
            if request.decision_kind is DecisionKind.SECONDARY:
                if request.state.pending_choice is None:
                    raise ValueError('secondary_without_pending_choice')
                entry['source'] = 'local_strong_reference_secondary'
                local_start = time.perf_counter()
                self._secondary_calls += 1
                # A declared new local-choice window, not recovery of a prior
                # incomplete window. Environment seeds are never consulted.
                seed = self.planning_seed + 1000000 + self._secondary_calls
                bridge = self.secondary_factory(self.kernel, planning_seed=seed,
                    deadline=min(self.deadline, local_start + 300.))
                try:
                    action = bridge.choose(request, legal)
                finally:
                    elapsed = (time.perf_counter() - local_start) * 1000
                    entry['local_choice_ms'] = elapsed
                    self.timing['local_choice_ms'] += elapsed
                    plans = getattr(bridge, 'plans', ())
                    self.secondary_plans.extend(plans)
                    entry['secondary_plan_count'] = len(plans)
                    entry['secondary_planning_seed'] = seed
                if action not in legal:
                    raise ValueError('secondary_proposal_not_locally_legal')
            else:
                entry['source'] = 'gkms_uinet_original_argmax'
                stage = time.perf_counter()
                observed_request = self.hand_order.request(request)
                entry['observed_card_slots'] = observed_request.card_slots
                build_observation = self.observation_builder or build_shadow_observation
                options = {} if self.v2_effects is None else {'effects': self.v2_effects}
                projection = build_observation(observed_request, self.kernel, self.context, self.history, **options)
                elapsed = (time.perf_counter() - stage) * 1000
                self.timing['mapping_ms'] += elapsed
                entry.update(mapping_ms=elapsed, mapping_status=projection.status,
                    mapping_reasons=list(projection.reason_codes),
                    compression_notes=list(projection.compression_notes),
                    compression_evidence=getattr(projection, 'compression_evidence', ()))
                if projection.status != 'supported':
                    raise ValueError('ordinary_state_mapping_incomplete:' + '|'.join(projection.reason_codes))
                if getattr(self.context, 'use_host_legality', False):
                    projection = _tower_legal_projection(projection, request, legal)
                entry['observation'] = projection.observation
                stage = time.perf_counter()
                output = self.model.decide(projection.observation)
                elapsed = (time.perf_counter() - stage) * 1000
                self.timing['inference_ms'] += elapsed
                entry.update(inference_ms=elapsed, model_output=output)
                token = verify_model_output(output, projection.observation)
                stage = time.perf_counter()
                action = projection.bind_action(token, request=observed_request, kernel=self.kernel)
                entry['observed_action'] = asdict(action)
                if action.kind is ActionKind.PLAY_CARD:
                    action = replace(action, source_slot=dict(request.card_slots)[action.card_instance_id])
                elapsed = (time.perf_counter() - stage) * 1000
                self.timing['binding_ms'] += elapsed
                entry['binding_ms'] = elapsed
            if time.perf_counter() >= self.deadline:
                raise TimeoutError('continuous_policy_batch_timeout')
            self._pending = (request.state, action, bridge)
            entry.update(status='selected', action=asdict(action))
            return action
        except Exception as error:
            entry.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            self.timing['complete_decision_ms'] += elapsed
            entry['complete_decision_ms'] = elapsed

    def observe_transition(self, before, action, step):
        """Only the environment's actual completed step advances played history."""
        started = time.perf_counter()
        if self._pending is None or self._pending[:2] != (before, action):
            raise ValueError('network_commit_does_not_match_selected_action')
        bridge = self._pending[2]
        if self.use_observed_hand_order:
            after = step.request
            if after.decision_kind is DecisionKind.LIVE and not step.terminated:
                self.hand_order.ids = tuple(iid for iid, _ in sorted(after.card_slots, key=lambda pair: pair[1]))
            else:
                self.hand_order.ids = tuple(card.instance_id for card in after.state.deck.hand)
        else:
            self.hand_order.advance(before, step)
        if hasattr(self.history, 'advance_step'):
            self.history = self.history.advance_step(action, step)
        else:
            self.history = self.history.advance(action, step.request.state, step.events)
        if bridge is not None:
            bridge.committed(before, action, step.request.state)
        self.v2_effects = self.effect_state_after(before, action, step.request.state, step.events)
        self._pending = None
        self.decisions[-1]['status'] = 'committed'
        self.timing['history_ms'] += (time.perf_counter() - started) * 1000

    def effect_state_after(self, before, action, after, events):
        if self.v2_effects is None:
            return None
        return self.v2_effects.advance(before, action, after, events, self.v2_sources)
