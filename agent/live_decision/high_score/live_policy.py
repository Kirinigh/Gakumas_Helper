"""Bind a complete opening to fresh the Live observer card-selection observations."""

import copy
import json
import math
import time
from dataclasses import asdict, replace

from .hif import HIF_POLICY_PROFILE, build_hif_policy
from .kernel import EffectKind, CardCostKind
from .contracts import PlanType, ActionKind, GameAction, DecisionKind, DecisionRequest, CapabilityReport, EffectSourceKind
from ..contracts import LiveAction, LiveStatus, DecisionProposal


def status_only_resource_effects(effects):
    """Status changes, including their schedules, cannot directly pay or score."""
    statuses = {EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION}
    return bool(effects) and all(
        effect.kind in statuses or effect.kind is EffectKind.SCHEDULE
        and bool(effect.scheduled_effects)
        and all(nested.kind in statuses for nested in effect.scheduled_effects)
        for effect in effects)


def generated_draw_resource_effects(effects):
    """Either declared status/draw family leaves HP, shield and score unchanged."""
    kinds = {effect.kind for effect in effects}
    shared = {EffectKind.EXTRA_ACTION, EffectKind.DRAW, EffectKind.ADD_CARD_TO_DECK}
    return any(kinds == shared | {status}
               for status in (EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION))


def hand_exchange_resource_effects(definition):
    """Plain hand exchange and status/actions leave card-only HUD resources alone."""
    from .kernel import EffectSpec

    if (not any(effect.kind is EffectKind.EXCHANGE_HAND for effect in definition.effects)
            or definition.cost_kind is not CardCostKind.NORMAL or definition.persistent_effects):
        return False
    exchange = EffectSpec(EffectKind.EXCHANGE_HAND)
    return (definition.effects.count(exchange) == 1 and all(
        effect == exchange or effect.kind in {
            EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION, EffectKind.EXTRA_ACTION}
        and type(effect.amount) in (int, float) and math.isfinite(effect.amount)
        and effect.amount > 0 and float(effect.amount).is_integer()
        and effect == EffectSpec(effect.kind, amount=effect.amount)
        for effect in definition.effects))


def _ordinary_composite_resource_effects(definition):
    """Exact mechanisms present in the bound 1793 deck, without card-ID rules."""
    from .kernel import EffectSpec, ConditionKind, EffectCondition
    from .contracts import SchedulePhase

    if definition.cost_kind is not CardCostKind.NORMAL or definition.persistent_effects:
        return False
    effects = definition.effects
    kinds = tuple(effect.kind for effect in effects)
    focus, good, extra = EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION, EffectKind.EXTRA_ACTION

    def positive(value, *, integer=True):
        return (type(value) in (int, float) and math.isfinite(value) and value > 0
            and (not integer or float(value).is_integer()))

    def plain(effect):
        return positive(effect.amount) and effect == EffectSpec(effect.kind, amount=effect.amount)

    if kinds == (focus, good, EffectKind.HALF_COST_TURNS):
        return all(plain(effect) for effect in effects)
    generated = (
        (focus, extra, EffectKind.DRAW, EffectKind.ADD_CARD_TO_DECK),
        (good, extra, EffectKind.DRAW, EffectKind.ADD_CARD_TO_DECK),
        (good, focus, extra, EffectKind.ADD_CARD_TO_DECK),
    )
    if kinds in generated:
        added = effects[-1]
        return (all(plain(effect) for effect in effects[:-1])
            and isinstance(added.card_id, str) and bool(added.card_id)
            and added == EffectSpec(EffectKind.ADD_CARD_TO_DECK, card_id=added.card_id))
    if kinds in ((focus, EffectKind.SCHEDULE), (good, EffectKind.SCHEDULE)):
        direct, schedule = effects
        if not plain(direct) or len(schedule.scheduled_effects) != 1:
            return False
        child = schedule.scheduled_effects[0]
        if (kinds[0] is focus and child.kind is focus and plain(child)
                and schedule == EffectSpec(EffectKind.SCHEDULE, delay_turns=0,
                    schedule_phase=SchedulePhase.CARD_USED, schedule_limit=None,
                    trigger_card_type='active', scheduled_effects=(child,))):
            return True
        phase_delay = {SchedulePhase.END_OF_TURN: 0, SchedulePhase.START_OF_TURN: 1}
        return (child.kind is EffectKind.SCORE and plain(child)
            and schedule.schedule_phase in phase_delay
            and type(schedule.schedule_limit) is int and schedule.schedule_limit > 0
            and schedule == EffectSpec(EffectKind.SCHEDULE,
                delay_turns=phase_delay[schedule.schedule_phase], schedule_phase=schedule.schedule_phase,
                schedule_limit=schedule.schedule_limit, scheduled_effects=(child,)))
    if kinds not in ((focus, extra, EffectKind.SCORE),
                     (focus, EffectKind.CONCENTRATION_EFFECT_BUFF, EffectKind.SCORE)):
        return False
    direct, preparation, score = effects
    condition = score.condition
    expected_condition = (ConditionKind.REMOVED_CARD_COUNT_MIN if kinds[1] is extra
        else ConditionKind.CONCENTRATION_MIN)
    if (not plain(direct) or condition is None or not positive(condition.threshold)
            or condition != EffectCondition(expected_condition, threshold=condition.threshold)
            or not positive(score.amount) or not positive(score.score_concentration_multiplier, integer=False)
            or score != EffectSpec(EffectKind.SCORE, amount=score.amount, condition=condition,
                score_concentration_multiplier=score.score_concentration_multiplier)):
        return False
    if kinds[1] is extra:
        return plain(preparation)
    return (positive(preparation.amount, integer=False) and type(preparation.duration) is int
        and preparation.duration > 0 and preparation == EffectSpec(EffectKind.CONCENTRATION_EFFECT_BUFF,
            amount=preparation.amount, duration=preparation.duration))


def _counted_good_condition_score_schedule(effect):
    """The observed after-card counter defers scoring beyond its registration."""
    from .kernel import EffectSpec, ConditionKind, EffectCondition
    from .contracts import SchedulePhase

    active = EffectCondition(ConditionKind.GOOD_CONDITION_MAX, threshold=20)
    trigger = EffectCondition(ConditionKind.ALL, all_of=(active,
        EffectCondition(ConditionKind.EFFECT_COUNTER_MIN, threshold=2),
        EffectCondition(ConditionKind.EFFECT_COUNTER_MOD_EQ, modulus=2, remainder=0)))
    return any(effect == EffectSpec(EffectKind.SCHEDULE, delay_turns=0,
        schedule_phase=SchedulePhase.AFTER_CARD_USED, schedule_limit=None,
        schedule_counter_condition=active, scheduled_effects=(
            EffectSpec(EffectKind.SCORE, amount=score, condition=trigger,
                score_concentration_multiplier=4),
            EffectSpec(EffectKind.GOOD_CONDITION, amount=-3, condition=trigger)))
        for score in (6, 7))


class BoundHifOpeningPolicy:
    """Opening-only intent adapter. The caller must still prove A/B input gates.

    No serialized log or caller-supplied card id enters this adapter. Its bundle
    comes directly from the same native observation context. The first binding
    computes the policy; subsequent A/B refreshes must match the complete base.
    """

    provider_name = 'task275-hif-bound-opening'
    provider_version = HIF_POLICY_PROFILE

    def __init__(self, bundle, *, action_order, decision_policy=None, hold_ready=False):
        self.bundle = bundle
        self.action_order = tuple(action_order)
        self.response = None
        self.bound_snapshot = None
        self.observed_normal_score_preview = None
        self._normal_score_projection = None
        self.selected_single_plus_estimate = None
        self._single_plus_state = None
        self._single_plus_projected_state = None
        self.selected_skill_support = None
        self._support_definition = None
        self._support_state = None
        self.decision_policy = decision_policy
        self.hold_ready = hold_ready
        self._hand_support_receipt_state = None
        self.ordinary_plus_projection = None

    def bind_verified_hand_support(self, receipt):
        """Add a completely rechecked held-card fact, never old input evidence.

        The caller restores the receipt's exact public state before this call.
        Cancellation, process continuity and fresh physical binding remain the
        host's responsibility. Multiple receipts from that same unchanged state
        may be added; facts never modify permanent inventory or carry slots.
        """
        from .hif_support_planning import verified_hand_support_fact, uses_hand_support_for_strategy

        current = self.bundle.state
        original = self._hand_support_receipt_state or current
        if replace(current, hand_support=original.hand_support) != original:
            raise ValueError('HIF hand support receipt public state changed')
        source_bundle = replace(self.bundle, state=original)
        fact = verified_hand_support_fact(source_bundle, receipt)
        existing = [value for value in current.hand_support
                    if value.instance_id == fact.instance_id]
        if existing:
            if existing != [fact]:
                raise ValueError('HIF hand support conflicts with the held entity')
            return fact
        if not uses_hand_support_for_strategy(fact):
            return fact
        after = replace(current, hand_support=tuple(sorted(
            (*current.hand_support, fact), key=lambda value: value.instance_id)))
        after.validate()
        self.bundle.kernel._validate_registered_state(after, plan_type=self.bundle.kernel.plan_type)
        if self.decision_policy is not None:
            update = getattr(self.decision_policy, 'bind_hand_support_context', None)
            if not callable(update):
                raise ValueError('HIF decision policy cannot revoke its old support context')
            update(run_id=self.bundle.run_id, revision=self.bundle.revision,
                   before=current, after=after)
        self._hand_support_receipt_state = original
        self.bundle = replace(self.bundle, state=after)
        self.response = self.bound_snapshot = self.observed_normal_score_preview = None
        self.selected_single_plus_estimate = self._single_plus_state = None
        self.selected_skill_support = self._support_definition = self._support_state = None
        return fact

    def bind_selected_skill_support(self, receipt):
        """Bind complete selected-detail evidence to this one planned play."""
        from .skill_support import supported_definition
        from ..hif_skill_support import verify_selected_skill_support_observations

        self.selected_skill_support = self._support_definition = self._support_state = None
        self.selected_single_plus_estimate = self._single_plus_state = None
        action = self.response.action if self.response else None
        if (action is None or action.kind is not ActionKind.PLAY_CARD
                or not isinstance(receipt, dict) or receipt.get('accepted') is not True
                or receipt.get('effects_verified') is not True
                or receipt.get('selected_slot') != action.source_slot
                or receipt.get('card_instance_id') != action.card_instance_id
                or receipt.get('run_id') != self.bundle.run_id or receipt.get('revision') != self.bundle.revision):
            raise ValueError('HIF selected support differs from the planned action or epoch')
        verified = verify_selected_skill_support_observations(receipt.get('observations', ()),
            bundle=self.bundle, selected_slot=action.source_slot,
            selection_evidence=receipt.get('selection_evidence', {}))
        if (verified.get('accepted') is not True
                or json.dumps(verified, sort_keys=True) != json.dumps(receipt, sort_keys=True)):
            raise ValueError('HIF selected support differs from its independent complete observations')
        ref = self.bundle.state.deck.hand[action.source_slot]
        definition = supported_definition(self.bundle.kernel.cards[ref.card_id],
            enhancement_count=verified.get('enhancement_count'))
        if (verified.get('printed_cost') != definition.stamina_cost
                or verified.get('cost_kind') != definition.cost_kind.value):
            raise ValueError('HIF selected support cost differs from the fixed definition')
        self.selected_skill_support = copy.deepcopy(verified)
        self._support_definition = definition
        self._support_state = self.bundle.state
        self.observed_normal_score_preview = None
        return True

    def bind_selected_single_plus_estimate(self, receipt, *, require_submitted=False):
        """Bind D-047 title evidence to this action, never its unknown effects."""
        from ..hif_single_plus import verify_selected_single_plus_estimate

        self.selected_single_plus_estimate = self._single_plus_state = None
        self.observed_normal_score_preview = None
        if receipt is None:
            return False
        if self.selected_skill_support is not None or type(require_submitted) is not bool:
            raise ValueError('HIF single-plus estimate conflicts with exact support')
        action = self.response.action if self.response else None
        verified = verify_selected_single_plus_estimate(receipt, bundle=self.bundle,
            action=action, require_submitted=require_submitted)
        projected = None
        if verified.get('ordinary_plus_projection') is True:
            from .contracts import VerifiedHandSupport

            state = self.bundle.state
            ref = state.deck.hand[action.source_slot]
            fact = VerifiedHandSupport(ref.instance_id, ref.card_id, state.turn, 1,
                                       projection_kind='ordinary_plus_estimate')
            existing = [value for value in state.hand_support if value.instance_id == ref.instance_id]
            if existing and existing != [fact]:
                raise ValueError('HIF selected ordinary-plus conflicts with current held support')
            projected = state if existing else replace(state, hand_support=tuple(sorted(
                (*state.hand_support, fact), key=lambda value: value.instance_id)))
            projected.validate()
            self.bundle.kernel._validate_registered_state(projected, plan_type=self.bundle.kernel.plan_type)
        self.selected_single_plus_estimate = copy.deepcopy(verified)
        self._single_plus_state = self.bundle.state
        self._single_plus_projected_state = projected
        return True

    def _selected_model_state(self):
        """A selected title changes only this play's model, not its planned intent."""
        receipt = self._current_single_plus_estimate()
        if receipt is not None and receipt.get('ordinary_plus_projection') is True:
            if self._single_plus_projected_state is None:
                raise ValueError('HIF selected ordinary-plus projection is missing')
            return self._single_plus_projected_state
        return self.bundle.state

    def _current_single_plus_estimate(self):
        receipt = self.selected_single_plus_estimate
        if receipt is None:
            return None
        action = self.response.action if self.response else None
        if (self.bundle.state != self._single_plus_state or action is None
                or action.kind is not ActionKind.PLAY_CARD
                or action.card_instance_id != receipt['card_instance_id']
                or action.source_slot != receipt['selected_slot']
                or (self.bundle.run_id, self.bundle.revision) != (receipt['run_id'], receipt['revision'])
                or self.selected_skill_support is not None):
            raise ValueError('HIF single-plus estimate cannot carry to another state or action')
        return receipt

    def _current_support_definition(self):
        if self.selected_skill_support is None:
            return None
        receipt = self.selected_skill_support
        action = self.response.action if self.response else None
        if (self.bundle.state != self._support_state or action is None
                or action.kind is not ActionKind.PLAY_CARD
                or action.card_instance_id != receipt['card_instance_id']
                or action.source_slot != receipt['selected_slot']
                or (self.bundle.run_id, self.bundle.revision) != (receipt['run_id'], receipt['revision'])):
            raise ValueError('HIF selected support cannot carry to another state or action')
        return self._support_definition

    def _current_genki_support_definition(self):
        """A revealed tier may predict B; only independent B can commit it."""
        receipt = self._current_single_plus_estimate()
        if receipt is None or receipt.get('ordinary_plus_projection') is True:
            return None
        from .support_master import genki_only_support

        state = self._selected_model_state()
        definition = self.bundle.kernel.card_definition(state, state.deck.hand[self.response.action.source_slot])
        try:
            return genki_only_support(definition, receipt['plus_count'])
        except (ValueError, KeyError):
            return None

    def bind_visible_hand(self, live, *, identity_events=None):
        """Use the already verified visible form as a current-turn estimate.

        The observer owns title/instance identity. This adds no input authority
        or claim about random extra-plus effects, and never rewrites the deck.
        """
        from .contracts import VerifiedHandSupport
        from .hif_support_planning import observed_plus_hand, ordinary_plus_definition

        if self.response is None:
            self.ordinary_plus_projection = None
        if len(live.hand) != len(self.bundle.state.deck.hand):
            raise ValueError('HIF visible hand binding is not at initial observation')
        before = self.bundle.state
        if (live.turn, live.turns_remaining) != (before.turn, before.turns_remaining):
            raise ValueError('HIF visible hand binding is not in the current turn')
        inventory_ids = {ref.instance_id for ref in before.deck.all_cards}
        if any(receipt.get('card_instance_id') not in inventory_ids
               for receipt in self.bundle.opening_skill_support):
            raise ValueError('HIF opening support differs from the permanent inventory')
        original_live = live
        proof_events = ()
        if identity_events is not None:
            live, proof_events = observed_plus_hand(self.bundle, live, identity_events)
        additions = []
        for observed, ref in zip(live.hand, self.bundle.state.deck.hand, strict=True):
            current = self.bundle.kernel.cards[ref.card_id]
            visible = self.bundle.kernel.cards.get(observed.card_id)
            if (not observed.identity_accepted or observed.box is None or visible is None
                    or visible.base_card_id != current.base_card_id):
                raise ValueError('HIF visible hand family is not verified')
            if observed.card_id == (current.canonical_card_id or current.card_id):
                continue
            # Revalidate any existing opening receipt; it cannot silently
            # acquire complete-effect authority from this catalog estimate.
            self._opening_support_visible_id(ref)
            if observed.card_id == self._hand_support_visible_id(ref):
                continue
            if current.canonical_card_id and current.canonical_card_id != current.card_id:
                raise ValueError('HIF customized visible variant requires its compiled definition')
            if (ref.upgraded or observed.upgraded is not True
                    or observed.card_id != current.upgrade_card_id):
                raise ValueError('HIF visible enhancement has no current projection; confirmed deck is unchanged')
            ordinary_plus_definition(self.bundle.kernel.cards, current)
            additions.append(VerifiedHandSupport(ref.instance_id, ref.card_id, before.turn, 1,
                                                projection_kind='ordinary_plus_estimate'))
        if additions:
            if self.response is not None:
                raise ValueError('HIF new visible hand support invalidates the evaluated intention')
            after = replace(before, hand_support=tuple(sorted(
                (*before.hand_support, *additions), key=lambda fact: fact.instance_id)))
            after.validate()
            self.bundle.kernel._validate_registered_state(after, plan_type=self.bundle.kernel.plan_type)
            if self.decision_policy is not None:
                update = getattr(self.decision_policy, 'bind_hand_support_context', None)
                if not callable(update):
                    raise ValueError('HIF decision policy cannot revoke its old support context')
                update(run_id=self.bundle.run_id, revision=self.bundle.revision, before=before, after=after)
            self.bundle = replace(self.bundle, state=after)
            self.bound_snapshot = self.observed_normal_score_preview = None
            self.selected_single_plus_estimate = self._single_plus_state = None
            self.selected_skill_support = self._support_definition = self._support_state = None
            if identity_events is not None:
                from .contracts import _jsonable

                self.ordinary_plus_projection = dict(event='live_hand_projection',
                    source='hif_current_turn_ordinary_plus_v1', accepted=True,
                    run_id=self.bundle.run_id, revision=self.bundle.revision,
                    bound_state=_jsonable(asdict(before)), inventory_keys=_jsonable(self.bundle.inventory_keys),
                    visible_state=original_live.to_dict(), identity_events=copy.deepcopy(proof_events),
                    projected_facts=_jsonable([asdict(fact) for fact in additions]),
                    effects_verified=False, executed=False, execution_submit_count=0, total_input_click_count=0)
        return self.bundle

    def _hand_support_visible_id(self, ref):
        """An already verified held fact may explain a visible ordinary plus."""
        state, kernel = self.bundle.state, self.bundle.kernel
        if ref.upgraded or ref not in state.deck.hand or not any(
                (fact.instance_id, fact.card_id, fact.turn) == (ref.instance_id, ref.card_id, state.turn)
                for fact in state.hand_support):
            return None
        ordinary = kernel.cards[ref.card_id]
        effective = kernel.card_definition(state, ref)
        visible = kernel.cards.get(ordinary.upgrade_card_id)
        if (effective == ordinary or visible is None or visible.base_card_id != ordinary.base_card_id):
            return None
        return visible.card_id

    def _opening_support_visible_id(self, ref):
        from ..opening_support import verify_opening_skill_support_receipt

        receipts = [row for row in self.bundle.opening_skill_support
                    if row.get('card_instance_id') == ref.instance_id]
        if not receipts or self.bundle.state.turn != 1:
            return None
        if len(receipts) != 1 or ref not in self.bundle.state.deck.hand:
            raise ValueError('HIF opening support instance is ambiguous')
        origin = self.bundle.opening_support_origin
        if (not isinstance(origin, tuple) or len(origin) != 2
                or not isinstance(origin[0], str) or not origin[0]
                or type(origin[1]) is not int):
            raise ValueError('HIF opening support lacks its verified original epoch')
        receipt = verify_opening_skill_support_receipt(receipts[0],
            run_id=origin[0], revision=origin[1],
            card_instance_id=ref.instance_id, card_id=ref.card_id,
            slot=receipts[0].get('slot'))
        if (ref.upgraded or (ref.instance_id, ref.card_id, ()) not in self.bundle.inventory_keys):
            raise ValueError('HIF opening support differs from the permanent inventory')
        return receipt['visible_card_id']

    def decide_empty_hand(self):
        state = self.bundle.state
        if state.deck.hand or (state.drinks and self.bundle.drink_slots is None):
            raise ValueError('HIF empty-hand decision requires complete physical inventory')
        request = DecisionRequest('hif-empty-hand', self.bundle.kernel.plan_type, DecisionKind.LIVE, state,
            CapabilityReport(True, True, True, True, True),
            drink_slots=self.bundle.drink_slots or (), ruleset_id=self.bundle.kernel.ruleset_id)
        self.response = (self.decision_policy or build_hif_policy(self.bundle.kernel)).decide(request)
        return self.response

    def prepare_state(self, live, fields):
        self.bound_snapshot = None
        try:
            return self._prepare_state(live, fields)
        except (ValueError, KeyError, TypeError):
            # A rejected new observation revokes the old intent too. The next
            # accepted native binding must recompute, never reuse stale input.
            self.response = None
            self.observed_normal_score_preview = None
            self.selected_single_plus_estimate = self._single_plus_state = None
            self.selected_skill_support = self._support_definition = self._support_state = None
            raise

    def bind_generated_card_resource_preview(self, transaction, score):
        """D-047 estimates effects; independently read B owns dynamic payment.

        These narrow families add/draw cards and a status without directly
        changing score or shield. Keep every draw outcome and after-card item
        effect; never replace a resource from the later settlement itself.
        """
        state = self.bundle.state
        action = self.response.action if self.response else None
        slot = transaction.get('selected_slot')
        if (action is None or action.kind is not ActionKind.PLAY_CARD or type(slot) is not int
                or not 0 <= slot < len(state.deck.hand) or action.source_slot != slot
                or action.card_instance_id != state.deck.hand[slot].instance_id):
            return False
        definition = self.bundle.kernel.card_definition(state, state.deck.hand[slot])
        status_only = status_only_resource_effects(definition.effects)
        composite = (_ordinary_composite_resource_effects(definition)
            and not any(effect.kind is EffectKind.SCORE for effect in definition.effects))
        if (definition.cost_kind is not CardCostKind.NORMAL
                or not status_only and not generated_draw_resource_effects(definition.effects)
                and not hand_exchange_resource_effects(definition) and not composite):
            return False
        if transaction.get('card_effect_preview') is not None:
            # All already admitted NORMAL families consume the same independently
            # checked model contract. Cost OCR corroborates that known cost;
            # only legacy observed-payment receipts need its .95 threshold.
            # Preserve each family's existing score contract: status/draw uses
            # the accepted score reader (including its strict dedicated zero),
            # while composite effects additionally require .95 confidence.
            contract = self.selected_resource_preview(slot)
            confidence = transaction.get('displayed_effective_cost_confidence')
            score_confidence = score.get('confidence')
            frames = transaction.get('frame_ids', ())
            if (contract is None or transaction['card_effect_preview'] != contract
                    or transaction.get('accepted') is not True
                    or len(frames) != 3 or len(set(frames)) != 3
                    or any(not isinstance(frame, str) or not frame for frame in frames)
                    or transaction.get('card_id', contract['card_id']) != contract['card_id']
                    or transaction.get('normalized_stamina') != state.stamina
                    or transaction.get('normalized_shield') != state.genki
                    or transaction.get('displayed_effective_cost')
                        != contract.get('displayed_card_cost', contract['effective_cost'])
                    or transaction.get('effective_payment_cost') != contract['effective_cost']
                    or type(confidence) not in (int, float) or not 0 <= confidence <= 1
                    or any(transaction.get('conservation', {}).get(key) is not True
                        for key in ('declared_cost', 'modeled_card_preview', 'stamina', 'shield'))
                    or transaction.get('preview_stamina') != contract['preview_stamina']
                    or transaction.get('preview_shield') != contract['preview_shield']
                    or score.get('accepted') is not True or score.get('value') != contract['preview_score']
                    or contract['preview_score'] != state.score
                    or composite and (type(score_confidence) not in (int, float)
                                      or not .95 <= score_confidence <= 1)):
                raise ValueError('HIF composite resource preview is not independently verified')
            self.observed_normal_score_preview = dict(score=state.score,
                stamina=contract['preview_stamina'], genki=contract['preview_shield'])
            self._normal_score_projection = None
            return True
        cost = transaction.get('displayed_effective_cost')
        confidence = transaction.get('displayed_effective_cost_confidence')
        conservation = transaction.get('conservation',{})
        if (transaction.get('accepted') is not True or len(set(transaction.get('frame_ids',()))) != 3
                or any(conservation.get(k) is not True for k in ('effective_cost','stamina','shield'))
                or transaction.get('normalized_stamina') != state.stamina
                or transaction.get('normalized_shield') != state.genki
                or type(cost) is not int or not 0 <= cost <= state.stamina+state.genki
                or type(confidence) not in (int,float) or not .95 <= confidence <= 1
                or transaction.get('preview_stamina') != state.stamina-max(cost-state.genki,0)
                or transaction.get('preview_shield') != max(state.genki-cost,0)
                or score.get('accepted') is not True or score.get('value') != state.score):
            raise ValueError('HIF generated card resource preview is not independently verified')
        self.observed_normal_score_preview = dict(score=state.score,
            stamina=transaction['preview_stamina'],genki=transaction['preview_shield'])
        self._normal_score_projection = None
        return True

    def bind_normal_card_score_preview(self, transaction, score):
        """Share the selected NORMAL-card forecast with native record replay.

        The caller supplies the original accepted selected observation, never
        a later settlement. The catalog and complete pre-action state stay
        unchanged. A fixed pure-score tier can instead be identified by its
        complete title and independent card-only HUD, then replay all triggers.
        """
        self._normal_score_projection = None
        state = self.bundle.state
        action = self.response.action if self.response else None
        slot = transaction.get('selected_slot')
        if action is None or action.kind is not ActionKind.PLAY_CARD:
            return False
        if (type(action.source_slot) is not int or not 0 <= action.source_slot < len(state.deck.hand)
                or action.card_instance_id != state.deck.hand[action.source_slot].instance_id):
            raise ValueError('HIF normal card score preview has no bound physical action')
        definition = self.bundle.kernel.card_definition(state, state.deck.hand[action.source_slot])
        receipt = self._current_single_plus_estimate()
        genki_support = self._current_genki_support_definition()
        if genki_support is not None:
            return self._bind_pure_score_projection(transaction, score, receipt, definition,
                supported=genki_support)
        if receipt is not None and definition.card_id == '35' and receipt.get('plus_count') != 1:
            self.observed_normal_score_preview = None
            return self._bind_pure_score_projection(transaction, score, receipt, definition)
        if (definition.cost_kind is not CardCostKind.NORMAL or not definition.effects
                or not any(effect.kind is EffectKind.SCORE for effect in definition.effects)
                or any(effect.kind not in (EffectKind.SCORE, EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION)
                       for effect in definition.effects)):
            return False
        if (type(slot) is not int or slot != action.source_slot
                or transaction.get('accepted') is not True or len(set(transaction.get('frame_ids', ()))) != 3
                or score.get('accepted') is not True or type(score.get('value')) is not int or score['value'] < 0
                or transaction.get('normalized_stamina') != state.stamina
                or transaction.get('normalized_shield') != state.genki
                or any(type(transaction.get(key)) is not int or transaction[key] < 0
                       for key in ('preview_stamina', 'preview_shield'))):
            raise ValueError('HIF normal card score preview is not independently verified')
        self.observed_normal_score_preview = dict(score=score['value'],
            stamina=transaction['preview_stamina'], genki=transaction['preview_shield'])
        return True

    def _bind_pure_score_projection(self, transaction, score, receipt, definition, *, supported=None):
        """One current pinned support play, not a claim about unrevealed tiers.

        A pure-score tier or an otherwise unchanged GENKI tier must match
        independent card-only B. Replay the complete transition with that
        definition, preserving counters, repeats, costs and automatic effects.
        """
        from .skill_support import supported_definition, project_supported_card_once

        state, kernel = self.bundle.state, self.bundle.kernel
        action = self.response.action
        ref = state.deck.hand[action.source_slot]
        if supported is None:
            supported = supported_definition(definition, enhancement_count=receipt.get('plus_count'))
        preview = self.selected_resource_preview(action.source_slot)
        frames = transaction.get('frame_ids', ())
        confidence = score.get('confidence')
        cost_confidence = transaction.get('displayed_effective_cost_confidence')
        cost = kernel._effective_stamina_cost(state, ref, supported)
        if (receipt.get('effects_verified') is not False or receipt.get('ordinary_plus_projection') is True
                or preview is None or transaction.get('card_effect_preview') != preview
                or transaction.get('accepted') is not True
                or type(transaction.get('selected_slot')) is not int
                or transaction['selected_slot'] != action.source_slot
                or transaction.get('card_id', preview['card_id']) != preview['card_id']
                or len(frames) != 3 or any(not isinstance(frame, str) or not frame for frame in frames)
                or len(set(frames)) != 3 or set(frames) & set(receipt['frame_ids'])
                or any(transaction.get('conservation', {}).get(name) is not True
                       for name in ('declared_cost', 'modeled_card_preview', 'stamina', 'shield'))
                or transaction.get('normalized_stamina') != state.stamina
                or transaction.get('normalized_shield') != state.genki
                or type(transaction.get('displayed_effective_cost')) is not int
                or transaction['displayed_effective_cost'] != cost
                or type(transaction.get('effective_payment_cost')) is not int
                or transaction['effective_payment_cost'] != cost
                or type(cost_confidence) not in (int, float) or not 0 <= cost_confidence <= 1
                or score.get('accepted') is not True or type(score.get('value')) is not int
                or type(confidence) not in (int, float) or not .95 <= confidence <= 1):
            raise ValueError('HIF pure-score projection lacks independent title and B evidence')
        card_only = replace(self.bundle, state=replace(state,
            items=(), memory_abilities=(), scheduled_effects=()))
        _, projections = project_supported_card_once(card_only, action,
            effective_definition=supported, settle_automatic_turn=False)
        values = dict(score=score['value'], stamina=transaction.get('preview_stamina'),
                      genki=transaction.get('preview_shield'))
        if (len(projections) != 1 or any(type(value) is not int or value < 0
                                       for value in values.values())
                or any(getattr(projections[0], name) != value for name, value in values.items())):
            raise ValueError('HIF pure-score projection differs from the independent card-only HUD')
        self.observed_normal_score_preview = values
        self._normal_score_projection = (state, action, copy.deepcopy(receipt), definition, supported, values.copy())
        return True

    def bind_single_plus_typed_score_preview(self, transaction, score):
        """Use independent B score only for a bound D-047 concentration payment.

        The ordinary-plus model still owns cost, action count and all triggers.
        The observed card-only score enters the existing commit correction;
        neither this forecast nor the receipt changes the complete prestate.
        """
        receipt = self._current_single_plus_estimate()
        if receipt is None or receipt.get('ordinary_plus_projection') is True:
            return False
        state, kernel = self.bundle.state, self.bundle.kernel
        action = self.response.action
        ref = state.deck.hand[action.source_slot]
        definition = kernel.card_definition(state, ref)
        if (definition.cost_kind is not CardCostKind.CONCENTRATION or definition.persistent_effects
                or not any(effect.kind is EffectKind.SCORE for effect in definition.effects)
                or any(effect.kind not in (EffectKind.SCORE, EffectKind.GOOD_CONDITION, EffectKind.EXTRA_ACTION)
                    or effect.condition is not None or effect.value_source is not None
                    for effect in definition.effects)):
            return False
        preview = self.selected_resource_preview(action.source_slot)
        frames = transaction.get('frame_ids', ())
        confidence = score.get('confidence')
        cost_confidence = transaction.get('displayed_effective_cost_confidence')
        if (preview is None or transaction.get('card_effect_preview') != preview
                or transaction.get('accepted') is not True
                or transaction.get('selected_slot') != action.source_slot
                or transaction.get('card_id', preview['card_id']) != preview['card_id']
                or len(frames) != 3 or any(not isinstance(frame, str) or not frame for frame in frames)
                or len(set(frames)) != 3 or set(frames) & set(receipt['frame_ids'])
                or any(transaction.get('conservation', {}).get(name) is not True
                       for name in ('declared_cost', 'modeled_card_preview', 'stamina', 'shield'))
                or transaction.get('normalized_stamina') != state.stamina
                or transaction.get('normalized_shield') != state.genki
                or transaction.get('preview_stamina') != preview['preview_stamina']
                or transaction.get('preview_shield') != preview['preview_shield']
                or type(transaction.get('displayed_effective_cost')) is not int
                or transaction['displayed_effective_cost'] != preview['effective_cost']
                or transaction.get('effective_payment_cost') != preview['effective_cost']
                or type(cost_confidence) not in (int, float) or not 0 <= cost_confidence <= 1
                or not kernel._non_full_power_cost_payable(state, ref, definition)
                or score.get('accepted') is not True or type(score.get('value')) is not int
                or score['value'] < state.score
                or type(confidence) not in (int, float) or not .95 <= confidence <= 1):
            raise ValueError('HIF single-plus typed score preview is not independently verified')
        self.observed_normal_score_preview = dict(score=score['value'],
            stamina=transaction['preview_stamina'], genki=transaction['preview_shield'])
        self._normal_score_projection = None
        return True

    def _bind_typed_single_precision_display(self, transaction, score):
        """Bind 2004's exact float32 HUD boundary for this ordinary play only.

        The decimal kernel remains unchanged. Its paid card-only parameters
        must reproduce both the original forecast and the independently read
        float32 display. Settlement still has to match the actual new score.
        """
        import struct

        from .kernel import EffectSpec

        state, kernel = self.bundle.state, self.bundle.kernel
        action = self.response.action if self.response else None
        if (action is None or action.kind is not ActionKind.PLAY_CARD
                or self._current_single_plus_estimate() is not None
                or self._current_support_definition() is not None
                or state.double_card_effects_remaining or state.nullify_cost_cards):
            return False
        ref = state.deck.hand[action.source_slot]
        definition = kernel.card_definition(state, ref)
        if (definition.cost_kind is not CardCostKind.CONCENTRATION or definition.persistent_effects
                or not definition.effects or definition.effects[0].kind is not EffectKind.SCORE
                or sum(effect.kind is EffectKind.SCORE for effect in definition.effects) != 1
                or ref.score_repeat_bonus
                or any(effect.kind not in {EffectKind.SCORE, EffectKind.GOOD_CONDITION, EffectKind.EXTRA_ACTION}
                    or effect != EffectSpec(effect.kind, amount=effect.amount) for effect in definition.effects)):
            return False
        preview = self.selected_resource_preview(action.source_slot)
        frames = transaction.get('frame_ids', ())
        cost = kernel._effective_stamina_cost(state, ref, definition)
        if (preview is None or not 0 < cost <= state.concentration
                or transaction.get('accepted') is not True
                or transaction.get('selected_slot') != action.source_slot
                or transaction.get('card_effect_preview') != preview
                or len(frames) != 3 or len(set(frames)) != 3
                or any(not isinstance(frame, str) or not frame for frame in frames)
                or transaction.get('normalized_stamina') != state.stamina
                or transaction.get('normalized_shield') != state.genki
                or transaction.get('preview_stamina') != preview['preview_stamina']
                or transaction.get('preview_shield') != preview['preview_shield']
                or transaction.get('displayed_effective_cost') != cost
                or transaction.get('effective_payment_cost') != cost
                or any(transaction.get('conservation', {}).get(key) is not True
                    for key in ('declared_cost', 'modeled_card_preview', 'stamina', 'shield'))
                or score.get('accepted') is not True or type(score.get('value')) is not int
                or type(score.get('confidence')) not in (int, float)
                or not .95 <= score['confidence'] <= 1
                or type(transaction.get('displayed_effective_cost_confidence')) not in (int, float)
                or not 0 <= transaction['displayed_effective_cost_confidence'] <= 1):
            return False
        paid = replace(state, concentration=state.concentration-cost)
        effect = definition.effects[0]
        exact = kernel.score_effect_gain(paid, effect, ref.instance_id)
        parameters = kernel.score_effect_gain(replace(paid, score_multiplier=1), effect, ref.instance_id)
        def single(value):
            return struct.unpack('f', struct.pack('f', value))[0]
        displayed = math.ceil(single(single(parameters) * single(state.score_multiplier)))
        if (preview['preview_score'] != state.score+exact or abs(displayed-exact) != 1
                or score['value'] != state.score+displayed):
            return False
        self.observed_normal_score_preview = dict(score=score['value'],
            stamina=preview['preview_stamina'], genki=preview['preview_shield'])
        self._normal_score_projection = None
        return True

    def _prepare_state(self, live, fields):
        live.validate()
        state, kernel = self._selected_model_state(), self.bundle.kernel
        transaction = fields.get('resource_transaction', {})
        if transaction.get('accepted') is True:
            self.bind_generated_card_resource_preview(transaction,fields.get('score',{}))
            if (transaction.get('selected_slot') is not None
                    and fields.get('score', {}).get('value') == live.score):
                try:
                    normalized = (self.bind_normal_card_score_preview(transaction, fields.get('score', {}))
                        or self.bind_single_plus_typed_score_preview(transaction, fields.get('score', {}))
                        or self._bind_typed_single_precision_display(transaction, fields.get('score', {})))
                except ValueError as error:
                    if live.score != state.score:
                        raise ValueError('complete HIF state no longer matches Live: score') from error
                    raise
                if normalized:
                    # The selected HUD is a forecast, not a changed preinput score.
                    live = replace(live, score=state.score)
        if (live.score != state.score and transaction.get('accepted') is True
                and transaction.get('selected_slot') is not None and self.response is not None):
            preview = self.selected_resource_preview(transaction['selected_slot'])
            if (preview is not None and transaction.get('card_effect_preview') == preview
                    and live.score == preview['preview_score']):
                # The selected HUD displays the prospective total. Keep the
                # actual observed fields as evidence, but decide on preinput state.
                live = replace(live,score=state.score)
        for name in ('turn','turns_remaining','stamina','max_stamina','score','score_multiplier'):
            if getattr(live,name) != getattr(state,name):
                raise ValueError('complete HIF state no longer matches Live: '+name)
        if (live.action_order != self.action_order or live.contradictory_fields
                or fields.get('shield',{}).get('accepted') is not True
                or fields['shield']['value'] != state.genki):
            raise ValueError('complete HIF state order/resources are not revalidated')
        if len(live.hand) != len(state.deck.hand):
            raise ValueError('complete HIF hand count changed')
        cards = []
        for observed, ref in zip(live.hand,state.deck.hand,strict=True):
            definition = kernel.card_definition(state, ref)
            support_id = self._opening_support_visible_id(ref)
            if support_id is None and observed.card_id == self._hand_support_visible_id(ref):
                support_id = observed.card_id
            expected_id = support_id or definition.canonical_card_id or definition.card_id
            expected_upgraded = True if support_id else ref.upgraded
            if (not observed.identity_accepted or observed.card_id != expected_id
                    or observed.upgraded is not expected_upgraded or observed.box is None):
                raise ValueError('complete HIF hand identity changed')
            # Preserve the legacy gate for cost resources whose selected HUD
            # transaction is not supported. Fixed stamina still needs B proof.
            tags = observed.tags
            cost = observed.cost
            if definition.cost_kind is CardCostKind.STAMINA:
                cost = definition.stamina_cost
                tags = tuple(t for t in tags if t != 'transaction:unsupported_typed_cost')
            elif definition.cost_kind in (CardCostKind.CONCENTRATION, CardCostKind.GOOD_CONDITION):
                if not kernel._non_full_power_cost_payable(state,ref,definition):
                    raise ValueError('complete HIF '+definition.cost_kind.value.lower()+' cannot pay the declared cost')
                # Legacy safety's cost field means stamina only. The typed
                # amount remains independently read and checked in B.
                cost=0
                tags=tuple(t for t in tags if t!='transaction:unsupported_typed_cost')
            cards.append(replace(observed,cost=cost,tags=tags))
        statuses = tuple(status for status in (
            LiveStatus('concentration',state.concentration),
            LiveStatus('good_condition',remaining_turns=state.good_condition_turns),
        ) if status.stacks or status.remaining_turns)
        rebound = replace(live,hand=tuple(cards),items=state.items,statuses=statuses,
            field_effects=tuple(f'{e.source_id}:{e.definition_effect_index}' for e in state.scheduled_effects),
            unknown_fields=tuple(x for x in live.unknown_fields if x not in ('items','statuses','field_effects')))
        rebound.validate()
        self.bound_snapshot = rebound.snapshot_id
        return rebound

    def decide(self, live):
        started = time.perf_counter()
        def proposal(reason, action=None):
            return DecisionProposal('PROPOSE' if action else 'STOP',reason,action,1.,
                self.provider_name,self.provider_version,(time.perf_counter()-started)*1000,
                candidate_traces={'high_score': self.response.to_dict()} if self.response else {})
        if live.snapshot_id != self.bound_snapshot or live.unknown_fields or live.contradictory_fields:
            return proposal('hif_complete_state_binding_missing')
        state, kernel = self.bundle.state, self.bundle.kernel
        if self.response is None:
            if state.drinks and self.bundle.drink_slots is None:
                return proposal('hif_physical_drink_slots_missing')
            legal = kernel.legal_actions(state,plan_type=self.bundle.kernel.plan_type,decision_kind=DecisionKind.LIVE)
            slots_ready = all(c.box is not None and c.identity_accepted for c in live.hand)
            supported = all(kernel.card_definition(state, c).cost_kind in (CardCostKind.NORMAL,CardCostKind.STAMINA,CardCostKind.CONCENTRATION,CardCostKind.GOOD_CONDITION)
                            for c in state.deck.hand)
            request = DecisionRequest(live.snapshot_id,self.bundle.kernel.plan_type,DecisionKind.LIVE,state,
                CapabilityReport(observation_valid=True,game_legal=bool(legal),
                    model_supported=state.deck.conservation_valid,policy_eligible=state.turn>=1,
                    execution_ready=slots_ready and supported),
                card_slots=tuple((ref.instance_id,card.slot) for ref,card in zip(state.deck.hand,live.hand,strict=True)),
                drink_slots=self.bundle.drink_slots or (),
                ruleset_id=kernel.ruleset_id)
            self.response = (self.decision_policy or build_hif_policy(kernel)).decide(request)
        action = self.response.action
        if action is None:
            return proposal('hif_policy_stop:'+self.response.reason)
        if action.kind in (ActionKind.USE_DRINK, ActionKind.END_TURN):
            return proposal(self.response.reason, LiveAction(action.kind.value))
        if action.kind is not ActionKind.PLAY_CARD:
            return proposal('hif_secondary_transaction_required')
        chosen = live.hand[action.source_slot]
        if not chosen.playable:
            return proposal('hif_policy_card_not_playable')
        return proposal(self.response.reason,LiveAction('PLAY_CARD',action.source_slot))

    def selected_resource_preview(self, slot):
        """Project the observed card-only HUD preview, not the committed state.

        Native 432/435 show card cost/recovery in the HUD while after-card item
        stamina costs are excluded. Require the supported card effects and a
        deterministic resource projection, including typed non-stamina costs.
        """
        action = self.response.action if self.response else None
        if (type(slot) is not int or not 0 <= slot < len(self.bundle.state.deck.hand)
                or action is None or action.kind is not ActionKind.PLAY_CARD or action.source_slot != slot):
            raise ValueError('HIF preview has no bound selected decision')
        state,kernel = self._selected_model_state(),self.bundle.kernel
        card = state.deck.hand[slot]
        if action.card_instance_id != card.instance_id:
            raise ValueError('HIF preview physical card does not match the decision')
        supported = self._current_support_definition() or self._current_genki_support_definition()
        if supported is not None and kernel.card_definition(state, card) != supported:
            from .skill_support import project_supported_card_once

            projected = replace(self.bundle, state=replace(state,
                items=(), memory_abilities=(), scheduled_effects=()))
            branches, _ = project_supported_card_once(projected, action,
                effective_definition=supported, settle_automatic_turn=False)
            if len({(branch.state.stamina, branch.state.genki, branch.state.score) for branch in branches}) != 1:
                raise ValueError('HIF supported card preview is not deterministic')
            preview = branches[0].state
            contract = dict(source='native_hif_card_effect_preview', run_id=self.bundle.run_id,
                revision=self.bundle.revision, card_id=supported.canonical_card_id or supported.card_id,
                selected_slot=slot, baseline_stamina=state.stamina, baseline_shield=state.genki,
                effective_cost=kernel._effective_stamina_cost(state,card,supported),
                preview_stamina=preview.stamina, preview_shield=preview.genki, preview_score=preview.score)
            if (state.nullify_cost_cards > 0
                    and supported.cost_kind in (CardCostKind.NORMAL, CardCostKind.STAMINA)):
                contract.update(displayed_card_cost=kernel._effective_stamina_cost(
                    replace(state,nullify_cost_cards=0),card,supported), nullify_cost_cards=state.nullify_cost_cards)
            return contract
        from ..hif_gray_selection import ordinary_nullified_resource_preview

        nullified = ordinary_nullified_resource_preview(self.bundle, slot)
        if nullified is not None:
            return nullified
        definition = kernel.card_definition(state, card)
        if (hand_exchange_resource_effects(definition)
                and kernel._declared_cost_kind(definition) is CardCostKind.NORMAL):
            # The admitted exchange/status/action effects cannot change the
            # card-only HUD resources or score. Use the real payment rules
            # without enumerating random replacement hands or firing items.
            cost = kernel._effective_stamina_cost(state, card, definition)
            preview, _ = kernel._pay_stamina_cost(state, cost, card.instance_id)
            contract = dict(source='native_hif_card_effect_preview', run_id=self.bundle.run_id,
                revision=self.bundle.revision, card_id=definition.canonical_card_id or definition.card_id,
                selected_slot=slot, baseline_stamina=state.stamina, baseline_shield=state.genki,
                effective_cost=cost, preview_stamina=preview.stamina,
                preview_shield=preview.genki, preview_score=state.score)
            if state.nullify_cost_cards > 0:
                contract.update(displayed_card_cost=kernel._effective_stamina_cost(
                    replace(state, nullify_cost_cards=0), card, definition),
                    nullify_cost_cards=state.nullify_cost_cards)
            return contract
        from .kernel import EffectSpec

        # National Idol's known counter changes no HUD resource directly.
        # Keep its exact single-use repeat primitive and typed payment; other
        # families or conditional repeat effects remain outside this preview.
        national_repeat = (definition.base_card_id == '162'
            and (definition.canonical_card_id or definition.card_id) in {'162', '163'}
            and definition.cost_kind is CardCostKind.GOOD_CONDITION
            and definition.card_type == 'mental' and definition.rarity == 'SSR'
            and tuple(effect for effect in definition.effects
                      if effect.kind is EffectKind.DOUBLE_CARD_EFFECT)
            == (EffectSpec(EffectKind.DOUBLE_CARD_EFFECT, amount=1),))
        ordinary_score_status = (definition.cost_kind is CardCostKind.NORMAL
            and any(effect.kind is EffectKind.SCORE for effect in definition.effects)
            and all(effect.kind in {EffectKind.SCORE,EffectKind.CONCENTRATION,EffectKind.GOOD_CONDITION}
                    for effect in definition.effects))
        # 1673's 668 has direct genki/status effects. Let the same kernel apply
        # real cost modifiers and NULLIFY_GENKI; a catalog recovery heuristic
        # cannot describe this HUD. Keep score/schedules outside this new domain.
        ordinary_genki_status = (definition.cost_kind is CardCostKind.NORMAL
            and not definition.persistent_effects
            and any(effect.kind is EffectKind.GENKI for effect in definition.effects)
            and all(effect.kind in {EffectKind.GENKI,EffectKind.CONCENTRATION,EffectKind.GOOD_CONDITION}
                    for effect in definition.effects))
        # 1793's preparation card grants focus, a half-cost turn and an action.
        # Payment precedes those effects: preserve the pre-play modifiers and
        # the existing nullification contract instead of paying its face cost.
        ordinary_half_cost_preparation = (definition.cost_kind is CardCostKind.NORMAL
            and not definition.persistent_effects
            and any(effect.kind is EffectKind.HALF_COST_TURNS for effect in definition.effects)
            and all(effect.kind in {EffectKind.CONCENTRATION, EffectKind.HALF_COST_TURNS,
                    EffectKind.EXTRA_ACTION}
                and type(effect.amount) in (int, float) and math.isfinite(effect.amount)
                and effect.amount > 0 and float(effect.amount).is_integer()
                and effect == EffectSpec(effect.kind, amount=effect.amount)
                for effect in definition.effects))
        # 1680's ordinary card grants EXCELLENT now and registers future score.
        # Only plain, positive-delay TURN score schedules enter this domain;
        # the kernel keeps every schedule and applies no future turn here.
        future_scores = tuple(effect for effect in definition.effects if effect.kind is EffectKind.SCHEDULE)
        ordinary_future_score = (definition.cost_kind is CardCostKind.NORMAL
            and not definition.persistent_effects and bool(future_scores)
            and any(effect.kind is EffectKind.EXCELLENT_CONDITION for effect in definition.effects)
            and all(effect in future_scores or effect == EffectSpec(EffectKind.EXCELLENT_CONDITION,
                amount=effect.amount) for effect in definition.effects)
            and all(effect.delay_turns >= 1 and bool(effect.scheduled_effects)
                and effect == EffectSpec(EffectKind.SCHEDULE, delay_turns=effect.delay_turns,
                    scheduled_effects=effect.scheduled_effects)
                and all(child == EffectSpec(EffectKind.SCORE, amount=child.amount,
                    score_good_condition_multiplier=child.score_good_condition_multiplier)
                    for child in effect.scheduled_effects) for effect in future_scores))
        ordinary_composite = _ordinary_composite_resource_effects(definition)
        if (not (ordinary_score_status or ordinary_genki_status or ordinary_future_score
                 or ordinary_half_cost_preparation or ordinary_composite)
                and definition.cost_kind not in (CardCostKind.STAMINA,CardCostKind.CONCENTRATION,CardCostKind.GOOD_CONDITION)
                or not definition.effects or any(effect.kind not in {
                    EffectKind.STAMINA,EffectKind.CONCENTRATION_ADDITION_BUFF,EffectKind.EXTRA_ACTION,
                    EffectKind.GENKI,EffectKind.GOOD_CONDITION,EffectKind.EXCELLENT_CONDITION,
                    EffectKind.HALF_COST_TURNS,EffectKind.SCORE,EffectKind.SCHEDULE,
                    EffectKind.DRAW,EffectKind.CONCENTRATION,EffectKind.CONCENTRATION_EFFECT_BUFF,EffectKind.NULLIFY_GENKI_TURNS
                } and not (national_repeat and effect.kind is EffectKind.DOUBLE_CARD_EFFECT)
                    and not (ordinary_composite and effect.kind is EffectKind.ADD_CARD_TO_DECK)
                    for effect in definition.effects)):
            return None
        if not (ordinary_future_score or ordinary_composite) and any(effect.kind is EffectKind.SCHEDULE and (
                not effect.scheduled_effects or any(child.kind not in
                    {EffectKind.DRAW,EffectKind.CONCENTRATION}
                    for child in effect.scheduled_effects))
                and not (definition.cost_kind is CardCostKind.GOOD_CONDITION
                    and _counted_good_condition_score_schedule(effect))
                for effect in definition.effects):
            from .kernel import EffectSpec, CardDefinition
            from .contracts import SchedulePhase

            # Fixed CSV 301: its score trigger is registered now, and fires on
            # later active cards. Do not admit arbitrary scheduled score effects.
            entertainment = CardDefinition('301','至高のエンタメ+',PlanType.SENSE,
                stamina_cost=2,cost_kind=CardCostKind.CONCENTRATION,
                effects=(EffectSpec(EffectKind.SCHEDULE,
                    scheduled_effects=(EffectSpec(EffectKind.DRAW,amount=1),)),
                    EffectSpec(EffectKind.SCHEDULE,delay_turns=0,
                        schedule_phase=SchedulePhase.CARD_USED,schedule_limit=None,
                        trigger_card_type='active',
                        scheduled_effects=(EffectSpec(EffectKind.SCORE,amount=5),))),
                limited=True,card_type='active',rarity='SSR',base_card_id='300',source_type='produce')
            if definition != entertainment:
                return None
        projected = replace(state,items=(),memory_abilities=(),scheduled_effects=tuple(
            ref for ref in state.scheduled_effects
            if not ordinary_composite and ref.source_kind is EffectSourceKind.CARD))
        projected, excluded_concentration = self._selected_hud_concentration_projection(
            projected, definition)
        projected, excluded_after_card = self._selected_hud_after_card_projection(projected, definition)
        branches = kernel.transition(projected,action,plan_type=self.bundle.kernel.plan_type)
        # A random card identity is irrelevant only when every outcome has
        # exactly the same visible resource projection. Never select a draw.
        if (not branches or abs(sum(b.probability for b in branches)-1)>1e-9
                or any(b.state.pending_choice or b.state.resolution_stack for b in branches)
                or len({(b.state.stamina,b.state.genki,b.state.score,b.state.turn)
                        for b in branches}) != 1):
            raise ValueError('HIF card resource preview is not deterministic')
        preview = branches[0].state
        score_projections = tuple((branch.state.score, []) if ordinary_composite else
            self._selected_card_score_preview(action, branch) for branch in branches)
        if any(value != score_projections[0] for value in score_projections[1:]):
            raise ValueError('HIF card-only score preview is not deterministic')
        preview_score, excluded = score_projections[0]
        contract = dict(source='native_hif_card_effect_preview',run_id=self.bundle.run_id,
            revision=self.bundle.revision,card_id=definition.canonical_card_id or definition.card_id,
            selected_slot=slot,baseline_stamina=state.stamina,baseline_shield=state.genki,
            effective_cost=kernel._effective_stamina_cost(state,card,definition),
            preview_stamina=preview.stamina,preview_shield=preview.genki,preview_score=preview_score)
        if (state.nullify_cost_cards > 0
                and definition.cost_kind in (CardCostKind.NORMAL, CardCostKind.STAMINA)):
            # Recovery changes the HUD even when payment is nullified. The
            # physical card still displays its ordinary cost, as in the
            # supported-card branch; both must match fresh observations.
            contract.update(displayed_card_cost=kernel._effective_stamina_cost(
                replace(state,nullify_cost_cards=0),card,definition),
                nullify_cost_cards=state.nullify_cost_cards)
        if excluded:
            contract.update(preview_score_scope='card_without_verified_card_used_score',
                excluded_card_used_scores=excluded)
        if excluded_concentration:
            contract['excluded_card_used_concentration'] = excluded_concentration
        if excluded_after_card:
            contract['excluded_after_card_score_schedules'] = excluded_after_card
        return contract

    def _selected_hud_after_card_projection(self, state, definition):
        """1906 HUD omits the known counted score after the selected card.

        Only the already supported score/good-condition counter is omitted
        from this display calculation. The committed transition is untouched.
        """
        from .contracts import SchedulePhase

        if (definition.cost_kind is not CardCostKind.CONCENTRATION
                or definition.card_type != 'active'
                or not any(effect.kind is EffectKind.SCORE for effect in definition.effects)
                or any(effect.kind not in {EffectKind.SCORE, EffectKind.GOOD_CONDITION,
                    EffectKind.EXTRA_ACTION} for effect in definition.effects)):
            return state, []
        refs = []
        for ref in state.scheduled_effects:
            if (ref.source_kind is not EffectSourceKind.CARD
                    or ref.trigger_phase is not SchedulePhase.AFTER_CARD_USED
                    or ref.turns_until_fire != 0 or ref.remaining_uses is not None
                    or ref.remaining_turns is not None or ref.persistent or ref.direct_on_trigger
                    or ref.duration_fresh or type(ref.effect_counter) is not int or ref.effect_counter < 0
                    or ref.card_id not in self.bundle.kernel.cards
                    or not _counted_good_condition_score_schedule(self.bundle.kernel._scheduled_effect_spec(ref))):
                continue
            identities = [card for card in state.deck.all_cards if card.instance_id == ref.source_id]
            if len(identities) != 1 or identities[0].card_id != ref.card_id:
                raise ValueError('HIF preview after-card score source is ambiguous')
            refs.append(ref)
        if not refs:
            return state, []
        return replace(state, scheduled_effects=tuple(
            ref for ref in state.scheduled_effects if ref not in refs)), [dict(
                source_id=ref.source_id,card_id=ref.card_id,definition_effect_index=ref.definition_effect_index,
                trigger_phase=ref.trigger_phase.value,effect_counter=ref.effect_counter) for ref in refs]

    def _selected_hud_concentration_projection(self, state, definition):
        """1281/1570/1906 HUD omits verified 108/109 triggers before scoring.

        This only changes the preview state. The actual transition retains
        both 108's concentration and 301's score, as verified by 1244/1247.
        1570 proves the same display for an ordinary NORMAL pure-score card.
        A bound extra-plus receipt retains its original catalog-only contract;
        its independent HUD is reconciled by the existing support projection.
        Unknown schedules and mixed NORMAL effects stay outside that extension.
        """
        from .kernel import EffectSpec
        from .contracts import SchedulePhase

        receipt = self._current_single_plus_estimate()
        normal_pure_score = (definition.cost_kind is CardCostKind.NORMAL
            and self.selected_skill_support is None
            and (receipt is None or receipt.get('plus_count') == 1)
            and bool(definition.effects)
            and all(effect.kind is EffectKind.SCORE for effect in definition.effects))
        if (not normal_pure_score and definition.cost_kind is not CardCostKind.CONCENTRATION
                or definition.card_type != 'active'
                or not any(effect.kind is EffectKind.SCORE for effect in definition.effects)
                or any(effect.kind not in {EffectKind.SCORE, EffectKind.GOOD_CONDITION,
                                           EffectKind.EXTRA_ACTION} for effect in definition.effects)):
            return state, []
        fixed = {card_id: EffectSpec(EffectKind.SCHEDULE, delay_turns=0,
            schedule_phase=SchedulePhase.CARD_USED, schedule_limit=None,
            trigger_card_type='active', scheduled_effects=(EffectSpec(EffectKind.CONCENTRATION, amount=amount),))
            for card_id,amount in (('108',1),('109',1))}
        refs = tuple(ref for ref in state.scheduled_effects
            if ref.source_kind is EffectSourceKind.CARD and ref.card_id in fixed
            and ref.definition_effect_index == 1 and ref.trigger_phase is SchedulePhase.CARD_USED)
        if not refs or any(ref.turns_until_fire != 0 or ref.remaining_uses is not None
                or ref.remaining_turns is not None or ref.persistent or ref.direct_on_trigger
                or ref.effect_counter != 0 or ref.duration_fresh
                or ref.card_id not in self.bundle.kernel.cards
                or self.bundle.kernel._scheduled_effect_spec(ref) != fixed[ref.card_id] for ref in refs):
            return state, []
        for ref in refs:
            identities = [card for card in state.deck.all_cards if card.instance_id == ref.source_id]
            if len(identities) != 1 or identities[0].card_id != ref.card_id:
                raise ValueError('HIF preview automatic concentration source is ambiguous')
        return replace(state, scheduled_effects=tuple(
            ref for ref in state.scheduled_effects if ref not in refs)), [
                dict(source_id=ref.source_id, card_id=ref.card_id, definition_effect_index=1,
                     trigger_phase='CARD_USED', declared_concentration=1) for ref in refs]

    def _selected_card_score_preview(self, action, branch):
        """Separate 1056's proven 301 trigger from the selected card HUD only.

        Keep the complete transition and its schedules intact. Unknown effects
        stay in the prediction; this is not a blanket automatic-score filter.
        """
        from .kernel import EffectSpec
        from .contracts import SchedulePhase

        state, kernel = self.bundle.state, self.bundle.kernel
        card = state.deck.hand[action.source_slot]
        if kernel.card_definition(state, card).card_type != 'active':
            return branch.state.score, []
        fixed = EffectSpec(EffectKind.SCHEDULE, delay_turns=0,
            schedule_phase=SchedulePhase.CARD_USED, schedule_limit=None,
            trigger_card_type='active', scheduled_effects=(EffectSpec(EffectKind.SCORE, amount=5),))
        sources = {}
        matching_sources = {}
        for ref in state.scheduled_effects:
            if (ref.source_kind is EffectSourceKind.CARD and ref.card_id == '301'
                    and ref.definition_effect_index == 1 and ref.trigger_phase is SchedulePhase.CARD_USED):
                matching_sources[ref.source_id] = matching_sources.get(ref.source_id, 0) + 1
            if (ref.source_kind is EffectSourceKind.CARD and ref.card_id == '301'
                    and ref.definition_effect_index == 1
                    and ref.trigger_phase is SchedulePhase.CARD_USED and ref.turns_until_fire == 0
                    and ref.remaining_uses is None and ref.remaining_turns is None
                    and not ref.persistent and not ref.direct_on_trigger
                    and kernel._scheduled_effect_spec(ref) == fixed):
                identities = [c for c in state.deck.all_cards if c.instance_id == ref.source_id]
                if len(identities) != 1 or identities[0].card_id != ref.card_id:
                    raise ValueError('HIF preview automatic score source is ambiguous')
                sources[ref.source_id] = sources.get(ref.source_id, 0) + 1
        # The kernel registers a separate ref for each National Idol pass.
        # Only identical known lifecycles may share the same event identity.
        if any(count != matching_sources[source] for source, count in sources.items()):
            raise ValueError('HIF preview automatic score source is ambiguous')
        excluded = []
        observed_counts = {}
        for index, event in enumerate(branch.events):
            if (event.event_type != 'TRIGGER_SCHEDULED' or event.source_id not in sources
                    or event.detail != '301:1:CARD_USED'):
                continue
            score_event = branch.events[index + 1] if index + 1 < len(branch.events) else None
            if (score_event is None or score_event.event_type != 'SCORE'
                    or score_event.source_id != event.source_id or type(score_event.amount) is not int
                    or score_event.amount < 0
                    or observed_counts.get(event.source_id, 0) >= sources[event.source_id]):
                raise ValueError('HIF preview automatic score lacks its exact source event')
            observed_counts[event.source_id] = observed_counts.get(event.source_id, 0) + 1
            excluded.append(dict(source_id=event.source_id, card_id='301', definition_effect_index=1,
                trigger_phase='CARD_USED', amount=score_event.amount))
        if observed_counts != sources:
            raise ValueError('HIF preview automatic score trigger evidence is incomplete')
        return branch.state.score - sum(row['amount'] for row in excluded), excluded

    def committed_transition(self):
        """Keep every settled outcome; fresh observation resolves random draws."""
        self.committed_event_branches = ()
        action = self.response.action if self.response else None
        if action is None or action.kind not in (ActionKind.PLAY_CARD, ActionKind.USE_DRINK, ActionKind.END_TURN):
            raise ValueError('HIF committed transition has no bound joint decision')
        model_state = self._selected_model_state()
        supported = self._current_support_definition()
        if self._normal_score_projection is not None and self.observed_normal_score_preview is not None:
            state, bound_action, receipt, ordinary, projection, values = self._normal_score_projection
            if (state != model_state or action != bound_action
                    or receipt != self._current_single_plus_estimate()
                    or values != self.observed_normal_score_preview
                    or self.bundle.kernel.card_definition(state, state.deck.hand[action.source_slot]) != ordinary):
                raise ValueError('HIF pure-score projection cannot carry to another state or action')
            supported = projection
        current_definition = None
        if supported is not None:
            ref = self.bundle.state.deck.hand[action.source_slot]
            current_definition = self.bundle.kernel.card_definition(self.bundle.state, ref)
        projected_support = supported is not None and current_definition != supported
        if projected_support:
            from .skill_support import project_supported_card_once

            branches, _ = project_supported_card_once(self.bundle, action,
                effective_definition=supported, settle_automatic_turn=False)
        else:
            branches = self.bundle.kernel.transition(model_state, action, plan_type=self.bundle.kernel.plan_type)
        if not branches or abs(sum(b.probability for b in branches)-1)>1e-9:
            raise ValueError('HIF commit has incomplete outcome probabilities')
        if self.hold_ready and any(branch.state.pending_choice for branch in branches):
            from ..hif_hold_transaction import hold_origin_supported

            states = tuple(dict.fromkeys(branch.state for branch in branches))
            if len(states) == 1 and hold_origin_supported(states[0], action):
                self.committed_event_branches = tuple(branches)
                return states[0]
        states=[]
        correction = None
        if (not projected_support and action.kind is ActionKind.PLAY_CARD
                and self.observed_normal_score_preview is not None):
            card_only = replace(model_state, items=(), memory_abilities=(), scheduled_effects=())
            projections = self.bundle.kernel.transition(card_only, action, plan_type=self.bundle.kernel.plan_type)
            if (not projections or abs(sum(p.probability for p in projections)-1)>1e-9
                    or any(p.state.pending_choice or p.state.resolution_stack for p in projections)
                    or len({(p.state.score,p.state.stamina,p.state.genki,p.state.turn) for p in projections}) != 1):
                raise ValueError('HIF observed score preview projection is not deterministic')
            correction = {name: value - getattr(projections[0].state, name)
                          for name, value in self.observed_normal_score_preview.items()}
        event_branches = []
        for branch in branches:
            state=branch.state
            if correction is not None:
                # Keep catalog strategy values; reconcile independently read
                # card-only resources before the automatic next-turn model.
                # The subsequent full native settlement must still match.
                state = replace(state, **{name: getattr(state, name) + delta
                                          for name, delta in correction.items()})
            if state.pending_choice or state.resolution_stack:
                raise ValueError('HIF card continuation has pending resolution')
            if not state.terminal and state.actions_remaining == 0:
                next_branches=self.bundle.kernel.transition(state, GameAction(ActionKind.END_TURN),
                                                             plan_type=self.bundle.kernel.plan_type)
                if not next_branches or abs(sum(b.probability for b in next_branches)-1)>1e-9:
                    raise ValueError('HIF automatic turn transition is incomplete')
                states.extend(b.state for b in next_branches)
                event_branches.extend(replace(b, probability=branch.probability*b.probability,
                    events=branch.events+b.events) for b in next_branches)
            else:
                states.append(state)
                event_branches.append(replace(branch, state=state))
        if any(state.pending_choice or state.resolution_stack
               or state.turn not in (self.bundle.state.turn, self.bundle.state.turn+1) for state in states):
            raise ValueError('HIF card continuation is not settled')
        self.committed_event_branches = tuple(event_branches)
        states=tuple(dict.fromkeys(states))
        return states[0] if len(states)==1 else states
