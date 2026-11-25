"""Bind native battle policy decisions to the pinned upstream card UI."""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, replace

from route.battle_policy import FORMAL_PROFILES

from .partial_card import MIN_LIVE_HAND_TOP_RATIO
from .hif_upstream_ui import UpstreamHifCardUI, unselected_art_footprints
from .high_score.contracts import DecisionKind, DecisionRequest, CapabilityReport


def unselected_art_candidates(candidates, frame_shape, *, image=None):
    """Recover full artwork footprints from contained upstream visible fragments.

    The caller must first
    establish an unselected row; original boxes remain the physical evidence.
    """
    if (image is not None and frame_shape[:2] == (1280, 720) and 1 <= len(candidates) <= 3
            and any(card.box[2:] != (190, 250) for card in candidates)):
        from .hif_visual import hif_unselected_row_candidates

        # A playable detector fragment supplies the click box, but the model
        # was trained on complete art. Reuse this frame's title-plate geometry
        # for classification only; a missing plate keeps the original crop.
        bodies = hif_unselected_row_candidates(frame_shape, candidates, image=image)
        if bodies and len(bodies) == len(candidates):
            return tuple(replace(card, box=body.box) if card.slot == body.slot and body.box[2:] == (190, 250)
                         else card for card, body in zip(candidates, bodies, strict=True))
    footprints = unselected_art_footprints([(card.slot, card.box) for card in candidates], frame_shape)
    if footprints is None:
        return candidates
    return tuple(replace(card, box=box) for card, box in zip(candidates, footprints, strict=True))


def unselected_row_anchors(candidates, frame_shape):
    """Identify stable physical slots despite changes in visible detector width.

    This does not establish SELECT absence or authorize input. Callers retain
    the raw boxes and independently check the current selection on every frame.
    """
    count = len(candidates)
    if frame_shape[:2] != (1280, 720) or not 1 <= count <= 5:
        return None
    spacing = 218 if count <= 3 else 494 / (count - 1)
    origin = (720 - 190 - (count - 1) * spacing) / 2
    anchors = tuple(round(origin + slot * spacing) for slot in range(count))
    for slot, (card, anchor) in enumerate(zip(candidates, anchors, strict=True)):
        x, y, width, height = card.box
        if (card.slot != slot or not anchor - 8 <= x <= anchor + 190 - width + 8
                or not 878 <= y <= 904 or not 90 <= width <= 206
                or not 200 <= height <= 266):
            return None
    return anchors


def unselected_resource_reobserve_allowed(baseline):
    """Permit only another screenshot batch; no incomplete batch grants input."""
    try:
        rows = baseline['resource_observations']
        return (baseline.get('accepted') is False and baseline.get('geometry_stable') is False
            and baseline.get('select_consensus') is None and baseline.get('select_evidence') == []
            and len(rows) == 3
            and all(all(r[name].get('accepted') is True and type(r[name].get('value')) is int
                and .95 <= r[name].get('confidence', 0) <= 1
                and r[name]['value'] == rows[0][name]['value'] for r in rows)
                for name in ('stamina', 'shield')))
    except (KeyError, TypeError, ValueError):
        return False


def _reobserve_unsubmitted(context, ui, dispatch_count):
    """Keep rejected intents separate from positions in the actual Maa dispatch list."""
    if context.gkh_hif_upstream_dispatch_count() != dispatch_count:
        return False
    inputs = ui.receipt['inputs']
    if not inputs:
        return True  # The upstream recognizer never reached a click.
    read_preparation = getattr(context, 'gkh_hif_upstream_preparation', None)
    preparation = read_preparation() if callable(read_preparation) else None
    if (not isinstance(preparation, dict) or preparation.get('dispatch_state') != 'rejected_before_dispatch'
            or preparation.get('input_may_have_been_dispatched') is not False
            or preparation.get('reason') not in ('upstream UI input evidence expired',
                                                'snapshot_expired_after_input_preparation')
            or (preparation.get('x'), preparation.get('y'), preparation.get('authorization_floor')) !=
               (inputs[-1].get('x'), inputs[-1].get('y'), inputs[-1].get('captured_at'))
            or any(item.get('dispatched') is not False or item.get('dispatch_index') != dispatch_count
                   for item in inputs)):
        return False
    ui.receipt.setdefault('undispatched_inputs', []).extend(
        dict(item, rejection=preparation['reason']) for item in inputs)
    inputs.clear()
    ui.publish()
    return True


def run_strong_end_turn(module, context, params, policy, baseline):
    """Use fb8's SKIP recognition/Click; rule settlement remains our business owner."""
    from .high_score.contracts import ActionKind
    from .high_score.maintenance import complete_action_baseline
    from .high_score.hif_fast_native import persist_projected_events

    action = policy.response.action
    event = dict(event='live_end_turn', accepted=False, executed=False,
        action=asdict(action), proposal=policy.response.to_dict(),
        end_turn_submit_count=0, execution_submit_count=0)
    ui = UpstreamHifCardUI(context, frame_identifier=module.frame_identifier,
        sink=context.gkh_hif_upstream_ui_sink)
    try:
        bundle = context.gkh_hif_opening_state
        if (policy.bundle != bundle or not complete_action_baseline(bundle, baseline)
                or action.kind is not ActionKind.END_TURN
                or action not in bundle.kernel.legal_actions(bundle.state,
                    plan_type=bundle.kernel.plan_type, decision_kind=DecisionKind.LIVE)):
            raise ValueError('strong END lacks its complete legal decision')
        # fb8 _skip_round has no extra HUD/inventory consensus before SKIP.
        # The bound decision above and the actual settlement below own business
        # state; the upstream target capture owns the fresh input evidence.
        branches = bundle.kernel.transition(bundle.state, action, plan_type=bundle.kernel.plan_type)
        persist_projected_events(context, policy, branches)
        # fb8 battle_recognition._skip_round issues this same recognition/Click.
        # The existing rule/terminal observer owns the subsequent wait, so the
        # upstream main loop cannot also advance a nonbattle result page.
        poll = getattr(context, 'gkh_hif_poll_battle_progress', None)
        while True:
            before = context.gkh_hif_upstream_dispatch_count()
            try:
                if ui.end_turn(action=asdict(action)):
                    break
            except (ValueError, RuntimeError) as error:
                if not ui.receipt['inputs'] or poll is None or not _reobserve_unsubmitted(context, ui, before):
                    raise
                event['last_reobservation_reason'] = str(error)
            else:
                if poll is None or not _reobserve_unsubmitted(context, ui, before):
                    raise ValueError('upstream SKIP was not confirmed by Maa')
            from .hif_result import try_displayed_terminal_frame

            terminal = try_displayed_terminal_frame(context, ui.image, (), frame_identifier=module.frame_identifier)
            if terminal is not None:
                context.gkh_hif_terminal_event = dict(terminal, event='live_committed_state',
                    execution_submit_count=0, executed=False)
                event.update(accepted=True, settlement=context.gkh_hif_terminal_event)
                return event
            if poll(ui.image) or context.gkh_hif_close_unexpected_drink(ui.image):
                continue
            time.sleep(ui.CARD_HAND_POLL_INTERVAL)
        settlement = module._observe_hif_opening_state(context, params,
            committed_expected=tuple(branch.state for branch in branches),
            baseline_event=baseline, committed_action=action)
        event.update(settlement=settlement, accepted=settlement.get('accepted') is True,
            executed=settlement.get('accepted') is True, reason=settlement.get('reason'))
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as error:
        event['reason'] = str(error)
    finally:
        count = context.gkh_hif_upstream_dispatch_count()
        submitted = any(item.get('phase') == 'end_turn'
            and type(item.get('dispatch_index')) is int and item['dispatch_index'] < count
            for item in ui.receipt['inputs'])
        event.update(ui=ui.receipt, end_turn_submit_count=int(submitted),
            execution_submit_count=int(submitted), input_may_have_been_dispatched=submitted)
        module._confirm_hif_mcts_observed_commit(context, policy, event, submitted=submitted)
        module._publish_hif_component(context, event)
    return event


def _playable_slots(module, ui, image, results, boxes, hand_size):
    """Translate this detection's upstream boxes to slots in the maintained hand."""
    if not any(row.label == 'useless' and row.box[1] >= image.shape[0] * MIN_LIVE_HAND_TOP_RATIO
               for row in results):
        return tuple(range(hand_size)) if len(boxes) == hand_size else None

    physical = module._normalise_observed_live_candidates(ui.base, image, results)
    playable = [card for card in physical if card.detector_label in ('cards', 'suggestions')]
    if len(physical) != hand_size or len(playable) != len(boxes):
        return None
    for card, (box, _) in zip(playable, boxes, strict=True):
        if tuple(box) == card.box:
            continue
        # The existing row reader can replace a suggestion halo with its body.
        # Keep the upstream box for input only if that body is its sole match.
        if card.detector_label != 'suggestions' or [
            other.slot for other in physical if ui._hif_duplicate_box(box, other.box)
        ] != [card.slot]:
            return None
    return tuple(card.slot for card in playable)


def _stable_hand_geometry(module, ui, state=None, *, target_slot=None, hand_size=None):
    """Bind each upstream playable box to its business identity on the same frame."""
    if state is not None:
        ui._last_turn_count = state.turn
        hand_size = len(state.deck.hand)
    ui._reset_card_hand_stability()
    deadline = time.monotonic() + ui.TIME_OUT
    poll = getattr(getattr(ui, 'base', None), 'gkh_hif_poll_battle_progress', None)
    while poll is not None or time.monotonic() < deadline:
        image = ui.context.tasker.controller.post_screencap().wait().get()
        recognition = ui.context.run_recognition('ProduceHIF__ProduceRecognitionCards', image)
        results = recognition.all_results if recognition else []
        boxes = ui._hif_card_boxes(results)
        slots = _playable_slots(module, ui, image, results, boxes, hand_size)
        if slots is not None:
            if state is not None and (target_slot is None or target_slot in slots):
                # The SELECT path takes its own current frame before input.
                # The whole hand need not repeat on a fixed number of frames.
                return image, boxes, slots
            # Opening inspection has only physical slots before inventory.
            identified = [dict(key=state.deck.hand[slot].instance_id if state is not None else None,
                               box=list(box), upgraded=None)
                for slot, (box, _) in zip(slots, boxes, strict=True)]
            if ui._confirm_card_hand(identified) == 'stable':
                if target_slot is None or target_slot in slots:
                    return image, boxes, slots
                if poll is None:
                    raise ValueError(f'strong target in physical slot {target_slot} has no upstream playable box')
                # The current UI cannot select the intended card. The automatic
                # battle observes again under miyu's shared no-progress clock.
        else:
            ui._reset_card_hand_stability()
        if poll is not None:
            if slots is None:
                from .hif_result import try_displayed_terminal_frame

                if state is not None:
                    result = try_displayed_terminal_frame(ui.base, image, (),
                        frame_identifier=module.frame_identifier)
                    if result is not None:
                        ui.base.gkh_hif_terminal_event = dict(result, event='live_committed_state',
                            execution_submit_count=0, executed=False)
                        return image, (), ()
                if poll(image) or ui.base.gkh_hif_close_unexpected_drink(image):
                    continue
            else:
                poll(image)
        time.sleep(ui.CARD_HAND_POLL_INTERVAL)
    raise ValueError('upstream target hand did not stabilize')


def run_strong_card_step(module, context, params, policy, baseline):
    """Submit the bound policy's card or END intent through the upstream UI.

    The complete/rule-maintained bundle owns card identities and resources.
    Current upstream geometry only binds physical targets; it is not a new
    source for strategy fields and does not turn hidden cards into known ones.
    """
    from .high_score.contracts import ActionKind

    bundle = policy.bundle
    state = bundle.state
    if params.get('hif_policy_profile') not in FORMAL_PROFILES:
        raise ValueError('upstream card input requires a native battle profile')
    if policy.response is not None and policy.response.action is not None:
        if policy.response.action.kind is ActionKind.END_TURN:
            return run_strong_end_turn(module, context, params, policy, baseline)
        if policy.response.action.kind is ActionKind.USE_DRINK:
            return None
    if not state.deck.hand:
        return None
    sink = context.gkh_hif_upstream_ui_sink
    ui = UpstreamHifCardUI(context, frame_identifier=module.frame_identifier, sink=sink)
    if policy.response is None:
        # The maintained state supplies the choices. A drink or SKIP has no
        # physical card target, even when this is a newly computed decision.
        image = ui.context.tasker.controller.post_screencap().wait().get()
        legal = bundle.kernel.legal_actions(state, plan_type=bundle.kernel.plan_type, decision_kind=DecisionKind.LIVE)
        request = DecisionRequest(module.frame_identifier(image), bundle.kernel.plan_type, DecisionKind.LIVE, state,
            CapabilityReport(True, bool(legal), state.deck.conservation_valid, state.turn >= 1, True),
            card_slots=tuple((ref.instance_id, slot) for slot, ref in enumerate(state.deck.hand)),
            drink_slots=bundle.drink_slots or (), ruleset_id=bundle.kernel.ruleset_id)
        policy.response = policy.decision_policy.decide(request)
        if (policy.response.action is None and policy.response.reason
                == 'native_mcts_requires_replan:no_evaluated_continuation'):
            policy.response = policy.decision_policy.replan_exhausted_continuation(request, policy.response)
        module._persist_hif_mcts_computation(context, policy)
    action = policy.response.action
    if action is None:
        event = dict(event='live_policy_stop', accepted=False, executed=False,
            execution_submit_count=0, execution_selection_click_count=0,
            input_may_have_been_dispatched=False, reason=policy.response.reason,
            proposal=policy.response.to_dict())
        module._publish_hif_component(context, event)
        return event
    if action.kind is ActionKind.END_TURN:
        return run_strong_end_turn(module, context, params, policy, baseline)
    if action.kind is not ActionKind.PLAY_CARD:
        return None
    if (type(action.source_slot) is not int or not 0 <= action.source_slot < len(state.deck.hand)
            or replace(action, source_slot=None) not in bundle.kernel.legal_actions(
            state, plan_type=bundle.kernel.plan_type, decision_kind=DecisionKind.LIVE)
            or state.deck.hand[action.source_slot].instance_id != action.card_instance_id):
        raise ValueError('strong target is not a current legal hand action')
    def selected(_context, _image):
        from .hif_upstream_support import read_current_support

        receipt = read_current_support(ui, policy)
        if receipt is not None:
            ui.receipt.update(selected_skill_support=receipt,
                selected_detail=receipt['selection_evidence'])
            ui.publish()

    ui.selected = selected
    from .hif_card_retry import read_retry_scene, observe_after_unconfirmed

    ui.read_retry_scene = lambda frame: read_retry_scene(module, context, frame)
    event = dict(event='live_upstream_card', accepted=False, executed=False,
        execution_submit_count=0, action=asdict(action), proposal=policy.response.to_dict())
    def confirmation_dispatched():
        count = context.gkh_hif_upstream_dispatch_count()
        return any(row['phase'] == 'confirm' and row['dispatch_index'] < count for row in ui.receipt['inputs'])

    try:
        poll = getattr(context, 'gkh_hif_poll_battle_progress', None)
        while True:
            marker = getattr(context, 'gkh_hif_reobserved_selected_target', None)
            if isinstance(marker, dict) and marker.get('action') == asdict(action):
                binding = marker['selection_retry']
                index = binding['bound_index']
                boxes = [(box, None) for box in binding['bound_boxes']]
                if not ui.receipt['inputs'] and marker.get('ui') is not None:
                    ui.receipt = deepcopy(marker['ui'])
            else:
                _, boxes, slots = _stable_hand_geometry(module, ui, state, target_slot=action.source_slot)
                terminal = getattr(context, 'gkh_hif_terminal_event', None)
                if terminal is not None:
                    event = terminal
                    return event
                index = slots.index(action.source_slot)
            before = context.gkh_hif_upstream_dispatch_count()
            confirmed = ui.submit_card_once(index=index, hand_boxes=boxes, action=asdict(action))
            if (confirmed or poll is None or ui.receipt.get('selection_retry', {}).get('outcome') == 'observe_current_page'
                    or not _reobserve_unsubmitted(context, ui, before)):
                break
            poll()

        dispatched = confirmation_dispatched()
        event.update(ui=ui.receipt, execution_submit_count=int(dispatched))
        if not dispatched and ui.receipt.get('selection_retry', {}).get('outcome') == 'observe_current_page':
            observed = observe_after_unconfirmed(module, context, params, baseline, ui)
            context.gkh_hif_reobserved_selection_source = baseline, policy
            event.update(accepted=observed['classification'] == 'terminal', reobserved_battle=observed,
                reason='unconfirmed_selection_observed_current_page')
            return event
        if not dispatched:
            event['reason'] = 'upstream_response_unconfirmed_no_resubmit'
            return event
        context.gkh_hif_upstream_confirmation = None
        if ui.receipt.get('selected_skill_support') is not None:
            context.gkh_hif_upstream_confirmation = dict(selected_skill_support=ui.receipt['selected_skill_support'])
        expected = policy.committed_transition()
        from .hif_hold_transaction import complete_hif_hold
        from .high_score.hif_fast_native import persist_projected_events

        persist_projected_events(context, policy)
        expected = complete_hif_hold(context, policy, expected)
        settlement = module._observe_hif_opening_state(context, params,
            committed_expected=expected, baseline_event=baseline, committed_action=action)
        for _ in range(2):
            if settlement.get('accepted') is True or not ui.retry_persistent_confirmation(index):
                break
            settlement = module._observe_hif_opening_state(context, params,
                committed_expected=expected, baseline_event=baseline, committed_action=action)
        accepted = settlement.get('accepted') is True
        event.update(settlement=settlement, accepted=accepted, executed=accepted)
        module._confirm_hif_mcts_observed_commit(context, policy, event, submitted=dispatched)
        return event
    finally:
        if event.get('event') == 'live_upstream_card':
            event.update(ui=ui.receipt, execution_submit_count=int(confirmation_dispatched()))
        context.gkh_hif_upstream_confirmation = None
        module._publish_hif_component(context, event)


def decode_upstream_action(row, *, bundle=None):
    """Recover actual audited submission, including a stop before final event."""
    from .high_score.contracts import ActionKind, GameAction

    receipt = row.get('upstream_ui_journal')
    if not isinstance(receipt, dict):
        return None
    from .hif_upstream_ui import UPSTREAM_REVISION

    if receipt.get('kind') != 'upstream_hif_card_ui' or receipt.get('revision') != UPSTREAM_REVISION:
        raise ValueError('unsupported upstream UI receipt')
    raw = dict(receipt['action'])
    raw['kind'] = ActionKind(raw['kind'])
    action = GameAction(**raw)
    action.validate()
    if action.kind not in (ActionKind.PLAY_CARD, ActionKind.END_TURN):
        raise ValueError('upstream receipt contains an unsupported action')
    if action.kind is ActionKind.PLAY_CARD and bundle is not None and (action.source_slot >= len(bundle.state.deck.hand)
            or bundle.state.deck.hand[action.source_slot].instance_id != action.card_instance_id):
        raise ValueError('upstream submitted card differs from its original hand')
    phase_name = 'end_turn' if action.kind is ActionKind.END_TURN else 'confirm'
    confirmations = [item for item in receipt.get('inputs', ()) if item.get('phase') == phase_name]
    if action.kind is ActionKind.END_TURN and len(confirmations) != 1:
        raise ValueError('upstream END must contain exactly one submission attempt')
    actual = row.get('native_input_dispatches', ())
    matched = []
    for item in confirmations:
        index = item.get('dispatch_index')
        if type(index) is not int or index < 0:
            raise ValueError('upstream confirmation has no dispatch index')
        if index < len(actual):
            entry = actual[index]
            if (entry.get('kind') == 'swipe'
                    or (entry.get('x'), entry.get('y')) != (item.get('x'), item.get('y'))):
                raise ValueError('upstream confirmation differs from its audited dispatch')
            matched.append(index)
    if not matched:
        raise ValueError('upstream confirmation has no actual audited dispatch')
    if len(set(matched)) != len(matched):
        raise ValueError('upstream confirmation reuses an audited dispatch')
    return action
