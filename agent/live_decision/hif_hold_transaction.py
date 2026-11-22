"""HIF deck/discard hold menu, following the pinned miyu input sequence."""

import time
from dataclasses import asdict, replace

from .hif_choice_panel import (
    CHOICE_BOXES,
    hif_choice_grid,
    confirm_hif_choice_title,
    _choice_selected_title_rows,
)
from .hif_choice_scroll import low_move_layout, scan_move_viewports
from .idol_road_runtime import TowerChoicePolicy
from .high_score.opening import expected_permanent_inventory_keys
from .high_score.planner import RobustHighScorePlanner
from .hif_choice_execution import _dispatch
from .high_score.contracts import ActionKind, GameAction, DecisionKind, DecisionRequest, CapabilityReport
from .hif_choice_transaction import _poll_move_title
from .inventory_customization import measure_customization_badge, stable_customization_badge_presence


def hold_origin_supported(state, action):
    pending = state.pending_choice
    if pending is None or pending.action_kind is not ActionKind.HOLD_CARD:
        return False
    if action.kind is ActionKind.PLAY_CARD:
        source = action.card_instance_id
    elif action.kind is ActionKind.USE_DRINK:
        source = f'drink:{action.drink_id}'
    else:
        return False
    sources = {card.instance_id for card in (*state.deck.draw_pile, *state.deck.discard)}
    return (pending.source_id == source and set(pending.candidate_instance_ids) <= sources
            and (pending.selections_remaining, pending.selection_optional) in ((1, False), (2, True)))


def _control(context, image, node):
    match = context.run_recognition(node, image)
    return tuple(match.best_result.box) if match and match.hit else None


# Fixed miyu fb8 _select_move_cards multi-hold cancel center.
_HOLD_CANCEL = (219, 1159, 2, 2)


def _wait_move_confirm(context, capture, *, multi):
    # fb8 _wait_for_move_enabled requires two positive reads for the
    # single-selection menu. Its multi path runs ProduceMoveCards directly.
    deadline = time.monotonic() + (5 if multi else 3)
    hits = 0
    while time.monotonic() < deadline and not context.tasker.stopping:
        image, at = capture()
        target = _control(context, image, 'ProduceMoveCards')
        hits = hits + 1 if target is not None else 0
        if hits >= (1 if multi else 2):
            return image, at, target
        time.sleep(.2)
    raise ValueError('HIF hold confirmation button is not enabled')


def _poll(context, capture, read):
    progress = getattr(context, 'gkh_hif_poll_battle_progress', None)
    while not context.tasker.stopping:
        image, at = capture()
        target = read(image)
        if target is not None:
            return image, at, target
        if not callable(progress):
            break
        progress(image)
        time.sleep(.2)
    raise ValueError('HIF hold menu or expected control was not observed')


def _resolved_branches(kernel, branches, choices):
    for choice in choices:
        branches = tuple(replace(after, probability=before.probability * after.probability,
            events=before.events + after.events) for before in branches
            for after in kernel.transition(before.state, choice, plan_type=kernel.plan_type))
    if any(b.state.pending_choice or b.state.resolution_stack or b.state.pending_after_card_ids for b in branches):
        raise ValueError('HIF hold would open another unresolved menu')
    return tuple(settled for branch in branches for settled in (
        tuple(replace(after, probability=branch.probability * after.probability,
            events=branch.events + after.events) for after in kernel.transition(branch.state,
                GameAction(ActionKind.END_TURN), plan_type=kernel.plan_type))
        if not branch.state.terminal and branch.state.actions_remaining == 0 else (branch,)))


def execute_hif_hold(context, bundle, action, branches, *, capture, frame_id, reader, catalog,
                     journal, choose=None):
    """Execute one newly opened menu, with separate dispatch receipts per tap."""
    states = tuple(dict.fromkeys(branch.state for branch in branches))
    if len(states) != 1 or not hold_origin_supported(states[0], action):
        raise ValueError('HIF hold has no supported unique primary action outcome')
    state, = states
    pending = state.pending_choice
    multi = pending.selection_optional
    count = len(pending.candidate_instance_ids)
    kernel = bundle.kernel
    record = dict(run_id=bundle.run_id, revision=bundle.revision, origin=asdict(action),
                  actions=[], scrolls=[], inputs=[])

    def save(stage):
        journal(stage, record)

    def click(phase, target, at, frame, **evidence):
        entry = dict(phase=phase, frame_id=frame_id(frame, 0), **evidence)
        record['inputs'].append(entry)
        dispatch_count = context.gkh_hif_upstream_dispatch_count
        index = dispatch_count()
        _dispatch(context, target, at, entry, lambda stage, _: save(stage), phase)
        entry.update(submitted=True, dispatch_index=index)
        save(phase + '_submitted')

    def scroll(stage, payload):
        record['scrolls'] = payload['choice_scrolls']
        save(stage)

    image, _, _ = _poll(context, capture,
        lambda frame: _control(context, frame, 'ProduceRecognitionChooseMoveCards'))
    record['opening_frame_id'] = frame_id(image, 0)
    save('hold_opened')

    if choose is None:
        choose = TowerChoicePolicy(kernel).choose
    choices = []
    while state.pending_choice is not None:
        pending = state.pending_choice
        if pending.source_id != states[0].pending_choice.source_id or len(choices) > int(multi):
            raise ValueError('HIF hold would open another unresolved menu')
        # The shared secondary planner requires action indexes. These are private
        # planning indexes over legal instances, never positions in the UI.
        request = DecisionRequest(record['opening_frame_id'] + f':hold:{len(choices)}',
            kernel.plan_type, DecisionKind.SECONDARY, state, CapabilityReport(True, True, True, True, True),
            card_slots=tuple((instance, index) for index, instance in enumerate(pending.candidate_instance_ids)),
            ruleset_id=kernel.ruleset_id)
        request.validate()
        legal = tuple(RobustHighScorePlanner._attach_source_slot(choice, request) for choice in
                kernel.legal_actions(state, plan_type=kernel.plan_type, decision_kind=DecisionKind.SECONDARY))
        choice = choose(request, legal)
        if choice not in legal:
            raise ValueError('HIF hold strategy returned an illegal choice')
        choice = replace(choice, source_slot=None)
        choices.append(choice)
        transitions = kernel.transition(state, choice, plan_type=kernel.plan_type)
        if all(row.state.pending_choice is None for row in transitions):
            break
        if (len(transitions) != 1 or transitions[0].state.pending_choice is None
                or choice.kind is not ActionKind.HOLD_CARD
                or transitions[0].state.pending_choice.selections_remaining != pending.selections_remaining - 1):
            raise ValueError('HIF hold would open another unresolved menu')
        state = transitions[0].state
    expected = _resolved_branches(kernel, branches, choices)
    record['actions'] = [asdict(choice) for choice in choices]
    save('hold_planned')

    origin = states[0]
    inventory = {iid: (card_id, mods)
                 for iid, card_id, mods in expected_permanent_inventory_keys(bundle, origin)}
    targets = {}
    for choice in choices:
        if choice.kind is not ActionKind.HOLD_CARD:
            continue
        instance = choice.card_instance_id
        card_id, mods = inventory[instance]
        same_id = [iid for iid in origin.pending_choice.candidate_instance_ids if inventory[iid][0] == card_id]
        if same_id != [instance]:
            raise ValueError('HIF hold target has multiple same-art candidates in the draw/discard ledger')
        targets[card_id] = (instance, 'DETAIL_CANDIDATE' if mods else 'CONFIDENT_ZERO')

    low = low_move_layout(image)
    options = dict(capture=lambda: capture()[0], identify_frame=frame_id, journal=scroll)
    plan = kernel.plan_type.value.lower()

    def matches_badge(frame, box, card_id):
        badge = stable_customization_badge_presence(
            (measure_customization_badge(frame, box),), required_frames=1).state.value
        return badge == targets[card_id][1]

    slots = {}
    if targets:
        views = (scan_move_viewports(context, image, count, **options) if low
                 else ((image, None, enumerate(CHOICE_BOXES[:count])),))
        for frame, offset, cells in views:
            geometry = {} if offset is None else dict(scroll_offset=offset)
            for slot, box in cells:
                card_id = reader.identify(frame, box, plan=plan).card_id
                if card_id not in targets:
                    continue
                hif_choice_grid(frame, expected_count=count, target_slot=slot, **geometry)
                if card_id in slots or not matches_badge(frame, box, card_id):
                    raise ValueError('HIF hold target has ambiguous physical identity')
                slots[card_id] = slot
            if slots.keys() == targets.keys():
                break
        if slots.keys() != targets.keys():
            raise ValueError('HIF hold strategy target is not located on the current menu')
        record['targets'] = [dict(instance=instance, card_id=card_id, slot=slots[card_id])
                             for card_id, (instance, _) in targets.items()]
        save('hold_targets_bound')

    picked = []
    for choice in choices:
        if choice.kind is ActionKind.FINISH_SELECTION:
            break
        card_id = inventory[choice.card_instance_id][0]
        slot = slots[card_id]
        current, at = capture()
        geometry = {}
        if low:
            _, offset, _ = next(scan_move_viewports(context, current, count, **options,
                previous=record['scrolls'], target_slot=slot))
            geometry = dict(scroll_offset=offset)
            current, at = capture()
        target = hif_choice_grid(current, expected_count=count, target_slot=slot, **geometry)[slot]
        if not (_control(context, current, 'ProduceRecognitionChooseMoveCards')
                or _control(context, current, 'ProduceMoveCards')):
            raise ValueError('HIF hold page is no longer visible')
        if (reader.identify(current, target, plan=plan).card_id != card_id
                or not matches_badge(current, target, card_id)):
            raise ValueError('HIF hold target changed physical identity before input')
        click('hold_preview' if multi else 'hold_select', target, at, current, slot=slot,
              instance=choice.card_instance_id)

        def detail(frame):
            try:
                box = hif_choice_grid(frame, expected_count=count, target_slot=slot, **geometry)[slot]
                titles = _choice_selected_title_rows(context, frame)
                confirm_hif_choice_title(catalog, ''.join(row.text for row in titles),
                                        candidate_card_ids=(int(card_id),))
            except ValueError:
                return None
            return dict(selected_slot=slot, pages=({'detail_title_rows': tuple(dict(text=row.text) for row in titles)},),
                        image=frame, target=box)

        preview = _poll_move_title(context, slot, capture, frame_id, detail)
        record['inputs'][-1]['target_title_frames'] = list(preview['frame_ids'])
        save('hold_target_title_observed')
        if multi:
            click('hold_select', preview['target'], preview['title_capture_monotonic'][-1], preview['image'],
                  slot=slot, instance=choice.card_instance_id)
            time.sleep(.8)
        picked.append(slot)

    if picked:
        image, at, target = _wait_move_confirm(context, capture, multi=multi)
        click('hold_confirm', target, at, image, picked_slots=tuple(picked))
    else:
        image, at, _ = _poll(context, capture,
            lambda frame: _control(context, frame, 'ProduceRecognitionChooseMoveCards'))
        click('hold_cancel', _HOLD_CANCEL, at, image)
    save('hold_settlement_pending')
    return expected


def replay_confirmed_hif_hold(policy, row, *, origin_dispatch_index):
    """Rebuild a completed hold from its current-format Maa input receipt."""
    from .high_score.hif_mcts_cache import json_value, action_from_json

    saved = row['hold_journal']
    action = policy.response.action
    bundle = policy.bundle
    if (saved.get('stage') != 'hold_settlement_pending'
            or (saved.get('run_id'), saved.get('revision')) != (bundle.run_id, bundle.revision)
            or saved.get('origin') != json_value(asdict(action))):
        raise ValueError('HIF hold has no complete confirmation receipt')
    state = policy.committed_transition()
    if isinstance(state, tuple) or not hold_origin_supported(state, action):
        raise ValueError('HIF hold receipt does not match the original pending choice')
    choices = tuple(action_from_json(raw) for raw in saved['actions'])
    targets = saved.get('targets', ())
    picks = tuple(choice.card_instance_id for choice in choices if choice.kind is ActionKind.HOLD_CARD)
    if (not choices or any(choice.kind not in (ActionKind.HOLD_CARD, ActionKind.FINISH_SELECTION)
            or choice.source_slot is not None for choice in choices)
            or len(picks) != len(set(picks))
            or len(targets) != len(picks) or {target['instance'] for target in targets} != set(picks)
            or not saved.get('inputs')):
        raise ValueError('HIF hold receipt does not match its selected instances')
    slots = {target['instance']: target['slot'] for target in targets}
    inventory = {iid: canon for iid, canon, _ in expected_permanent_inventory_keys(bundle, state)}
    count = len(state.pending_choice.candidate_instance_ids)
    if (len(set(slots.values())) != len(slots)
            or any(type(slot) is not int or not 0 <= slot < count for slot in slots.values())
            or any(inventory.get(target['instance']) != target['card_id'] for target in targets)
            or tuple(saved['inputs'][-1].get('picked_slots', ())) != tuple(slots[iid] for iid in picks)):
        raise ValueError('HIF hold receipt has inconsistent target slots')
    phases = []
    for instance in picks:
        if state.pending_choice.selection_optional:
            phases.append(('hold_preview', instance, slots[instance]))
        phases.append(('hold_select', instance, slots[instance]))
    phases.append(('hold_confirm' if picks else 'hold_cancel', None, None))
    actual = row['native_input_dispatches']
    if type(origin_dispatch_index) is not int or not 0 <= origin_dispatch_index < len(actual):
        raise ValueError('HIF hold receipt lacks its audited primary submission')
    previous = origin_dispatch_index
    for entry, (phase, instance, slot) in zip(saved['inputs'], phases, strict=True):
        index = entry.get('dispatch_index')
        if (entry.get('submitted') is not True or entry.get('phase') != phase
                or entry.get('instance') != instance or entry.get('slot') != slot
                or type(index) is not int or not previous < index < len(actual)):
            raise ValueError('HIF hold receipt does not match its audited inputs')
        x, y, w, h = entry['box']
        dispatch = actual[index]
        if (dispatch.get('kind') == 'swipe'
                or (dispatch.get('x'), dispatch.get('y')) != (x + w//2, y + h//2)):
            raise ValueError('HIF hold receipt differs from its audited input coordinates')
        previous = index
    branches = _resolved_branches(bundle.kernel, policy.committed_event_branches, choices)
    states = tuple(dict.fromkeys(branch.state for branch in branches))
    return states[0] if len(states) == 1 else states


def complete_hif_hold(context, policy, expected):
    """Continue a verified primary action into its supported hold menu."""
    states = expected if isinstance(expected, tuple) else (expected,)
    if not any(state is not None and state.pending_choice for state in states):
        return expected
    handler = getattr(context, 'gkh_hif_hold_transaction', None)
    if not callable(handler):
        raise ValueError('HIF hold requires the formal battle host')
    policy.committed_event_branches = handler(policy)
    from .high_score.hif_fast_native import persist_projected_events

    persist_projected_events(context, policy)
    states = tuple(dict.fromkeys(branch.state for branch in policy.committed_event_branches))
    return states[0] if len(states) == 1 else states
