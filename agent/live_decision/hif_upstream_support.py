"""Current selected effects over the pinned upstream UI, without new input."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np

from .hif_upstream_ui import UPSTREAM_REVISION
from .hif_skill_support import (
    _text,
    _qualified,
    _panel_bounds,
    _read_detail_title,
    _read_selected_support_details,
    locate_hif_selected_detail_title,
    read_selected_skill_support_frame,
    verify_selected_skill_support_observations,
)


def selection_evidence(frame_ids, slot):
    return dict(kind='target_select_detail' if len(frame_ids) == 1 else 'upstream_selected_detail',
        revision=UPSTREAM_REVISION,
        selected_slot=slot, frame_ids=list(frame_ids), select_and_detail_confirmed=True)


def read_current_support(ui, policy):
    """Use the pinned selection frames to distinguish ordinary and enhanced effects."""
    frames = ui.selected_frames
    current_target = ui.receipt.get('confirmation_mode') == 'observe_next_page'
    if current_target and (len(frames) != 1 or ui.receipt.get('selected_target_frame') != frames[0][0]):
        raise ValueError('selected effect observation differs from the current target marker')
    if not current_target and len(frames) != 2:
        raise ValueError('upstream selected detail frames are incomplete')
    action = policy.response.action
    state = policy.bundle.state
    ref = state.deck.hand[action.source_slot]
    if current_target and ref.card_id.split('@', 1)[0] == '816':
        return None
    if any(fact.instance_id == ref.instance_id and fact.turn == state.turn
           and fact.projection_kind == 'verified_effects' for fact in state.hand_support):
        return None
    ordinary = policy.bundle.kernel.cards[ref.card_id]
    last_id, image = frames[-1]
    roi = locate_hif_selected_detail_title(image)
    if roi is None:
        raise ValueError('upstream selected detail disappeared')
    panel = _panel_bounds(image, roi)
    x, y, w, h = roi
    crop = image[y:y+h, x:x+w]
    cache = getattr(ui.base, 'gkh_hif_ordinary_title_pixels', None)
    if cache is None or cache[0] != policy.bundle.run_id:
        cache = (policy.bundle.run_id, {})
        ui.base.gkh_hif_ordinary_title_pixels = cache
    key = (ref.card_id, ordinary.name)
    previous = cache[1].get(key)
    if previous is not None and crop.dtype == previous.dtype and np.array_equal(crop, previous):
        return None

    title_reading = _read_detail_title(ui.base, image, roi, panel)
    titles, _, excluded = title_reading
    if len(titles) != 1 or titles[0]['confidence'] < .95:
        raise ValueError('upstream selected title is unresolved')
    if _text(titles[0]['text']) == _text(ordinary.name):
        # The complete title (including its suffix) was read from this crop.
        # A cost-icon repair also consults pixels outside it, so cannot seed the cache.
        if not excluded and _qualified(titles, roi):
            cache[1][key] = crop.copy()
        return None
    hand_count = len(state.deck.hand)
    observations = [read_selected_skill_support_frame(ui.base, earlier, frame_id=key,
        selected_slot=action.source_slot, hand_count=hand_count) for key, earlier in frames[:-1]]
    observations.append(_read_selected_support_details(ui.base, image, roi, panel, frame_id=last_id,
        selected_slot=action.source_slot, hand_count=hand_count, title_reading=title_reading))
    receipt = verify_selected_skill_support_observations(observations, bundle=policy.bundle,
        selected_slot=action.source_slot, selection_evidence=selection_evidence(
            [key for key, _ in frames], action.source_slot))
    if receipt.get('effects_verified') is not True:
        ui.receipt['unresolved_selected_skill_support'] = receipt
        ui.publish()
        raise ValueError('upstream current support is not resolved: '+str(receipt.get('reason')))
    policy.bind_selected_skill_support(receipt)
    ui.receipt.update(selected_skill_support=receipt, selected_detail=receipt['selection_evidence'])
    ui.publish()
    return receipt


def bind_recorded_support(policy, journal):
    receipt = journal.get('selected_skill_support')
    if receipt is None:
        return False
    if (journal.get('revision') != UPSTREAM_REVISION
            or receipt['selection_evidence'] != journal.get('selected_detail')
            or receipt.get('selected_slot') != journal.get('action', {}).get('source_slot')):
        raise ValueError('upstream support differs from the original selected action')
    confirmations = [r for r in journal.get('inputs', ()) if r.get('phase') == 'confirm']
    if receipt['selection_evidence'].get('kind') == 'target_select_detail' and (
            journal.get('selected_target_frame') != receipt['frame_ids'][0]
            or any(row.get('frame_id') != journal.get('selected_target_frame') for row in confirmations)):
        raise ValueError('upstream support differs from the confirmed target frame')
    if confirmations and confirmations[0].get('frame_id') not in receipt['frame_ids']:
        refreshed = journal.get('refreshed_detail_frames', ())
        if (journal.get('detail_refreshed_for_input') is not True or len(set(refreshed)) != 2
                or confirmations[0].get('frame_id') != refreshed[-1]):
            raise ValueError('upstream support lost its selected input binding')
    policy.bind_selected_skill_support(receipt)
    return True


def supported_boundary(bundle, action, boundary, confirmation):
    receipt = (confirmation or {}).get('selected_skill_support')
    if receipt is None:
        return boundary
    from .high_score.live_policy import BoundHifOpeningPolicy

    policy = BoundHifOpeningPolicy(bundle, action_order=(), hold_ready=True)
    policy.response = SimpleNamespace(action=action)
    policy.bind_selected_skill_support(receipt)
    expected = policy.committed_transition()
    states = expected if isinstance(expected, tuple) else (expected,)
    if any(state.pending_choice for state in states):
        from .high_score.hif_information_boundary import InformationBoundary

        return InformationBoundary('synchronize', 'pending_resolution', ('choice', 'core'))
    if boundary.successor is not None:
        if len(states) != 1 or states[0].turn != bundle.state.turn:
            raise ValueError('observed support changed the deterministic action boundary')
        return replace(boundary, successor=states[0], reason='observed_current_play_support')
    return boundary
