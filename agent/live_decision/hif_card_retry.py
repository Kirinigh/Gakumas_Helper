"""Compare the visible hand markers and HUD after an unconfirmed selection tap."""
import json
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np


@lru_cache
def _letter_templates():
    # A, M and T from our adjacent-gray acceptance frame, not upstream assets.
    strip = cv2.imread(str(Path(__file__).with_name('assets') / 'hif_hand_letters_0067.png'),
                       cv2.IMREAD_GRAYSCALE)
    return () if strip is None or strip.shape != (42, 18) else np.vsplit(strip, 3)


def hand_marks(image):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return ()
    band = cv2.cvtColor(image[1040:1100], cv2.COLOR_BGR2GRAY)
    markers = set()
    for template in _letter_templates():
        scores = np.abs(cv2.matchTemplate(band, template, cv2.TM_CCOEFF_NORMED))
        maxima = scores == cv2.dilate(scores, np.ones((9, 15), np.uint8))
        ys, xs = np.where(maxima & (scores >= .85))
        markers.update((int(x) // 8, (1040 + int(y)) // 8) for x, y in zip(xs, ys, strict=True))
    return tuple(sorted(markers))


def read_retry_scene(module, context, image):
    marks = hand_marks(image)
    if not marks:
        return None
    fields = {}
    for name in ('turns_remaining', 'stamina', 'shield'):
        roi, pattern = ((module.LIVE_SHIELD_ROI, r'^\d{1,3}$') if name == 'shield'
                        else module.LIVE_LAYOUT_V1[name])
        try:
            row = module._ocr_field(context, image, name, roi, pattern)
            fields[name] = row.get('value') if row.get('accepted') else None
        except (ValueError, RuntimeError, OSError):
            fields[name] = None
    return dict(fields, marks=marks) if fields['turns_remaining'] is not None else None


def same_retry_scene(before, after):
    if before is None or after is None:
        return False
    return (before['marks'] == after['marks'] and before['turns_remaining'] == after['turns_remaining']
            and all(before[name] is None or after[name] is None or before[name] == after[name]
                    for name in ('stamina', 'shield')))


def observe_after_unconfirmed(module, context, params, baseline, ui=None, *, image=None, selection_retry=None):
    """Read the actual current page without submitting or binding a planned action."""
    from .hif_result import try_displayed_terminal_frame

    if image is None:
        image = ui.context.tasker.controller.post_screencap().wait().get()
    if ui is not None:
        selection_retry = ui.receipt.get('selection_retry')
    observed = dict(event='live_reobserved_battle', reason='unconfirmed_selection',
        frame_id=module.frame_identifier(image), classification='unreadable', core=None,
        selection_retry=selection_retry, execution_submit_count=0, executed=False)
    terminal = try_displayed_terminal_frame(context, image, (), frame_identifier=module.frame_identifier)
    if terminal is not None:
        context.gkh_hif_terminal_event = dict(terminal, event='live_committed_state',
            execution_submit_count=0, executed=False)
        observed.update(classification='terminal', result=context.gkh_hif_terminal_event)
    else:
        try:
            recognizer = module.EmbeddingCardRecognizer.load(module.MODEL_ROOT)
            current = module._inventory_live_checkpoint(context, (image,), recognizer, single_frame=True,
                identity_threshold=params['debug_identity_threshold'], identity_min_margin=params['debug_identity_min_margin'])
            core = {key:current[key] for key in ('fields', 'families', 'frame_ids')}
            prior = baseline['core']
            unchanged = (json.dumps(core['fields'], sort_keys=True) == json.dumps(prior['fields'], sort_keys=True)
                and tuple(map(tuple, core['families'])) == tuple(map(tuple, prior['families'])))
            observed.update(core=core, classification='unchanged' if unchanged else 'changed')
        except ValueError as error:
            observed.update(reason=str(error), observations=list(getattr(error, 'events', ())))
    context.gkh_hif_reobserved_battle = observed
    return observed
