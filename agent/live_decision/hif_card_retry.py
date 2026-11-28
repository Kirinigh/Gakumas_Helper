"""Compare the visible hand markers and HUD after an unconfirmed selection tap."""
import json
import time

import cv2
import numpy as np

from .exam_hand import exam_hand_marks as hand_marks

CARD_SELECT_TIMEOUT = 1.6
CARD_SELECT_POLL_INTERVAL = .2
_SELECT_MARKER = r'^\s*[Ss]\s*[Ee]\s*[Ll]\s*[Ee]\s*[Cc].{0,5}\s*$'
_SELECT_SUFFIX = r'[Ss][Ee][Ll][Ee][Cc][Tt]$'


def read_exam_hand_strip(context, image):
    node = 'ProduceHIF__ProduceIdentitySelected'
    return context.run_recognition(node, image, pipeline_override={node: dict(
        recognition='OCR', roi=[0, 1040, 720, 160], expected=[_SELECT_MARKER, _SELECT_SUFFIX], only_rec=False)})


def read_exam_select_markers(context, image, *, read_strip=None):
    """Hiro's tolerant SELECT strip read, with a crop retry for joined OCR rows."""
    node = 'ProduceHIF__ProduceIdentitySelected'

    def ocr(roi, expected, *, only_rec=False):
        return context.run_recognition(node, image, pipeline_override={node: dict(
            recognition='OCR', roi=list(roi), expected=expected, only_rec=only_rec)})

    result = read_exam_hand_strip(context, image) if read_strip is None else read_strip(image)
    if not getattr(result, 'hit', False):
        return (), False
    rows = getattr(result, 'filtered_results', ()) or ()
    markers = []
    for row in rows:
        label = ''.join((row.text or '').split()).upper()
        if label.startswith('SELEC') and len(label) <= 10:
            markers.append(tuple(row.box))
            continue
        if (not isinstance(image, np.ndarray) or not label.endswith('SELECT')
                or row.box[2] < 105):
            continue
        x, top, width, height = map(int, row.box)
        y = top + height // 2
        if not 1080 < y < 1140:
            continue
        right = x + width
        left, right = max(0, right - 94), min(image.shape[1], right + 5)
        crop = image[y - 10:y + 25, left:right]
        if not crop.size:
            continue
        orange = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), (14, 130, 178), (30, 255, 255))
        if cv2.countNonZero(orange) > 200:
            roi = (left, y - 10, right - left, 35)
            confirmed = ocr(roi, [_SELECT_MARKER], only_rec=True)
            if confirmed and confirmed.hit:
                markers.append(roi)
    return tuple(markers), bool(markers) or not rows


def recheck_persistent_select(image, anchor, *, selected, capture, stopping):
    """Ignore a stale SELECT during a card animation before tapping the same card."""
    first = selected(image, anchor)
    if first is None:
        return None
    time.sleep(.8)
    return None if stopping() else selected(capture(), first)


def wait_for_target_select(capture, probe, stopping):
    """Read the target marker during the same Hiro selection wait in each host."""
    deadline = time.monotonic() + CARD_SELECT_TIMEOUT
    seen_marker = False
    while not stopping() and time.monotonic() < deadline:
        time.sleep(CARD_SELECT_POLL_INTERVAL)
        if stopping():
            break
        box, any_marker = probe(capture())
        if box is not None:
            return box, True
        seen_marker |= any_marker
    return None, seen_marker


def same_readable_resources(before, after):
    """A missed HUD read alone does not show that the game progressed."""
    return all(old is None or new is None or old == new
               for old, new in zip(before, after, strict=True))


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
            and same_readable_resources((before['stamina'], before['shield']),
                                        (after['stamina'], after['shield'])))


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
