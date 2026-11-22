# SPDX-License-Identifier: AGPL-3.0
"""HIF move-list geometry and measured scrolling from miyu fb8fd00c.

Source: extensions/hif/agent/hif/action/battle_recognition.py and common.py.
The caller binds the physical slot to the planned card; Maa owns all inputs.
"""
import time

import cv2
import numpy as np


def low_move_layout(image):
    if getattr(image, 'shape', None) != (1280, 720, 3):
        return False
    probe = image[440:476, 60:660]
    return float(probe.mean()) >= 200 and float(probe.std()) <= 6


def visible_move_cells(count, offset):
    for row in range(min(40, (count + 3) // 4)):
        y = 569 + 147 * row - offset
        if 560 <= y <= 1040:
            for column, x in enumerate((139, 286, 433, 580)):
                slot = row * 4 + column
                if slot < count:
                    yield slot, (x - 59, y - 59, 118, 118)


def _view(image):
    return image[500:1100, 75:645]


def _still(first, second):
    return np.abs(_view(first).astype(np.int16) - _view(second).astype(np.int16)).mean() <= 2.5


def measured_move_shift(before, after, direction):
    """fb8 _hif_list_scroll_shift validates each candidate against the entire overlap."""
    first, second = (cv2.GaussianBlur(cv2.cvtColor(_view(image), cv2.COLOR_BGR2GRAY), (5, 5), 0)
                     for image in (before, after))
    band = 120
    start = len(first) - band - 10 if direction > 0 else 10
    template = first[start:start + band]
    if float(template.std()) < 8:
        return None
    scores = cv2.matchTemplate(second, template, cv2.TM_CCOEFF_NORMED).ravel()
    shifts = start - np.arange(len(scores))
    scores[shifts * direction < -3] = -1
    verified = []
    for _ in range(4):
        best = int(np.argmax(scores))
        if float(scores[best]) < .80:
            break
        scores[max(0, best-8):best+9] = -1
        delta = int(shifts[best])
        overlap = len(first) - abs(delta)
        if overlap >= band:
            a, b = (first[delta:], second[:overlap]) if delta >= 0 else (first[:overlap], second[-delta:])
            verified.append((float(cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED).max()), delta))
    verified.sort(reverse=True)
    if not verified or verified[0][0] < .85 or (len(verified) > 1 and verified[0][0] - verified[1][0] < .06):
        return None
    delta = verified[0][1]
    return 0 if abs(delta) <= 3 else delta


def _move_control(context, image):
    for node in ('ProduceRecognitionChooseMoveCards', 'ProduceMoveCards'):
        found = context.run_recognition(node, image)
        if found and found.hit:
            return True
    return False


def scan_move_viewports(context, initial, count, *, capture, identify_frame, journal, previous=(),
                        target_slot=None):
    """Yield each newly visible slot. A resume continues from its recorded pixel offset."""
    history = list(previous)
    image = initial
    offset = history[-1].get('offset_after') if history else 0
    scanned = set()
    if not _move_control(context, initial):
        raise ValueError('HIF move list is not visible')

    def swipe(direction):
        nonlocal image, offset
        if context.tasker.stopping:
            raise ValueError('HIF move-list navigation stopped')
        start, end = (1000, 860) if direction == 1 else (860, 1000)
        controller = context.tasker.controller
        from utils.maa_input import verify_maa_input_endpoints

        endpoints = verify_maa_input_endpoints(controller.info,
            expected_client_size=getattr(context, 'gkh_hif_client_size', (720, 1280)))
        receipt = dict(direction=direction, start=(360, start), end=(360, end), duration=1000,
            offset_before=offset, offset_after=None, before_frame=identify_frame(image, 0),
            endpoints=endpoints, state='dispatch_pending')
        history.append(receipt)
        journal('pending_ui_captured', {'choice_scrolls': history})
        job = controller.post_swipe(360, start, 360, end, 1000).wait()
        if not job.succeeded:
            raise ValueError('HIF move-list swipe did not complete')
        before, last = image, None
        deadline = time.monotonic() + 2.
        while last is None or time.monotonic() < deadline:
            time.sleep(.2)
            image = capture()
            if last is not None and _still(last, image):
                break
            last = image
        delta = measured_move_shift(before, image, direction)
        if direction < 0 and _still(before, image):
            offset = 0
        elif delta is None or offset is None:
            offset = None
        else:
            offset += delta
        receipt.update(state='dispatched', after_frame=identify_frame(image, 0), delta=delta, offset_after=offset)
        journal('pending_ui_captured', {'choice_scrolls': history})
        return delta

    def recover_top():
        for _ in range(16):
            swipe(-1)
            if offset == 0:
                return
        raise ValueError('HIF move-list top could not be located')

    if offset is None:
        # A dispatched or interrupted scroll may have moved a different distance.
        # miyu recovers the top by observing that an upward list navigation no longer moves it.
        recover_top()
    if target_slot is not None:
        for _ in range(16):
            cells = dict(visible_move_cells(count, offset))
            if target_slot in cells:
                yield image, offset, [(target_slot, cells[target_slot])]
                return
            if not cells or not swipe(1 if target_slot > max(cells) else -1) or offset is None:
                break
        raise ValueError('HIF move-list target could not be brought into view')
    for attempt in range(15):
        cells = dict(visible_move_cells(count, offset))
        fresh = [(slot, box) for slot, box in cells.items() if slot not in scanned]
        if fresh:
            yield image, offset, fresh
            scanned.update(slot for slot, _ in fresh)
        if count - 1 in cells or attempt == 14:
            break
        delta = swipe(1)
        if not delta:
            if delta is None:
                recover_top()
            break
