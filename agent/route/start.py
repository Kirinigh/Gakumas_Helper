"""Observe a supported start confirmation; never grant or send its AP input."""
import re
import unicodedata

import cv2
import numpy as np

from .controls import read_checkbox


def _independent_start_quote(context, image):
    """Use the SDK's match, without an additional local confidence veto."""
    node = 'HifRecoveryStartAPQuote'
    roi = (315, 1095, 130, 35)
    x, y, width, height = roi
    hit = context.run_recognition(node, image[y:y + height, x:x + width].copy(),
        pipeline_override={node: dict(recognition='OCR', only_rec=True, expected='.+')})
    rows = list(hit.filtered_results) if hit and hit.hit else []
    if len(rows) != 1:
        raise ValueError('current start AP quote is not uniquely observed')
    row = rows[0]
    text = ''.join(unicodedata.normalize('NFKC', row.text).split())
    if not re.fullmatch(r'\d{1,3}→\d{1,3}', text):
        raise ValueError('current start AP quote is incomplete')
    return dict(text=text, confidence=float(row.score), box=roi)


def start_price_arrow(image):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('unsupported HIF start capture resolution')
    crop = image[1095:1130, 350:405]
    light = crop.min(2)
    mask = ((light >= 185) & (light <= 210)).astype('uint8')
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    matches = []
    for label, row in enumerate(stats[1:], 1):
        x, y, width, height, area = map(int, row)
        if not (2 <= x and x + width <= 53 and 2 <= y and y + height <= 33
                and 20 <= width <= 22 and 13 <= height <= 15 and 60 <= area <= 90):
            continue
        glyph = labels[y:y + height, x:x + width] == label
        sums = glyph.sum(1)
        if np.any(sums == 0):
            continue
        right = np.array([np.flatnonzero(line)[-1] for line in glyph])
        middle = np.flatnonzero(sums >= width - 1)
        if (len(middle) != 2 or middle[1] != middle[0] + 1
                or not 5 <= middle[0] <= 7 or right[middle[0]] != width - 1
                or not width // 2 <= right[0] <= width // 2 + 3
                or not width // 2 <= right[-1] <= width // 2 + 3
                or np.any(np.diff(right[:middle[0] + 1]) < 0)
                or np.any(np.diff(right[middle[1]:]) > 0)):
            continue
        matches.append((x + 350, y + 1095, width, height))
    if len(matches) != 1:
        raise ValueError('current AP price is not one complete rightward arrow')
    return matches[0]


def read_start_confirmation(context, image, rows):
    from live_decision.hif_session import _text as identity_field
    from live_decision.hif_recovery import _field

    from live_decision.hif_profiles import notebook_hif_identity

    from .production import _row
    # The payable control, selected idol and page identify this confirmation.
    # The explanatory caption above the memory is not an input or an identity.
    identity_field(rows, r'2[.．]開始確認', (30, 15, 210, 65))
    title = identity_field(rows, r'\[.+\]', (355, 104, 365, 55))
    idol = identity_field(rows, r'.+', (500, 145, 215, 90))
    profile = notebook_hif_identity(title[1:-1], idol)
    target = _row(rows, r'プロデュース開始', (210, 1035, 310, 115))
    boxes = ((531, 1007, 30, 30), (619, 1007, 30, 30))
    # The second start checkbox has a 62-pixel selected glyph in retained 013;
    # the interval's separate, unchanged profile keeps its original 60 maximum.
    boosters = tuple(read_checkbox(image, box, selected_area_max=65) for box in boxes)
    try:
        quote = _row(rows, r'\d{1,3}→\d{1,3}', (265, 1090, 190, 50))
    except ValueError:
        complete = [row for row in rows if re.fullmatch(r'\d{1,3}→\d{1,3}', row['text'])
            and 265 <= row['box'][0] and 1090 <= row['box'][1]
            and row['box'][0] + row['box'][2] <= 455 and row['box'][1] + row['box'][3] <= 1140]
        if len(complete) > 1:
            raise ValueError('current start AP quote is not uniquely observed')
        if not complete:
            # Older layouts without a complete quote retain their independently
            # observed arrow and two current number crops.
            x, y, width, height = start_price_arrow(image)
            before = _field(context, image, 'StartCurrentAP', (x - 48, y - 8, 43, 30), r'\d{1,3}')
            after = _field(context, image, 'StartRemainingAP', (x + width + 8, y - 8, 45, 30), r'\d{1,3}')
            current, remaining = int(before['text']), int(after['text'])
            price_evidence = dict(arrow=(x, y, width, height), before=before, after=after)
            quote = None
        else:
            # Read both numbers and direction together, excluding the AP icon.
            # Like upstream OCR consumers, use the SDK's recognized result.
            # Exact direction, amounts and independent agreement remain required.
            quote = _independent_start_quote(context, image)
            if quote['text'] != complete[0]['text']:
                raise ValueError('current start AP quote readings disagree')
    if quote is not None:
        current, remaining = map(int, quote['text'].split('→'))
        price_evidence = dict(quote=quote)
    if current < 20 or remaining < 0 or current - remaining != 20:
        raise ValueError('current start price is not exactly 20 AP')
    return dict(scene='start_confirmation', idol=profile.identity, plan=profile.plan,
        target=tuple(target['box']), booster_selected=boosters, booster_boxes=boxes,
        ap=current, remaining_ap=remaining, ap_cost=current - remaining,
        price_evidence=price_evidence)
