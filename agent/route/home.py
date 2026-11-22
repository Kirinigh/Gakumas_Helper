"""Read the active game home and AP without starting or completing a run."""
from pathlib import Path

import cv2
import numpy as np
from utils.runtime_paths import resolve_runtime_path


def _unique_home_template(image, name, region, threshold):
    template = cv2.imread(str(resolve_runtime_path(Path('assets/resource/base/image') / name,
        Path('resource/base/image') / name)))
    if template is None:
        raise ValueError('existing home template is unavailable')
    x, y, width, height = region
    values = cv2.matchTemplate(image[y:y + height, x:x + width], template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (left, top) = cv2.minMaxLoc(values)
    count, _ = cv2.connectedComponents((values >= threshold).astype('uint8'))
    if peak < threshold or count != 2:
        raise ValueError('current home control is not uniquely observed: ' + name)
    return dict(box=(x + left, y + top, template.shape[1], template.shape[0]),
        confidence=float(peak)), template


def _home_ap(context, image, rows):
    from .production import _text
    try:
        ap = _text(rows, r'\d{1,3}/\d{1,3}', (310, 20, 165, 55))
    except ValueError:
        if context is None:
            raise
        from live_decision.hif_recovery import _field
        # The full detector can join the left AP icon to the number box.
        ap = _field(context, image, 'ActiveHomeAP', (340, 20, 115, 45), r'\d{1,3}/\d{1,3}')['text']
    current, maximum = map(int, ap.split('/'))
    if maximum == 0 or not 0 <= current <= maximum:
        raise ValueError('contradictory current home AP')
    return current, maximum


def read_active_home(image, rows, *, saved_day=None, context=None, preview_saved=False, optional_preview_ap=False,
                     free_start=False):
    """Identity only; the caller still owns run, completion and start authority.

    Reuses the original home/produce templates. Correlation alone survives a
    dark modal overlay, so the selected home glyph must also retain its actual
    orange foreground and full brightness.
    """
    from .production import _row
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('unsupported HIF home capture resolution')
    # Locked upstream ProduceHomeFlag/Start/Continue identify the existing
    # buttons by image. Unrelated navigation labels are not prerequisites.
    if type(preview_saved) is not bool or preview_saved and saved_day is not None:
        raise ValueError('saved preview cannot claim an observed saved day')
    if preview_saved:
        # This only locates the existing in-progress control to open its
        # read-only save preview. The preview must identify role/day before
        # any resume input; no saved day is inferred from the button template.
        produce = None
        button_spec = ('home_1.png', (200, 930, 320, 140), .8)
    elif saved_day is None:
        produce = None if free_start else _row(rows, r'プロデュース', (225, 940, 300, 115))
        # The normal and in-progress buttons use different templates.
        button_spec = ('home.png', (200, 920, 320, 170), .8)
    else:
        if type(saved_day) is not int or not 1 <= saved_day <= 7:
            raise ValueError('current saved home day is not explicitly bound')
        # The ordinal suffix carries no day value (actual 121 reads 目 as 月).
        # Keep the exact numeric day, full caption and strong confidence.
        day_caption = rf'{saved_day}日\D'
        try:
            produce = _row(rows, day_caption, (300, 1020, 140, 90))
        except ValueError:
            if context is None:
                raise
            from live_decision.hif_recovery import _field
            # 145's full-frame 1日目 is .902718. Independently read the complete
            # saved-day caption, retaining the full active-home/template gates.
            produce = _field(context, image, 'ActiveSavedDay', (315, 1030, 90, 50), day_caption)
        # Original ProduceContinue uses this existing in-progress template's .8
        # profile. The active home glyph and exact day are required as well.
        button_spec = ('home_1.png', (200, 930, 320, 140), .8)
    home, reference = _unique_home_template(image, 'homeflag.png', (315, 1100, 95, 100), .95)
    button, _ = _unique_home_template(image, *button_spec)
    reference_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    orange = ((reference_hsv[:, :, 0] >= 5) & (reference_hsv[:, :, 0] <= 30)
        & (reference_hsv[:, :, 1] >= 150) & (reference_hsv[:, :, 2] >= 230)).astype('uint8')
    orange = cv2.erode(orange, np.ones((3, 3), dtype='uint8')).astype(bool)
    if orange.sum() < 100:
        raise ValueError('existing home foreground template is incomplete')
    x, y, width, height = home['box']
    observed = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_BGR2HSV)
    active = ((observed[:, :, 0] >= 5) & (observed[:, :, 0] <= 30)
        & (observed[:, :, 1] >= 150) & (observed[:, :, 2] >= 230))
    fraction = float(active[orange].mean())
    if fraction < .95:
        raise ValueError('current home is obscured or inactive')
    try:
        current, maximum = _home_ap(context, image, rows)
    except ValueError:
        if not (free_start or preview_saved and optional_preview_ap):
            raise
        # These buttons only open the scenario or save preview. The actual
        # paid start still needs its own observed quote and single-run budget.
        current, maximum = None, None
    result = dict(scene='home', ap=current, max_ap=maximum, target=tuple((produce or button)['box']),
        home=home, produce=button, active_foreground_fraction=fraction)
    if saved_day is not None:
        result['saved_day_evidence'] = produce
    return result
