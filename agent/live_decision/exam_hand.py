"""Shared single-frame geometry for a settled exam hand."""

from dataclasses import dataclass

from .hiro_runtime import hiro_module, hiro_ui_templates


@dataclass(frozen=True)
class ExamHandRow:
    box: tuple
    label: str
    score: float


def hand_row_ready(boxes):
    from .upstream_hif_card_ui import BattleRecognitionMixin

    if not boxes:
        return True
    # A clipped detector box can cover only the part before the next card.
    return (abs(boxes[0][0] + boxes[-1][0] + max(box[2] for box in boxes) - 720) <= 24
            and not any(BattleRecognitionMixin._hif_card_is_raised(box) for box in boxes))


def _letter_hand(image, detected, detector):
    if (getattr(image, 'shape', ()) != (1280, 720, 3)
            or not detected or any(detector._hif_card_is_raised(box) for box in detected)):
        return None
    try:
        with hiro_ui_templates():
            locator = hiro_module('vision.hand_slots')
            slots = sorted(locator.hand_slots(image), key=lambda slot: slot.letter.center.x)
            if not 1 <= len(slots) <= 5 or locator.raised_card(image) is not None:
                return None
    except (ImportError, FileNotFoundError):
        return None
    centers = [slot.letter.center.x for slot in slots]
    gaps = [right - left for left, right in zip(centers, centers[1:])]
    tops = [slot.letter.y1 for slot in slots]
    if (gaps and (min(gaps) < 100 or max(gaps) > 225 or max(gaps) - min(gaps) > 12)
            or max(tops) - min(tops) > 8):
        return None
    boxes = tuple((x - 96, y - 200, 192, 248) for x, y in zip(centers, tops, strict=True))
    if not hand_row_ready(boxes) or boxes[0][0] < 0 or boxes[-1][0] + boxes[-1][2] > 720:
        return None

    # The two edges must still agree with Maa. The badge proves a missing
    # interior slot on this very image, irrespective of the card's name.
    matched = {}
    fragments = []
    for candidate in detected:
        x, y, _, _ = candidate
        options = [i for i, (left, top, _, _) in enumerate(boxes) if abs(x - left) <= 26 and abs(y - top) <= 24]
        if not options:
            fragments.append(candidate)
        elif len(options) != 1 or options[0] in matched:
            return None
        else:
            matched[options[0]] = candidate
    if 0 not in matched or len(boxes) - 1 not in matched:
        return None
    # A narrow detection can be an extra piece of a card that already has a
    # complete physical slot. Do not let the fragment become an extra slot.
    if any(box[2] >= 135 or not any(abs(box[1] - boxes[i][1]) <= 24
               and detector._hif_duplicate_box(box, boxes[i]) for i in matched) for box in fragments):
        return None
    return boxes, matched, [slot.available for slot in slots]


def anchor_exam_hand(image, rows, detector):
    """Prefer badge positions and preserve actual Maa recommendations and scores."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return None
    by_box = {}
    for row in rows:
        if row.label not in ('cards', 'suggestions', 'useless'):
            continue
        x, y, width, height = row.box
        bottom = y + height
        # A recommendation can extend the detector box past the hand filter.
        # Only the badge-derived rectangle is used, and the same-frame
        # detector must still support this slot at the top and left.
        if 1170 < bottom <= 1178:
            height = 1170 - y
        box = (x, y, width, height)
        if box not in by_box or row.score > by_box[box].score:
            by_box[box] = row
    geometry = [ExamHandRow(box, 'cards', row.score) for box, row in by_box.items()]
    detected = tuple(tuple(box) for box, _ in detector._hif_card_boxes(geometry))
    binding = _letter_hand(image, detected, detector)
    if binding is None:
        return None
    boxes, matched, available = binding
    seen = {slot: by_box[box] for slot, box in matched.items()}
    restored = []
    for i, box in enumerate(boxes):
        row = seen.get(i)
        label, score = (row.label, row.score) if row is not None else ('cards', 0.)
        restored.append(ExamHandRow(box, label if available[i] else 'useless', score))
    return tuple(restored)


def playable_exam_boxes(anchored):
    return [(row.box, row.score) for row in anchored if row.label != 'useless']


def read_hiro_upgrade_suffix(image, box, *, center_x=None):
    """Read the upgrade glyphs from a title, keeping clipped suffixes unknown."""
    x, y, width, height = box
    try:
        return hiro_module('vision.hif').plus_count(image[y:y + height, x:x + width], center_x=center_x)
    except (ImportError, FileNotFoundError):
        return None


def read_hiro_hand_slot(image, box, slot, ocr):
    """Read one card using Hiro's original artwork and title reader."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return None
    from .hiro_maa_device import hiro_maa_ocr

    x, y, _, _ = box
    try:
        with hiro_ui_templates():
            locator = hiro_module('vision.hand_slots')
            letters = sorted(locator.card_letters(image), key=lambda letter: letter.center.x)
            matches = [i for i, letter in enumerate(letters)
                       if abs(letter.center.x - (x + 96)) <= 26 and abs(letter.y1 - (y + 200)) <= 24]
            if len(matches) != 1:
                return None
            index, = matches
            next_center = letters[index + 1].center.x if index + 1 < len(letters) else None
            with hiro_maa_ocr(ocr):
                return hiro_module('vision.hif')._read_card(image, slot, letters[index], next_center)
    except (ImportError, FileNotFoundError):
        return None


def exam_hand_marks(image):
    """Hiro's visible letter positions for a before/after retry comparison."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return ()
    try:
        with hiro_ui_templates() as templates:
            locator = hiro_module('vision.hand_slots')
            letters = locator.card_letters(image)
            # A slightly lifted edge card can leave Hiro's normal letter strip
            # without reaching its central raised-card region. Reuse the same
            # upstream templates and raised-card threshold in the gap.
            upper_strip = (0, 1040, 720, 39)
            letters.extend(hit.rect for kind in ('A', 'M')
                           for hit in templates.tpl(f'InProduce.{kind}').find_all(
                               image, region=upper_strip, threshold=locator.RAISED_MIN))
    except (ImportError, FileNotFoundError):
        return ()
    return tuple(sorted({(letter.x1 // 8, letter.y1 // 8) for letter in letters}))
