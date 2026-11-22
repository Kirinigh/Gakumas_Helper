"""Read the remaining HIF turn table without predicting its attribute order."""

from __future__ import annotations

import re
import math
import time
import unicodedata
from dataclasses import asdict, dataclass
from collections.abc import Mapping

import numpy as np

from .hif_session import resolve_hif_round_context
from .hif_capture_adapter import captured_at as frame_captured_at

ATTRIBUTES = {"ボーカル": "vocal", "ダンス": "dance", "ビジュアル": "visual"}


@dataclass(frozen=True)
class TurnInformation:
    current_turn: int
    turn_attributes: tuple[str, ...]
    score_percentages: tuple[int, int, int]

    @property
    def multipliers(self) -> tuple[float, ...]:
        values = dict(zip(ATTRIBUTES.values(), self.score_percentages))
        return tuple(values[name] / 100 for name in self.turn_attributes)


def current_turn_from_remaining(turns_remaining, *, round_context=None):
    """Compute a turn only within the explicitly bound round's known length."""
    _, base_turn_count = resolve_hif_round_context(round_context)
    if type(turns_remaining) is not int or not 1 <= turns_remaining <= base_turn_count:
        raise ValueError('turn information remaining count contradicts its notebook round')
    return base_turn_count + 1 - turns_remaining


def parse_turn_information(rows, *, current_turn: int, turns_remaining: int,
                           round_context=None) -> TurnInformation:
    """Parse a complete observed table; round-two navigation is not calibrated.

    The three percentage columns are the calibrated vocal/dance/visual header.
    The table keeps past rows. Their explicit 前 labels establish ordering;
    only the current and future rows are returned to the strategy.
    """
    round_number, base_turn_count = resolve_hif_round_context(round_context)
    if (type(current_turn) is not int or type(turns_remaining) is not int
            or min(current_turn, turns_remaining) < 1
            or current_turn + turns_remaining != base_turn_count + 1):
        raise ValueError("HIF turn information requires the observed complete round")
    normalized = []
    for row in rows:
        if isinstance(row, Mapping):
            text, score, box = row.get("text"), row.get("score"), row.get("box")
        else:
            text, score, box = row.text, row.score, row.box
        if (not isinstance(text, str) or type(score) not in (int, float)
                or not math.isfinite(score) or not 0 <= score <= 1
                or not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(value) is not int for value in box)):
            raise ValueError("turn information contains malformed OCR evidence")
        x, y, width, height = box
        if width <= 0 or height <= 0 or min(x, y) < 0 or x + width > 720 or y + height > 1280:
            raise ValueError("turn information OCR box is outside the image")
        normalized.append((unicodedata.normalize("NFKC", text).replace(" ", ""), score, tuple(box)))

    headings = [row for row in normalized if re.fullmatch(r"現在\d+ターン目", row[0])]
    if len(headings) != 1 or headings[0][0] != f"現在{current_turn}ターン目" or headings[0][1] < .95:
        raise ValueError("turn information current-turn heading is unresolved")
    heading = headings[0][2]
    # The round objective is an independent panel anchor, not a battle score.
    goals = [row for row in normalized if re.fullmatch(r"\d+位以上でラウンド\d+突破", row[0])]
    if len(goals) != 1 or goals[0][1] < .95 or not heading[1] < goals[0][2][1] < heading[1] + 90:
        raise ValueError("turn information round objective is unresolved")
    if not re.fullmatch(rf'\d+位以上でラウンド{round_number}突破', goals[0][0]):
        raise ValueError('turn information objective contradicts its notebook round')

    # The OCR detector can split a leading digit from ターン後. Join only an
    # independently observed, unique adjacent digit on the same visible row.
    for suffix in tuple(row for row in normalized if row[0] in {"ターン後", "ターン前"}):
        sx, sy, _, sh = suffix[2]
        digits = [row for row in normalized if re.fullmatch(r"[1-9]\d?", row[0])
                  and row[2][0] < sx and -15 <= sx - (row[2][0] + row[2][2]) <= 15
                  and abs(row[2][1] + row[2][3] / 2 - (sy + sh / 2)) <= 10]
        if len(digits) != 1 or min(suffix[1], digits[0][1]) < .95:
            raise ValueError("turn information split row label is unresolved")
        digit = digits[0]
        x, y = digit[2][0], min(digit[2][1], sy)
        right = max(digit[2][0] + digit[2][2], sx + suffix[2][2])
        bottom = max(digit[2][1] + digit[2][3], sy + sh)
        normalized.remove(suffix)
        normalized.remove(digit)
        normalized.append((digit[0] + suffix[0], min(digit[1], suffix[1]), (x, y, right-x, bottom-y)))

    labels = {}
    for text, confidence, box in normalized:
        match = re.fullmatch(r"([1-9]\d?)ターン(前|後)", text)
        if text != "現在のターン" and not match:
            continue
        offset = int(match[1]) * (-1 if match[2] == '前' else 1) if match else 0
        # A label also has to occur once in the complete, geometrically ordered
        # table. This redundant numbering tolerates the small-font 541px OCR
        # case (3ターン後 at .943); attributes and score values remain at .95.
        if (offset in labels or not 1-current_turn <= offset < turns_remaining or confidence < .90
                or not 20 <= box[0] < 190 or box[0] + box[2] > 220):
            raise ValueError("turn information has ambiguous or out-of-range row labels")
        labels[offset] = box
    offsets = tuple(range(1-current_turn, turns_remaining))
    if set(labels) != set(offsets):
        raise ValueError("turn information full round rows are incomplete")
    centers = [labels[index][1] + labels[index][3] / 2 for index in offsets]
    if any(b - a < 45 or b - a > 120 for a, b in zip(centers, centers[1:])):
        raise ValueError("turn information rows are not in visible sequential order")

    attributes = []
    attribute_rows = [row for row in normalized if row[0] in ATTRIBUTES and row[2][0] >= 230]
    for offset, center in zip(offsets, centers):
        candidates = [row for row in attribute_rows
                      if abs(row[2][1] + row[2][3] / 2 - center) <= 20]
        if len(candidates) != 1 or candidates[0][1] < .95 or candidates[0][2][0] + candidates[0][2][2] > 390:
            raise ValueError(f"turn information attribute is unresolved at offset {offset}")
        if offset >= 0:
            attributes.append(ATTRIBUTES[candidates[0][0]])
    if len(attribute_rows) != base_turn_count:
        raise ValueError("turn information contains an unmatched attribute row")

    percentages = [row for row in normalized if re.fullmatch(r"\d+%", row[0])
                   and goals[0][2][1] + goals[0][2][3] < row[2][1] < labels[offsets[0]][1]]
    percentages.sort(key=lambda row: row[2][0])
    if len(percentages) != 3 or any(row[1] < .95 for row in percentages):
        raise ValueError("turn information three attribute percentages are unresolved")
    percent_centers = [row[2][1] + row[2][3] / 2 for row in percentages]
    if max(percent_centers) - min(percent_centers) > 8:
        raise ValueError("turn information percentages are not in one header row")
    for row, (left, right) in zip(percentages, ((90, 280), (290, 475), (490, 675))):
        if not left <= row[2][0] < row[2][0] + row[2][2] <= right:
            raise ValueError("turn information percentage column is ambiguous")
    values = tuple(int(row[0][:-1]) for row in percentages)
    if any(not 0 < value <= 100000 for value in values):
        raise ValueError("turn information percentage is outside the supported range")
    return TurnInformation(current_turn, tuple(attributes), values)


def turn_information_consensus(observations, *, current_score_multiplier: float) -> TurnInformation:
    if not observations or any(not isinstance(row, TurnInformation) for row in observations):
        raise ValueError("turn information requires a complete observation")
    if any(row != observations[0] for row in observations[1:]):
        raise ValueError("turn information changed across frames")
    if (type(current_score_multiplier) not in (int, float) or not math.isfinite(current_score_multiplier)
            or not math.isclose(observations[0].multipliers[0], current_score_multiplier, abs_tol=1e-8, rel_tol=0)):
        raise ValueError("turn information contradicts the current Live multiplier")
    return observations[0]


@dataclass(frozen=True)
class RoundTwoTurnRow:
    offset: int
    attribute: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class RoundTwoTurnPage:
    current_turn: int
    score_percentages: tuple[int, int, int]
    rows: tuple[RoundTwoTurnRow, ...]
    thumb: tuple[int, int]
    at_top: bool
    at_bottom: bool


@dataclass(frozen=True)
class VerifiedRoundTwoTurnPage:
    page: RoundTwoTurnPage
    frame_ids: tuple[str, ...]


def _round_two_past_row_boxes(image):
    """Past turns have dark outlines around an opaque, lighter gray fill."""
    import cv2

    body = image[303:1147, 40:680].astype(np.int16)
    gray = ((body.min(axis=2) >= 90) & (body.max(axis=2) < 190)
            & (np.ptp(body, axis=2) < 8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(gray.astype(np.uint8), 8)
    boxes = []
    for label in range(1, count):
        x, y, width, height, _ = map(int, stats[label])
        if not (0 <= x <= 2 and x+width >= 638 and 72 <= height <= 80
                and y >= 5 and y+height+5 <= len(body)):
            continue
        # Each side independently supplies both complete outlines, the lighter
        # inner fill and white space outside. A detached stroke is not a row.
        valid = True
        for columns in (slice(20, 270), slice(372, 622)):
            owned = labels[y:y+height, columns] == label
            pixels = body[y:y+height, columns]
            inner_top = pixels[5:10].min(axis=(0, 2))
            inner_bottom = pixels[-10:-5].min(axis=(0, 2))
            top = pixels[:5].min(axis=(0, 2)) < inner_top
            bottom = pixels[-5:].min(axis=(0, 2)) < inner_bottom
            outside_top = (body[y-5:y, columns].min(axis=2) >= 245).mean(axis=1)
            outside_bottom = (body[y+height:y+height+5, columns].min(axis=2) >= 245).mean(axis=1)
            if (any(owned[index].mean() < .97 for index in (0, height-1, 5, height-6))
                    or min(top.mean(), bottom.mean()) < .97
                    or min(outside_top.max(), outside_bottom.max()) < .97):
                valid = False
                break
        if valid:
            boxes.append((40, y+303, 640, height))
    return tuple(boxes)


def round_two_turn_geometry(image):
    """Read the actual 988/989 opaque list borders and independent scrollbar."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('round-two turn panel resolution is uncalibrated')
    strip = image[303:1147, 689:692].astype(np.int16)
    solid = ((strip.min(axis=(1, 2)) >= 90) & (strip.max(axis=(1, 2)) <= 180)
             & (np.ptp(strip, axis=(1, 2)) <= 32)
             & (np.ptp(strip, axis=1).max(axis=1) <= 2))
    edges = np.diff(np.r_[False, solid, False].astype(np.int8))
    spans = [(int(a+303), int(b+303)) for a, b in
             zip(np.where(edges == 1)[0], np.where(edges == -1)[0]) if b-a >= 30]
    if len(spans) != 1:
        raise ValueError('round-two turn scrollbar is not unique')
    thumb = spans[0]
    # Horizontal borders extend independently across both text columns. Only
    # two complete boundaries can establish a readable row; edge fragments
    # never become rows just because their label happens to be recognized.
    body = image[303:1147, 60:662].astype(np.int16)
    gray = ((body.min(axis=2) >= 190) & (body.max(axis=2) < 245)
            & (np.ptp(body, axis=2) < 8))
    # The fixed close button overlaps the middle of the last row's lower
    # border at the bottom. Both independent side spans remain fully visible.
    lines = (gray[:, :250].mean(axis=1) >= .97) & (gray[:, -250:].mean(axis=1) >= .97)
    edges = np.diff(np.r_[False, lines, False].astype(np.int8))
    borders = [(int(a+303), int(b+303)) for a, b in
               zip(np.where(edges == 1)[0], np.where(edges == -1)[0]) if 1 <= b-a <= 5]
    boxes = tuple((40, a[0], 640, b[1]-a[0]) for a, b in zip(borders, borders[1:])
                  if 72 <= b[1]-a[0] <= 80)
    past = _round_two_past_row_boxes(image)
    if past:
        # Antialiasing at a dark row's edge can also resemble a light border.
        # Its independently proven full row owns those overlapping fragments.
        boxes = tuple(sorted((*past, *(box for box in boxes if not any(
            box[1] < row[1]+row[3] and row[1] < box[1]+box[3] for row in past))), key=lambda box: box[1]))
    if not 8 <= len(boxes) <= 10 or any(not 86 <= b[1]-a[1] <= 90
                                      for a, b in zip(boxes, boxes[1:])):
        raise ValueError('round-two complete row borders are unresolved')
    return boxes, thumb, thumb[0] <= 307, thumb[1] >= 1142


def _round_two_header(rows, current_turn):
    normalized = []
    for row in rows:
        text, score, box = ((row.get('text'), row.get('score'), row.get('box'))
                            if isinstance(row, Mapping) else (row.text, row.score, row.box))
        if (not isinstance(text, str) or type(score) not in (int, float)
                or not math.isfinite(score) or not 0 <= score <= 1
                or not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(value) is not int for value in box)
                or min(box[:2]) < 0 or min(box[2:]) <= 0
                or box[0]+box[2] > 720 or box[1]+box[3] > 1280):
            raise ValueError('round-two turn OCR evidence is malformed')
        normalized.append((unicodedata.normalize('NFKC', text).replace(' ', ''), score, tuple(box)))
    for pattern, roi in ((f'現在{current_turn}ターン目', (40, 125, 280, 42)),
                         ('1位でラウンド2突破', (40, 165, 290, 40))):
        x, y, w, h = roi
        hits = [row for row in normalized if row[0] == pattern and row[1] >= .95
                and x <= row[2][0] and y <= row[2][1]
                and row[2][0]+row[2][2] <= x+w and row[2][1]+row[2][3] <= y+h]
        if len(hits) != 1:
            raise ValueError('round-two turn heading or objective is unresolved')
    return normalized


def read_round_two_turn_page(context, image, *, current_turn, turns_remaining, round_context):
    from .inventory import _inventory_ocr

    if resolve_hif_round_context(round_context) != (2, 12):
        raise ValueError('round-two turn page needs explicit notebook round context')
    if current_turn_from_remaining(turns_remaining, round_context=round_context) != current_turn:
        raise ValueError('round-two turn page contradicts its current turn')
    boxes, thumb, at_top, at_bottom = round_two_turn_geometry(image)
    rows = _round_two_header(_inventory_ocr(context, image, 'LiveStatusRoundTwoTurnInformation'), current_turn)
    percentages = []
    for left, right in ((90, 280), (290, 475), (490, 675)):
        hits = [row for row in rows if re.fullmatch(r'\d+%', row[0])
                and left <= row[2][0] < row[2][0]+row[2][2] <= right
                and 210 <= row[2][1] < row[2][1]+row[2][3] <= 280]
        if len(hits) != 1 or hits[0][1] < .95:
            raise ValueError('round-two percentage column is unresolved')
        percentages.append(hits[0])
    centers = [row[2][1]+row[2][3]/2 for row in percentages]
    values = tuple(int(row[0][:-1]) for row in percentages)
    if max(centers)-min(centers) > 8 or any(not 0 < value <= 100000 for value in values):
        raise ValueError('round-two percentage header is inconsistent')
    output = []
    for box in boxes:
        x, y, w, h = box
        inside = [row for row in rows if x <= row[2][0] and y+3 <= row[2][1]
                  and row[2][0]+row[2][2] <= x+w and row[2][1]+row[2][3] <= y+h-3]
        labels = [row for row in inside if row[0] == '現在のターン'
                  or re.fullmatch(r'[1-9]\d?ターン[前後]', row[0])]
        # The existing OCR sometimes separates a large numeral from its
        # smaller suffix. Join only two independently observed adjacent parts.
        if not labels:
            suffixes = [row for row in inside if row[0] in ('ターン前', 'ターン後')]
            digits = [row for row in inside if re.fullmatch(r'[1-9]\d?', row[0]) and row[2][0] < 190]
            if len(suffixes) == len(digits) == 1:
                digit, suffix = digits[0], suffixes[0]
                if (-15 <= suffix[2][0]-digit[2][0]-digit[2][2] <= 15
                        and abs(digit[2][1]+digit[2][3]/2-suffix[2][1]-suffix[2][3]/2) <= 20):
                    labels = [(digit[0]+suffix[0], min(digit[1], suffix[1]),
                               (digit[2][0], min(digit[2][1], suffix[2][1]),
                                suffix[2][0]+suffix[2][2]-digit[2][0],
                                max(digit[2][1]+digit[2][3], suffix[2][1]+suffix[2][3])
                                - min(digit[2][1], suffix[2][1])))]
        attributes = [row for row in inside if row[0] in ATTRIBUTES and row[2][0] >= 230]
        # Whole-screen detection can give a weak score to a complete label
        # containing a large digit and smaller suffix. Independently reread
        # the full label column in this same bounded row, without changing
        # its literal text or accepting a missing/partial label.
        if (len(labels) == 1 and labels[0][1] < .95
                and labels[0][2][0]+labels[0][2][2] <= 220):
            reread = _inventory_ocr(context, image[y+3:y+h-3, x:220].copy(),
                                    'LiveStatusRoundTwoTurnLabel', only_rec=True)
            if (len(reread) == 1 and isinstance(reread[0].text, str)
                    and type(reread[0].score) in (int, float) and math.isfinite(reread[0].score)
                    and .95 <= reread[0].score <= 1
                    and unicodedata.normalize('NFKC', reread[0].text).replace(' ', '') == labels[0][0]):
                labels = [(labels[0][0], reread[0].score, labels[0][2])]
        if (len(labels) != 1 or len(attributes) != 1 or min(labels[0][1], attributes[0][1]) < .95
                or labels[0][2][0]+labels[0][2][2] > 220
                or attributes[0][2][0]+attributes[0][2][2] > 390
                or abs(labels[0][2][1]+labels[0][2][3]/2
                       - attributes[0][2][1]-attributes[0][2][3]/2) > 20):
            raise ValueError('round-two complete row label or attribute is unresolved')
        text = labels[0][0]
        match = re.fullmatch(r'([1-9]\d?)ターン([前後])', text)
        offset = int(match[1])*(-1 if match[2] == '前' else 1) if match else 0
        output.append(RoundTwoTurnRow(offset, ATTRIBUTES[attributes[0][0]], box))
    offsets = [row.offset for row in output]
    if (offsets != list(range(offsets[0], offsets[0]+len(offsets)))
            or offsets[0] < 1-current_turn or offsets[-1] >= turns_remaining
            or at_top and offsets[0] != 1-current_turn
            or at_bottom and offsets[-1] != turns_remaining-1):
        raise ValueError('round-two row sequence or scroll boundary is incomplete')
    return RoundTwoTurnPage(current_turn, values, tuple(output), thumb, at_top, at_bottom)


def round_two_turn_page_consensus(observations, frame_ids):
    observations, frame_ids = tuple(observations), tuple(frame_ids)
    if (not observations or not all(isinstance(row, RoundTwoTurnPage) for row in observations)
            or len(frame_ids) != len(observations) or len(set(frame_ids)) != len(frame_ids)
            or any(not isinstance(key, str) or not key for key in frame_ids)):
        raise ValueError('round-two turn page lacks its observed frame')
    if any(row != observations[0] for row in observations[1:]):
        raise ValueError('round-two turn page has not stopped scrolling')
    return VerifiedRoundTwoTurnPage(observations[0], frame_ids)


def join_round_two_turn_pages(pages, *, current_score_multiplier, require_bottom=True):
    pages = tuple(pages)
    if not pages or not all(isinstance(page, VerifiedRoundTwoTurnPage) for page in pages):
        raise ValueError('round-two turn pages are unverified')
    first = pages[0].page
    if not first.at_top or require_bottom and not pages[-1].page.at_bottom:
        raise ValueError('round-two turn table needs both scroll boundaries')
    all_ids = [key for page in pages for key in page.frame_ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('round-two turn pages reuse capture frames')
    known = {row.offset: row.attribute for row in first.rows}
    for old_verified, new_verified in zip(pages, pages[1:]):
        old, new = old_verified.page, new_verified.page
        if (new.current_turn != first.current_turn or new.score_percentages != first.score_percentages
                or old.at_bottom or new.at_top or new.thumb[0] <= old.thumb[0]
                # The rounded track clips up to two pixels at either end;
                # actual top/bottom captures retain 662/658 solid pixels.
                or abs(new.thumb[1]-new.thumb[0]-old.thumb[1]+old.thumb[0]) > 4):
            raise ValueError('round-two turn page context or scrollbar changed')
        shared = sorted(set(row.offset for row in old.rows) & set(row.offset for row in new.rows))
        old_rows, new_rows = ({row.offset: row for row in page.rows} for page in (old, new))
        if len(shared) < 2 or new.rows[-1].offset <= old.rows[-1].offset:
            raise ValueError('round-two turn pages need advancing overlapping rows')
        shifts = [old_rows[key].box[1]-new_rows[key].box[1] for key in shared]
        if min(shifts) <= 0 or max(shifts)-min(shifts) > 2:
            raise ValueError('round-two turn overlap geometry changed')
        for row in new.rows:
            if row.offset in known and known[row.offset] != row.attribute:
                raise ValueError('round-two turn overlapping attribute changed')
            known[row.offset] = row.attribute
    expected = list(range(1-first.current_turn, 13-first.current_turn))
    if sorted(known) != (expected if require_bottom else expected[:len(known)]):
        raise ValueError('round-two full turn table has missing rows')
    information = TurnInformation(first.current_turn, tuple(known[key] for key in sorted(known) if key >= 0),
                                  first.score_percentages)
    if require_bottom and not information.turn_attributes:
        raise ValueError('round-two turn page does not include the current turn')
    if (type(current_score_multiplier) not in (int, float) or not math.isfinite(current_score_multiplier)
            or information.turn_attributes and not math.isclose(
                information.multipliers[0], current_score_multiplier, abs_tol=1e-8, rel_tol=0)):
        raise ValueError('round-two turn table contradicts the current Live multiplier')
    return information


def _observe_round_two_turn_pages(context, *, capture_frames, frame_identifier, current_turn,
                                  turns_remaining, current_score_multiplier, round_context,
                                  maximum_age_ms, event, started):
    from utils.maa_input import verify_maa_input_endpoints

    from .inventory import _inventory_ocr

    controller = context.tasker.controller
    limit_batch_time = getattr(context, 'gkh_hif_upstream_card_ui', False) is not True

    def stable_page(phase):
        for attempt in range(5):
            if limit_batch_time and time.perf_counter()-started > 40:
                raise ValueError('round-two turn reading time budget exceeded')
            frames = tuple(capture_frames())
            ids = tuple(frame_identifier(frame) for frame in frames)
            try:
                if not frames:
                    raise ValueError('round-two turn page lacks its observed frame')
                geometries = [round_two_turn_geometry(frame) for frame in frames]
                if any(geometry != geometries[0] for geometry in geometries[1:]):
                    raise ValueError('round-two turn page has not stopped scrolling')
                pages = [read_round_two_turn_page(context, frame, current_turn=current_turn,
                                                  turns_remaining=turns_remaining,
                                                  round_context=round_context) for frame in frames]
                return round_two_turn_page_consensus(pages, ids)
            except ValueError as error:
                event.setdefault('reading_rejections', []).append(dict(
                    phase=phase, attempt=attempt, reason=str(error), frame_ids=ids))
                failure_sink = getattr(context, 'gkh_hif_status_failure_triplet_sink', None)
                if callable(failure_sink):
                    failure_sink(frames)
                if attempt == 4:
                    raise
                time.sleep(.5)

    def scroll(page, toward_top):
        image = controller.post_screencap().wait().get()
        captured_at = frame_captured_at(controller, image)
        geometry = round_two_turn_geometry(image)
        if geometry != (tuple(row.box for row in page.rows), page.thumb, page.at_top, page.at_bottom):
            raise ValueError('round-two turn list moved before navigation')
        _round_two_header(_inventory_ocr(context, image, 'LiveStatusRoundTwoTurnNavigation',
                                       roi=[40, 125, 290, 80]), current_turn)
        endpoints = verify_maa_input_endpoints(controller.info,
            expected_client_size=getattr(context, 'gkh_hif_client_size', (720, 1280)))
        if (time.perf_counter()-captured_at)*1000 > maximum_age_ms:
            raise ValueError('round-two turn navigation input frame expired')
        if limit_batch_time and time.perf_counter()-started > 40:
            raise ValueError('round-two turn navigation budget exceeded')
        swipe = (560, 700, 560, 940 if toward_top else 460, 500)
        event['input_click_count'] += 1
        event['inputs'].append(dict(phase='scroll', toward_top=toward_top, swipe=swipe,
                                    endpoints=endpoints, frame_id=frame_identifier(image)))
        if not controller.post_swipe(*swipe).wait().succeeded:
            raise ValueError('Maa round-two turn navigation failed')
        time.sleep(.5)

    first = stable_page('initial')
    while not first.page.at_top:
        event.setdefault('seeking_pages', []).append(asdict(first))
        scroll(first.page, True)
        new = stable_page('seek_top')
        if (new.page.thumb[0] >= first.page.thumb[0]
                or abs(new.page.thumb[1]-new.page.thumb[0]-first.page.thumb[1]+first.page.thumb[0]) > 4):
            raise ValueError('round-two turn list did not advance toward top')
        first = new
    pages = [first]
    event['pages'] = [asdict(first)]
    while not pages[-1].page.at_bottom:
        scroll(pages[-1].page, False)
        new = stable_page('read')
        join_round_two_turn_pages([*pages, new], current_score_multiplier=current_score_multiplier,
                                 require_bottom=False)
        pages.append(new)
        event['pages'].append(asdict(new))
    information = join_round_two_turn_pages(pages, current_score_multiplier=current_score_multiplier)
    event['panel_frame_ids'] = [key for page in pages for key in page.frame_ids]
    return information


def observe_turn_information_transaction(context, *, capture_frames, checkpoint, frame_identifier,
                                         run_id, revision, maximum_age_ms=1000, round_context=None):
    """Open the turn table once and restore the same Live; no gameplay input."""
    round_number, base_turn_count = resolve_hif_round_context(round_context, run_id=run_id, revision=revision)
    from utils.maa_input import verify_maa_input_endpoints

    from .inventory import (
        _inventory_ocr,
        _unique_inventory_text,
        capture_initial_live_checkpoint,
        fresh_identity_reobserve_allowed,
        unresolved_live_resource_evidence,
    )
    from .status_panel import StatusInspectionError
    from .hif_entry_round import read_entry_turn

    event = dict(phase='turn_information_observation', accepted=False, input_click_count=0,
                 inputs=[], strategy_state_bound=False, run_id=run_id, revision=revision)
    if round_context is not None:
        event.update(round_number=round_number, base_turn_count=base_turn_count)
    controller = context.tasker.controller
    started = time.perf_counter()
    limit_batch_time = getattr(context, 'gkh_hif_upstream_card_ui', False) is not True

    def fresh():
        image = controller.post_screencap().wait().get()
        return image, frame_captured_at(controller, image)

    def click(box, captured_at, phase):
        endpoints = verify_maa_input_endpoints(controller.info, expected_client_size=getattr(context, "gkh_hif_client_size", (720, 1280)))
        if (time.perf_counter()-captured_at)*1000 > maximum_age_ms:
            raise ValueError('turn information input frame expired')
        if limit_batch_time and phase != 'close' and time.perf_counter()-started > (40 if round_number == 2 else 20):
            raise ValueError('turn information navigation budget exceeded')
        event['input_click_count'] += 1
        event['inputs'].append(dict(phase=phase, box=box, endpoints=endpoints))
        x, y, width, height = box
        if not controller.post_click(x+width//2, y+height//2).wait().succeeded:
            raise ValueError('Maa turn information click failed')

    try:
        if (not isinstance(run_id, str) or not run_id.strip() or type(revision) is not int or revision < 0
                or type(maximum_age_ms) not in (int, float) or not 0 < maximum_age_ms <= 1000):
            raise ValueError('turn information context or freshness gate is invalid')
        if controller.set_screenshot_target_long_side(1280) is not True:
            raise ValueError('turn information capture profile was not applied')
        _, before = capture_initial_live_checkpoint(capture_frames, checkpoint, event)
        remaining = before['fields'][0]
        current_turn = current_turn_from_remaining(remaining, round_context=round_context)
        event['before_frame_ids'] = before['frame_ids']
        for entry_attempt in range(3):
            image, captured_at = fresh()
            try:
                current = checkpoint((image,), single_frame=True)
            except ValueError as error:
                resource_evidence = unresolved_live_resource_evidence(error)
                event.setdefault('preinput_checkpoint_rejections', []).append(dict(
                    reason=str(error), frame_id=frame_identifier(image),
                    failure_events=list(getattr(error, 'events', ()))))
                if (entry_attempt == 2 or (not resource_evidence
                        and not fresh_identity_reobserve_allowed(error, before)
                        and str(error) != 'inventory fresh recheck is not an unselected hand')):
                    raise
                _, confirmed = capture_initial_live_checkpoint(capture_frames, checkpoint, event)
                if (confirmed['families'], confirmed['fields']) != (before['families'], before['fields']):
                    raise ValueError('Live changed before opening turn information') from error
                observation_key = ('preinput_resource_reobservations' if resource_evidence
                                   else 'preinput_hand_reobservations')
                event.setdefault(observation_key, []).append(confirmed['frame_ids'])
                continue
            if (current['families'], current['fields']) != (before['families'], before['fields']):
                raise ValueError('Live changed before opening turn information')
            # This fresh screenshot already passed the unselected-hand and
            # public-state check. Use miyu's ring reader for the same control.
            target_box = (40, 68, 43, 42)
            if getattr(context, 'gkh_hif_upstream_card_ui', False) is True:
                observed = read_entry_turn(context, image)
                if observed == remaining:
                    break
                reason = f'turn ring reads {observed!r}, expected {remaining}'
                event.setdefault('entry_rejections', []).append(dict(
                    reason=reason, frame_id=frame_identifier(image)))
                if entry_attempt == 2:
                    raise ValueError(reason)
                continue
            rows = _inventory_ocr(context, image, 'LiveStatusTurnEntry',
                                  roi=[0, 0, 140, 50 if round_number == 2 else 130])
            number_rows = _inventory_ocr(context, image, 'LiveStatusTurnEntryNumber',
                                        roi=list(target_box), only_rec=True)
            try:
                _unique_inventory_text(rows, '残りターン', (0, 0, 140, 50), threshold=.95)
                target_box = tuple(_unique_inventory_text(number_rows, str(remaining), target_box, threshold=.95).box)
                break
            except ValueError as error:
                event.setdefault('entry_rejections', []).append(dict(reason=str(error),
                    frame_id=frame_identifier(image), rows=[dict(text=row.text,score=row.score,box=tuple(row.box))
                                                          for row in (*rows,*number_rows)]))
                if entry_attempt == 2:
                    raise
        click(target_box, captured_at, 'open')
        failure = None
        observation = None
        try:
            if round_number == 2:
                observation = _observe_round_two_turn_pages(context, capture_frames=capture_frames,
                    frame_identifier=frame_identifier, current_turn=current_turn, turns_remaining=remaining,
                    current_score_multiplier=before['fields'][4], round_context=round_context,
                    maximum_age_ms=maximum_age_ms, event=event, started=started)
            else:
                for attempt in range(6):
                    if limit_batch_time and time.perf_counter()-started > 20:
                        raise ValueError('turn information reading time budget exceeded')
                    frames = capture_frames()
                    try:
                        observation = turn_information_consensus([
                            parse_turn_information(_inventory_ocr(context, frame, 'LiveStatusTurnInformation'),
                                                   current_turn=current_turn, turns_remaining=remaining,
                                                   round_context=round_context)
                            for frame in frames], current_score_multiplier=before['fields'][4])
                        event['panel_frame_ids'] = [frame_identifier(frame) for frame in frames]
                        break
                    except ValueError as error:
                        event.setdefault('reading_rejections', []).append(str(error))
                        if attempt == 5:
                            raise
                        time.sleep(.08)
        except (ValueError, RuntimeError) as error:
            failure = str(error)
            event['read_failure'] = failure
        # The current-turn heading and round objective independently authorize
        # closing even when a future row is unreadable. No blind cleanup click.
        image, captured_at = fresh()
        rows = _inventory_ocr(context, image, 'LiveStatusTurnClose')
        headings = [row for row in rows if row.text.replace(' ', '') == f'現在{current_turn}ターン目' and row.score >= .95]
        goal_pattern = r'1位でラウンド2突破' if round_number == 2 else r'\d+位以上でラウンド1突破'
        goals = [row for row in rows if re.fullmatch(goal_pattern, row.text.replace(' ', '')) and row.score >= .95]
        if len(headings) != 1 or len(goals) != 1:
            raise ValueError('turn information close page is unresolved')
        node = 'LiveStatusTurnCloseButton'
        hit = context.run_recognition(node, image, pipeline_override={node: {
            'recognition': 'TemplateMatch', 'template': ['close_icon.png'],
            'roi': [300, 1090, 120, 130], 'threshold': .80}})
        targets = list(hit.filtered_results or ()) if hit and hit.hit else []
        if len(targets) != 1:
            raise ValueError('turn information close control is unresolved')
        click(tuple(targets[0].box), captured_at, 'close')
        if callable(getattr(checkpoint, '__dict__', {}).get('capture_complete')):
            _, after = capture_initial_live_checkpoint(capture_frames, checkpoint, event)
        else:
            for attempt in range(6):
                try:
                    after = checkpoint(capture_frames())
                    break
                except ValueError as error:
                    event.setdefault('return_rejections', []).append(str(error))
                    if attempt == 5:
                        raise
                    time.sleep(.08)
        if (after['families'], after['fields']) != (before['families'], before['fields']):
            raise ValueError('Live changed after closing turn information')
        event.update(returned_unchanged=True, after_frame_ids=after['frame_ids'])
        if failure or observation is None:
            raise ValueError(failure or 'turn information observation is missing')
        event.update(accepted=True, current_turn=current_turn, turn_attributes=observation.turn_attributes,
                     score_percentages=observation.score_percentages, score_multipliers=observation.multipliers,
                     future_score_multipliers=observation.multipliers[1:])
    except (ValueError, RuntimeError) as error:
        event.update(reason=str(error), elapsed_ms=(time.perf_counter()-started)*1000)
        if getattr(error, 'events', ()):
            event['checkpoint_failure_events'] = list(error.events)
        raise StatusInspectionError(str(error), [event]) from error
    event['elapsed_ms'] = (time.perf_counter()-started)*1000
    return [event]
