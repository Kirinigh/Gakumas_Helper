"""Current ordinary HIF pages, using the existing route host and catalog.

A first-day lesson selection can reach either its preview or the classroom.
Both destinations preserve resources until a classroom option is confirmed.
"""
from __future__ import annotations

import re
import time
from copy import deepcopy
from pathlib import Path
from functools import lru_cache
from dataclasses import replace

import cv2
from utils.runtime_paths import resolve_runtime_path

from .hif import active_hif_role, HifConfig, decide_hif, training_main_attribute
from .host import HifRoutePage
from .engine import Observation, source_frames_match
from .ledger import LedgerEffect

LESSON_OPTION_CAPTION = r'.*[\u3040-\u30ff\u4e00-\u9fff].*'


def _daily_page_id(page, *, scene=None):
    scene = page['scene'] if scene is None else scene
    offers = page.get('offer_ids', [])
    if scene == 'card_choice' and len(offers) == 3:
        # Learning an unused neighbour is not a new physical page.
        target = page.get('selected_offer_index', 2)
        offers = tuple(card_id if index == target else None for index, card_id in enumerate(offers))
    training = ';'.join(f'{option["attribute"]}:{option["special"]}:{option["cost"]}' for option in page.get('options', []))
    return (f'daily:{page["day"]}:{scene}:Vi:'
        + ','.join(key if key is not None else '?' for key in offers)
        + (':' + training if training else ''))


@lru_cache(maxsize=1)
def _attribute_glyphs():
    root = resolve_runtime_path(Path('assets/resource/base/image/produce'), Path('resource/base/image/produce'))
    result = {}
    for attribute, scale in (('Vo', .55), ('Da', .55), ('Vi', .57)):
        image = cv2.imread(str(root / (attribute + '.png')))
        if image is None:
            raise ValueError('existing lesson attribute reference is unavailable')
        mask = (image.min(2) > 230).astype('uint8') * 255
        # Rounded tile corners are background, not part of the white glyph.
        mask[:8] = mask[-8:] = 0
        mask[:, :8] = mask[:, -8:] = 0
        x, y, width, height = cv2.boundingRect(mask)
        result[attribute] = cv2.resize(mask[y:y + height, x:x + width], None,
            fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return result


def lesson_attribute(image, caption):
    x, y, _, _ = caption
    left, top = x + 20, y - 85
    if left < 0 or top < 0 or left + 80 > 720 or top + 90 > 1280:
        raise ValueError('lesson attribute glyph is outside its caption tile')
    white = (image[top:top + 90, left:left + 80].min(2) > 230).astype('uint8') * 255
    matches = []
    scores = {}
    for attribute, reference in _attribute_glyphs().items():
        response = cv2.matchTemplate(white, reference, cv2.TM_CCOEFF_NORMED)
        _, score, _, point = cv2.minMaxLoc(response)
        scores[attribute] = float(score)
        if score >= .8:
            count, _ = cv2.connectedComponents((response >= .8).astype('uint8'))
            if count != 2:
                raise ValueError('lesson attribute glyph has multiple accepted targets')
            matches.append((attribute, point))
    if len(matches) != 1:
        return None, scores
    attribute, _ = matches[0]
    if scores[attribute] - max(score for key, score in scores.items() if key != attribute) < .2:
        raise ValueError('lesson attribute glyph identity is ambiguous')
    return attribute, scores


def read_lesson_schedule(image, rows, *, confirmation_pending=False, calendar=None):
    from .calendar import read_calendar, read_route_resources
    from .production import _row
    day, family = calendar or read_calendar(rows)
    if family != 'lesson':
        raise ValueError('lesson schedule requires its actual first or fourth day')
    captions = [_row(rows, r'授業', region) for region in (
        (140, 1035, 100, 85), (300, 1035, 100, 85), (460, 1035, 100, 85))]
    # Upstream LessonAuto confirms the countdown and all three lesson
    # columns, whose order is Vo/Da/Vi. Re-reading tiny attribute glyphs is
    # not a prerequisite for opening the configured column's preview.
    attributes = ('Vo', 'Da', 'Vi')
    selected = [row for row in rows if row['score'] >= .95 and row['text'] == 'SELECT'
        and 100 <= row['box'][0] and 1065 <= row['box'][1]
        and row['box'][0] + row['box'][2] <= 610 and row['box'][1] + row['box'][3] <= 1150]
    common = dict(read_route_resources(rows), day=day,
        attribute_evidence=dict(source='confirmed_lesson_columns', order=attributes),
        caption_boxes=[row['box'] for row in captions])
    if not selected and confirmation_pending:
        # The old buttons can persist after the actual confirmation was issued.
        # Wait for its destination without treating the animation as a new turn.
        raise ValueError('lesson preview awaits its current SELECT or destination')
    if selected:
        if len(selected) != 1:
            raise ValueError('lesson preview does not uniquely select the supported visual lesson')
        index = attributes.index('Vi')
        caption, select = captions[index]['box'], selected[0]['box']
        if abs(caption[0] + caption[2] / 2 - select[0] - select[2] / 2) > 15:
            raise ValueError('lesson SELECT does not belong to the selected attribute')
        # SELECT is an animated selection anchor below the card. Its glyph box
        # can move outside the actual button and cancel selection when clicked.
        return dict(common, scene='lesson_selected', targets={'confirm_lesson': tuple(caption)}, selected_attribute='Vi')
    return dict(common, scene='lesson', targets={attribute: tuple(row['box'])
        for attribute, row in zip(attributes, captions, strict=True)})


def read_day_one_lesson(image, rows):
    page = read_lesson_schedule(image, rows)
    if page['day'] != 1:
        raise ValueError('first-day reader cannot identify another actual calendar day')
    return page


def lesson_selection_effect(state):
    return LedgerEffect(f'{state.run_id}:{state.revision}:select-lesson:Vi', 'lesson:Vi')


def _daily_stamina_segments(context, image, unresolved):
    """Read two actual numeral fields and the complete intervening slash."""
    from live_decision.hif_recovery import _field
    if not re.fullmatch(r'\d{3,5}', unresolved.get('text', '')):
        raise ValueError('ordinary stamina is not a segmented fraction candidate')
    # The ordinary HUD uses grey digits; the notebook's analogous existing
    # separator proof uses white. Both require the whole single diagonal.
    x, y, width, height = 373, 65, 20, 38
    pixels = image[y:y+height, x:x+width].astype('int16')
    mask = ((pixels.min(2) > 145) & (pixels.max(2)-pixels.min(2) < 25)).astype('uint8')
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count != 2:
        raise ValueError('ordinary stamina fraction lacks one complete separator')
    left, top, wide, high, area = map(int, stats[1])
    if not (2 <= left and left+wide <= width-2 and 10 <= wide <= 16
            and 2 <= top and top+high <= height-2 and 18 <= high <= 26 and 40 <= area <= 110):
        raise ValueError('ordinary stamina fraction separator is clipped')
    centers = []
    for row in mask[top:top+high]:
        points = row.nonzero()[0]
        if not 2 <= len(points) <= 4 or int(points[-1])-int(points[0])+1 != len(points):
            raise ValueError('ordinary stamina fraction separator is discontinuous')
        centers.append((int(points[0])+int(points[-1]))/2)
    if (not 9 <= centers[0]-centers[-1] <= 13
            or any(not 0 <= before-after <= 1 for before, after in zip(centers, centers[1:]))):
        raise ValueError('ordinary stamina fraction separator is not a slash')
    current = _field(context, image, 'DailyStaminaCurrent', (329, 65, 44, 36), r'\d{1,2}')
    maximum = _field(context, image, 'DailyStaminaMaximum', (393, 68, 36, 32), r'\d{1,2}')
    if current['text']+'1'+maximum['text'] != unresolved['text']:
        raise ValueError('ordinary stamina fraction segments contradict the original crop')
    return dict(text=current['text']+'/'+maximum['text'], score=min(current['score'], maximum['score']),
        box=[329, 65, 100, 36], source='independent_numerals_and_complete_diagonal_separator',
        segments=[current, maximum], original_crop=unresolved,
        separator=dict(roi=[x,y,width,height], box=[x+left,y+top,wide,high], row_centers=centers))


def complete_daily_resources(context, image, rows, *, submitted_stamina=None):
    from live_decision.hif_recovery import _field
    accepted, evidence = list(rows), []
    for name, pattern, region, crop in (
            ('DailyStamina', r'\d{1,3}/\d{1,3}', (285, 35, 165, 85), (300, 40, 145, 75)),
            ('DailyCurrency', r'\d{1,6}', (350, 80, 110, 75), (350, 80, 110, 75))):
        x, y, width, height = region
        located = [row for row in rows if re.fullmatch(pattern, row['text'])
            and x <= row['box'][0] and y <= row['box'][1]
            and row['box'][0] + row['box'][2] <= x + width and row['box'][1] + row['box'][3] <= y + height]
        strong = [row for row in located if row['score'] >= .95]
        if len(strong) > 1:
            raise ValueError('ordinary resource field has contradictory high-confidence observations')
        if strong:
            continue
        if (name == 'DailyStamina' and submitted_stamina is not None and len(located) == 1
                and tuple(map(int, located[0]['text'].split('/'))) == submitted_stamina):
            # The course result HUD moves during the animation. A complete
            # current fraction agreeing with this submission needs no raster slash.
            proof = dict(located[0], source='observed_submitted_course_fraction')
            accepted.append(proof)
            evidence.append(proof)
            continue
        if len(located) == 1:
            left, top, wide, high = located[0]['box']
            right, bottom = min(x + width, left + wide + 4), min(y + height, top + high + 4)
            left, top = max(x, left - 4), max(y, top - 4)
            crop = (left, top, right - left, bottom - top)
        try:
            proof = _field(context, image, name, crop, pattern)
        except ValueError as error:
            unresolved = getattr(error,'evidence',{})
            if (name != 'DailyStamina' or unresolved.get('field') != name
                    or located and not re.fullmatch(r'\d{3,5}',unresolved.get('text',''))):
                raise
            # Exclude the heart before reading the unequal-size numeral pair.
            # The complete crop can still misread the actual slash as "1";
            # only independent numerals plus its full stroke may resolve it.
            if located:
                proof = _daily_stamina_segments(context,image,unresolved)
            else:
                try:
                    proof = _field(context, image, 'DailyStaminaTight', (325, 55, 115, 50), pattern)
                except ValueError as tight_error:
                    if getattr(tight_error, 'evidence', {}).get('field') != 'DailyStaminaTight':
                        raise
                    proof = _daily_stamina_segments(context, image, tight_error.evidence)
                    proof['original_wide_crop'] = error.evidence
        # Resource consumers use integers as well: a complete 029/42 and
        # 29/42 are the same observation, without repairing OCR characters.
        weak_values = {tuple(map(int, row['text'].split('/'))) for row in located}
        if weak_values and weak_values != {tuple(map(int, proof['text'].split('/')))}:
            raise ValueError('ordinary resource crop contradicts the full-frame observation')
        accepted.append(proof)
        evidence.append(dict(proof, original_observations=located,
            source=proof.get('source', 'independent_complete_resource_crop')))
    return accepted, evidence


def _lesson_option_selected(image, top):
    button = image[top:top + 65, 60:660]
    return (((button[:, :, 2] >= 230) & (button[:, :, 1] >= 140)
        & (button[:, :, 1] <= 220) & (button[:, :, 0] <= 100)).mean() >= .65)


def _first_lesson_captions(rows):
    # Right-hand cost badges do not name a choice.
    return [row for row in rows if re.fullmatch(LESSON_OPTION_CAPTION, row['text'])
        and 50 <= row['box'][0] < 300 and 650 <= row['box'][1]
        and row['box'][0] + row['box'][2] <= 670 and row['box'][1] + row['box'][3] <= 740]


def _first_lesson_target(image, rows):
    captions = [row for row in _first_lesson_captions(rows) if row['score'] >= .95]
    if len(captions) == 1:
        return tuple(captions[0]['box'])
    if captions:
        raise ValueError('lesson first option has ambiguous captions')
    # Fixed hiro vision/classify.py option_regions on BoxCommuEventButtonsArea.
    # A highlighted option adds a description above the physical buttons.
    panel = image[412:1089, 14:703]
    _, mask = cv2.threshold(panel, 190, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(mask, 200, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    outline = cv2.dilate(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel), kernel)
    contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = sorted((cv2.boundingRect(contour) for contour in contours), key=lambda box: box[1])
    regions = [(x, y, w, h) for x, y, w, h in regions if 350 < w < 660 and h > 60]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    if any(h < 100 and (gray[y:y+h, x:x+w] > 190).mean() < .5 for x, y, w, h in regions):
        regions = regions[1:]
    # A lower option can be the first one still visible. Its center cannot
    # represent the first physical row of this strategy.
    first = [(14+x, 412+y, w, h) for x, y, w, h in regions if h < 100 and 650 <= 412+y+h/2 <= 740]
    if len(first) != 1:
        raise ValueError('lesson first option button is missing or ambiguous')
    return first[0]


def read_lesson_options(image, rows, *, day=1):
    from .production import _text
    _text(rows, r'授業', (65, 25, 110, 75))
    current, maximum = map(int, _text(rows, r'\d{1,3}/\d{1,3}', (300, 40, 170, 80)).split('/'))
    currency = int(_text(rows, r'\d{1,6}', (340, 100, 130, 65)))
    if day not in (1, 4):
        raise ValueError('lesson options have no bound first/fourth-day source')
    preview = _lesson_option_selected(image, 665)
    selected_neighbors = sum(_lesson_option_selected(image, top) for top in (765, 865))
    if preview and selected_neighbors:
        raise ValueError('lesson first option and another option both appear selected')
    targets = {'lesson_option_1': _first_lesson_target(image, rows)}
    # A quote belongs to the visibly selected row. Switching from another
    # identified row to the first is the ordinary zero-cost preview action.
    if not preview and selected_neighbors != 1:
        from .nonbattle_observation import _within
        if any(re.fullmatch(r'-\d+', row['text']) and _within(row['box'], (190, 45, 90, 70))
                and row['score'] >= .95 for row in rows):
            raise ValueError('lesson already quotes a stamina cost without an identified selection')
    # The strategy chooses the first row. Its highlight and displayed cost
    # identify the paid confirmation even when the effect text is obscured.
    if preview:
        _text(rows, r'-5', (190, 45, 90, 70))
        if current < 5:
            raise ValueError('lesson first-option stamina is insufficient')
    return dict(scene='lesson_choice_preview' if preview else 'lesson_choice', day=day,
        stamina=current, max_stamina=maximum, currency=currency, targets=targets,
        first_option_cost=5 if preview else None)


def complete_lesson_option_captions(context, image, rows, day):
    from live_decision.hif_recovery import _field

    accepted, evidence = list(rows), []
    if day not in (1, 4):
        raise ValueError('lesson captions require the current bound lesson day')
    try:
        _first_lesson_target(image, rows)
        return accepted, evidence
    except ValueError:
        pass
    candidates = _first_lesson_captions(rows)
    if len(candidates) == 1:
        row = candidates[0]
        crop = (55, row['box'][1] - 4, 530, row['box'][3] + 8)
        proof = _field(context, image, 'DailyLessonCaption0', crop, LESSON_OPTION_CAPTION)
        accepted.append(proof)
        evidence.append(dict(proof, original_observation=row, source='independent_complete_lesson_caption'))
    return accepted, evidence


def read_lesson_exchange_offer(rows, *, day=1):
    from .exchange import exchange_offer_next
    from .production import _text
    if exchange_offer_next(rows) is None:
        raise ValueError('lesson exchange next control is not uniquely observed')
    current, maximum = map(int, _text(rows, r'\d{1,3}/\d{1,3}', (300, 40, 170, 80)).split('/'))
    currency = int(_text(rows, r'\d{1,6}', (340, 100, 130, 65)))
    hint = [row for row in rows if row['score'] >= .95 and row['text'] == 'スキルカードを選んでください'
        and 150 <= row['box'][0] and 615 <= row['box'][1]
        and row['box'][0] + row['box'][2] <= 570 and row['box'][1] + row['box'][3] <= 690]
    if len(hint) > 1:
        raise ValueError('lesson exchange empty preview hint is ambiguous')
    if day not in (1, 4):
        raise ValueError('lesson exchange has no bound lesson-day source')
    return dict(scene='card_choice', day=day, stamina=current, max_stamina=maximum, currency=currency,
        targets={}, has_unselected_hint=bool(hint))


def read_day_two_schedule(rows):
    from .calendar import read_calendar
    from .production import _text
    day, family = read_calendar(rows)
    if family != 'training':
        raise ValueError('public-course schedule requires its actual second or fifth day')
    for attribute, x in (('Vo', 70), ('Da', 255), ('Vi', 440)):
        _text(rows, attribute + r'[.]公開レッスン', (x, 1050, 195, 75))
    current, maximum = map(int, _text(rows, r'\d{1,3}/\d{1,3}', (285, 35, 165, 85)).split('/'))
    currency = int(_text(rows, r'\d{1,6}', (350, 80, 110, 75)))
    return dict(scene='training', day=day, stamina=current, max_stamina=maximum, currency=currency, targets={})


def complete_training_captions(context, image, rows, *, attributes=('Vo', 'Da', 'Vi')):
    from live_decision.hif_recovery import _field

    from .training import _course_caption_fragment
    from .production import _row
    accepted, evidence = list(rows), []
    for attribute, x in (('Vo', 70), ('Da', 255), ('Vi', 440)):
        if attribute not in attributes:
            continue
        region = (x, 1050, 195, 75)
        expected = attribute + '.公開レッスン'
        try:
            _row(rows, re.escape(expected), region)
            continue
        except ValueError:
            pass
        candidates = [row for row in rows if row['text'].startswith(attribute + '.')
            and x <= row['box'][0] and 1050 <= row['box'][1]
            and row['box'][0] + row['box'][2] <= x + 195 and row['box'][1] + row['box'][3] <= 1125]
        # A prefix or one OCR character error may request an independent
        # whole-name read. The imperfect string itself never proves the
        # course; a different complete course still cannot be overridden.
        if any(row['score'] >= .95 and not _course_caption_fragment(row['text'], expected) for row in candidates):
            raise ValueError('training caption crop cannot override a different full-frame course')
        crop = (x, max(1050, candidates[0]['box'][1] - 4), 195, candidates[0]['box'][3] + 8) if len(candidates) == 1 else region
        proof = _field(context, image, 'DailyTrainingCaption' + attribute, crop, re.escape(expected))
        accepted.append(dict(proof, source='independent_complete_course_caption',
            detected_prefix_box=candidates[0]['box'] if len(candidates) == 1 else None))
        evidence.append(dict(proof, original_observations=candidates, source='independent_complete_course_caption'))
    return accepted, evidence


def complete_training_selection(context, image, rows, attribute):
    """A located SELEC prefix never stands in for the complete SELECT word."""
    if attribute not in ('Vo', 'Da', 'Vi'):
        return rows, None
    from live_decision.hif_recovery import _field
    center = {'Vo': 168, 'Da': 354, 'Vi': 541}[attribute]
    candidates = [row for row in rows if row['text'] in ('SELEC', 'SELECT')
        and 1080 <= row['box'][1] and row['box'][1] + row['box'][3] <= 1150
        and abs(row['box'][0] + row['box'][2] / 2 - center) <= 20]
    if len(candidates) != 1:
        return rows, None
    row = candidates[0]
    if row['text'] == 'SELECT' and row['score'] >= .95:
        return rows, None
    x, y, width, height = row['box']
    proof = _field(context, image, 'DailyTrainingSelect' + attribute,
        (x - 6, y - 6, width + 16, height + 12), r'SELECT')
    evidence = dict(proof, source='independent_complete_selected_word', original_observation=row,
        detected_prefix_box=row['box'])
    return [*rows, evidence], evidence


def current_handbook_transition(rows):
    from .production import _text
    if any(row['score'] >= .95 and row['text'] == 'H.I.F本戦まで' for row in rows):
        return False
    try:
        _text(rows, r'HATSUBOSHI', (20, 760, 330, 150))
        _text(rows, r'GAKUEN', (280, 850, 310, 180))
        _text(rows, r'要確認!', (220, 470, 280, 140))
    except ValueError:
        return False
    return True


def daily_effect(state, key):
    if key == 'lesson:Vi' and state.phase == 'schedule':
        return lesson_selection_effect(state)
    if key == 'confirm_lesson' and state.phase == 'schedule':
        return LedgerEffect(f'{state.run_id}:{state.revision}:confirm-lesson:Vi', key, phase='lesson_choice')
    if key == 'lesson_option_1' and state.phase in ('lesson_choice', 'lesson_choice_preview'):
        execute = state.phase == 'lesson_choice_preview'
        return LedgerEffect(f'{state.run_id}:{state.revision}:lesson-first:{"execute" if execute else "preview"}', key,
            phase='card_choice' if execute else 'lesson_choice_preview', stamina_delta=-5 if execute else 0)
    if key == 'select_exchange_offer' and state.phase == 'card_choice':
        return LedgerEffect(f'{state.run_id}:{state.revision}:select-exchange-offer', key, phase='card_choice_preview')
    if key == 'open_exchange_replacement' and state.phase == 'card_choice_preview':
        return LedgerEffect(f'{state.run_id}:{state.revision}:open-exchange-replacement', key, phase='card_exchange')
    if key == 'select_exchange_source' and state.phase == 'card_exchange':
        return LedgerEffect(f'{state.run_id}:{state.revision}:select-exchange-source', key, phase='card_exchange_selected')
    if key == 'dismiss_exchange_result' and state.phase == 'card_exchange_result' and state.day in (1, 4):
        suffix = 'finish-first-lesson' if state.day == 1 else 'finish-lesson:4'
        return LedgerEffect(f'{state.run_id}:{state.revision}:{suffix}', key, day=state.day + 1, phase='schedule')
    if key in ('training:Vo', 'training:Da', 'training:Vi') and state.phase == 'schedule' and state.day in (2, 5):
        return LedgerEffect(f'{state.run_id}:{state.revision}:select-public-course', key, phase='training_selected')
    raise ValueError('ordinary HIF action has no bound source-stage effect')


def daily_contract(state, key, *, observed_scene=None):
    if state.day == 1 and state.phase == 'schedule' and key == 'lesson:Vi' and observed_scene == 'lesson_choice':
        return 'lesson', 'lesson_choice'
    contracts = {('schedule', 'lesson:Vi'): ('lesson', 'lesson_selected'),
        ('schedule', 'confirm_lesson'): ('lesson_selected', 'lesson_choice'),
        ('lesson_choice', 'lesson_option_1'): ('lesson_choice', 'lesson_choice_preview'),
        ('lesson_choice_preview', 'lesson_option_1'): ('lesson_choice_preview', 'card_choice'),
        ('card_choice', 'select_exchange_offer'): ('card_choice', 'card_choice_preview'),
        ('card_choice_preview', 'open_exchange_replacement'): ('card_choice_preview', 'card_exchange'),
        ('card_exchange', 'select_exchange_source'): ('card_exchange', 'card_exchange_selected'),
        ('card_exchange_selected', 'execute_lesson_exchange'): ('card_exchange_selected', 'card_exchange_result'),
        ('card_exchange_result', 'dismiss_exchange_result'): ('card_exchange_result', 'training')}
    if key in ('training:Vo', 'training:Da', 'training:Vi') and state.phase == 'schedule' and state.day in (2, 5):
        return 'training', 'training_selected'
    if key == 'confirm_training' and state.phase == 'training_selected' and state.day in (2, 5):
        return 'training_selected', 'training_result'
    if (state.phase, key) not in contracts:
        raise ValueError('ordinary HIF transition source is unbound')
    return contracts[state.phase, key]


def reconcile_unselected_preview(host,gate,report,proof,*,target_kind,unselected_scene,source_phase,outcome,
        confirmed_phase=None):
    """Preserve a dispatched zero-cost preview that verifiably was not entered."""
    from dataclasses import replace

    from . import activity_reward
    from .engine import TransitionGate, validate_observation
    from .ledger import HifLedger
    from .checkpoint import transition_checkpoint
    route, reserved = gate.pending, host.ledger.pending if host.ledger is not None else None
    if confirmed_phase is not None and (target_kind,unselected_scene,source_phase,confirmed_phase) != (
            'select_consult','consult_selected','schedule','consult_selected'):
        raise ValueError('only the actual consultation preview can replace its expired single entry')
    if route is None or reserved is None:
        raise ValueError('unselected preview has no original reserved confirmation')
    seen, effect = proof.observation, reserved.effect
    events = [event for event in report.get('route',{}).get('input_events',())
        if event.get('target_key') == route.target.key and event.get('target_kind') == route.target.kind
        and event.get('source_revision') == route.source.revision and event.get('issued_at_ms') == gate.issued_at_ms
        and event.get('dispatch_attempted') is True and event.get('success') is True and event.get('uncertain') is False]
    history = list((host.handoff or {}).get('recovery_navigation_receipts',()))
    if (not host._active or host.ownership.get('owner') != 'task600' or host.ownership.get('phase') != 'nonbattle'
            or report.get('route',{}).get('run_id') != host.run_id
            or report.get('endpoints',{}).get('game',{}).get('pid') != host.game_pid
            or route.target.kind != target_kind or len(events) != 1
            or any((effect.currency_delta,effect.stamina_delta,effect.max_stamina_delta,effect.add_cards,
                effect.remove_cards,effect.add_drinks,effect.remove_drinks,effect.counter_deltas))
            or effect.day is not None or host.clock.now() < gate.issued_at_ms+gate.timeout_ms
            or seen.scene != unselected_scene or reserved.before.phase != source_phase
            or seen.run_id != host.run_id or (seen.idol,seen.plan,seen.scenario) != (
                route.source.idol,route.source.plan,route.source.scenario)
            or proof.baseline != reserved.before or seen.day != route.source.day
            or proof.activity is not None and tuple(proof.activity.get(key) for key in ('stamina','max_stamina','currency')) != (
                reserved.before.stamina,reserved.before.max_stamina,reserved.before.currency)
            or validate_observation(seen) is not None or not source_frames_match(seen, proof.frame_ids)
            or seen.revision <= route.source.revision
            or not 0 <= seen.capture_age_ms+host.clock.now()-seen.revision <= 1000
            or set(proof.frame_ids).intersection(row.get('frame_id') for row in report.get('frames',()))
            or any(row.get('outcome') == outcome and row.get('day') == seen.day and (
                not isinstance(row.get('original_transition'),dict)
                or row['original_transition'].get('issued_at_ms') == gate.issued_at_ms) for row in history)):
        raise ValueError('confirmation lacks an independent unchanged unselected before-state')
    activity = activity_reward.request(host)
    old_ledger, old_handoff = host.ledger, host.handoff
    history.append(dict(outcome=outcome,day=seen.day,source_result=report['id'],
        original_transition=transition_checkpoint(gate,host.clock.epoch),original_ledger_pending=host.ledger.record()['pending'],
        original_activity=deepcopy(activity),original_input_event=deepcopy(events[0]),receipt_frame_ids=list(proof.frame_ids)))
    state = old_ledger.state if source_phase == 'schedule' else replace(old_ledger.state,
        phase='schedule',revision=old_ledger.state.revision+1)
    if confirmed_phase is not None:
        state = replace(old_ledger.state,phase=confirmed_phase,revision=old_ledger.state.revision+1)
    host.ledger = HifLedger(state,card_catalog=old_ledger.card_catalog,drink_ids=old_ledger.drink_ids,settled=old_ledger.settled)
    host.handoff = dict(old_handoff or {},recovery_navigation_receipts=history)
    if activity is not None:
        host.handoff = activity_reward.with_request(host.handoff,dict(activity,stage='cancelled',
            unselected_frame_ids=list(proof.frame_ids)))
    result = TransitionGate(timeout_ms=gate.timeout_ms)
    try:
        host._save(transition_checkpoint(result,host.clock.epoch))
    except BaseException:
        host.ledger,host.handoff = old_ledger,old_handoff
        raise
    return result


class HifDailyProducer:
    def __init__(self, recovery):
        self.parent = recovery
        self.context, self.host, self.clock = recovery.context, recovery.host, recovery.clock
        self.offer_reader = None

    def effect(self, key):
        from .ledger import PartialDeckLedgerState
        if key == 'execute_lesson_exchange':
            from .exchange import lesson_exchange_effect
            return lesson_exchange_effect(self.host)
        if key == 'confirm_training':
            from .training import training_effect
            return training_effect(self.host)
        state = self.host.ledger.state
        effect = daily_effect(state, key)
        if key == 'select_exchange_source' and isinstance(state, PartialDeckLedgerState):
            from .exchange import exchange_selection_effect
            return exchange_selection_effect(self.host, effect)
        return effect

    def observe(self, *, reconciliation=False, _deadline=None):
        from .engine import Target
        from .calendar import ScheduleCountdownChanged, read_route_resources, refresh_schedule_home
        from .exchange import exchange_offer_next, exchange_dialogs_absent, read_exchange_replacement
        from .production import _current_rows, _common_caption_target
        from .nonbattle_observation import exchange_receipt_text
        host, state = self.host, self.host.ledger.state
        identity = (host.handoff or {}).get('recovery_identity', {})
        if (host.ownership['owner'] != 'task600' or host.ownership['phase'] != 'nonbattle'
                or state.day not in range(1, 7)
                or state.phase not in ('schedule', 'lesson_choice', 'lesson_choice_preview',
                    'card_choice', 'card_choice_preview', 'card_exchange', 'card_exchange_selected', 'card_exchange_result',
                    'training_selected', 'training_result')
                or identity.get('run_id') != host.run_id or (identity.get('idol'), identity.get('plan')) != active_hif_role(host) or identity.get('max_stamina') != state.max_stamina):
            raise ValueError('ordinary lesson requires its bound same-run initial ledger and identity')
        if state.phase in ('card_choice', 'card_choice_preview', 'card_exchange', 'card_exchange_selected') and self.offer_reader is None:
            from .nonbattle_inventory import load_flat_inventory_readers
            self.offer_reader = load_flat_inventory_readers()[0]
        pending = host.hooks.transition_gate.pending
        if host.ledger.pending is not None and (host.ledger.pending.effect.add_drinks or host.ledger.pending.effect.remove_drinks):
            raise ValueError('ordinary lesson navigation cannot confirm a drink-changing transaction')
        training_request = (host.handoff or {}).get('nonbattle_context', {}).get('training_choice', {})
        selection_frames = training_request.get('selected_frame_ids', ())
        selection_attribute = (pending.target.attribute if pending is not None and pending.target.kind == 'training' else None)
        if ((pending is None or pending.target.kind == 'confirm_training'
                and host.ledger.pending is not None and host.ledger.pending.effect.target_key == 'confirm_training')
                and state.phase == 'training_selected'
                and training_request.get('status') == 'active'
                and training_request.get('source_ledger_revision') == state.revision - 1
                and training_request.get('source_stamina') == state.stamina
                and training_request.get('source_currency') == state.currency
                and host.ledger.settled and host.ledger.settled[-1] ==
                    f'{host.run_id}:{state.revision - 1}:select-public-course'):
            if (not isinstance(selection_frames, (list, tuple)) or len(selection_frames) != 1
                    or not isinstance(selection_frames[0], str) or not selection_frames[0]):
                raise ValueError('confirmed public-course preview requires its single selection frame')
            selection_attribute = training_request.get('selected_attribute')
        observation_budget = (host.hooks.transition_gate.timeout_ms if pending is not None
            and pending.target.key in ('dismiss_exchange_result', 'confirm_training') and not reconciliation else 8000)
        deadline = self.clock.now() + observation_budget if _deadline is None else _deadline
        if pending is not None and not reconciliation:
            deadline = min(deadline, host.hooks.transition_gate.issued_at_ms + host.hooks.transition_gate.timeout_ms)
        checking_exchange = (pending is not None and pending.target.key == 'execute_lesson_exchange'
            and host.ledger.pending is not None and host.ledger.pending.effect.target_key == pending.target.key)
        previous_absent = None
        while True:
            started = self.clock.now()
            if started >= deadline or self.context.tasker.stopping:
                raise ValueError('ordinary HIF observation deadline reached')
            capture = self.context.tasker.controller.post_screencap().wait()
            if not capture.succeeded:
                raise ValueError('ordinary HIF capture failed')
            image = capture.get()
            key = f'route-frame-{self.parent.game_pid}-{started}'
            rows = _current_rows(self.context, image)
            countdown_frame_id = None
            calendar = None
            if state.phase == 'schedule' and pending is None and host.ledger.pending is None and not reconciliation:
                try:
                    current = refresh_schedule_home(self.parent, image, rows, key, day=state.day)
                except ScheduleCountdownChanged:
                    continue
                if current is not None:
                    countdown_frame_id = key
                    started, image, rows, calendar = current
                    key = f'route-frame-{self.parent.game_pid}-{started}'
            if (pending is not None and pending.target.key == 'dismiss_exchange_result'
                    and state.phase == 'card_exchange_result' and current_handbook_transition(rows)):
                metadata = dict(frame_id=key, phase='day_transition',
                    reason='observed_handbook_animation_requires_actual_next_day_receipt', rows=rows,
                    original_issued_at_ms=host.hooks.transition_gate.issued_at_ms,
                    original_timeout_ms=host.hooks.transition_gate.timeout_ms)
                if self.parent.evidence_sink:
                    self.parent.evidence_sink(image, metadata)
                self.parent.events.append(metadata)
                time.sleep(.1)
                continue
            exchange_result = re.fullmatch(r'.+を.+にチェンジしました[。]?', exchange_receipt_text(rows))
            # A return to the offer page has its own same-transaction census.
            # Read that destination before using the disappearance fallback.
            restored_offer = checking_exchange and reconciliation and exchange_offer_next(rows) is not None
            if checking_exchange and not exchange_result and not restored_offer:
                context = (host.handoff or {}).get('nonbattle_context', {})
                exchange = context.get('lesson_exchange', {})
                if exchange.get('status') == 'active' and not exchange.get('closed_frame_ids'):
                    if exchange_dialogs_absent(image, rows):
                        if previous_absent is None:
                            previous_absent = key
                            metadata = dict(phase='exchange_dialog_absent', frame_id=key, rows=rows)
                            if self.parent.evidence_sink:
                                self.parent.evidence_sink(image, metadata)
                            self.parent.events.append(metadata)
                            time.sleep(.5)
                            continue
                        if previous_absent != key:
                            # A disappearing dialog acknowledges only the UI, not either card instance.
                            frames = [previous_absent, key]
                            host.handoff = dict(host.handoff, nonbattle_context=dict(context,
                                lesson_exchange=dict(exchange, closed_frame_ids=frames)))
                            self.parent.events.append(dict(phase='exchange_dialog_closed', frame_ids=frames,
                                transaction_id=host.ledger.pending.effect.transaction_id))
                    else:
                        previous_absent = None
            elif checking_exchange:
                previous_absent = None
            try:
                replacement = None
                if state.day in (1, 4) and state.phase in ('card_choice_preview', 'card_exchange', 'card_exchange_selected'):
                    request = (host.handoff or {}).get('nonbattle_context', {}).get('lesson_exchange')
                    try:
                        replacement = read_exchange_replacement(rows, image, day=state.day,
                            catalog=host.ledger.card_catalog, request=request)
                    except ValueError:
                        pass
                if replacement:
                    page = replacement
                    # This explicit, cancelable chooser opens without a
                    # resource mutation. Its opaque modal hides the footer;
                    # maintain only the already-bound zero-action baseline,
                    # and require its original incoming-card proof below.
                    page.update(stamina=state.stamina, max_stamina=state.max_stamina, currency=state.currency,
                        maintained_resource_revision=state.revision)
                    resource_crops = []
                else:
                    course_header = any(re.fullmatch(r'(?:Vo|Da|Vi)[.]公開レッスン', row['text'])
                        and row['score'] >= .95 and row['box'][0] < 200 and row['box'][1] < 80 for row in rows)
                    expectation = host.ledger.pending
                    submitted = (expectation.expected if course_header and expectation is not None
                        and expectation.effect.target_key == 'confirm_training' else None)
                    parsed_rows, resource_crops = complete_daily_resources(self.context, image, rows,
                        submitted_stamina=(submitted.stamina, submitted.max_stamina) if submitted else None)
                from .activity_reward import support_source
                from .activity_production import support_award
                support = (support_award(image, parsed_rows, host) if not replacement
                    and support_source(pending) is not None else None)
                if replacement:
                    pass
                elif support is not None:
                    page = dict(read_route_resources(parsed_rows), scene='support_reward', day=state.day,
                        targets={'activity_ack_support_card': support['target']}, grant=support['grant'],
                        maintained_resource_revision=state.revision)
                elif state.day in (1, 4) and exchange_offer_next(rows) is not None:
                    # CardSwapFlag identifies the physical dialog even when
                    # a date interruption returned a submitted exchange to the
                    # offers. Its transaction is settled only by the full census.
                    page = read_lesson_exchange_offer(parsed_rows, day=state.day)
                elif calendar is not None or any(row['text'] == 'H.I.F本戦まで' and row['score'] >= .95 for row in rows):
                    from .calendar import read_calendar
                    calendar = calendar or read_calendar(parsed_rows)
                    _, family = calendar
                    if family == 'training':
                        from .training import read_training_page
                        parsed_rows, selection_crop = complete_training_selection(self.context, image, parsed_rows, selection_attribute)
                        caption_crops = []
                        if selection_attribute is None:
                            parsed_rows, caption_crops = complete_training_captions(self.context, image, parsed_rows)
                        request = (host.handoff or {}).get('nonbattle_context', {}).get('training_choice')
                        page = dict(read_training_page(self.context, image, parsed_rows, request,
                            expected_selection=selection_attribute, calendar=calendar),
                            caption_crops=caption_crops, selection_crop=selection_crop)
                    elif family == 'lesson':
                        page = read_lesson_schedule(image, parsed_rows, calendar=calendar, confirmation_pending=(not reconciliation
                            and pending is not None and pending.target.kind == 'confirm_lesson'))
                    else:
                        raise ValueError('ordinary calendar page family is not yet bound: ' + family)
                elif exchange_result:
                    from .exchange import read_lesson_exchange_result
                    request = (host.handoff or {}).get('nonbattle_context', {}).get('lesson_exchange', {})
                    page = read_lesson_exchange_result(parsed_rows, host.ledger.card_catalog, request, day=state.day)
                elif course_header:
                    from .training import complete_training_sp_field, read_training_after_submission
                    request = (host.handoff or {}).get('nonbattle_context', {}).get('training_choice', {})
                    parsed_rows, sp_proof = complete_training_sp_field(self.context, image, parsed_rows,
                        expected_special=request.get('selected_special'))
                    page = read_training_after_submission(parsed_rows, request)
                    page['sp_field_evidence'] = sp_proof
                    page['maintained_resource_revision'] = state.revision
                else:
                    if state.day not in (1, 4):
                        raise ValueError('ordinary page does not match a bound current-day family')
                    parsed_rows, option_crops = complete_lesson_option_captions(self.context, image, parsed_rows, state.day)
                    page = read_lesson_options(image, parsed_rows, day=state.day)
                    page['option_caption_crops'] = option_crops
                page['resource_crops'] = resource_crops
                if 'options' in page:
                    page['option_contract'] = [(option['attribute'], option['special'], option['cost'], option['available'])
                        for option in page['options']]
                maintained = state
                if host.ledger.pending is not None:
                    _, destination = daily_contract(state, host.ledger.pending.effect.target_key)
                    if page['scene'] == destination:
                        maintained = host.ledger.pending.expected
                course_result = (page['scene'] in ('training_result', 'training_animation') and host.ledger.pending is not None
                    and host.ledger.pending.effect.target_key == 'confirm_training')
                if course_result:
                    from .rewards import request, validate_resources
                    if request(host) is None or page['day'] != state.day:
                        raise ValueError('public-course result lacks its original reserved reward intent')
                    validate_resources(host, page)
                elif page['day'] != maintained.day or (page['stamina'], page['max_stamina'], page['currency']) != (
                        maintained.stamina, maintained.max_stamina, maintained.currency):
                    raise ValueError('lesson navigation resources differ from their bound complete ledger')
                hidden_footer = replacement is not None or page['scene'] in ('training_result', 'training_animation', 'support_reward')
                # The menu and cup positions do not supply a schedule action.
                # Keep the existing dialog layout where the world HUD is visible.
                page['footer_evidence'] = None
                page['footer_offset'] = None if hidden_footer or page['scene'] in (
                    'lesson', 'lesson_selected', 'training', 'training_selected') else 0
                # These lesson/course actions never change or use drinks.
                # Keep the verified ledger source without claiming a fresh HUD
                # identity. Drink rewards, purchases, use and battle readiness
                # retain their own actual inventory/detail observations.
                page['drink_slots'] = None
                page['maintained_drink_ledger_revision'] = state.revision
                page['current_drink_observation'] = False
            except ValueError as error:
                metadata = dict(frame_id=key, phase='unknown', reason=str(error), rows=rows)
                if isinstance(getattr(error, 'evidence', None), dict):
                    metadata['field_evidence'] = error.evidence
                from .training import TrainingBadgeNotObserved
                if isinstance(error, TrainingBadgeNotObserved):
                    metadata['badge_rejection'] = error.evidence
                if self.parent.evidence_sink:
                    self.parent.evidence_sink(image, metadata)
                self.parent.events.append(metadata)
                transient_preview = (state.phase == 'schedule' and str(error) in (
                    'lesson preview does not uniquely select the supported visual lesson',
                    'lesson preview awaits its current SELECT or destination',
                    'all three current lesson attributes must be independently observed'))
                transient_training = (state.day in (2, 5) and state.phase == 'schedule'
                    and any(row['text'] == 'H.I.F本戦まで' and row['score'] >= .95 for row in rows))
                unresolved = getattr(error, 'evidence', {})
                weak_complete_crop = (unresolved.get('field', '').startswith('DailyTraining')
                    and type(unresolved.get('confidence')) in (int, float) and unresolved['confidence'] < .95
                    and unresolved.get('text') in ('SELECT', *(attribute + '.公開レッスン' for attribute in ('Vo', 'Da', 'Vi'))))
                missing_preview_crop = str(error).startswith('HIF recovery field is missing: DailyTraining')
                transient_selected = (selection_attribute is not None and state.phase == 'training_selected'
                    and (weak_complete_crop or missing_preview_crop
                        or str(error) == 'public-course caption has no independently located input target'))
                transient_selected = transient_selected or (selection_attribute is not None and state.phase == 'training_selected'
                    and str(error) == 'public-course original preview requires a complete current SELECT')
                retry_badge = isinstance(error, TrainingBadgeNotObserved) and selection_attribute == error.evidence['attribute']
                if (pending is not None or retry_badge or transient_preview or transient_training or transient_selected) and self.clock.now() < deadline:
                    time.sleep(.1)
                    continue
                raise
            if self.parent.evidence_sink:
                self.parent.evidence_sink(image, dict(frame_id=key, phase=page['scene'], observation=page, rows=rows))
            keys, images = [key], [image]
            break
        if (pending is not None and not reconciliation
                and self.clock.now() >= host.hooks.transition_gate.issued_at_ms + host.hooks.transition_gate.timeout_ms):
            raise ValueError('ordinary HIF observation deadline reached')
        if page['scene'] == 'support_reward':
            from .engine import Target
            from .production import _common_caption_target
            from .activity_reward import support_source
            from .activity_production import ACQUIRE_SKIP, activity_page_id
            box = _common_caption_target([dict(target=page['targets']['activity_ack_support_card'])])
            if support_source(pending) == 'lesson:Vi' and page['grant'].get('kind') != 'upgrade':
                box = ACQUIRE_SKIP
            revision = self.clock.now()
            observation = Observation(host.run_id, revision, 'support_reward',
                activity_page_id(state.day, 'support_reward', host.ledger.pending.effect.transaction_id, page),
                state.day, *active_hif_role(host), revision-started, len(keys), 'HIF',
                (Target('activity_ack_support_card', 'activity_ack_support_card', available=True),))
            self.parent.last_frame_ids = tuple(keys)
            return HifRoutePage(observation, {'activity_ack_support_card': box}, baseline=state,
                frame_ids=tuple(keys), activity=dict(page, kind='support_reward', frame_ids=list(keys)))
        scene = page['scene']
        if scene in ('training', 'training_selected'):
            context = (host.handoff or {}).get('nonbattle_context', {})
            request = context.get('training_choice')
            if request is not None and request.get('status') == 'complete' and request.get('day') != page['day']:
                request = None
            if request is not None and (request.get('schema_version') != 1 or request.get('run_id') != host.run_id
                    or request.get('game_pid') != host.game_pid or request.get('day') != page['day']):
                raise ValueError('public-course request belongs to another scope')
            if scene == 'training' and pending is None:
                options = page['options']
                request = dict(schema_version=1, run_id=host.run_id, game_pid=host.game_pid, day=state.day,
                    source_ledger_revision=state.revision, source_stamina=state.stamina, source_currency=state.currency,
                    source_frame_ids=list(keys), options=options, status='active')
                host.handoff = dict(host.handoff or {}, nonbattle_context=dict(context, training_choice=request))
            elif scene == 'training_selected':
                if request is None or (pending is not None and pending.target.key.startswith('training:')
                        and pending.target.attribute != page['selected_attribute']):
                    raise ValueError('public-course selected preview is not the original chosen target')
                if state.phase == 'training_selected' and any(request.get(key) != page[key] for key in (
                        'selected_attribute', 'selected_special', 'selected_cost')):
                    raise ValueError('public-course selected preview differs from its confirmed original choice')
                request = dict(request, selected_attribute=page['selected_attribute'], selected_special=page['selected_special'],
                    selected_cost=page['selected_cost'], selected_frame_ids=list(keys))
                host.handoff = dict(host.handoff or {}, nonbattle_context=dict(context, training_choice=request))
        arriving_offer = (pending is not None and host.ledger.pending is not None
            and state.phase == 'lesson_choice_preview' and host.ledger.pending.effect.target_key == 'lesson_option_1'
            and maintained.phase == 'card_choice')
        if scene == 'card_choice' and maintained.phase in ('card_choice', 'card_choice_preview'):
            try:
                from .exchange import read_exchange_offers, selected_exchange_offer
                from .candidate_pool import candidate_policy, maybe_edit_pool, preferred_index
                if self.offer_reader is None:
                    from .nonbattle_inventory import load_flat_inventory_readers
                    self.offer_reader = load_flat_inventory_readers()[0]
                context = (host.handoff or {}).get('nonbattle_context', {})
                request = context.get('lesson_exchange')
                if request is not None and request.get('status') == 'complete' and request.get('day', 1) != state.day:
                    request = None
                registered = ([offer['box'] for offer in request['offers']]
                    if request is not None and not page['has_unselected_hint'] else None)
                preferred, _ = candidate_policy(self.context)
                locked_index = request.get('selected_index', 2) if registered is not None else None
                selection = dict(preferred=preferred, selected_index=locked_index) if preferred or locked_index not in (None, 2) else {}
                offers = read_exchange_offers(self.offer_reader, images, registered_boxes=registered,
                    plan=active_hif_role(host)[1], **selection)
                ids = [offer['card_id'] for offer in offers]
                selected_index = preferred_index(offers, preferred, 2) if locked_index is None else locked_index
                if registered is not None and (request.get('schema_version') != 1
                        or request.get('run_id') != host.run_id or request.get('game_pid') != host.game_pid
                        or request.get('day', 1) != state.day
                        or request['offers'][selected_index]['card_id'] != ids[selected_index]):
                    raise ValueError('lesson exchange original offer snapshot changed')
                if page['has_unselected_hint']:
                    if maintained.phase != 'card_choice' or pending is not None and not arriving_offer:
                        raise ValueError('lesson exchange pending preview did not reach its selected detail')
                    maybe_edit_pool(host, self.context, image, rows, offers, family='lesson', default=selected_index)
                    request = dict(schema_version=1, run_id=host.run_id, game_pid=host.game_pid, day=state.day,
                        source_ledger_revision=maintained.revision, offers=offers, selected_index=selected_index,
                        source_frame_ids=list(keys), status='active')
                    host.handoff = dict(host.handoff or {}, nonbattle_context=dict(context, lesson_exchange=request))
                    page['targets'] = {'select_exchange_offer': tuple(offers[selected_index]['box'])}
                else:
                    if request is None:
                        raise ValueError('lesson exchange cold detail has no original offer request')
                    detail = selected_exchange_offer(self.parent, self.offer_reader, images, [rows], request, plan=active_hif_role(host)[1])
                    page.update(scene='card_choice_preview', chosen_card=detail['card_id'],
                        targets={'open_exchange_replacement': detail['continue_target']})
                    scene = 'card_choice_preview'
                page['offer_ids'] = ids
                page['selected_offer_index'] = selected_index
            except ValueError as error:
                from .candidate_pool import CandidatePoolObserved

                if isinstance(error, CandidatePoolObserved):
                    raise
                self.parent.events.append(dict(phase='unresolved_exchange_offers', frame_ids=keys, reason=str(error)))
                if self.clock.now() < deadline:
                    return self.observe(reconciliation=reconciliation, _deadline=deadline)
                raise
        if scene in ('card_exchange', 'card_exchange_selected'):
            from .exchange import basic_source_group, selected_source_identity, replacement_chosen_identity
            request = (host.handoff or {}).get('nonbattle_context', {}).get('lesson_exchange')
            if (not isinstance(request, dict) or request.get('schema_version') != 1
                    or request.get('run_id') != host.run_id or request.get('game_pid') != host.game_pid
                    or request.get('status') != 'active' or request.get('selected_index') not in range(3)):
                raise ValueError('replacement chooser lacks its original same-run incoming offer')
            try:
                detail = replacement_chosen_identity(self.offer_reader, images, request, catalog=host.ledger.card_catalog, plan=active_hif_role(host)[1])
                group = basic_source_group(self.parent, self.offer_reader, images, tuple(keys), plan=active_hif_role(host)[1])
                if 'outgoing_card' in request and (request['outgoing_card'] != group['outgoing_card']
                        or request['outgoing_box'] != group['outgoing_box']):
                    raise ValueError('replacement original physical source instance changed')
                if scene == 'card_exchange':
                    if pending is not None and pending.target.key != 'open_exchange_replacement':
                        raise ValueError('replacement source selection has not reached its detail')
                    context = (host.handoff or {}).get('nonbattle_context', {})
                    request = dict(request, **group)
                    host.handoff = dict(host.handoff or {}, nonbattle_context=dict(context, lesson_exchange=request))
                    page['targets'] = {'select_exchange_source': tuple(group['outgoing_box'])}
                else:
                    selected_source_identity(self.offer_reader, images, request, catalog=host.ledger.card_catalog, plan=active_hif_role(host)[1])
                    request = dict(request, selected_incoming=dict(detail,
                        original_offer_card_id=request['offers'][request['selected_index']]['card_id'], frame_ids=list(keys)))
                    context = (host.handoff or {}).get('nonbattle_context', {})
                    host.handoff = dict(host.handoff or {}, nonbattle_context=dict(context, lesson_exchange=request))
            except ValueError as error:
                self.parent.events.append(dict(phase='unresolved_incoming_replacement_card', frame_ids=keys, reason=str(error)))
                if pending is not None and self.clock.now() < deadline:
                    return self.observe(reconciliation=reconciliation, _deadline=deadline)
                raise
            page['incoming_card_id'] = detail['card_id']
        if scene == 'card_exchange_result' and host.ledger.pending is None and state.phase == scene:
            context = (host.handoff or {}).get('nonbattle_context', {})
            request = context.get('lesson_exchange')
            host.handoff = dict(host.handoff or {}, nonbattle_context=dict(context,
                lesson_exchange=dict(request, status='complete', exchange_receipt_frame_ids=list(keys))))
        if scene == 'training_result':
            page['targets'] = {'training_reward_ack': page['result_target']}
        boxes = {(f'lesson:{key}' if scene == 'lesson' else key): _common_caption_target([
            dict(target=page['targets'][key])]) for key in page['targets']}
        targets = tuple(Target(key, 'lesson' if scene == 'lesson' else 'training' if scene == 'training'
            else 'lesson_option' if key.startswith('lesson_option_') else key,
            attribute=key.split(':')[1] if scene in ('lesson', 'training') else None,
            special=next(option['special'] for option in page['options'] if option['attribute'] == key.split(':')[1])
                if scene == 'training' else None,
            available=next(option['available'] for option in page['options'] if option['attribute'] == key.split(':')[1])
                if scene == 'training' else True) for key in boxes)
        revision = self.clock.now()
        day = page['day']
        observation = Observation(host.run_id, revision, scene, _daily_page_id(page, scene=scene), day,
            *active_hif_role(host), revision - started, len(keys), 'HIF', targets)
        if scene == 'training_result':
            observation = replace(observation, page_id=f'reward:{host.ledger.pending.effect.transaction_id}:result')
        receipt = None
        if host.ledger.pending is not None:
            effect = host.ledger.pending.effect
            source, destination = daily_contract(state, effect.target_key, observed_scene=scene)
            if (effect != self.effect(effect.target_key)
                    or pending is None or pending.target.key != effect.target_key or pending.source.scene != source):
                raise ValueError('lesson preview cannot settle another pending resource action')
            if scene == destination and effect.target_key != 'confirm_training':
                receipt = host.ledger.pending.expected
                if effect.target_key == 'lesson:Vi' and scene == 'lesson_choice':
                    receipt = replace(receipt, phase='lesson_choice')
        self.parent.events.append(dict(phase=scene, frame_ids=keys, capture_age_ms=observation.capture_age_ms))
        self.parent.last_frame_ids = tuple(keys)
        effect_keys = {'lesson': 'lesson:Vi', 'lesson_selected': 'confirm_lesson',
            'lesson_choice': 'lesson_option_1', 'lesson_choice_preview': 'lesson_option_1',
            'card_choice': 'select_exchange_offer', 'card_choice_preview': 'open_exchange_replacement',
            'card_exchange': 'select_exchange_source', 'card_exchange_selected': 'execute_lesson_exchange',
            'card_exchange_result': 'dismiss_exchange_result', 'training_selected': 'confirm_training'}
        effects = ()
        if targets and scene != 'training_result' and (host.ledger.pending is None
                or scene == daily_contract(state, host.ledger.pending.effect.target_key)[0]):
            if scene == 'training':
                plan = decide_hif(observation, HifConfig(mode='advance', main_attribute=training_main_attribute(self.context)))
                if plan.proposal is not None:
                    effects = (self.effect(plan.proposal.target.key),)
            else:
                effects = (self.effect(effect_keys[scene]),)
        return HifRoutePage(observation, boxes, baseline=state, frame_ids=tuple(keys),
            schedule_countdown_frame_id=countdown_frame_id if scene in ('lesson', 'training') else None,
            effects=effects,
            receipt=receipt, receipt_frame_ids=tuple(keys) if receipt is not None else (),
            reward=dict(page, kind='result' if scene == 'training_result' else 'animation', frame_ids=list(keys))
                if scene in ('training_result', 'training_animation') else None)

    def decide(self, observation, config):
        if observation.scene == 'lesson_selected':
            from .engine import propose_unique
            return propose_unique(observation, observation.targets, 'hif-confirm-verified-visual-lesson', ('lesson_choice', 'support_reward'))
        if observation.scene == 'card_choice' and observation.targets:
            from .engine import propose_unique
            return propose_unique(observation, observation.targets, 'hiro-known-rightmost-exchange-offer', ('card_choice_preview',))
        if observation.scene == 'card_choice_preview':
            from .engine import propose_unique
            return propose_unique(observation, observation.targets, 'hif-open-original-exchange-replacement', ('card_exchange',))
        if observation.scene in ('card_exchange', 'card_exchange_selected', 'card_exchange_result'):
            from .engine import propose_unique
            expected = {'card_exchange': 'card_exchange_selected', 'card_exchange_selected': 'card_exchange_result',
                'card_exchange_result': 'training'}[observation.scene]
            destinations = (expected, 'support_reward') if observation.scene == 'card_exchange_result' else (expected,)
            return propose_unique(observation, observation.targets, 'hif-current-physical-lesson-exchange', destinations)
        return decide_hif(observation, config)

    def _reconcile_unselected_lesson(self, gate, report, proof, events):
        return reconcile_unselected_preview(self.host,gate,report,proof,target_kind='confirm_lesson',
            unselected_scene='lesson',source_phase='schedule',outcome='verified_unselected_after_lesson_confirmation')

    def prepare(self, gate, report):
        from .engine import TransitionGate
        from .checkpoint import transition_checkpoint
        from .input_receipt import partial_first_lesson
        host, pending = self.host, gate.pending
        if pending is None:
            return gate
        source, _ = daily_contract(host.ledger.state, pending.target.key)
        if (not host._active or host.ownership['owner'] != 'task600' or host.ownership['phase'] != 'nonbattle'
                or pending.source.scene != source or pending.source.day != host.ledger.state.day
                or host.ledger.pending is None or host.ledger.pending.effect != self.effect(pending.target.key)
                or report.get('route', {}).get('run_id') != host.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != host.game_pid):
            raise ValueError('late lesson preview cannot discard another pending resource action')
        events = [event for event in report['route'].get('input_events', [])
            if event.get('target_key') == pending.target.key and event.get('target_kind') == pending.target.kind
            and event.get('source_revision') == pending.source.revision and event.get('issued_at_ms') == gate.issued_at_ms
            and event.get('dispatch_attempted') is True
            and (event.get('success') is True and event.get('uncertain') is False or partial_first_lesson(event, pending))]
        proof = self.observe(reconciliation=True)
        if pending.target.kind == 'execute_lesson_exchange' and proof.observation.scene == 'card_choice':
            from .exchange_census import reconcile_before_exchange
            return reconcile_before_exchange(host, gate, report, proof, events)
        if pending.target.kind == 'confirm_lesson' and proof.observation.scene == 'lesson':
            return self._reconcile_unselected_lesson(gate, report, proof, events)
        from .activity_reward import support_source
        if support_source(pending) is not None and proof.observation.scene == 'support_reward':
            return host.reconcile_activity_navigation(gate, report, proof)
        if pending.target.kind == 'confirm_training':
            return host.reconcile_submitted_training(gate, report, proof)
        expected_day = host.ledger.pending.expected.day
        _, destination = daily_contract(host.ledger.state, pending.target.key, observed_scene=proof.observation.scene)
        if (len(events) != 1 or destination not in pending.expected_scenes
                or proof.observation.run_id != host.run_id or proof.observation.scene != destination
                or proof.observation.day != expected_day or not source_frames_match(proof.observation, proof.frame_ids)
                or proof.observation.revision <= pending.source.revision
                or not 0 <= proof.observation.capture_age_ms + self.clock.now() - proof.observation.revision <= 1000
                or proof.receipt_frame_ids != proof.frame_ids
                or set(proof.frame_ids) & {frame.get('frame_id') for frame in report.get('frames', [])}):
            raise ValueError('late lesson preview lacks independent current destination evidence')
        original = transition_checkpoint(gate, self.clock.epoch)
        old_ledger, old_handoff = host.ledger, host.handoff
        settled = host.confirm_page_ledger(proof)
        receipts = list((host.handoff or {}).get('recovery_navigation_receipts', []))
        receipts.append(dict(outcome='late_verified_lesson_destination_observed', original_transition=original,
            original_ledger_pending=host.ledger.record()['pending'], source_result=report['id'],
            receipt_frame_ids=list(proof.frame_ids)))
        host.ledger, host.handoff = settled, dict(host.handoff or {}, recovery_navigation_receipts=receipts)
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            host._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            host.ledger, host.handoff = old_ledger, old_handoff
            raise
        return result
