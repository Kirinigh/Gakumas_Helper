"""Ordinary public-course pages for both training days, never battle control."""
import re
from pathlib import Path
from functools import lru_cache

import cv2
from utils.runtime_paths import resolve_runtime_path

from .calendar import read_calendar, read_route_resources


class TrainingBadgeNotObserved(ValueError):
    """Only the pulsing badge is unknown; no SP value is supplied for it."""
    def __init__(self, proof):
        super().__init__('public-course selected SP badge is temporarily unobserved')
        self.evidence = proof


SP_HEADER_ROI = (110, 80, 85, 70)


def _sp_header_rows(rows):
    x, y, width, height = SP_HEADER_ROI
    return [row for row in rows if row['text'] == 'SP' and x <= row['box'][0] and y <= row['box'][1]
        and row['box'][0] + row['box'][2] <= x + width and row['box'][1] + row['box'][3] <= y + height]


def complete_training_sp_field(context, image, rows, *, expected_special):
    from live_decision.hif_recovery import _field
    if type(expected_special) is not bool:
        raise ValueError('public-course original submitted SP type is incomplete')
    candidates = _sp_header_rows(rows)
    if len(candidates) > 1:
        raise ValueError('public-course result SP field is ambiguous')
    if candidates and candidates[0]['score'] >= .95 or not candidates and expected_special is False:
        return rows, None
    region = SP_HEADER_ROI
    if candidates:
        x, y, width, height = candidates[0]['box']
        region = (x - 6, y - 6, width + 12, height + 12)
    proof = _field(context, image, 'CourseResultSP', region, r'SP')
    evidence = dict(proof, source='independent_complete_course_sp',
        original_observations=candidates)
    return [*rows, evidence], evidence


def unresolved_training_sp_field(error):
    return str(error) in ('HIF recovery field is unresolved: CourseResultSP',
        'HIF recovery field is missing: CourseResultSP')


@lru_cache(maxsize=1)
def _sp_reference():
    image = cv2.imread(str(resolve_runtime_path(Path('assets/resource/base/image/produce/sp.png'),
        Path('resource/base/image/produce/sp.png'))))
    if image is None or image.shape[:2] != (39, 40):
        raise ValueError('existing public-course SP reference is unavailable')
    return image


def read_special_course(image, index, *, selected=False):
    if getattr(image, 'shape', ()) != (1280, 720, 3) or index not in (0, 1, 2):
        raise ValueError('public-course image or physical column is uncalibrated')
    left = (86, 272, 459)[index]
    top = 903 if selected else 919
    x, y = (70, 250, 430)[index], top - 19
    response = cv2.matchTemplate(image[y:y + 80, x:x + 80], _sp_reference(), cv2.TM_CCOEFF_NORMED)
    _, score, _, point = cv2.minMaxLoc(response)
    if score >= .8:
        count, _ = cv2.connectedComponents((response >= .8).astype('uint8'))
        if count != 2 or abs(x + point[0] - (left - 3)) > 2 or abs(y + point[1] - top) > 2:
            raise ValueError('public-course SP badge is not uniquely anchored')
        return dict(special=True, confidence=float(score), box=[x + point[0], y + point[1], 40, 39], source='existing_sp_template')
    # Positive complete normal-card corner, independent of a badge miss.
    # Black/white occlusion or an intermediate badge response is UNKNOWN.
    corner = image[top:top + 39, left - 3:left + 37].min(2) > 235
    if selected:
        # The selected normal card's white cap ends before its coloured body
        # (084-after, 085-before, 086-before/preinput). Its complete cap and
        # separate lower edge must both exist; do not borrow the root profile.
        proof = dict(inside=float(corner[10:24, 20:37].mean()), outside=float(corner[:7, :6].mean()),
            top=float(corner[3:8, 18:37].mean()), left=float(corner[14:24, 3:8].mean()),
            below_cap=float(corner[30:38, 20:37].mean()))
    else:
        proof = dict(inside=float(corner[10:35, 20:37].mean()), outside=float(corner[:7, :6].mean()),
            top=float(corner[3:8, 18:37].mean()), left=float(corner[20:37, 3:8].mean()))
    if score > .4 or not (proof['inside'] >= .98 and proof['outside'] <= .05 and proof['top'] >= .9 and proof['left'] >= .55):
        raise ValueError('public-course SP absence is not positively observed')
    if selected and proof['below_cap'] > .05:
        raise ValueError('public-course normal selected cap has no complete lower boundary')
    return dict(special=False, confidence=float(score), normal_corner=proof, source='complete_normal_course_corner')


def read_training_cost(context, image, rows, index):
    from live_decision.hif_recovery import _field

    from .production import _row
    left = (86, 272, 459)[index]
    region = (left + 95, 900, 80, 65)
    try:
        row = _row(rows, r'-\d{1,2}', region)
        return int(row['text'][1:]), dict(row, source='complete_signed_cost_row')
    except ValueError:
        pass
    x, y, width, height = region
    numeric = [row for row in rows if row['score'] >= .95 and re.fullmatch(r'\d{1,2}', row['text'])
        and x <= row['box'][0] and y <= row['box'][1]
        and row['box'][0] + row['box'][2] <= x + width and row['box'][1] + row['box'][3] <= y + height]
    if len(numeric) == 1:
        row = numeric[0]
        bx, by, _, high = row['box']
        top = by + high // 2 - 4
        mask = (image[top:top + 9, bx - 12:bx].min(2) > 230).astype('uint8')
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        strokes = [tuple(map(int, value)) for value in stats[1:] if 7 <= value[2] <= 8
            and 1 <= value[3] <= 2 and value[4] >= value[2] * value[3] * .8]
        if len(strokes) == 1:
            sx, sy, wide, high, _ = strokes[0]
            return int(row['text']), dict(source='independent_numeric_and_complete_minus_core', numeric=row,
                minus_box=[bx - 12 + sx, top + sy, wide, high])
        if context is None:
            raise ValueError('public-course numeric cost lacks its complete minus stroke')
        # A differently rasterised minus can miss the old pixel geometry.
        # Read the whole signed quote through the existing OCR crop instead;
        # an unsigned numeral still cannot authorize any stamina debit.
    elif numeric:
        raise ValueError('public-course cost has multiple numeric candidates')
    proof = _field(context, image, 'PublicCourseCost' + str(index), (left + 115, 915, 60, 45), r'-\d{1,2}')
    return int(proof['text'][1:]), dict(proof, source='independent_complete_cost_crop')


def _caption(rows, attribute, index):
    from .production import _row
    region = ((70, 255, 440)[index], 1050, 195, 75)
    caption = _row(rows, re.escape(attribute + '.公開レッスン'), region)
    # A crop proves a whole name; use its already located original prefix
    # as the physical text target, never the entire crop as a click fallback.
    if caption.get('source') == 'independent_complete_course_caption':
        if not caption.get('detected_prefix_box'):
            raise ValueError('public-course caption has no independently located input target')
        return tuple(caption['detected_prefix_box'])
    return tuple(caption['box'])


def _course_caption_fragment(value, expected):
    """A fragment is unresolved text, never an independent course identity."""
    if expected.startswith(value):
        return True
    if len(value) == len(expected):
        return sum(a != b for a, b in zip(value, expected, strict=True)) <= 1
    if abs(len(value) - len(expected)) == 1:
        longer, shorter = (value, expected) if len(value) > len(expected) else (expected, value)
        return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))
    return False


def _selected_caption(rows, attribute, index):
    from .production import _row
    try:
        return _caption(rows, attribute, index)
    except ValueError:
        pass
    region = ((70, 255, 440)[index], 1050, 195, 75)
    # The original three-column request already proved the course. A current
    # located fragment supplies its target only; SELECT, cost and SP below
    # must still agree with that same original option before confirmation.
    caption = _row(rows, r'(?:Vo|Da|Vi)[.].+', region, minimum=.85)
    if not caption['text'].startswith(attribute + '.公開'):
        raise ValueError('selected course caption contradicts its original course')
    return tuple(caption.get('detected_prefix_box') or caption['box'])


def read_training_page(context, image, rows, request=None, *, expected_selection=None, calendar=None):
    from .production import _row
    day, family = calendar or read_calendar(rows)
    if family != 'training':
        raise ValueError('public courses require the actual second or fifth day')
    resources = read_route_resources(rows)
    attributes = ('Vo', 'Da', 'Vi')
    selected = [row for row in rows if row['score'] >= .95 and row['text'] == 'SELECT'
        and 70 <= row['box'][0] and 1080 <= row['box'][1]
        and row['box'][0] + row['box'][2] <= 650 and row['box'][1] + row['box'][3] <= 1150]
    if selected:
        if len(selected) != 1 or request is None or request.get('day') != day:
            raise ValueError('public-course cold preview lacks its original current-day options')
        if (len(request.get('options', ())) != 3
                or [option.get('attribute') for option in request['options']] != list(attributes)):
            raise ValueError('public-course preview lacks its original complete three-column source')
        matches = [index for index, option in enumerate(request['options'])
            if abs(option['box'][0] + option['box'][2] / 2 - selected[0]['box'][0] - selected[0]['box'][2] / 2) <= 20]
        if len(matches) != 1:
            raise ValueError('public-course SELECT is not bound to one attribute')
        index = matches[0]
        if expected_selection is not None and attributes[index] != expected_selection:
            raise ValueError('public-course selected course differs from its original physical column')
        # The root transaction already observed all three options. A preview
        # executes only this independently selected, fully named course;
        # faded neighbours do not supply its identity or authorise its input.
        target = _selected_caption(rows, attributes[index], index)
        option = request['options'][index]
        region = (190, 30, 90, 75)
        signed = [row for row in rows if row['score'] >= .95 and re.fullmatch(r'[+-]\d{1,2}', row['text'])
            and 190 <= row['box'][0] and 30 <= row['box'][1]
            and row['box'][0]+row['box'][2] <= 280 and row['box'][1]+row['box'][3] <= 105]
        if len(signed) > 1 or signed and signed[0]['text'].startswith('+'):
            raise ValueError('public-course selected signed quote is ambiguous or contradictory')
        try:
            quote = _row(rows, r'-\d{1,2}', region)
        except ValueError:
            from live_decision.hif_recovery import _field
            quote = dict(_field(context, image, 'TrainingSelectedCost', region, r'-\d{1,2}'),
                source='independent_complete_selected_quote', source_rows=[row for row in rows
                    if 190 <= row['box'][0] < 280 and 30 <= row['box'][1] < 105])
        if int(quote['text'][1:]) != option['cost'] or resources['stamina'] < option['cost']:
            raise ValueError('public-course selected stamina quote changed or is unaffordable')
        try:
            badge = read_special_course(image, index, selected=True)
        except ValueError as error:
            if option['special'] is True and str(error) == 'public-course SP absence is not positively observed':
                raise TrainingBadgeNotObserved(dict(resources, day=day, attribute=attributes[index],
                    quoted_cost=option['cost'], quote=quote, selected=selected[0],
                    expected_special=True, original_option=option)) from error
            raise
        if badge['special'] is not option['special'] or option['attribute'] != attributes[index]:
            raise ValueError('public-course selected attribute or SP changed from its original options')
        return dict(resources, scene='training_selected', day=day, selected_attribute=attributes[index],
            selected_special=option['special'], selected_cost=option['cost'], sp_evidence=badge, selected_quote_evidence=quote,
            targets={'confirm_training': target}, options=request['options'])
    if expected_selection is not None:
        raise ValueError('public-course original preview requires a complete current SELECT')
    captions = [_caption(rows, attribute, index) for index, attribute in enumerate(attributes)]
    options = []
    for index, attribute in enumerate(attributes):
        badge = read_special_course(image, index)
        cost, quote = read_training_cost(context, image, rows, index)
        if cost > resources['max_stamina']:
            raise ValueError('public-course quoted cost exceeds the supported stamina range')
        options.append(dict(attribute=attribute, special=badge['special'], cost=cost,
            available=resources['stamina'] >= cost, box=list(captions[index]), badge=badge, cost_evidence=quote))
    return dict(resources, scene='training', day=day, options=options,
        targets={'training:' + option['attribute']: tuple(option['box']) for option in options})


def _submitted_course(rows, request):
    from .production import _row
    day, attribute = request['day'], request['selected_attribute']
    if day not in (2, 5):
        raise ValueError('public-course result has no original training-day request')
    _row(rows, re.escape(attribute + '.公開レッスン'), (15, 15, 195, 70))
    present = _sp_header_rows(rows)
    special = [row for row in present if row['score'] >= .95]
    if (present or request['selected_special']) and not special:
        raise ValueError('public-course result SP field is unobserved')
    if bool(special) is not request['selected_special'] or len(special) > 1:
        raise ValueError('public-course result SP type differs from the selected course')
    return dict(read_route_resources(rows), day=day, attribute=attribute, special=request['selected_special'])


def read_training_result(rows, request):
    from .production import _row
    current = _submitted_course(rows, request)
    # Result rows keep Vo/Da/Vi order. The selected Da gain in 180 is the
    # second line at y1044; selected attribute does not mean the first line.
    gain = _row(rows, current['attribute'] + r'[.]が\d{1,3}上昇した!', (190, 970, 420, 160))
    return dict(current, scene='training_result', result_target=tuple(gain['box']),
        result_text=gain['text'], targets={})


def read_training_after_submission(rows, request):
    if any(row['score'] >= .95 and '上昇' in row['text'] and 970 <= row['box'][1] <= 1130 for row in rows):
        return read_training_result(rows, request)
    # Header + original SP type + actual resources prove only an in-flight
    # course surface, never its result or any free reward.
    return dict(_submitted_course(rows, request), scene='training_animation', targets={})


def training_effect(host):
    from .ledger import LedgerEffect
    state = host.ledger.state
    request = (host.handoff or {}).get('nonbattle_context', {}).get('training_choice', {})
    options = [option for option in request.get('options', []) if option['attribute'] == request.get('selected_attribute')]
    if (state.day not in (2, 5) or state.phase != 'training_selected' or len(options) != 1
            or request.get('schema_version') != 1 or request.get('run_id') != state.run_id
            or request.get('game_pid') != host.game_pid or request.get('day') != state.day
            or request.get('status') != 'active' or request.get('source_stamina') != state.stamina
            or request.get('source_currency') != state.currency or request.get('selected_cost') != options[0]['cost']
            or request.get('selected_special') is not options[0]['special']
            or type(options[0]['cost']) is not int or not 0 <= options[0]['cost'] <= state.stamina):
        raise ValueError('public-course execution has no exact original affordable stamina quote')
    return LedgerEffect(f'{state.run_id}:{state.revision}:execute-public-course', 'confirm_training',
        stamina_delta=-options[0]['cost'], phase='training_result')
