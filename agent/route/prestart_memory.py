"""Read this cultivation's HIF qualifier memory and navigate its selection list."""
import re
import unicodedata
from pathlib import Path
from functools import lru_cache

import cv2
from utils.runtime_paths import resolve_runtime_path

from .nonbattle_observation import _within


@lru_cache(maxsize=4)
def _reference(name):
    path = Path('resource/base/image/produce') / (name + '.png')
    image = cv2.imread(str(resolve_runtime_path(Path('assets') / path, path)))
    if image is None:
        raise ValueError('HIF preparation image reference is unavailable: ' + name)
    return image


def _match(image, name, region):
    x, y, width, height = region
    sample = _reference(name)
    scores = cv2.matchTemplate(image[y:y + height, x:x + width], sample, cv2.TM_CCOEFF_NORMED)
    count, _ = cv2.connectedComponents((scores >= .8).astype('uint8'))
    if count == 1:
        return None
    if count != 2:
        raise ValueError('HIF preparation image matches multiple candidates: ' + name)
    _, _, _, (left, top) = cv2.minMaxLoc(scores)
    return x + left, y + top, sample.shape[1], sample.shape[0]


def _present(rows, pattern, region):
    return any(row['score'] >= .95 and _within(row['box'], region)
               and re.fullmatch(pattern, row['text']) for row in rows)


def role_page(rows):
    return (any(re.fullmatch(r'1[.．]アイドル選択', row['text'])
                and _within(row['box'], (10, 12, 235, 69)) for row in rows)
            or _present(rows, '選抜試験メモリー', (426, 197, 286, 67)))


def memory_sort_page(rows):
    return any('ソート' in row['text'] and 'フィルタ' in row['text'] for row in rows) or (
        any(row['text'] == '入手' and row['box'][1] < 510 for row in rows)
        and any(row['text'] == '決定' and row['box'][1] > 1000 for row in rows))


def memory_list_page(rows):
    if memory_sort_page(rows):
        return True
    header = any(row['text'] in ('Pアイテム', 'メモリー') and row['box'][1] < 300 for row in rows)
    ok = any(row['text'] == 'OK' and _within(row['box'], (255, 1025, 207, 113)) for row in rows)
    sort = any(re.search('入手|スター|タグ', row['text']) and _within(row['box'], (437, 814, 183, 94))
               for row in rows)
    # The sort control and OK identify the list even when a header went unread.
    # A closed/background dialog with OK alone is insufficient.
    return (header and (ok or sort)) or (sort and ok)


def read_idol(context, image, rows):
    from live_decision.hif_profiles import notebook_hif_identity

    from .production import _row, _text

    if getattr(image, 'shape', ()) != (1280, 720, 3) or not role_page(rows):
        raise ValueError('HIF idol selection page is not visible')
    actor = _text(rows, r'[一-龯々ぁ-ゖァ-ヺー]+', (470, 142, 249, 80))
    title = _text(rows, r'\[.+\]', (355, 104, 365, 55))
    profile = notebook_hif_identity(title[1:-1], actor)
    try:
        next_box = tuple(_row(rows, '次へ', (310, 1045, 155, 85))['box'])
    except ValueError:
        from .home import _unique_home_template
        control, _ = _unique_home_template(image, 'next.png', (210, 1020, 300, 120), .7)
        next_box = control['box']
    return dict(phase='start_idol', target_kind='start_selected_idol_next',
                target=next_box, identity=(profile.identity, profile.plan))


# Vo, Da and Vi are on the confirmation page's middle line. On the list they
# appear in the selected memory's header, between star power and stamina.
_CONFIRM_COLUMNS = ((345, 462), (470, 580), (585, 705))
_LIST_COLUMNS = ((210, 337), (341, 473), (479, 607))
_LIST_LIMIT = 12


def _number(text):
    text = unicodedata.normalize('NFKC', text).replace(',', '').replace(' ', '')
    digits = re.fullmatch(r'\D*(\d{3,4})\D*', text)
    return int(digits[1]) if digits and '%' not in text else None


def _read_confirm_fields(context, image, rows):
    top, bottom = 475, 510
    values = []
    for left, right in _CONFIRM_COLUMNS:
        candidates = [value for row in rows
            if left <= row['box'][0] + row['box'][2] / 2 < right
            and top <= row['box'][1] + row['box'][3] / 2 < bottom
            and (value := _number(row['text'])) is not None]
        value = candidates[0] if len(candidates) == 1 else None
        if context is not None:
            # Whole-screen OCR may include the attribute icon or truncate the
            # final digit. Read the entire number up to the next column.
            region = image[top:bottom, left:right]
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            icon = ((hsv[:, :48, 1] >= 120) & (hsv[:, :48, 2] >= 150)).sum(axis=0)
            solid = (icon >= region.shape[0] // 4).nonzero()[0]
            if len(solid) >= 8:
                region = region[:, int(solid[-1]) + 1:]
                zoom = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                node = 'HifStartMemoryNumber'
                detail = context.run_recognition(node, zoom,
                    pipeline_override={node: dict(recognition='OCR', only_rec=True, expected='.+')})
                hits = [_number(row.text) for row in detail.filtered_results] if detail and detail.hit else []
                if len(hits) == 1 and hits[0] is not None:
                    value = hits[0]
        values.append(value)
    return tuple(values)


def read_confirm_memory(context, image, rows):
    """Return the three observed numbers, leaving an incomplete frame unread."""
    values = _read_confirm_fields(context, image, rows)
    return values if None not in values else None


def _digit_difference(expected, value):
    if value is None:
        return None
    a, b = str(expected), str(value)
    if len(a) != len(b):
        return None
    return [(x, y) for x, y in zip(a, b) if x != y]


_CONFUSED_DIGITS = {'3': '5', '5': '3', '6': '8', '8': '6', '0': '8', '1': '7', '7': '1'}


def _recognizes_digits(diff):
    return diff == [] or (diff is not None and len(diff) == 1 and diff[0][1] in _CONFUSED_DIGITS.get(diff[0][0], ''))


def memory_stats_match(expected, values, *, confirm=False):
    # Upstream allows a single commonly confused glyph per field. On the
    # confirmation page only, two verified fields can resolve one other glyph.
    uncertain = []
    for want, value in zip(expected, values, strict=True):
        diff = _digit_difference(want, value)
        if diff is None:
            return False
        if not _recognizes_digits(diff):
            uncertain.append(diff)
    return not uncertain or (confirm and len(uncertain) == 1 and len(uncertain[0]) == 1)


def confirm_memory_matches(context, image, rows, expected):
    values = read_confirm_memory(context, image, rows)
    return None if values is None else memory_stats_match(expected, values, confirm=True)


def _list_fields_match(expected, rows):
    for want, (left, right) in zip(expected, _LIST_COLUMNS, strict=True):
        tokens = [unicodedata.normalize('NFKC', row['text']).replace(',', '').replace(' ', '')
                  for row in rows if left <= row['box'][0] + row['box'][2] / 2 < right
                  and 126 <= row['box'][1] + row['box'][3] / 2 < 165 and '%' not in row['text']]
        # A stat's icon is sometimes OCR'd as a digit or a circled number.
        # Search within that column only, excluding the adjacent percentages.
        size = len(str(want))
        if not any(_recognizes_digits(_digit_difference(want, part))
                   for token in tokens for part in re.findall(rf'(?=(\d{{{size}}}))', token)):
            return False
    return True


def _selected_list_matches(context, image, rows, expected):
    if _list_fields_match(expected, rows):
        return True
    if context is None:
        return False
    # The upstream reader OCRs the selected memory's header (720px viewport).
    top = 100
    node = 'HifStartMemoryListHeader'
    header = image[top:220].copy()
    hit = context.run_recognition(node, header,
        pipeline_override={node: dict(recognition='OCR', only_rec=False, expected='.+')})
    local_rows = []
    for row in hit.filtered_results if hit and hit.hit else ():
        x, y, width, height = row.box
        local_rows.append(dict(text=row.text, box=(x, y + top, width, height)))
    return _list_fields_match(expected, local_rows)


def _label_or_point(rows, label, region, point):
    for row in rows:
        if label in row['text'] and _within(row['box'], region):
            return tuple(row['box'])
    return (*point, 1, 1)


def memory_change_target(rows):
    return _label_or_point(rows, '変更', (570, 541, 139, 71), (636, 577))


def read_memory_list(context, image, rows, *, expected, stage, page=0, cell=0):
    """Return one upstream action for the cursor following the last delivered input."""
    if not memory_list_page(rows):
        raise ValueError('HIF qualifier memory list is not visible')
    if not (0 <= page < _LIST_LIMIT and 0 <= cell < 9):
        raise ValueError('invalid HIF memory list cursor')
    matched = False
    if stage == 'inspect_cell':
        matched = _selected_list_matches(context, image, rows, expected)
        if matched or (page, cell) == (_LIST_LIMIT - 1, 8):
            stage = 'close'
        elif cell == 8:
            stage = 'scroll'
        else:
            stage, cell = 'select_cell', cell + 1
    if stage == 'open_sort':
        target = (527, 860, 1, 1)
    elif stage == 'choose_sort':
        target = _label_or_point(rows, '入手', (0, 0, 720, 780), (413, 273))
    elif stage == 'apply_sort':
        target = _label_or_point(rows, '決定', (0, 990, 720, 270), (500, 1160))
    elif stage == 'select_cell':
        target = ((140, 360, 580)[cell % 3], (487, 627, 767)[cell // 3], 1, 1)
    elif stage == 'scroll':
        target = (360, 620, 1, 281)
    elif stage == 'close':
        target = _label_or_point(rows, 'OK', (255, 1025, 207, 113), (360, 1085))
    else:
        raise ValueError('unknown HIF memory list action: ' + stage)
    kind = 'start_memory_ok' if stage == 'close' else 'start_memory_' + stage
    result = dict(phase='start_memory_' + stage, target_kind=kind, target=target,
        memory_page=page, memory_cell=cell)
    if stage == 'close':
        result['memory_match'] = matched
    return result
