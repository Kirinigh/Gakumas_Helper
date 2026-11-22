"""Recognize the AP drink route from the HIF game's insufficient-AP dialog."""

import re

from .new_run import start_record

AP_REFILL_INSUFFICIENT = 'start_refill_insufficient'
AP_REFILL_APPLY = 'start_apply_ap_drink'
AP_REFILL_CANCEL = 'start_cancel_ap_recovery'
AP_REFILL_CANCELLED = 'start_refill_cancelled'
_RETURN = ('start_home', 'start_hif_scenario', 'start_idol', 'start_confirmation',
    'start_memory_mismatch', AP_REFILL_INSUFFICIENT)
AP_REFILL_CONTRACTS = {
    'start_open_ap_recovery': (AP_REFILL_INSUFFICIENT, ('start_refill_list',)),
    'start_select_ap_drink': ('start_refill_list', ('start_refill_confirm', 'start_refill_empty_count')),
    'start_add_ap_drink': ('start_refill_empty_count', ('start_refill_confirm',)),
    AP_REFILL_APPLY: ('start_refill_confirm', ('start_refill_close', *_RETURN)),
    'start_close_ap_recovery': ('start_refill_close', _RETURN),
    AP_REFILL_CANCEL: ('start_refill_cancel', (AP_REFILL_CANCELLED,)),
}
AP_REFILL_SCENES = frozenset({source for source, _ in AP_REFILL_CONTRACTS.values()} | {AP_REFILL_CANCELLED})


def allows_ap_refill(host):
    start = start_record(host.handoff) if host is not None else None
    return bool(start and start.get('status') == 'available'
        and (start.get('creation_task') or {}).get('entry') == 'Produce'
        and start.get('additional_start') is None)


def ap_refill_prompt(host):
    return (host.handoff.get('start_preparation') or {}).get('ap_refill_prompt')


def ap_drinks_enabled(context):
    """Honor the existing Produce task's 使用体力药 override."""
    read_node = getattr(context, 'get_node_data', None)
    if not callable(read_node):
        return False
    node = read_node('ProduceLackAP') or {}
    targets = node.get('next') or ()
    if isinstance(targets, str):
        targets = (targets,)
    return any((target.get('name') if isinstance(target, dict) else target) == 'ProduceLackAPRestore'
               for target in targets)


def _center(row):
    x, y, w, h = row['box']
    return x + w / 2, y + h / 2


def _button(rows, text, *, min_y=0):
    hits = [row for row in rows if row['text'] == text and _center(row)[1] >= min_y]
    if not hits:
        raise ValueError('AP recovery control is missing: ' + text)
    return tuple(max(hits, key=lambda row: _center(row)[1])['box'])


def _use_count(rows):
    for row in reversed(sorted(rows, key=lambda row: _center(row)[1])):
        match = re.fullmatch(r'使用数(?:\d+[→➝])?(\d+)', row['text'])
        if match:
            return int(match[1])
        if row['text'] != '使用数':
            continue
        x, y = _center(row)
        numbers = [(xx, int(other['text'])) for other in rows if other['text'].isdecimal()
                   for xx, yy in [_center(other)] if xx > x and abs(yy - y) <= 15]
        return max(numbers)[1] if numbers else None
    return None


def read_ap_refill_page(rows, host, context):
    """Only the AP dialog and its APドリンク controls are eligible."""
    if not allows_ap_refill(host):
        return None
    start = start_record(host.handoff)
    submitted = start.get('ap_drink_request') is not None
    words = [row['text'] for row in rows]
    # The confirmation can overlay the recovery list.
    if any('APドリンクを消費' in word for word in words):
        count = _use_count(rows)
        if (not ap_drinks_enabled(context) or any('ジュエル' in word for word in words)
                or count not in (0, 1)):
            raise ValueError('AP recovery confirmation is not for one AP drink')
        if count == 0:
            # Hiro's fixed plus control for the AP drink quantity dialog.
            return dict(phase='start_refill_empty_count', target=(579, 1028, 10, 10))
        return dict(phase='start_refill_confirm', target=_button(rows, '回復する', min_y=1000))
    shortage = [int(match[1]) for word in words if (match := re.search(r'[AÄ]Pが(\d+)不足', word))]
    if shortage:
        if submitted:
            # A second shortage is an observed failure after the one permitted attempt.
            return dict(phase=AP_REFILL_INSUFFICIENT, target=None)
        if not ap_drinks_enabled(context):
            return dict(phase='start_refill_cancel', target=_button(rows, 'キャンセル'))
        if any('ジュエル' in word for word in words) or len(set(shortage)) != 1 or not 0 < shortage[0] <= 20:
            raise ValueError('AP recovery shortage does not identify a one-drink HIF start')
        return dict(phase=AP_REFILL_INSUFFICIENT, target=_button(rows, '回復する'), ap_shortfall=shortage[0])
    if 'AP回復' not in words and not (submitted and any('APを回復しました' in word for word in words)):
        return None
    if submitted:
        return dict(phase='start_refill_close', target=_button(rows, '閉じる'))
    if not ap_drinks_enabled(context):
        raise ValueError('AP recovery is disabled by the Produce task')
    for label in rows:
        if not label['text'].startswith('APドリンク'):
            continue
        x, y = _center(label)
        same_row = [row for row in rows if abs(_center(row)[1] - y) <= 70]
        if any('ジュエル' in row['text'] for row in same_row):
            continue
        use = [row for row in same_row if row['text'] == '使う' and _center(row)[0] > x]
        if len(use) == 1:
            return dict(phase='start_refill_list', target=tuple(use[0]['box']))
    raise ValueError('AP recovery list has no AP drink use control')


def ap_refill_target(page):
    if page.get('target') is not None:
        return next((kind for kind, (source, _) in AP_REFILL_CONTRACTS.items()
                     if source == page['phase']), None)
    return None
