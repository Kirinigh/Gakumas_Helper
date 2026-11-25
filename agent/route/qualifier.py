"""Independent HIF qualifier preparation through the native Produce callback."""
from __future__ import annotations

import os
import re
import json
import time
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from utils.runtime_paths import resolve_runtime_path

from .memory_picker import MemoryPicker, picker_open
from .qualifier_result import finish_qualifier, is_qualifier_result, qualifier_certificate, qualifier_exam_number, read_qualifier_result
from .qualifier_support import FormationValue, SupportGallery, read_own_support, write_own_support


@dataclass(frozen=True)
class Screen:
    image: object
    rows: list
    at: float


def label(frame, text, *, top=0, bottom=1280):
    boxes = [row['box'] for row in frame.rows if row['score'] >= .85 and top <= row['box'][1] < bottom
        and re.fullmatch(text, row['text'].replace('サボート', 'サポート'))]
    return boxes[0] if len(boxes) == 1 else None


class QualifierUi:
    def __init__(self, context, authorised, *, clock=time.monotonic, sleep=time.sleep):
        self.context, self.authorised, self.clock, self.sleep = context, authorised, clock, sleep
        self.frame = None
        self.attempts = []
        self.memory_inventory = None
        self.support_changes = []
        self.rental_receipt = None

    def capture(self):
        from .production import _current_rows

        self.authorised()
        at = self.clock()
        job = self.context.tasker.controller.post_screencap().wait()
        if job.succeeded is not True:
            raise ValueError('qualifier capture failed')
        image = job.get()
        if image.shape != (1280, 720, 3):
            raise ValueError('qualifier viewport is not available')
        self.frame = Screen(image, _current_rows(self.context, image, date_probe=False, settings_probe=False), at)
        return self.frame

    def wait(self, predicate, *, timeout=12):
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            frame = self.capture()
            if predicate(frame):
                return frame
            self.sleep(.15)
        return None

    def need(self, predicate, page):
        frame = self.wait(predicate)
        if frame is None:
            raise ValueError('qualifier page not observed: ' + page)
        return frame

    def _act(self, frame, action):
        self.authorised()
        if frame is not self.frame or not 0 <= self.clock() - frame.at <= 1:
            raise ValueError('qualifier input requires a fresh current capture')
        self.frame = None
        entry = 'ProduceHifAdvanceClick'
        event = dict(action=action, sdk_success=None)
        self.attempts.append(event)
        result = self.context.run_action(entry, pipeline_override={entry: action})
        if getattr(result, 'success', None) is not True:
            raise ValueError('qualifier Maa input result is unknown')
        event['sdk_success'] = True

    def click(self, frame, box, *, settle=.4):
        self._act(frame, {'action': 'Click', 'target': list(box)})
        self.sleep(settle)

    def swipe(self, frame, begin, end):
        self._act(frame, dict(action='Swipe', begin=[*begin, 1, 1], end=[*end, 1, 1], duration=1000))
        self.sleep(.8)
        return self.capture()


def _frame_number(frame):
    numbers = {int(match[1]) for row in frame.rows if 890 <= row['box'][1] <= 1010
        and row['box'][0] < 250 and (match := re.fullmatch(r'(\d+)/20', row['text']))}
    return next(iter(numbers)) if len(numbers) == 1 else None


def _select_set(ui, frame, number, *, changed, arrows):
    for _ in range(20):
        actual = _frame_number(frame)
        if actual is None:
            raise ValueError('qualifier formation set number is unreadable')
        if actual == number:
            return frame
        x, y = arrows[int(number > actual)]
        ui.click(frame, (x, y, 1, 1), settle=.5)
        frame = ui.need(lambda f: changed(f) and _frame_number(f) not in (None, actual), 'formation set change')
    raise ValueError('configured qualifier formation set was not reached')


def _memory_menu(frame):
    return (label(frame, 'メモリー編成', top=1040) is not None and label(frame, 'サポート編成', top=1040) is not None
        and label(frame, 'おまかせ', top=890, bottom=1010) is not None and _frame_number(frame) is not None)


def _hif_menu(frame):
    return any(row['box'][1] < 90 and row['box'][0] < 320
        and re.search(r'H\.?I\.?F|FESTIVAL', row['text'], re.I) for row in frame.rows)


def scan_qualifier_memories(ui, number, catalog, desired=()):
    home = ui.need(lambda f: label(f, 'アイドル', top=1060) is not None and
        label(f, 'プロデュース', top=920, bottom=1070) is not None, 'home')
    ui.click(home, label(home, 'アイドル', top=1060))
    idol = ui.need(lambda f: label(f, 'プロデュース編成') is not None, 'idol menu')
    ui.click(idol, label(idol, 'プロデュース編成'))
    frame = ui.need(_memory_menu, 'formation menu')
    if not _hif_menu(frame):
        box = label(frame, 'プロデュース切替', bottom=120)
        if box is None:
            raise ValueError('formation mode selector is missing')
        ui.click(frame, box)
        mode = ui.need(lambda f: label(f, '決定', top=1080) is not None and
            any(row['box'][1] > 680 and 'FESTIVAL' in row['text'].upper() for row in f.rows), 'formation mode choices')
        ui.click(mode, (360, 796, 1, 1))
        confirm = ui.need(lambda f: label(f, '決定', top=1080) is not None, 'formation mode confirm')
        ui.click(confirm, label(confirm, '決定', top=1080))
        frame = ui.need(lambda f: _memory_menu(f) and _hif_menu(f), 'HIF formation menu')
    ui.click(frame, label(frame, 'メモリー編成', top=1040))
    frame = ui.need(_memory_menu, 'memory formation tab')
    frame = _select_set(ui, frame, number, changed=_memory_menu, arrows=((40, 635), (680, 635)))
    ui.click(frame, (197, 730, 1, 1))
    frame = ui.need(lambda f: picker_open(f.rows), 'memory picker')
    inventory = MemoryPicker(ui, catalog).scan(frame)
    ui.memory_inventory = inventory
    frame = ui.need(lambda f: picker_open(f.rows) and label(f, 'キャンセル', top=1080) is not None, 'memory picker cancel')
    ui.click(frame, label(frame, 'キャンセル', top=1080))
    frame = ui.need(_memory_menu, 'closed memory picker')
    if desired:
        frame = _write_memories(ui, frame, catalog, inventory, desired)
    frame = frame if label(frame, 'ホーム', top=1070) is not None else ui.need(
        lambda f: label(f, 'ホーム', top=1070) is not None, 'return to home')
    ui.click(frame, label(frame, 'ホーム', top=1070))
    return inventory


def _memory_slot_id(frame, slot, reader):
    return reader.identify(frame.image, ((75, 222, 370, 518)[slot], 345, 130, 130), plan=None).card_id


def _write_memories(ui, frame, catalog, inventory, desired):
    from .nonbattle_inventory import load_flat_inventory_readers

    changes = []
    for slot, card_id in enumerate(desired):
        matches = [entry for entry in inventory['entries'] if (entry.get('card') or {}).get('card_id') == card_id]
        if len(matches) != 1 or not inventory['complete']:
            raise ValueError('configured memory is unknown or ambiguous in the scanned inventory')
        changes.append((slot, card_id, matches[0]))
    reader, _ = load_flat_inventory_readers()
    for slot, card_id, entry in changes:
        if _memory_slot_id(frame, slot, reader) == card_id:
            continue
        x, y = ((197, 730), (522, 730), (197, 861), (522, 861))[slot]
        ui.click(frame, (x, y, 1, 1))
        picker = ui.need(lambda f: picker_open(f.rows), 'editable memory slot')
        located = MemoryPicker(ui, catalog).scan(picker, stop_at=(entry['row'], entry['column']))
        current = ui.frame
        if (located.get('target', {}).get('card') or {}).get('card_id') != card_id:
            cancel = label(current, 'キャンセル', top=1080)
            if cancel is not None and picker_open(current.rows):
                ui.click(current, cancel)
            raise ValueError('the memory at the selected physical slot differs from the scanned card')
        commit = label(current, '決定', top=1080)
        if commit is None or not picker_open(current.rows):
            raise ValueError('memory picker confirmation is not visible')
        ui.click(current, commit)
        frame = ui.need(_memory_menu, 'memory write return')
        if _memory_slot_id(frame, slot, reader) != card_id:
            raise ValueError('the configured memory was not observed in its formation slot')
    return frame


def _support_step(frame):
    return label(frame, r'(?:2[.．])?サポート選択', bottom=105) is not None


def _rental_rows(frame, *, context=None):
    found = []
    for row in frame.rows:
        if not re.fullmatch(r'PLv\d+', row['text'], re.I):
            continue
        _, y, _, h = row['box']
        y = y + h // 2 - 13
        if not 555 <= y <= 872:
            continue
        icon = cv2.cvtColor(frame.image[y+80:y+98, 227:245], cv2.COLOR_BGR2HSV)
        color = icon[:, :, 0][(icon[:, :, 1] >= 120) & (icon[:, :, 2] >= 150)]
        if len(color) < 30:
            continue
        hue = np.median(color)
        attr = next((name for name, lo, hi in (('Vi', 5, 35), ('HP', 40, 85), ('Da', 90, 120), ('Vo', 150, 180))
            if lo <= hue <= hi), None)
        level, breaks = support_level(frame, 44, y, context=context)
        if attr is not None and level is not None:
            found.append((attr, level, breaks, (360, y + 30, 1, 1)))
    return found


def support_level(frame, x, y, *, context=None):
    from .production import _text

    level_box, number = (x, y+73, 51, 34), r'[1-9]\d?'
    try:
        try:
            level = int(_text(frame.rows, number, level_box))
        except ValueError:
            if context is None:
                return None, None
            from live_decision.hif_recovery import _field

            level = int(_field(context, frame.image, 'HifQualifierSupportLevel', level_box, number)['text'])
    except ValueError:
        return None, None
    if not 1 <= level <= 70:
        return None, None
    left = x + (42 if level < 10 else 61)
    lit = []
    for i in range(4):
        center = left + 25 * i
        badge = cv2.cvtColor(frame.image[y+86:y+104, center-9:center+9], cv2.COLOR_BGR2HSV)
        lit.append(float(((badge[:, :, 2] >= 215) & (badge[:, :, 1] <= 70)).mean()) >= .4)
    return level, sum(lit) if lit == sorted(lit, reverse=True) else None


def _rental_selected(frame, candidate):
    y = candidate[3][1] - 30
    area = cv2.cvtColor(frame.image[max(0, y-55):y-20], cv2.COLOR_BGR2HSV)
    orange = (area[:, :, 0] >= 8) & (area[:, :, 0] <= 22) & (area[:, :, 1] >= 150) & (area[:, :, 2] >= 200)
    return int(orange[:, 15:50].sum() + orange[:, 670:705].sum()) >= 200


def _rental_pager(frame):
    values = {int(match[1]) for row in frame.rows if 500 <= row['box'][1] <= 560
        and (match := re.fullmatch(r'([12])/2', row['text']))}
    return next(iter(values)) if len(values) == 1 else None


def _rental_art(frame, candidate):
    y = candidate[3][1] - 30
    return cv2.resize(frame.image[y:y+70, 44:245], (42, 14))


def _rental_player(frame, candidate):
    y = candidate[3][1] - 30 + 13
    return next((row['text'] for row in frame.rows if row['box'][0] < 300
        and -50 <= row['box'][1] + row['box'][3]//2 - y <= -25), None)


def _matching_rental_rows(ui, frame, best):
    _, ranked, art, _, player = best
    return [row for row in _rental_rows(frame, context=ui.context) if row[:3] == ranked[:3]
        and (player is None or _rental_player(frame, row) == player)
        and cv2.mean(cv2.absdiff(_rental_art(frame, row), art))[0] < 8]


def _rental_moved(previous, current):
    return cv2.mean(cv2.absdiff(previous.image[555:990], current.image[555:990]))[0] >= 2


def _rental_top(ui, frame):
    for _ in range(30):
        new = ui.swipe(frame, (360, 640), (360, 1000))
        if not _rental_moved(frame, new):
            return new
        frame = new
    return frame


def _rank_rental_page(ui, frame, attr, *, gallery, scorer, own, excluded):
    best = None
    for _ in range(10):
        for candidate in _rental_rows(frame, context=ui.context):
            y = candidate[3][1]-30
            card = gallery.identify(frame.image, 44, y) if gallery else None
            if card and (card['id'] in excluded or card['id'] in {c.get('id') for c in own}):
                continue
            if scorer is None:
                if candidate[0] != attr:
                    continue
                value = 0.
            else:
                if card is None or card['attribute'] != candidate[0]:
                    continue
                value = scorer.score(own, dict(card, level=candidate[1], breaks=candidate[2]))
                if value is None:
                    continue
            key = (value, candidate[1], candidate[2] if candidate[2] is not None else -1)
            if best is None or key > best[0]:
                best = (key, candidate, _rental_art(frame, candidate), card, _rental_player(frame, candidate))
        new = ui.swipe(frame, (360, 1000), (360, 640))
        if not _rental_moved(frame, new):
            return new, best
        frame = new
    return frame, best


def _pick_rental(ui, frame, attr, *, gallery, scorer, own, excluded):
    confirm = label(frame, '決定', top=1080)
    if confirm is None or label(frame, 'プラン適正', top=950) is None:
        raise ValueError('qualifier rental list was not observed')
    checkbox = frame.image[1006:1027, 545:566]
    hsv = cv2.cvtColor(checkbox, cv2.COLOR_BGR2HSV)
    if float(((hsv[:, :, 2] >= 150) & (hsv[:, :, 2] <= 215) & (hsv[:, :, 1] < 40)).mean()) < .4:
        ui.click(frame, (551, 1016, 1, 1))
        frame = ui.need(lambda f: label(f, 'プラン適正', top=950) is not None, 'plan-compatible rentals')
    frame = _rental_top(ui, frame)
    first = frame
    best_page = _rental_pager(first)
    ranking = dict(gallery=gallery, scorer=scorer, own=own, excluded=excluded)
    frame, best = _rank_rental_page(ui, first, attr, **ranking)
    if best_page == 1:
        ui.click(frame, (485, 533, 1, 1))
        second = ui.wait(lambda f: _rental_pager(f) == 2)
        if second is not None:
            second = _rental_top(ui, second)
            # Some presentations page only the upper details, so the list
            # may still be the same even when the pager moved.
            if _rental_moved(first, second):
                frame, second_best = _rank_rental_page(ui, second, attr, **ranking)
                if second_best is not None and (best is None or second_best[0] > best[0]):
                    best_page, best = 2, second_best
            else:
                frame = second
        else:
            frame = ui.capture()
    if best is None:
        return frame, None
    if best_page == 1 and _rental_pager(frame) == 2:
        ui.click(frame, (233, 533, 1, 1))
        frame = ui.need(lambda f: _rental_pager(f) == 1, 'earlier rental list')
    # Start at the bottom on the page just ranked, or the top on returning
    # to a prior page. Rescan toward both ends if the original row moved.
    descending = best_page == 1
    for _ in range(40):
        matches = _matching_rental_rows(ui, frame, best)
        if len(matches) == 1:
            ui.click(frame, matches[0][3])
            selected = ui.need(lambda f: label(f, '決定', top=1080) is not None
                and any(_rental_selected(f, row) for row in _matching_rental_rows(ui, f, best)), 'selected rental')
            ui.click(selected, label(selected, '決定', top=1080))
            support = ui.need(_support_step, 'rental return to support formation')
            identity = best[3]['id'] if best[3] else None
            return support, dict(card_id=identity, level=best[1][1], breaks=best[1][2],
                attribute=best[1][0], formation_value=best[0][0] if scorer else None,
                basis='formation_value' if scorer else 'attribute_level',
                mismatch=any('不一致' in row['text'] and row['box'][0] >= 480 and 600 <= row['box'][1] <= 900
                    for row in support.rows))
        top, bottom = (360, 640), (360, 1000)
        current = ui.swipe(frame, bottom if descending else top, top if descending else bottom)
        if not _rental_moved(frame, current):
            descending = not descending
        frame = current
    raise ValueError('the previously ranked rental is no longer visible')


def _open_rental(ui, frame):
    ui.click(frame, (588, 900, 1, 1))
    return ui.need(lambda f: label(f, 'プラン適正', top=950) is not None, 'support rental')


def rent_support(ui, frame, attr, *, gallery=None, scorer=None, own=()):
    excluded = set()
    ranking = dict(gallery=gallery, scorer=scorer, own=own, excluded=excluded)
    for _ in range(3):
        frame, receipt = _pick_rental(ui, frame, attr, **ranking)
        for _ in range(2):
            if receipt is not None:
                break
            before = _rental_top(ui, frame)
            ui.click(before, (400, 1071, 1, 1), settle=2.5)
            refreshed = ui.capture()
            if label(refreshed, 'プラン適正', top=950) is None or not _rental_moved(before, refreshed):
                frame = refreshed
                break
            frame, receipt = _pick_rental(ui, refreshed, attr, **ranking)
        if receipt is None:
            cancel = label(frame, 'キャンセル', top=1080)
            if cancel is not None:
                ui.click(frame, cancel)
            raise ValueError('no matching rental was observed')
        if not receipt['mismatch']:
            ui.rental_receipt = receipt
            return frame
        if receipt['card_id'] is None:
            break
        excluded.add(receipt['card_id'])
        frame = _open_rental(ui, frame)
    raise ValueError('the selected rental does not match the qualifier plan')


def prepare_qualifier(ui, plan, catalog, *, gallery=None, value=None, first=None):
    from .home import read_active_home

    frame = first or ui.need(lambda f: label(f, 'プロデュース', top=920, bottom=1070) is not None, 'produce home')
    read_active_home(frame.image, frame.rows, context=ui.context, free_start=True)
    inventory = scan_qualifier_memories(ui, plan['memory_set'], catalog, plan.get('memory_cards', ()))
    frame = ui.need(lambda f: label(f, 'プロデュース', top=920, bottom=1070) is not None, 'produce home')
    home = read_active_home(frame.image, frame.rows, context=ui.context, free_start=True)
    ui.click(frame, home['target'])
    scenario = ui.need(lambda f: label(f, '選抜試験', top=810, bottom=950) is not None and
        label(f, '本戦', top=810, bottom=950) is not None, 'HIF scenario')
    ui.click(scenario, label(scenario, '選抜試験', top=810, bottom=950))
    idol = ui.need(lambda f: label(f, r'1[.．]アイドル選択', bottom=100) is not None and
        label(f, '次へ', top=1000) is not None, 'qualifier idol')
    ui.click(idol, label(idol, '次へ', top=1000))
    support = ui.need(_support_step, 'support formation')
    support = _select_set(ui, support, plan['support_set'], changed=_support_step, arrows=((20, 833), (700, 833)))
    if plan.get('support_cards'):
        support = write_own_support(ui, support, gallery, plan['support_cards'], ui.support_changes)
    if value is not None or label(support, 'レンタルのみ.*', top=810, bottom=990) is not None:
        own = read_own_support(ui, support, gallery, stats=value is not None) if gallery else ()
        if value is not None:
            support = ui.need(_support_step, 'support formation before rental')
        rental = _open_rental(ui, support)
        support = rent_support(ui, rental, plan['attribute'], gallery=gallery, scorer=value, own=own)
    next_box = label(support, '次へ', top=1000)
    if next_box is None:
        raise ValueError('qualifier support next is not available')
    ui.click(support, next_box)
    def memory_step(current):
        return label(current, r'(?:3[.．])?メモリー選択', bottom=110) is not None

    memory = ui.need(memory_step, 'memory formation')
    memory = _select_set(ui, memory, plan['memory_set'], changed=memory_step, arrows=((20, 833), (700, 833)))
    next_box = label(memory, '次へ', top=1000)
    if next_box is None:
        raise ValueError('qualifier memory next is not available')
    ui.click(memory, next_box)
    quote = ui.need(lambda f: label(f, r'4[.．]開始確認', bottom=115) is not None, 'qualifier start confirmation')
    prices = {tuple(map(int, match.groups())) for row in quote.rows if 1035 <= row['box'][1] <= 1190
        and (match := re.fullmatch(r'(\d{1,3})\s*→\s*(\d{1,3})', row['text']))}
    price = next(iter(prices)) if len(prices) == 1 else None
    return dict(status='prepared', paid_start=False, memory_inventory=inventory,
        quoted_ap=dict(before=price[0], after=price[1], cost=price[0]-price[1])
        if price and price[0] >= price[1] else None)


def play_qualifier_exams(ui, frame, play_exam, *, report, finish=None):
    """Hiro's briefing → common exam → observed result → memory sequence."""
    results = report.setdefault('exams', [])
    context = ui.context
    # These are the existing Maa non-battle handlers; each returns control to
    # this loop to classify the new frame, just as Hiro's page dispatcher does.
    between_exams = ('ProduceChooseGetFlag', 'ProduceChooseStrengthenFlag',
                     'ProduceChooseDeleteFlag', 'ProduceRecognitionWorkOptions',
                     'ProduceChooseEventFlag', 'ProduceSkipChatFlag', 'ProduceSkip')
    last_result = None
    advanced_result = None
    stage = None
    awaiting = ui.clock()
    while not context.tasker.stopping:
        if qualifier_certificate(frame):
            report.update((finish or finish_qualifier)(ui, frame))
            return report

        if result := read_qualifier_result(frame):
            key = result['exam'], result['score'], result['passed']
            if key != last_result:
                results.append(result)
                last_result = key
                awaiting = ui.clock()
                if not result['passed']:
                    report['status'] = 'failed'
                    return report
            if key != advanced_result:
                if target := label(frame, '次へ', top=1030):
                    ui.click(frame, target)
                    advanced_result = key
                    stage = None
        elif is_qualifier_result(frame):
            pass  # Wait for the player row and the real score to become readable.
        elif qualifier_exam_number(frame, result=False) and label(frame, '合格条件', bottom=850):
            number = qualifier_exam_number(frame, result=False)
            if number != stage and (target := label(frame, r'次へ|試験(?:を)?開始(?:する)?', top=1080)):
                ui.click(frame, target)
                stage = number
                awaiting = ui.clock()
        elif getattr(context.run_recognition('ProduceRecognitionHealthFlag', frame.image), 'hit', False):
            def result_reader(image):
                from .production import _current_rows

                rows = _current_rows(context, image, date_probe=False, settings_probe=False)
                observed = read_qualifier_result(Screen(image, rows, ui.clock()))
                return observed if observed and (stage is None or observed['exam'] == stage) else None

            is_qualifier = stage is not None or any('突破ライン' in row['text'] for row in frame.rows
                                                    if row['box'][1] < 245)
            played = play_exam(context, result_reader=result_reader if is_qualifier else None,
                               authorize=ui.authorised)
            if not played:
                break
            awaiting = ui.clock()
        elif not any(re.search(r'購入|有償|ジュエル|マニー', row['text']) for row in frame.rows):
            for node in between_exams:
                if getattr(context.run_recognition(node, frame.image), 'hit', False):
                    ui.authorised()
                    job = context.run_task(node)
                    if getattr(job, 'succeeded', True):
                        awaiting = ui.clock()
                    break
            else:
                if target := label(frame, r'TAPで次へ|次へ|試験へ|レッスンへ', top=1070):
                    ui.click(frame, target)
                    awaiting = ui.clock()

        if ui.clock() - awaiting >= 30.:
            break
        ui.sleep(.2)
        frame = ui.capture()
    report['status'] = 'exam_scene_unresolved'
    return report


def start_qualifier_exams(ui, report):
    frame = ui.frame
    target = label(frame, r'プロデュース開始|選抜試験(?:を)?開始(?:する)?', top=1030)
    if report.get('quoted_ap') is None or target is None:
        raise ValueError('qualifier start button or AP quote unread')
    ui.click(frame, target, settle=1.)
    report['paid_start'] = True
    return ui.capture()


def run_native_qualifier(context, argv, options):
    """Bind only the real Produce task and the existing exclusive device lease."""
    from maa.context import Context
    from maa.custom_action import CustomAction
    from live_decision.catalog import SkillCardFeatureCatalog
    from live_decision.hif_capture_adapter import adapt_hif_context

    from .hiro_hif import finish_hiro_qualifier
    from .bootstrap import _endpoint, _product_paths, _instance_endpoint
    from .ownership import write_hif_record, locked_hif_record
    from .hiro_product import prepare_auto_qualifier
    from .hiro_formation import auto_formation_selected

    if (not isinstance(context, Context) or not isinstance(argv, CustomAction.RunArg)
            or argv.node_name != 'ProduceHifQualifierPrepareEntry' or argv.custom_action_name != 'ProduceHifQualifierPrepare'
            or argv.task_detail.entry != 'Produce'):
        raise ValueError('qualifier preparation requires the native Produce task')
    plan = {key: int(options[key]) for key in ('support_set', 'memory_set')}
    params = argv.custom_action_param
    params = json.loads(params) if isinstance(params, str) and params else params
    run_exams = isinstance(params, dict) and params.get('run_exams') is True
    plan['attribute'] = options['attribute']
    if any(not 1 <= plan[key] <= 20 for key in ('support_set', 'memory_set')) or plan['attribute'] not in ('Vo', 'Da', 'Vi'):
        raise ValueError('qualifier requires support/memory sets 1–20 and a valid attribute')
    memories = options.get('memory_cards', '')
    if memories and re.fullmatch(r'[1-9]\d*(?:,[1-9]\d*){0,3}', memories) is None:
        raise ValueError('qualifier memory plan needs up to four comma-separated card IDs')
    plan['memory_cards'] = memories.split(',') if memories else []
    if len(set(plan['memory_cards'])) != len(plan['memory_cards']):
        raise ValueError('select distinct physical memories when entering a card ID plan')
    supports = options.get('support_cards', '').strip()
    plan['support_cards'] = [entry.strip() for entry in supports.split(',')] if supports else []
    if (len(plan['support_cards']) > 5 or len(set(plan['support_cards'])) != len(plan['support_cards'])
            or any(re.fullmatch(r's_card-[\w-]+', card) is None for card in plan['support_cards'])):
        raise ValueError('qualifier needs up to five distinct support card IDs')
    gallery_path = options.get('support_gallery', '').strip()
    value_path = options.get('support_effects', '').strip()
    if (plan['support_cards'] or value_path) and not gallery_path:
        raise ValueError('support formation editing and scoring require the local support art gallery')
    gallery = SupportGallery(os.path.expandvars(gallery_path)) if gallery_path else None
    if plan['support_cards'] and not set(plan['support_cards']).issubset(card['id'] for card in gallery.cards):
        raise ValueError('the local support gallery lacks configured card art')
    value = FormationValue.load(os.path.expandvars(value_path), plan['attribute']) if value_path else None
    path, runtime, maa = _product_paths()
    task_id = argv.task_detail.task_id
    endpoint = _instance_endpoint(_endpoint(context, expected_maa=maa))

    def authorised():
        current = context.get_task_job()
        if current.job_id != task_id or not current.status.running or context.tasker.stopping:
            raise ValueError('qualifier native task is no longer active')
        if _instance_endpoint(_endpoint(context, expected_maa=maa)) != endpoint:
            raise ValueError('qualifier input endpoint changed')

    with locked_hif_record(path):
        if path.is_file():
            record = json.loads(path.read_bytes())
            if (record['ownership']['owner'] == 'task275' or record['transition']['pending'] is not None
                    or (record.get('ledger') or {}).get('pending') is not None):
                raise ValueError('qualifier cannot take over an unresolved main cultivation')
        authorised()
        bound = adapt_hif_context(context)
        if bound.tasker.controller.set_screenshot_target_long_side(1280) is not True:
            raise ValueError('qualifier capture shape could not be configured')
        data = resolve_runtime_path(Path('assets/data/skill_cards.csv'), Path('data/skill_cards.csv'))
        ui = QualifierUi(bound, authorised)
        report = dict(status='unresolved', paid_start=False)
        try:
            frame = ui.need(lambda f: qualifier_certificate(f) or label(f, 'プロデュース', top=920, bottom=1070)
                or (run_exams and (qualifier_exam_number(f, result=True)
                    or (qualifier_exam_number(f, result=False) and label(f, '合格条件', bottom=850)))),
                'qualifier entry or settlement')
            if qualifier_certificate(frame):
                report = dict(finish_hiro_qualifier(ui, frame), paid_start=False)
            else:
                if label(frame, 'プロデュース', top=920, bottom=1070):
                    if auto_formation_selected(context):
                        report = prepare_auto_qualifier(context, ui, options, report)
                    else:
                        report = prepare_qualifier(ui, plan, SkillCardFeatureCatalog.load(data),
                            gallery=gallery, value=value, first=frame)
                    if run_exams:
                        frame = start_qualifier_exams(ui, report)
                if run_exams:
                    from custom.action.produce import ProduceCardsAuto

                    play_qualifier_exams(ui, frame, ProduceCardsAuto()._run_recommended_exam, report=report,
                                         finish=finish_hiro_qualifier)
            return report
        finally:
            context.gkh_hif_qualifier_report = dict(report, plan=plan, inputs=ui.attempts,
                support_changes=ui.support_changes, rental=ui.rental_receipt)
            if ui.memory_inventory is not None:
                context.gkh_hif_qualifier_report['memory_inventory'] = ui.memory_inventory
            write_hif_record(runtime / f'qualifier-preparation-{task_id}-{time.time_ns()}.json',
                context.gkh_hif_qualifier_report)
