"""Follow the qualifier certificate, photo and memory settlement pages."""
from __future__ import annotations

import re
import unicodedata

import cv2


def _heading(frame, bottom):
    return ''.join(unicodedata.normalize('NFKC', row['text']).replace(' ', '')
                   for row in frame.rows if row['box'][1] < bottom)


def is_qualifier_result(frame):
    head = _heading(frame, 230)
    return '選抜試験' in head and '結果' in head


def qualifier_exam_number(frame, *, result):
    """Hiro identifies an exam by its header or the 128/64/32 entrants."""
    head = _heading(frame, 230 if result else 800)
    heading = (r'選抜試験.*?第([1-3])回.*?結果' if result else r'選抜試験(?:第)?([1-3])(?:回)?突破ライン')
    if match := re.search(heading, head):
        return int(match[1])
    if (is_qualifier_result(frame) if result else '突破ライン' in head):
        entrants = {int(match[1]) for row in frame.rows if row['box'][0] < 450
                    and 200 <= row['box'][1] <= 1050
                    and (match := re.search(r'(?<!\d)(128|64|32)\s*人', row['text']))}
        if len(entrants) == 1:
            return {128: 1, 64: 2, 32: 3}[entrants.pop()]
    return None


def read_qualifier_result(frame):
    """Use the actual result and the player's white border, as in Hiro."""
    if (number := qualifier_exam_number(frame, result=True)) is None:
        return None
    if getattr(frame.image, 'shape', ()) != (1280, 720, 3):
        return None
    edge = cv2.cvtColor(frame.image[:, 20:32], cv2.COLOR_BGR2HSV)
    white = (edge[..., 1] <= 40) & (edge[..., 2] >= 225)
    rows, statuses, cutoffs = [], [], []
    for row in frame.rows:
        x, y, _, height = row['box']
        if not 250 <= y <= 1050:
            continue
        cy = y + height // 2
        value = unicodedata.normalize('NFKC', row['text']).replace(' ', '')
        if x >= 200 and (match := re.fullmatch(r'(\d[\d,.]*)(?:[Pp][Tt])?', value)):
            digits = re.sub(r'\D', '', match[1])
            if len(digits) >= 2:
                rows.append((cy - 22, int(digits)))
        elif x < 160 and value in ('突破', '敗退'):
            statuses.append((cy, value == '突破'))
        elif x < 160 and re.fullmatch(r'\d+人', value):
            cutoffs.append(cy)
    mine = [(cy, score) for cy, score in rows if white[cy - 50:cy + 50].mean() >= .08]
    if len(mine) != 1:
        return None
    cy, score = mine[0]
    passed = {status for y, status in statuses if abs(y - cy) <= 30}
    if not passed and len(cutoffs) == 1:
        passed.add(cy < cutoffs[0])
    if len(passed) != 1:
        return None
    return dict(exam=number, score=score, passed=passed.pop())


def qualifier_certificate(frame):
    from .qualifier import label

    return (label(frame, '完了する', top=700) is not None
        and any('選抜試験突破者' in row['text'] for row in frame.rows))


def finish_qualifier(ui, frame):
    """Advance the distinct qualifier result and stop before a new cultivation."""
    from .ending import _completion_next
    from .qualifier import label

    def texts(f):
        return tuple(row['text'].replace(' ', '') for row in f.rows)

    def target(f):
        words = texts(f)
        if any('メモリーとして保存しますか' in word for word in words):
            return label(f, '決定', top=900) or (500, 1157, 1, 1)
        if any('メモリーにするフォト' in word for word in words):
            return label(f, '次へ', top=900) or (360, 1157, 1, 1)
        if qualifier_certificate(f):
            return label(f, '完了する', top=700)
        generate = label(f, '生成', top=850)
        if generate is not None:
            return generate
        complete = label(f, 'メモリー生成完了', bottom=460)
        if complete is not None:
            return complete
        try:
            return _completion_next(f.rows, f.image)
        except ValueError:
            return label(f, r'閉じる|OK|スキップ|決定|完了する', top=700)

    def end(f):
        words = texts(f)
        if any('開始確認' in word or word == '本戦へ' for word in words):
            return 'next_start'
        if label(f, 'プロデュース', top=920, bottom=1070) and label(f, 'アイドル', top=1060):
            return 'home'
        return None

    def paid(f):
        return any(any(key in word for key in ('購入', 'ジュエル', '有償', '課金')) for word in texts(f))

    if not qualifier_certificate(frame):
        raise ValueError('qualifier settlement was not identified')
    while True:
        if boundary := end(frame):
            result = dict(status='settled', boundary=boundary)
            if boundary == 'next_start':
                from .prestart_memory import read_confirm_memory

                memory = read_confirm_memory(getattr(ui, 'context', None), frame.image, frame.rows)
                if memory is not None:
                    result['selected_memory'] = dict(zip(('Vo', 'Da', 'Vi'), memory, strict=True))
            return result
        box = None if paid(frame) else target(frame)
        if box is None:
            frame = ui.need(lambda f: end(f) or (not paid(f) and target(f)), 'qualifier settlement continuation')
            continue
        ui.click(frame, box, settle=1.2)
        frame = ui.capture()
