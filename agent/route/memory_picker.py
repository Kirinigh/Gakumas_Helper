"""Read formation memories by physical row, using the Maa preparation driver."""
from __future__ import annotations

import cv2
import numpy as np

from .memory_details import read_memory_detail


def picker_open(rows):
    labels = [row['text'] for row in rows if row['box'][1] >= 1000]
    return 'キャンセル' in labels and '決定' in labels and any('成詳' in text for text in labels)


def grid_rows(image):
    """Use complete, unobscured groups of cards in Hiro's formation picker."""
    hsv = cv2.cvtColor(image[560:1040, 35:687], cv2.COLOR_BGR2HSV)
    occupancy = hsv[:, :, 1].mean(axis=1) >= 40
    edges = np.flatnonzero(np.diff(np.r_[False, occupancy, False]))
    result = []
    for top, bottom in edges.reshape(-1, 2):
        if not top or bottom + 560 > 995 or not 108 <= bottom - top <= 128:
            continue
        y0, y1 = int(top) + 560, int(bottom) + 560
        row = {}
        for col, x in enumerate((140, 360, 580)):
            tile = image[y0:y1, x-105:x+105]
            if cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)[:, :, 1].mean() >= 40:
                art = cv2.cvtColor(tile[8:-8, 8:-8], cv2.COLOR_BGR2GRAY)
                row[col] = ((x, (y0 + y1) // 2, 1, 1), cv2.resize(art, (48, 24)).astype('float32'))
        if row:
            result.append(row)
    return result


def overlap_position(previous, current):
    """Find the last previous row again, using the visible overlap."""
    for index in range(len(current) - 1, -1, -1):
        length = min(len(previous), index + 1)
        if length and all(a.keys() == b.keys() and all(
                np.abs(a[slot][1] - b[slot][1]).mean() <= 14 for slot in a)
                for a, b in zip(previous[-length:], current[index + 1 - length:index + 1])):
            return index
    return None


def _viewport_changed(first, current):
    first = cv2.cvtColor(first[560:1040, 35:687], cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(current[560:1040, 35:687], cv2.COLOR_BGR2GRAY)
    return cv2.mean(cv2.absdiff(first, current))[0] >= 2


class MemoryPicker:
    def __init__(self, ui, catalog):
        self.ui, self.catalog = ui, catalog

    def _detail_page(self, frame, number):
        for _ in range(3):
            detail = read_memory_detail(frame.rows, self.catalog)
            if detail is not None:
                if detail['page'] == number:
                    return frame, detail
                self.ui.click(frame, ((485 if number == 2 else 230), 530, 1, 1), settle=.3)
            else:
                self.ui.sleep(.4)
            frame = self.ui.capture()
            if not picker_open(frame.rows):
                raise ValueError('formation memory picker closed while reading details')
        return frame, None

    def _selected_detail(self, frame, box, last_pane):
        # Exactly one tap per slot: a repeat tap can commit the selection.
        before = frame.image[170:510, 40:680]
        self.ui.click(frame, box, settle=.3)
        next_frame = self.ui.wait(lambda f: picker_open(f.rows) and cv2.mean(
            cv2.absdiff(before, f.image[170:510, 40:680]))[0] > 6, timeout=3)
        changed = next_frame is not None
        frame = next_frame or self.ui.capture()
        if not picker_open(frame.rows):
            raise ValueError('formation memory picker closed during a read-only selection')
        if not changed and last_pane:
            return frame, dict(card=None, expected_timing=None, effects=None, abilities=None, detail_changed=False)
        frame, card = self._detail_page(frame, 1)
        frame, abilities = self._detail_page(frame, 2)
        frame, _ = self._detail_page(frame, 1)
        return frame, dict(card=(card or {}).get('card'), expected_timing=(card or {}).get('expected_timing'),
            effects=(card or {}).get('effects'), abilities=(abilities or {}).get('abilities'), detail_changed=changed)

    def scan(self, frame, *, stop_at=None):
        if not picker_open(frame.rows):
            raise ValueError('current formation memory picker is required')
        entries = []
        complete, row_number = False, 0
        still = 0
        for _ in range(80):
            next_frame = self.ui.swipe(frame, (360, 600), (360, 875))
            still = still + 1 if not _viewport_changed(frame.image, next_frame.image) else 0
            frame = next_frame
            if still == 2:
                break
        if still != 2:
            return dict(entries=entries, complete=False)
        start = 0
        for _ in range(600):
            rows = grid_rows(frame.image)
            if not rows:
                break
            for index, cells in enumerate(rows[start:], start):
                for col, (box, _) in cells.items():
                    frame, detail = self._selected_detail(frame, box, bool(entries))
                    entries.append(dict(detail, row=row_number + index, column=col))
                    if stop_at == (row_number + index, col):
                        return dict(entries=entries, complete=False, target=entries[-1])
            previous = grid_rows(frame.image)
            if not previous:
                break
            scroll = max(137, previous[-1][next(iter(previous[-1]))][0][1] - previous[0][next(iter(previous[0]))][0][1])
            next_frame = self.ui.swipe(frame, (360, 995), (360, 995 - min(scroll, 410)))
            if not _viewport_changed(frame.image, next_frame.image):
                check = self.ui.swipe(next_frame, (360, 995), (360, 995 - min(scroll, 410)))
                if not _viewport_changed(next_frame.image, check.image):
                    complete = True
                    break
                next_frame = check
            match = overlap_position(previous, grid_rows(next_frame.image))
            for _ in range(6):
                if match is not None:
                    break
                next_frame = self.ui.swipe(next_frame, (360, 600), (360, 737))
                match = overlap_position(previous, grid_rows(next_frame.image))
            if match is None:
                break
            advance = len(previous) - 1 - match
            if advance <= 0:
                break
            row_number += advance
            start, frame = match + 1, next_frame
        return dict(entries=entries, complete=complete)
