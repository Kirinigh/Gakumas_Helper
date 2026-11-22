"""Replay latest-history navigation, including a bounded observed turn tail."""
import re
import json
from dataclasses import asdict


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _page(raw):
    from live_decision.history import normalize_history_text
    from live_decision.history_chain import HistoryLine, HistoryPage, VerifiedHistoryPage, _line_kind

    try:
        page = raw['page']
        lines = tuple(HistoryLine(line['text'], tuple(line['box']), line['kind']) for line in page['lines'])
        parsed = VerifiedHistoryPage(HistoryPage(lines, tuple(page['thumb']), page['at_top'],
            page['at_bottom'], page['score'], page['clipped_top'], page['clipped_bottom']), tuple(raw['frame_ids']))
        if (not _same(asdict(parsed), raw) or not parsed.frame_ids
                or len(set(parsed.frame_ids)) != len(parsed.frame_ids)
                or any(not isinstance(key, str) or not key for key in parsed.frame_ids)
                or any(type(page[key]) is not bool for key in ('at_top', 'at_bottom', 'clipped_top', 'clipped_bottom'))
                or type(page['score']) is not int or page['score'] < 0
                or len(parsed.page.thumb) != 2 or any(type(v) is not int for v in parsed.page.thumb)
                or not 262 <= parsed.page.thumb[0] <= parsed.page.thumb[1]-59 <= 1136-59
                or page['at_top'] is not (parsed.page.thumb[0] == 262)
                or page['at_bottom'] is not (parsed.page.thumb[1] >= 1129)
                or page['at_top'] and page['clipped_top'] or page['at_bottom'] and page['clipped_bottom']
                or not lines):
            raise ValueError('history turn-tail page metadata is incomplete')
        for index, line in enumerate(lines):
            if (not isinstance(line.text, str) or not line.text or normalize_history_text(line.text) != line.text
                    or len(line.box) != 4 or any(type(v) is not int for v in line.box)
                    or min(line.box[:2]) < 0 or min(line.box[2:]) <= 0
                    or line.box[0]+line.box[2] > 720 or line.box[1]+line.box[3] > 1280):
                raise ValueError('history turn-tail line metadata is incomplete')
            # The page reader splits a merged turn/multiplier into two exact
            # texts sharing the original box. Retain that existing exception.
            split_multiplier = (index > 0 and line.kind == 'detail' and re.fullmatch(r'\d+%', line.text)
                and lines[index-1].kind == 'turn-start' and line.box == lines[index-1].box)
            if line.kind != _line_kind(line.text, line.box) and not split_multiplier:
                raise ValueError('history turn-tail line ownership differs from its text')
        return parsed
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('history turn-tail page record is incomplete') from error


def history_navigation_proofs(history):
    """Return original inputs only after replaying any additional page proof.

    Callers retain their old two-click checks and complete state/history
    binding. This function grants no input or settlement authority.
    """
    if 'latest_bottom_alignment' in history:
        return _aligned_navigation_proofs(history)
    inputs = history.get('inputs', ())
    tail = {key for key in history if key.startswith('turn_tail_')}
    if not tail:
        return [dict(entry, phase='close' if entry.get('closing') else 'open') for entry in inputs]

    from live_decision.history_turn_tail import join_turn_tail_pages

    if (tail not in ({'turn_tail_pages', 'turn_tail_frame_ids', 'turn_tail_overlaps'},
                     {'turn_tail_pages', 'turn_tail_frame_ids', 'turn_tail_overlaps', 'turn_tail_bottom_restored'})
            or history.get('event') != 'live_history_observe' or history.get('accepted') is not True
            or history.get('returned_unchanged') is not True
            or type(history.get('input_click_count')) is not int
            or history['input_click_count'] != len(inputs)
            or len(inputs) < 2 or inputs[0].get('closing') is not False
            or inputs[-1].get('closing') is not True
            or any('swipe' in entry or entry.get('phase') == 'turn_tail_scroll' for entry in (inputs[0], inputs[-1]))):
        raise ValueError('history turn-tail navigation is incomplete')
    pages = tuple(_page(raw) for raw in history['turn_tail_pages'])
    observation = history.get('observation', {})
    lines = observation.get('lines', ())
    turn = re.fullmatch(r'([1-9]\d*)ターン目', lines[0]) if lines and isinstance(lines[0], str) else None
    card = bool(lines and isinstance(lines[0], str)
                and re.fullmatch(r'スキルカード「[^「」]+」', lines[0]))
    if not 1 <= len(pages) <= 4 or turn is None and not card:
        raise ValueError('history turn-tail current heading or page bound is missing')
    observed, frame_ids, overlaps = join_turn_tail_pages(pages, int(turn[1]) if turn else None)
    if (not _same(observed, observation) or not _same(frame_ids, history['turn_tail_frame_ids'])
            or not _same(overlaps, history['turn_tail_overlaps'])
            or not _same(pages[-1].frame_ids, history.get('frame_ids'))):
        raise ValueError('history turn-tail differs from its independently joined pages')
    if len(pages) == 1:
        if len(inputs) != 2 or 'turn_tail_bottom_restored' in history:
            raise ValueError('history turn-tail contains unobserved navigation')
        return [dict(entry, phase='close' if entry['closing'] else 'open') for entry in inputs]
    if 'turn_tail_bottom_restored' not in history or len(inputs) != len(pages)+2:
        raise ValueError('history turn-tail lacks its bounded return navigation')
    restored = _page(history['turn_tail_bottom_restored'])
    old, new = pages[-1].page, restored.page
    if (set(restored.frame_ids) & set(frame_ids) or not new.at_bottom
            or new.score != old.score
            or any(abs(a-b) > 2 for a, b in zip(new.thumb, old.thumb))
            or tuple(line.content_key for line in new.lines) != tuple(line.content_key for line in old.lines)):
        raise ValueError('history turn-tail bottom was reused or changed')
    # Reads are stored top-to-bottom, whereas navigation starts at the bottom.
    scroll_pages = (*reversed(pages[1:]), pages[0])
    for index, (entry, page) in enumerate(zip(inputs[1:-1], scroll_pages, strict=True)):
        thumb = entry.get('thumb', ())
        returning = index == len(scroll_pages)-1
        if (entry.get('phase') != 'turn_tail_scroll' or 'closing' in entry
                or entry.get('direction') != ('to_bottom' if returning else 'toward_top')
                or len(thumb) != 2 or any(type(v) is not int for v in thumb)
                or any(abs(a-b) > 2 for a, b in zip(thumb, page.page.thumb))):
            raise ValueError('history turn-tail scroll differs from its observed page')
        expected = [685, (thumb[0]+thumb[1])//2, 685, 1134, 650] if returning else [560, 610, 560, 850, 500]
        if not _same(entry.get('swipe'), expected):
            raise ValueError('history turn-tail scroll coordinates or duration changed')
    return [dict(inputs[0], phase='open'), *[dict(entry) for entry in inputs[1:-1]],
            dict(inputs[-1], phase='close')]


def _aligned_navigation_proofs(history):
    """Replay one observed initial bottom alignment before the existing reader."""
    alignment = history['latest_bottom_alignment']
    inputs = history.get('inputs', ())
    keys = {'accepted', 'before_frame_ids', 'before_geometries', 'after_frame_ids',
            'after_geometries', 'input_index'}
    if (not isinstance(alignment, dict) or set(alignment) != keys
            or alignment['accepted'] is not True or type(alignment['input_index']) is not int
            or alignment['input_index'] != 1 or history.get('accepted') is not True
            or history.get('event') != 'live_history_observe'
            or history.get('returned_unchanged') is not True
            or type(history.get('input_click_count')) is not int
            or history['input_click_count'] != len(inputs) or len(inputs) < 3
            or inputs[0].get('closing') is not False or inputs[-1].get('closing') is not True):
        raise ValueError('history initial bottom alignment is incomplete')
    groups, geometries = [], []
    for prefix in ('before', 'after'):
        frames, values = alignment[prefix+'_frame_ids'], alignment[prefix+'_geometries']
        if (not isinstance(frames, (list, tuple)) or not frames
                or any(not isinstance(frame, str) or not frame for frame in frames)
                or len(set(frames)) != len(frames) or any(set(frames) & set(prior) for prior in groups)
                or not isinstance(values, (list, tuple)) or len(values) != len(frames)
                or any(not _same(value, values[0]) for value in values[1:])):
            raise ValueError('history alignment lacks distinct observations of the scrollbar')
        value = values[0]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError('history alignment geometry is malformed')
        thumb, top, bottom = value
        if (not isinstance(thumb, (list, tuple)) or len(thumb) != 2
                or any(type(number) is not int for number in thumb)
                or not 262 <= thumb[0] <= thumb[1]-59 <= 1136-59
                or type(top) is not bool or type(bottom) is not bool
                or top is not (thumb[0] == 262) or bottom is not (thumb[1] >= 1129)):
            raise ValueError('history alignment geometry contradicts its boundaries')
        groups.append(frames)
        geometries.append(value)
    before, after = geometries
    if (before[2] or not after[2] or after[0][0] <= before[0][0]
            or abs((after[0][1]-after[0][0])-(before[0][1]-before[0][0])) > 2):
        raise ValueError('history alignment did not reach the unchanged log bottom')
    entry = inputs[1]
    thumb = entry.get('thumb')
    if (entry.get('phase') != 'turn_tail_scroll' or entry.get('direction') != 'to_bottom'
            or 'closing' in entry or not isinstance(thumb, (list, tuple)) or len(thumb) != 2
            or any(type(value) is not int for value in thumb)
            or any(abs(a-b) > 2 for a, b in zip(thumb, before[0]))
            or not _same(entry.get('swipe'), [685, sum(thumb)//2, 685, 1134, 650])):
        raise ValueError('history alignment input differs from its observed scrollbar')
    # Preserve original ordering and data. The old consumer still proves any
    # current-turn pages and their own return-to-bottom navigation separately.
    remainder = {key:value for key, value in history.items() if key != 'latest_bottom_alignment'}
    remainder.update(inputs=[*inputs[:1], *inputs[2:]], input_click_count=len(inputs)-1)
    if not any(key.startswith('turn_tail_') for key in history) and len(remainder['inputs']) != 2:
        raise ValueError('history alignment contains unexplained extra navigation')
    proofs = history_navigation_proofs(remainder)
    return [proofs[0], dict(entry), *proofs[1:]]
