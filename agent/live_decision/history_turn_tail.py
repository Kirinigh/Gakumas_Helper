"""Observe a complete turn tail spanning the bottom and its preceding pages."""

import re
import time
from dataclasses import asdict

from .history_chain import (
    _group_lines,
    read_history_page,
    _join_history_prefix,
    history_chain_geometry,
    history_page_consensus,
)


def join_turn_tail_pages(pages, expected_turn):
    """Keep the full-history overlap proof, ending at the actual log bottom."""
    if expected_turn is not None and (type(expected_turn) is not int or expected_turn < 1):
        raise ValueError('history turn tail lacks its observed current turn')
    combined, frames, overlaps, score = _join_history_prefix(
        pages, require_bottom=True, require_top=False)
    boundaries = [index for index, line in enumerate(combined)
                  if line.kind in ('card', 'drink', 'turn-start')]
    if boundaries and combined[boundaries[-1]].kind == 'card':
        groups = _group_lines(combined[boundaries[-1]:])
        if (len(groups) != 1 or groups[0].kind != 'card'
                or not re.fullmatch(r'スキルカード「[^「」]+」', groups[0].title)
                or len(groups[0].lines) < 2 or not any('→' in line for line in groups[0].lines[1:])):
            raise ValueError('history latest card tail is incomplete')
        return dict(lines=groups[0].lines, score=score, thumb=pages[-1].page.thumb), frames, overlaps
    if expected_turn is None:
        raise ValueError('history turn tail lacks its observed current turn')
    starts = [i for i, line in enumerate(combined)
              if line.kind == 'turn-start' and line.text == f'{expected_turn}ターン目']
    if not starts:
        raise ValueError('history turn tail needs an earlier page')
    if len(starts) != 1:
        raise ValueError('history turn tail has repeated turn headings')
    groups = _group_lines(combined[starts[0]:])
    header = groups[0].lines
    # The complete-page reader also observes the same-row attribute multiplier
    # and hand button which latest-history excludes below the heading's box.
    if (len(header) not in (1, 2, 3)
            or len(header) >= 2 and not re.fullmatch(r'\d+%', header[1])
            or len(header) == 3 and header[2] != '手札情報'
            or len(groups) < 2
            or any(group.kind != 'system' or len(group.lines) < 2 for group in groups[1:])):
        raise ValueError('history turn tail is not a complete current turn-start group')
    lines = (header[0], *(line for group in groups[1:] for line in group.lines))
    if not any('→' in line for line in lines[1:]):
        from .history import _complete_history_draw

        for group in groups[1:]:
            effects = group.lines[1:]
            if (not group.title.startswith('持続効果「') or not group.title.endswith('」')
                    or len(effects) % 2 or any(not _complete_history_draw(name, effect)
                        for name, effect in zip(effects[::2], effects[1::2]))):
                raise ValueError('history turn-start draw group is incomplete')
    return dict(lines=lines, score=score, thumb=pages[-1].page.thumb), frames, overlaps


def scroll_open_history(context, *, expected_thumb, fresh, navigation_target, event, to_bottom=False):
    """Use the same fresh control/scrollbar proof for both existing scroll directions."""
    from utils.maa_input import verify_maa_input_endpoints

    controller = context.tasker.controller
    first, _ = fresh()
    first_box = navigation_target(first, True)
    image, stamp = fresh()
    box = navigation_target(image, True)
    for sample in (first, image):
        thumb, _, _ = history_chain_geometry(sample, top_thumb_y=262)
        if any(abs(a-b) > 2 for a, b in zip(thumb, expected_thumb)):
            raise ValueError('history moved before turn-tail navigation')
    if box != first_box:
        raise ValueError('history navigation control changed between frames')
    endpoints = verify_maa_input_endpoints(controller.info,
        expected_client_size=getattr(context, 'gkh_hif_client_size', (720, 1280)))
    if time.perf_counter()-stamp > 1:
        raise ValueError('history navigation snapshot expired')
    swipe = ([685, (thumb[0]+thumb[1])//2, 685, 1134, 650] if to_bottom
             else [560, 610, 560, 850, 500])
    event['input_click_count'] += 1
    event['inputs'].append(dict(phase='turn_tail_scroll', box=box, swipe=swipe,
        direction='to_bottom' if to_bottom else 'toward_top', thumb=thumb, endpoints=endpoints))
    if not controller.post_swipe(*swipe).wait().succeeded:
        raise ValueError('history turn-tail Maa navigation failed')
    time.sleep(2.5)


def align_open_history_bottom(context, frames, *, capture_frames, frame_identifier,
                              fresh, navigation_target, event):
    """One proved scroll for an already-open log; all following readers stay strict."""
    def observed(values):
        if not values:
            raise ValueError('history bottom alignment requires an observed frame')
        ids = [frame_identifier(frame) for frame in values]
        geometries = [history_chain_geometry(frame, top_thumb_y=262) for frame in values]
        if len(set(ids)) != len(values) or any(geometry != geometries[0] for geometry in geometries[1:]):
            raise ValueError('history bottom alignment scrollbar is unstable')
        return ids, geometries

    ids, geometries = observed(frames)
    if geometries[0][2]:
        return frames
    if event.get('latest_bottom_alignment') is not None or len(event['inputs']) != 1:
        raise ValueError('history bottom alignment was already attempted')
    proof = dict(accepted=False, before_frame_ids=ids, before_geometries=geometries,
        after_frame_ids=[], after_geometries=[], input_index=len(event['inputs']))
    event['latest_bottom_alignment'] = proof
    scroll_open_history(context, expected_thumb=geometries[0][0], fresh=fresh,
        navigation_target=navigation_target, event=event, to_bottom=True)
    after = tuple(capture_frames())
    new_ids, new_geometries = observed(after)
    proof.update(after_frame_ids=new_ids, after_geometries=new_geometries)
    old, new = geometries[0][0], new_geometries[0][0]
    if (set(ids) & set(new_ids) or not new_geometries[0][2] or new[0] <= old[0]
            or abs((new[1]-new[0])-(old[1]-old[0])) > 2):
        raise ValueError('history bottom alignment did not reach an unchanged complete scrollbar')
    proof['accepted'] = True
    return after


def observe_open_turn_tail(context, *, capture_frames, frame_identifier, expected_turn,
                          fresh, navigation_target, event):
    """Read at most three earlier pages, then restore and recheck the bottom."""
    started = time.perf_counter()
    pages = []
    moved = False

    def read_page():
        frames = tuple(capture_frames())
        ids = tuple(frame_identifier(frame) for frame in frames)
        return history_page_consensus(
            [read_history_page(context, frame, top_thumb_y=262) for frame in frames], ids)

    def scroll(expected_thumb, *, to_bottom=False):
        nonlocal moved
        if not to_bottom and time.perf_counter()-started > 30:
            raise ValueError('history turn tail observation budget reached')
        count = len(event['inputs'])
        try:
            scroll_open_history(context, expected_thumb=expected_thumb, fresh=fresh,
                navigation_target=navigation_target, event=event, to_bottom=to_bottom)
        finally:
            moved = moved or len(event['inputs']) > count

    original = read_page()
    if not original.page.at_bottom:
        raise ValueError('history turn tail must start at the verified bottom')
    pages.append(original)
    try:
        for earlier in range(4):
            try:
                observation, frame_ids, overlaps = join_turn_tail_pages(pages, expected_turn)
                event['turn_tail_pages'] = [asdict(page) for page in pages]
                event['turn_tail_frame_ids'] = frame_ids
                event['turn_tail_overlaps'] = overlaps
                return observation, original.frame_ids
            except ValueError as error:
                if str(error) != 'history turn tail needs an earlier page' or earlier == 3:
                    raise
            if pages[0].page.at_top:
                raise ValueError('current turn heading is absent from the loaded history')
            scroll(pages[0].page.thumb)
            pages.insert(0, read_page())
    except ValueError as error:
        # A cleanup failure must not hide the reader failure that caused it.
        event['turn_tail_read_error'] = str(error)
        raise
    finally:
        if moved:
            image, _ = fresh()
            thumb, _, _ = history_chain_geometry(image, top_thumb_y=262)
            scroll(thumb, to_bottom=True)
            restored = read_page()
            old, new = original.page, restored.page
            used = {frame for page in pages for frame in page.frame_ids}
            if (used.intersection(restored.frame_ids)
                    or not new.at_bottom or new.score != old.score
                    or any(abs(a-b) > 2 for a, b in zip(new.thumb, old.thumb))
                    or tuple(line.content_key for line in new.lines)
                        != tuple(line.content_key for line in old.lines)):
                raise ValueError('history bottom changed during turn-tail observation')
            event['turn_tail_bottom_restored'] = asdict(restored)
