"""Read complete visible history groups on an already open page; no navigation."""

import re
from dataclasses import asdict

from live_decision.history_chain import (
    HistoryLine,
    HistoryPage,
    _group_lines,
    read_history_page,
    history_page_consensus,
)


def _visible_groups(page):
    if not page.at_bottom or page.clipped_bottom:
        raise ValueError('visible history requires an unclipped bottom page')
    start = 0
    while start < len(page.lines) and page.lines[start].kind == 'detail':
        start += 1
    lines = page.lines[start:]
    if not lines or any(line.kind not in ('card', 'drink', 'system', 'turn-start', 'detail') for line in lines):
        raise ValueError('visible history has no complete recognized heading')
    groups = _group_lines(lines)
    if any(group.kind != 'turn-start' and len(group.lines) < 2 for group in groups):
        raise ValueError('visible history group has no complete effects')
    # Mirror the latest reader: same-row turn multiplier and hand-information
    # text remain in raw groups, but are outside its latest-effect fingerprint.
    headings = [line for line in lines if line.kind in ('card', 'turn-start')]
    if not headings:
        raise ValueError('visible history has no latest card or turn heading')
    last = headings[-1]
    tail = tuple(line.text for line in lines if line is last or (
        line.box[1] >= last.box[1] + (last.box[3] if last.kind == 'turn-start' else 0)))
    return dict(groups=[asdict(group) for group in groups],
                skipped_prefix_lines=[asdict(line) for line in page.lines[:start]],
                latest_lines=tail,
                turns=[int(match[1]) for group in groups
                       if (match := re.fullmatch(r'([1-9]\d*)ターン目', group.title))])


def read_visible_history(context, image, *, frame_id, top_thumb_y=264):
    """One actual page is diagnostic until independently observed three times."""
    run = getattr(context, 'gkh_hif_live_context', None)
    if (not isinstance(frame_id, str) or not frame_id
            or not isinstance(getattr(run, 'run_id', None), str) or not run.run_id
            or type(getattr(run, 'revision', None)) is not int or run.revision < 0):
        raise ValueError('visible history frame or Live epoch is missing')
    page = read_history_page(context, image, top_thumb_y=top_thumb_y)
    return dict(accepted=False, read_accepted=True, three_frame_consensus=False,
                frame_id=frame_id, run_id=run.run_id, revision=run.revision,
                score=page.score, page=asdict(page), **_visible_groups(page))


def _restore_page(observation):
    if observation.get('read_accepted') is not True:
        raise ValueError('visible history observation is unread')
    raw = observation['page']
    page = HistoryPage(lines=tuple(HistoryLine(row['text'], tuple(row['box']), row['kind'])
                                  for row in raw['lines']),
                       thumb=tuple(raw['thumb']), at_top=raw['at_top'],
                       at_bottom=raw['at_bottom'], score=raw['score'],
                       clipped_top=raw['clipped_top'], clipped_bottom=raw['clipped_bottom'])
    return page


def visible_history_consensus(observations, *, latest_event):
    """Bind full groups to the latest reader's exact same independent frames."""
    observations = tuple(observations)
    ids = tuple(row.get('frame_id') for row in observations)
    expected_ids = tuple(latest_event.get('frame_ids', ()))
    if (len(observations) != 3 or len(set(ids)) != 3 or ids != expected_ids
            or any(not isinstance(key, str) or not key for key in ids)
            or latest_event.get('accepted') is not True
            or latest_event.get('returned_unchanged') is not True):
        raise ValueError('visible history lacks the latest three independent frames')
    epochs = {(row.get('run_id'), row.get('revision')) for row in observations}
    if len(epochs) != 1:
        raise ValueError('visible history Live epoch changed')
    run_id, revision = next(iter(epochs))
    if not isinstance(run_id, str) or not run_id or type(revision) is not int or revision < 0:
        raise ValueError('visible history Live epoch is missing')
    before, after = latest_event.get('before', {}), latest_event.get('after', {})
    if (any(not before.get(key) or before.get(key) != after.get(key) for key in ('fields', 'families'))
            or len(before['fields']) < 4
            or any(len(set(core.get('frame_ids', ()))) != 3
                   or set(core['frame_ids']) & set(ids) for core in (before, after))):
        raise ValueError('visible history lacks unchanged Live checkpoints')
    pages = tuple(_restore_page(row) for row in observations)
    verified = history_page_consensus(pages, ids)
    projections = [_visible_groups(page) for page in pages]
    first = projections[0]
    if any((row['groups'], row['latest_lines'], row['turns']) !=
           (first['groups'], first['latest_lines'], first['turns']) for row in projections[1:]):
        raise ValueError('visible history groups changed')
    latest = latest_event.get('observation', {})
    if (tuple(latest.get('lines', ())) != first['latest_lines']
            or latest.get('score') != verified.page.score
            or before['fields'][3] != verified.page.score
            or tuple(latest.get('thumb', ())) != verified.page.thumb):
        raise ValueError('visible history disagrees with the latest observation')
    return dict(accepted=True, event='live_visible_history_observe', input_click_count=0,
                run_id=run_id, revision=revision, frame_ids=list(ids),
                three_frame_consensus=True, score=verified.page.score,
                at_bottom=True, clipped_bottom=False,
                complete_from_start=False, complete_to_end=True,
                **first, observations=list(observations), latest_observation=latest,
                before=before, after=after, returned_unchanged=True)
