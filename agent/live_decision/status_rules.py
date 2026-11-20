"""Offline status-row and memory-effect evidence rules."""

import re
import unicodedata


def _normalize_status_text(text):
    # CJK glyph equivalences; 1586 independently reread 变化 as 変化.
    # Do not fold kana sizes or infer a catalogue name from similar text.
    normalized = ''.join(unicodedata.normalize('NFKC',text).split()).translate(
        str.maketrans({'增':'増','絕':'絶','变':'変','轨':'軌'})).replace('消費体カ','消費体力')
    # A duplicated OCR unit (15%％) has one amount; keep every digit and all
    # surrounding wording for the original independent-read comparisons.
    return re.sub(r'(?<=\d)%{2,}', '%', normalized)


def status_panel_consensus(pages):
    pages = tuple(pages)
    if not pages or any(page.content_key != pages[0].content_key for page in pages):
        raise ValueError('status panel has no consistent semantic observation')
    if not _status_geometry_stable(tuple((tuple(row.box for row in page.rows),page.thumb,
                                          page.at_top,page.at_bottom,page.partial_row_count) for page in pages)):
        raise ValueError('status panel is still scrolling')
    return pages[-1]


def _status_geometry_stable(geometries):
    if not geometries:
        return False
    first = geometries[0]
    # Partial rows touch the viewport boundary and supply no readable content.
    # Their background connectivity can flicker while complete rows and the
    # thumb stay fixed; keep the count as telemetry, outside the motion proof.
    if any(len(row[0]) != len(first[0]) or row[2:4] != first[2:4] for row in geometries):
        return False
    if any(max(row[1][i] for row in geometries)-min(row[1][i] for row in geometries) > 1 for i in range(2)):
        return False
    return all(max(row[0][i][j] for row in geometries)-min(row[0][i][j] for row in geometries) <= 2
               for i in range(len(first[0])) for j in range(4))


def join_status_panel_pages(pages):
    """Join only unique suffix/prefix overlap from top to bottom.

    Missing pages, same-named but different abilities and conflicting counts
    cannot be repaired by dropping duplicates or assuming an exhausted effect.
    """
    pages = tuple(pages)
    if not pages or not pages[0].at_top or not pages[-1].at_bottom:
        raise ValueError('status list endpoints are incomplete')
    return _join_status_panel_prefix(pages)


def _join_status_panel_prefix(pages):
    if not pages or not pages[0].at_top:
        raise ValueError('status list prefix must start at its verified top')
    return _join_status_panel_rows(pages)


def _join_status_panel_rows(pages):
    """Join pages from an established entry using their unique overlaps."""
    combined = list(pages[0].rows)
    reference_thumb_length = pages[0].thumb[1]-pages[0].thumb[0]
    scroll_ratio = None
    for previous, page in zip(pages, pages[1:]):
        if page.thumb[0] <= previous.thumb[0]:
            raise ValueError('status list did not advance')
        old = [row.content_key for row in combined]
        new = [row.content_key for row in page.rows]
        overlaps = [size for size in range(1,min(len(old),len(new))+1) if old[-size:] == new[:size]]
        semantic_unique = len(overlaps) == 1
        if len(overlaps) > 1 and scroll_ratio is not None:
            # Repeated effects are separate rows. As in history-chain joining,
            # use independent scrollbar motion rather than picking the largest
            # textual overlap. Calibrate only from earlier unique text matches;
            # the status viewport height need not be guessed from its crop.
            overlaps = [size for size in overlaps
                if (interval := _status_overlap_scroll_ratio(previous, page, size,
                    reference_thumb_length)) is not None
                and max(interval[0], scroll_ratio[0]) <= min(interval[1], scroll_ratio[1])]
        if len(overlaps) != 1:
            raise ValueError('status list overlap is missing or ambiguous')
        if semantic_unique:
            interval = _status_overlap_scroll_ratio(previous, page, overlaps[0], reference_thumb_length)
            if interval is not None:
                scroll_ratio = interval if scroll_ratio is None else (
                    max(scroll_ratio[0], interval[0]), min(scroll_ratio[1], interval[1]))
        combined.extend(page.rows[overlaps[0]:])
    return tuple(combined)


def _status_overlap_scroll_ratio(previous, page, count, reference_thumb_length):
    """Possible content/scrollbar motion ratio from complete corresponding rows.

    Two-pixel row and scrollbar-edge uncertainty follows the existing stable
    geometry and history-join rules. Text remains mandatory for every pair.
    """
    delta = page.thumb[0]-previous.thumb[0]
    if (delta <= 2 or not 0 < count <= min(len(previous.rows), len(page.rows))
            or any(abs(p.thumb[1]-p.thumb[0]-reference_thumb_length) > 2
                   for p in (previous, page))):
        return None
    shifts = []
    for left, right in zip(previous.rows[-count:], page.rows[:count]):
        if (left.content_key != right.content_key
                or any(abs(left.box[i]-right.box[i]) > 2 for i in (0, 2, 3))):
            return None
        shifts.append(left.box[1]+left.box[3]/2-right.box[1]-right.box[3]/2)
    if min(shifts) <= 2 or max(shifts)-min(shifts) > 2:
        return None
    return (min(shifts)-2)/(delta+2), (max(shifts)+2)/(delta-2)


def memory_effect_description(row):
    """Remove exactly the independently observed counter, not effect digits.

    Joined OCR lines can end in e.g. ``追加+12回``: effect +1 and counter
    2. A greedy digit suffix would erase the actual effect magnitude.
    """
    if row.title != 'メモリーアビリティ' or row.unit != '回' or type(row.count) is not int or row.count < 1:
        raise ValueError('visible memory ability count is inconsistent with disappearance semantics')
    suffix = f'{row.count}回'
    if not row.description.endswith(suffix):
        raise ValueError('memory description does not end in its observed use count')
    description = _normalize_status_text(row.description[:-len(suffix)]).replace(
        '·スキルカードを引く', '・スキルカードを引く')
    if not description:
        raise ValueError('memory ability description is missing')
    return description


def memory_use_counts(previous_pages, current_pages, *, previous_run_id, current_run_id):
    """D-067: exhausted memory abilities disappear from the detail list.

    Current absence requires complete endpoints and unique page overlap.
    Earlier ownership may be established by one three-frame page explicitly
    showing that ability. Only those known abilities can acquire zero; this
    does not claim a complete historical inventory or infer other effect types.
    """
    if not previous_run_id or previous_run_id != current_run_id:
        raise ValueError('memory observations must belong to the same verified run')

    def collect(groups, *, require_complete):
        pages = tuple(status_panel_consensus(group) for group in groups)
        rows = (pages[0].rows if not require_complete and len(pages) == 1
                else join_status_panel_pages(pages))
        abilities = {}
        for row in rows:
            if row.title != 'メモリーアビリティ':
                continue
            # The final red use count can change; retain every number in the
            # trigger/effect itself so different abilities cannot alias.
            key = memory_effect_description(row)
            if not key or key in abilities:
                raise ValueError('memory ability identity is missing or duplicated')
            abilities[key] = row
        return abilities

    previous = collect(previous_pages,require_complete=False)
    current = collect(current_pages,require_complete=True)
    if current.keys() - previous.keys():
        raise ValueError('current memory ability was not in the known run inventory')
    observations = []
    for key, known in previous.items():
        observed = current.get(key)
        count = observed.count if observed is not None else 0
        if count > known.count:
            raise ValueError('memory use count increased without a confirmed replenishment event')
        observations.append(dict(description_key=key,previous_count=known.count,remaining_uses=count,
                                 source='visible_count' if observed else 'known_ability_absent_from_complete_list'))
    return tuple(observations)
