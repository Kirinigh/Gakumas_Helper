"""Read and join observed Live history pages without sending game input."""

import re
import time
from dataclasses import asdict, dataclass

import numpy as np

from .history import (
    _history_line,
    _history_score,
    normalize_history_text,
    _history_trigger_icon_row,
    history_navigation_target,
    _merge_split_card_reference_rows,
    qualified_history_card_reference,
)
from .inventory import (
    _inventory_ocr,
    capture_fresh_live_checkpoint,
    capture_initial_live_checkpoint,
)
from .hif_capture_adapter import captured_at as frame_captured_at

# 858 pixels: the header divider ends at y=248; body text starts below it.
# Its bounds are independent of the scrollbar's sampling strip (y=262).
_HISTORY_BODY_TOP = 250
# Content extends to y=1147, but the close button occludes it from y=1120.
# Preserve the existing two-pixel exact-line margin above that obstruction.
_HISTORY_BODY_SAFE_BOTTOM = 1118
# The scroll viewport continues behind the close button to y=1147. Use its
# full height, not the OCR-safe crop, to relate thumb motion to content motion.
_HISTORY_VIEWPORT_HEIGHT = 1148-_HISTORY_BODY_TOP


@dataclass(frozen=True)
class HistoryLine:
    text: str
    box: tuple[int, int, int, int]
    kind: str

    @property
    def content_key(self):
        return self.kind, self.text


@dataclass(frozen=True)
class HistoryPage:
    lines: tuple[HistoryLine, ...]
    thumb: tuple[int, int]
    at_top: bool
    at_bottom: bool
    score: int
    clipped_top: bool = False
    clipped_bottom: bool = False


@dataclass(frozen=True)
class VerifiedHistoryPage:
    page: HistoryPage
    frame_ids: tuple[str, ...]


@dataclass(frozen=True)
class HistoryGroup:
    kind: str
    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class HistoryChain:
    groups: tuple[HistoryGroup, ...]
    score: int
    frame_ids: tuple[str, ...]
    overlaps: tuple[int, ...]
    turns: tuple[int, ...]
    first_turn_observed: bool

    @property
    def complete_from_start(self):
        # This is the observed scrollable log's beginning. Its relationship to
        # the beginning of the Live is proved by the separate restart contract.
        return True

    @property
    def complete_to_end(self):
        return True

    def as_observation(self):
        return dict(
            groups=[dict(kind=g.kind, title=g.title, lines=list(g.lines)) for g in self.groups],
            score=self.score, turn=self.turns[-1] if self.turns else None,
            turns=list(self.turns), at_top=True, at_bottom=True,
            page_count=len(self.overlaps)+1, overlaps=list(self.overlaps),
            first_turn_observed=self.first_turn_observed,
        )


def history_chain_geometry(image, *, top_thumb_y, allow_top_candidate=False):
    """Use the existing bottom calibration and an independently observed top."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('history image size is uncalibrated')
    if (type(allow_top_candidate) is not bool
            or top_thumb_y is None and not allow_top_candidate
            or top_thumb_y is not None and (type(top_thumb_y) is not int or not 262 <= top_thumb_y <= 271)):
        raise ValueError('history top scrollbar calibration is missing')
    strip = image[262:1137, 683:688].astype(np.int16)
    dark = ((strip.min(2) >= 100) & (strip.max(2) <= 180)
            & (np.ptp(strip, axis=2) <= 30)).mean(1) >= .8
    indices = np.flatnonzero(dark)
    if len(indices) < 60 or np.any(np.diff(indices) != 1):
        raise ValueError('history scrollbar is not unique and continuous')
    thumb = int(indices[0]+262), int(indices[-1]+262)
    if top_thumb_y is not None and thumb[0] < top_thumb_y:
        raise ValueError('history scrollbar contradicts its top calibration')
    # None is used only while seeking a new viewport's top. It identifies a
    # candidate boundary; the observer must still confirm it by pulling upward.
    at_top = thumb[0] <= 271 if top_thumb_y is None else thumb[0] == top_thumb_y
    return thumb, at_top, bool(indices[-1] >= 867)


def _line_kind(text, box):
    # Drawn/generated card names are indented effect text, including wrapped
    # lines. They must never increase the observed submission count.
    if not 130 <= box[0] <= 150:
        return 'detail'
    if re.fullmatch(r'[1-9]\d*ターン目', text):
        return 'turn-start'
    if text.startswith('スキルカード「'):
        if text.endswith('」'):
            return 'card'
        if '」を' in text:
            return 'detail'
        raise ValueError('history card action heading is incomplete')
    if text.startswith(('Pドリンク', 'ドリンク')):
        return 'drink'
    if text.startswith(('メモリーアビリティ', '持続効果「', 'Pアイテム')):
        return 'system'
    return 'unknown'


def _history_turn_multiplier_row(title_box, multiplier_box):
    """Independently read fields share the existing two-pixel ink interval."""
    x,y,width,height = title_box
    nx,ny,_new_width,new_height = multiplier_box
    return (x+width < nx <= x+width+60
            and 16 <= height <= 40 and 16 <= new_height <= 40
            and min(y+height,ny+new_height)-max(y,ny) >= min(height,new_height)-2)


def _normalize_history_turn_rows(lines):
    """Keep observed turn/multiplier text in one physical row's reading order."""
    replacements, consumed = {}, set()
    for index, line in enumerate(lines):
        x, y, width, height = line.box
        if index in consumed or not 130 <= x <= 150:
            continue
        joined = re.fullmatch(r'([1-9]\d*ターン目)(\d+%)', line.text)
        if joined:
            title, multiplier = joined.groups()
            indices = [index]
        elif line.kind == 'turn-start' and re.fullmatch(r'[1-9]\d*ターン目', line.text):
            matches = [i for i, other in enumerate(lines) if i not in consumed
                and other.kind == 'detail' and re.fullmatch(r'\d+%', other.text)
                and _history_turn_multiplier_row(line.box, other.box)]
            if not matches:
                continue  # No missing multiplier is supplied to a partial row.
            if len(matches) != 1:
                raise ValueError('history turn row multiplier is ambiguous')
            title, multiplier = line.text, lines[matches[0]].text
            indices = [index, matches[0]]
        else:
            continue
        left = min(lines[i].box[0] for i in indices)
        top = min(lines[i].box[1] for i in indices)
        right = max(lines[i].box[0]+lines[i].box[2] for i in indices)
        bottom = max(lines[i].box[1]+lines[i].box[3] for i in indices)
        box = (left, top, right-left, bottom-top)
        hand = [i for i, other in enumerate(lines) if i not in consumed
            and other.text == '手札情報' and other.box[0] > right
            and min(bottom,other.box[1]+other.box[3])-max(top,other.box[1])
                >= min(bottom-top,other.box[3])-2]
        if len(hand) > 1:
            raise ValueError('history turn row hand information is ambiguous')
        indices += hand
        replacements[min(indices)] = [HistoryLine(title,box,'turn-start'),
            HistoryLine(multiplier,box,'detail'), *(lines[i] for i in hand)]
        consumed.update(indices)
    return tuple(part for index, line in enumerate(lines)
                 for part in replacements.get(index, [] if index in consumed else [line]))


def _history_thumbnail_trigger_header(image, box, detected, *, memory_header=False):
    """Locate an independent coloured thumbnail left of a complete title."""
    import cv2

    x,y,width,height = box
    if not (235 <= x <= 260 and 20 <= height <= 40 and width >= 40
            and _HISTORY_BODY_TOP <= y and y+height <= _HISTORY_BODY_SAFE_BOTTOM):
        return None
    center = y+height//2
    # The saturated artwork need not occupy the whole thumbnail (1837's
    # white/purple art). Its non-white physical rim owns the entity; colour
    # inside that same connected entity independently distinguishes it from
    # ordinary text. Include both rim ends outside the title's ink padding.
    crop = image[center-42:center+42,150:232]
    corners = (crop[:5,:5],crop[:5,-5:],crop[-5:,:5],crop[-5:,-5:])
    if any(not np.all(corner >= 235) for corner in corners):
        return None
    color = (np.ptp(crop.astype(np.int16),axis=2) >= 40).astype(np.uint8)
    _,labels,stats,_ = cv2.connectedComponentsWithStats((crop.min(2) < 235).astype(np.uint8))
    bodies = [(int(cx)+150,int(cy)+center-42,int(cw),int(ch))
              for label,(cx,cy,cw,ch,_area) in enumerate(stats[1:],1)
              if 60 <= cw <= 80 and 60 <= ch <= 80
              and cy > 0 and cy+ch < crop.shape[0]
              # The independently read memory header has a pastel icon;
              # its complete rim remains identical to the saturated cards.
              and np.count_nonzero(color & (labels == label)) >= (500 if memory_header else 1000)
              and abs(cx+cw/2-40) <= 3 and abs(cy+ch/2-42) <= 4]
    if len(bodies) != 1:
        return None
    left,top,bw,bh = bodies[0]
    owned = []
    for row in detected:
        rx, ry, rw, rh = row.box
        if not (130 <= rx and rw > 0 and rh > 0 and rx+rw < x
                and 0 <= ry < ry+rh <= image.shape[0]):
            continue
        ink_y, ink_x = np.nonzero(image[ry:ry+rh, rx:rx+rw].min(2) < 235)
        # Text detection padding can extend beyond the coloured thumbnail.
        # Only blank pixels may do so; every foreground pixel still belongs
        # to the independently located physical entity.
        if (len(ink_x) and left <= rx+ink_x.min() and rx+ink_x.max() < left+bw
                and top <= ry+ink_y.min() and ry+ink_y.max() < top+bh):
            owned.append(row)
    return tuple(owned[0].box) if len(owned) == 1 else None


def read_history_page(context, image, *, top_thumb_y):
    """Read every complete body line, not only the expected cards' names."""
    thumb, at_top, at_bottom = history_chain_geometry(image, top_thumb_y=top_thumb_y)
    detected = _inventory_ocr(context, image, 'HifHistoryLines', roi=[30,110,645,1008])
    # The union is only a crop proposal: interior references need an exact
    # complete-line read, while a boundary fragment is deferred as one row.
    detected = _merge_split_card_reference_rows(detected)
    exact_headers = {}
    trigger_headers = []
    thumbnail_icon_boxes = set()
    turn_headers = []
    for row in detected:
        raw = normalize_history_text(row.text)
        x,y,width,height = tuple(row.box)
        if (240 <= x <= 260 and _HISTORY_BODY_TOP <= y
                and y+height <= _HISTORY_BODY_SAFE_BOTTOM
                and (raw.startswith('メモリー') or raw == 'スキルカードサポート')):
            box = tuple(int(value) for value in row.box)
            text = _history_line(context,image[y-2:y+height+2,x-2:x+width+2].copy())
            exact_headers[box] = text
            if text in ('メモリーアビリティ','スキルカードサポート'):
                trigger_headers.append(box)
            if text == 'メモリーアビリティ':
                owned_icon = _history_thumbnail_trigger_header(image, box, detected, memory_header=True)
                if owned_icon is not None:
                    thumbnail_icon_boxes.add(owned_icon)
        elif (owned_icon := _history_thumbnail_trigger_header(image, tuple(row.box), detected)) is not None:
            # Triggered card effects use the same indented title/thumbnail
            # header as memory effects. The coloured physical thumbnail and
            # independent full title own its ink; its OCR spelling is unused.
            box = tuple(int(value) for value in row.box)
            text = _history_line(context,image[y-2:y+height+2,x-2:x+width+2].copy())
            if text != raw:
                raise ValueError('history thumbnail trigger title differs from its complete read')
            exact_headers[box] = text
            thumbnail_icon_boxes.add(owned_icon)
        elif (130 <= x <= 150 and _HISTORY_BODY_TOP <= y and y+height <= _HISTORY_BODY_SAFE_BOTTOM
                and re.fullmatch(r'[1-9]\d*ターン目', raw)):
            box = tuple(int(value) for value in row.box)
            text = _history_line(context,image[y-2:y+height+2,x-2:x+width+2].copy())
            exact_headers[box] = text
            if re.fullmatch(r'[1-9]\d*ターン目', text):
                turn_headers.append(box)
    turn_attribute_badges = set()
    for hx,hy,hw,hh in turn_headers:
        multipliers = [row for row in detected if .95 <= row.score <= 1
            and re.fullmatch(r'\d+%',normalize_history_text(row.text))
            and _history_turn_multiplier_row((hx,hy,hw,hh), row.box)]
        if len(multipliers) != 1:
            continue
        multiplier_x = multipliers[0].box[0]
        for row in detected:
            x,y,width,height = tuple(row.box)
            if (hx+hw < x+width/2 and x+width < multiplier_x and 16 <= width <= 40 and 16 <= height <= 40
                    and min(y+height,hy+hh)-max(y,hy) >= min(height,hh)-2):
                # 878's low-confidence "日" is white microphone ink within
                # the saturated attribute badge between a read turn heading
                # and multiplier. Ordinary text outside this layout remains.
                # The title's OCR padding can cover the badge's left edge;
                # its center must still be outside the independently read
                # heading, and the full badge must precede the multiplier.
                # Use the existing two-pixel common-ink interval vertically:
                # 1470's badge box exceeds the title by one pixel at each end.
                crop = image[y:y+height,x:x+width].astype(np.int16)
                if (np.ptp(crop,axis=2) >= 60).mean() >= .5:
                    turn_attribute_badges.add(tuple(row.box))
    lines = []
    clipped_top = clipped_bottom = False
    for row in sorted(detected, key=lambda row: (row.box[1], row.box[0])):
        raw_box = tuple(row.box)
        if len(raw_box) != 4 or any(isinstance(value, (bool, np.bool_))
                                   or not isinstance(value, (int, np.integer)) for value in raw_box):
            raise ValueError('history body line bounds are not integer pixels')
        x, y, width, height = (int(value) for value in raw_box)
        if x < 130 or y+height <= _HISTORY_BODY_TOP or y >= _HISTORY_BODY_SAFE_BOTTOM:
            continue
        if width <= 0 or height <= 0 or x+width > 675:
            raise ValueError('history body line bounds are invalid')
        if y < _HISTORY_BODY_TOP:
            clipped_top = True
            continue
        if y+height > _HISTORY_BODY_SAFE_BOTTOM:
            clipped_bottom = True
            continue
        if raw_box in turn_attribute_badges:
            continue
        # 850 OCR detects a 'C' inside the tall orange trigger icon. Exclude
        # only ink wholly left of an independently read trigger title and
        # vertically overlapping it. Ordinary effect text is not this tall.
        if raw_box in thumbnail_icon_boxes or _history_trigger_icon_row(image,row,trigger_headers):
            continue
        box = (x, y, width, height)
        text = exact_headers.get(box)
        if text is None:
            try:
                text = _history_line(context, image[max(_HISTORY_BODY_TOP,y-2):min(_HISTORY_BODY_SAFE_BOTTOM,y+height+2),
                                                  max(0,x-2):x+width+2].copy())
            except ValueError as exc:
                # 882's detector ends a partially visible numeric line at
                # y=1117, one pixel inside the OCR ROI. The crop truncates
                # "0" into "0.". 892 cuts the arrow at that same boundary.
                # Defer that edge fragment to the next page;
                # the actual bottom page must still read every complete row.
                if (not at_bottom and y+height+2 >= _HISTORY_BODY_SAFE_BOTTOM
                        and (str(exc).startswith('history line confidence is unresolved:')
                             or str(exc) == 'history numeric arrow direction is unresolved')):
                    clipped_bottom = True
                    continue
                # A tight crop can lose glyph context even when the complete
                # page already has a qualified card reference (862), or a full
                # draw sentence whose crop agrees including its digits (1461).
                try:
                    if getattr(row, 'split_reference', False):
                        raise exc
                    text = qualified_history_card_reference(row,exc,allow_draw_count=True)
                except ValueError:
                    sink = getattr(context, 'gkh_hif_history_failure_sink', None)
                    if callable(sink):
                        sink(image)
                    raise
        if not text or normalize_history_text(text) != text:
            raise ValueError('history exact body line is empty or unnormalized')
        if getattr(row, 'split_reference', False) and text != normalize_history_text(row.text):
            raise ValueError('history split reference differs from its complete-line read')
        lines.append(HistoryLine(text, box, _line_kind(text, box)))
    if not lines:
        raise ValueError('history page contains no complete body lines')
    if len({line.box for line in lines}) != len(lines):
        raise ValueError('history body line detection is duplicated')
    if at_top and clipped_top or at_bottom and clipped_bottom:
        raise ValueError('history endpoint clips a body line')
    return HistoryPage(_normalize_history_turn_rows(lines), thumb, at_top, at_bottom,
                       _history_score(context, image), clipped_top, clipped_bottom)


def history_page_consensus(pages, frame_ids):
    pages, frame_ids = tuple(pages), tuple(frame_ids)
    if (not pages or len(frame_ids) != len(pages) or len(set(frame_ids)) != len(frame_ids)
            or any(not isinstance(key, str) or not key for key in frame_ids)):
        raise ValueError('history page lacks its observed frames')
    signatures = [(tuple(line.content_key for line in page.lines),page.thumb,page.at_top,page.at_bottom,
                   page.score,page.clipped_top,page.clipped_bottom) for page in pages]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError('history page observations disagree')
    # 881 has identical high-confidence text and a stable thumb, but one
    # detector box changes from (158,668,224,25) to (158,667,223,25).
    # Use the same two-pixel common ink interval as cross-page continuity.
    for lines in zip(*(page.lines for page in pages)):
        for axis in (0,1):
            common = min(line.box[axis]+line.box[axis+2] for line in lines)-max(line.box[axis] for line in lines)
            if common < min(line.box[axis+2] for line in lines)-2:
                raise ValueError('history page line geometry changed between observations')
    return VerifiedHistoryPage(pages[0], frame_ids)


def _group_lines(lines):
    groups = []
    for line in lines:
        if line.kind != 'detail':
            groups.append(HistoryGroup(line.kind, line.text, (line.text,)))
        elif not groups:
            raise ValueError('history starts with an incomplete action group')
        else:
            previous = groups[-1]
            groups[-1] = HistoryGroup(previous.kind, previous.title,
                                      (*previous.lines, line.text))
    if any(group.kind in ('card', 'drink') and len(group.lines) < 2 for group in groups):
        raise ValueError('history action effects are missing')
    return tuple(groups)


def _join_history_prefix(pages, *, require_bottom, require_top=True):
    pages = tuple(pages)
    if not pages or require_top and not pages[0].page.at_top or require_bottom and not pages[-1].page.at_bottom:
        raise ValueError('history endpoints are incomplete')
    frames = tuple(key for page in pages for key in page.frame_ids)
    if (any(not page.frame_ids for page in pages)
            or len(set(frames)) != len(frames)
            or any(not isinstance(key, str) or not key for key in frames)):
        raise ValueError('history pages reuse or lack independent frames')
    if require_top and pages[0].page.clipped_top or require_bottom and pages[-1].page.clipped_bottom:
        raise ValueError('history endpoints contain partial lines')
    score = pages[0].page.score
    if type(score) is not int or score < 0 or any(page.page.score != score for page in pages):
        raise ValueError('history score changed between pages')
    combined, overlaps = list(pages[0].page.lines), []
    reference_thumb_length = pages[0].page.thumb[1]-pages[0].page.thumb[0]
    for previous, current in zip(pages, pages[1:]):
        old, new = previous.page, current.page
        if (old.at_bottom or new.at_top or new.thumb[0] <= old.thumb[0]
                or abs(new.thumb[1]-new.thumb[0]-reference_thumb_length) > 2):
            raise ValueError('history scrollbar changed size or did not advance')
        old_keys, new_keys = ([line.content_key for line in page.lines] for page in (old,new))
        matches = [size for size in range(1,min(len(old_keys),len(new_keys))+1)
                   if old_keys[-size:] == new_keys[:size]]
        if len(matches) > 1:
            # 879 repeats four identical memory grants. Their text alone
            # permits 1/3/5/7-line overlaps; the observed thumb displacement
            # independently selects the matching content translation.
            lengths = [page.thumb[1]-page.thumb[0]+1 for page in (old,new)]
            delta = new.thumb[0]-old.thumb[0]
            if min(lengths) <= 2:
                raise ValueError('history scrollbar translation is unresolved')
            # Retain two-pixel thumb-edge uncertainty and the existing
            # two-pixel ink-interval tolerance; never rank overlap sizes.
            low = (delta-2)*_HISTORY_VIEWPORT_HEIGHT/(max(lengths)+2)-2
            high = (delta+2)*_HISTORY_VIEWPORT_HEIGHT/(min(lengths)-2)+2
            matches = [count for count in matches if low <= float(np.median([
                left.box[1]+left.box[3]/2-right.box[1]-right.box[3]/2
                for left,right in zip(old.lines[-count:],new.lines[:count])])) <= high]
        if len(matches) != 1:
            raise ValueError('history overlap is missing or ambiguous')
        count = matches[0]
        shifts = []
        for left, right in zip(old.lines[-count:], new.lines[:count]):
            x, y, width, height = left.box
            nx, ny, new_width, new_height = right.box
            # 859 has identical text but OCR estimates different glyph-box
            # widths/heights after scrolling. Position, unique text overlap
            # and the independently stable pages establish continuity.
            overlap = min(x+width,nx+new_width)-max(x,nx)
            if overlap < min(width,new_width)-2 or y <= ny:
                raise ValueError('history overlap geometry is inconsistent')
            shifts.append(y+height/2-ny-new_height/2)
        # 863 shows a shrinking title box and an old bottom-edge box whose
        # height is truncated. Compare shared ink intervals after one common
        # translation, not the detector's variable top/left edges. At most two
        # pixels of the smaller interval may lie outside the other interval.
        shift = float(np.median(shifts))
        for left, right in zip(old.lines[-count:], new.lines[:count]):
            y,height = left.box[1],left.box[3]
            ny,new_height = right.box[1]+shift,right.box[3]
            overlap = min(y+height,ny+new_height)-max(y,ny)
            if overlap < min(height,new_height)-2:
                raise ValueError('history overlap lines moved inconsistently')
        if count == len(new.lines):
            raise ValueError('history page adds no observed content')
        overlaps.append(count)
        combined.extend(new.lines[count:])
    return combined, frames, tuple(overlaps), score


def join_history_pages(pages):
    """Join only continuous, independently observed top-to-bottom pages."""
    combined, frames, overlaps, score = _join_history_prefix(pages, require_bottom=True)
    groups = _group_lines(combined)
    turns = tuple(int(group.title.removesuffix('ターン目')) for group in groups
                  if group.kind == 'turn-start')
    if turns and turns != tuple(range(turns[0], turns[-1]+1)):
        raise ValueError('history turn sequence repeats or skips a turn')
    first_turn_index = next((i for i,g in enumerate(groups) if g.kind == 'turn-start'), None)
    first_turn_observed = (bool(turns) and turns[0] == 1 and first_turn_index is not None
                           and all(g.kind == 'system' for g in groups[:first_turn_index]))
    return HistoryChain(groups, score, frames, overlaps, turns, first_turn_observed)


def _history_seek_progress(previous_thumb, thumb):
    """Classify seek-only progress; a shorter thumb can expose older log rows."""
    previous_length = previous_thumb[1]-previous_thumb[0]
    length = thumb[1]-thumb[0]
    # 877 loaded earlier records while seeking upward: 288..604 became
    # 464..707. The stable thumb shrank by 73 pixels and its position reset.
    # Formal top-to-bottom reading still requires one fixed content length.
    if length < previous_length-2:
        return 'content_grew'
    if thumb[0] < previous_thumb[0] and abs(length-previous_length) <= 2:
        return 'toward_top'
    raise ValueError('history did not advance toward its top')


def observe_complete_history(context, *, capture_frames, checkpoint, frame_identifier,
                             top_thumb_y=264, maximum_age_ms=1000, max_pages=12,
                             max_seconds=None, drag_to_top=False, page_scroll_pixels=240):
    """Observe the whole current log through audited Maa navigation only.

    No action is submitted. Each input uses fresh geometry and permission
    checks after the slower three-frame semantic observation has finished.
    """
    from utils.maa_input import verify_maa_input_endpoints

    if type(maximum_age_ms) is not int or not 0 < maximum_age_ms <= 1000:
        raise ValueError('history input evidence cannot exceed 1000 milliseconds')
    if (type(drag_to_top) is not bool or type(max_pages) is not int or not 1 <= max_pages <= 48
            or max_seconds is not None and (type(max_seconds) is not int or not 1 <= max_seconds <= 180)
            or type(page_scroll_pixels) is not int or page_scroll_pixels not in (240,400)):
        raise ValueError('history observation budget is invalid')
    controller = context.tasker.controller
    event = dict(accepted=False,event='live_complete_history_observe',
                 input_click_count=0,scroll_count=0,inputs=[],pages=[],seeking_pages=[])
    started = time.perf_counter()
    before = None
    closing = False
    top_drags = 0

    def budget():
        if not closing and max_seconds is not None and time.perf_counter()-started >= max_seconds:
            raise ValueError('history complete observation time budget reached')

    def fresh():
        budget()
        image = controller.post_screencap().wait().get()
        return image, frame_captured_at(controller, image)

    def dispatch(captured_at, phase, box, *, direction=None, thumb=None):
        nonlocal top_drags
        budget()
        if (drag_to_top and direction == 'toward_top' and thumb is not None
                and thumb[0] > 271 and top_drags >= 3):
            raise ValueError('history scrollbar drag limit reached before top')
        endpoints = verify_maa_input_endpoints(controller.info,
            expected_client_size=getattr(context,'gkh_hif_client_size',(720,1280)))
        if (time.perf_counter()-captured_at)*1000 > maximum_age_ms:
            raise ValueError('history navigation snapshot expired')
        event['input_click_count'] += 1
        record = dict(phase=phase,box=box,endpoints=endpoints)
        event['inputs'].append(record)
        if direction is None:
            if phase == 'open':
                event['open_dispatched'] = True
            x,y,width,height = box
            result = controller.post_click(x+width//2,y+height//2)
        else:
            event['scroll_count'] += 1
            offset = 240 if direction == 'toward_top' else -240
            swipe = [560,610,560,610+offset,500]
            if direction != 'toward_top' and page_scroll_pixels == 400:
                # Reuse the diagnostic reader's observed 1467/1468 gesture.
                # Each resulting page must still pass three-frame consensus
                # and the same independently checked overlap before proceeding.
                swipe = [560,1000,560,600,650]
            if drag_to_top and direction == 'toward_top' and thumb is not None and thumb[0] > 271:
                # Reuse the diagnostic long-log reader's visible scrollbar
                # drag. At the loaded top, retain the normal upward pull which
                # reveals older records; a drag alone does not prove completeness.
                top_drags += 1
                swipe = [685,(thumb[0]+thumb[1])//2,685,264,650]
            record.update(direction=direction,swipe=swipe)
            result = controller.post_swipe(*swipe)
        if not result.wait().succeeded:
            raise ValueError('history Maa navigation input failed')
        # 860 showed a sub-pixel inertia tail after an apparently identical
        # three-frame thumb (658..994 -> 657..993). Let that animation finish
        # before observing; navigation still takes separate fresh input frames.
        time.sleep(2.5 if direction is not None else .5)

    def overlay_input(phase, *, expected_thumb=None, direction=None):
        first,_ = fresh()
        first_box = history_navigation_target(context,first,True)
        if expected_thumb is not None:
            thumb,_,_ = history_chain_geometry(first,top_thumb_y=top_thumb_y,
                allow_top_candidate=top_thumb_y is None)
            if any(abs(a-b) > 2 for a,b in zip(thumb,expected_thumb)):
                raise ValueError('history moved before navigation')
        second,stamp = fresh()
        box = history_navigation_target(context,second,True)
        if box != first_box:
            raise ValueError('history navigation control changed between frames')
        if expected_thumb is not None:
            thumb,_,_ = history_chain_geometry(second,top_thumb_y=top_thumb_y,
                allow_top_candidate=top_thumb_y is None)
            if any(abs(a-b) > 2 for a,b in zip(thumb,expected_thumb)):
                raise ValueError('history moved before navigation')
        dispatch(stamp,phase,box,direction=direction,thumb=thumb if expected_thumb is not None else None)

    def stable_geometry(*, phase='seek', page=None):
        deadline = time.perf_counter()+4
        for attempt in range(9):
            budget()
            frames = tuple(capture_frames())
            ids = tuple(frame_identifier(frame) for frame in frames)
            observations = []
            try:
                if not frames or len(set(ids)) != len(frames):
                    raise ValueError('history geometry lacks its observed frames')
                for frame in frames:
                    observations.append(history_chain_geometry(frame,top_thumb_y=top_thumb_y,
                        allow_top_candidate=top_thumb_y is None))
                if any(row != observations[0] for row in observations[1:]):
                    raise ValueError('history has not stopped scrolling')
                history_navigation_target(context,frames[-1],True)
                event['seeking_pages'].append(dict(
                    phase=phase,page=page,frame_ids=ids,geometry=observations[0]))
                return observations[0],frames,ids
            except ValueError as error:
                event.setdefault('reobservations',[]).append(dict(
                    phase=phase,page=page,attempt=attempt,reason=str(error),
                    frame_ids=ids,geometries=observations))
                remaining = deadline-time.perf_counter()
                if attempt == 8 or remaining <= 0:
                    raise
                time.sleep(min(.5,remaining))

    try:
        frames,before = capture_initial_live_checkpoint(capture_frames,checkpoint,event)
        initial_box = history_navigation_target(context,frames[-1],False)
        image,stamp,_ = capture_fresh_live_checkpoint(
            fresh,capture_frames,checkpoint,before,event)
        box = history_navigation_target(context,image,False)
        if box != initial_box:
            raise ValueError('history opening control changed between frames')
        dispatch(stamp,'open',box)
        previous_thumb = None
        top_confirmation_used = False
        confirming_top = False
        for index in range(max_pages+1):
            (thumb,at_top,_),_,ids = stable_geometry()
            # 893 landed exactly at the currently loaded top (263..580),
            # before the upward pull which exposed 892's older 243px-thumb
            # content. Confirm that boundary once; never loop on a static top.
            if confirming_top and at_top and thumb == previous_thumb:
                event['top_confirmation'].update(result='stable_top',frame_ids=ids,thumb=thumb)
                break
            progress = _history_seek_progress(previous_thumb,thumb) if previous_thumb is not None else None
            if confirming_top:
                if progress != 'content_grew':
                    raise ValueError('history top confirmation moved without exposing older content')
                event['top_confirmation'].update(result='content_grew',frame_ids=ids,thumb=thumb)
                confirming_top = False
            if progress == 'content_grew':
                event.setdefault('seeking_growth_events',[]).append(dict(
                    seek_index=index,frame_ids=ids,previous_thumb=previous_thumb,thumb=thumb,
                    previous_length=previous_thumb[1]-previous_thumb[0],length=thumb[1]-thumb[0]))
            if at_top:
                if top_confirmation_used:
                    break
                if index == max_pages:
                    raise ValueError('history top confirmation exceeds seeking page budget')
                top_confirmation_used = confirming_top = True
                event['top_confirmation'] = dict(previous_thumb=thumb,source_frame_ids=ids)
            if index == max_pages:
                raise ValueError('history top seeking page budget reached')
            overlay_input('scroll',expected_thumb=thumb,direction='toward_top')
            previous_thumb = thumb
        if top_thumb_y is None:
            if not top_confirmation_used or not 262 <= thumb[0] <= 271:
                raise ValueError('history observed top lacks boundary confirmation')
            top_thumb_y = thumb[0]
            event['observed_top_calibration'] = dict(top_thumb_y=top_thumb_y, frame_ids=ids)
        pages = []
        for index in range(max_pages):
            for attempt in range(3):
                # Settle first and read those exact frames, so movement never
                # starts a full page of expensive OCR or another swipe.
                _,frames,ids = stable_geometry(phase='read',page=index)
                try:
                    observations = [read_history_page(context,frame,top_thumb_y=top_thumb_y)
                                    for frame in frames]
                    budget()
                    page = history_page_consensus(observations,ids)
                    _join_history_prefix([*pages,page],require_bottom=False)
                    break
                except ValueError as error:
                    event.setdefault('reobservations',[]).append(dict(
                        phase='read',page=index,attempt=attempt,reason=str(error),frame_ids=ids))
                    if attempt == 2:
                        raise
                    time.sleep(.5)
            pages.append(page)
            event['pages'].append(asdict(page))
            if page.page.at_bottom:
                break
            if index == max_pages-1:
                raise ValueError('history complete reading page budget reached')
            overlay_input('scroll',expected_thumb=page.page.thumb,direction='toward_bottom')
        chain = join_history_pages(pages)
        event.update(observation=chain.as_observation(),frame_ids=chain.frame_ids,
                     complete_from_start=chain.complete_from_start,
                     complete_to_end=chain.complete_to_end)
    except (ValueError,RuntimeError) as error:
        event['reason'] = str(error)
    finally:
        closing = True
        if event.get('open_dispatched'):
            try:
                # A failed page read does not block closing an independently
                # recognized overlay; no guessed close coordinate is used.
                overlay_input('close')
                _,after = capture_initial_live_checkpoint(capture_frames,checkpoint,event)
                if before is None or any(before[key] != after[key] for key in ('families','fields')):
                    raise ValueError('Live changed during complete history inspection')
                event.update(returned_unchanged=True,
                    before={key:before[key] for key in ('families','fields','frame_ids')},
                    after={key:after[key] for key in ('families','fields','frame_ids')})
            except (ValueError,RuntimeError) as error:
                event['close_failure'] = str(error)
                event.setdefault('reason',str(error))
        event['elapsed_ms'] = (time.perf_counter()-started)*1000
        if max_seconds is not None and event['elapsed_ms'] >= max_seconds*1000:
            event.setdefault('reason','history complete observation time budget reached')
    event['accepted'] = bool('reason' not in event and event.get('returned_unchanged')
                             and event.get('complete_from_start') and event.get('complete_to_end'))
    return event
