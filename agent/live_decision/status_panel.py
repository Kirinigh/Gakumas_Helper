"""Read HIF status rows and resolve known memory counts with audited navigation.

The list has several rows with the same heading (memory abilities). Preserve
their descriptions and source order rather than collapsing them by heading.
This visual layer does not resolve rule identities or initialize schedules.
"""

from __future__ import annotations

import re
import json
import math
import time
import unicodedata
from copy import deepcopy
from types import SimpleNamespace
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .hif_capture_adapter import captured_at as frame_captured_at


def _normalize_status_text(text):
    # CJK glyph equivalences; 1586 independently reread 变化 as 変化.
    # Do not fold kana sizes or infer a catalogue name from similar text.
    normalized = ''.join(unicodedata.normalize('NFKC',text).split()).translate(
        str.maketrans({'增':'増','絕':'絶','变':'変','轨':'軌'})).replace('消費体カ','消費体力')
    # A duplicated OCR unit (15%％) has one amount; keep every digit and all
    # surrounding wording for the original independent-read comparisons.
    return re.sub(r'(?<=\d)%{2,}', '%', normalized)


_STATIC_STATUS_HEADINGS = {
    '集中':'focus', '好調':'good', '絶好調':'excellent',
    'メモリーアビリティ':'memory', 'スキルカード使用数追加':'extra_action',
}

# 1670 independently read this standard heading at .980240 with the existing
# complete 8px border. It has no static template and supplies no numeric value.
_COMPLETE_STATUS_HEADINGS = _STATIC_STATUS_HEADINGS.keys() | {'消費体力削減'}

_COUNTLESS_STATUS_DESCRIPTIONS = {
    '天真爛漫': 'ターン終了時、集中が3以上の場合、集中+2',
    '厳選初星マキアート': 'ターン終了時、集中+1',
    '至高のエンタメ+': 'アクティブスキルカード使用時、スコア+5',
    '願いの力': 'アクティブスキルカード使用時、集中+1',
    '自己肯定感爆上げ↑↑+': 'メンタルスキルカード使用時、集中+2',
}


def _static_heading_confidence(context,image,row,boxes,index):
    title = _normalize_status_text(row['text'])
    key = _STATIC_STATUS_HEADINGS.get(title)
    if key is None or not any(_inside(row,(x+90,y+8,w-110,90)) for x,y,w,h in boxes):
        return None
    x,y,w,h = row['box']
    yy,xx = np.where(image[y:y+h,x:x+w].min(axis=2) < 235)
    if not len(xx):
        return None
    ink = (x+int(xx.min()),y+int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))
    node = f'LiveStatusPanelHeading{index}'
    hit = context.run_recognition(node,image,pipeline_override={node: {
        'recognition':'TemplateMatch','template':[f'produce/hif_status_heading_{key}.png'],
        'roi':[x-6,y-6,w+12,h+12],'threshold':.95,
    }})
    found = list(hit.filtered_results) if hit and hit.hit else []
    if len(found) != 1 or not math.isfinite(found[0].score) or found[0].score < .95:
        return None
    tx,ty,tw,th = found[0].box
    # Require the entire glyph extent, so 好調 inside 絶好調 cannot match.
    if (abs(tx+2-ink[0]) > 3 or abs(ty+2-ink[1]) > 3
            or abs(tw-4-ink[2]) > 4 or abs(th-4-ink[3]) > 4):
        return None
    return float(found[0].score)


@dataclass(frozen=True)
class StatusPanelRow:
    box: tuple[int, int, int, int]
    title: str
    count: int | None
    unit: str
    description: str
    confidence: float
    remaining_turns: int | None = None
    heading_source: str = 'ocr'
    per_turn_limit: int | None = None
    heading_observations: tuple[tuple[str, float], ...] = ()
    detail_confidence: float | None = None

    @property
    def content_key(self):
        description = self.description
        if (self.title == 'パラメータ上昇量増加' and self.unit == '%'
                and type(self.count) is int and self.count > 0
                and type(self.remaining_turns) is int and self.remaining_turns > 0):
            # Native 1616: the complete supported clause can attach its opening
            # parenthesis to the adjacent icon or text line. Preserve raw OCR;
            # only this exact, amount-matched clause shares a semantic key.
            clause = re.fullmatch(r'スコア上昇量を([1-9]\d*)%増加\(?好印象による上昇も含む\)', description)
            if clause and int(clause[1]) == self.count:
                description = description.replace('増加(', '増加')
        base=(self.title, self.count, self.unit, description, self.remaining_turns)
        return base if self.per_turn_limit is None else (*base,self.per_turn_limit)


def _corroborated_heading(title, confidence, observations):
    """Two complete-crop readings identify text, never an effect or its value."""
    if (not isinstance(title, str) or not title or re.search(r'\d', title)
            or type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not .8 <= confidence <= 1
            or not isinstance(observations, (tuple, list)) or len(observations) != 2):
        return False
    for reading in observations:
        if (not isinstance(reading, (tuple, list)) or len(reading) != 2
                or not isinstance(reading[0], str)
                or _normalize_status_text(reading[0]).removesuffix('おすすめ効果') != title
                or type(reading[1]) not in (int, float) or not math.isfinite(reading[1])
                or not .8 <= reading[1] <= 1):
            return False
    return confidence == min(reading[1] for reading in observations)


def status_row_confidence_valid(row):
    """Keep observed confidence; weak headings cannot exempt numbers/body text."""
    if type(row.confidence) not in (int, float) or not math.isfinite(row.confidence):
        return False
    if row.heading_source != 'corroborated_complete_heading':
        return .95 <= row.confidence <= 1
    return (type(row.detail_confidence) in (int, float)
            and math.isfinite(row.detail_confidence) and .95 <= row.detail_confidence <= 1
            and _corroborated_heading(row.title, row.confidence, row.heading_observations))


@dataclass(frozen=True)
class StatusPanelPage:
    rows: tuple[StatusPanelRow, ...]
    thumb: tuple[int, int]
    at_top: bool
    at_bottom: bool
    partial_row_count: int

    @property
    def content_key(self):
        return tuple(row.content_key for row in self.rows), self.at_top, self.at_bottom


def status_panel_geometry(image, *, require_complete_layout=True, row_white_min=250, allow_unscrolled=False):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('status panel resolution is uncalibrated')
    strip = image[30:738, 695].astype(np.int16)
    # The thumb is a long, uniform solid interior; scene tint can move its
    # channels apart by more than twelve. The track follows the background.
    # Require a repeated color AND a contiguous, horizontally solid strip,
    # rather than accepting every approximately gray background pixel.
    eligible = strip[(np.ptp(strip,axis=1) <= 32) & (strip.min(axis=1) >= 90)
                     & (strip.max(axis=1) <= 180)]
    colors,counts = np.unique(eligible,axis=0,return_counts=True)
    spans = []
    # At the smaller client raster x=697 is the blended right edge. The
    # three inner columns remain solid along the original full thumb span.
    interior = image[30:738,694:697].astype(np.int16)
    for color,count in zip(colors,counts):
        if count < 30:
            continue
        solid = np.max(np.abs(interior-color),axis=(1,2)) <= 2
        edges = np.diff(np.r_[False,solid,False].astype(np.int8))
        for a,b in zip(np.where(edges == 1)[0],np.where(edges == -1)[0]):
            span = (int(a+30),int(b+30))
            if b-a >= 30 and not any(max(abs(x-y) for x,y in zip(span,old)) <= 3 for old in spans):
                spans.append(span)
    if len(spans) != 1 and (spans or not allow_unscrolled):
        raise ValueError('status panel scrollbar is not unique')
    if row_white_min not in (235, 250):
        raise ValueError('panel white-row profile is uncalibrated')
    # Both item and status rows are opaque white. Native 930 confirms that
    # the stricter existing 250 profile excludes animated pale background
    # attached to status borders; 235 remains available for explicit replay.
    mask = (image[20:738,39:681].min(axis=2) >= row_white_min).astype(np.uint8)
    if row_white_min == 235:
        # A bright scene band can join adjacent opaque rows through their
        # central gap. Both blank side margins and complete white borders
        # independently identify the existing 7--16px row separator.
        gaps = ~mask[:,:20].any(axis=1) & ~mask[:,-20:].any(axis=1)
        edges = np.diff(np.r_[False,gaps,False].astype(np.int8))
        for a,b in zip(np.where(edges == 1)[0],np.where(edges == -1)[0]):
            if (7 <= b-a <= 16 and a >= 3 and b+3 <= len(mask)
                    and mask[a:b].any(axis=1).all()
                    and mask[a-3:a].sum(axis=1).max() >= 625
                    and mask[b:b+3].sum(axis=1).max() >= 625):
                mask[a:b] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for label,(x,y,w,h,area) in enumerate(stats[1:],1):
        if not (625 <= w <= 642 and h >= 40):
            continue
        if area/(w*h) < .80 and row_white_min == 250 and y > 0 and y+h < len(mask):
            # Dense text/art are enclosed holes in the same opaque card.
            # Preserve open edge notches (including the close-button shadow)
            # and all original bounds; do not repair or erase source pixels.
            component = labels[y:y+h,x:x+w] == label
            top = component[:20,20:-20].all(axis=1).any()
            sides = (component[20:-20,:20].all(axis=0).any()
                     and component[20:-20,-20:].all(axis=0).any())
            bottom = component[-20:,20:-20].all(axis=1).any()
            if not bottom and 718 <= y+20+h <= 728:
                # The calibrated close ROI is x=300..420. Require intact
                # bottom rails on BOTH sides; its open shadow is never filled.
                left, right = 300-(x+39),420-(x+39)
                bottom = (component[-20:,20:left].all(axis=1).any()
                          and component[-20:,right:-20].all(axis=1).any())
            if top and sides and bottom:
                exterior = np.pad(component.astype(np.uint8),1)
                cv2.floodFill(exterior,None,(0,0),1)
                area = np.count_nonzero(component | (exterior[1:-1,1:-1] == 0))
        if area/(w*h) >= .80:
            boxes.append((int(x+39),int(y+20),int(w),int(h)))
    boxes.sort(key=lambda box: box[1])
    complete = tuple(box for box in boxes if box[1] > 20 and box[1]+box[3] < 738)
    if not complete and (require_complete_layout or not boxes):
        raise ValueError('status panel has no complete readable rows')
    def spacing_valid(a,b):
        clipped_edge = a[1] == 20 or b[1]+b[3] == 738
        lower,upper = (0,24) if clipped_edge else (7,16)
        return lower <= b[1]-a[1]-a[3] <= upper
    if require_complete_layout and any(not spacing_valid(a,b) for a,b in zip(boxes,boxes[1:])):
        raise ValueError(f'status panel contains an unobserved row or inconsistent spacing: {boxes}')
    if not spans:
        if len(complete) != len(boxes) or not 40 <= boxes[0][1] <= 48 or boxes[-1][1] + boxes[-1][3] > 708:
            raise ValueError('unscrolled status panel rows are incomplete')
        return complete, (0, 0), True, True, 0
    top, bottom = spans[0]
    if require_complete_layout and ((30 <= top <= 36 and not 40 <= boxes[0][1] <= 48)
            or (731 <= bottom <= 736 and not 718 <= boxes[-1][1]+boxes[-1][3] <= 728)):
        raise ValueError('status panel endpoint rows are missing')
    return complete, (top,bottom), 30 <= top <= 36, 731 <= bottom <= 736, len(boxes)-len(complete)


def _inside(row, roi):
    x,y,w,h = roi
    rx,ry,rw,rh = row['box']
    return x <= rx and y <= ry and rx+rw <= x+w and ry+rh <= y+h


def _join_lines(rows):
    lines = []
    for row in sorted(rows, key=lambda value: (value['box'][1]+value['box'][3]/2, value['box'][0])):
        center = row['box'][1]+row['box'][3]/2
        if lines and abs(center-lines[-1][0]) <= 12:
            lines[-1][1].append(row)
        else:
            lines.append((center,[row]))
    return ''.join(''.join(row['text'] for row in sorted(line, key=lambda value: value['box'][0]))
                   for _,line in lines)


def parse_status_panel(image, ocr_rows, *, minimum_confidence=.95):
    boxes, thumb, at_top, at_bottom, partial = status_panel_geometry(image)
    if (type(minimum_confidence) not in (int,float) or not math.isfinite(minimum_confidence)
            or not .95 <= minimum_confidence <= 1):
        raise ValueError('status panel confidence is invalid')
    normalized = []
    for raw in ocr_rows:
        row = raw if isinstance(raw, dict) else dict(text=raw.text,score=raw.score,box=raw.box)
        box = row.get('box')
        if (not isinstance(box,(list,tuple)) or len(box) != 4 or
                any(type(value) is not int for value in box) or min(box[2:]) <= 0):
            raise ValueError('status OCR geometry is invalid')
        normalized.append(dict(row,text=_normalize_status_text(row['text']))
                          if isinstance(row.get('text'),str) else row)
    output = []
    for box in boxes:
        x,y,w,h = box
        # Header text and numerical count exclude the pictogram on the left.
        header = [row for row in normalized if _inside(row,(x+90,y+8,w-110,90))]
        # Descriptions begin below the fixed separator. Tiny decorative stars
        # next to the separator are outside this region and never become text.
        body = [row for row in normalized if _inside(row,(x+15,y+110,w-42,h-120))]
        text_rows = header + body
        if not header or not body:
            raise ValueError('status row text is incomplete')
        for row in text_rows:
            corroborated = (row in header and isinstance(row.get('text'), str)
                and row.get('heading_source') == 'corroborated_complete_heading'
                and _corroborated_heading(row.get('text', '').removesuffix('おすすめ効果'),
                    row.get('score'), row.get('heading_observations')))
            if (type(row.get('score')) not in (int, float) or not math.isfinite(row['score'])
                    or not 0 <= row['score'] <= 1 or not isinstance(row.get('text'), str)
                    or (row['score'] < minimum_confidence and not corroborated)):
                raise ValueError('status row text has insufficient confidence')
        count_pattern = r'(\d+)(ターン|回|%)?(?:[|｜]?(\d+)ターン)?'
        limit_pattern = r'(\d+)回ターン内(\d+)回(\d+)ターン'
        counts = [row for row in header if re.fullmatch(count_pattern, row['text'])
                  or re.fullmatch(limit_pattern,row['text'])]
        titles = [row for row in header if row not in counts]
        # OCR can split the recommendation badge from the status heading.
        # Accept only the exact badge beside one heading on the same line;
        # an extra heading, repeated badge or displaced text stays ambiguous.
        badges = [row for row in titles if row['text'] == 'おすすめ効果']
        if len(titles) == 2 and len(badges) == 1:
            badge = badges[0]
            heading = next(row for row in titles if row is not badge)
            hx,hy,hw,hh = heading['box']
            bx,by,bw,bh = badge['box']
            if (-min(hh,bh)/2 <= bx-hx-hw <= 24 and abs(by+bh/2-hy-hh/2) <= 12
                    and not heading['text'].endswith('おすすめ効果')):
                titles = [heading]
        if len(counts) == 2:
            left,right = sorted(counts,key=lambda row:row['box'][0])
            lx,ly,lw,lh = left['box']
            rx,ry,rw,rh = right['box']
            fixed_addition = len(titles) == 1 and titles[0]['text'].removesuffix('おすすめ効果') == '集中増加量追加'
            score_percentage = (len(titles) == 1
                and titles[0]['text'].removesuffix('おすすめ効果') == 'パラメータ上昇量増加')
            if ((re.fullmatch(r'\d+回',left['text']) or (fixed_addition and re.fullmatch(r'\d+',left['text']))
                    or (score_percentage and re.fullmatch(r'\d+%',left['text'])))
                    and re.fullmatch(r'\d+ターン',right['text'])
                    and -min(lh,rh)/2 <= rx-lx-lw <= 36 and abs(ry+rh/2-ly-lh/2) <= 12):
                counts = [dict(left,text=left['text']+'|'+right['text'])]
        if not counts and len(titles) == 1:
            # Native 910/1016/1049/1184 show these effects without a counter.
            # Require their full independently read wording
            # and an actually blank counter strip, so missed digits cannot
            # be interpreted as an unlimited duration or an invented use.
            title = titles[0]['text']
            description = _join_lines(body)
            counter_strip = image[y+57:y+98, x+90:x+w-20]
            if (_COUNTLESS_STATUS_DESCRIPTIONS.get(title) == description
                    and counter_strip.shape == (41, w-110, 3)
                    and counter_strip.min() >= 235):
                output.append(StatusPanelRow(box, title, None, '', description,
                    min(row['score'] for row in text_rows),
                    heading_source=titles[0].get('heading_source', 'ocr'),
                    heading_observations=titles[0].get('heading_observations', ()),
                    detail_confidence=min(row['score'] for row in body)
                    if titles[0].get('heading_observations') else None))
                continue
        if len(counts) != 1 or len(titles) != 1:
            raise ValueError('status row heading or count is not unique')
        title = titles[0]['text'].removesuffix('おすすめ効果')
        limit_match=re.fullmatch(limit_pattern,counts[0]['text'])
        per_turn_limit=int(limit_match[2]) if limit_match else None
        if limit_match:
            counts=[dict(counts[0],text=limit_match[1]+'回|'+limit_match[3]+'ターン')]
        match = re.fullmatch(count_pattern, counts[0]['text'])
        if title == '集中増加量追加' and match[2] == 'ターン' and match[3] is None:
            # Mixed-size amount/duration can be detected as one OCR line.
            # Require the independent complete description to split it;
            # never infer either number from the predicted game state.
            description = re.fullmatch(r'集中の増加量を(\d+)追加(\d+)ターン', _join_lines(body))
            if description is None or match[1] != description[1] + description[2]:
                raise ValueError('fixed addition compound counter is unresolved')
            match = re.fullmatch(count_pattern, description[1] + '|' + description[2] + 'ターン')
        score_percentage = title == 'パラメータ上昇量増加'
        if score_percentage and not (match[2] == '%' and match[3] is not None
                                     and int(match[1]) > 0 and int(match[3]) > 0):
            raise ValueError('score percentage requires its amount, unit and finite duration')
        if (match[3] is not None and match[2] != '回'
                and not (title == '集中増加量追加' and match[2] is None)
                and not score_percentage):
            raise ValueError('status compound counter must give uses and duration')
        if titles[0]['box'][1] >= counts[0]['box'][1] or not title:
            raise ValueError('status row heading and count are inconsistent')
        output.append(StatusPanelRow(box,title,int(match[1]),match[2] or '',
                                     _join_lines(body),min(row['score'] for row in text_rows),
                                     int(match[3]) if match[3] is not None else None,
                                     titles[0].get('heading_source','ocr'),per_turn_limit,
                                     titles[0].get('heading_observations', ()),
                                     min(row['score'] for row in text_rows if row is not titles[0])
                                     if titles[0].get('heading_observations') else None))
    return StatusPanelPage(tuple(output),thumb,at_top,at_bottom,partial)


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


def _unchanged_status_scroll(previous, observations, geometries, completed_scroll, frame_ids):
    """Prove a completed navigation left the same complete, non-bottom page.

    The anchor is the fresh image immediately before that swipe, not an older
    OCR frame that may still have been coasting. This proves no visible advance;
    it does not identify why the first input had no visible effect.
    """
    if (completed_scroll is None or not previous.rows or previous.at_bottom
            or not observations or len(observations) != len(frame_ids) or len(geometries) != len(frame_ids)
            or len(set(frame_ids)) != len(frame_ids)
            or not all(isinstance(value,str) and value for value in frame_ids)
            or not _status_geometry_stable(geometries)
            or any(page.content_key != previous.content_key or page.at_bottom
                   or page.at_top != previous.at_top for page in observations)
            or completed_scroll['content_key'] != previous.content_key
            or not completed_scroll['completed']):
        return None
    original = completed_scroll['geometry']
    if original[3]:
        return None
    for current in geometries:
        reference = original
        if (completed_scroll.get('restored') and original[2] and current[2]
                and all(30 <= item[1][0] <= 36 for item in (original,current))
                and abs(original[1][0]-current[1][0]) <= 2):
            # 0033: both complete views independently locate the top endpoint.
            # Its first solid pixel moved by two after client resampling; keep
            # the opposite scrollbar edge and every complete row in the check.
            reference = (original[0],(current[1][0],original[1][1]),*original[2:])
        if not _status_geometry_stable((reference,current)):
            return None
    for page, geometry in zip(observations, geometries):
        parsed = (tuple(row.box for row in page.rows),page.thumb,
                  page.at_top,page.at_bottom,page.partial_row_count)
        if not _status_geometry_stable((parsed,geometry)):
            return None
    return dict(reason='completed scroll without visible page advance',
        frame_ids=list(frame_ids),original_input_index=completed_scroll['input_index'],
        original_geometry=original,observed_geometries=geometries,
        scroll_distance=completed_scroll['distance'],failed_batch_accepted=False)


def _read_short_status_line(context, crop, original_text, index):
    """Independently read separated glyphs; never infer a missing character."""
    if not (16<=crop.shape[0]<=48 and 12<=crop.shape[1]<=180):
        return None
    ink=crop.min(axis=2)<220
    edges=np.diff(np.r_[False,ink.any(axis=0),False].astype(np.int8))
    spans=list(zip(np.where(edges==1)[0],np.where(edges==-1)[0]))
    if (not 2<=len(spans)<=6 or spans[0][0]<2 or spans[-1][1]>crop.shape[1]-2
            or any(not 6<=right-left<=40 for left,right in spans)):
        return None
    expected = _normalize_status_text(original_text)
    duration = re.fullmatch(r'(\d+)ターン', expected)
    tokens = list(expected)
    if duration and len(spans) == len(duration[1])+3:
        # The numeral and its smaller unit are printed at different sizes.
        # Read both complete runs, using their visible inter-glyph gap.
        tokens = [duration[1], 'ターン']
        spans = [(spans[0][0],spans[-4][1]), (spans[-3][0],spans[-1][1])]
    if len(tokens) != len(spans):
        return None
    text=[]
    scores=[]
    for part,(left,right) in enumerate(spans):
        # Detector height belongs to the whole line, not each glyph. 1627's
        # complete ')' retained large vertical margins and failed recognition.
        # Keep every visible glyph pixel, prove blank top/bottom margins, then
        # restore the same two-pixel border already used horizontally.
        vertical=np.flatnonzero((crop[:,left:right].min(axis=2)<235).any(axis=1))
        if not len(vertical) or vertical[0]<2 or vertical[-1]>=crop.shape[0]-2:
            return None
        glyph=cv2.copyMakeBorder(crop[vertical[0]:vertical[-1]+1,left:right],
                                2,2,2,2,cv2.BORDER_CONSTANT,value=(255,255,255))
        node=f'LiveStatusPanelGlyph{index}_{part}'
        hit=context.run_recognition(node,glyph,pipeline_override={node:dict(
            recognition='OCR',only_rec=True,expected='.+')})
        results=list(hit.all_results) if hit and hit.hit else []
        if (len(results)!=1 or not math.isfinite(results[0].score) or results[0].score<.95
                or _normalize_status_text(results[0].text) != tokens[part]):
            return None
        text.append(_normalize_status_text(results[0].text))
        scores.append(results[0].score)
    joined=''.join(text)
    if joined!=_normalize_status_text(original_text):
        return None
    return SimpleNamespace(text=joined,score=min(scores))


def _wrapped_tail_has_single_middle_dot(crop):
    """Separate one neutral centered dot from a colon or adjacent icon ink."""
    ink = crop.min(2) < 220
    yy,xx = np.where(ink)
    visible = crop.min(2) < 235
    if (not len(xx) or any(edge.any() for edge in
            (visible[0],visible[-1],visible[:,0],visible[:,-1]))):
        return False
    edges = np.diff(np.r_[False,ink.any(0),False].astype(np.int8))
    spans = list(zip(np.where(edges==1)[0],np.where(edges==-1)[0]))
    if len(spans) < 2:
        return False
    left,right = spans[-1]
    previous_left,previous_right = spans[-2]
    previous_y = np.flatnonzero(ink[:,previous_left:previous_right].any(1))
    count,_,stats,_ = cv2.connectedComponentsWithStats(ink[:,left:right].astype(np.uint8))
    if count != 2:
        return False
    _,top,width,height,_ = stats[1]
    line_height = yy.max()-yy.min()+1
    center = top+(height-1)/2
    pixels = crop[:,left:right][ink[:,left:right]].astype(np.int16)
    return bool(3 <= width <= line_height/3 and 3 <= height <= line_height/3
        and left-previous_right >= width
        and previous_y[0]+len(previous_y)/3 <= center <= previous_y[-1]-len(previous_y)/3
        and np.all(pixels.max(1)-pixels.min(1) <= 24))


def _read_wrapped_status_tail(context, image, row, rows, boxes, original_text, index, *, pictograms=()):
    """Read a short closing tail with its independently observed previous line."""
    tail = _normalize_status_text(original_text)
    if re.fullmatch(r'[^()]{1,6}\):',tail):
        x,y,w,h = row['box']
        if not _wrapped_tail_has_single_middle_dot(image[y:y+h,x:x+w]):
            return None
        # Only this independently visible single dot may be compared with the
        # strong complete-line reread; no global punctuation normalization.
        tail = tail[:-1]+'・'
    if not re.fullmatch(r'[^()]{1,6}\)[·・]?', tail):
        return None
    owners = [(x+15,y+110,w-42,h-120) for x,y,w,h in boxes
              if _inside(row,(x+15,y+110,w-42,h-120))]
    if len(owners) != 1:
        return None
    x,y,w,h = row['box']
    previous = []
    for other in rows:
        px,py,pw,ph = other['box']
        text = _normalize_status_text(other['text'])
        if (other is not row and _inside(other,owners[0])
                and math.isfinite(other['score']) and .95 <= other['score'] <= 1
                and 0 <= y-py-ph <= 12 and abs(px-x) <= 12 and pw > w
                and text.count('(') == text.count(')')+1
                and text.rfind('(') > text.rfind(')')):
            previous.append(other)
    if len(previous) != 1:
        return None
    prior = previous[0]
    # A separately detected inline pictogram can split the preceding physical
    # line. Retain all independently strong text in that same line, not only
    # the left fragment ending with the opening parenthesis.
    parts = sorted([other for other in rows if other is not row
        and _inside(other,owners[0]) and math.isfinite(other['score'])
        and .95 <= other['score'] <= 1
        and abs(other['box'][1]+other['box'][3]/2-prior['box'][1]-prior['box'][3]/2) <= 3],
        key=lambda item:item['box'][0])
    if len(parts) > 1:
        def adjacent(a,b):
            end,start = a['box'][0]+a['box'][2],b['box'][0]
            if -12 <= start-end <= 24:
                return True
            if not 24 < start-end <= 44:
                return False
            for left,top,owned in pictograms:
                yy,xx = np.where(owned)
                if (left+int(xx.min()) <= end and left+int(xx.max())+1 >= start
                        and top+int(yy.min()) < a['box'][1]+a['box'][3]
                        and top+int(yy.max())+1 > b['box'][1]):
                    return True
            return False
        if parts[0] is not prior or not all(adjacent(a,b) for a,b in zip(parts,parts[1:])):
            return None
        px = prior['box'][0]
        py = min(part['box'][1] for part in parts)
        bottom = max(part['box'][1]+part['box'][3] for part in parts)
        right=max(part['box'][0]+part['box'][2] for part in parts)
        prior=dict(prior,text=''.join(part['text'] for part in parts),box=(px,py,right-px,bottom-py))
    px,py,pw,ph = prior['box']
    if not (16 <= min(h,ph) <= max(h,ph) <= 48 and w+pw <= 640):
        return None
    height = max(h,ph)
    pieces = []
    for bx,by,bw,bh in (prior['box'],row['box']):
        crop = image[by:by+bh,bx:bx+bw].copy()
        for left,top,owned in pictograms:
            a,b=max(bx,left),max(by,top)
            c,d=min(bx+bw,left+owned.shape[1]),min(by+bh,top+owned.shape[0])
            if a<c and b<d:
                crop[b-by:d-by,a-bx:c-bx][owned[b-top:d-top,a-left:c-left]]=255
        top = (height-bh)//2
        pieces.append(cv2.copyMakeBorder(crop,top,height-bh-top,0,0,
                      cv2.BORDER_CONSTANT,value=(255,255,255)))
    node = f'LiveStatusPanelLineWrapped{index}'
    hit = context.run_recognition(node,cv2.hconcat(pieces),pipeline_override={node: {
        'recognition':'OCR','only_rec':True,'expected':'.+',
    }})
    found = list(hit.all_results) if hit and hit.hit else []
    prefix = _normalize_status_text(prior['text'])
    if (len(found) != 1 or not math.isfinite(found[0].score) or not .95 <= found[0].score <= 1
            or _normalize_status_text(found[0].text).replace('·','・')
            != (prefix+tail).replace('·','・')):
        return None
    # Keep the actual strong suffix. The middle-dot glyph equivalence is
    # local to this complete-text comparison, never an inferred missing word.
    return SimpleNamespace(text=_normalize_status_text(found[0].text)[len(prefix):],score=found[0].score)


def _read_addition_counter(context, image, row, rows, boxes, original_text, index):
    """Read both sides of a complete amount/duration divider independently."""
    if not re.fullmatch(r'\d+タ[ー-]ン', _normalize_status_text(original_text)):
        return None
    owners = [box for box in boxes if _inside(row,(box[0]+90,box[1]+8,box[2]-110,90))]
    if len(owners) != 1:
        return None
    bx,by,bw,bh = owners[0]
    headings = [value for value in rows if value is not row
        and _inside(value,(bx+90,by+8,bw-110,90))]
    body = [value for value in rows if _inside(value,(bx+15,by+110,bw-42,bh-120))]
    if (len(headings) != 1 or _normalize_status_text(headings[0]['text']) != '集中増加量追加'
            or not body or any(not math.isfinite(value['score']) or not .95 <= value['score'] <= 1
                               for value in headings+body)):
        return None
    description = re.fullmatch(r'集中の増加量を(\d+)追加(\d+)ターン',
        _normalize_status_text(_join_lines(body)))
    x,y,w,h = row['box']
    if description is None or not (16 <= h <= 48 and 32 <= w <= 180):
        return None
    crop = image[y:y+h,x:x+w]
    low,high = crop.min(2),crop.max(2)
    _,_,stats,_ = cv2.connectedComponentsWithStats(
        ((low >= 220) & (low < 235) & (high-low <= 8)).astype(np.uint8))
    dividers = [(int(sx),int(sw)) for sx,sy,sw,sh,area in stats[1:]
        if 1 <= sw <= 3 and sh >= h*.65 and area >= sw*sh*.95
        and sy > 0 and sy+sh < h and sx >= 12 and sx+sw <= w-12]
    if len(dividers) != 1:
        return None
    left,width = dividers[0]
    right = left+width
    if np.any(low[:,left-2:left] < 235) or np.any(low[:,right:right+2] < 235):
        return None
    ranges = ((0,left),(right,w))
    for start,end in ranges:
        ink = low[:,start:end] < 235
        if (not ink.any() or any(edge.any() for edge in
                (ink[:2],ink[-2:],ink[:,:2],ink[:,-2:]))):
            return None
    parts = []
    for part,(start,end) in enumerate(ranges):
        node = f'LiveStatusPanelCounterPart{index}_{part}'
        hit = context.run_recognition(node,crop[:,start:end].copy(),pipeline_override={node:dict(
            recognition='OCR',only_rec=True,expected='.+')})
        found = list(hit.all_results) if hit and hit.hit else []
        if (len(found) != 1 or not math.isfinite(found[0].score) or not .95 <= found[0].score <= 1
                or tuple(found[0].box) != (0,0,end-start,h)):
            return None
        parts.append(found[0])
    amount,turns = (_normalize_status_text(part.text) for part in parts)
    if (amount != description[1] or turns != description[2]+'ターン'
            or any(re.findall(r'\d+',text) != [amount+description[2]]
                   for text in (original_text,row['text']))):
        return None
    return SimpleNamespace(text=amount+'|'+turns,score=min(part.score for part in parts))


def _read_scaled_status_line(context, crop, original_text, index, *, alternate_text=None):
    """One fixed resampling of the same complete text, with unchanged gates."""
    if not (16 <= crop.shape[0] <= 48 and 12 <= crop.shape[1] <= 640):
        return None
    node = f'LiveStatusPanelLineScaled{index}'
    enlarged = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    hit = context.run_recognition(node,enlarged,pipeline_override={node: {
        'recognition':'OCR','only_rec':True,'expected':'.+',
    }})
    rows = list(hit.all_results) if hit and hit.hit else []
    expected = {_normalize_status_text(original_text)}
    if alternate_text is not None:
        expected.add(_normalize_status_text(alternate_text))
    if (len(rows) != 1 or not math.isfinite(rows[0].score) or not .95 <= rows[0].score <= 1
            or _normalize_status_text(rows[0].text) not in expected):
        return None
    return rows[0]


def _read_ink_status_line(context, crop, original_text, index):
    """Remove color contrast in mixed red/black text without changing its ink."""
    if not (16 <= crop.shape[0] <= 48 and 12 <= crop.shape[1] <= 640):
        return None
    darkest = crop.min(axis=2)
    if not np.any((crop.max(axis=2)-darkest >= 64) & (darkest < 220)):
        return None
    node = f'LiveStatusPanelLineInk{index}'
    monochrome = cv2.cvtColor(darkest, cv2.COLOR_GRAY2BGR)
    hit = context.run_recognition(node,monochrome,pipeline_override={node: {
        'recognition':'OCR','only_rec':True,'expected':'.+',
    }})
    rows = list(hit.all_results) if hit and hit.hit else []
    if (len(rows) != 1 or not math.isfinite(rows[0].score) or not .95 <= rows[0].score <= 1
            or _normalize_status_text(rows[0].text) != _normalize_status_text(original_text)):
        return None
    return rows[0]


def _confirm_countless_body(context, image, rows, boxes, consistent_strong_boxes):
    """One complete-line read before applying the existing countless grammar."""
    for x,y,w,h in boxes:
        header = [row for row in rows if _inside(row,(x+90,y+8,w-110,90))]
        body = [row for row in rows if _inside(row,(x+15,y+110,w-42,h-120))]
        if (len(header) != 1 or len(body) != 1
                or any(not math.isfinite(row['score']) or not .95 <= row['score'] <= 1
                       for row in header+body)):
            continue
        expected = _COUNTLESS_STATUS_DESCRIPTIONS.get(_normalize_status_text(header[0]['text']))
        row = body[0]
        observed = _normalize_status_text(row['text'])
        counter = image[y+57:y+98,x+90:x+w-20]
        if (expected is None or observed == expected or tuple(row['box']) not in consistent_strong_boxes
                or re.findall(r'\d+',observed) != re.findall(r'\d+',expected)
                or counter.shape != (41,w-110,3) or counter.min() < 235):
            continue
        bx,by,bw,bh = row['box']
        confirmed = _read_scaled_status_line(context,image[by:by+bh,bx:bx+bw].copy(),
            row['text'],rows.index(row),alternate_text=expected)
        if confirmed is not None:
            row.update(text=confirmed.text,score=confirmed.score)


def _without_status_body_pictograms(image, rows, boxes, *, pictograms=None):
    """Exclude separately detected icon text only with complete pixel ownership."""
    excluded = set()
    for x, y, w, h in boxes:
        left, top, width, height = x+15, y+110, w-42, h-120
        crop = image[top:top+height, left:left+width]
        ink = (crop.min(2) < 235).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        colors = (cv2.inRange(hsv, (85, 100, 120), (112, 255, 255)) > 0,
                  # Actual1785's red good-condition icon has hue zero. Keep
                  # the same saturation/value, shape and complete ownership.
                  cv2.inRange(hsv, (0, 100, 120), (0, 255, 255)) > 0)
        body = [(i, row) for i, row in enumerate(rows)
                if _inside(row, (left, top, width, height))]
        for label, (bx, by, bw, bh, _area) in enumerate(stats[1:], 1):
            # Reuse the detail-icon size/color gates. The enclosing ink
            # component joins cyan parts split by the white central glyph.
            if (not (24 <= bw <= 44 and 22 <= bh <= 44)
                    or bx <= 0 or by <= 0 or bx+bw >= width or by+bh >= height):
                continue
            shape = labels[by:by+bh, bx:bx+bw] == label
            if max(np.count_nonzero(color[by:by+bh, bx:bx+bw] & shape)
                   for color in colors)/(bw*bh) < .2:
                continue
            cy, cx = (bh-1)/2, (bw-1)/2
            yy, xx = np.where(shape)
            # A complete diamond has four centered tips. Its three-pixel
            # antialiased outline cannot justify a rectangle, letter or cut edge.
            # Even-sized rasters have two central pixels; measure the same
            # three-pixel outline from the nearer one, not a half-pixel gap.
            if (np.max(np.abs(xx-cx)+np.abs(yy-cy)) > max(cx, cy)+3
                    or any(np.any(np.minimum(
                            np.abs(np.flatnonzero(edge)-math.floor(center)),
                            np.abs(np.flatnonzero(edge)-math.ceil(center))) > 3)
                           for edge, center in ((shape[0], cx), (shape[-1], cx),
                                                (shape[:, 0], cy), (shape[:, -1], cy)))):
                continue
            if pictograms is not None:
                pictograms.append((left,top,labels == label))
            for index, row in body:
                rx, ry, rw, rh = row['box']
                rx, ry = rx-left, ry-top
                # The independently bounded diamond keeps its original size
                # gate. A larger OCR box may contain only blank padding or
                # pixels already owned by another complete strong text line.
                if not (bx <= rx+rw/2 <= bx+bw and by <= ry+rh/2 <= by+bh):
                    continue
                own = labels[ry:ry+rh, rx:rx+rw] == label
                if not own.any():
                    continue
                covered = own.copy()
                for other_index, other in body:
                    # An OCR box may include a sliver of its neighbour. Those
                    # pixels must be owned by another independently read full line.
                    if (other_index == index or len(other['text']) < 2
                            or not math.isfinite(other['score']) or other['score'] < .95):
                        continue
                    ox, oy, ow, oh = other['box']
                    a, b = max(0, ox-left-rx), max(0, oy-top-ry)
                    c, d = min(rw, ox-left-rx+ow), min(rh, oy-top-ry+oh)
                    if a < c and b < d:
                        covered[b:d, a:c] = True
                if np.all(covered | ~ink[ry:ry+rh, rx:rx+rw].astype(bool)):
                    excluded.add(index)
    return [row for index, row in enumerate(rows) if index not in excluded]


def _reread_status_body_pictograms(context,image,rows,pictograms):
    """Read merged text after clearing only independently owned diamond ink."""
    for index,row in enumerate(rows):
        rx,ry,rw,rh=row['box']
        for left,top,owned in pictograms:
            yy,xx=np.where(owned)
            ix,iy,iw,ih=left+int(xx.min()),top+int(yy.min()),int(np.ptp(xx))+1,int(np.ptp(yy))+1
            if not (left <= rx and rx+rw <= left+owned.shape[1]
                    and top <= ry and ry+rh <= top+owned.shape[0]
                    and rx <= ix < ix+iw < rx+rw and ry < iy+ih and ry+rh > iy):
                continue
            original=image[ry:ry+rh,rx:rx+rw]
            ownership=owned[ry-top:ry-top+rh,rx-left:rx-left+rw]
            prefix=ix+iw-rx
            if np.any((original[:,:prefix].min(2)<235)&~ownership[:,:prefix]):
                continue
            cleaned=original.copy()
            cleaned[ownership]=255
            node=f'LiveStatusPanelPictogramLine{index}'
            hit=context.run_recognition(node,cleaned,pipeline_override={node:dict(
                recognition='OCR',only_rec=True,expected='.+')})
            found=list(hit.all_results) if hit and hit.hit else []
            if len(found)!=1 or not math.isfinite(found[0].score) or not .95 <= found[0].score <= 1:
                continue
            text,previous=_normalize_status_text(found[0].text),_normalize_status_text(row['text'])
            if (len(text)<2 or not previous.endswith(text)
                    or re.findall(r'\d+',text)!=re.findall(r'\d+',previous)):
                continue
            row.update(text=found[0].text,score=found[0].score,
                pictogram_original_text=row['text'],pictogram_box=(ix,iy,iw,ih))
    return rows


def _complete_wrapped_tail_box(image,row,boxes):
    """Complete a neutral dot cut by a tail detector, without supplying text."""
    if not re.fullmatch(r'[^()]{1,6}\)[·・]?',_normalize_status_text(row['text'])):
        return
    owners=[box for box in boxes if _inside(row,(box[0]+15,box[1]+110,box[2]-42,box[3]-120))]
    if len(owners)!=1:
        return
    x,y,w,h=row['box']
    right=min(x+w+h//3+3,owners[0][0]+owners[0][2]-27)
    crop=image[y:y+h,x:right]
    _,labels,stats,_=cv2.connectedComponentsWithStats((crop.min(2)<220).astype(np.uint8))
    dots=[(label,p) for label,p in enumerate(stats[1:],1)
          if 3<=p[2]<=h/3 and 3<=p[3]<=h/3 and p[0]<w<=p[0]+p[2]]
    if len(dots)!=1:
        return
    _,(dx,dy,dw,dh,_)=dots[0]
    width=int(dx+dw+2)
    if width>crop.shape[1] or not _wrapped_tail_has_single_middle_dot(crop[:,:width]):
        return
    row.update(original_box=row['box'],box=(x,y,width,h),complete_tail_dot=True)


def _reread_status_body_type_icons(context, image, rows, boxes):
    """Read complete text beside a separately proven grey type-circle icon."""
    for x,y,w,h in boxes:
        left,top,width,height = x+15,y+110,w-42,h-120
        crop = image[top:top+height,left:left+width]
        # Reuse the explicit item-counter grey-circle geometry, without
        # closing a broken contour or interpreting any enclosed character.
        gray = ((np.ptp(crop,axis=2) <= 12) & (crop.min(2) >= 90)
                & (crop.max(2) <= 200)).astype(np.uint8)*255
        contours,_ = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        ink = (crop.min(2) < 235).astype(np.uint8)
        ink_contours,_ = cv2.findContours(ink,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        icons = []
        for contour in contours:
            bx,by,bw,bh = cv2.boundingRect(contour)
            if (not (24 <= min(bw,bh) <= max(bw,bh) <= 44)
                    or bx <= 0 or by <= 0 or bx+bw >= 75 or by+bh >= height):
                continue
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area,perimeter = cv2.contourArea(hull),cv2.arcLength(hull,True)
            if (.85 <= bw/bh <= 1.18 and .60 <= area/(bw*bh) <= .86
                    and hull_area > 0 and perimeter > 0 and area/hull_area >= .85
                    and 4*math.pi*hull_area/perimeter**2 >= .75):
                # 1797's complete antialiased ring extends one pixel beyond
                # the solid-grey seed. Bind its actual enclosing ink contour;
                # do not clear the surrounding rectangle or other grey text.
                outer = []
                for candidate in ink_contours:
                    ox,oy,ow,oh = cv2.boundingRect(candidate)
                    if (ox <= bx and oy <= by and ox+ow >= bx+bw and oy+oh >= by+bh
                            and max(bx-ox,by-oy,ox+ow-bx-bw,oy+oh-by-bh) <= 1):
                        outer.append((candidate,(left+ox,top+oy,ow,oh)))
                if len(outer) == 1:
                    owned = np.zeros((height,width),np.uint8)
                    cv2.drawContours(owned,[outer[0][0]],-1,1,cv2.FILLED)
                    icons.append((outer[0][1],owned.astype(bool)))
        if len(icons) != 1:
            continue
        (ix,iy,iw,ih),owned = icons[0]
        for index,row in enumerate(rows):
            rx,ry,rw,rh = row['box']
            if (not _inside(row,(left,top,width,height))
                    or not ix <= rx < ix+iw < rx+rw or not ry < iy+ih or ry+rh <= iy):
                continue
            original = image[ry:ry+rh,rx:rx+rw]
            ownership = owned[ry-top:ry-top+rh,rx-left:rx-left+rw]
            prefix_ink = original[:,:ix+iw-rx].min(2) < 235
            if np.any(prefix_ink & ~ownership[:,:ix+iw-rx]):
                continue
            cleaned = original.copy()
            cleaned[ownership] = 255
            node = f'LiveStatusPanelTypeIconLine{index}'
            hit = context.run_recognition(node,cleaned,pipeline_override={node:dict(
                recognition='OCR',only_rec=True,expected='.+')})
            found = list(hit.all_results) if hit and hit.hit else []
            if len(found) != 1 or not math.isfinite(found[0].score) or not .95 <= found[0].score <= 1:
                continue
            text,previous = _normalize_status_text(found[0].text),_normalize_status_text(row['text'])
            if (len(text) < 2 or not previous.endswith(text)
                    or re.findall(r'\d+',text) != re.findall(r'\d+',previous)):
                continue
            row.update(text=found[0].text,score=found[0].score,
                pictogram_original_text=row['text'],pictogram_box=icons[0][0])
    return rows


class StaticStatusHeadingUnreadable(ValueError):
    """A complete unchanged heading may request navigation, never supply a row."""

    def __init__(self, message, *, heading, box, rows):
        super().__init__(message)
        self.heading = heading
        self.box = tuple(box)
        self.rows = tuple(dict(row) for row in rows)


def _static_heading_reframe(error, frames, geometries, pages, frame_ids):
    """Prove one small scroll can retain an independently identified overlap."""
    if (not isinstance(error, StaticStatusHeadingUnreadable) or not pages or not frames
            or len(frames) != len(frame_ids) or len(set(frame_ids)) != len(frames)
            or not all(isinstance(value,str) and value for value in frame_ids)
            or not _status_geometry_stable(geometries)
            or any(geometry[2] or geometry[3] for geometry in geometries)):
        return None
    previous = pages[-1]
    boxes, thumb, *_ = geometries[0]
    if not previous.rows or thumb[0] <= previous.thumb[0]:
        return None
    weak_rows = [box for box in boxes if _inside({'box':error.box},
        (box[0]+90,box[1]+8,box[2]-110,90))]
    if len(weak_rows) != 1:
        return None
    overlap = previous.rows[-1]
    weak_overlap = error.heading == overlap.title
    if error.heading not in _STATIC_STATUS_HEADINGS and not weak_overlap:
        return None
    anchors = [(row,box) for box in boxes for row in error.rows
        if _normalize_status_text(row['text']) == overlap.title
        and type(row['score']) in (int,float) and math.isfinite(row['score']) and .95 <= row['score'] <= 1
        and _inside(row,(box[0]+90,box[1]+8,box[2]-110,90))]
    if weak_overlap and not anchors:
        # The previous page already read this complete row. Its current full
        # description (including all numbers) may anchor navigation only;
        # neither the old heading nor its confidence supplies a new page row.
        anchors = [(row,box) for box in weak_rows for row in error.rows
            if overlap.description and _normalize_status_text(row['text']) == overlap.description
            and type(row['score']) in (int,float) and math.isfinite(row['score']) and .95 <= row['score'] <= 1
            and _inside(row,(box[0]+15,box[1]+110,box[2]-42,box[3]-120))]
    else:
        weak_overlap = False
    if len(anchors) != 1:
        return None
    anchor, box = anchors[0]
    if weak_overlap:
        x,y,w,h = anchor['box']
        anchor = dict(anchor,box=(x-2,y-2,w+4,h+4))
        if not _inside(anchor,(box[0]+15,box[1]+110,box[2]-42,box[3]-120)):
            return None
    if (any(abs(box[i]-overlap.box[i]) > 2 for i in (0,2,3))
            or box[1] >= overlap.box[1] or (box[1]-40)//2 < 24
            or (not weak_overlap and weak_rows[0][1] < box[1]+box[3])):
        return None
    # A complete overlap label anchors the next scroll. Preserve any supplied
    # image disagreement, and leave the unreadable heading unresolved.
    for roi in (error.box, tuple(anchor['box'])):
        x,y,w,h = roi
        crops = [frame[y:y+h,x:x+w] for frame in frames]
        if (any(crop.shape != (h,w,3) for crop in crops)
                or not all(np.array_equal(crops[0],crop) for crop in crops[1:])):
            return None
        ink = crops[0].min(2) < 235
        if not ink.any() or any(edge.any() for edge in (ink[0],ink[-1],ink[:,0],ink[:,-1])):
            return None
    proof = dict(heading=error.heading, heading_roi=list(error.box),
        overlap_title=overlap.title, overlap_box=list(box), overlap_title_roi=list(anchor['box']),
        frame_ids=list(frame_ids), scroll_distance=24, failed_batch_accepted=False)
    if weak_overlap:
        proof['overlap_anchor'] = 'current_complete_body'
    return proof


class _StatusLineRecognitionCache:
    """Reuse strong OCR computations only inside one observed three-frame batch.

    Each frame still detects, parses and checks its own complete page. Only an
    identical line image and recognition configuration can reuse text results;
    recognition/frame identifiers and captured times are never retained here.
    """

    def __init__(self, context):
        self.context = context
        self.gkh_hif_client_size = getattr(context, 'gkh_hif_client_size', ())
        self.entries = {}
        self.reused = 0

    def run_recognition(self, node, image, *, pipeline_override=None):
        params = (pipeline_override or {}).get(node, {})
        eligible = (re.fullmatch(r'LiveStatusPanel(?:Line|TriggerComplete|TriggerPadded)\d+', node) is not None
                    and params.get('recognition') == 'OCR'
                    and params.get('only_rec') is True)
        if eligible:
            for config, pixels, rows in self.entries.get(node, ()):
                if (config == pipeline_override and pixels.dtype == image.dtype
                        and np.array_equal(pixels, image)):
                    self.reused += 1
                    return SimpleNamespace(hit=True, all_results=deepcopy(rows))
        hit = self.context.run_recognition(node, image, pipeline_override=pipeline_override)
        rows = list(hit.all_results) if hit and hit.hit else []
        if (eligible and len(rows) == 1 and isinstance(rows[0].text, str)
                and math.isfinite(rows[0].score) and .95 <= rows[0].score <= 1):
            self.entries.setdefault(node, []).append(
                (deepcopy(pipeline_override), image.copy(), deepcopy(rows)))
        return hit


def _complete_memory_trigger_line(context, image, row, boxes, index):
    """Read a small-window trigger from complete ink, never alias card names.

    1937's tight detector box reads a small kana as its large form. Two
    independently framed complete crops must agree above the existing gate.
    Counts and effect amounts never enter this name-only recognition path.
    """
    from .layout import calibrated_live_client_size
    text = _normalize_status_text(row['text'])
    if (calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', ())) != (480, 853)
            or not re.fullmatch(r'[^\d]+使用後、', text)):
        return None
    x,y,w,h = row['box']
    expanded = dict(row, box=(x-3,y-3,w+6,h+6))
    if not any(_inside(expanded,(bx+15,by+110,bw-42,bh-120)) for bx,by,bw,bh in boxes):
        return None
    crop = image[y-3:y+h+3,x-3:x+w+3].copy()
    ink = crop.min(2) < 235
    if not ink.any() or any(edge.any() for edge in (ink[0],ink[-1],ink[:,0],ink[:,-1])):
        return None
    yy,xx = np.where(ink)
    padded = cv2.copyMakeBorder(crop[yy.min():yy.max()+1,xx.min():xx.max()+1],
        4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))
    reads = []
    for suffix,picture in (('Complete',crop),('Padded',padded)):
        node = f'LiveStatusPanelTrigger{suffix}{index}'
        hit = context.run_recognition(node,picture,pipeline_override={node: {
            'recognition':'OCR','only_rec':True,'expected':'.+',
        }})
        found = list(hit.all_results) if hit and hit.hit else []
        if (len(found) != 1 or not math.isfinite(found[0].score) or not .95 <= found[0].score <= 1
                or not re.fullmatch(r'[^\d]+使用後、', _normalize_status_text(found[0].text))):
            return None
        reads.append(found[0])
    if _normalize_status_text(reads[0].text) != _normalize_status_text(reads[1].text):
        return None
    return SimpleNamespace(text=reads[0].text,score=min(value.score for value in reads))


def read_status_panel(context, image):
    """Detect text once, then read each semantic line from the same image.

    Panel-scale detection alone confused small kana in a memory trigger name.
    Native line recognition fixes that crop issue without fuzzy text aliases
    or selecting a preferred catalogue name. Every line has one bounded read.
    """
    boxes, *_ = status_panel_geometry(image)
    node = 'LiveStatusPanelText'
    hit = context.run_recognition(node,image,pipeline_override={node: {
        'recognition':'OCR','roi':[40,20,642,718],'expected':'.+',
    }})
    raw = [dict(text=row.text,score=row.score,box=tuple(row.box))
           for row in hit.all_results] if hit and hit.hit else []
    semantic = []
    for row in raw:
        if any(_inside(row,(x+90,y+8,w-110,90)) or _inside(row,(x+15,y+110,w-42,h-120))
               for x,y,w,h in boxes):
            semantic.append(row)
    if not 1 <= len(semantic) <= 32:
        raise ValueError('status panel text budget or coverage is invalid')
    # Independently owned icon pixels need no text-confidence promotion. Keep
    # the final pass for ownership that requires a neighbour's strong line read.
    pictograms=[]
    semantic = _without_status_body_pictograms(image, semantic, boxes,pictograms=pictograms)
    semantic = _reread_status_body_pictograms(context,image,semantic,pictograms)
    for row in semantic:
        _complete_wrapped_tail_box(image,row,boxes)
    consistent_strong_boxes = set()
    for index,row in enumerate(semantic):
        x,y,w,h = row['box']
        node = f'LiveStatusPanelLine{index}'
        trigger = _complete_memory_trigger_line(context,image,row,boxes,index)
        if trigger is not None:
            results = [trigger]
        elif 'pictogram_original_text' in row:
            # The raw line still contains the icon. Use the independently read
            # cleaned line, never reintroduce that icon as a letter.
            results=[SimpleNamespace(text=row['text'],score=row['score'])]
        else:
            hit = context.run_recognition(node,image[y:y+h,x:x+w].copy(),pipeline_override={node: {
                'recognition':'OCR','only_rec':True,'expected':'.+',
            }})
            results = list(hit.all_results) if hit and hit.hit else []
        if len(results) != 1:
            raise ValueError('status panel line is not uniquely readable')
        if (math.isfinite(row['score']) and .95 <= row['score'] <= 1
                and math.isfinite(results[0].score) and .95 <= results[0].score <= 1
                and _normalize_status_text(row['text']) == _normalize_status_text(results[0].text)):
            consistent_strong_boxes.add(tuple(row['box']))
        if (trigger is None and math.isfinite(row['score']) and .95 <= row['score'] <= 1
                and math.isfinite(results[0].score) and .95 <= results[0].score <= 1
                and _normalize_status_text(row['text']) != _normalize_status_text(results[0].text)
                and any(_inside(row,(bx+15,by+110,bw-42,bh-120))
                        for bx,by,bw,bh in boxes)):
            # A tight line crop can confuse kana that the full panel read
            # correctly. A single fixed complete-crop reread must independently
            # confirm either observation, never a preferred rule name.
            if re.findall(r'\d+',row['text']) != re.findall(r'\d+',results[0].text):
                raise ValueError('status panel body numeric readings conflict')
            confirmed = _read_scaled_status_line(context,image[y:y+h,x:x+w].copy(),row['text'],index,
                                                  alternate_text=results[0].text)
            if confirmed is None:
                raise ValueError('status panel body readings conflict')
            results = [confirmed]
        if results[0].score < .95:
            counter = _read_addition_counter(context,image,row,semantic,boxes,results[0].text,index)
            if counter is not None:
                results = [counter]
        if results[0].score < .95:
            confidence = _static_heading_confidence(context,image,
                dict(row,text=results[0].text),boxes,index)
            if confidence is not None:
                row['heading_source'] = 'static_template_with_matching_ocr_text'
                results = [SimpleNamespace(text=results[0].text,score=confidence)]
        if results[0].score < .95:
            # Small glyphs occupy very different fractions of detector boxes.
            # One bounded reread trims the blank margin and restores a fixed
            # border. Short fixed headings need room around their outer glyphs;
            # keep body text's existing border and the original confidence gate.
            crop = image[max(0,y-2):y+h+2,max(0,x-2):x+w+2].copy()
            yy,xx = np.where(crop.min(axis=2) < 235)
            if not len(xx):
                raise ValueError('status panel line has no visible text')
            complete_ink = (yy.min() > 0 and yy.max() < crop.shape[0]-1
                            and xx.min() > 0 and xx.max() < crop.shape[1]-1)
            heading_text = _normalize_status_text(results[0].text)
            # The same complete-heading border applies to independently
            # bounded nonnumeric titles, not only the old static-template set.
            # Counter lines and clipped new headings keep their original path.
            heading = ((heading_text in _COMPLETE_STATUS_HEADINGS
                        or (complete_ink and heading_text and not re.search(r'\d',heading_text)))
                       and any(_inside(row,(bx+90,by+8,bw-110,90))
                               for bx,by,bw,bh in boxes))
            if heading and heading_text == '消費体力削減':
                ink = crop.min(axis=2) < 235
                if any(edge.any() for edge in (ink[0],ink[-1],ink[:,0],ink[:,-1])):
                    raise ValueError('status panel complete heading crop is clipped')
            border = 8 if heading else 2
            crop = cv2.copyMakeBorder(crop[yy.min():yy.max()+1,xx.min():xx.max()+1],
                                     border,border,border,border,cv2.BORDER_CONSTANT,value=(255,255,255))
            node = f'LiveStatusPanelLineNormalized{index}'
            retry = context.run_recognition(node,crop,pipeline_override={node: {
                'recognition':'OCR','only_rec':True,'expected':'.+',
            }})
            corrected = list(retry.all_results) if retry and retry.hit else []
            # The fixed line box can clip a small kana. When the full panel
            # and the complete, padded heading both read the same full text
            # strongly, retain their reading over the weak tight crop.
            if (heading and complete_ink and len(corrected) == 1
                    and .95 <= row['score'] <= 1 and .95 <= corrected[0].score <= 1
                    and _normalize_status_text(row['text']) == _normalize_status_text(corrected[0].text)):
                row.update(text=corrected[0].text, score=min(row['score'], corrected[0].score))
                continue
            # Identical complete title text in two different crops is enough
            # for visual identity. Do not keep enlarging it solely to reach
            # .95, or promote its score. Rule identity, body and values remain
            # independently validated; this applies to all nonnumeric titles.
            if heading and complete_ink and len(corrected) == 1 and corrected[0].score < .95:
                readings = ((results[0].text, results[0].score),
                            (corrected[0].text, corrected[0].score))
                confidence = min(reading[1] for reading in readings)
                if _corroborated_heading(heading_text.removesuffix('おすすめ効果'), confidence, readings):
                    row.update(text=results[0].text, score=confidence,
                               heading_source='corroborated_complete_heading',
                               heading_observations=readings)
                    continue
            if (len(corrected) == 1 and corrected[0].score < .95
                    and _normalize_status_text(corrected[0].text) == _normalize_status_text(results[0].text)):
                scaled = _read_scaled_status_line(context,image[y:y+h,x:x+w].copy(),results[0].text,index)
                if scaled is None:
                    scaled = _read_ink_status_line(context,image[y:y+h,x:x+w].copy(),results[0].text,index)
                if (scaled is None and complete_ink
                        and (heading or any(_inside(row,(bx+15,by+110,bw-42,bh-120))
                                            for bx,by,bw,bh in boxes))):
                    # 1777's full body agrees in both weak reads. Reuse its
                    # existing normalized ink and fixed border at the same 2x
                    # scale, also for 1829's complete heading. Clipped ink or
                    # a different full text cannot pass.
                    scaled = _read_scaled_status_line(context,crop,results[0].text,index)
                if scaled is not None:
                    corrected = [scaled]
            if (len(corrected) != 1
                    or _normalize_status_text(corrected[0].text) != _normalize_status_text(results[0].text)
                    or not math.isfinite(corrected[0].score) or not .95 <= corrected[0].score <= 1):
                glyphs=_read_short_status_line(context,image[y:y+h,x:x+w].copy(),results[0].text,index)
                if glyphs is None:
                    glyphs=_read_wrapped_status_tail(context,image,row,semantic,boxes,results[0].text,index,
                                                     pictograms=pictograms)
                if glyphs is None:
                    message = ('status panel normalized line is low confidence or contradictory: '
                               f'original={results[0].text!r}/{results[0].score:.6f}, '
                               f'normalized={[(r.text,r.score) for r in corrected]!r}, '
                               f'box={(x,y,w,h)}')
                    if (heading and len(corrected) == 1
                            and all(math.isfinite(item.score) and 0 <= item.score < .95
                                    for item in (results[0],corrected[0]))
                            and _normalize_status_text(corrected[0].text) == _normalize_status_text(results[0].text)):
                        raise StaticStatusHeadingUnreadable(message,
                            heading=_normalize_status_text(results[0].text),box=row['box'],rows=raw)
                    raise ValueError(message)
                corrected=[glyphs]
            results = corrected
        row.update(text=results[0].text,score=results[0].score)
    semantic = _without_status_body_pictograms(image, semantic, boxes)
    semantic = _reread_status_body_type_icons(context,image,semantic,boxes)
    _confirm_countless_body(context,image,semantic,boxes,consistent_strong_boxes)
    return parse_status_panel(image, semantic)


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


class StatusInspectionError(ValueError):
    def __init__(self, message, events):
        super().__init__(message)
        self.events = events


def _status_navigation_target(context, image, template, roi, node):
    hit = context.run_recognition(node,image,pipeline_override={node: {
        'recognition':'TemplateMatch','template':[template],'roi':list(roi),'threshold':.95,
    }})
    found = list(hit.filtered_results) if hit and hit.hit else []
    client_size = getattr(context, 'gkh_hif_client_size', None)
    if (not found and node == 'LiveStatusOpen' and template == 'produce/hif_live_status.png'
            and isinstance(client_size, (tuple, list)) and len(client_size) == 2
            and all(type(value) is int for value in client_size)
            and tuple(client_size) in ((359, 638), (540, 960), (541, 961), (541, 962), (542, 962))
            and getattr(image, 'shape', ()) == (1280, 720, 3)):
        if tuple(client_size) == (359, 638):
            # This raster softens both the cyan tile and its white glyph.
            fallback, source = 'LiveStatusOpenSmall', image
            alternative = 'produce/hif_live_status_359x638.png'
        else:
            fallback = 'LiveStatusOpenMinimum'
            source = cv2.cvtColor(image.min(2), cv2.COLOR_GRAY2BGR)
            alternative = 'produce/hif_live_status_minimum.png'
        hit = context.run_recognition(fallback,source,pipeline_override={fallback: {
            'recognition':'TemplateMatch','template':[alternative],
            'roi':list(roi),'threshold':.95,
        }})
        found = list(hit.filtered_results) if hit and hit.hit else []
    if len(found) != 1:
        raise ValueError('status navigation target is not unique: '+node)
    box = tuple(found[0].box)
    if len(box) != 4 or any(type(value) is not int for value in box) or min(box[2:]) <= 0 or not _inside({'box':box},roi):
        raise ValueError('status navigation target is outside its control region')
    return box


def _resume_status_scroll(source, event):
    """Recover the observed prefix and optional actual pre-swipe geometry."""
    from route.engine import has_single_frame_id

    saved = source['status']
    inputs = saved['inputs']
    before = saved.get('before_frame_ids', ())
    if not has_single_frame_id(before) or not has_single_frame_id(source['before'].get('frame_ids')):
        raise ValueError('status resume requires its current single-frame entry')
    if (saved['phase'] == 'status_inspection' and saved['accepted'] is False and not saved['pages']
            and len(inputs) == 1 and inputs[0].get('completed') is True and inputs[0]['phase'] == 'open'):
        event.update(before_frame_ids=list(saved['before_frame_ids']), resumed_status_source=source['source_result'])
        return [], None
    if (saved['phase'] != 'status_inspection' or saved['accepted'] is not False
            or not inputs or inputs[-1]['phase'] != 'scroll'
            or any(step.get('completed') is not True or step['phase'] == 'close' for step in inputs)):
        raise ValueError('status resume lacks a completed open-panel scroll')
    pages = []
    frame_ids = set(saved['before_frame_ids'])
    for group in saved['pages']:
        ids = group['frame_ids']
        if not has_single_frame_id(ids) or len(group['observations']) != 1 or frame_ids.intersection(ids):
            raise ValueError('status resume page lacks independent observations')
        frame_ids.update(ids)
        observations = tuple(StatusPanelPage(
            rows=tuple(StatusPanelRow(**dict(row,box=tuple(row['box']))) for row in item['rows']),
            thumb=tuple(item['thumb']), at_top=item['at_top'], at_bottom=item['at_bottom'],
            partial_row_count=item['partial_row_count']) for item in group['observations'])
        if any(not status_row_confidence_valid(row) for item in observations for row in item.rows):
            raise ValueError('status resume page contains an unresolved row')
        pages.append(status_panel_consensus(observations))
    _join_status_panel_prefix(pages)
    if pages[-1].at_bottom or len(pages) >= 12:
        raise ValueError('status resume has no pending page')
    event.update(before_frame_ids=list(saved['before_frame_ids']),pages=deepcopy(saved['pages']),
                 resumed_status_source=source['source_result'])
    anchor = source.get('scroll_anchor')
    completed = None
    if anchor is not None:
        event['resumed_scroll_frame_id'] = anchor['frame_id']
        boxes,thumb,top,bottom,partial = anchor['geometry']
        completed = dict(geometry=(tuple(map(tuple,boxes)),tuple(thumb),top,bottom,partial),
            content_key=pages[-1].content_key,
            distance=inputs[-1]['scroll_distance'],input_index=len(inputs)-1,
            completed=True,restored=True)
    return pages,completed


def observe_status_transaction(context, *, capture_frames, checkpoint, frame_identifier,
                               maximum_age_ms=1000, restore_open_panel=False,
                               return_checkpoint=None, resume_open_panel=None):
    """Open, read up to twelve overlapping pages, close and verify unchanged Live.

    The caller supplies the same shared Live checkpoint used by inventory
    inspection. Every actual input uses native Maa and checks all endpoint
    tokens plus a fresh, uniquely recognized page before dispatch.
    """
    from utils.maa_input import verify_maa_input_endpoints

    if type(maximum_age_ms) not in (int,float) or not 0 < maximum_age_ms <= 1000:
        raise ValueError('status navigation freshness gate is invalid')
    if return_checkpoint is not None and not callable(return_checkpoint):
        raise ValueError('status return checkpoint callback is invalid')
    if restore_open_panel and resume_open_panel is not None:
        raise ValueError('status resume and close-only recovery are mutually exclusive')
    controller = context.tasker.controller
    event = dict(phase='status_inspection',accepted=False,input_click_count=0,inputs=[],pages=[])
    events = [event]
    started = time.perf_counter()

    def fresh():
        image = controller.post_screencap().wait().get()
        return image,frame_captured_at(controller,image)

    def target(image, template, roi, node):
        return _status_navigation_target(context, image, template, roi, node)

    def panel_anchor(image, *, closing=False):
        # Closing an already identified overlay does not claim its entire
        # effect list was readable. Keep independent thumb/white-row/X anchors.
        try:
            status_panel_geometry(image,require_complete_layout=not closing)
        except ValueError:
            sink=getattr(context,'gkh_hif_status_failure_sink',None)
            if callable(sink): sink(image)
            raise
        return target(image,'produce/hif_status_close.png',(300,690,120,130),'LiveStatusClose')

    def dispatch(captured_at, phase, box=None, *, scroll_distance=80):
        endpoints = verify_maa_input_endpoints(controller.info, expected_client_size=getattr(context, "gkh_hif_client_size", (720, 1280)))
        if (time.perf_counter()-captured_at)*1000 > maximum_age_ms:
            raise ValueError('status navigation snapshot expired')
        if (getattr(context, 'gkh_hif_upstream_card_ui', False) is not True
                and phase != 'close' and time.perf_counter()-started > 55):
            raise ValueError('status inspection wall-clock budget reached')
        event['input_click_count'] += 1
        event['inputs'].append(dict(phase=phase,box=box,endpoints=endpoints))
        if box is None:
            # Keep the last complete row in overlap even with observed coast
            # (402px for a 240px gesture). Prefix joining verifies it afresh.
            event['inputs'][-1]['scroll_distance']=scroll_distance
            result = controller.post_swipe(560,610,560,610-scroll_distance,400)
        else:
            x,y,w,h = box
            result = controller.post_click(x+w//2,y+h//2)
        if not result.wait().succeeded:
            raise ValueError('Maa status navigation input failed')
        event['inputs'][-1]['completed'] = True
        time.sleep(.35)

    try:
        if restore_open_panel:
            frames = capture_frames()
            geometries = tuple(status_panel_geometry(frame,require_complete_layout=False) for frame in frames)
            geometries = tuple(((),*row[1:]) for row in geometries)
            if not _status_geometry_stable(geometries):
                raise ValueError('status recovery requires a stable open panel')
            for frame in frames:
                panel_anchor(frame,closing=True)
            image,captured_at = fresh()
            geometry = status_panel_geometry(image,require_complete_layout=False)
            geometry = ((),*geometry[1:])
            if not _status_geometry_stable((geometries[-1],geometry)):
                raise ValueError('status recovery panel changed before close')
            close = panel_anchor(image,closing=True)
            dispatch(captured_at,'close',close)
            after = checkpoint(capture_frames())
            event.update(accepted=True,recovery_only=True,live_return_verified=True,
                         after_frame_ids=after['frame_ids'],elapsed_ms=(time.perf_counter()-started)*1000)
            return (),events
        from .inventory import (
            capture_initial_live_checkpoint,
            fresh_identity_reobserve_allowed,
            unresolved_live_resource_evidence,
            inspected_single_frame_reobserve_allowed,
        )
        from .partial_card import LiveCardLayoutError
        if resume_open_panel is not None:
            pages,completed_scroll = _resume_status_scroll(resume_open_panel,event)
            before = resume_open_panel['before']
        else:
            _,before = capture_initial_live_checkpoint(capture_frames,checkpoint,event,
                poll_progress=getattr(context, 'gkh_hif_poll_battle_progress', None))
            event['before_frame_ids'] = before['frame_ids']
            for attempt in range(3):
                image,captured_at = fresh()
                try:
                    current = checkpoint((image,),single_frame=True)
                    break
                except ValueError as error:
                    recoverable = (
                        str(error) == 'inventory fresh recheck is not an unselected hand'
                        or unresolved_live_resource_evidence(error)
                        or fresh_identity_reobserve_allowed(error, before)
                        or (isinstance(error, LiveCardLayoutError)
                            and str(error) == 'HIF ordinary card row is unstable')
                        or inspected_single_frame_reobserve_allowed(checkpoint, error, image, before))
                    if not recoverable or attempt == 2:
                        raise
                    _, confirmed = capture_initial_live_checkpoint(capture_frames, checkpoint, event)
                    if any(confirmed[key] != before[key] for key in ('families','fields')):
                        raise ValueError('Live changed before opening status list') from error
                    event.setdefault('preinput_hand_reobservations', []).append(confirmed['frame_ids'])
            if any(current[key] != before[key] for key in ('families','fields')):
                raise ValueError('Live changed before opening status list')
            button = target(image,'produce/hif_live_status.png',(10,230,100,100),'LiveStatusOpen')
            dispatch(captured_at,'open',button)
            pages,completed_scroll = [],None
        failure = None
        heading_reframed = False
        scroll_resent = False
        try:
            for attempt in range(3):
                image,_ = fresh()
                try:
                    # Read the overlay controls first, then its complete visible rows.
                    panel_anchor(image,closing=True)
                    break
                except ValueError:
                    if attempt == 2:
                        raise
                    time.sleep(.2)
            for index in range(len(pages),12):
                # A swipe can keep coasting after Maa reports dispatch done.
                # Observe arrival, without another input, before expensive OCR.
                for settle_attempt in range(8):
                    frames = capture_frames()
                    try:
                        geometries = tuple(status_panel_geometry(frame) for frame in frames)
                    except ValueError as error:
                        # Animated background can briefly join a row border.
                        # Reobserve only while independent overlay anchors hold.
                        panel_anchor(frames[-1],closing=True)
                        event.setdefault('settling',[]).append(dict(page=index,attempt=settle_attempt,
                            reason=str(error)))
                        if settle_attempt == 7:
                            triplet_sink=getattr(context,'gkh_hif_status_failure_triplet_sink',None)
                            if callable(triplet_sink):
                                triplet_sink(frames)
                            raise ValueError('status row layout did not settle within observation budget') from error
                        time.sleep(.08)
                        continue
                    if _status_geometry_stable(geometries):
                        observations=()
                        try:
                            # Cache no state across retries, scrolling or input.
                            line_context = _StatusLineRecognitionCache(context)
                            observations = tuple(read_status_panel(line_context,frame) for frame in frames)
                            page = status_panel_consensus(observations)
                            # A scene highlight can temporarily join the upper
                            # overlap row to the panel edge. Reobserve this page
                            # within the existing budget before another swipe.
                            _join_status_panel_prefix([*pages, page])
                            break
                        except ValueError as error:
                            sink=getattr(context,'gkh_hif_status_failure_sink',None)
                            if callable(sink):
                                sink(frames[-1])
                            event.setdefault('semantic_reobservations', []).append(dict(
                                page=index,attempt=settle_attempt,reason=str(error),
                                frame_ids=[frame_identifier(frame) for frame in frames],
                                contents=[observation.content_key for observation in observations]))
                            retry = (_unchanged_status_scroll(pages[-1],observations,geometries,
                                completed_scroll,[frame_identifier(frame) for frame in frames])
                                if str(error) == 'status list did not advance' and pages
                                and not scroll_resent and settle_attempt < 7 else None)
                            if retry is not None:
                                triplet_sink=getattr(context,'gkh_hif_status_failure_triplet_sink',None)
                                if callable(triplet_sink):
                                    triplet_sink(frames)
                                image,captured_at = fresh()
                                panel_anchor(image,closing=True)
                                current_geometry = status_panel_geometry(image)
                                if not _status_geometry_stable((geometries[-1],current_geometry)):
                                    raise ValueError('status scroll retry layout changed') from error
                                # Preserve the accepted prefix and consume this
                                # group's budget; neither loop nor clock restarts.
                                scroll_resent = True
                                event['scroll_retry'] = dict(retry,page=index,attempt=settle_attempt,
                                    remaining_capture_groups=7-settle_attempt,dispatched=False)
                                dispatch(captured_at,'scroll_retry',scroll_distance=retry['scroll_distance'])
                                event['scroll_retry']['dispatched'] = True
                                completed_scroll = None
                                continue
                            reframe = (_static_heading_reframe(error,frames,geometries,pages,
                                [frame_identifier(frame) for frame in frames])
                                if not heading_reframed and settle_attempt < 7 else None)
                            if reframe is not None:
                                image,captured_at = fresh()
                                panel_anchor(image,closing=True)
                                current_geometry = status_panel_geometry(image)
                                if not _status_geometry_stable((geometries[-1],current_geometry)):
                                    raise ValueError('status heading reframe layout changed') from error
                                for roi in (reframe['heading_roi'],reframe['overlap_title_roi']):
                                    x,y,w,h = roi
                                    if not np.array_equal(image[y:y+h,x:x+w],frames[0][y:y+h,x:x+w]):
                                        raise ValueError('status heading reframe pixels changed') from error
                                dispatch(captured_at,'heading_reframe',scroll_distance=24)
                                completed_scroll = None
                                heading_reframed = True
                                event['heading_reframe'] = dict(reframe,page=index,attempt=settle_attempt,
                                    remaining_capture_groups=7-settle_attempt)
                                continue
                            if settle_attempt == 7:
                                triplet_sink=getattr(context,'gkh_hif_status_failure_triplet_sink',None)
                                if callable(triplet_sink):
                                    triplet_sink(frames)
                                raise
                            time.sleep(.08)
                            continue
                    event.setdefault('settling',[]).append(dict(page=index,attempt=settle_attempt,
                        thumbs=[row[1] for row in geometries],geometries=geometries))
                    sink=getattr(context,'gkh_hif_status_failure_sink',None)
                    if callable(sink):
                        sink(frames[-1])
                    if settle_attempt == 7:
                        triplet_sink=getattr(context,'gkh_hif_status_failure_triplet_sink',None)
                        if callable(triplet_sink):
                            triplet_sink(frames)
                        raise ValueError('status panel did not stop scrolling within observation budget')
                    time.sleep(.08)
                pages.append(page)
                event['pages'].append(dict(frame_ids=[frame_identifier(frame) for frame in frames],
                                          observations=[asdict(item) for item in observations]))
                _join_status_panel_prefix(pages)
                if page.at_bottom:
                    break
                if index == 11:
                    raise ValueError('status list page budget reached')
                image,captured_at = fresh()
                # Recheck the same overlay and scrollbar before scrolling.
                panel_anchor(image,closing=True)
                geometry = status_panel_geometry(image,require_complete_layout=False)
                if abs(geometry[1][0]-page.thumb[0]) > 3:
                    raise ValueError('status list moved before scroll')
                distance=max(24,min(160,(page.rows[-1].box[1]-40)//2))
                dispatch(captured_at,'scroll',scroll_distance=distance)
                completed_scroll = dict(geometry=geometry,content_key=page.content_key,
                    distance=distance,input_index=len(event['inputs'])-1,completed=True)
            rows = join_status_panel_pages(pages)
            event['rows'] = [asdict(row) for row in rows]
        except (ValueError,RuntimeError) as error:
            failure = str(error)
            event['read_failure'] = failure
            if resume_open_panel is not None:
                raise
        # Close only the verified status overlay, even after a read failure.
        image,captured_at = fresh()
        close = panel_anchor(image,closing=True)
        if resume_open_panel is not None:
            geometry = status_panel_geometry(image,require_complete_layout=False)
            if not geometry[3] or abs(geometry[1][0]-pages[-1].thumb[0]) > 3:
                raise ValueError('status list moved before closing the resumed bottom page')
        dispatch(captured_at,'close',close)
        after_frames,after = capture_initial_live_checkpoint(capture_frames,checkpoint,event)
        event['after_frame_ids'] = after['frame_ids']
        if any(json.dumps(after[key]) != json.dumps(before[key]) for key in ('families','fields')):
            raise ValueError('status inspection changed the Live hand or resources')
        event['returned_unchanged'] = True
        if failure:
            raise ValueError(failure)
        event.update(accepted=True,elapsed_ms=(time.perf_counter()-started)*1000)
        if return_checkpoint is not None:
            # Pass the actual accepted observation within this call. Frame IDs
            # in the persisted event are not enough to recreate a checkpoint.
            return_checkpoint(after_frames, after, event)
        return tuple(pages),events
    except Exception as error:
        event['accepted'] = False
        event['reason'] = str(error)
        if getattr(error,'events',None):
            event['failure_events'] = error.events
        event['elapsed_ms'] = (time.perf_counter()-started)*1000
        raise StatusInspectionError(str(error),events) from error
