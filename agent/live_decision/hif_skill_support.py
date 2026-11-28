"""Read one already-selected card's support details without sending input.

The returned receipt is observation evidence only. A separate action-bound
projection may consume verified effects; this module never changes a model.
"""

import re
import json
import math
from copy import deepcopy
from types import SimpleNamespace
from dataclasses import asdict

import cv2
import numpy as np

if __package__.startswith('agent.'):
    from agent.utils.effect_text import normalize_score_effect_text
else:
    from utils.effect_text import normalize_score_effect_text

from .layout import calibrated_live_client_size
from .upgrade import compact_ocr_text, normalize_upgrade_title
from .inventory import _inventory_ocr, inventory_white_text
from .hif_visual import locate_hif_selected_detail_title
from .high_score.contracts import _jsonable
from .high_score.skill_support import supported_definition

SOURCE = 'gakumas-core@88d73149d926c5a5d1cc3f520e73d0efe4b80a68/src/data/cards.ts'
_SUPPORTED = {
    '145': dict(name='成就', source_id='145:++', cost=9,
        source='native-hif-0064-step6-20261111:145:++',
        body=r'好調7ターン次のターン、スコア\+44試験・ステージ中1回',
        effects=[dict(kind='GOOD_CONDITION', amount=7),
            dict(kind='SCHEDULE', phase='TURN', delay_turns=1, uses=1, score=44)], limited=True),
    '597@54=1': dict(name='破竹の勢い', source_id='597@54=1:++', cost=2,
        source='native-hif-1969:597@54=1:++',
        body=r'好調が8ターン以上の場合、使用可集中\+10以降の4ターンの間、ターン開始時、スコア\+7試験・ステージ中1回',
        effects=[dict(kind='CONCENTRATION', amount=10),
            dict(kind='SCHEDULE', phase='START_OF_TURN', uses=4, score=7)], limited=True),
    '762': dict(name='脚光', source_id='762:++', cost=2,
        source='native-hif-1779:762:++', cost_kind='GOOD_CONDITION',
        typed_cost=r'好調消費(\d{1,2})ターン',
        body=r'好調消費2ターン集中\+7スキルカード使用数追加\+1'
             r'以降、好調が20ターン以下の場合、スキルカード使用後2回ごとに、'
             r'スコア\+7\(集中効果を4倍適用\)・好調減少3試験・ステージ中1回',
        effects=[dict(kind='CONCENTRATION', amount=7), dict(kind='EXTRA_ACTION', amount=1),
            dict(kind='SCHEDULE', phase='AFTER_CARD_USED', uses=None, score=7,
                score_concentration_multiplier=4, good_condition_delta=-3,
                counter_good_condition_max=20, counter_min=2, counter_modulus=2)], limited=True),
    '597@14=1,54=1': dict(name='破竹の勢い', source_id='597@14=1,54=1:++', cost=2,
        source='native-hif-1074-1075:597@14=1,54=1:++',
        body=r'好調が8ターン以上の場合、使用可集中\+10以降の4ターンの間、ターン開始時、スコア\+7好調2ターン試験・ステージ中1回',
        effects=[dict(kind='CONCENTRATION', amount=10),
            dict(kind='SCHEDULE', phase='START_OF_TURN', uses=4, score=7),
            dict(kind='GOOD_CONDITION', amount=2)], limited=True),
    '29': dict(name='準備運動', source_id='jumbiundo', cost=3,
        body=r'スコア\+10集中\+3',
        effects=[dict(kind='SCORE', amount=10), dict(kind='CONCENTRATION', amount=3)]),
    '35': dict(name='ハイタッチ', source_id='haitatchi', cost=3,
        body=r'スコア\+24\(集中効果を2倍適用\)(?:レッスン中1回|試験・ステージ中1回)',
        effects=[dict(kind='SCORE', amount=24, score_concentration_multiplier=2)], limited=True),
    '49': dict(name='気合十分！', source_id='kiaijubun', cost=0,
        body=r'元気\+4消費体力減少(?:3ターン|\(3ターン\))',
        effects=[dict(kind='GENKI', amount=4), dict(kind='HALF_COST_TURNS', amount=3)]),
    '55': dict(name='楽観的', source_id='rakkanteki', cost=1,
        body=r'好調(?:4ターン|\(4ターン\))好調状態の場合、?集中\+1',
        effects=[dict(kind='GOOD_CONDITION', amount=4),
                 dict(kind='CONCENTRATION', amount=1, condition='GOOD_CONDITION_MIN', threshold=1)]),
    '57': dict(name='深呼吸', source_id='shinkokyu', cost=3,
        body=r'集中\+4集中が3以上の場合、?好調(?:3ターン|\(3ターン\))',
        effects=[dict(kind='CONCENTRATION', amount=4),
                 dict(kind='GOOD_CONDITION', amount=3, condition='CONCENTRATION_MIN', threshold=3)]),
    '87': dict(name='シュプレヒコール', source_id='shupurehikoru', cost=2,
        cost_kind='CONCENTRATION', typed_cost=r'集中消費(\d{1,2})',
        body=r'集中消費2スコア\+12好調(?:3ターン|\(3ターン\))スキルカード使用数追加\+1',
        effects=[dict(kind='SCORE', amount=12), dict(kind='GOOD_CONDITION', amount=3),
                 dict(kind='EXTRA_ACTION', amount=1)]),
    '117': dict(name='存在感', source_id='117:++', cost=1,
        source='native-hif-0052-20261111:117:++',
        cost_kind='GOOD_CONDITION', typed_cost=r'好調消費(\d{1,2})ターン',
        body=r'好調消費1ターン集中\+6スキルカード使用数追加\+1',
        effects=[dict(kind='CONCENTRATION', amount=6), dict(kind='EXTRA_ACTION', amount=1)]),
    '116': dict(name='存在感', source_id='116->117', cost=1,
        source='assets/data/skill_cards.csv:116->117', enhancement_count=1,
        require_support_label=True, require_permanent_inventory=True,
        selected_support_visual=True,
        cost_kind='GOOD_CONDITION', typed_cost=r'好調消費(\d{1,2})ターン',
        body=r'好調消費1ターン集中\+5スキルカード使用数追加\+1',
        effects=[dict(kind='CONCENTRATION', amount=5), dict(kind='EXTRA_ACTION', amount=1)]),
    '288': dict(name='ひと呼吸', source_id='288->289', cost=7,
        source='assets/data/skill_cards.csv:288->289', enhancement_count=1,
        require_support_label=True,
        body=r'好調(?:4ターン|\(4ターン\))集中\+5(?:レッスン中1回|試験・ステージ中1回)',
        effects=[dict(kind='GOOD_CONDITION', amount=4), dict(kind='CONCENTRATION', amount=5)],
        limited=True),
    '707': dict(name='スリリング', source_id='707:++', cost=1,
        source='native-hif-1984:707:++',
        body=r'好調7ターンスキルカード使用数追加\+1スキルカードを引く眠気を山札のランダムな位置に生成試験・ステージ中1回',
        effects=[dict(kind='GOOD_CONDITION', amount=7), dict(kind='EXTRA_ACTION', amount=1),
                 dict(kind='DRAW', amount=1), dict(kind='ADD_CARD_TO_DECK', card_id='23')], limited=True),
    '709': dict(name='大胆不敵', source_id='709:++', cost=2,
        source='native-hif-1117-1118:709:++',
        body=r'好調(?:3ターン|\(3ターン\))集中\+5スキルカード使用数追加\+1眠気を山札のランダムな位置に生成',
        effects=[dict(kind='GOOD_CONDITION', amount=3), dict(kind='CONCENTRATION', amount=5),
                 dict(kind='EXTRA_ACTION', amount=1), dict(kind='ADD_CARD_TO_DECK', card_id='23')]),
    '818': dict(name='あなたがくれた夢', source_id='818:++', cost=3,
        source='native-hif-1229:818:++', cost_kind='STAMINA', typed_cost=r'体力消費(\d{1,2})',
        body=r'体力消費3試験・ステージ開始時手札に入る最大体力の20%分体力回復'
             r'集中[増增]加量追加\+2\(2ターン\)体力が80%以上の場合、スキルカード使用数追加\+1',
        effects=[dict(kind='STAMINA', amount=0, value_source='MAX_STAMINA', value_scale=.2),
                 dict(kind='CONCENTRATION_ADDITION_BUFF', amount=2, duration=2),
                 dict(kind='EXTRA_ACTION', amount=1, condition='STAMINA_RATIO_MIN', threshold=.8)]),
}


def _text(value):
    return compact_ocr_text(value)


def _panel_bounds(image, title):
    """Recover the same white component as the existing selected-title locator."""
    tx, ty, tw, _ = title
    # The title locator only needs the upper panel. 918's complete body extends
    # below its y=830 crop, so follow the same component up to the hand row.
    mask = (image[260:884,70:650,:3].min(axis=2) > 225).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    matches = [(int(x)+70,int(y)+260,int(w),int(h)) for x,y,w,h,_area in stats[1:]
               if (int(x)+70,int(y)+260,int(w)) == (tx-16,ty-6,tw+84)]
    if len(matches) != 1 or matches[0][1]+matches[0][3] >= 884:
        raise ValueError('selected support detail panel is incomplete')
    return matches[0]


def _read_rows(context, image, roi, node, **params):
    x,y,w,h = roi
    rows = _inventory_ocr(context,image[y:y+h,x:x+w].copy(),node,**params)
    # Colored effect labels can have a slightly taller OCR box than the
    # condition immediately to their left. Read shared text lines left-right.
    lines = []
    for row in sorted(rows,key=lambda row:(row.box[1],row.box[0])):
        if lines and any(min(row.box[1]+row.box[3],previous.box[1]+previous.box[3])
                         - max(row.box[1],previous.box[1]) >= min(row.box[3],previous.box[3])*.5
                         for previous in lines[-1]):
            lines[-1].append(row)
        else:
            lines.append([row])
    return [dict(text=str(row.text), box=[int(row.box[0])+x,int(row.box[1])+y,
        int(row.box[2]),int(row.box[3])], confidence=float(row.score))
        for line in lines for row in sorted(line,key=lambda row:row.box[0])]


def _qualified(rows, roi):
    x,y,w,h = roi
    return bool(rows) and all(type(row['confidence']) in (int,float)
        and math.isfinite(row['confidence']) and .95 <= row['confidence'] <= 1
        and isinstance(row['text'],str) and row['text']
        and len(row['box']) == 4 and all(type(value) is int for value in row['box'])
        and x <= row['box'][0] and y <= row['box'][1]
        and row['box'][2] > 0 and row['box'][3] > 0
        and row['box'][0]+row['box'][2] <= x+w and row['box'][1]+row['box'][3] <= y+h
        for row in rows)


def _read_separated_support_duration_line(context, image, row, body_roi):
    """Read both sides of one observed blank gap; never supply catalog text."""
    from .status_panel import _normalize_status_text

    text = _normalize_status_text(row['text'])
    if (not re.fullmatch(r'集中増加量追加\+\d+\(\d+ターン\)', text)
            or not .85 <= row['confidence'] < .95):
        return None
    x,y,w,h = row['box']
    bx,by,bw,bh = body_roi
    # Keep the actual left edge clear of the adjacent blue icon; two pixels
    # to the right retain the closing parenthesis's antialiased outer edge.
    roi = (x,y-2,w+2,h+4)
    if (not 16 <= h <= 48 or not 80 <= w <= 480
            or x < bx or y-2 < by or x+w+2 > bx+bw or y+h+2 > by+bh):
        return None
    crop = image[roi[1]:roi[1]+roi[3],x:x+roi[2]]
    ink = crop.min(axis=2) < 200
    edges = np.diff(np.r_[False,ink.any(axis=0),False].astype(np.int8))
    spans = list(zip(np.flatnonzero(edges == 1),np.flatnonzero(edges == -1)))
    if (len(spans) < 2 or spans[0][0] < 2 or spans[-1][1] > crop.shape[1]-2):
        return None
    gaps = [(int(left[1]),int(right[0])) for left,right in zip(spans,spans[1:])
            if right[0]-left[1] >= 12]
    if len(gaps) != 1:
        return None
    left,right = gaps[0]
    ranges = ((0,left+2),(right-2,crop.shape[1]))
    if any(not 40 <= end-start <= 360 for start,end in ranges):
        return None
    parts = []
    for index,(start,end) in enumerate(ranges):
        found = _inventory_ocr(context,crop[:,start:end].copy(),
            f'HifSkillSupportDurationPart{index}',only_rec=True)
        if (len(found) != 1 or not math.isfinite(found[0].score) or not .95 <= found[0].score <= 1
                or tuple(found[0].box) != (0,0,end-start,crop.shape[0])):
            return None
        parts.append(dict(text=found[0].text,confidence=float(found[0].score),
            box=[x+start,roi[1],end-start,roi[3]]))
    joined = ''.join(part['text'] for part in parts)
    if _normalize_status_text(joined) != text:
        return None
    return dict(row=dict(row,text=joined,confidence=min(part['confidence'] for part in parts)),
        evidence=dict(original_row=deepcopy(row),roi=list(roi),blank_gap=[left,right],parts=parts))


def _cost_match(rows, roi):
    return (re.fullmatch(r'[-−]?(\d{1,2})',_text(rows[0]['text']))
        if len(rows) == 1 and _qualified(rows,roi) else None)


def _read_printed_cost(context, image, roi, *, require_signed=False):
    original = _read_rows(context,image,roi,'HifSkillSupportDetailCost')
    if (_cost_match(original,roi) is not None
            and (not require_signed or _text(original[0]['text']).startswith(('-', '−')))):
        return original, original, 'original'
    x,y,w,h = roi
    # The white popup backing touches the ROI edge; the complete white signed
    # numeral is enclosed by its colored heart. Keep enclosed lettering only.
    white = inventory_white_text(image[y:y+h,x:x+w].copy())[2:-2,2:-2,0]
    _, labels, stats, _ = cv2.connectedComponentsWithStats((white == 0).astype(np.uint8))
    clean = np.full((h,w),255,np.uint8)
    for label,(sx,sy,sw,sh,_area) in enumerate(stats[1:],1):
        if sx > 0 and sy > 0 and sx+sw < w and sy+sh < h:
            clean[labels == label] = 0
    text_image = image.copy()
    text_image[y:y+h,x:x+w] = cv2.cvtColor(clean,cv2.COLOR_GRAY2BGR)
    rows = _read_rows(context,text_image,roi,'HifSkillSupportDetailCostWhite',only_rec=True)
    match = _cost_match(rows,roi)
    previous = [re.fullmatch(r'[-−]?(\d{1,2})',_text(row['text'])) for row in original]
    if (match is not None
            and (not require_signed or _text(rows[0]['text']).startswith(('-', '−')))
            and all(old is None or int(old[1]) == int(match[1]) for old in previous)):
        return rows, original, 'enclosed_white_text'
    return original, original, 'original_unresolved'


def read_complete_zero_cost(context, image, roi):
    """Keep one complete zero glyph; reject signs, cut glyphs and side ink."""
    x,y,w,h = roi
    mask = (image[y:y+h,x:x+w].min(axis=2) >= 220).astype(np.uint8)
    _,labels,stats,_ = cv2.connectedComponentsWithStats(mask)
    glyphs = [(i,tuple(map(int,s))) for i,s in enumerate(stats[1:],1)
              if 7 <= s[2] <= 24 and 12 <= s[3] <= 26]
    if len(glyphs) != 1:
        return None
    index,(gx,gy,gw,gh,_area) = glyphs[0]
    if not (gx > 0 and gy > 0 and gx+gw < w and gy+gh < h):
        return None
    decoration = [tuple(map(int,s)) for i,s in enumerate(stats[1:],1) if i != index]
    if any(not (sw <= 2 and sh <= 3 and area <= 4 and sy+sh <= gy)
           for _sx,sy,sw,sh,area in decoration):
        return None
    glyph = (labels[gy:gy+gh,gx:gx+gw] == index).astype(np.uint8)
    _,hierarchy = cv2.findContours(glyph,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or sum(row[3] >= 0 for row in hierarchy[0]) != 1:
        return None
    clean = cv2.cvtColor(cv2.copyMakeBorder(255-glyph*255,4,4,4,4,
        cv2.BORDER_CONSTANT,value=255),cv2.COLOR_GRAY2BGR)
    rows = _inventory_ocr(context,clean,'HifSelectedCostCompleteZero',only_rec=True)
    if len(rows) != 1 or rows[0].text.strip() != '0' or not .95 <= rows[0].score <= 1:
        return None
    return dict(text='0',confidence=rows[0].score,roi=list(roi),
        glyph_box=[x+gx,y+gy,gw,gh],decoration_components=decoration,
        source='selected_card_face_complete_zero')


def _effect_body(rows, panel):
    text = ''.join(_text(row['text']) for row in rows
        if not (_text(row['text']) in ('Mental','Active') and row['box'][1] < panel[1]+80)
        and _text(row['text']) != 'スキルカードサポート')
    return normalize_score_effect_text(text)


def _original_support_body_complete(title_rows, title_roi, body_rows, body_roi, panel):
    if len(title_rows) != 1 or not _qualified(title_rows,title_roi) or not _qualified(body_rows,body_roi):
        return False
    title = _text(title_rows[0]['text'])
    specification = next((entry for entry in _SUPPORTED.values()
        if title == _text(entry['name'])+'+'*entry.get('enhancement_count',2)), None)
    if specification is None or (specification.get('require_support_label')
            and not specification.get('selected_support_visual') and sum(
                _text(row['text']) == 'スキルカードサポート' for row in body_rows) != 1):
        return False
    return re.fullmatch(
        specification['body'],_effect_body(body_rows,panel)) is not None


def revalidate_original_support_body_observations(observations):
    """Select exact original text in a copy for subsequent full receipt replay."""
    result = deepcopy(list(observations))
    for row in result:
        if not _original_support_body_complete(row['title_rows'],row['title_roi'],
                row['raw_body_rows'],row['body_roi'],row['panel_box']):
            raise ValueError('original support body is not complete qualified fixed text')
        row['prior_body_rows'] = row['body_rows']
        row['body_rows'] = deepcopy(row['raw_body_rows'])
        row['body_source'] = 'raw_body_rows_selected_via_new_validator'
        row['text_accepted'] = True
    return result


def _without_detail_pictograms(image, panel):
    """Exclude observed cyan diamonds, and their isolated left bullet column."""
    x,y,w,h = panel
    top = y+80
    hsv = cv2.cvtColor(image[top:y+h,x:x+w],cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(85,100,120),(112,255,255))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned, boxes = image.copy(), []
    for ix,iy,iw,ih,area in stats[1:]:
        if not (24 <= iw <= 44 and 22 <= ih <= 44 and area/(iw*ih) >= .2):
            continue
        bx,by = int(x+ix),int(top+iy)
        leading_ink = np.any(image[by:by+int(ih),x+12:x+38,:3].min(axis=2) < 170)
        if 65 <= ix <= 90 and leading_ink:
            # A wrapped line can start with text before an inline diamond.
            # Keep that complete line for OCR instead of splitting its prefix.
            continue
        box = [bx-1,by-1,int(iw)+2,int(ih)+2]
        boxes.append(box)
        cleaned[box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 255
        if 65 <= ix <= 90:
            # 906's ★ is the gray bullet immediately left of this independently
            # found first-column diamond. Wrapped text without a diamond stays.
            # 1779 wraps "+7 (" before an inline diamond in the same column.
            # Real leading text rules out a standalone bullet; do not erase it.
            bullet = [x+38,by,x+65-(x+38),int(ih)]
            boxes.append(bullet)
            cleaned[bullet[1]:bullet[1]+bullet[3],bullet[0]:bullet[0]+bullet[2]] = 255
    return cleaned, boxes


def _resampled_separator_star(shape):
    """A resampled four-point star has solid, tapering rows and columns."""
    height,width = shape.shape
    if not (7 <= width <= 13 and 7 <= height <= 15 and abs(width-height) <= 2):
        return False
    for oriented in (shape,shape.T):
        counts = oriented.sum(axis=1)
        peak = int(counts.argmax())
        if (abs(peak-(len(counts)-1)/2) > 1 or counts[peak] != oriented.shape[1]
                or max(counts[0],counts[-1]) > 3
                or np.any(np.diff(counts[:peak+1]) < 0) or np.any(np.diff(counts[peak:]) > 0)):
            return False
        for line in oriented:
            ink = np.flatnonzero(line)
            if (not len(ink) or len(ink) != ink[-1]-ink[0]+1
                    or abs((ink[0]+ink[-1])/2-(len(line)-1)/2) > 1):
                return False
    return True


def _support_header_decoration_boxes(crop, panel, rows, *, resampled=False):
    """Prove both separator stars, independently of their OCR spelling."""
    x,y,w,_h = panel
    roi = (x+12,y+50,w-24,24)
    if crop.dtype != np.uint8 or crop.shape != (24,w-24,3):
        return []
    labels = [row for row in rows if _text(row['text']) in ('Active','Mental')]
    if (len(labels) != 1 or not _qualified(labels,roi)
            or abs(labels[0]['box'][0]+labels[0]['box'][2]/2-(x+w/2)) > 8):
        return []
    low,high = crop.min(axis=2),crop.max(axis=2)
    if np.any(high-low > 8) or low.min() < 170:
        return []
    _,components,stats,_ = cv2.connectedComponentsWithStats((low < 240).astype(np.uint8))
    tx,ty,tw,th = labels[0]['box']
    tx,ty = tx-roi[0],ty-roi[1]
    # At the calibrated small size the final antialiased label row can
    # extend one pixel below its OCR box (0032 Mental).
    label_bottom = ty+th+int(resampled)
    parts = [(index,tuple(map(int,stat))) for index,stat in enumerate(stats[1:],1)
        if not (tx <= stat[0] and ty <= stat[1]
            and stat[0]+stat[2] <= tx+tw and stat[1] < ty+th
            and stat[1]+stat[3] <= label_bottom)]
    lines = sorted((part for part in parts if part[1][2] >= 100),key=lambda part:part[1][0])
    stars = sorted((part for part in parts if part not in lines),key=lambda part:part[1][0])
    if len(lines) != 2 or len(stars) != 2:
        return []
    for _index,(sx,sy,sw,sh,area) in lines:
        if not (w*.3 <= sw <= w*.4 and 1 <= sh <= 3 and area == sw*sh):
            return []
    left,right = lines[0][1],lines[1][1]
    if (left[1:4] != right[1:4] or not left[0]+left[2] < tx < tx+tw < right[0]
            or not 4 <= tx-left[0]-left[2] <= 36
            or not 4 <= right[0]-tx-tw <= 36):
        return []
    boxes,shapes = [],[]
    for index,(sx,sy,sw,sh,_area) in stars:
        pad = 1 if resampled else 2
        xpad = 2
        if not ((resampled or 7 <= sw == sh <= 11 and sw % 2 == 1)
                and xpad <= sx and sx+sw+xpad <= w-24 and pad <= sy and sy+sh+pad <= 24
                and abs(sy+sh/2-(left[1]+left[3]/2)) <= 1):
            return []
        shape = components[sy:sy+sh,sx:sx+sw] == index
        if resampled:
            if not _resampled_separator_star(shape):
                return []
            # A pale tip can reach the original white rim. Prove the entire
            # lighter silhouette is still a four-point star before masking it.
            ink = low[sy:sy+sh,sx-xpad:sx+sw+xpad] < 254
            if ink[:,[0,-1]].any():
                ix,iy,iw,ih = cv2.boundingRect(ink.astype(np.uint8))
                shape = ink[iy:iy+ih,ix:ix+iw]
                xpad += 1
                outward_left = shape if sx < crop.shape[1]//2 else shape[:,::-1]
                if sx < xpad or sx+sw+xpad > w-24 or not _resampled_separator_star(outward_left):
                    return []
        elif (not np.array_equal(shape,shape[::-1]) or not np.array_equal(shape,shape[:,::-1])
                or not shape[sh//2].all() or not shape[:,sw//2].all()
                or not 1 <= int(shape[0].sum()) <= 3 or shape[0,0]):
            return []
        patch = crop[sy-pad:sy+sh+pad,sx-xpad:sx+sw+xpad]
        if min(patch[0].min(),patch[-1].min(),patch[:,0].min(),patch[:,-1].min()) < 254:
            return []
        # Small-client OCR may include one blank column beyond the white rim.
        margin = xpad+int(resampled)
        boxes.append([roi[0]+sx-margin,roi[1]+sy-2,sw+2*margin,sh+4])
        shapes.append(shape)
    matching_stars = np.array_equal(*shapes) or resampled and np.array_equal(shapes[0],shapes[1][:,::-1])
    if (not matching_stars or stars[0][1][0] > 24 or stars[1][1][0] < w-24-24
            or not 3 <= left[0]-sum(stars[0][1][0:3:2]) <= 12
            or not 3 <= stars[1][1][0]-right[0]-right[2] <= 12):
        return []
    for row in rows:
        rx,ry,rw,rh = row['box']
        if ry < roi[1]+roi[3] and row != labels[0] and not any(
                bx <= rx and by <= ry and rx+rw <= bx+bw and ry+rh <= by+bh
                for bx,by,bw,bh in boxes):
            return []
    return boxes


def _read_detail_title(context, image, title_roi, panel):
    rows = _read_rows(context,image,title_roi,'HifSkillSupportDetailTitle')
    if len(rows) == 1 and _qualified(rows,title_roi):
        return rows, rows, []
    if (len(rows) == 1 and .85 <= rows[0]['confidence'] < .95
            and _qualified([dict(rows[0],confidence=1)],title_roi)):
        # Keep every dark title pixel; remove only the wide white side margins.
        # At small native resolutions those margins change the OCR line scale.
        tx,ty,tw,th = title_roi
        crop = image[ty:ty+th,tx:tx+tw,:3]
        ink = crop.min(axis=2) < 200
        ix,iy,iw,ih = cv2.boundingRect(ink.astype(np.uint8))
        if (iw > 0 and ix >= 8 and ix+iw <= tw-8 and iy >= 2 and iy+ih <= th-2
                and iw+16 < tw):
            trimmed = (tx+ix-8,ty,iw+16,th)
            reread = _read_rows(context,image,trimmed,'HifSkillSupportCompleteTitleTrim')
            if (len(reread) == 1 and _qualified(reread,trimmed)
                    and _text(reread[0]['text']) == _text(rows[0]['text'])):
                return [dict(reread[0],acquisition_mode='complete_title_horizontal_trim',
                             source_roi=list(trimmed))], rows, []
            if len(reread) == 1 and _text(reread[0]['text']) != _text(rows[0]['text']):
                return rows, rows, []
        separated = _read_title_with_observed_suffix(context, image, title_roi, rows[0])
        if separated is not None:
            return [separated], rows, []
    box = detail_cost_icon_box(image, panel)
    if box is None:
        return rows, rows, []
    title = [row for row in rows if _qualified([row],title_roi)
             and row['box'][0]+row['box'][2] <= box[0]]
    noise = [row for row in rows if row not in title]

    if len(title) != 1 or not noise:
        return rows, rows, []
    # A second detection can cover the right end of the same printed title
    # (0036). Its padding must contain no other ink. Only the cost diamond is
    # masked; the full title, including every suffix pixel, is read again.
    icons = [detail_row_within_region(image, row['box'], box) for row in noise]
    if not any(icons) or any(not inside and not detail_row_within_region(image, row['box'], title[0]['box'])
                             for row,inside in zip(noise,icons,strict=True)):
        return rows, rows, []
    cleaned = image.copy()
    cleaned[box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 255
    reread = _read_rows(context,cleaned,title_roi,'HifSkillSupportDetailTitleText')
    if (len(reread) != 1 or not _qualified(reread,title_roi)
            or _text(reread[0]['text']) != _text(title[0]['text'])
            or reread[0]['box'][0]+reread[0]['box'][2] > box[0]):
        return rows, rows, []
    return reread, rows, [box]


def detail_cost_icon_box(image, panel):
    """1036's concentration-cost diamond crossing the title ROI's right edge."""
    x,y,w,_h = panel
    left,top = x+w-96,y+4
    hsv = cv2.cvtColor(image[top:top+46,left:left+54],cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(85,100,120),(112,255,255))
    ix,iy,iw,ih = cv2.boundingRect(mask)
    if 28 <= iw <= 44 and 28 <= ih <= 44 and cv2.countNonZero(mask)/(iw*ih) >= .2:
        return [left+ix-3,top+iy-3,iw+6,ih+6]
    return None


def detail_row_within_region(image, row_box, box):
    """OCR padding may extend beyond an observed ink region only over pure white."""
    rx,ry,rw,rh = row_box
    bx,by,bw,bh = box
    if not (0 <= rx < rx+rw <= image.shape[1] and 0 <= ry < ry+rh <= image.shape[0]
            and max(rx,bx) < min(rx+rw,bx+bw) and max(ry,by) < min(ry+rh,by+bh)):
        return False
    outside = np.ones((rh,rw),dtype=bool)
    outside[max(0,by-ry):min(rh,by+bh-ry),max(0,bx-rx):min(rw,bx+bw-rx)] = False
    return bool(np.all(image[ry:ry+rh,rx:rx+rw,:3][outside] >= 250))


def _read_title_with_observed_suffix(context, image, roi, original):
    """Read the complete name and independently visible plus suffix separately.

    No catalog text is supplied. Every suffix pixel must belong to a complete
    cross; the independent name still needs the normal OCR confidence gate.
    """
    from .opening_support import _cross, _components

    text = _text(original['text'])
    count = len(text)-len(text.rstrip('+'))
    if not count or count == len(text):
        return None
    x,y,w,h = roi
    mask = image[y:y+h,x:x+w,:3].min(axis=2) < 200
    ix,iy,iw,ih = cv2.boundingRect(mask.astype(np.uint8))
    if not (ix >= 8 and ix+iw <= w-8 and iy >= 2 and iy+ih <= h-2):
        return None
    parts = sorted(_components(mask,(x,y)),key=lambda row:row['box'][0])
    suffix = []
    while parts and _cross(parts[-1],7,19):
        suffix.insert(0,parts.pop())
    if len(suffix) != count or not parts:
        return None
    first = suffix[0]['box'][0]
    name_right = max(row['box'][0]+row['box'][2] for row in parts)
    if not 2 <= first-name_right <= 24:
        return None
    for index,part in enumerate(suffix):
        sx,sy,sw,sh = part['box']
        if (min(part['row_coverage'],part['column_coverage']) < .95
                or not (x+ix <= sx and y+iy <= sy and sy+sh <= y+iy+ih)
                or index and not 2 <= sx-sum(suffix[index-1]['box'][::2]) <= 16):
            return None
    # Split only inside the observed blank gap. Neither side loses ink.
    boundary = (name_right+first)//2
    base_roi = (x+ix-8,y,boundary-(x+ix-8),h)
    base = _read_rows(context,image,base_roi,'HifSkillSupportTitleWithoutSuffix')
    if (len(base) != 1 or not _qualified(base,base_roi)
            or _text(base[0]['text']) != text[:-count]):
        return None
    bx,by,bw,bh = base[0]['box']
    right = max(bx+bw,*(p['box'][0]+p['box'][2] for p in suffix))
    top = min(by,*(p['box'][1] for p in suffix))
    bottom = max(by+bh,*(p['box'][1]+p['box'][3] for p in suffix))
    return dict(text=base[0]['text']+'+'*count,box=[bx,top,right-bx,bottom-top],
        confidence=min(base[0]['confidence'],*(min(p['row_coverage'],p['column_coverage'])
            for p in suffix)),acquisition_mode='complete_name_and_observed_plus_suffix',
        source_roi=list(base_roi),base_title_rows=base,suffix_components=suffix)


def _read_selected_support_visual(context, image, *, frame_id, selected_slot, hand_count, title):
    """1199: move the raised four-card row back to the existing badge geometry.

    Only coordinates change. The original blue suffix, title and portrait plus
    are all read through the established support reader and its original gates.
    """
    from .opening_support import _verify_frame, _read_support_frame

    if (tuple(image.shape) != (1280, 720, 3) or hand_count != 4
            or type(selected_slot) is not int or not 0 <= selected_slot < 4):
        raise ValueError('selected support visual requires the verified four-card row')
    anchor, dy = round(18+selected_slot*494/3), 26
    normalized = image.copy()
    normalized[dy:] = image[:-dy]
    rows = _read_rows(context, normalized, (anchor, 1090, 190, 44),
        'HifSelectedSupportCardTitle')
    if len(rows) != 1:
        raise ValueError('selected support card title is not unique')
    row = rows[0]
    mark = _read_support_frame(normalized, frame_id=frame_id, slot=selected_slot,
        title_row=SimpleNamespace(text=row['text'], score=row['confidence'], box=row['box']),
        anchor=anchor)
    try:
        _verify_frame(mark, selected_slot, anchor=anchor, title_text=title, suffix_right_margin=6)
    except ValueError:
        px, py, pw, ph = mark['title_plate']
        x, y, w, h = row['box']
        #1205: the broad OCR ROI included one pixel above the actual plate,
        # producing a33px box for an otherwise exact title. Read the complete
        # physical plate; never clamp the old box or expand the verifier gate.
        if not (row['text'] == title and row['confidence'] >= .95
                and px <= x < x+w <= px+pw and y < py and h > ph
                and y+h <= py+ph):
            raise
        reread = _read_rows(context, normalized, tuple(mark['title_plate']),
            'HifSelectedSupportPlateTitle')
        if len(reread) != 1 or reread[0]['text'] != row['text']:
            raise ValueError('selected support complete plate title is unresolved') from None
        mark['title_before_plate_reread'] = mark['title']
        mark['title_reread_roi'] = list(mark['title_plate'])
        mark['title'] = dict(text=reread[0]['text'], score=reread[0]['confidence'],
            box=list(reread[0]['box']))
        _verify_frame(mark, selected_slot, anchor=anchor, title_text=title, suffix_right_margin=6)
    return dict(normalization_y=dy, hand_count=hand_count, observation=mark)


def _verify_selected_support_visual(observed, *, selected_slot, hand_count, title):
    from .opening_support import _verify_frame

    if hand_count != 4:
        raise ValueError('selected support visual requires the verified four-card row')
    anchor = round(18+selected_slot*494/3)
    for row in observed:
        visual = row.get('selected_support_visual') or {}
        mark = visual.get('observation') or {}
        if (visual.get('normalization_y') != 26 or visual.get('hand_count') != hand_count
                or mark.get('frame_id') != row['frame_id']):
            raise ValueError('selected support lacks current complete visual evidence')
        _verify_frame(mark, selected_slot, anchor=anchor, title_text=title, suffix_right_margin=6)


def read_selected_skill_support_frame(context, image, *, frame_id, selected_slot=None,
                                      hand_count=None):
    """Retain all detected detail text, including unresolved rows; no binding."""
    title_roi = locate_hif_selected_detail_title(image)
    if title_roi is None:
        raise ValueError('selected support detail title is not unique')
    panel = _panel_bounds(image,title_roi)
    return _read_selected_support_details(context, image, title_roi, panel, frame_id=frame_id,
        selected_slot=selected_slot, hand_count=hand_count)


def _read_selected_support_details(context, image, title_roi, panel, *, frame_id,
                                   selected_slot, hand_count, title_reading=None):
    x,y,w,h = panel
    # The component includes the speech pointer below the rounded rectangle.
    # 906 exposes background lettering beside that pointer; it is not body text.
    white_rows = (image[y:y+h,x+6:x+w-6,:3].min(axis=2) > 225).mean(axis=1)
    plate_rows = np.flatnonzero(white_rows >= .85)
    if not len(plate_rows) or plate_rows[-1] < 90:
        raise ValueError('selected support detail body plate is unresolved')
    body_bottom = y+int(plate_rows[-1])+1
    body_roi = (x+12,y+50,w-24,body_bottom-y-50)
    cost_roi = (x+w-68,y+4,62,50)
    if title_reading is None:
        title_reading = _read_detail_title(context,image,title_roi,panel)
    title_rows,raw_title_rows,title_excluded = title_reading
    raw_body_rows = _read_rows(context,image,body_roi,'HifSkillSupportDetailBody')
    raw_title = title_rows[0]['text'] if len(title_rows) == 1 else None
    normalized = _text(raw_title) if raw_title is not None else ''
    # 942's intact 35++ body was already exact; masking its inline icon
    # created a spurious OCR row. Keep complete qualified original text.
    raw_complete = _original_support_body_complete(
        title_rows,title_roi,raw_body_rows,body_roi,panel)
    excluded = []
    header_excluded = []
    body_rows, body_source = raw_body_rows, 'original'
    text_image = image
    if not raw_complete:
        header_crop = image[y+50:y+74,x+12:x+w-12]
        header_boxes = _support_header_decoration_boxes(header_crop,panel,raw_body_rows)
        if not header_boxes and calibrated_live_client_size(getattr(context,'gkh_hif_client_size',None)) == (480,853):
            header_boxes = _support_header_decoration_boxes(header_crop,panel,raw_body_rows,resampled=True)
        if header_boxes:
            text_image = image.copy()
            for bx,by,bw,bh in header_boxes:
                text_image[by:by+bh,bx:bx+bw] = 255
            candidate = _read_rows(context,text_image,body_roi,'HifSkillSupportHeaderDecorationText')
            if (_original_support_body_complete(title_rows,title_roi,candidate,body_roi,panel)
                    and [_text(line['text']) for line in raw_body_rows if line['box'][1] >= y+74]
                    == [_text(line['text']) for line in candidate if line['box'][1] >= y+74]):
                body_rows,body_source = candidate,'header_decoration_masked'
                header_excluded = header_boxes
    if not raw_complete and not header_excluded:
        text_image, excluded = _without_detail_pictograms(text_image,panel)
        if excluded:
            body_rows = _read_rows(context,text_image,body_roi,'HifSkillSupportDetailText')
            body_source = 'pictogram_masked'
            if header_boxes:
                header_excluded = header_boxes
    line_reobservations = []
    if normalized == _text(_SUPPORTED['818']['name'])+'++':
        for index,row in enumerate(body_rows):
            reread = _read_separated_support_duration_line(
                context, text_image if body_source == 'pictogram_masked' else image, row, body_roi)
            if reread is not None:
                body_rows = list(body_rows)
                body_rows[index] = reread['row']
                line_reobservations.append(reread['evidence'])
    cost_rows,raw_cost_rows,cost_source = _read_printed_cost(context,image,cost_roi)
    cost_match = _cost_match(cost_rows,cost_roi)
    plus_count = len(normalized)-len(normalized.rstrip('+'))
    result = dict(frame_id=frame_id,panel_box=list(panel),title_roi=list(title_roi),
        body_roi=list(body_roi),raw_title=raw_title,plus_count=plus_count,
        title_rows=title_rows,body_rows=body_rows,body_source=body_source,
        raw_title_rows=raw_title_rows,excluded_title_pictogram_boxes=title_excluded,
        title_source='cost_pictogram_masked' if title_excluded else 'original',
        raw_body_rows=raw_body_rows,excluded_pictogram_boxes=excluded,
        **(dict(excluded_header_decoration_boxes=header_excluded) if header_excluded else {}),
        **(dict(body_line_reobservations=line_reobservations) if line_reobservations else {}),
        printed_cost=int(cost_match[1]) if cost_match else None,
        printed_cost_roi=list(cost_roi),printed_cost_rows=cost_rows,
        raw_printed_cost_rows=raw_cost_rows,printed_cost_source=cost_source,
        title_accepted=len(title_rows) == 1 and _qualified(title_rows,title_roi),
        text_accepted=len(title_rows) == 1 and _qualified(title_rows,title_roi)
            and _qualified(body_rows,body_roi))
    visual_spec = next((entry for entry in _SUPPORTED.values()
        if entry.get('selected_support_visual')
        and normalized == _text(entry['name'])+'+'*entry.get('enhancement_count',2)), None)
    if visual_spec is not None and selected_slot is not None:
        try:
            result['selected_support_visual'] = _read_selected_support_visual(context, image,
                frame_id=frame_id, selected_slot=selected_slot, hand_count=hand_count,
                title=normalized)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            result['selected_support_visual_error'] = str(error)
    return result


def _verify_selection(frame_ids, selected_slot, selection_evidence):
    evidence_kind = selection_evidence.get('kind')
    if evidence_kind in ('upstream_selected_detail', 'target_select_detail'):
        from .hif_upstream_ui import UPSTREAM_REVISION

        count = 1 if evidence_kind == 'target_select_detail' else 2
        if (selection_evidence.get('revision') != UPSTREAM_REVISION
                or selection_evidence.get('selected_slot') != selected_slot
                or len(frame_ids) != count or len(set(frame_ids)) != count
                or any(not isinstance(value, str) or not value for value in frame_ids)
                or selection_evidence.get('frame_ids') != frame_ids
                or selection_evidence.get('select_and_detail_confirmed') is not True):
            raise ValueError('upstream support lacks its confirmed selected detail evidence')
        return
    consensus = selection_evidence.get('select_consensus') or {}
    observations = selection_evidence.get('select_evidence', ())
    if (len(frame_ids) != 3 or any(not isinstance(value,str) or not value for value in frame_ids)
            or len(set(frame_ids)) != 3
            or selection_evidence.get('frame_ids') != frame_ids
            or selection_evidence.get('geometry_stable') is not True
            or consensus.get('accepted') is not True or consensus.get('selected_slot') != selected_slot
            or list(map(list,consensus.get('observations',()))) != [[selected_slot]]*3
            or len(observations) != 3 or any(row.get('geometry_available') is not True
                or row.get('geometry_ambiguous') is not False
                or row.get('ocr_slots') != [selected_slot]
                or row.get('geometry_raised_slots') != [selected_slot]
                or row.get('resolved_slots') != [selected_slot] for row in observations)):
        raise ValueError('selected support target lacks independent unique SELECT evidence')


def _verify_variant(bundle, ref, definition):
    canonical = definition.canonical_card_id or definition.card_id
    if definition.card_id != ref.card_id:
        raise ValueError('selected support definition differs from the actual instance')
    if '@' in ref.card_id:
        keys = [row for row in bundle.inventory_keys if row[0] == ref.instance_id]
        if len(keys) != 1 or keys[0][1] != canonical or not keys[0][2]:
            raise ValueError('selected permanent variant has no exact inventory binding')
        levels = sorted(keys[0][2],key=lambda row:int(row[0]))
        expected = canonical+'@'+','.join(f'{key}={level}' for key,level in levels)
        base = bundle.kernel.cards[canonical]
        if expected != ref.card_id or definition.name != base.name:
            raise ValueError('selected permanent variant differs from its compiled title')
    elif canonical != ref.card_id:
        raise ValueError('selected canonical definition differs from the actual instance')
    name = _text(definition.name)
    count = len(name)-len(name.rstrip('+'))
    if count not in (0,1) or ref.upgraded is not bool(count):
        raise ValueError('selected ordinary enhancement differs from the actual instance')
    return canonical, name


def verify_selected_skill_support_observations(observations, *, bundle, selected_slot,
                                                selection_evidence):
    """Replay original OCR rows and target evidence; derived flags grant nothing.

    ``ordinary_title_verified`` retains the already verified ordinary model,
    including its exact permanent variant. It does not depend on support-only
    body OCR. Extra plus marks require separately verified effects.
    """
    result = dict(event='hif_selected_skill_support',accepted=False,effects_verified=False,
        observation_accepted=False,submission_authorized=False,model_updated=False,
        title_verified=False,ordinary_title_verified=False,support_required=None,
        run_id=bundle.run_id,revision=bundle.revision,selected_slot=selected_slot,
        bound_state=_jsonable(asdict(bundle.state)),
        observations=deepcopy(list(observations)),selection_evidence=deepcopy(selection_evidence))
    try:
        if type(selected_slot) is not int or not 0 <= selected_slot < len(bundle.state.deck.hand):
            raise ValueError('selected support target slot is invalid')
        ref = bundle.state.deck.hand[selected_slot]
        definition = bundle.kernel.cards[ref.card_id]
        canonical, ordinary_title = _verify_variant(bundle,ref,definition)
        result.update(card_instance_id=ref.instance_id,card_id=canonical,definition_id=ref.card_id)
        observed = result['observations']
        frame_ids = [row['frame_id'] for row in observed]
        result['frame_ids'] = frame_ids
        _verify_selection(frame_ids,selected_slot,selection_evidence)
        titles = []
        for row in observed:
            panel, roi = row['panel_box'], row['title_roi']
            if (len(panel) != 4 or any(type(value) is not int for value in panel)
                    or panel[2] <= 84 or panel[3] <= 90
                    or list(roi) != [panel[0]+16,panel[1]+6,panel[2]-84,44]
                    or len(row['title_rows']) != 1 or not _qualified(row['title_rows'],roi)):
                raise ValueError('selected complete title is unresolved')
            titles.append(_text(row['title_rows'][0]['text']))
        base_single_support = (canonical == ref.card_id and canonical in ('116', '288')
            and ordinary_title == _SUPPORTED[canonical]['name'] and titles[0] == ordinary_title+'+')
        if len(set(titles)) != 1 or (normalize_upgrade_title(titles[0]) != ordinary_title
                and not base_single_support):
            raise ValueError('selected complete title differs from the ordinary family or enhancement')
        title = titles[0]
        plus_count = len(title)-len(title.rstrip('+'))
        result.update(panel_observation_accepted=True,title_verified=True,
            raw_title=observed[0]['title_rows'][0]['text'],plus_count=plus_count)
        if title == ordinary_title:
            result.update(ordinary_title_verified=True,support_required=False,
                observation_accepted=True,reason='selected_ordinary_title_unanimous')
            return result
        result['support_required'] = True
        # Exact definition id distinguishes the one verified permanent variant.
        specification = _SUPPORTED.get(ref.card_id)
        if specification is None:
            from .high_score.support_master import cost_only_support
            try:
                _, specification = cost_only_support(definition, plus_count)
            except (ValueError, KeyError):
                pass
        if specification is None or plus_count != specification.get('enhancement_count',2):
            raise ValueError('selected extra enhancement is outside the verified slice')
        if specification.get('require_permanent_inventory'):
            matches = [key for key in bundle.inventory_keys if key[0] == ref.instance_id]
            if matches != [(ref.instance_id, ref.card_id, ())]:
                raise ValueError('selected base-card support lacks its exact permanent inventory binding')
        if ref.card_id != canonical:
            supported_definition(definition, enhancement_count=plus_count)
        signatures = []
        for row in observed:
            x,y,w,h = row['panel_box']
            body_roi, cost_roi = row['body_roi'], row['printed_cost_roi']
            if (list(body_roi[:3]) != [x+12,y+50,w-24]
                    or type(body_roi[3]) is not int or not 0 < body_roi[3] <= h-50
                    or list(cost_roi) != [x+w-68,y+4,62,50]
                    or not _qualified(row['body_rows'],body_roi)):
                raise ValueError('selected support complete body or printed cost is unresolved')
            if specification.get('typed_cost'):
                # 932's focus icon is not a stamina heart. Its explicit,
                # complete cost line independently names both resource and amount.
                matches = [match for line in row['body_rows']
                    if (match := re.fullmatch(specification['typed_cost'],_text(line['text'])))]
                match = matches[0] if len(matches) == 1 else None
            else:
                match = (re.fullmatch(r'[-−]?(\d{1,2})',_text(row['printed_cost_rows'][0]['text']))
                    if len(row['printed_cost_rows']) == 1
                    and _qualified(row['printed_cost_rows'],cost_roi) else None)
            if match is None:
                raise ValueError('selected support printed cost is unresolved')
            if row.get('printed_cost_source') == 'enclosed_white_text':
                prior = [re.fullmatch(r'[-−]?(\d{1,2})',_text(line['text']))
                    for line in row['raw_printed_cost_rows']]
                if any(old is not None and int(old[1]) != int(match[1]) for old in prior):
                    raise ValueError('selected support printed cost contradicts the original OCR')
            if (specification.get('typed_cost') and len(row['printed_cost_rows']) == 1
                    and _qualified(row['printed_cost_rows'],cost_roi)):
                header = re.fullmatch(r'[-−]?(\d{1,2})',_text(row['printed_cost_rows'][0]['text']))
                if header is not None and int(header[1]) != int(match[1]):
                    raise ValueError('selected support typed body cost contradicts the header')
            body_lines = tuple(normalize_score_effect_text(_text(line['text'])) for line in row['body_rows'])
            signatures.append((body_lines, int(match[1])))
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError('selected support complete body or printed cost changed')
        result.update(observation_accepted=True,body_lines=observed[0]['body_rows'])
        body = _effect_body(observed[0]['body_rows'],observed[0]['panel_box'])
        if (title != _text(specification['name'])+'+'*specification.get('enhancement_count',2)
                or signatures[0][1] != specification['cost']
                or re.fullmatch(specification['body'],body) is None):
            raise ValueError('selected support complete effects do not match the fixed entry')
        explicit_support_label = any(_text(row['text']) == 'スキルカードサポート'
            for row in observed[0]['body_rows'])
        if specification.get('require_support_label'):
            if any(sum(_text(line['text']) == 'スキルカードサポート'
                    for line in row['body_rows']) != 1 for row in observed):
                from .high_score.live_policy import BoundHifOpeningPolicy
                if any(_text(line['text']) == 'スキルカードサポート'
                        for row in observed for line in row['body_rows']):
                    raise ValueError('selected base-card support lacks an explicit support label')
                if specification.get('selected_support_visual'):
                    _verify_selected_support_visual(observed, selected_slot=selected_slot,
                        hand_count=len(bundle.state.deck.hand), title=title)
                    result['support_source'] = 'selected_blue_suffix_and_portrait_badge'
                elif (canonical == '288' and BoundHifOpeningPolicy(
                        bundle,action_order=())._opening_support_visible_id(ref) == '289'):
                    result['support_source'] = 'verified_opening_temporary_support'
                else:
                    raise ValueError('selected base-card support lacks an explicit support label')
            else:
                result['support_source'] = 'explicit_support_label'
            supported_definition(definition,enhancement_count=plus_count)
        result.update(accepted=True,effects_verified=True,reason='selected_support_complete_text_unanimous',
            effect_source=specification.get('source',SOURCE),source_id=specification['source_id'],
            enhancement_count=plus_count,
            cost_kind=specification.get('cost_kind','NORMAL'),printed_cost=specification['cost'],
            effects=[dict(effect) for effect in specification['effects']],
            explicit_support_label=explicit_support_label)
        if specification.get('typed_cost'):
            result['cost_source'] = 'complete_body_typed_cost'
        if specification.get('limited'):
            result['limited'] = True
    except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:
        result['reason'] = str(error)
    return result


def observe_selected_skill_support(context, frames, *, bundle, selected_slot,
                                   selection_evidence, frame_identifier):
    """Read three confirmed current-target frames without input or model changes."""
    observed = []
    try:
        frame_ids = [frame_identifier(frame) for frame in frames]
        _verify_selection(frame_ids,selected_slot,selection_evidence)
        read_options = {}
        ref = bundle.state.deck.hand[selected_slot]
        if _SUPPORTED.get(ref.card_id, {}).get('selected_support_visual'):
            read_options = dict(selected_slot=selected_slot, hand_count=len(bundle.state.deck.hand))
        for frame,key in zip(frames,frame_ids,strict=True):
            observed.append(read_selected_skill_support_frame(context,frame,frame_id=key,
                **read_options))
    except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:
        result = verify_selected_skill_support_observations(observed,bundle=bundle,
            selected_slot=selected_slot,selection_evidence=selection_evidence)
        result['reason'] = str(error)
        return result
    return verify_selected_skill_support_observations(observed,bundle=bundle,
        selected_slot=selected_slot,selection_evidence=selection_evidence)


def selected_support_visible_upgrade(bundle, policy, certificate, *, slot,
                                     frame_ids, upgrade_evidence, identity_evidence):
    """Prove a fresh visible plus without changing the permanent certificate.

    Only a currently action-bound complete support receipt can explain this
    difference. Identity-only marks and saved body text without that binding
    cannot authorize a changed visible enhancement.
    """
    from .high_score.contracts import ActionKind

    try:
        if (bundle is None or policy.bundle != bundle or type(slot) is not int
                or not 0 <= slot < len(bundle.state.deck.hand)
                or certificate.get('slot') != slot or certificate.get('upgraded') is not False):
            return None
        ref = bundle.state.deck.hand[slot]
        ordinary = bundle.kernel.cards[ref.card_id]
        canonical = ordinary.canonical_card_id or ordinary.card_id
        action = policy.response.action if policy.response else None
        receipt = policy.selected_skill_support
        effective = policy._current_support_definition()
        if (ref.upgraded or certificate.get('card_id') != canonical
                or canonical not in certificate.get('family', ())
                or ordinary.upgrade_card_id not in certificate.get('family', ())
                or action is None or action.kind is not ActionKind.PLAY_CARD
                or action.source_slot != slot or action.card_instance_id != ref.instance_id
                or effective is None or not isinstance(receipt, dict)
                or receipt.get('enhancement_count') != 1
                or receipt.get('accepted') is not True or receipt.get('effects_verified') is not True
                or receipt.get('card_instance_id') != ref.instance_id
                or receipt.get('definition_id') != ref.card_id or receipt.get('selected_slot') != slot
                or (receipt.get('run_id'), receipt.get('revision')) != (bundle.run_id, bundle.revision)):
            return None
        verified = verify_selected_skill_support_observations(receipt.get('observations', ()),
            bundle=bundle, selected_slot=slot, selection_evidence=receipt.get('selection_evidence', {}))
        if (verified.get('accepted') is not True or json.dumps(verified, sort_keys=True)
                != json.dumps(receipt, sort_keys=True)
                or effective != supported_definition(ordinary, enhancement_count=1)):
            return None
        observations = upgrade_evidence.get('observations', ())
        fresh = identity_evidence.get('fresh_prediction') or {}
        if (len(frame_ids) != 3 or len(set(frame_ids)) != 3
                or upgrade_evidence.get('accepted') is not True
                or upgrade_evidence.get('upgraded') is not True
                or upgrade_evidence.get('plus_votes') != 1 or len(observations) != 1
                or observations[0].get('upgraded') is not True
                or observations[0].get('frame_id') != frame_ids[-1]
                or identity_evidence.get('slot') != slot
                or identity_evidence.get('standard_accepted') is not True
                or identity_evidence.get('accepted') is not True
                or tuple(identity_evidence.get('fresh_family', ())) != tuple(certificate['family'])
                or tuple(identity_evidence.get('expected_family', ())) != tuple(certificate['family'])
                or fresh.get('accepted') is not True or fresh.get('slot') != slot
                or fresh.get('frame_id') != frame_ids[-1]):
            return None
        return dict(source='bound_selected_support_visible_upgrade', selected_slot=slot,
            card_instance_id=ref.instance_id, permanent_card_id=ref.card_id,
            permanent_upgraded=False, visible_card_id=ordinary.upgrade_card_id,
            visible_upgraded=True, frame_ids=list(frame_ids),
            support_frame_ids=list(receipt['frame_ids']),
            upgrade_evidence=deepcopy(upgrade_evidence))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None
