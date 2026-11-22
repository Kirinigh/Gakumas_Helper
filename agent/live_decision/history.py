"""Observe the latest visible Live log without replaying a game action."""

import re
import math
import time
import unicodedata

import cv2
import numpy as np

from .inventory import (
    _inventory_ocr,
    capture_fresh_live_checkpoint,
    capture_initial_live_checkpoint,
)
from .hif_capture_adapter import captured_at as frame_captured_at


def history_navigation_target(context, image, closing):
    node='HifHistoryClose' if closing else 'HifHistoryOpen'
    attempts = (235,) if closing else (235, 215)
    client_size = getattr(context, 'gkh_hif_client_size', None)
    if (not closing and isinstance(client_size, (tuple, list)) and len(client_size) == 2
            and all(type(value) is int for value in client_size) and tuple(client_size) in {
            (540, 960), (541, 961), (541, 962), (542, 962)} and image.shape == (1280, 720, 3)):
        # Display resampling leaves grey edges on the white glyph. Keep their
        # contrast, while removing the moving dark button background (1328).
        # Only enter after both binary paths find no candidate; ambiguity stops.
        attempts = (*attempts, None)
    for white_min in attempts:
        gray=image.min(2)
        if not closing and white_min is None:
            gray=np.clip((gray.astype(np.float32)-150)*(255/105),0,255).astype(np.uint8)
        source=image if closing else cv2.cvtColor(cv2.GaussianBlur(
            gray if white_min is None else (gray>white_min).astype('uint8')*255,
            (3,3),0),cv2.COLOR_GRAY2BGR)
        hit=context.run_recognition(node,source,pipeline_override={node:{
            'recognition':'TemplateMatch','template':(['produce/hif_history_close.png'] if closing else
                ['produce/hif_live_history.png','produce/hif_live_history_480.png','produce/hif_live_history_479.png']),
            'roi':[300,1110,120,100] if closing else [425,1160,80,85],'threshold':.95}})
        matches=list(hit.filtered_results) if hit and hit.hit else []
        boxes={tuple(match.box) for match in matches}
        if len(boxes)==1:
            return boxes.pop()
        if boxes:
            break
    raise ValueError('history navigation control is not unique')


def normalize_history_text(text):
    text=re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))
    return re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',text)


def history_bottom(image):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('history image size is uncalibrated')
    strip = image[262:1137, 683:688].astype(np.int16)
    dark = ((strip.min(2) >= 100) & (strip.max(2) <= 180) & (np.ptp(strip, axis=2) <= 30)).mean(1) >= .8
    indices = np.flatnonzero(dark)
    if len(indices) < 60 or indices[-1] < 867 or np.any(np.diff(indices) != 1):
        raise ValueError('history is not at its verified latest entry')
    return int(indices[0]+262), int(indices[-1]+262)


def _history_fixed_effect_label(context, crop, readings):
    """Match a complete static label with an independently read reference.

    OCR still supplies only the observed label association. Its original weak
    score is unchanged; this is the existing static-title visual proof pattern.
    Numbers and the arrow are outside the template and remain separately read.
    """
    if (crop.dtype != np.uint8 or not callable(getattr(context, 'run_recognition', None))
            or not readings or any(len(rows) != 1 or not math.isfinite(rows[0].score)
                or not 0 < rows[0].score <= 1 for rows in readings)):
        return None
    labels = {normalize_history_text(rows[0].text) for rows in readings}
    excellent = labels == {'絶好調'}
    if excellent:
        client = getattr(context, 'gkh_hif_client_size', None)
        if (crop.shape not in ((32,90,3), (33,90,3)) or len(readings) != 2
                or any(rows[0].score >= .95 for rows in readings)
                or not isinstance(client, (tuple, list)) or len(client) != 2
                or any(type(value) is not int for value in client)
                or tuple(client) not in {(540,960), (541,961), (541,962), (542,962)}):
            return None
    elif labels != {'消費体力削減'} or crop.shape != (29,160,3):
        return None
    # Complete glyphs must fit; no erased first/last character or
    # additional detected label can be hidden by a central template match.
    ink = crop.min(axis=2) < 220
    edges = np.diff(np.r_[False,ink.any(axis=0),False].astype(np.int8))
    spans = list(zip(np.flatnonzero(edges == 1),np.flatnonzero(edges == -1)))
    if excellent:
        # 1693's three strong bottom-page reads establish this whole label.
        # 好/調 touch at their antialiased edge, forming one connected span.
        # Check the original ink before blur; smoothing cannot hide clipped
        # first/last glyphs, extra ink, or an incomplete outer blank margin.
        background = crop[0,0]
        complete = (spans == [(6,30),(31,77)]
            and all(ink[:,left:right].any() for left,right in ((6,30),(31,54),(54,77)))
            and background.min() >= 220 and np.ptp(background) == 0
            and all(np.all(edge == background) for edge in (crop[0],crop[-1],crop[:,0],crop[:,-1])))
        template = 'produce/hif_history_label_excellent_1693.png'
        source = 'native_1693_complete_label_template'
        image = cv2.GaussianBlur(crop,(3,3),0)
    else:
        complete = (len(spans) == 6 and spans[0][0] >= 2 and spans[-1][1] <= crop.shape[1]-2
            and all(6 <= right-left <= 40 for left,right in spans)
            and not ink[0].any() and not ink[-1].any())
        template = 'produce/hif_history_label_cost_reduction.png'
        source = 'native_1326_complete_label_template'
        image = crop
    if not complete:
        return None
    node = 'HifHistoryFixedEffectLabel'
    hit = context.run_recognition(node,image,pipeline_override={node:dict(
        recognition='TemplateMatch', template=[template],
        roi=[0,0,crop.shape[1],crop.shape[0]], threshold=.95)})
    rows = list(hit.filtered_results) if hit and hit.hit else []
    if (len(rows) != 1 or not math.isfinite(rows[0].score) or not .95 <= rows[0].score <= 1):
        return None
    box = tuple(rows[0].box)
    if excellent:
        # A 33px source has one extra blank row. The matched 32px reference
        # must still own all original ink, with its complete two-pixel margin.
        if (box not in {(0,0,90,32), (0,1,90,32)} or box[1]+32 > crop.shape[0]
                or ink[:box[1]+2].any() or ink[box[1]+30:].any()):
            return None
    elif box != (0,0,160,29):
        return None
    return dict(source=source, confidence=float(rows[0].score))


def _history_complete_label_prefix(crop):
    """Separate faint colored numeral edges after a complete grayscale label."""
    background = crop[0,0]
    if background.min() < 200 or np.ptp(background) != 0:
        return None
    blank = np.flatnonzero(np.all(crop == background,axis=(0,2)))
    if not len(blank) or blank[-1] == crop.shape[1]-1:
        return None
    right = int(blank[-1])+1
    prefix, tail = crop[:,:right],crop[:,right:]
    changed = np.any(tail != background,axis=2)
    # A dark letter stroke cannot be removed: each non-background tail pixel
    # must retain a bright channel and be chromatic, as a colored digit edge.
    colored = (np.ptp(tail.astype(np.int16),axis=2) > 0) & (tail.max(2) >= background.max())
    if (right < 12 or not changed.any() or not colored[changed].all()
            or np.ptp(prefix.astype(np.int16),axis=2).any()
            or not (prefix.min(2) < 220).any()
            or any(not np.all(edge == background)
                   for edge in (prefix[0],prefix[-1],prefix[:,0],prefix[:,-1]))):
        return None
    return prefix.copy()


def _history_complete_label_border(crop):
    """Preserve all grayscale label pixels inside a proven blank border."""
    complete = _history_complete_label_prefix(crop)
    source = crop if complete is None else complete
    background = source[0,0]
    if (background.min() < 200 or np.ptp(background) != 0
            or np.ptp(source.astype(np.int16),axis=2).any()
            or any(not np.all(edge == background)
                   for edge in (source[0],source[-1],source[:,0],source[:,-1]))):
        return None
    # Include every antialiased pixel, not just the dark core of each glyph.
    yy,xx = np.where(np.any(source != background,axis=2))
    if not len(yy) or not (source.min(2) < 220).any():
        return None
    return cv2.copyMakeBorder(source[yy.min():yy.max()+1,xx.min():xx.max()+1],
        8,8,8,8,cv2.BORDER_CONSTANT,value=tuple(int(value) for value in background))


def _history_line(context, crop):
    rows=_inventory_ocr(context,crop,'HifHistoryExactLine',only_rec=True)
    if len(rows)!=1:
        raise ValueError('history line is not unique')
    text=normalize_history_text(rows[0].text)
    if .95 <= rows[0].score <= 1:
        return text
    if re.fullmatch(r'Pアイテム「[^「」]+」', text) and min(crop.shape[:2]) > 4:
        # Re-read the entire heading at the fixed 2x sampling. At the calibrated
        # small client sizes, 1445/1446 show correct, agreeing headings below
        # .95. The resolution-scoped adjustment uses .90 for this heading alone; numeric,
        # card and input-evidence gates retain their original requirements.
        enlarged = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        heading = _inventory_ocr(context,enlarged,'HifHistoryItemHeading',only_rec=True)
        client_size = getattr(context, 'gkh_hif_client_size', None)
        small_client = (isinstance(client_size, (tuple, list)) and len(client_size) == 2
                        and all(type(value) is int for value in client_size)
                        and tuple(client_size) in {(540, 960), (541, 961), (541, 962), (542, 962)})
        heading_floor = .90 if small_client and .90 <= rows[0].score <= 1 else .95
        if (len(heading) == 1 and normalize_history_text(heading[0].text) == text
                and heading_floor <= heading[0].score <= 1):
            return text
    if 1 <= len(text) <= 8 and not re.search(r'[0-9→]',text) and min(crop.shape[:2]) > 4:
        # Two-pixel padding may touch a rounded button border at this scale.
        # Re-read the same label with one pixel of that margin removed.
        tighter = _inventory_ocr(context,crop[1:-1,1:-1].copy(),'HifHistoryTightLabel',only_rec=True)
        if (len(tighter)==1 and normalize_history_text(tighter[0].text)==text
                and .95<=tighter[0].score<=1):
            return text
    # Effect labels are dark, while both sides of a numeric change are colored.
    # Read the label independently at full confidence; numeric fields must also
    # agree with the whole line. No predicted resource value is substituted.
    colored=(np.ptp(crop.astype(np.int16),axis=2)>60)&(crop.max(2)>150)
    columns=np.flatnonzero(colored.sum(0)>=2)
    if not len(columns) or columns[0]<12 or not re.fullmatch(r'[^0-9]+\d+→\d+',text):
        error = ValueError('history line confidence is unresolved: '+repr([(r.text,r.score) for r in rows]))
        if 0 < rows[0].score < .95:
            error.normalized_history_text = text
        raise error
    split=int(columns[0])
    label_crop=crop[:,:split]
    label_image=cv2.resize(label_crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    label=_inventory_ocr(context,label_image,'HifHistoryEffectLabel',only_rec=True)
    label_readings=[label]
    if (len(label)==1 and 0<label[0].score<.95 and min(label_crop.shape[:2])>4
            and not re.search(r'[0-9→]',label[0].text)):
        background=label_crop[0,0]
        complete = (_history_complete_label_prefix(label_crop)
                    if normalize_history_text(label[0].text) == re.split(r'\d',text,maxsplit=1)[0]
                    else None)
        # 1326 has two uniformly blank top rows. Remove only the first one
        # before the same 2x sampling, retaining a complete blank row and
        # every bottom/right antialiased edge. Never trim any visible ink.
        if complete is not None or (background.min()>=200 and np.ptp(background)==0
                and np.all(label_crop[:2]==background)):
            # Reuse this one bounded reread. 1792 has two faint digit columns
            # before the strong-color split; preserve the whole label instead
            # of reading that partial neighboring numeral with it.
            tightened=cv2.resize(complete if complete is not None else label_crop[1:],
                None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            reread=_inventory_ocr(context,tightened,'HifHistoryEffectTightLabel',only_rec=True)
            label_readings.append(reread)
            if (len(reread)==1 and .95<=reread[0].score<=1
                    and normalize_history_text(reread[0].text)==normalize_history_text(label[0].text)):
                label=reread
    if (len(label)==1 and 0<label[0].score<.95
            and not re.search(r'[0-9→]',label[0].text)
            and all(len(reading)==1 and 0<reading[0].score<.95 for reading in label_readings)):
        # Reuse complete-heading padding at the existing fixed 2x sampling.
        # 1798's long gray label needs this full border; neither a weak whole
        # line spelling nor predicted values may supply or replace its text.
        bordered = _history_complete_label_border(label_crop)
        if bordered is not None:
            enlarged = cv2.resize(bordered,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            reread = _inventory_ocr(context,enlarged,'HifHistoryEffectCompleteLabel',only_rec=True)
            label_readings.append(reread)
            if (len(reread)==1 and .95<=reread[0].score<=1
                    and normalize_history_text(reread[0].text)==normalize_history_text(label[0].text)):
                label=reread
    label_visual = (_history_fixed_effect_label(context,label_crop,label_readings)
        if len(label)==1 and 0<label[0].score<.95 else None)
    number_image=np.where(colored[:,max(0,split-2):,None],0,255).astype('uint8')
    number_image=np.repeat(number_image,3,axis=2)
    number_mask=number_image[:,:,0]==0
    occupied=np.flatnonzero(number_mask.any(0))
    groups=np.split(occupied,np.flatnonzero(np.diff(occupied)>1)+1)
    widths=[len(group) for group in groups]
    arrow_index=int(np.argmax(widths)) if widths else -1
    if not 0<arrow_index<len(groups)-1 or widths[arrow_index]<14:
        raise ValueError('history numeric change has no separated arrow: '+repr(widths))
    arrow=number_mask[:,groups[arrow_index]]
    ys=np.flatnonzero(arrow.any(1))
    arrow=arrow[ys[0]:ys[-1]+1]
    if (arrow.any(0).mean()<.95 or arrow.sum(1).max()<arrow.shape[1]*.85
            or arrow[:,:3].any(1).sum()>arrow.shape[0]*.4
            or arrow[:,arrow.shape[1]//2:].any(1).sum()<arrow.shape[0]*.8):
        raise ValueError('history numeric arrow direction is unresolved')
    parts=[]
    for left,right in ((0,int(groups[arrow_index][0])),(int(groups[arrow_index][-1])+1,number_image.shape[1])):
        number_crop=number_image[:,left:right].copy()
        part=_inventory_ocr(context,number_crop,'HifHistoryEffectNumber',only_rec=True)
        if (len(part)==1 and normalize_history_text(part[0].text).isdigit()
                and 0<part[0].score<.95 and min(number_crop.shape[:2])>4):
            # 1559's complete binary numeral is weak at native sampling. Use
            # the same fixed 2x sampling as labels, retaining every pixel and
            # requiring an independently confident, identical numeric read.
            enlarged=cv2.resize(number_crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            reread=_inventory_ocr(context,enlarged,'HifHistoryEffectNumberScaled',only_rec=True)
            if (len(reread)==1 and .95<=reread[0].score<=1
                    and normalize_history_text(reread[0].text)==normalize_history_text(part[0].text)):
                part=reread
        if len(part)!=1 or not normalize_history_text(part[0].text).isdigit() or not .95<=part[0].score<=1:
            raise ValueError('history numeric field is unresolved: '+repr([(r.text,r.score) for r in part]))
        parts.append(part[0])
    from types import SimpleNamespace
    numbers=[SimpleNamespace(text=normalize_history_text(parts[0].text)+'→'+normalize_history_text(parts[1].text),score=min(r.score for r in parts))]
    if (len(label)!=1 or len(numbers)!=1 or (not .95<=label[0].score<=1 and label_visual is None)
            or not .95<=numbers[0].score<=1
            or re.search(r'[0-9→]',label[0].text)
            or not text.endswith(normalize_history_text(numbers[0].text))):
        raise ValueError('history split fields are unresolved: '+ascii((text,split,[(r.text,r.score) for r in [*label,*numbers]])))
    return normalize_history_text(label[0].text+numbers[0].text)


def qualified_history_card_reference(row, error, *, allow_draw_count=False):
    """Keep an independently qualified full-page card reference (862/948).

    Only a weak tight crop can use this path. The complete-page reader may
    also keep a full draw/exclusion sentence when its weak crop agrees including digits.
    Other numeric/effect text and ambiguous recognition remain rejected.
    """
    text = normalize_history_text(row.text)
    qualified_draw = (allow_draw_count is True
                      and re.fullmatch(r'[1-9]\d*枚(?:山札から引いた|除外に移動)', text)
                      and getattr(error, 'normalized_history_text', None) == text)
    if (not str(error).startswith('history line confidence is unresolved:')
            or not .95 <= row.score <= 1
            or not (re.fullmatch(r'スキルカード「[^「」]+」を?', text) or qualified_draw)):
        raise error
    return text


def _complete_history_draw(name, effect):
    return (re.fullmatch(r'スキルカード「[^「」]+」を',name) is not None
            and re.fullmatch(r'[1-9]\d*枚山札から引いた',effect) is not None)


def _merge_split_card_reference_rows(rows):
    """OCR may split a black prefix from the blue name on the same line."""
    from types import SimpleNamespace

    def same_row(left, right, *, normal_height=True):
        ly, lh = left.box[1], left.box[3]
        ry, rh = right.box[1], right.box[3]
        common = min(ly+lh,ry+rh)-max(ly,ry)
        return ((not normal_height or 16 <= lh <= 40 and 16 <= rh <= 40)
                and common > 0 and common >= min(lh,rh)-2)

    merged, consumed = [], set()
    for index, prefix in enumerate(rows):
        if index in consumed:
            continue
        label = normalize_history_text(prefix.text)
        # 1236 splits a persistent heading, and its draw prefix overlaps the
        # opening quote of the colored name. 1256 reads that same partial
        # quote as 『. These fragments only propose a complete-line crop;
        # the independent exact reader must verify the entire reference.
        exact_required = label in ('持続効果', 'スキルカード「', 'スキルカード『')
        if label not in ('スキルカード', 'スキルカード「', 'スキルカード『', '持続効果'):
            continue
        if not (0 < prefix.score <= 1 if exact_required else .95 <= prefix.score <= 1):
            continue
        suffix_pattern = r'「[^「」]+」' if label == '持続効果' else r'「[^「」]+」を(?:山札から)?'
        x, y, w, h = prefix.box

        def suffix_text(suffix):
            text = normalize_history_text(suffix.text)
            # 1839's overlapping name crop includes the tail of the black
            # prefix as an extra quote. Propose the same complete-line crop;
            # overlap still requires an independent, matching exact read.
            if label == 'スキルカード' and suffix.box[0] < x+w and text.startswith('“「'):
                return text[1:]
            return text

        matches = [(other, suffix) for other, suffix in enumerate(rows) if other not in consumed
            and .95 <= suffix.score <= 1
            and re.fullmatch(suffix_pattern, suffix_text(suffix))
            and -min(h, suffix.box[3])//2-2 <= suffix.box[0]-(x+w) <= 24
            and suffix.box[0] > x+w/2
            and same_row(prefix,suffix)]
        if len(matches) != 1:
            continue
        other, suffix = matches[0]
        sx, sy, sw, sh = suffix.box
        # 1459's independently strong plain prefix also has overlapping OCR
        # padding. Reuse the quoted-prefix geometry, but require a whole read.
        # A source-zone suffix is only a crop proposal, even without overlap.
        # Both readers must independently read the entire reference (1467).
        # Newly proved common-ink rows also need a whole read; preserve the
        # old nonoverlapping, <=2px proposal contract for existing callers.
        requires_exact = (exact_required or sx < x+w or abs(sy-y) > 2
                          or normalize_history_text(suffix.text).endswith('を山札から'))
        if requires_exact and any(
                candidate != index and candidate != other
                and row.box[0] < sx and row.box[0]+row.box[2] > x
                and (abs(row.box[1]-y) <= 2 or same_row(prefix,row,normal_height=False))
                for candidate, row in enumerate(rows)):
            continue
        consumed.update((index, other))
        merged.append(SimpleNamespace(text=label.rstrip('「『')+suffix_text(suffix),
            score=min(prefix.score, suffix.score),
            box=(x, min(y,sy), sx+sw-x, max(y+h,sy+sh)-min(y,sy)),
            split_reference=True, split_requires_exact=requires_exact))
    return [row for index,row in enumerate(rows) if index not in consumed]+merged


def _complete_draw_and_grant_card_group(lines, rows):
    # 949 draws before both grants. 1080's full hand omits the draw entirely.
    # This proves a complete log group, not whether a draw should have occurred;
    # the independent committed-state verifier still checks every card zone.
    if (len(lines) not in (3, 5) or lines[0] != 'スキルカード「わたしだけの思い出+」'
            or (len(lines) == 5 and (not _complete_history_draw(lines[1],lines[2])
                                    or lines[2] != '1枚山札から引いた'))
            or lines[-2:] != ['集中強化が付与','わたしだけの思い出+が付与']):
        return False
    from .history_chain import HistoryLine, _line_kind, _group_lines

    groups = _group_lines(tuple(HistoryLine(text,tuple(row.box),_line_kind(text,row.box))
                               for row,text in zip(rows,lines,strict=True)))
    return len(groups) == 1 and groups[0].kind == 'card' and groups[0].lines == tuple(lines)


def _complete_no_change_card_group(lines, rows, detected):
    # Native 1671: card 23 explicitly logs its whole effect as 変化なし.
    # This proves the receipt grammar, not unchanged actions/card zones; the
    # independently verified settlement and fresh core retain those checks.
    if (lines != ['スキルカード「眠気」', '変化なし'] or len(rows) != 2
            or any(normalize_history_text(row.text) != text
                   for row, text in zip(rows, lines, strict=True))
            or rows[1].box[1] < rows[0].box[1]+rows[0].box[3]
            or any(row.box[0] >= 130 and row.box[1] >= rows[0].box[1]
                   and row.box[1]+row.box[3] > 1118 for row in detected)):
        return False
    from .history_chain import HistoryLine, _line_kind, _group_lines

    groups = _group_lines(tuple(HistoryLine(text, tuple(row.box), _line_kind(text, row.box))
                               for row, text in zip(rows, lines, strict=True)))
    return len(groups) == 1 and groups[0].kind == 'card' and groups[0].lines == tuple(lines)


def _complete_retrieval_drink_effects(lines):
    # 1588's complete black-vinegar receipt includes a wrapped transfer and
    # a grant before its numeric changes. This is log grammar only; the
    # separately verified settlement proves the actual selected card/state.
    # 1768's longer card name wraps the independently read source onto its
    # own line. Validate both complete lines; preserve their original receipt.
    if (len(lines) == 7 and re.fullmatch(r'スキルカード「[^「」]+」を', lines[1])
            and lines[2] == '山札から'):
        lines = [lines[0], lines[1]+lines[2], *lines[3:]]
    return (len(lines) == 6
            and re.fullmatch(r'(?:Pドリンク|ドリンク)「初星黒酢」', lines[0]) is not None
            and re.fullmatch(r'スキルカード「[^「」]+」を山札から', lines[1]) is not None
            and lines[2:4] == ['1枚手札に移動', '消費体力変化が付与']
            and re.fullmatch(r'体力\d+→\d+', lines[4]) is not None
            and re.fullmatch(r'消費体力[増增]加\d+→\d+', lines[5]) is not None)


def _complete_senburi_drink_effects(lines):
    # Native 1617: at four cards the drink draws one card before its stamina
    # payment and recurring-effect grant. Require the whole six-line receipt;
    # actual card zones and effect values belong to complete settlement.
    return (len(lines) == 6
            and re.fullmatch(r'(?:Pドリンク|ドリンク)「センブリソーダ」', lines[0]) is not None
            and lines[1] == 'スコア上昇量増加が付与'
            and _complete_history_draw(lines[2], lines[3])
            and lines[3] == '1枚山札から引いた'
            and re.fullmatch(r'体力\d+→\d+', lines[4]) is not None
            and lines[5] == 'センブリソーダが付与')


def _complete_boost_drink_effects(lines):
    # Native 1640: the nonnumeric score grant precedes both numeric changes.
    # This recognizes the complete receipt; settlement verifies effect values.
    return (len(lines) == 4
            and re.fullmatch(r'(?:Pドリンク|ドリンク)「ブーストエキス」', lines[0]) is not None
            and lines[1] == 'スコア上昇量増加が付与'
            and re.fullmatch(r'消費体力減少\d+→\d+', lines[2]) is not None
            and re.fullmatch(r'体力\d+→\d+', lines[3]) is not None)


def _complete_skip_history_tail(lines, rows):
    """1023's unused-card marker owns the following stamina and end-turn groups."""
    if (len(lines) < 2 or lines[0] != 'スキルカード未使用'
            or not re.fullmatch(r'体力\d+→\d+', lines[1])
            or not 150 <= rows[1].box[0] <= 175):
        return False
    from .history_chain import HistoryLine, _line_kind, _group_lines

    groups = _group_lines(tuple(HistoryLine(text, tuple(row.box), _line_kind(text, row.box))
        for row, text in zip(rows[2:], lines[2:], strict=True))) if lines[2:] else ()
    return all(group.kind == 'system'
        and re.fullmatch(r'持続効果「[^「」]+」', group.title)
        and len(group.lines) >= 2
        and all(re.fullmatch(r'[^0-9→]+\d+→\d+', text) for text in group.lines[1:])
        for group in groups)


def _history_score(context, image):
    rows = _inventory_ocr(context, image, 'HifHistoryScore', roi=[40,145,345,32], only_rec=True)
    if len(rows) == 1 and .95 <= rows[0].score <= 1 and rows[0].text.isdigit():
        return int(rows[0].text)
    # A single digit in the wide score area may be read as a letter. Keep all
    # dark text pixels and re-read their bounds; never substitute the Live score.
    crop = image[145:177,40:385]
    ys, xs = np.where(crop.max(2) < 100)
    if not len(xs) or xs.min() < 2 or ys.min() < 2 or xs.max() >= 343 or ys.max() >= 30:
        raise ValueError('history score is unresolved')
    tight = crop[max(0,ys.min()-3):ys.max()+4,max(0,xs.min()-3):xs.max()+4].copy()
    rows = _inventory_ocr(context, tight, 'HifHistoryScoreTight', only_rec=True)
    if len(rows) != 1 or not .95 <= rows[0].score <= 1 or not rows[0].text.isdigit():
        raise ValueError('history score is unresolved')
    return int(rows[0].text)


def _history_trigger_icon_row(image, row, trigger_headers):
    """Reuse the full-history trigger lane, allowing only blank left padding."""
    x,y,width,height = row.box
    if not any(x+width < hx and y < hy+hh and y+height > hy and height > hh
               for hx,hy,_hw,hh in trigger_headers):
        return False
    if x >= 150:
        return True
    if (not 130 <= x < 150 or width <= 0 or height <= 0 or y < 0
            or y+height > image.shape[0] or x+width > image.shape[1]):
        return False
    # 1663's detector starts at x=141 around the memory thumbnail, but every
    # foreground pixel remains in the existing x>=150 icon lane. A letter or
    # border in that padding keeps the row subject to exact text recognition.
    ink = image[y:y+height,x:x+width].min(2) < 235
    return bool(ink.any() and not ink[:, :150-x].any())


def read_latest_history(context, image, *, observed_turn=None, round_context=None):
    """Require a complete latest action/turn tail at the bottom, with exact OCR."""
    from .hif_session import resolve_hif_round_context

    _, base_turn_count = resolve_hif_round_context(round_context)
    thumb = history_bottom(image)
    rows = _inventory_ocr(context, image, 'HifHistoryLines', roi=[30,110,645,1008])
    rows = _merge_split_card_reference_rows(rows)
    action_prefixes = ('スキルカード「', 'Pドリンク「', 'ドリンク「')
    headings = [row for row in rows if normalize_history_text(row.text).startswith(action_prefixes)
                and 130 <= row.box[0] <= 150 and row.box[1] >= 270]
    expected_turn = getattr(context, 'gkh_hif_history_expected_turn', None)
    if observed_turn is not None:
        if (type(observed_turn) is not int or not 1 <= observed_turn <= base_turn_count
                or expected_turn is not None and expected_turn != observed_turn):
            raise ValueError('history observed current turn is invalid or contradictory')
        expected_turn = observed_turn
    turn_headings = [row for row in rows if type(expected_turn) is int
                     and normalize_history_text(row.text) == f'{expected_turn}ターン目'
                     and 130 <= row.box[0] <= 150 and row.box[1] >= 270]
    turn_group = bool(len(turn_headings) == 1 and
                      (not headings or turn_headings[0].box[1] > max(r.box[1] for r in headings)))
    if turn_group:
        headings = turn_headings
    skip_headings = [row for row in rows if type(expected_turn) is int and expected_turn > 1
                     and normalize_history_text(row.text) == 'スキルカード未使用'
                     and 150 <= row.box[0] <= 175 and row.box[1] >= 270]
    skip_group = bool(len(skip_headings) == 1 and
                      (not headings or skip_headings[0].box[1] > max(r.box[1] for r in headings)))
    if skip_group:
        headings, turn_group = skip_headings, False
    if not headings:
        raise ValueError('history latest card heading is not fully visible')
    last = max(headings, key=lambda row: row.box[1])
    drink_group = normalize_history_text(last.text).startswith(('Pドリンク「', 'ドリンク「'))
    if not turn_group and not skip_group and not normalize_history_text(last.text).endswith('」'):
        raise ValueError('history card heading is incomplete')
    if (skip_group or drink_group) and any(row.box[0] >= 130 and row.box[1] >= last.box[1]
                          and row.box[1]+row.box[3] > 1118 for row in rows):
        raise ValueError('history drink tail contains a clipped line' if drink_group
                         else 'history skip tail contains a clipped line')
    tail = sorted((row for row in rows if row.box[0] >= 130 and row.box[1] >= last.box[1]
                   and row.box[1]+row.box[3] <= 1118
                   and (not turn_group or row is last or row.box[1] >= last.box[1]+last.box[3])),
                  key=lambda row: (row.box[1], row.box[0]))
    if len(tail) < 2:
        raise ValueError('history card effects are missing')
    exact_headers = {}
    trigger_headers = []
    for row in tail:
        x,y,w,h = row.box
        raw = normalize_history_text(row.text)
        if 240 <= x <= 260 and (raw.startswith('メモリー') or raw == 'スキルカードサポート'):
            text = _history_line(context,image[y-2:y+h+2,x-2:x+w+2].copy())
            exact_headers[tuple(row.box)] = text
            if text in ('メモリーアビリティ','スキルカードサポート'):
                trigger_headers.append(tuple(row.box))
    tail = [row for row in tail if not _history_trigger_icon_row(image,row,trigger_headers)]
    # Detect complete lines, then recognize their exact crop independently.
    # A missing/extra line or changed value changes the fingerprint.
    lines = []
    for row in tail:
        x,y,w,h = row.box
        try:
            text = exact_headers.get(tuple(row.box))
            if text is None:
                text = _history_line(context,image[max(0,y-2):y+h+2,max(0,x-2):x+w+2].copy())
        except ValueError as error:
            if getattr(row, 'split_requires_exact', False):
                raise
            text = qualified_history_card_reference(row,error)
        if getattr(row, 'split_reference', False) and text != normalize_history_text(row.text):
            raise ValueError('history split card name changed during complete-line recognition')
        lines.append(text)
    if skip_group:
        if not _complete_skip_history_tail(lines, tail):
            raise ValueError('history skip effect groups are incomplete')
    elif turn_group:
        if lines[0] != f'{expected_turn}ターン目':
            raise ValueError('history turn heading changed during recognition')
        # Reuse the complete-history heading/indentation grammar. A new turn
        # can also contain item grants and persistent score effects (915),
        # including wrapped card transfers; it is not just a draw reservation.
        from .history_chain import HistoryLine, _line_kind, _group_lines

        groups = _group_lines(tuple(HistoryLine(text, tuple(row.box),
            _line_kind(text, row.box)) for row, text in zip(tail, lines)))
        if (len(groups) < 2 or groups[0].kind != 'turn-start'
                or groups[0].lines != (f'{expected_turn}ターン目',)
                or any(group.kind != 'system' or len(group.lines) < 2 for group in groups[1:])):
            raise ValueError('history turn-start effect groups are incomplete')
        # Keep the existing completeness proof when no numeric transition is
        # present: each draw needs both its complete card name and draw count.
        if not any('→' in line for line in lines[1:]):
            for group in groups[1:]:
                effects = group.lines[1:]
                if (not group.title.startswith('持続効果「') or not group.title.endswith('」')
                        or len(effects) % 2 or any(
                            not _complete_history_draw(name,effect)
                            for name, effect in zip(effects[::2], effects[1::2]))):
                    raise ValueError('history turn-start draw group is incomplete')
    elif drink_group:
        from .history_chain import HistoryLine, _line_kind, _group_lines

        groups = _group_lines(tuple(HistoryLine(text, tuple(row.box),
            _line_kind(text, row.box)) for row, text in zip(tail, lines)))
        if (lines[0] != normalize_history_text(last.text)
                or not re.fullmatch(r'(?:Pドリンク|ドリンク)「[^「」]+」', lines[0])
                or len(groups) != 1 or groups[0].kind != 'drink'
                or (any(not re.fullmatch(r'[^0-9→]+\d+→\d+', line) for line in lines[1:])
                    and not _complete_retrieval_drink_effects(lines)
                    and not _complete_senburi_drink_effects(lines)
                    and not _complete_boost_drink_effects(lines))):
            raise ValueError('history drink heading or numeric effects are incomplete')
    elif not lines[0].startswith('スキルカード「') or not lines[0].endswith('」'):
        raise ValueError('history card heading changed during recognition')
    if (not turn_group and not skip_group and not any('→' in line for line in lines[1:])
            and not _complete_draw_and_grant_card_group(lines,tail)
            and not _complete_no_change_card_group(lines,tail,rows)):
        raise ValueError('history latest group has no numeric transition')
    return dict(lines=tuple(lines), score=_history_score(context, image), thumb=thumb)


def observe_latest_history(context, *, capture_frames, checkpoint, frame_identifier, round_context=None):
    """Audited navigation, at most one initial bottom scroll, and exact log reads."""
    from utils.maa_input import verify_maa_input_endpoints

    from .hif_session import resolve_hif_round_context

    if round_context is not None:
        run = getattr(context, 'gkh_hif_live_context', None)
        if run is None:
            raise ValueError('history round metadata requires its bound Live context')
        _, base_turn_count = resolve_hif_round_context(
            round_context, run_id=run.run_id, revision=run.revision)
    else:
        _, base_turn_count = resolve_hif_round_context()

    controller=context.tasker.controller
    event=dict(accepted=False,event='live_history_observe',input_click_count=0,inputs=[])
    started=time.perf_counter()
    def fresh():
        image=controller.post_screencap().wait().get()
        return image,frame_captured_at(controller,image)
    if getattr(context, 'gkh_hif_upstream_card_ui', False) is True:
        capture_frames = lambda: (fresh()[0],)
    def target(image,closing):
        return history_navigation_target(context,image,closing)
    def click(image,captured_at,closing):
        box=target(image,closing)
        endpoints=verify_maa_input_endpoints(controller.info,
            expected_client_size=getattr(context,'gkh_hif_client_size',(720,1280)))
        if time.perf_counter()-captured_at>1 or (not closing and time.perf_counter()-started>30):
            raise ValueError('history navigation snapshot expired')
        event['input_click_count']+=1
        event['inputs'].append(dict(box=box,closing=closing,endpoints=endpoints))
        x,y,w,h=box
        if not controller.post_click(x+w//2,y+h//2).wait().succeeded:
            raise ValueError('history Maa input failed')
        time.sleep(.35)
    _,before=capture_initial_live_checkpoint(capture_frames,checkpoint,event)
    remaining=before['fields'][0]
    if type(remaining) is not int or not 1 <= remaining <= base_turn_count:
        raise ValueError('history remaining turns contradict the observed HIF round')
    observed_turn=base_turn_count+1-remaining
    image,stamp,_=capture_fresh_live_checkpoint(fresh,capture_frames,checkpoint,before,event)
    click(image,stamp,False)
    failure=None
    try:
        from .history_turn_tail import align_open_history_bottom

        frames = align_open_history_bottom(context, tuple(capture_frames()), capture_frames=capture_frames,
            frame_identifier=frame_identifier, fresh=fresh, navigation_target=target, event=event)
        for attempt in range(3):
            if attempt:
                frames=capture_frames()
            try:
                options = {'round_context': round_context} if round_context is not None else {}
                observations=[read_latest_history(context,frame,observed_turn=observed_turn,**options)
                              for frame in frames]
                if not observations or any((row['lines'],row['score']) !=
                        (observations[0]['lines'],observations[0]['score']) for row in observations[1:]):
                    raise ValueError('history needs a readable current observation')
                event.update(observation=observations[0],frame_ids=[frame_identifier(frame) for frame in frames])
                break
            except ValueError as error:
                event.setdefault('reobservations',[]).append(str(error))
                if str(error) == 'history latest card heading is not fully visible':
                    from .history_turn_tail import observe_open_turn_tail

                    observation, ids = observe_open_turn_tail(context,
                        capture_frames=capture_frames, frame_identifier=frame_identifier,
                        expected_turn=observed_turn, fresh=fresh,
                        navigation_target=target, event=event)
                    event.update(observation=observation, frame_ids=ids)
                    break
                if attempt==2:
                    raise
                time.sleep(.1)
    except ValueError as error:
        failure=error
    image,stamp=fresh()
    # A failed reader need not be at the bottom to leave an already-open log.
    # The ordinary close template, current endpoint and 1000ms gate still apply.
    if failure is None:
        history_bottom(image)
    click(image,stamp,True)
    _,after=capture_initial_live_checkpoint(capture_frames,checkpoint,event)
    if any(before[k]!=after[k] for k in ('families','fields')):
        raise ValueError('Live changed during history inspection')
    event.update(returned_unchanged=True,
        before={k:before[k] for k in ('fields','families','frame_ids')},
        after={k:after[k] for k in ('fields','families','frame_ids')},
        elapsed_ms=(time.perf_counter()-started)*1000)
    if failure:
        event['reason']=str(failure)
        return event
    event['accepted']=True
    return event


def restore_verified_history(bundle, expected, *, settlement, reference, current,
                             old_run, new_run, old_pid, new_pid):
    """Rebind a verified ledger only after fresh log and Live continuity proof.

    The host supplies a previously verified settlement and its reviewed log
    reference. This is not an opening constructor or a way to fill unknowns.
    """
    import copy
    import json
    from dataclasses import asdict, replace

    from .high_score.opening import (
        observed_hif_drink_slots,
        verified_expired_start_drinks,
        verified_expired_start_scores,
        verify_hif_committed_observation,
        expected_permanent_inventory_keys,
    )

    if (type(old_pid) is not int or old_pid <= 0 or old_pid != new_pid
            or any(getattr(old_run,k) != getattr(new_run,k) for k in ('idol','plan','max_stamina'))
            or bundle.run_id != old_run.run_id or bundle.revision != old_run.revision):
        raise ValueError('history resume process or identity changed')
    if (settlement.get('accepted') is not True or not settlement.get('checks')
            or not all(value is True for value in settlement['checks'].values())
            or settlement.get('run_id') != old_run.run_id or settlement.get('revision') != old_run.revision
            or json.dumps(settlement.get('expected_state'),sort_keys=True) != json.dumps(asdict(expected),sort_keys=True)):
        raise ValueError('history resume lacks a matching complete settlement')
    if (current.get('accepted') is not True or current.get('returned_unchanged') is not True
            or not current.get('frame_ids') or len(set(current['frame_ids'])) != len(current['frame_ids'])
            or not reference.get('lines')
            or tuple(current.get('observation',{}).get('lines',())) != tuple(reference['lines'])
            or current['observation'].get('score') != reference.get('score')):
        raise ValueError('history resume latest log is unverified or changed')
    core = current['after']
    for key in ('fields','families'):
        if (json.dumps(current['before'][key]) != json.dumps(core[key])
                or json.dumps(core[key]) != json.dumps(settlement['core'][key])):
            raise ValueError('history resume current Live changed')
    if current['observation']['score'] != expected.score:
        raise ValueError('history resume score disagrees')
    # Recheck the actual recorded component data, not merely its old flags.
    comparison = verify_hif_committed_observation(bundle,expected,core=core,
        components=settlement['components'],baseline_components=settlement['components'])
    if not comparison['accepted']:
        raise ValueError('history resume ledger does not explain the current hand')
    by_id = {ref.instance_id:ref for ref in expected.deck.hand}
    state = replace(expected,deck=expected.deck.replace_zones(hand=tuple(
        by_id[key] for key in comparison['observed_hand_instance_ids'])))
    rebound = replace(bundle,run_id=new_run.run_id,revision=new_run.revision,state=state,
        expired_start_scores=verified_expired_start_scores(bundle, state),
        expired_start_drinks=verified_expired_start_drinks(bundle, state),
        inventory_keys=expected_permanent_inventory_keys(bundle,expected),
        hand_instance_ids=tuple(ref.instance_id for ref in state.deck.hand),
        # Validate the recorded action against its original bottle mapping
        # first; only the completed new binding receives the post-use slots.
        drink_slots=(observed_hif_drink_slots(settlement['components']['drinks'])
                     if 'slots' in settlement['components']['drinks'] else bundle.drink_slots))
    event = copy.deepcopy(settlement)
    event.update(run_id=new_run.run_id,revision=new_run.revision,core=core,
        observation_source='verified_history_resume',history_observation=current,
        source_epoch=dict(run_id=old_run.run_id,revision=old_run.revision))
    for component in event['components'].values():
        component.update(run_id=new_run.run_id,revision=new_run.revision,
            observation_source='retained_verified_component',source_epoch=event['source_epoch'])
    # The comparison above independently proved the original temporary-body /
    # permanent-ledger separation. Carry only that permanent identity into the
    # new epoch, as normal retained inventory; the old raw support receipt stays
    # in the original settlement. A later selected card still needs fresh body
    # evidence, and cannot inherit this old observation as a playable support.
    _retain_verified_inventory_support(event['components']['cards'], new_run)
    return rebound,state,event


def _retain_verified_inventory_support(component, run):
    for row in component['cards']:
        if row.pop('temporary_support', None) is not None:
            row['source'] = 'retained_verified_detail'
            row['temporary_support_identity_only'] = True
        if row.get('temporary_support_identity_only') is True:
            row['retained_from'] = dict(run_id=run.run_id,revision=run.revision)
