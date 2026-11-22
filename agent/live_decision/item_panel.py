"""Read the owned-item overview without interpreting missing counters as zero.

This produces observed list rows, not a complete high-score item ledger.
"""

import re
import json
import math
import time
from types import SimpleNamespace
from itertools import count
from dataclasses import asdict, dataclass

import cv2
import numpy as np

try:
    from agent.p_item_recognition.reference import PItemReferenceRuntime, measure_p_item_content_generation
except ModuleNotFoundError:
    from p_item_recognition.reference import PItemReferenceRuntime, measure_p_item_content_generation

from .inventory import _inventory_ocr, _unique_inventory_text, capture_fresh_live_checkpoint, capture_initial_live_checkpoint
from .status_panel import _inside, status_panel_geometry, _normalize_status_text
from .hif_capture_adapter import captured_at as frame_captured_at


@dataclass(frozen=True)
class ItemDescriptionRow:
    box: tuple[int, int, int, int]
    name: str
    remaining: 'ItemRemainingCounter | None'
    use_limit: int | None
    confidence: float

    @property
    def content_key(self):
        return self.name, self.remaining.value if self.remaining is not None else None, self.use_limit


@dataclass(frozen=True)
class OpeningOnlyItemDescriptionRow(ItemDescriptionRow):
    """A complete start-only effect, with its counter absence observed."""

    description: str
    counter_region: tuple[int, int, int, int]
    counter_absence_evidence: dict
    opening_resolved_visual: dict

    @property
    def content_key(self):
        return (*super().content_key, self.description)


@dataclass(frozen=True)
class ItemDescriptionHeader:
    box: tuple[int, int, int, int]
    name: str
    remaining: 'ItemRemainingCounter | None'

    @property
    def content_key(self):
        return self.name, self.remaining.value if self.remaining is not None else None


@dataclass(frozen=True)
class ItemDescriptionPage:
    rows: tuple[ItemDescriptionRow, ...]
    thumb: tuple[int, int]
    at_top: bool
    at_bottom: bool
    partial_row_count: int
    headers: tuple[ItemDescriptionHeader, ...] = ()

    @property
    def content_key(self):
        return (tuple(row.content_key for row in self.rows), self.at_top, self.at_bottom,
                tuple(header.content_key for header in self.headers))


@dataclass(frozen=True)
class _FourthSlotEntryPage(ItemDescriptionPage):
    """The first live popup page after opening the confirmed fourth HUD slot."""


def join_item_description_pages(pages, *, complete=True):
    """Join visible headers, then require a complete reading of every item.

    Two tall rows need not fit together. A partial row may establish order
    only with its fully visible name and counter; it never supplies a limit.
    """
    from .status_panel import _join_status_panel_rows, join_status_panel_pages, _join_status_panel_prefix

    views = [SimpleNamespace(rows=p.headers,thumb=p.thumb,at_top=p.at_top,at_bottom=p.at_bottom)
             for p in pages]
    if any(not view.rows for view in views):
        raise ValueError('item description page has no continuity headers')
    tail_index = next((i for i, page in enumerate(pages) if isinstance(page, _FourthSlotEntryPage)), None)
    previous = _join_status_panel_prefix(views[:tail_index]) if tail_index is not None else ()
    if previous and views[tail_index].rows[0].name not in {r.name for r in previous}:
        # The fourth sidebar slot identifies the first new header when the
        # third is already off screen. Following pages use ordinary overlaps.
        tail = views[tail_index:]
        if (len(previous) != 3 or tail[0].at_top or tail[0].thumb[0] <= views[tail_index-1].thumb[0]
                or (complete and not tail[-1].at_bottom)):
            raise ValueError('fourth item slot does not follow the observed prefix')
        following = _join_status_panel_rows(tail)
        if {r.name for r in previous} & {r.name for r in following}:
            raise ValueError('fourth item slot conflicts with the observed prefix')
        headers = (*previous, *following)
    else:
        headers = (join_status_panel_pages(views) if complete else _join_status_panel_prefix(views))
    if tail_index is not None and complete and len(headers) < 4:
        raise ValueError('fourth item slot was not observed at the bottom')
    keys = [header.content_key for header in headers]
    if len(set(keys)) != len(keys):
        raise ValueError('item description header identity is ambiguous')
    resolved = {}
    for page in pages:
        for row in page.rows:
            key = (row.name,row.remaining.value if row.remaining is not None else None)
            if key not in keys:
                raise ValueError('complete item row is outside the header sequence')
            if key in resolved and resolved[key].content_key != row.content_key:
                raise ValueError('item description complete readings disagree')
            resolved[key] = row
    if complete and any(key not in resolved for key in keys):
        raise ValueError('item description has headers without complete readings')
    return tuple(resolved[key] for key in keys if key in resolved)


def resolve_item_descriptions(pages, metadata, compiled_catalog, *, plan):
    """Resolve observed complete names and counts against the fixed catalog.

    Same-name upgrade variants remain ambiguous; no nearest-name or artwork
    fallback supplies an identity. Multiple schedules need a separate count
    interpretation and are rejected by this first adapter.
    """
    if plan not in ('sense','logic','anomaly'):
        raise ValueError('item description plan is unknown')
    rows = join_item_description_pages(pages)
    candidates = [row for row in metadata if row.get('mode')=='stage'
                  and row.get('plan') in ('free',plan)]
    result = []
    for row in rows:
        matched = [item for item in candidates if _normalize_status_text(item['name'])==row.name]
        if len(matched)!=1:
            raise ValueError('item description catalog identity is missing or ambiguous: '+row.name)
        item = matched[0]
        if isinstance(row, OpeningOnlyItemDescriptionRow):
            from .high_score.opening_items import STAGE_START_ITEM_SCOPE, bind_observed_stage_start_item

            record = dict(item_id=item['id'], name=row.name, upgraded=item['upgraded']=='TRUE',
                          remaining_uses=None, use_limit=None, confidence=row.confidence,
                          definition_effect_index=None, effect_scope=STAGE_START_ITEM_SCOPE,
                          description=row.description, counter_region=row.counter_region,
                          counter_absent=True, counter_absence_evidence=row.counter_absence_evidence,
                          opening_resolved=True, opening_resolved_visual=row.opening_resolved_visual)
            if row.remaining is not None or row.use_limit is not None:
                raise ValueError('opening-only item cannot supply a fabricated count')
            bind_observed_stage_start_item(record, item)
            result.append(record)
            continue
        definition = compiled_catalog.items.get(item['id'])
        if (definition is None or len(definition.persistent_effects)!=1
                or definition.persistent_effects[0].schedule_limit!=row.use_limit):
            raise ValueError('item description count has no matching compiled schedule: '+row.name)
        if item.get('upgraded') not in ('TRUE','FALSE') or not 0<=row.remaining.value<=row.use_limit:
            raise ValueError('item description upgrade or remaining count is invalid')
        result.append(dict(item_id=item['id'],name=row.name,upgraded=item['upgraded']=='TRUE',
                           remaining_uses=row.remaining.value,use_limit=row.use_limit,
                           confidence=row.confidence,definition_effect_index=0))
    return tuple(result)


def _read_opening_only_item(context, image, box, name, text):
    """Read the actual 996 green HIF item, including its disabled appearance."""
    from .high_score.opening_items import STAGE_START_ITEM_NAME, STAGE_START_ITEM_DESCRIPTION

    expected = _normalize_status_text(STAGE_START_ITEM_NAME)
    if _normalize_status_text(name.text) != expected or not math.isfinite(name.score) or name.score < .85:
        raise ValueError('opening-only item full name is unresolved')
    confidence = name.score
    if confidence < .95:
        nx, ny, nw, nh = name.box
        crop = image[ny:ny+nh, nx:nx+nw]
        if getattr(context, 'gkh_hif_client_size', None) == (359, 638):
            # The spent item is pale grey. Preserve every character and its
            # delimiters while restoring contrast and the recognizer's margin.
            crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
            crop = cv2.copyMakeBorder(crop, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255,255,255))
        else:
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        found = _inventory_ocr(context, crop, 'LiveItemOpeningName', only_rec=True)
        if (len(found) != 1 or not math.isfinite(found[0].score) or found[0].score < .95
                or _normalize_status_text(found[0].text) != expected):
            raise ValueError('opening-only item exact title reread is unresolved')
        confidence = found[0].score
    x, y, w, h = box
    body = [row for row in text if _inside({'box': tuple(row.box)}, (x+15, y+110, w-42, h-120))]
    if not body or any(not math.isfinite(row.score) or row.score < .95 for row in body):
        raise ValueError('opening-only item full body confidence is insufficient')
    lines = []
    for row in sorted(body, key=lambda item: item.box[1]+item.box[3]/2):
        center = row.box[1]+row.box[3]/2
        if not lines or abs(center-lines[-1][0]) > 12:
            lines.append((center, [row]))
        else:
            lines[-1][1].append(row)
    description = ''
    for _, group in lines:
        for row in sorted(group, key=lambda item: item.box[0]):
            value = _normalize_status_text(row.text)
            left, top, width, height = row.box
            if (getattr(context, 'gkh_hif_client_size', None) == (359, 638)
                    and value.startswith('+') and left < x+70 < left+width):
                # A diamond bullet can be returned as '+' with the next line.
                # Reread the text column without the bullet; keep real signs.
                found = _inventory_ocr(context, image[top:top+height, x+70:left+width],
                                       'LiveItemOpeningBody', only_rec=True)
                if (len(found) == 1 and .95 <= found[0].score <= 1
                        and _normalize_status_text(found[0].text) == value[1:]):
                    value = value[1:]
            description += value
    if description != STAGE_START_ITEM_DESCRIPTION:
        raise ValueError('opening-only item complete start effect is unsupported')
    triggers = [row for row in body if _normalize_status_text(row.text) == '試験開始時、']
    if len(triggers) != 1:
        raise ValueError('opening-only item start trigger is not unique')
    region = (x+75, y+13, 24, 12)
    rx, ry, rw, rh = region
    crop = image[ry:ry+rh, rx:rx+rw]
    if crop.shape != (rh, rw, 3) or crop.min() < 250:
        raise ValueError('opening-only item expected counter cap is not blank')
    absence = dict(profile='opaque_badge_upper_cap', minimum_channel=int(crop.min()),
                   white_fraction=float((crop.min(axis=2) >= 250).mean()))
    visual = dict(profile='faded_name_and_start_trigger')
    for key, row in (('name', name), ('trigger', triggers[0])):
        tx, ty, tw, th = row.box
        source = image[ty:ty+th, tx:tx+tw]
        ink = (source.min(axis=2) < 230) & (np.ptp(source, axis=2) <= 12)
        if source.min() < 180 or int(ink.sum()) < 20 or int(source[ink].min()) > 224:
            raise ValueError('opening-only item complete name and trigger are not visibly resolved')
        visual.update({key+'_box': tuple(row.box), key+'_ink_min': int(source[ink].min()),
                       key+'_ink_pixels': int(ink.sum())})
    return OpeningOnlyItemDescriptionRow(box, expected, None, None,
        min(confidence, *(row.score for row in body)), description, region, absence, visual)


def _item_description_name_rows(text, box, header_height=90):
    """Separate an explicit lower countdown from the complete item title.

    Keep unknown or weak second lines in the candidate set so existing
    ambiguity rejection remains effective. The countdown is neither a name
    nor the separate remaining-use badge or stage-use limit.
    """
    x, y, width, _ = box
    names = [row for row in text if _inside({'box': tuple(row.box)},
        (x+90, y+8, width-110, header_height))]
    if len(names) != 2:
        return names
    title, countdown = sorted(names, key=lambda row: row.box[1])
    tx, ty, _, th = title.box
    cx, cy, _, _ = countdown.box
    if (re.fullmatch(r'[1-9]\d*ターン', _normalize_status_text(countdown.text))
            and type(countdown.score) in (int, float)
            and math.isfinite(countdown.score) and .95 <= countdown.score <= 1
            and ty+th <= y+55 and cy >= max(ty+th, y+40)
            and abs(cx-tx) <= 12):
        return [title]
    return names


def read_item_description(context, image, *, text_rows=None):
    boxes, thumb, top, bottom, partial = status_panel_geometry(image, row_white_min=250)
    text = _inventory_ocr(context, image, 'LiveStatusItemDescription') if text_rows is None else text_rows
    rows = []
    for box in boxes:
        x,y,w,h = box
        names = _item_description_name_rows(text, box)
        if len(names) == 1 and _normalize_status_text(names[0].text) == 'H.I.F応援棒(緑)':
            rows.append(_read_opening_only_item(context, image, box, names[0], text))
            continue
        limits = [r for r in text if _inside({'box': tuple(r.box)}, (x+15,y+110,w-42,h-120))
                  and ('試験' in r.text or 'ステージ内' in r.text)]
        if len(names) != 1 or len(limits) != 1:
            raise ValueError('item description name or limit is missing or ambiguous')
        if not math.isfinite(names[0].score) or names[0].score < .95:
            raise ValueError('item description name or limit confidence is insufficient')
        limit = limits[0]
        text_value, limit_confidence = _confirmed_item_text(context, image, dict(
            text=_normalize_status_text(limit.text), score=limit.score, box=tuple(limit.box)), usage_limit=True)
        semantics = _item_limit_semantics(text_value)
        if semantics is None or semantics[0] != '試験・ステージ内':
            raise ValueError('item description stage-use limit is incomplete')
        counter = read_explicit_item_counter(context, image, (x+60,y+5,45,50), use_limit=semantics[1])
        rows.append(ItemDescriptionRow(box, _normalize_status_text(names[0].text), counter,
                                      semantics[1], min(names[0].score,limit_confidence,counter.confidence)))
    # Keep clipped rows only as continuity evidence. Complete field reading
    # above remains mandatory before a final item can be emitted.
    mask = (image[20:738,39:681].min(axis=2)>=250).astype(np.uint8)
    _,_,stats,_ = cv2.connectedComponentsWithStats(mask)
    candidates = sorted(((int(x+39),int(y+20),int(w),int(h)) for x,y,w,h,area in stats[1:]
                         if 625<=w<=642 and h>=40 and area/(w*h)>=.80),key=lambda b:b[1])
    headers = []
    for box in candidates:
        x,y,w,h = box
        known = next((row for row in rows if row.box==box),None)
        if known is not None:
            headers.append(ItemDescriptionHeader(box,known.name,known.remaining))
            continue
        header_height = min(90,738-y-8)
        counter_height = min(50,738-y-5)
        if header_height < 20 or counter_height < 20:
            continue
        names = _item_description_name_rows(text, box, header_height)
        if len(names)!=1 or not math.isfinite(names[0].score) or names[0].score<.95:
            continue
        try:
            # 99 is only the counter format bound here. The actual stage
            # limit is checked on the mandatory complete row before output.
            counter = read_explicit_item_counter(context,image,(x+60,y+5,45,counter_height),use_limit=99)
        except ValueError:
            continue
        headers.append(ItemDescriptionHeader(box,_normalize_status_text(names[0].text),counter))
    return ItemDescriptionPage(tuple(rows),thumb,top,bottom,partial,tuple(headers))


def item_description_stable_geometry(images):
    if not images:
        raise ValueError('item description requires an observed frame')
    geometries = [status_panel_geometry(image, row_white_min=250) for image in images]
    first = geometries[0]
    if len(images) == 1:
        return first
    if any(len(g[0]) != len(first[0]) or g[2:] != first[2:]
           or max(abs(a-b) for a,b in zip(g[1],first[1])) > 1
           or any(max(abs(a-b) for a,b in zip(row,prior)) > 1
                  for row,prior in zip(g[0],first[0])) for g in geometries[1:]):
        raise ValueError('item description geometry has not settled')
    boxes = tuple((x+15,y+12,90,90) for x,y,_,_ in geometries[0][0])
    errors = measure_p_item_content_generation(images,boxes)
    if max(errors) > PItemReferenceRuntime().content_generation_max_mean_abs_error:
        raise ValueError('item description rendered content has not settled')
    return geometries[0]


def _exhausted_cloth_navigation_target(image):
    """Recognize the complete cloth silhouette despite translucent lighting.

    This is a navigation anchor only; exhaustion/counts are never inferred.
    The outline is calibrated from 651-before and held out on 651-after.
    """
    region = image[300:385, 620:720]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,10,30)
    if not edges.any():
        return None
    distance = cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    # Five sidebar items use the game's compact 80% icon layout. Keep the
    # original shape/edge gates; this adds geometry, not a lower match score.
    for scale, xs, ys in ((1., range(10,26), range(10,31)),
            (.8, range(20,38), range(0,16))):
        target = _cloth_outline_target(distance, scale=scale, xs=xs, ys=ys)
        if target is not None:
            return target
    return None


def _cloth_outline_target(distance, *, scale, xs, ys):
    polygon = np.rint(np.array([[29,0],[60,22],[37,47],[23,52],[0,25]])*scale).astype(np.int32)
    segments = []
    for start,end in zip(polygon,np.roll(polygon,-1,axis=0),strict=True):
        mask = np.zeros((53,61),np.uint8)
        cv2.line(mask,tuple(start),tuple(end),1,1)
        segments.append(np.argwhere(mask))
    candidates = []
    for y in ys:
        for x in xs:
            distances = [distance[line[:,0]+y,line[:,1]+x] for line in segments]
            # Every edge must be independently present. Background edges may
            # connect to it; a closed contour is not required for this reason.
            if all(np.mean(row)<=.75 and np.mean(row<=2)>=.95 for row in distances):
                score = float(np.mean(np.concatenate(distances)))
                if score <= .45:
                    candidates.append((score,x,y))
    candidates.sort()
    if not candidates:
        return None
    score,x,y = candidates[0]
    if any(other-score < .05 and max(abs(xx-x),abs(yy-y))>1
           for other,xx,yy in candidates[1:]):
        return None
    return (620+x-1,300+y,round(63*scale),round(53*scale))



def _item_description_target(context, image, *, opening=False, preferred_anchor=None, text_rows=None):
    """Recognize the calibrated HIF entry or an independently visible popup.

    The usual entries are the first two items. The fourth-slot mask is only
    used for its saved opening continuation. Details supply identity and count.
    """
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('item description resolution is uncalibrated')
    if preferred_anchor not in (None, 'item', 'cloth', 'fourth') or (preferred_anchor is not None and not opening):
        raise ValueError('item description preferred anchor is invalid')
    cloth_first = opening and preferred_anchor == 'cloth'
    roi = (620, 215, 100, 110) if opening else (250, 300, 220, 940)
    node = 'LiveStatusItemOpen' if opening else 'LiveStatusItemClose'
    params = dict(recognition='TemplateMatch', roi=list(roi), threshold=.95,
        template=['produce/hif_live_item_status_entry.png' if opening else 'produce/hif_status_close.png'])
    if opening:
        params.update(method=3, green_mask=True)
        if cloth_first:
            roi, node = (620, 300, 100, 85), 'LiveStatusItemClothOpen'
            params.update(roi=list(roi), template=['produce/hif_live_item_status_cloth_entry.png'], method=5)
        elif preferred_anchor == 'fourth':
            roi, node = (620, 435, 100, 95), 'LiveStatusItemFourthOpen'
            params.update(roi=list(roi), template=['produce/hif_live_item_slot4_hat.png'], method=5)
    elif getattr(context, 'gkh_hif_upstream_card_ui', False) is not True:
        # Neither an X alone nor a click acknowledgement proves this popup.
        rows = _inventory_ocr(context, image, 'LiveStatusItemDescription') if text_rows is None else text_rows
        scopes = [r for r in rows if math.isfinite(r.score) and r.score >= .95
                  and '試験' in _normalize_status_text(r.text)
                  and 'ステージ内' in _normalize_status_text(r.text)
                  and _inside({'box': tuple(r.box)}, (15, 100, 690, 1120))]
        white = (image.min(axis=2) >= 235).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(white)
        if not scopes or not any(600 <= w <= 700 and h >= 120 and area/(w*h) >= .65
                                 for _, _, w, h, area in stats[1:]):
            raise ValueError('item description popup anchors are missing')
    hit = context.run_recognition(node, image, pipeline_override={node: params})
    matches = list(hit.filtered_results) if hit and hit.hit else []
    if (opening and preferred_anchor == 'fourth' and not matches
            and getattr(context, 'gkh_hif_client_size', None) == (359, 638)):
        # Five icons use a shorter vertical pitch. The fourth hat sits above
        # the old four-icon slot, while the fifth icon occupies that old ROI.
        roi, node = (635, 409, 73, 54), 'LiveStatusItemFourthCompactOpen'
        params.update(roi=list(roi), template=['produce/hif_live_item_slot4_hat_compact.png'])
        hit = context.run_recognition(node, image, pipeline_override={node: params})
        matches = list(hit.filtered_results) if hit and hit.hit else []
    if cloth_first and not matches:
        outline = _exhausted_cloth_navigation_target(image)
        if outline is not None:
            return outline
    if opening and not matches and preferred_anchor != 'fourth':
        # The first item becomes translucent after exhaustion. The opaque
        # cloth provides an independently matched navigation control; its
        # remaining uses still come from the complete description list.
        roi = (620, 300, 100, 85)
        node = 'LiveStatusItemClothOpen'
        params = dict(recognition='TemplateMatch', roi=list(roi), threshold=.95,
            template=['produce/hif_live_item_status_cloth_entry.png'], method=5, green_mask=True)
        if cloth_first:
            roi, node = (620, 215, 100, 110), 'LiveStatusItemOpen'
            params.update(roi=list(roi), template=['produce/hif_live_item_status_entry.png'], method=3)
        hit = context.run_recognition(node, image, pipeline_override={node: params})
        matches = list(hit.filtered_results) if hit and hit.hit else []
        if not matches and not cloth_first:
            outline = _exhausted_cloth_navigation_target(image)
            if outline is not None:
                return outline
    if len(matches) != 1:
        sink = getattr(context, 'gkh_hif_item_failure_sink', None)
        if opening and callable(sink):
            sink(image)
        raise ValueError('item description target is not unique: ' + node)
    box = tuple(matches[0].box)
    if (len(box) != 4 or any(type(v) is not int for v in box) or min(box[2:]) <= 0
            or not _inside({'box': box}, roi)):
        raise ValueError('item description target is outside its control region')
    return box


def _item_opening_anchor(box):
    """Distinguish only the two existing, independently matched entry regions."""
    matches = [name for name, roi in (
        ('item', (620, 215, 100, 110)), ('cloth', (620, 300, 100, 85)))
        if _inside({'box': box}, roi)]
    return matches[0] if len(matches) == 1 else None


def observe_item_description_transaction(context, *, capture_frames, checkpoint,
                                         frame_identifier, maximum_age_ms=1000, complete_list=False,
                                         expected_live=None):
    """Inspect one page or the complete list, close it, and verify the Live return."""
    from utils.maa_input import verify_maa_input_endpoints

    from .status_panel import StatusInspectionError

    if type(maximum_age_ms) not in (int, float) or not 0 < maximum_age_ms <= 1000:
        raise ValueError('item description freshness gate is invalid')
    if type(complete_list) is not bool:
        raise ValueError('item description completeness mode is invalid')
    controller = context.tasker.controller
    tail_resume = getattr(context, 'gkh_hif_item_tail_resume', None)
    event = dict(phase='item_description_inspection', accepted=False,
                 calibration_only=not complete_list, strategy_state_bound=False,
                 complete_list=False, input_click_count=0, inputs=[], pages=[], scroll_drag_pixels=80)
    started = time.perf_counter()
    limit_batch_time = getattr(context, 'gkh_hif_upstream_card_ui', False) is not True
    arrival_timeout_seconds = 5 if limit_batch_time else 10

    def fresh():
        image = controller.post_screencap().wait().get()
        return image, frame_captured_at(controller, image)

    def reobserve_scroll_anchor(error, image, phase, expected_thumb):
        if phase != 'scroll' or str(error) != 'item description popup anchors are missing':
            return False
        # A single-frame OCR miss cannot authorize a swipe. Let the existing
        # bounded page loop replace this page with three complete observations.
        event.setdefault('scroll_reobservations', []).append(dict(
            reason=str(error), expected_thumb=expected_thumb,
            frame_id=frame_identifier(image)))
        return True

    def dispatch(image, phase, *, expected_thumb=None):
        # OCR may take time. Reconfirm the same recognized control on a fresh
        # capture after popup semantics, before validating all input endpoints.
        for attempt in range(3):
            try:
                if phase == 'open' and tail_resume is not None:
                    box = _item_description_target(context, image, opening=True, preferred_anchor='fourth')
                else:
                    box = _item_description_target(context, image, opening=phase == 'open')
                break
            except ValueError as error:
                if reobserve_scroll_anchor(error, image, phase, expected_thumb):
                    return False
                if (phase != 'open' or attempt == 2
                        or not str(error).startswith('item description target is not unique:')):
                    raise
                event.setdefault('opening_target_reobservations', []).append(str(error))
                image, _, current_state = capture_fresh_live_checkpoint(
                    fresh, capture_frames, checkpoint, before, event)
                if any(current_state[k] != before[k] for k in ('families', 'fields')):
                    raise ValueError('Live changed while reobserving item description control')
        current, captured_at = fresh()
        if phase == 'open':
            current, captured_at, current_state = capture_fresh_live_checkpoint(
                fresh, capture_frames, checkpoint, before, event)
            if any(current_state[k] != before[k] for k in ('families', 'fields')):
                raise ValueError('Live changed immediately before item description input')
        try:
            preferred = None
            if phase == 'open':
                preferred = 'fourth' if tail_resume is not None else _item_opening_anchor(box)
            current_box = _item_description_target(context, current, opening=phase == 'open',
                preferred_anchor=preferred)
        except ValueError as error:
            if reobserve_scroll_anchor(error, current, phase, expected_thumb):
                return False
            raise
        if (phase == 'open' and event['input_click_count'] == 0
                and _item_opening_anchor(box) is not None
                and _item_opening_anchor(current_box) is not None
                and _item_opening_anchor(box) != _item_opening_anchor(current_box)):
            # A background highlight can make the first entry fall below its
            # existing threshold and select the independently matched cloth.
            # Neither entry was clicked. One more fresh unchanged Live may
            # reconfirm the original anchor, or the alternate twice in a row.
            # Both paths retain the original same-control/freshness gate.
            event.setdefault('opening_target_reanchors', []).append(dict(
                before=box, current=current_box, frame_id=frame_identifier(current),
                input_click_count=0, reason='known_item_entry_changed_before_input'))
            alternate_box = current_box
            current, captured_at, current_state = capture_fresh_live_checkpoint(
                fresh, capture_frames, checkpoint, before, event)
            if any(current_state[k] != before[k] for k in ('families', 'fields')):
                raise ValueError('Live changed while reanchoring item description control')
            current_box = _item_description_target(context, current, opening=True,
                preferred_anchor=_item_opening_anchor(box))
            if _item_opening_anchor(current_box) != _item_opening_anchor(box):
                box = alternate_box
            else:
                event['opening_target_reanchors'][-1]['original_anchor_reconfirmed'] = True
        if box != current_box:
            if (phase != 'open' or tuple(box[2:]) != tuple(current_box[2:])
                    or max(abs(a-b) for a,b in zip(box[:2],current_box[:2],strict=True)) > 1):
                raise ValueError('item description control moved before input')
            event.setdefault('control_raster_jitter', []).append(dict(before=box,current=current_box))
            box = current_box
        if phase in ('scroll', 'scroll_top'):
            geometry = status_panel_geometry(current, row_white_min=250)
            if (expected_thumb is None or geometry[2 if phase == 'scroll_top' else 3]
                    or max(abs(a-b) for a,b in zip(geometry[1],expected_thumb)) > 2):
                # The earlier three frames can agree while inertial scrolling
                # has not quite stopped. No second swipe is allowed from that
                # stale page; replace its observation at the current position.
                event.setdefault('scroll_reobservations', []).append(dict(
                    expected_thumb=expected_thumb, current_thumb=geometry[1],
                    current_at_bottom=geometry[3], frame_id=frame_identifier(current)))
                return False
        endpoints = verify_maa_input_endpoints(controller.info, expected_client_size=getattr(context, "gkh_hif_client_size", (720, 1280)))
        if (time.perf_counter()-captured_at)*1000 > maximum_age_ms:
            raise ValueError('item description snapshot expired')
        if limit_batch_time and phase != 'close' and time.perf_counter()-started > 50:
            raise ValueError('item description navigation budget reached')
        event['input_click_count'] += 1
        event['inputs'].append(dict(phase=phase, box=box, endpoints=endpoints))
        if phase == 'open' and tuple(box[2:]) == (50,42):
            # Compact five-item lists need more progress than the old 80px
            # stroke. Every following page must still overlap and reach the
            # real bottom within the unchanged page/time budgets.
            event['scroll_drag_pixels'] = 120
        x,y,w,h = box
        action = (controller.post_swipe(560,610,560,610-event['scroll_drag_pixels'],400) if phase == 'scroll'
                  else controller.post_swipe(560,220,560,610,600) if phase == 'scroll_top'
                  else controller.post_click(x+w//2,y+h//2))
        if not action.wait().succeeded:
            raise ValueError('Maa item description input failed')
        return True

    try:
        pages = []
        observed_fourth = None
        if tail_resume is not None:
            if expected_live is not None:
                raise ValueError('HUD item confirmation cannot resume a prior detail transaction')
            from route.engine import has_single_frame_id

            from .hif_item_tail_evidence import item_prefix_consensus

            if not complete_list or tail_resume['slot'] != 4 or not tail_resume['source_result']:
                raise ValueError('item tail resume requires the fourth slot and its recorded prefix')
            entries = tail_resume['pages']
            fourth = tail_resume.get('observed_fourth')
            if (any(not has_single_frame_id(entry.get('frame_ids')) for entry in entries)
                    or fourth is not None and not has_single_frame_id(fourth.get('frame_ids'))):
                raise ValueError('item tail resume requires a current single-frame receipt for each page')
            pages = [item_prefix_consensus(entry) for entry in entries]
            prefix = join_item_description_pages(pages, complete=False)
            if len(prefix) != 3 or pages[-1].at_bottom:
                raise ValueError('item tail resume requires three complete preceding items')
            if fourth is not None:
                observed_fourth, = item_prefix_consensus(fourth).rows
            event.update(resumed_item_source=tail_resume['source_result'], entered_item_slot=4,
                         prefix_page_frame_ids=[entry['frame_ids'] for entry in tail_resume['pages']])
        prefix_pages = len(pages)
        if tail_resume and tail_resume.get('open_before'):
            before = tail_resume['open_before']
            if not has_single_frame_id(before.get('frame_ids')) or not has_single_frame_id(tail_resume.get('before_frame_ids')):
                raise ValueError('open item tail resume requires current single-frame checkpoints')
            event.update(before_frame_ids=tail_resume['before_frame_ids'],
                resumed_open_item_source=tail_resume['opened_source_result'])
        else:
            _, before = capture_initial_live_checkpoint(capture_frames, checkpoint, event,
                poll_progress=getattr(context, 'gkh_hif_poll_battle_progress', None))
            event['before_frame_ids'] = before['frame_ids']
            if expected_live is not None and any(before[k] != expected_live[k] for k in ('families', 'fields')):
                raise ValueError('Live changed since the item HUD observation')
            image, _, current = capture_fresh_live_checkpoint(fresh, capture_frames, checkpoint, before, event)
            if any(current[k] != before[k] for k in ('families','fields')):
                raise ValueError('Live changed before item description navigation')
            dispatch(image, 'open')
        failure = None
        try:
            pages_to_read = count() if complete_list and not limit_batch_time else range(8 if complete_list else 1)
            for page_index in pages_to_read:
                arrival_started = time.perf_counter()
                for attempt in range(12):
                    frames = capture_frames()
                    try:
                        item_description_stable_geometry(frames)
                        # The popup anchor and body consume the same OCR of
                        # this exact frame. Every other frame is read afresh.
                        texts = [_inventory_ocr(context, frame, 'LiveStatusItemDescription') for frame in frames]
                        boxes = [_item_description_target(context, frame, text_rows=rows)
                                 for frame, rows in zip(frames, texts, strict=True)]
                        if any(box != boxes[0] for box in boxes[1:]):
                            raise ValueError('item description page has no three-frame consensus')
                        observations = [read_item_description(context, frame, text_rows=rows)
                                        for frame, rows in zip(frames, texts, strict=True)]
                        page = item_panel_consensus(observations)
                        for index in range(len(page.rows)):
                            if page.rows[index].remaining is not None:
                                item_counter_consensus(o.rows[index].remaining for o in observations)
                            elif not all(isinstance(o.rows[index], OpeningOnlyItemDescriptionRow)
                                         for o in observations):
                                raise ValueError('item counter absence has no complete start-only description')
                        for index in range(len(page.headers)):
                            if page.headers[index].remaining is not None:
                                item_counter_consensus(o.headers[index].remaining for o in observations)
                            elif not all(any(isinstance(row, OpeningOnlyItemDescriptionRow)
                                             and row.name == o.headers[index].name for row in o.rows)
                                         for o in observations):
                                raise ValueError('item header without counter has no complete description')
                        break
                    except ValueError as error:
                        event.setdefault('arrival_observations', []).append(str(error))
                        if (attempt == 11 or time.perf_counter()-arrival_started > arrival_timeout_seconds
                                or (limit_batch_time and time.perf_counter()-started > 45)):
                            sink = getattr(context, 'gkh_hif_item_failure_sink', None)
                            if callable(sink):
                                for frame in frames:
                                    sink(frame)
                            raise
                        time.sleep(.08)
                if complete_list and not pages and not page.at_top:
                    # Clicking a later item positions the list at that item.
                    # Establish the actual top before collecting any prefix.
                    attempts = event.get('top_navigation_attempts', 0)
                    if attempts >= 2:
                        raise ValueError('item description top was not reached')
                    event['top_navigation_attempts'] = attempts + 1
                    event['top_arrival_pages'] = [asdict(o) for o in observations]
                    image, _ = fresh()
                    dispatch(image, 'scroll_top', expected_thumb=page.thumb)
                    continue
                if tail_resume is not None and len(pages) == prefix_pages:
                    page = _FourthSlotEntryPage(**vars(page))
                pages.append(page)
                event['pages'].append(dict(frame_ids=[frame_identifier(frame) for frame in frames],
                                           observations=[asdict(o) for o in observations]))
                event['popup_confirmed'] = True
                if complete_list:
                    join_item_description_pages(pages,complete=False)
                if not complete_list or page.at_bottom:
                    break
                if limit_batch_time and page_index == 7:
                    raise ValueError('item description page budget reached')
                image, _ = fresh()
                if not dispatch(image, 'scroll', expected_thumb=page.thumb):
                    poll = getattr(context, 'gkh_hif_poll_battle_progress', None)
                    if not limit_batch_time and callable(poll):
                        poll(image)
                    pages.pop()
                    event['pages'].pop()
                    continue
            if complete_list:
                rows = join_item_description_pages(pages)
                if observed_fourth is not None and rows[3].content_key != observed_fourth.content_key:
                    raise ValueError('item description complete readings disagree')
                event['rows'] = [asdict(row) for row in rows]
                event['complete_list'] = True
        except (ValueError, RuntimeError) as error:
            failure = str(error)
            event['read_failure'] = failure
        image, _ = fresh()
        dispatch(image, 'close')
        if callable(getattr(checkpoint, '__dict__', {}).get('capture_complete')):
            _, after = capture_initial_live_checkpoint(capture_frames, checkpoint, event)
        else:
            for attempt in range(6):
                try:
                    after = checkpoint(capture_frames())
                    break
                except ValueError as error:
                    event.setdefault('return_observations', []).append(str(error))
                    if attempt == 5:
                        raise
                    time.sleep(.08)
        event['after_frame_ids'] = after['frame_ids']
        if any(json.dumps(after[k]) != json.dumps(before[k]) for k in ('families','fields')):
            raise ValueError('item description changed the Live hand or resources')
        event['returned_unchanged'] = True
        if failure:
            raise ValueError(failure)
        event.update(accepted=True, elapsed_ms=(time.perf_counter()-started)*1000)
        return tuple(pages), [event]
    except Exception as error:
        event.update(reason=str(error), elapsed_ms=(time.perf_counter()-started)*1000)
        raise StatusInspectionError(str(error), [event]) from error


@dataclass(frozen=True)
class ItemPanelRow:
    box: tuple[int, int, int, int]
    icon_box: tuple[int, int, int, int]
    name: str
    used_count: int | None
    use_limit: int | None
    limit_scope: str | None
    confidence: float

    @property
    def content_key(self):
        return self.name,self.used_count,self.use_limit,self.limit_scope


@dataclass(frozen=True)
class ItemPanelPage:
    rows: tuple[ItemPanelRow, ...]
    thumb: tuple[int, int]
    at_top: bool
    at_bottom: bool
    partial_row_count: int

    @property
    def content_key(self):
        return tuple(row.content_key for row in self.rows),self.at_top,self.at_bottom


@dataclass(frozen=True)
class ItemRemainingCounter:
    value: int
    box: tuple[int, int, int, int]
    confidence: float


def _explicit_counter_candidates(crop, mask, x, y, w, h):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        bx,by,bw,bh=cv2.boundingRect(contour)
        if bx==0 or by==0 or bx+bw>=w or by+bh>=h:
            continue
        area=cv2.contourArea(contour)
        hull=cv2.convexHull(contour)
        hull_area=cv2.contourArea(hull);perimeter=cv2.arcLength(hull,True)
        if not (10<=min(bw,bh)<=max(bw,bh)<=64 and .85<=bw/bh<=1.18
                and .60<=area/(bw*bh)<=.86 and hull_area>0 and perimeter>0
                and area/hull_area>=.85 and 4*math.pi*hull_area/perimeter**2>=.75):
            continue
        # Item artwork can touch the circle's lower edge. Fit its outer hull
        # only for geometry; retain the original contour for visible glyphs,
        # so an obscured digit is never reconstructed from the fitted circle.
        interior=np.zeros((h,w),np.uint8)
        cv2.drawContours(interior,[contour],-1,255,cv2.FILLED)
        glyph=(crop.min(axis=2)>=220)&(interior>0)
        yy,xx=np.where(glyph)
        if not len(xx) or len(xx)/(bw*bh)<.05:
            continue
        if xx.min()<=bx or xx.max()>=bx+bw-1 or yy.min()<=by or yy.max()>=by+bh-1:
            continue
        digit=255-glyph[by:by+bh,bx:bx+bw].astype(np.uint8)*255
        digit=cv2.copyMakeBorder(digit,6,6,6,6,cv2.BORDER_CONSTANT,value=255)
        candidates.append(((x+bx,y+by,bw,bh),cv2.cvtColor(digit,cv2.COLOR_GRAY2BGR)))
    return candidates


def read_explicit_item_counter(context,image,search_box, *, use_limit):
    """Read a visible grey-circle counter on the battle-item description page.

    D-068: this is the remaining count, including an explicit zero. Missing
    badges are unknown here; the HUD's exhausted appearance is a separate gate.
    """
    if type(use_limit) is not int or not 1<=use_limit<=99:
        raise ValueError('item counter requires an observed finite use limit')
    if getattr(image,'ndim',None)!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('item counter frame is invalid')
    if len(search_box)!=4 or any(type(v) is not int for v in search_box):
        raise ValueError('item counter search box is invalid')
    x,y,w,h=search_box
    if min(x,y)<0 or min(w,h)<1 or x+w>image.shape[1] or y+h>image.shape[0]:
        raise ValueError('item counter search box is clipped')
    crop=image[y:y+h,x:x+w]
    gray=((np.ptp(crop,axis=2)<=12)&(crop.min(axis=2)>=90)&(crop.max(axis=2)<=200)).astype(np.uint8)*255
    # Prefer the actual separated circle. Closing a healthy badge can join
    # adjacent item artwork (1929). Only repair a gap when no original circle
    # exists; multiple original circles remain ambiguous.
    candidates = _explicit_counter_candidates(crop, gray, x, y, w, h)
    if not candidates:
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        candidates = _explicit_counter_candidates(crop, closed, x, y, w, h)
    if not candidates and getattr(context, 'gkh_hif_client_size', None) == (359, 638):
        # The compact hat art can touch the grey badge through tinted edge
        # pixels. Keep its neutral pixels and read the original white glyph.
        neutral = gray & (np.ptp(crop, axis=2) == 0).astype(np.uint8) * 255
        candidates = _explicit_counter_candidates(crop, neutral, x, y, w, h)
    if len(candidates)!=1:
        raise ValueError('explicit remaining-use badge is missing or ambiguous')
    box,digit=candidates[0]
    rows=_inventory_ocr(context,digit,'LiveItemRemainingCounter',only_rec=True)
    if len(rows)!=1 or not math.isfinite(rows[0].score) or rows[0].score<.95:
        raise ValueError('item remaining-use digit is unresolved')
    text=_normalize_status_text(rows[0].text)
    if not re.fullmatch(r'0|[1-9][0-9]?',text):
        raise ValueError('item remaining-use digit is invalid')
    value=int(text)
    if value>use_limit:
        raise ValueError('item remaining-use count exceeds its limit')
    return ItemRemainingCounter(value,box,rows[0].score)


def item_counter_consensus(observations):
    observations=tuple(observations)
    if any(not isinstance(o,ItemRemainingCounter) or type(o.value) is not int
           or not 0<=o.value<=99 or type(o.confidence) not in (int,float)
           or not math.isfinite(o.confidence) or not .95<=o.confidence<=1
           or len(o.box)!=4 or any(type(v) is not int for v in o.box)
           or min(o.box[:2])<0 or min(o.box[2:])<1 for o in observations):
        raise ValueError('item remaining-use observation is invalid')
    if not observations or any(o.value!=observations[0].value for o in observations[1:]):
        raise ValueError('item remaining-use count has no three-frame consensus')
    if any(max(abs(a-b) for a,b in zip(o.box,observations[0].box))>2 for o in observations[1:]):
        raise ValueError('item remaining-use badge has not settled')
    return observations[len(observations)//2]


def item_panel_consensus(pages):
    pages=tuple(pages)
    if not pages or any(p.content_key!=pages[0].content_key for p in pages[1:]):
        raise ValueError('item overview semantics have not settled')
    for page in pages[1:]:
        if (max(abs(a-b) for a,b in zip(page.thumb,pages[0].thumb))>2
                or any(max(abs(x-y) for x,y in zip(a.box,b.box))>2
                       for a,b in zip(page.rows,pages[0].rows))):
            raise ValueError('item overview geometry has not settled')
    return pages[len(pages)//2]


def item_panel_stable_geometry(images, *, expected_thumb_height=None):
    """Read the visible geometry; compare rendering when multiple frames are supplied."""
    if not images:
        raise ValueError('item overview requires an observed frame')
    geometries=[item_panel_geometry(image,expected_thumb_height=expected_thumb_height) for image in images]
    first=geometries[0]
    if len(geometries) == 1:
        return first
    if any(len(g[0])!=len(first[0]) or g[2:]!=first[2:]
           or max(abs(a-b) for a,b in zip(g[1],first[1]))>1
           or any(max(abs(a-b) for a,b in zip(x,y))>1
                  for x,y in zip(g[0],first[0])) for g in geometries[1:]):
        raise ValueError('item overview geometry has not settled')
    boxes=tuple((x+14,y+14,142,142) for x,y,_,_ in first[0])
    errors=measure_p_item_content_generation(images,boxes)
    if max(errors)>PItemReferenceRuntime().content_generation_max_mean_abs_error:
        raise ValueError('item overview rendered content has not settled')
    return first


def item_panel_geometry(image, *, expected_thumb_height=None):
    if getattr(image,'shape',())!=(1280,720,3):
        raise ValueError('item overview resolution is uncalibrated')
    # Same white-row segmentation as the status list, with this overview's
    # larger icon area and viewport. Clipped rows never become observations.
    mask=(image[100:1007,20:700].min(axis=2)>=235).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    boxes=sorted(((int(x+20),int(y+100),int(w),int(h)) for x,y,w,h,area in stats[1:]
                  if 665<=w<=680 and h>=40 and area/(w*h)>=.65),key=lambda b:b[1])
    # The upper viewport has a 20-pixel fade: a clipped row can begin as a
    # separate bright component inside it. Only rows below the fade have a
    # visible top edge (the unscrolled first row begins at y=123).
    complete=tuple(b for b in boxes if b[1]>120 and b[1]+b[3]<995 and b[3]>=165)
    if not complete:
        raise ValueError('item overview has no complete row')
    if any(not 7<=b[1]-a[1]-a[3]<=20 for a,b in zip(boxes,boxes[1:])):
        raise ValueError('item overview row spacing is inconsistent')
    strip=image[115:1001,703].astype(np.int16)
    eligible=strip[(np.ptp(strip,axis=1)<=32)&(strip.min(axis=1)>=90)&(strip.max(axis=1)<=180)]
    colors,counts=np.unique(eligible,axis=0,return_counts=True)
    spans=[]
    for color,count in zip(colors,counts):
        if count<30:
            continue
        solid=np.max(np.abs(image[115:1001,703:705].astype(np.int16)-color),axis=(1,2))<=2
        edges=np.diff(np.r_[False,solid,False].astype(np.int8))
        for a,b in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
            span=(int(a+115),int(b+115))
            if b-a>=30 and not any(max(abs(x-y) for x,y in zip(span,old))<=3 for old in spans):
                spans.append(span)
    if expected_thumb_height is not None:
        if type(expected_thumb_height) is not int or not 30<=expected_thumb_height<=886:
            raise ValueError('item overview observed scrollbar height is invalid')
        # A scroll-only transaction keeps the scrollbar thumb's size. The
        # already observed first page distinguishes it from a short uniform
        # background strip without choosing the longest or darkest candidate.
        spans=[span for span in spans if abs(span[1]-span[0]-expected_thumb_height)<=2]
    if len(spans)!=1:
        raise ValueError('item overview scrollbar is not unique')
    top,bottom=spans[0]
    # The solid thumb interior ends at 992 on the actual bottom page; the
    # remaining viewport pixels are footer padding and the rounded end cap.
    return complete,(top,bottom),115<=top<=120,990<=bottom<=994,len(boxes)-len(complete)


def item_limit_text_image(source):
    """Exclude pale bullet stars and normalize the actual dark-text extent."""
    if source.ndim!=3 or source.shape[2]!=3 or source.dtype!=np.uint8 or not source.size:
        raise ValueError('item limit crop is invalid')
    ink=source.min(axis=2)<120
    yy,xx=np.where(ink)
    if not len(xx):
        raise ValueError('item limit has no visible text')
    glyphs=255-ink[yy.min():yy.max()+1,xx.min():xx.max()+1].astype(np.uint8)*255
    glyphs=cv2.copyMakeBorder(glyphs,6,6,6,6,cv2.BORDER_CONSTANT,value=255)
    return cv2.cvtColor(glyphs,cv2.COLOR_GRAY2BGR)


def _item_limit_semantics(text, *, repeated_delimiters=False):
    opening,closing=(r'\(+',r'\)+') if repeated_delimiters else (r'\(',r'\)')
    match=re.fullmatch(opening+r'(プロデュース中|試験・ステージ内)(\d{1,2})回'+closing,text)
    return (match[1],int(match[2])) if match else None


def _confirmed_item_text(context, image, row, *, usage_limit=False):
    if type(row['score']) not in {int,float} or not math.isfinite(row['score']):
        raise ValueError('item text confidence is invalid')
    if row['score']>=.95:
        return row['text'],row['score']
    if row['score']<.85:
        raise ValueError('item text confidence is insufficient')
    x,y,w,h=row['box']
    source=image[y:y+h,x:x+w]
    if usage_limit:
        if (getattr(context, 'gkh_hif_client_size', None) == (359, 638)
                and _item_limit_semantics(row['text']) is not None):
            # Binarization erases thin strokes at this scale. The original
            # line can independently confirm the exact stage limit and count.
            found = _inventory_ocr(context, source, 'LiveInventoryItemText', only_rec=True)
            if (len(found) == 1 and .95 <= found[0].score <= 1
                    and _normalize_status_text(found[0].text) == row['text']):
                return row['text'], found[0].score
        crop=item_limit_text_image(source)
    else:
        crop=cv2.copyMakeBorder(source,6,6,6,6,cv2.BORDER_CONSTANT,value=(255,255,255))
    found=_inventory_ocr(context,crop,'LiveInventoryItemText',only_rec=True)
    if len(found)!=1 or not math.isfinite(found[0].score) or found[0].score<.95:
        raise ValueError(f'item text reread is unresolved: {row["text"]}: '
                         f'{[(r.text,r.score) for r in found]}')
    text=_normalize_status_text(found[0].text)
    if text!=row['text']:
        # A duplicated outer parenthesis still has to agree on scope and count.
        canonical=_item_limit_semantics(text) if usage_limit else None
        original=_item_limit_semantics(row['text'],repeated_delimiters=True) if usage_limit else None
        if canonical is None or canonical!=original:
            raise ValueError('item text reread changed its meaning')
    return text,found[0].score


def read_item_panel(context,image, *, expected_thumb_height=None):
    boxes,thumb,at_top,at_bottom,partial=item_panel_geometry(image,expected_thumb_height=expected_thumb_height)
    rows=_inventory_ocr(context,image,'LiveInventoryItemsOverview')
    _unique_inventory_text(rows,'所持Pアイテム',(15,25,285,65))
    _unique_inventory_text(rows,'閉じる',(210,1120,290,85))
    rows=[dict(text=_normalize_status_text(r.text),score=r.score,box=tuple(r.box)) for r in rows]

    observed=[]
    for box in boxes:
        x,y,w,h=box
        headers=[r for r in rows if _inside(r,(180,y+3,485,42))]
        if len(headers)!=1:
            raise ValueError('item name is missing or ambiguous')
        name,confidence=_confirmed_item_text(context,image,headers[0])
        uses=[r for r in rows if _inside(r,(30,y+155,155,max(h-155,1))) and '回発動済み' in r['text']]
        if len(uses)>1:
            raise ValueError('item used counter is ambiguous')
        used=None
        if uses:
            value,score=_confirmed_item_text(context,image,uses[0]);confidence=min(confidence,score)
            match=re.fullmatch(r'(\d{1,2})回発動済み',value)
            if not match:
                raise ValueError('item used counter is incomplete')
            used=int(match[1])
        limits=[r for r in rows if _inside(r,(180,y+45,490,h-48))
                and ('プロデュース中' in r['text'] or '試験・ステージ内' in r['text'])]
        if len(limits)>1:
            raise ValueError('item usage scopes are ambiguous')
        limit,scope=None,None
        if limits:
            value,score=_confirmed_item_text(context,image,limits[0],usage_limit=True);confidence=min(confidence,score)
            match=re.fullmatch(r'\((プロデュース中|試験・ステージ内)(\d{1,2})回\)',value)
            if not match:
                raise ValueError('item usage limit is incomplete')
            scope,limit=match[1],int(match[2])
        if used is not None and limit is not None and used>limit:
            raise ValueError('item used count exceeds its limit')
        observed.append(ItemPanelRow(box,(x+14,y+14,142,142),name,used,limit,scope,confidence))
    return ItemPanelPage(tuple(observed),thumb,at_top,at_bottom,partial)
