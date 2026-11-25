"""Batch-readable inventory identities, independent of overlapping Live hands."""

from __future__ import annotations

import re
import math
import time
import unicodedata
from copy import deepcopy
from types import SimpleNamespace
from itertools import count
from collections import Counter
from dataclasses import asdict, replace, dataclass

import cv2
import numpy as np
from PIL import Image

from .layout import calibrated_live_client_size

try:
    from agent.card_selection.model import frame_identifier
    from agent.card_selection.embedding import EmbeddingHit, resolve_card_identity, extract_upgrade_marker
except ModuleNotFoundError:
    from card_selection.model import frame_identifier
    from card_selection.embedding import EmbeddingHit, resolve_card_identity, extract_upgrade_marker


def unresolved_live_resource_evidence(error):
    details=getattr(error,'events',())
    return details if details and all(e.get('phase')=='inventory_live_checkpoint'
                                     and e.get('unresolved_fields') for e in details) else ()


def inspected_single_frame_reobserve_allowed(checkpoint, error, image, before):
    """Only the explicitly bound observer may schedule a fresh confirmation."""
    allowed = getattr(checkpoint, '__dict__', {}).get('single_frame_reobserve_allowed')
    return allowed is not None and allowed(error, image, before) is True


def _complete_live_layout_reobserve_allowed(error, before):
    """A rejected row may request new pixels after a complete checkpoint."""
    from .partial_card import LiveCardLayoutError

    if (not isinstance(error, LiveCardLayoutError)
            or str(error) != 'HIF ordinary card row is unstable'
            or not isinstance(before, dict)):
        return False
    frame_ids, families, fields = (before.get(key) for key in ('frame_ids', 'families', 'fields'))
    # The existing checkpoint establishes an unselected hand; it has no
    # separate selected_slot field. Never use a partial/unknown envelope to
    # schedule this recovery, and never accept the failed single-frame row.
    return (isinstance(frame_ids, (tuple, list)) and bool(frame_ids)
        and all(isinstance(value, str) and value for value in frame_ids)
        and len(set(frame_ids)) == len(frame_ids)
        and isinstance(families, (tuple, list)) and len(families) in (4, 5)
        and all(isinstance(family, (tuple, list)) and family
                and all(isinstance(value, str) and value for value in family) for family in families)
        and isinstance(fields, (tuple, list)) and len(fields) == 6
        and all(value is not None for value in fields))


def capture_fresh_live_checkpoint(fresh, capture_frames, checkpoint, before, event):
    """Reobserve transient preinput failures; never replace observed values."""
    for attempt in range(3):
        image, captured_at = fresh()
        try:
            current = checkpoint((image,), single_frame=True)
        except ValueError as error:
            if (attempt == 2 or (str(error) != 'inventory fresh recheck is not an unselected hand'
                                and not unresolved_live_resource_evidence(error)
                                and not _complete_live_layout_reobserve_allowed(error, before)
                                and not fresh_identity_reobserve_allowed(error, before)
                                and not inspected_single_frame_reobserve_allowed(checkpoint, error, image, before))):
                raise
            _, confirmed = capture_initial_live_checkpoint(capture_frames, checkpoint, event)
            if any(confirmed[k] != before[k] for k in ('families', 'fields')):
                raise ValueError('Live changed during preinput reobservation') from error
            event.setdefault('preinput_reobservations', []).append(dict(
                reason=str(error),frame_ids=confirmed['frame_ids']))
            continue
        if any(current[k] != before[k] for k in ('families', 'fields')):
            raise ValueError('Live changed during preinput reobservation')
        return image,captured_at,current
    raise ValueError('Live preinput reobservation budget exhausted')


def initial_checkpoint_retry_evidence(error, frames):
    """The existing whole-group retry reasons, shared with scoped observers."""
    details = unresolved_live_resource_evidence(error)
    geometry = getattr(error, 'events', ())
    if (not details and geometry and all(e.get('phase') == 'inventory_live_geometry'
            and e.get('geometry_stable') is False for e in geometry)):
        details = geometry
    if not details and str(error) == 'HIF selection geometry invalid' and len(geometry) == 1:
        unresolved = geometry[0]
        frame_ids = unresolved.get('frame_ids', ())
        if (unresolved.get('phase') == 'inventory_live_geometry_unresolved'
                and unresolved.get('geometry_valid') is False
                and unresolved.get('click_dispatched') is False
                and isinstance(frame_ids, (tuple, list)) and bool(frame_ids)
                and all(isinstance(value, str) and value for value in frame_ids)
                and len(set(frame_ids)) == len(frame_ids) == len(frames)):
            details = geometry
    if (not details and geometry and all(e.get('phase') == 'hif_settlement_hand_pending'
            for e in geometry)):
        details = geometry
    # A single marginal pre-input read is retried by the caller against its
    # previous checkpoint. Do not spend a second retry budget here.
    if not details and len(frames) > 1:
        details = _weak_identity_retry_evidence(error, frame_count=len(frames))
    return details


def _weak_identity_retry_evidence(error, *, frame_count):
    geometry = getattr(error, 'events', ())
    if (str(error) == 'inventory navigation Live visual family is unresolved'
            and len(geometry) == 1):
        failed = geometry[0]
        ids, rows = failed.get('frame_ids', ()), failed.get('predictions', ())
        # A marginal detector frame is not a conflicting accepted identity.
        # Spend the existing fresh-set budget; never accept its rejected family.
        if (failed.get('phase') == 'inventory_live_identity'
                and len(ids) == frame_count and len(set(ids)) == frame_count
                and all(isinstance(fid, str) and fid for fid in ids)
                and failed.get('frame_id') in ids and 1 <= len(rows) <= 5
                and [row.get('slot') for row in rows] == list(range(len(rows)))
                and any(row.get('accepted') is False for row in rows)
                and all(row.get('accepted') is True or (
                    row.get('accepted') is False
                    and row.get('reason') in ('below_margin_threshold', 'below_acceptance_threshold')
                    and row.get('recognizer') == 'card_art_embedding_top5_hif_masked_v1'
                    and type(row.get('confidence')) in (float, int)
                    and .70 <= row['confidence'] <= 1
                    and type(row.get('margin')) in (float, int)
                    and (0 <= row['margin'] < .05 if row['reason'] == 'below_margin_threshold'
                         else .05 <= row['margin'] <= 1)
                    and row.get('top_k_card_ids')) for row in rows)):
            return geometry
    return ()


def fresh_identity_reobserve_allowed(error, before):
    """A weak single frame can request a new full group, never supply identity."""
    evidence = _weak_identity_retry_evidence(error, frame_count=1)
    if not evidence or not isinstance(before, dict):
        return False
    frames, fields, families = (before.get(key, ()) for key in ('frame_ids', 'fields', 'families'))
    rows = evidence[0]['predictions']
    return (isinstance(frames, (tuple, list)) and bool(frames) and len(set(frames)) == len(frames)
        and all(isinstance(fid, str) and fid for fid in frames)
        and evidence[0]['frame_id'] not in frames
        and isinstance(fields, (tuple, list)) and len(fields) == 6 and all(value is not None for value in fields)
        and isinstance(families, (tuple, list)) and len(families) == len(rows)
        and all(isinstance(family, (tuple, list)) and family
            and tuple((row.get('top_k_card_ids') or [()])[0]) == tuple(family)
            for row, family in zip(rows, families, strict=True)))


def capture_initial_live_checkpoint(capture_frames, checkpoint, event, *, max_attempts=3, retry_delay=.08,
                                    poll_progress=None):
    """Read whole fresh groups. The two automatic pre-open callers share their host's clock."""
    complete = getattr(checkpoint, '__dict__', {}).get('capture_complete') if max_attempts == 3 else None
    if complete is not None and poll_progress is None:
        return complete(capture_frames, event)

    def fresh_group():
        if poll_progress is not None:
            poll_progress()
        return capture_frames()

    for attempt in (range(max_attempts) if poll_progress is None else count()):
        if complete is None:
            frames = fresh_group()
        try:
            if complete is not None:
                return complete(fresh_group, event)
            return frames, checkpoint(frames)
        except ValueError as error:
            if complete is not None:
                # The scoped reader logs the exact rejected group itself.
                last = (event.get('inspected_live_observations') or [{}])[-1]
                if not last or tuple(last.get('detail', ())) != tuple(getattr(error, 'events', ())):
                    raise
                frames = last.get('frame_ids', ())
            details = initial_checkpoint_retry_evidence(error, frames)
            if not details:
                raise
            if complete is None:
                event.setdefault('initial_observations', []).append(details)
            if poll_progress is None and attempt == max_attempts - 1:
                raise
            time.sleep(retry_delay)


@dataclass(frozen=True)
class InventoryIdentity:
    card_id: str | None
    visual_group: str | None
    reason: str
    reference_error: float | None = None
    reference_margin: float | None = None
    marker_errors: tuple[float, float] | None = None
    embedding_similarity: float | None = None
    embedding_margin: float | None = None
    reference_evidence: dict | None = None


INVENTORY_ZONES = ("手札", "山札", "捨札", "除外")
# Shared by inline detail parsing and the selected-page check before scrolling.
INVENTORY_DETAIL_TITLE_ROI = (185, 105, 425, 45)


class InventoryObservationError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def inventory_identity_reobserve_allowed(error):
    """Reobserve isolated failures or agreeing weak retrieval, never vote IDs."""
    if str(error) != 'inventory page contains an unresolved card':
        return False
    slots = getattr(error,'evidence',{}).get('slots',())
    unresolved = [slot for slot in slots if slot.get('card_id') is None]
    if not unresolved:
        return False
    def weak_candidate(row):
        if row.get('reason') == 'inventory_embedding_low_confidence':
            return True
        proof = row.get('reference_evidence') or {}
        similarity, margin = row.get('embedding_similarity'), row.get('embedding_margin')
        return (row.get('reason') == 'inventory_ambiguous_reference_with_embedding'
            and proof.get('status') == 'MEASURED_CANDIDATES'
            and proof.get('top_group') == row.get('visual_group')
            and type(similarity) in (int, float) and .85 <= similarity <= 1
            and type(margin) in (int, float) and .05 <= margin <= 2)
    # Both independent recognizers can name the same family while the neural
    # score is weak during scrolling. Re-sample within the existing page
    # budget; no weak result supplies an accepted identity or an input target.
    weak_only = True
    for slot in unresolved:
        rows = slot.get('observations', ())
        groups = {row.get('visual_group') for row in rows}
        if (not rows or not any(row.get('card_id') is None for row in rows)
                or len(groups) != 1 or None in groups
                or len({row['card_id'] for row in rows if row.get('card_id') is not None}) > 1
                or any(not weak_candidate(row) for row in rows if row.get('card_id') is None)):
            weak_only = False
            break
    if weak_only:
        return True
    failed_frames = set()
    for slot in unresolved:
        observations = slot.get('observations',())
        accepted = [row for row in observations if row.get('card_id') is not None]
        rejected = [row for row in observations if row.get('card_id') is None]
        if (len(observations) != 3 or len(accepted) != 2 or len(rejected) != 1
                or len({row['card_id'] for row in accepted}) != 1
                or len({row.get('visual_group') for row in accepted}) != 1
                or accepted[0].get('visual_group') is None
                or rejected[0].get('visual_group') not in (None,accepted[0]['visual_group'])
                or rejected[0].get('reason') not in (
                    'inventory_embedding_disagreement','inventory_embedding_low_confidence',
                    'inventory_embedding_low_margin','reference_identity_unresolved')):
            return False
        failed_frames.add(next(index for index,row in enumerate(observations) if row.get('card_id') is None))
    return len(failed_frames)==1


def inventory_known_count_reobserve_allowed(error, sections):
    """Reread a consistent weak header; a proposed count is never admitted."""
    if str(error) != 'inventory section count is unresolved':
        return False
    evidence = getattr(error,'evidence',{})
    name = evidence.get('expected')
    if name not in {'手札', '山札', '捨札', '除外'}:
        return False
    direct = evidence.get('direct_text')
    if not isinstance(direct, str):
        return False
    proposed = re.sub(r'\s+', '', direct)
    if re.fullmatch(re.escape(name)+r'\(\d{1,3}\)', proposed) is None:
        return _segmented_count_reobserve_allowed(evidence, sections)
    expected = f'{name}({sections[name].count})' if name in sections else proposed
    rows = [dict(text=evidence.get('direct_text'),score=evidence.get('direct_score')),
            *evidence.get('reread',()), *evidence.get('cleaned',())]
    return len(rows) >= 2 and all(isinstance(row.get('text'),str)
        and re.sub(r'\s+','',row['text']) == expected
        and type(row.get('score')) in (int,float) and .85 <= row['score'] < .95 for row in rows)


def inventory_selected_title_reobserve_allowed(error):
    """A complete weak selected title may request pixels, never authorize input."""
    return (isinstance(error, InventoryObservationError)
        and str(error) == 'inventory selected-card page is unresolved'
        and error.evidence.get('weak_complete_selected_title') is True)


def _segmented_count_reobserve_allowed(evidence, sections):
    """Strong separated spans only request a fresh page, never supply its count."""
    name = evidence['expected']
    reread, cleaned, parts = (evidence.get(key, ()) for key in ('reread', 'cleaned', 'parts'))
    if len(reread) != 1 or len(cleaned) != 1 or len(parts) != 2 or any(len(group) != 1 for group in parts):
        return False
    direct = dict(text=evidence['direct_text'], score=evidence.get('direct_score'))
    rows = (direct, reread[0], cleaned[0], parts[0][0], parts[1][0])
    if any(not isinstance(row.get('text'), str) or type(row.get('score')) not in (int, float)
           or not math.isfinite(row['score']) or not .85 <= row['score'] <= 1 for row in rows):
        return False
    text = [re.sub(r'\s+', '', row['text']) for row in rows]
    if (text[0] != text[1] or text[3] != name or not .85 <= cleaned[0]['score'] < .95
            or any(group[0]['score'] < .95 for group in parts)
            or re.fullmatch(r'\(\d{1,3}\)', text[4]) is None):
        return False
    joined = text[3]+text[4]
    count = int(text[4][1:-1])
    # The existing white-text path isolates trailing stage digits. A different
    # complete parenthesized count, unknown suffix, or known count conflict stops.
    return (count <= 100 and text[2] == joined
            and re.fullmatch(re.escape(joined)+r'\d+', text[0]) is not None
            and (name not in sections or sections[name].count == count))


@dataclass(frozen=True)
class InventorySection:
    name: str
    count: int
    box: tuple[int, int, int, int]


def inventory_white_text(crop):
    """Separate white section lettering from its animated translucent backing."""
    if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8 or not crop.size:
        raise ValueError('inventory white text crop is invalid')
    text = 255-(crop.min(axis=2)>=220).astype(np.uint8)*255
    text = cv2.copyMakeBorder(text,2,2,2,2,cv2.BORDER_CONSTANT,value=255)
    return cv2.cvtColor(text,cv2.COLOR_GRAY2BGR)


def inventory_header_parts(crop):
    """Split two-character zone names from counts at a unique visible ink gap."""
    ink = inventory_white_text(crop)[2:-2,2:-2,0] == 0
    ys, xs = np.nonzero(ink)
    if not len(xs):
        return ()
    height = int(ys.max()-ys.min()+1)
    if not 22 <= height <= 31:
        return ()
    columns = np.flatnonzero(ink.any(axis=0))
    gaps = [(int(a),int(b)) for a,b in zip(columns,columns[1:])
            if b-a >= 6 and 1.6*height <= (a+b)/2-columns[0] <= 2.5*height]
    if len(gaps) != 1:
        return ()
    split = sum(gaps[0])//2
    return crop[:,:split].copy(), crop[:,split:].copy()


def _inventory_header_geometry(frame, box):
    """Use lettering ink, rather than variable OCR padding, as the baseline."""
    x,y,w,h = box
    white = frame[y:y+h,max(60,x):x+w].min(axis=2)>=220
    rows = np.flatnonzero(white.sum(axis=1)>=4)
    if not len(rows) or not 22 <= rows[-1]-rows[0]+1 <= 31:
        raise ValueError('inventory header ink geometry is unresolved')
    # Four pixels of standard baseline padding retain the grid's existing
    # 17-pixel card gap. OCR bounding-box padding is not a layout coordinate.
    return (x,y+int(rows[0]),w,int(rows[-1]-rows[0]+1)+4)


def verify_inventory_heading(context, frame, rows):
    """Use the same complete heading proof at arrival, reading and closing."""
    if tuple(frame.shape) != (1280, 720, 3):
        raise ValueError("inventory section resolution is not calibrated")
    try:
        _unique_inventory_text(rows, "所持スキルカード", (15, 25, 285, 65))
    except ValueError:
        if any(row.text == '所持スキルカード' for row in rows):
            raise
        # Whole-screen detection can split/misread the final kana. Read the
        # complete fixed heading line directly; do not complete a prefix.
        heading = _inventory_ocr(context, frame[34:76,55:279].copy(),
                                 'LiveInventoryHeadingLine', only_rec=True)
        if (len(heading) != 1 or heading[0].text.replace(' ', '') != '所持スキルカード'
                or not .95 <= heading[0].score <= 1):
            raise ValueError('inventory complete heading reread failed') from None


def read_inventory_sections(context, frame, *, headers_required=True, selected_card_title=None,
                            require_unselected_panel=True):
    """Read visible section headers and total; card text is not a header.

    After an inline detail has been identified, scrolling the thumbnail list
    does not consume its title or change the previously observed modification.
    """
    if tuple(frame.shape) != (1280, 720, 3):
        raise ValueError("inventory section resolution is not calibrated")
    rows = _inventory_ocr(context, frame, "LiveInventorySections")
    verify_inventory_heading(context, frame, rows)
    _unique_inventory_text(rows, "閉じる", (210, 1120, 290, 85))
    if selected_card_title is None and require_unselected_panel:
        _unique_inventory_text(rows, "スキルカードが選択されていません", (20, 100, 680, 290))
    elif selected_card_title is not None:
        if not isinstance(selected_card_title, str) or not selected_card_title.strip():
            raise ValueError('inventory selected-card title is invalid')
        # A verified inline detail remains visible while scrolling. Require
        # that exact title or an explicit cleared selection, never any title.
        anchors = []
        for text, roi in ((re.escape(selected_card_title), INVENTORY_DETAIL_TITLE_ROI),
                          ('スキルカードが選択されていません', (20,100,680,290))):
            try:
                anchors.append(_unique_inventory_text(rows,text,roi,threshold=.95))
            except ValueError:
                pass
        if len(anchors) != 1:
            try:
                weak = _unique_inventory_text(rows,re.escape(selected_card_title),
                    INVENTORY_DETAIL_TITLE_ROI,threshold=.85)
            except ValueError:
                weak = None
            raise InventoryObservationError('inventory selected-card page is unresolved',
                dict(weak_complete_selected_title=weak is not None and weak.score < .95))

    def normalized(text):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))

    total_rows = [row for row in rows if len(row.box) == 4
                  and 45 <= row.box[0] <= 130 and 1020 <= row.box[1] <= 1070
                  and normalized(row.text).startswith("スキルカード")]
    if len(total_rows) != 1:
        raise ValueError("inventory total count is missing or ambiguous")

    def count_text(row, expected):
        x, y, w, h = row.box
        if min(w,h) <= 0 or min(x,y) < 0 or x+w+4 > 720 or y+h > 1280:
            raise ValueError("inventory count box is invalid")
        direct = re.fullmatch(expected+r"\((\d{1,3})\)",normalized(row.text))
        if (direct and type(row.score) in {int,float} and math.isfinite(row.score)
                and row.score >= .95 and int(direct[1]) <= 100):
            return int(direct[1])
        left = 55 if expected in INVENTORY_ZONES else x
        crop = frame[y:y+h,left:x+w+4].copy()
        values = _inventory_ocr(context, crop,
                                "LiveInventorySectionCount", only_rec=True)
        valid = [normalized(value.text) for value in values
                 if type(value.score) in {int,float} and math.isfinite(value.score) and value.score >= .95]
        cleaned = []
        parts = []
        if (expected in INVENTORY_ZONES and direct is None and len(values) == 1
                and re.fullmatch(expected+r"\((\d{1,3})\)", normalized(values[0].text)) is None):
            # A translucent header can include stage-score digits. Independently
            # read the complete white lettering and both ink-separated spans;
            # never trim OCR text or replace an already valid count.
            cleaned = _inventory_ocr(context, inventory_white_text(crop),
                                     'LiveInventorySectionWhiteText', only_rec=True)
            for index, part in enumerate(inventory_header_parts(crop)):
                parts.append(_inventory_ocr(context, inventory_white_text(part),
                    f'LiveInventorySectionPart{index}', only_rec=True))
            # The name and complete parenthesized number are the two
            # independently read spans. An agreeing whole line is supporting
            # evidence; its CTC boundary score must not veto both strong spans
            # and repeatedly request the same page. Never trim trailing digits
            # from the original OCR or substitute a previously known count.
            groups = [(cleaned, .85), *((part, .95) for part in parts)]
            if (len(parts) == 2 and all(len(group) == 1
                    and type(group[0].score) in {int, float}
                    and math.isfinite(group[0].score) and threshold <= group[0].score <= 1
                    for group, threshold in groups)):
                joined = ''.join(normalized(group[0].text) for group in parts)
                if (normalized(parts[0][0].text) == expected
                        and re.fullmatch(r'\(\d{1,3}\)', normalized(parts[1][0].text))
                        and joined == normalized(cleaned[0].text)):
                    valid = [joined]
        if (not valid and expected in INVENTORY_ZONES and len(values)==1
                and type(values[0].score) in {int,float} and math.isfinite(values[0].score)
                and .85 <= values[0].score < .95
                and re.fullmatch(expected+r"\((\d{1,3})\)",normalized(values[0].text))):
            # As with status-panel ink normalization, permit one deterministic
            # reread of the same low-confidence, complete text. It must agree
            # exactly; this cannot select a different number or header.
            cleaned = _inventory_ocr(context,inventory_white_text(crop),
                                     "LiveInventorySectionWhiteText",only_rec=True)
            valid = [normalized(value.text) for value in cleaned
                     if type(value.score) in {int,float} and math.isfinite(value.score)
                     and value.score >= .95 and normalized(value.text)==normalized(values[0].text)]
            # CTC whole-line confidence can be dominated by the transition
            # between Japanese lettering and narrow parentheses. Independently
            # read both complete spans, retaining the original literal count.
            if (not valid and normalized(row.text)==normalized(values[0].text)
                    and all(normalized(v.text)==normalized(values[0].text) for v in cleaned)):
                crops = inventory_header_parts(crop)
                for index, part in enumerate(crops):
                    parts.append(_inventory_ocr(context,part,
                        f'LiveInventorySectionPart{index}',only_rec=True))
                if (len(parts)==2 and all(len(group)==1
                        and type(group[0].score) in {int,float}
                        and math.isfinite(group[0].score) and group[0].score>=.95 for group in parts)):
                    joined = ''.join(normalized(group[0].text) for group in parts)
                    if (normalized(parts[0][0].text)==expected
                            and re.fullmatch(r'\(\d{1,3}\)',normalized(parts[1][0].text))
                            and joined==normalized(values[0].text)):
                        valid = [joined]
        if len(valid) != 1 or not (match := re.fullmatch(expected+r"\((\d{1,3})\)",valid[0])):
            raise InventoryObservationError("inventory section count is unresolved",{
                'expected':expected,'box':tuple(row.box),'direct_text':row.text,'direct_score':row.score,
                'reread':[{'text':value.text,'score':value.score} for value in values],
                'cleaned':[{'text':value.text,'score':value.score} for value in cleaned],
                'parts':[[{'text':value.text,'score':value.score} for value in group] for group in parts]})
        count = int(match[1])
        if count > 100:
            raise ValueError("inventory count exceeds the supported bound")
        return count

    total = count_text(total_rows[0], "スキルカード")
    if not headers_required:
        return total, ()
    sections = []
    header_rows = _inventory_ocr(context, frame, "LiveInventoryZoneHeaders", roi=[55,395,155,610])
    for row in header_rows:
        x,y,w,h = row.box
        # A detection touching the crop's upper edge may be a clipped heading.
        # Previously established sections remain positioned by card overlap;
        # this fragment must not introduce or revise a section count.
        if not (55 <= x <= 70 and 395 < y < 965 and 22 <= h <= 43):
            continue
        text = normalized(row.text).lstrip("✦✧★☆")
        names = [name for name in INVENTORY_ZONES if text.startswith(name)]
        if not names:
            continue
        name, = names
        sections.append(InventorySection(name,count_text(row,name),
                                         _inventory_header_geometry(frame,tuple(row.box))))
    sections.sort(key=lambda section: section.box[1])
    names = [section.name for section in sections]
    if len(names) != len(set(names)) or names != sorted(names,key=INVENTORY_ZONES.index):
        raise ValueError("inventory section order is contradictory")
    return total, tuple(sections)


def _inventory_visible_boxes(sections, displacement):
    boxes = {}
    for name, section in sections.items():
        first_top = section.box[1] + section.box[3] + 17 - displacement
        for index in range(section.count):
            x = (48,208,368,529)[index % 4]
            y = first_top + index // 4 * 154
            if 406 <= y and y+142 <= 1007 and not (x+142 > 606 and y+142 > 947):
                boxes[(name,index)] = (x,y,142,142)
    return boxes


def _align_inventory_physical_rows(frames, boxes):
    """Recover a rasterized tile row from its own complete thin borders.

    Header OCR only proposes the row. Two separate 142px tiles must show the
    calibrated inset border on both horizontal and vertical sides; all three
    frames must give the same unique center. No recognition score chooses it.
    """
    if not frames or any(frame.shape != (1280,720,3) for frame in frames):
        return boxes
    grouped = {}
    for key, (x,y,w,h) in boxes.items():
        if (w,h) == (142,142):
            grouped.setdefault(y, []).append((key,x))
    edges = [cv2.Canny(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),80,180) for frame in frames]
    result = dict(boxes)
    for y, row in grouped.items():
        if len(row) < 2:
            continue
        centers = []
        for edge in edges:
            proposals = {}
            for key,x in row:
                for top in range(max(414,y-2),min(999-138,y+8)+1):
                    bottom = top+138  # 142px crop, two-pixel inset on each side.
                    if (float(np.mean(edge[top,x+12:x+130] > 0)) < .95
                            or float(np.mean(edge[bottom,x+12:x+130] > 0)) < .95
                            or float(np.mean(np.any(edge[top+8:bottom-8,x-3:x+3] > 0,axis=1))) < .95
                            or float(np.mean(np.any(edge[top+8:bottom-8,x+137:x+144] > 0,axis=1))) < .95):
                        continue
                    proposals.setdefault(top-2,set()).add(key)
            if len(proposals) != 1 or len(next(iter(proposals.values()))) < 2:
                break
            centers.append(next(iter(proposals)))
        if len(centers) != len(frames) or len(set(centers)) != 1 or abs(centers[0]-y) > 4:
            continue
        corrected = centers[0]
        if any(corrected < 406 or corrected+142 > 1007
                or x+142 > 606 and corrected+142 > 947 for _,x in row):
            continue
        for key,x in row:
            result[key] = (x,corrected,142,142)
    return result


def inventory_scroll_distance(previous_boxes):
    """Keep two artwork columns visible across sparse section boundaries."""
    limits = {}
    for x, y, _w, h in previous_boxes.values():
        limits[x] = max(limits.get(x, 0), y+h-8-414-48)
    # Native 480px inertia moves content roughly 1.5--1.8 times the swipe.
    # The shorter gesture retains overlapping pixels; actual movement still
    # requires the unchanged two-column registration after the gesture.
    return 40 if len(limits) >= 2 and sorted(limits.values())[-2] < 160 else 80


def inventory_scroll_displacement(previous, current, previous_boxes, *, minimum=-2, maximum=450,
                                  reference_sections=(), current_sections=()):
    """Measure one unique upward content shift from opaque thumbnail pixels.

    Every candidate must explain all previously visible thumbnails that remain
    fully visible. Duplicate pictures preserve ambiguity instead of collapsing
    physical card instances. The animated stage is excluded from each patch.
    """
    if tuple(previous.shape) != (1280,720,3) or current.shape != previous.shape:
        raise ValueError("inventory scroll frames are incompatible")
    header_shifts = []
    header_pairs = []
    for old in reference_sections:
        matches = [new for new in current_sections if new.name == old.name]
        if len(matches) != 1:
            continue
        new = matches[0]
        if new.count != old.count:
            raise ValueError("inventory section count changed during registration")
        header_shifts.append(old.box[1]+old.box[3]-new.box[1]-new.box[3])
        header_pairs.append((old,new))
    independent_headers = (len(header_shifts) >= 2
        and len({s.name for s in reference_sections}) == len(reference_sections)
        and len({s.name for s in current_sections}) == len(current_sections)
        and max(header_shifts)-min(header_shifts) <= 2)
    # An empty section can leave exactly one complete thumbnail between two
    # independently read headings. These three anchors still constrain one
    # translation; headings alone never generate a candidate displacement.
    complete_header_pairs = independent_headers and all(
        old.name in INVENTORY_ZONES
        and all(55 <= s.box[0] <= 70 and 395 < s.box[1] < 965
                and 22 <= s.box[3] <= 43 and s.box[0]+s.box[2] <= 210
                and s.box[1]+s.box[3] <= 1005 for s in (old,new))
        and all(abs(old.box[i]-new.box[i]) <= 2 for i in (0,2,3))
        for old,new in header_pairs)
    # The official 480->720 resize can move the vertical sampling phase by
    # half a pixel after scrolling. Apply the same small vertical low-pass to
    # both images for registration only. Card identities use original pixels.
    previous = cv2.GaussianBlur(previous,(1,5),1.)
    current = cv2.GaussianBlur(current,(1,5),1.)
    proposals = set()
    patches = []
    for box in previous_boxes.values():
        x,y,w,h = box
        patch = previous[y+h-56:y+h-8,x+8:x+w-8]
        if float(patch.std()) < 10:
            raise ValueError("inventory scroll anchor lacks visible artwork")
        area = current[414:999,x+8:x+w-8]
        scores = cv2.matchTemplate(area,patch,cv2.TM_CCOEFF_NORMED)[:,0]
        deltas = {y+h-56-(414+int(index)) for index in np.flatnonzero(scores >= .97)}
        deltas = {value for value in deltas if minimum <= value <= maximum}
        proposals.update(deltas)
        patches.append(box)
    accepted = []
    for delta in sorted(proposals):
        votes, columns = 0,set()
        good = True
        for x,y,w,h in patches:
            target = y-delta
            top = max(414,target+8)
            bottom = min(999,target+h-8,939 if x+w>606 else 999)
            if bottom-top < 48:
                continue
            patch = previous[top+delta:bottom+delta,x+8:x+w-8]
            measured = current[top:bottom,x+8:x+w-8]
            if float(cv2.matchTemplate(measured,patch,cv2.TM_CCOEFF_NORMED)[0,0]) < .97:
                good = False
                break
            votes += 1
            columns.add(x)
        header_agreement = independent_headers and all(abs(delta-shift) <= 2 for shift in header_shifts)
        single_card_anchors = (len(patches) == votes == 1 and complete_header_pairs
            and header_agreement and all(
                406 <= y and y+h <= 1007 and 406 <= y-delta and y-delta+h <= 1007
                and not (x+w > 606 and max(y,y-delta)+h > 947)
                for x,y,w,h in patches))
        if good and (votes >= 2 and (len(columns) >= 2 or header_agreement)
                     or single_card_anchors):
            accepted.append(delta)
    # Adjacent rasterization offsets can describe the same translation; a
    # second disjoint interval is a genuinely ambiguous physical row.
    if not accepted or max(accepted)-min(accepted) > 2:
        raise ValueError("inventory scroll has no unique overlapping translation")
    return accepted[len(accepted)//2]


def corroborate_inventory_marker(reader, frames, box, original, expected_id, *, plan, reason):
    """Test an independently identified card against the current artwork and upgrade marker."""
    if not frames or len(original) != len(frames) or len({row.visual_group for row in original}) != 1:
        return ()
    group = original[0].visual_group
    members = {reader.recognizer.gallery.card_ids[i] for i in reader.groups.get(group, ())}
    if (expected_id not in members or any(row.card_id is not None
            or row.reason not in ('inventory_embedding_low_confidence', 'inventory_embedding_low_margin')
            or any(type(value) not in (int,float) or not math.isfinite(value) for value in (
                row.reference_error,row.reference_margin,row.embedding_similarity,row.embedding_margin))
            or not 0 <= row.reference_error <= 20 or not 0 < row.reference_margin < 3
            or not 0 <= row.embedding_similarity < .85
            or not (row.embedding_margin >= .05 if row.reason == 'inventory_embedding_low_confidence'
                    else 0 < row.embedding_margin < .05)
            for row in original)):
        return ()
    resolved = tuple(reader._resolve_group_identity(np.asarray(frame), box, plan=plan,
        group=group, error=row.reference_error, margin=row.reference_margin,
        similarity=row.embedding_similarity, embedding_margin=row.embedding_margin,
        corroborated=True) for frame, row in zip(frames, original, strict=True))
    if any(row.card_id != expected_id or row.marker_errors is None
           or row.reason != 'accepted_with_live_hand_family_and_marker' for row in resolved):
        return ()
    return tuple(replace(row, reason=reason) for row in resolved)


class InventoryZoneCollector:
    """Accumulate indexed thumbnails with a fixed count and observed scrolling.

    All four displayed zones and every declared slot must be observed before
    returning a composition. This is not a complete high-score state producer.
    """

    def __init__(self, expected_hand_count, *, live_hand_evidence=None):
        if type(expected_hand_count) is not int or not 0 <= expected_hand_count <= 5:
            raise ValueError("inventory collector needs the observed Live hand count")
        self.expected_hand_count = expected_hand_count
        self.total = None
        self.sections = {}
        self.identities = {}
        self.displacement = 0
        self.previous = None
        self.previous_boxes = {}
        self.previous_sections = ()
        self.previous_independent_ids = {}
        self.previous_frame_ids = ()
        self.independent_pages = []
        self.page_count = 0
        self.selected_card_title = None
        self.live_hand_evidence = deepcopy(live_hand_evidence)

    def _current_live_hand_evidence(self, context, frames):
        evidence = self.live_hand_evidence
        if evidence is None:
            return None
        run = getattr(context, 'gkh_hif_live_context', None)
        old_ids = evidence.get('frame_ids', ()) if isinstance(evidence, dict) else ()
        new_ids = tuple(frame_identifier(frame) for frame in frames)
        families = evidence.get('families', ()) if isinstance(evidence, dict) else ()
        if (not isinstance(evidence, dict) or evidence.get('source') != 'inventory_navigation_before'
                or not evidence.get('run_id') or type(evidence.get('revision')) is not int
                or (evidence['run_id'], evidence['revision']) !=
                    (getattr(run, 'run_id', None), getattr(run, 'revision', None))
                or not old_ids or len(set(old_ids)) != len(old_ids)
                or any(not isinstance(value, str) or not value for value in old_ids)
                or not new_ids or len(set(new_ids)) != len(new_ids) or set(old_ids) & set(new_ids)
                or len(families) != self.expected_hand_count
                or any(not family or len(set(family)) != len(family)
                       or any(not isinstance(cid, str) or not cid.isdecimal() for cid in family)
                       for family in families)):
            raise ValueError('inventory Live hand evidence is stale or incomplete')
        return {**evidence, 'inventory_frame_ids': new_ids}

    def _corroborate_live_hand(self, frames, reader, boxes, batch, *, plan, evidence):
        """Use a current complete Live hand only for one weak thumbnail family.

        It supplies independent family evidence, never a model-predicted ID.
        The original thumbnail scores remain recorded, and normal/+ still
        requires the unchanged marker gates on all three inventory frames.
        """
        if evidence is None or len(batch) != len(boxes):
            return batch, None
        keys = tuple(boxes)
        unresolved = [i for i, row in enumerate(batch) if row.get('card_id') is None]
        hand_indexes = [i for i, key in enumerate(keys) if key[0] == '手札']
        if (len(unresolved) != 1 or unresolved[0] not in hand_indexes
                or [keys[i][1] for i in hand_indexes] != list(range(self.expected_hand_count))):
            return batch, None
        target = unresolved[0]
        groups, families = [], []
        for index in hand_indexes:
            observations = batch[index].get('observations', ())
            if len(observations) != len(frames) or len({row.visual_group for row in observations}) != 1:
                return batch, None
            group = observations[0].visual_group
            members = tuple(sorted({reader.recognizer.gallery.card_ids[i]
                for i in reader.groups.get(group, ())}, key=int))
            if not members or any(row.card_id != batch[index].get('card_id') for row in observations):
                return batch, None
            if index != target and batch[index]['card_id'] not in members:
                return batch, None
            groups.append(group)
            families.append(members)
        live_families = tuple(tuple(sorted(family, key=int)) for family in evidence['families'])
        target_family = families[hand_indexes.index(target)]
        if (Counter(live_families) != Counter(families)
                or families.count(target_family) != 1 or live_families.count(target_family) != 1):
            return batch, None
        original = batch[target]['observations']
        if any(any(type(value) not in (int, float) or not math.isfinite(value) for value in (
                   row.reference_error, row.reference_margin, row.embedding_similarity, row.embedding_margin))
               or not 0 <= row.reference_error <= 20 or not 0 < row.reference_margin < 3
               or row.embedding_margin < .05
               for row in original):
            return batch, None
        weak_embedding = all(row.reason == 'inventory_embedding_low_confidence'
                             and 0 <= row.embedding_similarity < .85 for row in original)
        ambiguous_reference = all(row.reason == 'inventory_ambiguous_reference_with_embedding'
            and .85 <= row.embedding_similarity <= 1
            and isinstance(row.reference_evidence,dict)
            and row.reference_evidence.get('status') == 'MEASURED_CANDIDATES'
            for row in original)
        if not (weak_embedding or ambiguous_reference):
            return batch, None
        box = boxes[keys[target]]
        resolved = tuple(replace(reader._resolve_group_identity(np.asarray(frame), box, plan=plan,
            group=row.visual_group, error=row.reference_error, margin=row.reference_margin,
            similarity=row.embedding_similarity, embedding_margin=row.embedding_margin,
            corroborated=True), reference_evidence=deepcopy(row.reference_evidence))
            for frame, row in zip(frames, original, strict=True))
        ids = {row.card_id for row in resolved}
        if (len(ids) != 1 or None in ids or any(row.marker_errors is None
                or row.reason != 'accepted_with_live_hand_family_and_marker' for row in resolved)):
            return batch, None
        updated = list(batch)
        updated[target] = {**batch[target], 'card_id': resolved[0].card_id, 'observations': resolved}
        return updated, dict(source='current_live_family_and_inventory_marker',
            run_id=evidence['run_id'], revision=evidence['revision'],
            live_frame_ids=tuple(evidence['frame_ids']), inventory_frame_ids=evidence['inventory_frame_ids'],
            live_families=live_families, inventory_families=tuple(families),
            zone=keys[target][0], index=keys[target][1], box=box, card_id=resolved[0].card_id,
            visual_group=groups[hand_indexes.index(target)],
            original_observations=tuple(asdict(row) for row in original),
            resolved_observations=tuple(asdict(row) for row in resolved), input_authorized=False)

    def _corroborate_registered_overlap(self, frames, reader, boxes, batch, *, plan, shifts, evidence,
                                       source=None, target_index=None):
        """Reconfirm one weak family on an independently registered old slot."""
        unresolved = [i for i, row in enumerate(batch) if row.get('card_id') is None]
        if source is None:
            source = dict(frame=self.previous, boxes=self.previous_boxes,
                independent_ids=self.previous_independent_ids, frame_ids=self.previous_frame_ids)
        elif evidence is None or (source['run_id'],source['revision']) != (evidence['run_id'],evidence['revision']):
            return batch, None
        if (evidence is None or source['frame'] is None or len(batch) != len(boxes)
                or (len(unresolved) != 1 if target_index is None else target_index not in unresolved)
                or not shifts or len(shifts) != len(frames)
                or max(shifts)-min(shifts) > 1
                or not source['frame_ids']
                or len(set(source['frame_ids'])) != len(source['frame_ids'])
                or set(source['frame_ids']) & set(evidence['inventory_frame_ids'])):
            return batch, None
        index = unresolved[0] if target_index is None else target_index
        key = tuple(boxes)[index]
        old_id = source['independent_ids'].get(key)
        old = source['boxes'].get(key)
        box = boxes[key]
        if (old_id is None or self.identities.get(key) != old_id or old is None
                or (old[0],old[2],old[3]) != (box[0],box[2],box[3])):
            return batch, None
        x,y,w,h = old
        # The existing registration checks every participating patch at .97.
        # Require this target to have participated in all three comparisons,
        # with its complete thumbnail visible at both ends of the translation.
        for delta in shifts:
            target = y-delta
            top = max(414,target+8)
            bottom = min(999,target+h-8,939 if x+w>606 else 999)
            if (not (406 <= y and y+h <= 1007 and 406 <= target and target+h <= 1007)
                    or (x+w>606 and max(y,target)+h>947)
                    or bottom-top < 48 or abs(target-box[1]) > 1):
                return batch, None
        original = batch[index].get('observations', ())
        resolved = corroborate_inventory_marker(reader, frames, box, original, old_id,
            plan=plan, reason='accepted_with_registered_overlap_and_marker')
        if not resolved:
            return batch, None
        updated = list(batch)
        updated[index] = {**batch[index], 'card_id':old_id, 'observations':resolved}
        return updated, dict(source='same_transaction_registered_overlap_and_current_marker',
            run_id=evidence['run_id'],revision=evidence['revision'],
            previous_frame_ids=source['frame_ids'],inventory_frame_ids=evidence['inventory_frame_ids'],
            previous_box=old,box=box,shifts=tuple(shifts),zone=key[0],index=key[1],card_id=old_id,
            visual_group=original[0].visual_group,original_observations=tuple(asdict(row) for row in original),
            resolved_observations=tuple(asdict(row) for row in resolved),input_authorized=False)

    def _corroborate_independent_pages(self, frames, reader, boxes, batch, *, plan, evidence, displacement, sections):
        """Register directly to original strong pixels; never chain a corroboration."""
        if evidence is None or len(batch) != len(boxes):
            return batch, None
        unresolved = [i for i,row in enumerate(batch) if row.get('card_id') is None]
        if not unresolved:
            return batch, None
        updated, proofs, registrations = list(batch), [], {}
        for index in unresolved:
            key = tuple(boxes)[index]
            for source in reversed(self.independent_pages):
                if (key not in source['independent_ids']
                        or not -2 <= displacement-source['displacement'] <= 450):
                    continue
                source_key = source['frame_ids']
                if source_key not in registrations:
                    try:
                        direct_shifts = tuple(inventory_scroll_displacement(
                            source['frame'],frame,source['boxes'],
                            reference_sections=source['sections'],current_sections=sections[frame_index][1])
                            for frame_index,frame in enumerate(frames))
                        if (max(direct_shifts)-min(direct_shifts) > 1
                                or any(abs(source['displacement']+delta-displacement) > 1
                                       for delta in direct_shifts)):
                            direct_shifts = None
                    except ValueError:
                        direct_shifts = None
                    registrations[source_key] = direct_shifts
                candidate, proof = self._corroborate_registered_overlap(
                    frames,reader,boxes,updated,plan=plan,shifts=registrations[source_key],
                    evidence=evidence,source=source,target_index=index)
                if proof is not None:
                    updated = candidate
                    proofs.append(dict(proof, independent_source_displacement=source['displacement'],
                        current_displacement=displacement))
                    break
            else:
                return batch, None
        return updated, dict(source='same_transaction_independent_pages_and_current_markers',
            corroborations=tuple(proofs),input_authorized=False)

    def observe(self, context, frames, reader, *, plan, recheck=False):
        if not frames or (not getattr(context, 'gkh_hif_upstream_card_ui', False) and self.page_count >= 12):
            raise ValueError("inventory collector requires a visible page within its observation scope")
        live_hand_evidence = self._current_live_hand_evidence(context, frames)
        positions = []
        if self.previous is None:
            # Later pages prove motion with opaque thumbnail registration;
            # these header positions are only consumed on entry.
            for frame in frames:
                detected = _inventory_ocr(context,frame,'LiveInventoryMotionHeaders',roi=[55,395,155,610])
                positions.append(tuple(sorted((name,row.box[1],row.box[1]+row.box[3])
                    for row in detected for name in INVENTORY_ZONES
                    if row.text.replace(' ','').startswith(name) and 22 <= row.box[3] <= 42)))
        displacement = self.displacement
        shifts = None
        observations = None
        if self.previous is not None and len({box[0] for box in self.previous_boxes.values()}) == 1:
            observations = [read_inventory_sections(context,frame,
                require_unselected_panel=self.selected_card_title is None) for frame in frames]
        if self.previous is not None:
            shifts = [inventory_scroll_displacement(self.previous,frame,self.previous_boxes,
                reference_sections=self.previous_sections,
                current_sections=observations[index][1] if observations is not None else ())
                      for index,frame in enumerate(frames)]
            if max(shifts)-min(shifts)>1:
                sink = getattr(context, 'gkh_hif_alignment_failure_sink', None)
                if callable(sink):
                    sink(self.previous, frames, self.previous_boxes)
                raise ValueError("inventory scroll has not settled across frames: " + repr(shifts))
            displacement += shifts[len(shifts)//2]
        else:
            for other in positions[1:]:
                if (len(other)!=len(positions[0]) or any(a[0]!=b[0] or abs(a[1]-b[1])>2 or abs(a[2]-b[2])>2
                                                       for a,b in zip(positions[0],other))):
                    raise ValueError('inventory header positions have not settled')
        if observations is None:
            observations = [read_inventory_sections(context,frame,
                require_unselected_panel=self.selected_card_title is None) for frame in frames]
        total, sections = observations[0]
        for other_total, other in observations[1:]:
            if (other_total != total or len(other) != len(sections)
                    or any((a.name,a.count) != (b.name,b.count)
                           or (self.previous is None and (abs(a.box[1]-b.box[1])>2
                               or abs(a.box[1]+a.box[3]-b.box[1]-b.box[3])>2))
                           or (self.previous is not None and abs(a.box[1]+a.box[3]-b.box[1]-b.box[3])>3)
                           for a,b in zip(sections,other))):
                raise ValueError("inventory section headers have not settled")
        if self.total is not None and total != self.total:
            raise ValueError("inventory total changed during observation")
        if self.previous is None:
            hand = [section for section in sections if section.name == "手札"]
            if (len(hand) != 1 or not 410 <= hand[0].box[1] <= 445
                    or hand[0].count != self.expected_hand_count):
                raise InventoryObservationError("inventory collection must start at the complete hand header", {
                    'expected_hand_count': self.expected_hand_count,
                    'sections': [asdict(section) for section in sections]})
        merged = dict(self.sections)
        for section in sections:
            x,y,w,h = section.box
            registered = InventorySection(section.name,section.count,(x,y+displacement,w,h))
            if section.name in merged:
                old = merged[section.name]
                if old.count != section.count or abs(old.box[1]+old.box[3]-registered.box[1]-h)>3:
                    raise InventoryObservationError("inventory header contradicts measured scrolling", {
                        'previous':asdict(old),'current':asdict(registered),
                        'displacement':displacement})
            else:
                merged[section.name] = registered
        ordered = sorted(merged.values(),key=lambda section:section.box[1])
        if [s.name for s in ordered] != sorted(merged,key=INVENTORY_ZONES.index):
            raise ValueError("inventory global section order is contradictory")
        if sum(s.count for s in merged.values()) > total:
            raise ValueError("inventory section counts exceed total")
        boxes = _align_inventory_physical_rows(frames, _inventory_visible_boxes(merged,displacement))
        if not boxes:
            raise ValueError("inventory page has no complete thumbnail anchors")
        # Every accepted slot is rechecked in all frames, including overlap;
        # inconsistent identities cannot silently overwrite an earlier page.
        batch = reader.identify_batch(frames,list(boxes.values()),plan=plan)
        independent_ids = {key:row['card_id'] for key,row in zip(boxes,batch)
            if row.get('card_id') and len(row.get('observations',())) == len(frames)
            and all(value.card_id == row['card_id'] for value in row['observations'])}
        batch, corroboration = self._corroborate_live_hand(
            frames, reader, boxes, batch, plan=plan, evidence=live_hand_evidence)
        batch, overlap_corroboration = self._corroborate_registered_overlap(
            frames,reader,boxes,batch,plan=plan,shifts=shifts,evidence=live_hand_evidence)
        if overlap_corroboration is None:
            batch, overlap_corroboration = self._corroborate_independent_pages(
                frames,reader,boxes,batch,plan=plan,evidence=live_hand_evidence,
                displacement=displacement,sections=observations)
        final_card = None
        opening_deck = getattr(context, 'gkh_hif_opening_deck_ids', ())
        if opening_deck and any(not row.get('card_id') for row in batch):
            from .hif_closed_inventory import corroborate_last_inventory_card

            batch, final_card = corroborate_last_inventory_card(
                self.identities, boxes, batch, frames, reader, plan=plan, total=total, ledger_ids=opening_deck)
        if len(batch) != len(boxes) or any(not row.get('card_id') for row in batch):
            raise InventoryObservationError("inventory page contains an unresolved card",{
                'displacement':displacement,'slots':[
                    {'zone':key[0],'index':key[1],'box':box,
                     'card_id':row.get('card_id'),
                     'observations':[asdict(value) for value in row.get('observations',())]}
                    for (key,box),row in zip(boxes.items(),batch)]})
        found = dict(self.identities)
        for key,row in zip(boxes,batch):
            if key in found and found[key] != row['card_id']:
                raise ValueError("inventory overlapping card identity changed")
            found[key] = row['card_id']
        self.total,self.sections,self.identities = total,merged,found
        self.displacement = displacement
        reference = len(frames)//2
        self.previous,self.previous_boxes = frames[reference],boxes
        self.previous_sections = observations[reference][1]
        self.previous_independent_ids = independent_ids
        self.previous_frame_ids = tuple(frame_identifier(frame) for frame in frames)
        if independent_ids and live_hand_evidence is not None:
            self.independent_pages.append(dict(frame=self.previous.copy(), boxes=dict(boxes),
                independent_ids=dict(independent_ids),frame_ids=self.previous_frame_ids,
                sections=self.previous_sections,displacement=displacement,
                run_id=live_hand_evidence['run_id'],revision=live_hand_evidence['revision']))
        self.page_count += 1
        return {'total':total,'displacement':displacement,
                'sections':tuple({'name':s.name,'count':s.count,'box':s.box} for s in sections),
                'slots':tuple((name,index,found[(name,index)],box) for (name,index),box in boxes.items()),
                'observed_slot_count':len(found),'complete':self.complete,
                **({'last_card_corroboration': final_card} if final_card is not None else {}),
                **({'live_hand_corroboration': corroboration} if corroboration is not None else {}),
                **({'overlap_corroboration': overlap_corroboration} if overlap_corroboration is not None else {})}

    @property
    def complete(self):
        return (set(self.sections) == set(INVENTORY_ZONES)
                and sum(s.count for s in self.sections.values()) == self.total
                and all((name,index) in self.identities
                        for name,s in self.sections.items() for index in range(s.count)))

    def composition(self):
        if not self.complete:
            raise ValueError("inventory zones are incomplete")
        return {name:tuple(self.identities[(name,index)] for index in range(self.sections[name].count))
                for name in INVENTORY_ZONES}


def match_inventory_hand(families, inventory_ids):
    """Map an unordered thumbnail hand to slots only when the assignment is unique.

    Equal copies are interchangeable; a normal/+ pair in the same visual
    family is not. Known slot identities can be supplied as singleton families.
    """
    if not 0 <= len(families) == len(inventory_ids) <= 5:
        raise ValueError("inventory and Live hand counts disagree")
    if any(not isinstance(cid, str) or not cid.isdecimal() for cid in inventory_ids):
        raise ValueError("inventory hand contains an unresolved identity")
    remaining = Counter(inventory_ids)
    solutions = set()

    def visit(prefix):
        if len(solutions) > 1:
            return
        if len(prefix) == len(families):
            solutions.add(prefix)
            return
        for cid in set(families[len(prefix)]):
            if remaining[cid] > 0:
                remaining[cid] -= 1
                visit((*prefix, cid))
                remaining[cid] += 1

    visit(())
    if len(solutions) != 1:
        raise ValueError("inventory hand has no unique Live slot assignment")
    return next(iter(solutions))


def _inventory_button_profile(client_size):
    # The 360x640 capture retains the sharp white bag strokes. The adjacent
    # 359x638 capture matches the softer small-client template.
    if client_size == (360, 640):
        return (720, 1280)
    return calibrated_live_client_size(client_size)


def inventory_button_image(frame, client_size=(720, 1280)):
    """Retain the white bag glyph; translucent stage colors are not the icon."""
    if tuple(frame.shape) != (1280, 720, 3) or frame.dtype != np.uint8:
        raise ValueError("inventory button image resolution is not calibrated")
    mask = (frame.min(axis=2) >= 220).astype(np.uint8) * 255
    # Apply the same anti-aliasing treatment as the pinned glyph template.
    # One extra bright raster row at the bag bottom is not a different icon.
    smoothed = cv2.GaussianBlur(mask,(3,3),0)
    if _inventory_button_profile(client_size) == (480, 853):
        smoothed = cv2.GaussianBlur(smoothed,(5,5),0)
    return cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)


def inventory_button_template(client_size=(720, 1280)):
    """Select the glyph raster for the verified native client profile."""
    client_size = _inventory_button_profile(client_size)
    if client_size == (720, 1280):
        return 'produce/hif_live_inventory_stable.png'
    if client_size == (480, 853):
        return 'produce/hif_live_inventory_480.png'
    raise ValueError('inventory button client profile is uncalibrated')


def inventory_button_templates(client_size=(720, 1280)):
    primary = inventory_button_template(client_size)
    # Restarting at native 720 can retain the game's softer glyph raster.
    # Both are existing calibrated assets; the original raster stays valid.
    return ([primary, 'produce/hif_live_inventory_480.png']
            if _inventory_button_profile(client_size) == (720,1280) else
            [primary, 'produce/hif_live_inventory_480_enlarged.png'])


def recognize_inventory_button(context, frame, client_size=(720, 1280)):
    """Match one current frame; an ambiguous primary result never falls back."""
    image = inventory_button_image(frame, client_size)
    templates = inventory_button_templates(client_size)
    node = 'ProduceLiveInventoryButton'
    attempts = [(templates, False), (['produce/hif_live_inventory_480_enlarged.png'], True)]
    if client_size in ((540, 960), (541, 961), (541, 962), (542, 962)):
        # 1306-1308 retain the normal-size glyph with the existing soft raster.
        # Only try this calibrated pairing after both prior paths find nothing.
        attempts.append((['produce/hif_live_inventory_480.png'], False))
        attempts.append((['produce/hif_live_inventory_480_enlarged.png'], True))
    for index, (templates, enlarged) in enumerate(attempts):
        if index == 1:
            # 1172's native 720 frame has the already calibrated larger,
            # softer glyph. Pair the existing asset with its own smoothing;
            # do not change the client profile or the original raster path.
            image = cv2.GaussianBlur(image, (5, 5), 0)
        elif index == 3:
            # 1360's resampled glyph loses grey edge pixels under the binary
            # mask. Reuse history's fixed continuous contrast projection and
            # this existing enlarged template's own 3x3 then 5x5 smoothing.
            # The complete template, ROI and confidence gate stay unchanged.
            gray = np.clip((frame.min(2).astype(np.float32)-150)*(255/105),
                           0, 255).astype(np.uint8)
            image = cv2.cvtColor(cv2.GaussianBlur(cv2.GaussianBlur(gray,
                (3, 3), 0), (5, 5), 0), cv2.COLOR_GRAY2BGR)
        detail = context.run_recognition(node, image, pipeline_override={node: {
            'recognition': 'TemplateMatch', 'template': templates,
            'roi': [510, 1150, 100, 100], 'threshold': .95,
        }})
        targets = list(detail.filtered_results or ()) if detail and detail.hit else []
        if targets and all(tuple(row.box) == tuple(targets[0].box) for row in targets):
            targets = targets[:1]
        if targets or index == len(attempts)-1 or _inventory_button_profile(client_size) != (720, 1280):
            return targets, image, templates, enlarged


def _inventory_ocr(context, image, node, **params):
    from .simd_ocr import read_optional_simd_ocr

    if rows := read_optional_simd_ocr(image, params):
        return rows
    hit = context.run_recognition(node, image, pipeline_override={node: {
        "recognition": "OCR", "expected": ".+", **params,
    }})
    return tuple(hit.filtered_results or ()) if hit and hit.hit else ()


def _unique_inventory_text(rows, text, roi, *, threshold=0.85):
    x, y, width, height = roi
    found, inside = [], []
    for row in rows:
        score = getattr(row, "score", None)
        box = getattr(row, "box", ())
        if (type(score) not in {int, float} or not math.isfinite(score)
                or score < threshold or len(box) != 4
                or any(type(v) is not int for v in box)):
            continue
        rx, ry, rw, rh = box
        if (min(rw, rh) > 0 and x <= rx and y <= ry
                and rx + rw <= x + width and ry + rh <= y + height):
            inside.append(row)
            if re.fullmatch(text, str(getattr(row, "text", ""))):
                found.append(row)
    # The native OCR detector sometimes splits the inventory heading into
    # "所持スキルカ" and "ード". Join only one complete, adjacent title line;
    # do not accept a prefix or search arbitrary combinations of fragments.
    if not found and text == "所持スキルカード" and 2 <= len(inside) <= 3:
        ordered = sorted(inside, key=lambda row: row.box[0])
        adjacent = all(-min(a.box[3], b.box[3]) / 2 <= b.box[0] - (a.box[0]+a.box[2]) <= 20
                       and abs(a.box[1]-b.box[1]) <= 6
                       and min(a.box[1]+a.box[3],b.box[1]+b.box[3])-max(a.box[1],b.box[1])
                           >= min(a.box[3],b.box[3])*.8
                       for a, b in zip(ordered, ordered[1:]))
        if adjacent and "".join(row.text.strip() for row in ordered) == text:
            left = min(row.box[0] for row in ordered)
            top = min(row.box[1] for row in ordered)
            right = max(row.box[0]+row.box[2] for row in ordered)
            bottom = max(row.box[1]+row.box[3] for row in ordered)
            found = [SimpleNamespace(text=text, box=(left, top, right-left, bottom-top),
                                     score=min(row.score for row in ordered))]
    if len(found) != 1:
        raise ValueError(f"inventory page anchor is missing or ambiguous: {text}")
    return found[0]


def observe_inventory_hand(context, frames, reader, *, plan, expected_count):
    """Read only the complete, currently visible hand section; never infer a count.

    Page title, close button, section count, grid geometry and exact identities
    must agree on the supplied images. Only the hand section is returned.
    """
    if not frames or type(expected_count) is not int or not 1 <= expected_count <= 5:
        raise ValueError("inventory hand requires a visible hand and one to five slots")
    layouts, close_boxes = [], []
    for frame in frames:
        if tuple(frame.shape) != (1280, 720, 3):
            raise ValueError("inventory hand resolution is not calibrated")
        rows = _inventory_ocr(context, frame, "LiveInventoryPage")
        _unique_inventory_text(rows, "所持スキルカード", (15, 25, 285, 65))
        close = _unique_inventory_text(rows, "閉じる", (210, 1120, 290, 85))
        header = _unique_inventory_text(rows, r".*手札.*", (20, 395, 250, 105))
        hx, hy, hw, hh = header.box
        # Use the observed text's right edge. The old fixed 205px edge adds
        # animated stage pixels and can depress an otherwise correct count.
        text_rows = _inventory_ocr(context, frame[hy:hy+hh, 55:hx+hw+4].copy(),
                                   "LiveInventoryHandCount", only_rec=True)
        texts = [row.text.replace("（", "(").replace("）", ")").replace(" ", "")
                 for row in text_rows if math.isfinite(row.score) and row.score >= 0.95]
        if len(texts) != 1 or texts[0] != f"手札({expected_count})":
            raise ValueError("inventory hand count is unresolved or changed")
        layout = inventory_section_boxes(
            frame.shape, _inventory_header_geometry(frame, tuple(header.box)), expected_count)
        if len(layout) != expected_count:
            raise ValueError("inventory hand is not completely visible")
        layouts.append(layout)
        close_boxes.append(tuple(close.box))
    if any(any(index != old_index or any(abs(a-b) > 2 for a, b in zip(box, old_box))
               for (index, box), (old_index, old_box) in zip(layout, layouts[0]))
           for layout in layouts[1:]):
        raise ValueError("inventory hand grid has not settled")
    # OCR text boxes can vary while the button is stationary. Each frame has
    # already proved a unique close label in its region; navigation rereads
    # the label on a fresh frame instead of clicking any stored box.
    batch = reader.identify_batch(frames, [box for _, box in layouts[0]], plan=plan)
    ids = tuple(row["card_id"] for row in batch)
    return {"accepted": all(ids), "card_ids": ids, "identities": batch,
            "layouts": tuple(layouts), "close_box": close_boxes[-1]}


def inventory_section_boxes(frame_shape, header_box, count):
    """Locate complete thumbnails relative to a currently observed section.

    Scrolling moves the section, so an earlier screenshot's row coordinates
    are never reused. The normalized HIF inventory has four 142px columns.
    """
    if tuple(frame_shape[:2]) != (1280, 720) or type(count) is not int or not 0 <= count <= 100:
        raise ValueError("unsupported inventory layout or section count")
    if len(header_box) != 4 or any(type(value) is not int for value in header_box):
        raise ValueError("invalid inventory section header")
    x, y, width, height = header_box
    if not (20 <= x <= 60 and 22 <= height <= 42 and width >= 75 and 395 <= y < 1000):
        raise ValueError("inventory section header is outside its calibrated region")
    first_top = y + height + 17
    return tuple(
        (index, (column, first_top + (index // 4) * 154, 142, 142))
        for index in range(count)
        for column in ((48, 208, 368, 529)[index % 4],)
        if 406 <= first_top + (index // 4) * 154 <= 1007 - 142
        # The floating layout toggle covers the bottom-right thumbnail.
        and not (column + 142 > 606 and first_top + (index // 4) * 154 + 142 > 947)
    )


class InventoryThumbnailRecognizer:
    """Reuse clean references and separately verify normal/+ marker evidence.

    The 24px marker patch permits only a four-pixel translation of its central
    16px reference. This absorbs thumbnail rasterization without changing the
    existing marker error and margin thresholds or shifting the card artwork.
    """

    def __init__(self, references, embedding_recognizer):
        self.references = references
        self.recognizer = embedding_recognizer
        self.trial_identity = None
        if getattr(embedding_recognizer, 'gkh_hif_trial_inventory', None) is not None:
            from .hif_trial_inventory import TrialInventoryIdentity

            self.trial_identity = TrialInventoryIdentity(embedding_recognizer)
            # The delta is an additional family proof, not a new ranking of
            # production thumbnails. Keep the original old-family consumer.
            self.trial_original = InventoryThumbnailRecognizer(references, SimpleNamespace(
                gallery=embedding_recognizer.gallery._old_gallery,
                embedder=embedding_recognizer.embedder,
                upgrade_maximum_distance=embedding_recognizer.upgrade_maximum_distance,
                upgrade_minimum_margin=embedding_recognizer.upgrade_minimum_margin))
        gallery = embedding_recognizer.gallery
        self.groups = {}
        for index, group in enumerate(gallery.visual_group_ids):
            self.groups.setdefault(group, []).append(index)

    def identify(self, image, box, *, plan):
        if len(box) != 4 or any(type(value) is not int for value in box):
            return InventoryIdentity(None, None, "invalid_thumbnail_geometry")
        x, y, width, height = box
        array = np.asarray(image)
        if (
            array.ndim != 3
            or array.dtype != np.uint8
            or array.shape[2] != 3
            or min(x, y) < 0
            or min(width, height) < 32
            or abs(width - height) > 2
            or x + width > array.shape[1]
            or y + height > array.shape[0]
        ):
            return InventoryIdentity(None, None, "invalid_thumbnail_geometry")
        if self.trial_identity is not None:
            original = self.trial_original.identify(array, box, plan=plan)
            trial = self.trial_identity.identify(array, box, plan=plan)
            if (trial is None or trial['reason'] == 'trial_inventory_art_unresolved'
                    or original.card_id is not None and trial['card_id'] is None):
                # A weak new-family guess is not a family proof. Preserve even
                # an unresolved old-family observation for exact detail OCR.
                return original
            if original.card_id is not None:
                return InventoryIdentity(None, trial['visual_group'], 'trial_inventory_reference_conflict',
                    reference_evidence={'original': asdict(original), 'trial': trial})
            return InventoryIdentity(**trial)
        reference = self.references.measure_identity(array, box)
        if reference.get('status') == 'MEASURED_CANDIDATES':
            # A visible temporary number can obscure the coarse upper-half
            # reference (native 947). Strong whole-art retrieval is only an
            # unresolved candidate here; the collector must independently
            # prove the same unique current Live family and all three markers.
            candidates = reference.get('candidates', ())
            error,margin = reference.get('top_error'),reference.get('group_margin')
            if (len(candidates) >= 2
                    and len({row.get('reference_visual_group') for row in candidates}) == len(candidates)
                    and all(row.get('status') == 'MEASURED' and row.get('identity_low_confidence') is False
                            for row in candidates)
                    and all(type(value) in (int,float) and math.isfinite(value) for value in (error,margin))
                    and 0 <= error <= 20 and 0 < margin < 3):
                hits = self.recognizer.gallery.search(self.recognizer.embedder.embed(array,box),top_k=5)
                if len(hits) >= 2:
                    similarity = hits[0].similarity
                    embedding_margin = similarity-hits[1].similarity
                    if (math.isfinite(similarity) and math.isfinite(embedding_margin)
                            and .85 <= similarity <= 1 and embedding_margin >= .05
                            and hits[0].visual_group_id in self.groups):
                        return InventoryIdentity(None,hits[0].visual_group_id,
                            'inventory_ambiguous_reference_with_embedding',error,margin,
                            embedding_similarity=similarity,embedding_margin=embedding_margin,
                            reference_evidence=deepcopy(reference))
            return InventoryIdentity(None,None,'reference_identity_unresolved',
                                     reference_evidence=deepcopy(reference))
        if reference.get("status") != "MEASURED" or reference.get("identity_low_confidence") is not False:
            return InventoryIdentity(None, None, "reference_identity_unresolved")
        group = reference["reference_visual_group"]
        error = reference["identity_top_error"]
        margin = reference["identity_group_margin"]
        # A separate, conservative inventory gate; never change the arena
        # reference asset or its production thresholds through this consumer.
        if not math.isfinite(error) or not math.isfinite(margin) or error > 20 or margin <= 0:
            return InventoryIdentity(None, group, "inventory_reference_gate", error, margin)
        similarity, embedding_margin = None, None
        if margin < 3:
            # The existing coarse reference can leave simple illustrations
            # close together. Require independent the card artwork recognizer retrieval of the
            # same unique family; it cannot override a contradictory reference.
            hits = self.recognizer.gallery.search(self.recognizer.embedder.embed(array,box),top_k=5)
            if len(hits) < 2:
                return InventoryIdentity(None,group,"inventory_embedding_unresolved",error,margin)
            similarity = hits[0].similarity
            embedding_margin = similarity-hits[1].similarity
            if (not math.isfinite(similarity) or not math.isfinite(embedding_margin)
                    or similarity < .85 or embedding_margin < .05 or hits[0].visual_group_id != group):
                reason = 'inventory_embedding_disagreement'
                if (math.isfinite(similarity) and math.isfinite(embedding_margin)
                        and hits[0].visual_group_id == group):
                    if embedding_margin >= .05:
                        reason = 'inventory_embedding_low_confidence'
                    elif embedding_margin > 0:
                        # Still unresolved. Only an independently registered
                        # old slot and three current markers may supply identity.
                        reason = 'inventory_embedding_low_margin'
                return InventoryIdentity(None,group,reason,error,margin,
                                         embedding_similarity=similarity,embedding_margin=embedding_margin)
        return self._resolve_group_identity(array, box, plan=plan, group=group, error=error,
            margin=margin, similarity=similarity, embedding_margin=embedding_margin)

    def _resolve_group_identity(self, array, box, *, plan, group, error, margin,
                                similarity, embedding_margin, corroborated=False):
        x, y, width, height = box
        gallery = self.recognizer.gallery
        indexes = self.groups.get(group, ())
        members = sorted({(gallery.card_ids[i], gallery.upgrade_counts[i]) for i in indexes}, key=lambda row: int(row[0]))
        if not members:
            return InventoryIdentity(None, group, "reference_group_missing", error, margin)
        crop = array[y : y + height, x : x + width]
        if self.recognizer.embedder.source_color_order == "BGR":
            crop = crop[:, :, ::-1]
        query = extract_upgrade_marker(Image.fromarray(np.ascontiguousarray(crop)))
        scores = []
        for upgraded in (0, 1):
            templates = [gallery.upgrade_markers[i] for i in indexes if gallery.upgrade_counts[i] == upgraded]
            if not templates:
                # A singleton has no normal/+ ambiguity in the fixed catalog.
                if len(members) == 1:
                    return InventoryIdentity(members[0][0], group, "accepted_singleton", error, margin,
                                             embedding_similarity=similarity,embedding_margin=embedding_margin)
                return InventoryIdentity(None, group, "marker_reference_incomplete", error, margin)
            scores.append(
                min(
                    float(
                        cv2.minMaxLoc(
                            cv2.matchTemplate(
                                query,
                                template[4:20, 4:20].copy(),
                                cv2.TM_SQDIFF,
                            )
                        )[0]
                    )
                    / (16 * 16 * 3)
                    for template in templates
                )
            )
        best = int(scores[1] < scores[0])
        marker_errors = tuple(scores)
        if scores[best] > self.recognizer.upgrade_maximum_distance or scores[1 - best] - scores[best] < self.recognizer.upgrade_minimum_margin:
            return InventoryIdentity(None, group, "marker_identity_unresolved", error, margin, marker_errors)
        hit = EmbeddingHit(members[0][0], "inventory", 1.0, 0, group, tuple(row[0] for row in members), tuple(row[1] for row in members))
        try:
            resolved = resolve_card_identity((hit,), plan=plan, upgraded=bool(best))
        except ValueError:
            return InventoryIdentity(None, group, "plan_identity_unresolved", error, margin, marker_errors)
        return InventoryIdentity(resolved.card_id, group,
                                 "accepted_with_live_hand_family_and_marker" if corroborated else
                                 "accepted_with_embedding_disambiguation" if similarity is not None else "accepted",
                                 error, margin, marker_errors, similarity, embedding_margin)

    def identify_batch(self, frames, boxes, *, plan):
        """Require a consistent identity for each thumbnail in every supplied frame."""
        if not frames or not boxes:
            raise ValueError("inventory identity requires a visible frame and nonempty slots")
        results = []
        for box in boxes:
            observations = tuple(self.identify(frame, box, plan=plan) for frame in frames)
            ids = {row.card_id for row in observations}
            results.append(
                {
                    "box": tuple(box),
                    "card_id": observations[0].card_id if len(ids) == 1 and None not in ids else None,
                    "observations": observations,
                }
            )
        return tuple(results)
