"""Shared badge/detail semantics for the HIF thumbnail overview."""

import re
import csv
import json
import math
import unicodedata
from types import SimpleNamespace
from pathlib import Path
from collections import Counter

import cv2

try:
    from agent.utils.effect_text import normalize_score_effect_text
    from agent.arena_winrate.catalog import ArenaEntityCatalog
    from agent.arena_winrate.customization_badge import (
        measure_customization_badge,
        stable_customization_badge_presence,
    )
except ModuleNotFoundError:
    from utils.effect_text import normalize_score_effect_text
    from arena_winrate.catalog import ArenaEntityCatalog
    from arena_winrate.customization_badge import (
        measure_customization_badge,
        stable_customization_badge_presence,
    )


def load_customization_catalog(card_path, customization_path):
    with Path(card_path).open(encoding='utf-8-sig', newline='') as stream:
        cards = list(csv.DictReader(stream))
    for row in cards:
        row['id'] = int(row['id'])
        row['upgraded'] = str(row['upgraded']).upper() == 'TRUE'
    customizations = json.loads(Path(customization_path).read_text(encoding='utf-8'))
    return ArenaEntityCatalog(cards, customizations)


def inspect_thumbnail_customizations(frames, slots):
    """Read the badge on the same physical card in each supplied frame."""
    if not frames:
        raise ValueError('customization inspection needs a visible thumbnail')
    result = []
    for zone, index, card_id, box in slots:
        observations = tuple(measure_customization_badge(frame, box) for frame in frames)
        decision = stable_customization_badge_presence(observations, required_frames=len(frames))
        if decision.state.value == 'AMBIGUOUS':
            raise ValueError('inventory customization badge is ambiguous')
        result.append({'zone': zone, 'index': index, 'card_id': card_id, 'box': box,
                       'presence': decision.state.value, 'reason': decision.reason,
                       'observations': observations})
    return result


def live_slot_badge_absence(frames, checkpoint, run):
    """Observe only exposed, enabled Live badges; never infer a positive tier."""
    from card_selection.model import frame_identifier

    core = {k:checkpoint[k] for k in ('fields','families','frame_ids')}
    boxes = checkpoint['candidate_boxes']
    if (not frames or [frame_identifier(f) for f in frames] != list(core['frame_ids'])
            or any(tuple(f.shape) != (1280,720,3) for f in frames) or len(boxes) != len(frames)):
        raise ValueError('Live slot badge evidence lacks the complete core frames')
    rows = []
    for candidate in checkpoint['candidates']:
        slot = candidate.slot
        family = core['families'][slot]
        if (len(family) < 2 or sum(set(f)==set(family) for f in core['families']) < 2
                or not candidate.clickable or candidate.detector_label not in ('cards','suggestions')):
            continue
        slot_boxes = [group[slot] for group in boxes]
        if not _live_badge_exposed(boxes,slot):
            continue
        observations = [measure_customization_badge(frame,tuple(box))
            for frame,box in zip(frames,slot_boxes,strict=True)]
        decision = stable_customization_badge_presence(observations, required_frames=len(frames))
        if decision.state.value == 'CONFIDENT_ZERO':
            rows.append(dict(slot=slot,enabled=True,boxes=slot_boxes,observations=observations))
    return dict(version=1,run_id=run.run_id,revision=run.revision,core=core,
        candidate_boxes=boxes,zero_slots=rows,input_click_count=0,
        source='fresh_live_exposed_badge_absence')


def _live_badge_exposed(groups,slot):
    """The existing presence reader's sampled domain must be fully visible."""
    if (not groups or any(not 0 <= slot < len(g) for g in groups)
            or any(len(b)!=4 or any(type(v) is not int for v in b)
                or b[2:] not in ([190,250],(190,250)) for g in groups for b in g)):
        return False
    reference = groups[0][slot]
    for boxes in groups:
        x,y,w,h = boxes[slot]
        if any(abs(v-old)>2 for v,old in zip(boxes[slot],reference)):
            return False
        # measure_customization_badge's badge_box_within_card is legacy report
        # metadata, not its sampling ROI. Presence actually uses the fixed
        # radius-12.5 disk at (46,72) after 96x96 INTER_AREA normalization.
        # Enclose every contributing source pixel, plus one pixel of padding.
        left = x + math.floor((46-12.5)*w/96) - 1
        top = y + math.floor((72-12.5)*h/96) - 1
        right = x + math.ceil((46+12.5+1)*w/96) + 1
        bottom = y + math.ceil((72+12.5+1)*h/96) + 1
        if not (0 <= left < right <= 720 and 0 <= top < bottom <= 1280):
            return False
        # In the calibrated unselected fan, later cards cover earlier ones.
        # A rectangle behind this card does not occlude its badge pixels.
        foreground_order = False
        if len(boxes) in (4, 5):
            from card_selection.types import CandidateBox

            from .partial_card import build_live_card_views

            try:
                build_live_card_views((1280, 720, 3),
                    tuple(CandidateBox(i, tuple(box)) for i, box in enumerate(boxes)),
                    layout_profile='hif-four-card-art-v1' if len(boxes) == 4 else 'hif-five-card-art-v2')
                foreground_order = True
            except ValueError:
                pass
        if any((i > slot if foreground_order else i != slot)
               and ox<right and ox+ow>left and oy<bottom and oy+oh>top
            for i,(ox,oy,ow,oh) in enumerate(boxes)):
            return False
    return True


def verified_live_slot_badge_absence(proof,core,run):
    """Bind negative badge observations to current physical slots, not effects."""
    from .opening_observer import valid_opening_frame_ids

    if (proof.get('version') != 1 or proof.get('source') != 'fresh_live_exposed_badge_absence'
            or (proof.get('run_id'),proof.get('revision')) != (run.run_id,run.revision)
            or json.dumps(proof.get('core'),sort_keys=True) != json.dumps(core,sort_keys=True)
            or proof.get('input_click_count') != 0 or not valid_opening_frame_ids(core.get('frame_ids'))):
        raise ValueError('Live slot badge proof changed its core or epoch')
    groups = proof.get('candidate_boxes',())
    slots = []
    for row in proof.get('zero_slots',()):
        slot = row.get('slot')
        if (type(slot) is not int or slot in slots or row.get('enabled') is not True
                or not _live_badge_exposed(groups,slot)
                or row.get('boxes') != [g[slot] for g in groups]
                or len(row.get('observations',())) != len(core['frame_ids'])
                or any(o.get('box') != box or o.get('count') != 0
                    or o.get('presence_state') != 'CONFIDENT_ZERO'
                    for o,box in zip(row['observations'],row['boxes'],strict=True))
                or stable_customization_badge_presence(row['observations'],
                    required_frames=len(core['frame_ids'])).state.value != 'CONFIDENT_ZERO'):
            raise ValueError('Live slot badge absence is incomplete, occluded or contradictory')
        slots.append(slot)
    if not slots:
        raise ValueError('Live slot badge absence cannot distinguish this hand')
    return tuple(slots)


class PermanentCustomizationEvidenceRequired(ValueError):
    """A combined Support body cannot establish a new permanent inventory."""


_PLAIN_DETAIL_BODIES = {
    # 1730: the unupgraded 596 has no customization capability. Its green
    # artwork produced only a badge candidate; the complete current body is
    # required to establish zero. The card-face cost is not read by this path.
    '596': ('破竹の勢い好調が8ターン以上の場合、使用可集中+5'
            '以降の3ターンの間、ターン開始時、スコア+5試験・ステージ中1回'),
}


def _complete_plain_detail(card_id, observations):
    expected = _PLAIN_DETAIL_BODIES.get(card_id)
    if expected is None or not observations:
        return False
    for rows in observations:
        if not rows or any(
                not isinstance(row.text, str) or type(row.score) not in (int, float)
                or not math.isfinite(row.score) or not .95 <= row.score <= 1
                or len(row.box) != 4 or any(type(v) is not int for v in row.box)
                or row.box[0] < 190 or row.box[1] < 105 or min(row.box[2:]) <= 0
                or row.box[0]+row.box[2] > 700 or row.box[1]+row.box[3] > 382
                for row in rows):
            return False
        text = ''.join(row.text for row in sorted(rows, key=lambda row: (row.box[1], row.box[0])))
        if normalize_score_effect_text(''.join(unicodedata.normalize('NFKC', text).split())) != expected:
            return False
    return True


def verified_zero_customization_detail(detail):
    """Verify the complete original description on the recorded images."""
    try:
        frames = detail['frame_ids']
        rows = tuple(tuple(SimpleNamespace(**row) for row in group) for group in detail['rows'])
        return (detail['customizations'] == {} and bool(frames) and len(rows) == len(frames)
                and all(isinstance(frame, str) and frame for frame in frames)
                and len(set(frames)) == len(frames) and _complete_plain_detail(detail['card_id'], rows))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def confirmed_inventory_detail_title(catalog, card_id, observations):
    """Keep the observed title as a navigation anchor after detail validation."""
    from .upgrade import normalize_upgrade_title
    from .inventory import INVENTORY_DETAIL_TITLE_ROI

    if not observations:
        raise ValueError('inventory detail title needs an observed frame')
    left, top, width, height = INVENTORY_DETAIL_TITLE_ROI
    expected = catalog.skill_card_title(int(card_id))
    titles = []
    for rows in observations:
        candidates = []
        for row in rows:
            x, y, w, h = row.box
            if left <= x and top <= y and x+w <= left+width and y+h <= top+height:
                candidates.append(row)
        if (len(candidates) != 1 or type(candidates[0].score) not in (int, float)
                or not math.isfinite(candidates[0].score) or not .95 <= candidates[0].score <= 1
                or not isinstance(candidates[0].text, str)
                or min(candidates[0].box[2:]) <= 0):
            raise ValueError('inventory detail title is unresolved')
        title = candidates[0].text
        if normalize_upgrade_title(title) != expected:
            raise ValueError('inventory detail title differs from its verified card')
        catalog.confirm_clicked_skill_card(title, predicted_card_id=int(card_id))
        titles.append(title)
    if len(set(titles)) != 1:
        raise ValueError('inventory detail title changes across frames')
    return titles[0]


def resolve_detail_customizations(catalog, card_id, observations, *,
                                  allow_bound_support_projection=False, expected_count=None,
                                  native_ocr_matches=False):
    """Resolve complete permanent effects; native nonbattle matches need no extra score veto."""
    if not observations:
        raise ValueError('customization detail needs an observed frame')
    if expected_count is not None and (type(expected_count) is not int or expected_count < 1):
        raise ValueError('customization detail badge count must be positive')
    texts = []
    for rows in observations:
        if not rows or any(type(row.score) not in (float,int) or not math.isfinite(row.score)
                or not 0 <= row.score <= 1 or not native_ocr_matches and row.score < .95 for row in rows):
            raise ValueError('customization detail contains unresolved text')
        text = '\n'.join(row.text for row in sorted(rows, key=lambda row: (row.box[1], row.box[0])))
        catalog.confirm_clicked_skill_card(text, predicted_card_id=int(card_id))
        texts.append(text)
    # The title's extra plus is a temporary Support tier (D-047). Its combined
    # numeric body may even coincide with a different permanent group: neither
    # a parser match nor a previous run's group is independent permanent proof.
    multi_plus = ['++' in ''.join(unicodedata.normalize('NFKC', ''.join(
        row.text for row in sorted(rows, key=lambda row: (row.box[1], row.box[0]))
        if 105 <= row.box[1] <= 135)).split()) for rows in observations]
    if all(multi_plus) and not allow_bound_support_projection:
        raise PermanentCustomizationEvidenceRequired(
            'current_run_permanent_customization_evidence_required: '
            'Support multi-plus body is combined; D-047 estimates ordinary plus, '
            'but cannot infer this run permanent customization from that body')
    if any(multi_plus) and not all(multi_plus):
        raise ValueError('customization detail title changes across frames')
    if expected_count is None and not catalog.available_customization_ids(int(card_id)):
        if not _complete_plain_detail(card_id, observations):
            raise ValueError('plain customization detail lacks its complete verified body')
        expected_count = 0
    groups = [(catalog.resolve_effective_customizations_without_badge_count(int(card_id), text)
               if expected_count is None else catalog.resolve_effective_customizations(
                   int(card_id), text, expected_count=expected_count, allow_positive_completion=False))
              for text in texts]
    if any(group != groups[0] for group in groups[1:]):
        raise ValueError('customization detail changes across frames')
    return groups[0]


def inventory_detail_text_rows(rows):
    """Select the detail text carrier, excluding its cost icon and bullets.

    These are fixed panel regions, not confidence-based deletion. A weak word
    or number inside the text carrier remains present and fails consensus.
    The cost glyph belongs to the separate card-face cost observation.
    """
    from .inventory import INVENTORY_DETAIL_TITLE_ROI

    left, top, width, height = INVENTORY_DETAIL_TITLE_ROI
    selected = []
    for row in rows:
        x,y,w,h = row.box
        # Native titles can begin at x=189; their small margin must not widen
        # the body region or include the separate cost glyph on the right.
        title = left <= x and x+w <= left+width and top <= y and y+h <= top+height
        body = 190 <= x and 150 <= y and y+h <= 382
        if not (title or body):
            continue
        if x >= 610 and y+h <= 160:
            continue
        if 205 <= x and x+w <= 244 and 155 <= y and y+h <= 380:
            continue
        selected.append(row)
    return tuple(selected)


def read_inventory_detail_text_rows(context, frame):
    """Reread text merged with the fixed left decoration column.

    Wrapped lines retain their full left edge. Only a Japanese phrase prefixed
    by a detected plus in the pictogram column gets an independent body read;
    numerical signs inside the actual text carrier are never removed.
    """
    from .inventory import _inventory_ocr
    rows = inventory_detail_text_rows(_inventory_ocr(context,frame,'LiveInventoryCustomizationDetail'))
    # Cyan pictograms can themselves produce S/C letters. Their fixed two
    # columns and visible colored extent, not OCR confidence, identify them.
    hsv = cv2.cvtColor(frame[155:382,190:290],cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(85,100,120),(112,255,255))
    _,_,stats,_ = cv2.connectedComponentsWithStats(mask)
    icons = [(int(x+190-6),int(y+155-6),int(w+12),int(h+12)) for x,y,w,h,area in stats[1:]
             if (8 <= x <= 17 or 56 <= x <= 64) and 30 <= w <= 38
             and 20 <= h <= 30 and area/(w*h) >= .25]
    # The good-condition pictogram has a white centre. On the small client
    # the cyan lower tip separates from the body; OCR spans both pieces.
    small_client = getattr(context, 'gkh_hif_client_size', None) in ((359, 638), (360, 640))
    condition_icons = []
    for x, y, w, h, area in (stats[1:] if small_client else ()):
        if not (8 <= x <= 17 and 30 <= w <= 38 and 20 <= h <= 30 and area >= w*h*.25):
            continue
        for tx, ty, tw, th, tip_area in stats[1:]:
            if (0 <= ty - (y+h) <= 3 and 10 <= tw <= 18 and 5 <= th <= 9
                    and abs((2*tx+tw) - (2*x+w)) <= 4 and tip_area >= tw*th*.35):
                condition_icons.append((int(x+190-6), int(y+155-6), int(w+12), int(ty+th-y+12)))
    icons.extend(condition_icons)
    # The extra-action glyph's white card outline splits its cyan diamond.
    # Keep the existing component path; accept this split shape only when all
    # four cyan tips are visible in the fixed right pictogram column (1725).
    for x, y, w, h, _ in stats[1:]:
        if not (68 <= x <= 72 and 9 <= w <= 13 and 5 <= h <= 7):
            continue
        left = int(x + w // 2 - 16)
        diamond = mask[y:y+33, left:left+33]
        if diamond.shape != (33, 33):
            continue
        tips = (diamond[:5,12:21], diamond[-5:,12:21],
                diamond[12:21,:5], diamond[12:21,-5:])
        if all(cv2.countNonZero(tip) >= 26 for tip in tips):
            # OCR can include the icon's white border. Its right edge stays
            # before the independently read body text at x=287.
            icons.append((left+190-9, int(y)+155-9, 47, 51))
    output = []
    for row in rows:
        x,y,w,h = row.box
        if any(ix <= x and iy <= y and x+w <= ix+iw and y+h <= iy+ih for ix,iy,iw,ih in icons):
            continue
        duration = small_client and re.match(r'^\+\d+ターン後[、,]', row.text)
        if (205 <= x < 235 and 155 <= y
                and (re.match(r'^\+[\u3040-\u30ff\u3400-\u9fff]',row.text) or duration)):
            crop = frame[y:y+h,235:x+w].copy()
            reread = _inventory_ocr(context,crop,'LiveInventoryCustomizationBodyLine',only_rec=True)
            if (len(reread)==1 and reread[0].text.replace(' ','')==row.text[1:].replace(' ','')
                    and type(reread[0].score) in (int,float) and math.isfinite(reread[0].score)
                    and .95 <= reread[0].score <= 1):
                row = SimpleNamespace(text=reread[0].text,score=reread[0].score,box=(235,y,x+w-235,h))
        elif re.search('[タ夕]一ン', row.text) and x >= 235 and any(
                abs(ix + iw - 6 - x) <= 3 and iy <= y+h//2 <= iy+ih for ix,iy,iw,ih in condition_icons):
            # Verify the long-vowel sign on the actual line, retaining every
            # number and effect character. The alternate text alone is no proof.
            reread = _inventory_ocr(context, frame[y:y+h,x:x+w].copy(), 'LiveInventoryCustomizationBodyLine', only_rec=True)
            if (len(reread) == 1 and reread[0].text == re.sub('[タ夕]一ン', 'ターン', row.text)
                    and .95 <= reread[0].score <= 1):
                row = SimpleNamespace(text=reread[0].text, score=reread[0].score, box=row.box)
        elif small_client and x >= 235 and 150 <= y and row.score < .95:
            # Retain one pixel of context around low-resolution text strokes.
            # A second reading can improve confidence only for identical text.
            crop = frame[y-1:y+h+1,x-1:x+w+1].copy()
            reread = _inventory_ocr(context, crop, 'LiveInventoryCustomizationBodyLine', only_rec=True)
            if len(reread) == 1 and reread[0].text == row.text and .95 <= reread[0].score <= 1:
                row = SimpleNamespace(text=row.text, score=reread[0].score, box=row.box)
        output.append(row)
    return tuple(output)


def confirmed_deck_card_rows(baseline, composition, *, presence=()):
    """Bind fresh physical slots to the user's confirmed complete final deck.

    A DETAIL_CANDIDATE remains compatible with every confirmed variant. Only
    observed badge absence removes positive variants; the complete
    variant counts must then leave exactly one physical assignment.
    """
    zones = ('手札', '山札', '捨札', '除外')
    if (not isinstance(baseline, dict) or not isinstance(baseline.get('source_id'), str)
            or re.fullmatch(r'[0-9]+-[a-z0-9-]+', baseline['source_id']) is None
            or not isinstance(baseline.get('cards'), (list, tuple)) or not baseline['cards']):
        raise ValueError('confirmed deck baseline source or complete cards are missing')
    counts = Counter()
    indexes = {zone: 0 for zone in zones}
    for row in baseline['cards']:
        if (not isinstance(row, dict) or row.get('zone') not in indexes
                or type(row.get('index')) is not int or row['index'] != indexes[row['zone']]
                or not isinstance(row.get('card_id'), str) or not row['card_id'].isdecimal()
                or str(int(row['card_id'])) != row['card_id']
                or not isinstance(row.get('customizations'), dict)):
            raise ValueError('confirmed deck baseline physical cards are incomplete')
        group = row['customizations']
        if (any(not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key
                or type(count) is not int or count < 1 for key, count in group.items())
                or row.get('source') not in ('observed_detail', 'observed_badge_absence')
                or (row['source'] == 'observed_detail') != bool(group)):
            raise ValueError('confirmed deck baseline customization source or group is invalid')
        counts[row['card_id'], tuple(sorted(group.items()))] += 1
        indexes[row['zone']] += 1
    if (not isinstance(composition, dict) or set(composition) != set(zones)
            or any(not isinstance(cards, (list, tuple)) for cards in composition.values())
            or any(not isinstance(cid, str) or not cid.isdecimal() or str(int(cid)) != cid
                   for cards in composition.values() for cid in cards)):
        raise ValueError('confirmed deck needs all four fresh canonical card zones')
    baseline_cards = Counter()
    for (canonical, _), count in counts.items():
        baseline_cards[canonical] += count
    if Counter(cid for cards in composition.values() for cid in cards) != baseline_cards:
        raise ValueError('confirmed deck complete canonical counts differ from the baseline')
    physical = {(zone, index): cid for zone in zones
                for index, cid in enumerate(composition[zone])}
    badges = {}
    for row in presence:
        if (not isinstance(row, dict) or type(row.get('index')) is not int
                or (row.get('zone'), row['index']) not in physical
                or physical[row['zone'], row['index']] != row.get('card_id')
                or (row['zone'], row['index']) in badges):
            raise ValueError('confirmed deck badge slot is extra, duplicated or mismatched')
        badges[row['zone'], row['index']] = row
    assigned = {}
    for canonical in baseline_cards:
        variants = {group: count for (cid, group), count in counts.items() if cid == canonical}
        slots = [slot for slot, cid in physical.items() if cid == canonical]
        if len(variants) == 1:
            assigned.update((slot, next(iter(variants))) for slot in slots)
            continue
        remaining = Counter(variants)
        candidates = []
        for slot in slots:
            badge = badges.get(slot)
            if badge is None:
                raise ValueError('confirmed deck mixed variants need fresh badge observations')
            frames = badge.get('frame_ids', ())
            observations = badge.get('observations', ())
            box = badge.get('box', ())
            if (not isinstance(frames, (tuple, list)) or not frames
                    or any(not isinstance(frame, str) or not frame for frame in frames)
                    or len(set(frames)) != len(frames) or not isinstance(observations, (tuple, list))
                    or len(observations) != len(frames) or not isinstance(box, (tuple, list)) or len(box) != 4
                    or any(type(value) is not int for value in box) or min(box[2:]) <= 0
                    or any(not isinstance(row, dict) or not isinstance(row.get('box'), (list, tuple))
                           or list(row['box']) != list(box)
                           or not isinstance(row.get('internal_features'), dict)
                           or type(row['internal_features'].get('seeded_plate_candidate')) is not bool
                           for row in observations)):
                raise ValueError('confirmed deck mixed badge evidence lacks its recorded frames')
            decision = stable_customization_badge_presence(observations, required_frames=len(frames))
            if (badge.get('presence') != decision.state.value
                    or decision.state.value not in ('CONFIDENT_ZERO', 'DETAIL_CANDIDATE')):
                raise ValueError('confirmed deck badge presence is unresolved or contradictory')
            if decision.state.value == 'CONFIDENT_ZERO':
                assigned[slot] = ()
                remaining[()] -= 1
                if remaining[()] < 0:
                    raise ValueError('confirmed deck badge absence exceeds the baseline count')
            else:
                candidates.append(slot)
        available = [group for group, count in remaining.items() if count > 0]
        # Candidate slots share the same unrestricted domain. Two remaining
        # groups can exchange slots, so their assignment cannot be unique.
        if (len(available) != 1 or sum(remaining.values()) != len(candidates)):
            raise ValueError('confirmed deck variants have no unique physical assignment')
        assigned.update((slot, available[0]) for slot in candidates)
    if Counter((physical[slot], group) for slot, group in assigned.items()) != counts:
        raise ValueError('confirmed deck assigned variant counts differ from the baseline')
    return [dict(zone=zone, index=index, card_id=canonical,
                 customizations=dict(assigned[zone, index]), source='confirmed_deck_baseline',
                 baseline_source_id=baseline['source_id'])
            for (zone, index), canonical in physical.items()]


def _retained_inventory_variants(bundle, expected):
    from .high_score.opening import expected_permanent_inventory_keys

    before = {card.instance_id:card for card in bundle.state.deck.all_cards}
    after = {card.instance_id:card for card in expected.deck.all_cards}
    if (len(before) != len(bundle.state.deck.all_cards)
            or len(after) != len(expected.deck.all_cards)
            or any(after.get(instance) != card for instance,card in before.items())):
        raise ValueError('permanent inventory changed across the committed transition')
    # This validates additions but supplies only their expected plain identity.
    # Every physical copy of an added family still requires fresh evidence.
    keys = expected_permanent_inventory_keys(bundle, expected)
    return {canonical: {customizations for _,cid,customizations in keys if cid==canonical}
            for _,canonical,_ in keys}


def retained_ambiguous_card_ids(bundle, expected):
    """Require fresh evidence for mixed variants and every newly added family."""
    variants = _retained_inventory_variants(bundle, expected)
    existing = {instance for instance,_,_ in bundle.inventory_keys}
    generated = {card.card_id for card in expected.deck.all_cards if card.instance_id not in existing}
    return frozenset(canonical for canonical, groups in variants.items()
                     if len(groups) != 1 or canonical in generated)


def retained_permanent_card_rows(bundle, expected, composition, *, presence=(), details=()):
    """Retain unique semantics; mixed copies need their own fresh evidence.

    A detail candidate is not positive customization proof. Only proven badge
    absence or complete current detail text can distinguish mixed variants.
    """
    variants_by_card = _retained_inventory_variants(bundle, expected)
    fresh_ids = retained_ambiguous_card_ids(bundle, expected)
    if expected.created_card_count != bundle.state.created_card_count:
        from .high_score.opening import expected_permanent_inventory_keys

        if (set(composition) != {'手札','山札','捨札','除外'}
                or Counter(cid for cards in composition.values() for cid in cards)
                != Counter(cid for _,cid,_ in expected_permanent_inventory_keys(bundle, expected))):
            raise ValueError('generated inventory needs complete fresh canonical composition')
    badges = {(row['zone'],row['index']):row for row in presence}
    resolved = {(row['zone'],row['index']):row for row in details}
    if len(badges)!=len(presence) or len(resolved)!=len(details):
        raise ValueError('retained customization observations are duplicated')
    used_badges, used_details = set(), set()
    rows=[]
    for zone,ids in composition.items():
        for index,canonical in enumerate(ids):
            variants=variants_by_card.get(canonical,set())
            if canonical in fresh_ids or len(variants)!=1:
                key = zone,index
                badge = badges.get(key)
                if not variants or badge is None or badge.get('card_id')!=canonical:
                    raise ValueError('retained customization variants require direct observation')
                used_badges.add(key)
                if badge.get('presence')=='CONFIDENT_ZERO' and key not in resolved:
                    group = {}
                    source = 'observed_badge_absence'
                elif badge.get('presence')=='DETAIL_CANDIDATE' and key in resolved:
                    detail = resolved[key]
                    group = detail.get('customizations')
                    if (detail.get('card_id')!=canonical or not isinstance(group,dict) or not group
                            or any(not isinstance(k,str) or not k.isdecimal() or type(v) is not int or v<1
                                   for k,v in group.items())):
                        raise ValueError('retained mixed customization detail is unresolved')
                    source = 'observed_detail'
                    used_details.add(key)
                else:
                    raise ValueError('retained customization variants require direct observation')
                if tuple(sorted((int(k),v) for k,v in group.items())) not in variants:
                    raise ValueError('retained customization differs from the proven variants')
                rows.append(dict(zone=zone,index=index,card_id=canonical,customizations=dict(group),source=source,
                    **({'temporary_support':resolved[key]['temporary_support']}
                       if key in resolved and resolved[key].get('temporary_support') else {})))
                continue
            rows.append(dict(zone=zone,index=index,card_id=canonical,
                customizations={str(k):v for k,v in next(iter(variants))},
                source='retained_verified_card',
                retained_from=dict(run_id=bundle.run_id,revision=bundle.revision)))
    if used_badges!=set(badges) or used_details!=set(resolved):
        raise ValueError('retained customization observations have extra physical targets')
    return rows


def partial_opening_details(source):
    """Recover permanent readings for individually located slots of the same opening."""
    panel = source['live_events'][0]['inventory_events'][0]
    slots = {(zone, index, card) for page in panel.get('pages', ())
             for zone, index, card, _box in page.get('slots', ())}
    candidates = {(row['zone'], row['index'], row['card_id']) for row in panel.get('customization_presence', ())
                  if row.get('presence') == 'DETAIL_CANDIDATE'}
    readings = {}
    for row in panel.get('detail_observations', ()):
        key = row['zone'], row['index'], row['card_id']
        frames, group = row.get('frame_ids', ()), row.get('customizations', {})
        if (key in slots and key in candidates and not row.get('temporary_support')
                and frames and len(frames) == len(set(frames)) == len(row.get('rows', ()))
                and group and all(isinstance(k, str) and k.isdecimal() and type(v) is int and v > 0
                                  for k, v in group.items())):
            retained = row.get('retained_from') or {}
            original = row.get('opening_detail_source') or dict(source_result=retained.get('source_result', source['id']),
                frame_ids=list(frames), run_id=source['observed_live_context']['run_id'],
                revision=source['observed_live_context']['revision'])
            reading = dict(row, opening_detail_source=original)
            if retained.get('source_result'):
                reading.pop('retained_from')
            readings[key] = reading
    return readings


def unread_opening_candidates(candidates, saved, details):
    """Match current zone, slot, card and badge before reusing the old detail."""
    unread = []
    for row in candidates:
        key = row['zone'], row['index'], row['card_id']
        if key in saved and row.get('presence') == 'DETAIL_CANDIDATE':
            details.append(saved[key])
        else:
            unread.append(row)
    return unread


def retained_customization_details(bundle, expected, candidates):
    """Reuse immutable details only when every copy has the same proven effect."""
    if ({card.instance_id:card for card in bundle.state.deck.all_cards}
            != {card.instance_id:card for card in expected.deck.all_cards}):
        raise ValueError('permanent inventory changed across the committed transition')
    keys = bundle.inventory_keys
    if {row[0] for row in keys} != {card.instance_id for card in expected.deck.all_cards}:
        raise ValueError('permanent inventory instance mapping is incomplete')
    details=[]
    for row in candidates:
        variants={customizations for _,canonical,customizations in keys if canonical==row['card_id']}
        if len(variants)!=1 or not next(iter(variants)):
            raise ValueError('retained customization variants require direct observation')
        details.append(dict(zone=row['zone'],index=row['index'],card_id=row['card_id'],
            customizations={str(k):v for k,v in next(iter(variants))},
            retained_from=dict(run_id=bundle.run_id,revision=bundle.revision)))
    return details


def bind_inventory_customizations(composition, presence, details):
    """Bind every physical slot; absence and positive details stay explicit."""
    expected = {(zone,index):card_id for zone, cards in composition.items()
                for index,card_id in enumerate(cards)}
    badges = {(row['zone'],row['index']):row for row in presence}
    resolved = {(row['zone'],row['index']):row for row in details}
    if len(badges) != len(presence) or len(resolved) != len(details) or set(badges) != set(expected):
        raise ValueError('card customization physical inventory is incomplete or duplicated')
    positive = {key for key,row in badges.items() if row['presence']=='DETAIL_CANDIDATE'}
    if set(resolved) != positive:
        raise ValueError('positive customization detail inventory is incomplete or extra')
    result = []
    for key, card_id in expected.items():
        row = badges[key]
        if row['card_id'] != card_id or row['presence'] not in ('DETAIL_CANDIDATE','CONFIDENT_ZERO'):
            raise ValueError('card customization identity or presence is contradictory')
        group = {}
        if key in resolved:
            detail = resolved[key]
            group = detail['customizations']
            if (detail['card_id'] != card_id or not isinstance(group, dict)
                    or (not group and not verified_zero_customization_detail(detail))
                    or any(not isinstance(k,str) or not k.isdecimal()
                           or type(v) is not int or v<1 for k,v in group.items())):
                raise ValueError('card customization detail is unresolved')
        if key in resolved:
            source = 'observed_detail'
            if group and resolved[key].get('retained_from'):
                source = 'retained_verified_detail'
        else:
            source = 'observed_badge_absence'
        result.append({'zone':key[0], 'index':key[1], 'card_id':card_id,
                       'customizations':dict(group),
                       'source':source,
                       **({'rows':resolved[key]['rows'], 'frame_ids':resolved[key]['frame_ids']}
                          if key in resolved and not group else {}),
                       **({'temporary_support':resolved[key]['temporary_support']}
                          if group and resolved[key].get('temporary_support') else {}),
                       **({'opening_detail_source':resolved[key]['opening_detail_source']}
                          if group and resolved[key].get('opening_detail_source') else {}),
                       **({'retained_from':resolved[key]['retained_from']}
                          if group and resolved[key].get('retained_from') else {})})
    return result
