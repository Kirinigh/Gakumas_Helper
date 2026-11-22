"""HIF run facts supplied by the host and the first calibrated ranking layout.

Run facts come from the verified route session, never from an expected-plan
parameter. Other ranking positions remain unknown until calibrated.
"""

from __future__ import annotations

import re
import math
from typing import Any
from pathlib import Path
from functools import lru_cache
from dataclasses import replace, dataclass

from .layout import _bgr_to_opencv_hsv
from .ocr_evidence import _valid_ocr_result


@dataclass(frozen=True)
class HifOpeningTurnInformation:
    turn_attributes: tuple[str, ...]
    vocal_percent: int
    dance_percent: int
    visual_percent: int

    @property
    def score_multipliers(self) -> tuple[float, ...]:
        values = {"vocal": self.vocal_percent, "dance": self.dance_percent, "visual": self.visual_percent}
        return tuple(values[attribute] / 100 for attribute in self.turn_attributes)


def hif_opening_turn_information_consensus(observations, *, frame_shape):
    """Read the complete nine-row opening information panel in three frames.

    This is an OCR result consumer, not a controller. Different round layouts
    must be calibrated separately; the ring colors do not supply hidden turns.
    """
    if tuple(frame_shape[:2]) != (1280, 720) or len(observations) != 3:
        raise ValueError("HIF turn information requires three calibrated frames")
    parsed = []
    for rows in observations:

        def inside(roi):
            x, y, width, height = roi
            found = []
            for row in rows:
                box, score = row.get("box"), row.get("score")
                if (
                    not isinstance(box, (list, tuple))
                    or len(box) != 4
                    or any(type(value) is not int for value in box)
                    or type(score) not in {int, float}
                    or not math.isfinite(score)
                    or score < 0.95
                ):
                    continue
                rx, ry, rw, rh = box
                if rw > 0 and rh > 0 and x <= rx and y <= ry and rx + rw <= x + width and ry + rh <= y + height:
                    found.append(row)
            return sorted(found, key=lambda row: row["box"][0])

        title = inside((35, 125, 235, 45))
        if len(title) != 1 or title[0]["text"] != "現在1ターン目":
            raise ValueError("HIF opening turn header unresolved")
        percentages = []
        for x in (110, 310, 505):
            hits = inside((x, 215, 150, 60))
            if len(hits) != 1 or not re.fullmatch(r"[1-9]\d*[%％]", hits[0]["text"]):
                raise ValueError("HIF attribute percentage unresolved")
            percentages.append(int(hits[0]["text"][:-1]))
        attributes = []
        for offset in range(9):
            top = 322 + offset * 88
            labels = inside((45, top, 150, 65))
            label = "".join(row["text"] for row in labels)
            if label != ("現在のターン" if offset == 0 else f"{offset}ターン後"):
                raise ValueError("HIF turn row offset unresolved")
            hits = inside((245, top, 125, 65))
            mapping = {"ボーカル": "vocal", "ダンス": "dance", "ビジュアル": "visual"}
            if len(hits) != 1 or hits[0]["text"] not in mapping:
                raise ValueError("HIF turn row attribute unresolved")
            attributes.append(mapping[hits[0]["text"]])
        parsed.append(HifOpeningTurnInformation(tuple(attributes), *percentages))
    if any(value != parsed[0] for value in parsed[1:]):
        raise ValueError("HIF turn information frames disagree")
    return parsed[0]


def locate_hif_selected_detail_title(image: Any) -> tuple[int, int, int, int] | None:
    """Locate the white selected-card panel independently of its variable height.

    This only proposes a title ROI; the caller must prove SELECT and resolve
    three complete titles against the retrieved visual family and catalog.
    """
    import cv2
    import numpy as np

    shape = getattr(image, "shape", ())
    if len(shape) != 3 or shape[:2] != (1280, 720) or shape[2] < 3:
        return None
    region = image[260:830, 70:650, :3]
    low = _bgr_channel_bounds(region)[0] if region.dtype == np.uint8 else region.min(axis=2)
    mask = (low > 225).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    panels = [
        (int(x) + 70, int(y) + 260, int(w), int(h))
        for x, y, w, h, area in stats[1:]
        if 520 <= w <= 540 and h >= 100 and area / (w * h) >= 0.65 and 90 <= x + 70 <= 100 and y > 0
    ]
    if len(panels) != 1:
        return None
    x, y, w, _h = panels[0]
    return x + 16, y + 6, w - 84, 44


def hif_selection_click_box(boxes, slot):
    """Keep a first-selection point outside both overlapping neighbour boxes."""
    box = boxes[slot]
    if len(boxes) < 4:
        return box
    x,y,width,height = box
    left = max(x, max((other[0]+other[2] for other in boxes[:slot]), default=x))
    right = min(x+width, min((other[0] for other in boxes[slot+1:]), default=x+width))
    if right-left < 32:
        raise ValueError('Live selection has no unique visible card strip')
    return box if left == x and right == x+width else (left+8,y,right-left-16,height)


def hif_selected_fragment_proposal(frame_shape: tuple[int, ...], candidates: Any):
    """Propose a single split-card repair; callers must prove SELECT separately.

    Six detections are not six cards when a raised card's right edge is also
    detected. Only the reviewed right fragment / wider left host pair is
    eligible. Ordinary overlapping neighbours have different top/bottom edges.
    """
    if frame_shape[:2] != (1280, 720):
        return None
    cards = sorted((c for c in candidates if c.box[1] >= 704), key=lambda c: c.box[0])
    if len(cards) != 6:
        return None
    solutions = []
    for index, (host, fragment) in enumerate(zip(cards, cards[1:])):
        x, y, w, h = host.box
        fx, fy, fw, fh = fragment.box
        right = max(x + w, fx + fw)
        ordinary_tops = [c.box[1] for i, c in enumerate(cards) if i not in (index, index + 1)]
        if not (
            140 <= w <= 190
            and 60 <= fw <= 125 and fw < w
            and x < fx < x + w
            and 180 <= right - x <= 200
            and abs(y - fy) <= 8
            and abs(y + h - fy - fh) <= 8
            and min(ordinary_tops) - max(y, fy) >= 12
            and host.detector_score >= 0.80
            and fragment.detector_score < host.detector_score
        ):
            continue
        top = min(y, fy)
        merged = replace(host, box=(x, top, right - x, max(y + h, fy + fh) - top))
        repaired = [*cards[:index], merged, *cards[index + 2 :]]
        repaired = tuple(replace(c, slot=i) for i, c in enumerate(repaired))
        solutions.append((repaired, index))
    return solutions[0] if len(solutions) == 1 else None


class HifFrameTitleCache:
    """Share pure title geometry only within one normalization of one image."""

    def __init__(self, image):
        self._image = image
        self._gray = {}
        self._unselected = None
        self._edges = None

    def _check_image(self, image):
        if image is not self._image:
            raise ValueError("HIF title preprocessing cannot be reused for another frame")

    def gray(self, image, *, selected=False):
        self._check_image(image)
        if selected not in self._gray:
            self._gray[selected] = _hif_gray_title_plates(image, selected=selected, title_cache=self)
        return self._gray[selected]

    def edges(self, image):
        self._check_image(image)
        if self._edges is None:
            self._edges = _hif_canny_edges(image)
        return self._edges

    def unselected(self, image):
        self._check_image(image)
        if self._unselected is None:
            self._unselected = _hif_unselected_title_plates(image, gray_plates=self.gray(image))
        return self._unselected


def _hif_canny_edges(image, title_cache=None):
    """Reuse only the existing full-frame 80/160 Canny within this image's cache."""
    import cv2

    return cv2.Canny(image, 80, 160) if title_cache is None else title_cache.edges(image)


def _hif_observed_vertical_rim(image, edge, center):
    """Count rim edges only when both sides contain observed pixels."""
    import numpy as np

    band = edge[880:1095, center-3:center+3] > 0
    observed = np.any(image[880:1095, center-4:center+4], axis=2)
    supported = band & observed[:, :-2] & observed[:, 1:-1] & observed[:, 2:]
    return np.count_nonzero(np.any(supported, axis=1)) >= 190


def _hif_complete_halo_title_matches(card, plates):
    """Existing unselected halo envelope, anchored by its actual title plate."""
    x, y, w, h = card.box
    return [(px-2, py+ph+2-250, 190, 250) for px,py,pw,ph in plates
            if 0 <= px-2-x <= 10 and pw <= w <= pw+40
            and 10 <= py+ph+2-250-y <= 20
            and 8 <= y+h-(py+ph+2) <= 20]


def _hif_exposed_unraised_body_edges(edge, left, end, *, both_sides, body_top=None):
    """Shared pixel proof for a title-anchored unraised physical body."""
    import numpy as np

    start = left+15
    # Complete titles fix the actual body. A screen-fixed band can include
    # background above it and discard the same number of real lower edge rows.
    side_top = 880 if body_top is None else body_top
    top_start, top_end = (882, 889) if body_top is None else (body_top-3, body_top+4)
    return (end-start >= 100
            and np.any(edge[top_start:top_end,start:end] > 0,axis=0).mean() >= .95
            and any(np.count_nonzero(np.any(edge[side_top:side_top+215,b:b+2] > 0,axis=1)) >= 190
                    for b in range(left-3,left+2))
            and (not both_sides or any(
                np.count_nonzero(np.any(edge[side_top:side_top+215,b:b+2] > 0,axis=1)) >= 190
                for b in range(left+187,left+191))))


def _hif_restore_cropped_unraised_body(image, card, plates, title_cache):
    """Current title and body edges own a short detector crop in either row."""
    x,y,w,h = card.box
    if card.detector_label not in ('cards','useless') or not (105 <= w <= 190 and 190 <= h < 250):
        return card
    owned = []
    for index, (px,py,pw,ph) in enumerate(plates):
        left, top, bottom = px-2, py+ph+2-250, py+ph+2
        if (abs(x-left) > 8 or not top+4 < y < bottom
                or abs(y+h-bottom) > 4 or x+w > left+198):
            continue
        end = min(left+175,plates[index+1][0]-4) if index+1 < len(plates) else left+175
        if _hif_exposed_unraised_body_edges(_hif_canny_edges(image,title_cache),
                left,end,both_sides=pw >= 180,body_top=top):
            owned.append((left,top,190,250))
    return replace(card,box=owned[0]) if len(owned) == 1 else card


def _hif_is_anchored_body_fragment(fragment, cards, body):
    """Check raw overlap with the unique detector body at a complete title."""
    left, top, width, height = body
    hosts = [card for card in cards
             if card.detector_label in ('cards', 'useless')
             and abs(card.box[0]-left) <= 8 and abs(card.box[1]-top) <= 4
             and 140 <= card.box[2] <= width and card.box[0]+card.box[2] <= left+width+8
             and abs(card.box[1]+card.box[3]-top-height) <= 4]
    if len(hosts) != 1:
        return False
    host = hosts[0]
    if (not host.detector_score >= .80 or host.detector_label != fragment.detector_label
            or host.clickable != fragment.clickable):
        return False
    hx, hy, hw, hh = host.box
    x, y, w, h = fragment.box
    overlap = max(0, min(hx+hw, x+w)-max(hx, x)) * max(0, min(hy+hh, y+h)-max(hy, y))
    return overlap >= min(hw*hh, w*h)*.50


def _hif_bordered_neighbour_halo_matches(card, plates, anchors, selected_box, edge,
                                        *, use_title_body_top=False):
    """Use exposed top/outer body edges when a neighbour's glow is taller.

    The full title fixes the body, while the halo remains only a detector
    hint. Reuse the existing190-row body-edge and95% exposed-top gates.
    A complete unselected three-card row has both sides exposed. In an
    overlapping four/five-card row only the final full title has both sides
    exposed. A preceding title's white component can stop before the next
    card; that card's actual left border must then prove the overlap boundary.
    """
    import numpy as np

    if selected_box is None:
        if (len(anchors) not in (3,4,5) or len(plates) != len(anchors)
                or any(abs(p[0]-2-anchor) > 8
                       for p, anchor in zip(plates, anchors, strict=True))):
            return []
        if len(anchors) == 3:
            if any(not 180 <= p[2] <= 190 for p in plates):
                return []
            allowed_anchors = anchors
        else:
            if not 180 <= plates[-1][2] <= 190:
                return []
            for p,next_plate in zip(plates,plates[1:]):
                next_left = next_plate[0]-2
                if not 105 <= p[2] < 180 or p[0]+p[2] > next_left+3:
                    return []
                if p[0]+p[2] < next_left-3 and not any(
                        np.count_nonzero(np.any(edge[880:1095,b:b+2] > 0,axis=1)) >= 190
                        for b in range(next_left-3,next_left+2)):
                    return []
            allowed_anchors = anchors[-1:]
    else:
        selected_slots = [i for i, anchor in enumerate(anchors) if abs(selected_box[0]-anchor) <= 8]
        if len(selected_slots) != 1 or selected_slots[0] == 0:
            return []
        allowed_anchors = (anchors[selected_slots[0]-1],)
    x, y, w, h = card.box
    if w <= 0 or h <= 0:
        return []
    matches = []
    for px, py, pw, ph in plates:
        left, top = px-2, py+ph+2-250
        end = min(left+175, selected_box[0]-2) if selected_box is not None else left+175
        overlap = max(0,min(x+w,left+190)-max(x,left)) * max(0,min(y+h,top+250)-max(y,top))
        # A detector proposal must predominantly belong to this one physical
        # body. Its glow's height/offset does not redefine that proven body.
        if (any(abs(left-anchor) <= 8 for anchor in allowed_anchors)
                and overlap >= w*h*.65
                and _hif_exposed_unraised_body_edges(
                    edge,left,end,both_sides=selected_box is None,
                    body_top=top if use_title_body_top else None)):
            matches.append((left, top, 190, 250))
    return matches


def hif_selected_row_proposal(image, candidates, *, hand_count=5, title_cache=None):
    """Corroborate a one-to-five-card selected row using visible titles.

    A raised card can hide its right neighbour from the neural detector. Only
    that neighbour may lack a detection, and its own visible title is required.
    Such a recovered observation is never clickable. The caller must prove the
    exact SELECT label and independently resolve card identities before use.
    """
    import cv2
    import numpy as np

    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return None
    cards = tuple(sorted((c for c in candidates if c.box[1] >= 704), key=lambda c: c.box[0]))
    if hand_count == 1:
        selected_gray = (_hif_gray_title_plates(image, selected=True) if title_cache is None
                         else title_cache.gray(image, selected=True))
        if (len(selected_gray) == 1 and 180 <= selected_gray[0][2] <= 190
                and abs(selected_gray[0][0]-2-265) <= 8):
            if (_hif_unselected_title_plates(image) if title_cache is None else title_cache.unselected(image)):
                return (), 0
            px, py, _pw, ph = selected_gray[0]
            box = (px-2, py+ph+2-250, 190, 250)
            x, y, width, height = box
            # 899's gray selected body is absent in two neural frames. Its
            # own complete title and two body borders establish observation
            # geometry; callers must still read the independent SELECT label.
            # 901's pulsing halo is independently detected as both a full
            # rectangle and a narrow fragment. Group detections by ownership
            # of the proven body, not by individual fragment dimensions. The
            # envelope includes the existing halo top and SELECT label bottom;
            # each detection must mostly overlap the body and stay inside it.
            consistent = True
            for card in cards:
                cx, cy, cw, ch = card.box
                overlap = max(0, min(cx+cw,x+width)-max(cx,x)) * max(0, min(cy+ch,y+height)-max(cy,y))
                if not (card.detector_label in ('cards', 'useless', 'suggestions')
                        and cw > 0 and ch > 0 and x-8 <= cx and cx+cw <= x+width+8
                        and 838 <= cy and cy+ch <= 1152 and overlap >= cw*ch*.65):
                    consistent = False
                    break
            if consistent:
                try:
                    from agent.card_selection import CandidateBox
                except ModuleNotFoundError:
                    from card_selection import CandidateBox
                return (CandidateBox(slot=0, box=box, detector_label='hif_gray_observation',
                                     detector_score=0., clickable=False),), 0
            return (), 0
    gray_plates = _hif_gray_title_plates(image) if title_cache is None else title_cache.gray(image)
    adjusted_gray = []
    for card in cards:
        x, y, width, height = card.box
        matches = [plate for plate in gray_plates if height > 0 and 105 <= width <= 205
                   and -8 <= x-(plate[0]-2) <= max(8, 190-width+8)
                   and min(y+height, plate[1]+plate[3]+2)
                   - max(y, plate[1]+plate[3]+2-250) >= max(250*.75, height*.85)]
        if len(matches) > 1:
            return None
        if matches:
            plate = matches[0]
            card = replace(card, box=(plate[0]-2, plate[1]+plate[3]+2-250, 190, 250),
                           detector_label='hif_gray_observation', clickable=False)
        adjusted_gray.append(card)
    cards = tuple(adjusted_gray)
    detected_count = len(cards)
    if hand_count not in (1, 2, 3, 4, 5) or not 1 <= detected_count <= 8:
        return None
    anchors = ((265,) if hand_count == 1 else (157,374) if hand_count == 2 else (48, 265, 482) if hand_count == 3 else
               tuple(round(18 + i * 494 / (hand_count - 1)) for i in range(hand_count)))
    selected_gray_body = None
    selected_disabled = ()
    selected_half_disabled = False
    if hand_count in (2, 3, 4, 5):
        selected_gray = (_hif_gray_title_plates(image, selected=True) if title_cache is None
                         else title_cache.gray(image, selected=True))
        selected_disabled = _hif_selected_disabled_title_plates(image, title_cache=title_cache)
        if not selected_gray and len(selected_disabled) == 1:
            selected_gray = selected_disabled
        if (len(selected_gray) == 1 and 180 <= selected_gray[0][2] <= 190
                and sum(abs(selected_gray[0][0]-2-anchor) <= 8 for anchor in anchors) == 1
                and (selected_gray == selected_disabled or hand_count <= 3
                     or abs(selected_gray[0][0]-2-anchors[-1]) <= 8)):
            px, py, _pw, ph = selected_gray[0]
            selected_gray_body = (px-2, py+ph+2-250, 190, 250)
            selected_half_disabled = selected_gray == selected_disabled
            x, y, width, height = selected_gray_body
            # 1348: the selected gray body can be absent from every neural
            # frame. Reuse the singleton's complete title, both inset borders
            # and ownership envelope; all neighbours still pass the row proof.
            # 1666's rightmost selected gray card has the identical complete
            # pixel proof in a four-card row. It covers no right neighbour;
            # keep the existing overlapping title/fragment checks below.
            owned = []
            for card in cards:
                cx, cy, cw, ch = card.box
                overlap = max(0, min(cx+cw,x+width)-max(cx,x)) * max(0, min(cy+ch,y+height)-max(cy,y))
                if ((card.detector_label in ('cards', 'useless', 'suggestions')
                        or card.detector_label == 'hif_gray_observation' and card.box == selected_gray_body)
                        and cw > 0 and ch > 0 and x-8 <= cx and cx+cw <= x+width+8
                        and 838 <= cy and cy+ch <= 1152 and overlap >= cw*ch*.65):
                    owned.append(card)
            try:
                from agent.card_selection import CandidateBox
            except ModuleNotFoundError:
                from card_selection import CandidateBox
            cards = (*tuple(card for card in cards if card not in owned), CandidateBox(
                slot=0, box=selected_gray_body, detector_label='hif_gray_observation',
                detector_score=0., clickable=False,
            ))
    selected_plates = []
    bordered_selected_plates = []
    unselected_plates = ()
    separated_disabled_plates = ()
    if hand_count in (1, 2, 3, 4, 5):
        # Two/three-card rows can lose a left border in the detector (907's
        # selected center included). Five-card 933 splits the selected body;
        # four-card 953 also crops a neighbour inside its complete title.
        # Reuse the complete visible title to restore that footprint; a
        # shifted full-width card still fails.
        plates = _hif_unselected_title_plates(image) if title_cache is None else title_cache.unselected(image)
        if hand_count == 3:
            # Separated half-dark cards expose their entire independently
            # bordered title even while another card is raised.
            separated_disabled_plates = _hif_disabled_title_plates(
                image,cards,plates,title_cache=title_cache)
            plates = tuple(sorted(set((*plates,*separated_disabled_plates))))
            gray_plates = tuple(sorted(set((*gray_plates,*separated_disabled_plates))))
        unselected_plates = plates
        selected_low, _selected_high, selected_range = _bgr_channel_bounds(image[1065:1112])
        selected_mask=((selected_low>=210)&(selected_range<=24)).astype(np.uint8)
        _,selected_labels,selected_stats,_=cv2.connectedComponentsWithStats(selected_mask)
        selected_plates=[(int(x)-2,int(y)+1065+int(h)+2-250,190,250)
            for x,y,w,h,area in selected_stats[1:]
            if y==0 and 180<=w<=190 and 39<=h<=43 and area/(w*h)>=.65
            and any(abs(int(x)-2-anchor)<=8 for anchor in anchors)]
        title_candidates = list(selected_plates)
        if hand_count in (1, 2, 3, 4, 5):
            for label, (x,y,w,h,_area) in enumerate(selected_stats[1:],1):
                # The upper nine rows of this component can contain card art.
                # Its actual bottom 32-row title must independently satisfy
                # the same white-coverage gate, plus both body borders below.
                # All calibrated two-to-five-card rows share this title and
                # 190x250 body contract, including 1258's four-card row.
                if (y==0 and 180<=w<=190 and 39<=h<=43
                        and any(abs(int(x)-2-anchor)<=8 for anchor in anchors)
                        and np.count_nonzero(selected_labels[y+h-32:y+h,x:x+w]==label)/(w*32)>=.65):
                    box=(int(x)-2,int(y)+1065+int(h)+2-250,190,250)
                    if box not in title_candidates:
                        title_candidates.append(box)
        if hand_count in (1, 2, 3, 4, 5) and len(title_candidates) == 1:
            # Native 1006: the complete selected title and both physical body
            # borders stay fixed while the same contained neural crop loses
            # confidence. Reuse the singleton's independent 190-row border
            # proof; never promote its detector score or infer its identity.
            edge = _hif_canny_edges(image, title_cache)
            left = title_candidates[0][0]
            # The inner/outer contours alternate under stage lighting. Count
            # actual edge-bearing rows inside the already calibrated rim bands,
            # not rows sharing one fixed pair of columns. Each side still needs
            # 190 observed rows; the independent complete title fixes the body.
            bordered = all(_hif_observed_vertical_rim(image, edge, side) for side in (left, left+190))
            if not bordered:
                # 1642: the complete white title is 185px wide, while the
                # calibrated card footprint is 190px. Locate its right rim
                # from that title's actual right edge, keeping the same 2px
                # inset, +/-3px rim band and 190 observed rows on both sides.
                box = title_candidates[0]
                right_edges = [int(x+w)+2 for x,y,w,h,_area in selected_stats[1:]
                    if (int(x)-2,int(y)+1065+int(h)+2-250,190,250) == box
                    and 180<=w<=190 and abs(int(x+w)+2-left-190)<=4]
                bordered = (len(right_edges)==1
                    and np.any(edge[box[1]-2:box[1]+3,left+15:left+175]>0,axis=0).mean()>=.95
                    and all(_hif_observed_vertical_rim(image, edge, side) for side in (left,right_edges[0])))
            if bordered:
                bordered_selected_plates = title_candidates
                selected_plates = title_candidates
        if selected_gray_body is not None:
            if (title_candidates or not selected_half_disabled and len(plates) != hand_count-1
                    or hand_count <= 3 and any(not 180 <= plate[2] <= 190 for plate in plates)):
                return None
            edge = _hif_canny_edges(image, title_cache)
            selected_plates = [selected_gray_body]
            bordered_selected_plates = [selected_gray_body]
        adjusted = []
        selected_title_sources = []
        neighbour_title_sources = {}
        for raw_card in cards:
            card = _hif_restore_cropped_unraised_body(image, raw_card, plates, title_cache)
            x,y,width,height = card.box
            if hand_count == 4 and card.detector_label == 'suggestions':
                #1206: a normal neighbour's glow extends nine pixels left
                # while its complete physical title stays fixed. Reuse the
                # unselected envelope before counting detector anchors.
                complete_plates = tuple(plate for plate in plates if 180 <= plate[2] <= 190
                    and any(abs(plate[0]-2-anchor) <= 8 for anchor in anchors))
                halo_matches = (_hif_complete_halo_title_matches(card, complete_plates)
                    if 866 <= y <= 874 and 270 <= height <= 286 else [])
                if not halo_matches and len(bordered_selected_plates) == 1:
                    halo_matches = _hif_bordered_neighbour_halo_matches(card, complete_plates,
                        anchors, bordered_selected_plates[0], edge)
                if len(halo_matches) > 1:
                    return None
                if halo_matches:
                    adjusted.append(replace(card, box=halo_matches[0]))
                    continue
            selected_matches=[box for box in selected_plates
                if len(selected_plates)==1 and card.detector_label=='cards'
                # 1698: an inset singleton uses the same complete title and
                # both physical borders, never a wider detector-anchor gate.
                and (hand_count != 1 or box in bordered_selected_plates)
                and (card.detector_score>=.80 or box in bordered_selected_plates)
                and 105<=width<=190
                and -8<=x-box[0]<=190-width+8
                and (852<=y<=866 and 240<=height<=262 and (
                        abs(y-box[1])<=4 and abs(y+height-box[1]-250)<=4
                        or box in bordered_selected_plates and box[0]<=x and x+width<=box[0]+box[2]
                        and box[1]<=y and y+height<=box[1]+box[3])
                     # 1385/1631: a narrow fragment may include light above
                     # the physical top. Its independently complete title,
                     # both sides and top rim own geometry, not detector y.
                     # Keep it inside the existing selected halo envelope.
                     # 1705: a wider inset detection also includes SELECT's
                     # lower padding. Its complete bordered body must contain
                     # the entire width before reusing this same envelope.
                     or box in bordered_selected_plates
                     and (width<180 or box[0]<=x and x+width<=box[0]+box[2])
                     and 240<=height<=286 and 838<=y and y+height<=1152
                     and max(0,min(x+width,box[0]+190)-max(x,box[0]))
                         *max(0,min(y+height,box[1]+250)-max(y,box[1]))>=width*height*.65
                     and np.any(edge[box[1]-2:box[1]+3,box[0]+15:box[0]+175]>0,axis=0).mean()>=.95)]
            if hand_count in (4, 5) and card.detector_label == 'suggestions':
                selected_matches = [box for box in selected_plates
                    if len(selected_plates)==1 and card.detector_score>=.80
                    and 180<=width<=210 and 838<=y<=846 and 270<=height<=286
                    and abs(x-box[0])<=8 and abs(y+14-box[1])<=4
                    and 8<=y+height-box[1]-250<=20]
            if (card.detector_label == 'suggestions'
                    and (hand_count in (1, 2, 3) or hand_count in (4, 5) and not selected_matches)):
                # 1011's recommendation halo includes SELECT below the body.
                # 1093's three-card row has the same independently proven
                # body inside a taller halo; use that body, not halo height.
                # The complete title and both edges prove the actual entity;
                # reuse the singleton halo ownership envelope, keeping scores.
                # 1248 splits that glow into overlapping weak/strong pieces.
                # 1679's four-card halo also owns this fully bordered body;
                # use the same ownership proof if the tight halo fit failed.
                # 1794's white singleton uses this same title/border proof;
                # its 288px halo must not become the physical title footprint.
                # A taller halo still needs the normal raised-host score.
                # The independently bordered title supplies geometry only.
                selected_matches = [box for box in bordered_selected_plates
                    if width>0 and height>=270
                    and (height<=286 or card.detector_score>=.80)
                    and box[0]-8<=x and x+width<=box[0]+box[2]+8
                    and 838<=y<=box[1] and box[1]+box[3]<=y+height<=1152
                    and max(0,min(x+width,box[0]+box[2])-max(x,box[0]))
                        *max(0,min(y+height,box[1]+box[3])-max(y,box[1]))>=width*height*.65]
            if len(selected_matches)==1:
                selected_title_sources.append((len(adjusted), card))
                adjusted.append(replace(card,box=selected_matches[0]))
                continue
            if hand_count == 5:
                adjusted.append(card)
                continue
            matches = [(px-2,py+ph+2-250,190,250) for px,py,pw,ph in plates
                if selected_gray_body is None and card.detector_label in ('cards','useless')
                and 140 <= width < 190 and 870 <= y <= 892 and 240 <= height <= 262
                and 180 <= pw <= 190 and any(abs(px-2-anchor)<=8 for anchor in anchors)
                and 8 < x-(px-2) <= 190-width+8
                and abs(y-(py+ph+2-250))<=4 and abs(y+height-(py+ph+2))<=4]
            if len(matches)>1:
                return None
            if matches and hand_count == 3:
                neighbour_title_sources.setdefault(matches[0], []).append((len(adjusted), raw_card))
            adjusted.append(replace(card,box=matches[0]) if matches else card)
        duplicates = set()
        for body, sources in neighbour_title_sources.items():
            if len(sources) == 1:
                index, fragment = sources[0]
                # Only a complete title with a distinct, left-anchored body
                # can consume its displaced duplicate before host counting.
                if _hif_is_anchored_body_fragment(fragment, cards, body):
                    duplicates.add(index)
        if len(selected_title_sources) > 1 and (hand_count == 4 or len(bordered_selected_plates) == 1):
            # Native 954: two overlapping detections each independently map
            # to the same complete selected title. Keep that one physical
            # footprint only when their union also spans its full width.
            ordered = sorted(selected_title_sources, key=lambda pair: pair[1].box[0])
            body = selected_plates[0]
            if (len(ordered) > 3
                    or abs(ordered[0][1].box[0]-body[0]) > 8
                    or abs(max(card.box[0]+card.box[2] for _,card in ordered)-body[0]-body[2]) > 8
                    or any(right.box[0] >= left.box[0]+left.box[2]
                           for (_,left),(_,right) in zip(ordered,ordered[1:]))):
                return None
            duplicates.update(index for index, _ in ordered[1:])
        cards = tuple(card for index, card in enumerate(adjusted) if index not in duplicates)
    if hand_count == 1:
        # A cropped recommendation halo can be the only detector output.
        # Require both independently visible full card edges before restoring
        # its footprint; exact SELECT and identity remain caller requirements.
        edges = _hif_canny_edges(image, title_cache)
        bordered=all(any(np.count_nonzero(np.any(edges[880:1095,x:x+2]>0,axis=1))>=190
                        for x in columns) for columns in (range(262,267),range(452,456)))
        halos=[c for c in cards if c.detector_label=='suggestions'
               and 261<=c.box[0]<=305 and 838<=c.box[1]<=846
               and 150<=c.box[2]<=210 and 450<=c.box[0]+c.box[2]<=470
               and 270<=c.box[3]<=286]
        bodies=[c for c in cards if c not in halos and abs(c.box[0]-265)<=8
                and 852<=c.box[1]<=866 and 140<=c.box[2]<=210 and 240<=c.box[3]<=262]
        if bordered and len(halos)==1 and len(bodies)<=1:
            cards=tuple(c for c in cards if c not in halos)
            if not bodies:
                cards=(*cards,replace(halos[0],box=(265,halos[0].box[1]+14,190,250)))
    # Native 663: the recommendation halo and its contained selected body can
    # both be detected. Only remove that duplicate UI halo; all neighbour
    # title plates and the independent SELECT recognition remain mandatory.
    bodies = [c for c in cards if c.detector_label != 'suggestions'
              and c.detector_score >= .80 and 140 <= c.box[2] <= 210
              and 852 <= c.box[1] <= 866 and 240 <= c.box[3] <= 262]
    if len(bodies) == 1 and (hand_count in (2,4)
            # Native 1046 also detects the same halo beside a three-card
            # selected body and its right fragment. Reuse the envelope only
            # when that body's complete title and both edges independently
            # prove its footprint on this frame.
            or hand_count in (3, 5) and bodies[0].box in bordered_selected_plates):
        bx,by,bw,bh = bodies[0].box
        duplicates = [c for c in cards if c.detector_label == 'suggestions'
            and 60 <= c.box[2] <= 210 and 838 <= c.box[1] <= 846 and 270 <= c.box[3] <= 286
            and any(abs(bx-anchor)<=12 and -8<=c.box[0]-anchor<=120
                    and abs(c.box[0]+c.box[2]-anchor-190)<=12 for anchor in anchors)
            and c.box[0]+c.box[2] >= bx+bw
            and c.box[1]+c.box[3] >= by+bh and abs(c.box[1]+14-by) <= 4]
        if 1 <= len(duplicates) <= 3:
            cards = tuple(c for c in cards if all(c is not duplicate for duplicate in duplicates))
    raised = [c for c in cards if c.box[2] >= 140
        and (c.detector_score >= (.70 if hand_count == 5 else .80)
             or c.box in bordered_selected_plates) and (
        852 <= c.box[1] <= 866 and 240 <= c.box[3] <= (268 if hand_count == 5 else 262)
        or c.detector_label == 'suggestions' and 838 <= c.box[1] <= 846 and 270 <= c.box[3] <= 286)]
    if len(raised) != 1:
        return None
    selected = raised[0]
    if separated_disabled_plates and selected.box not in bordered_selected_plates:
        return None
    selected_top = selected.box[1] + (14 if selected.box[3] >= 270 else 0)
    slots = [i for i, x in enumerate(anchors) if abs(selected.box[0] - x) <= (12 if hand_count in (2, 4) else 8)]
    if len(slots) != 1:
        return None
    selected_slot = slots[0]
    covered_slot = selected_slot + 1 if hand_count >= 4 and selected_slot < hand_count - 1 else None
    disabled_occluded = ()
    gray_fragments = ()
    if hand_count in (4, 5) and selected.box in bordered_selected_plates:
        # 1087: the raised body hides the grey neighbours' vertical borders.
        # Their dark titles plus the exposed top and thin inner bottom edges
        # still independently prove those physical cards on this same frame.
        # 1252 has the same exposed left neighbour in the calibrated five-row.
        gray_fragments = _hif_selected_gray_fragments(
            image, anchors, selected_slot, selected.box, unselected_plates=unselected_plates,
            title_cache=title_cache)
        disabled_occluded = _hif_selected_disabled_fragments(
            image, anchors, selected_slot, selected.box,
            unselected_plates, candidates=cards, title_cache=title_cache)
        occluded = tuple(sorted(set((*gray_fragments, *disabled_occluded))))
        gray_plates = tuple(sorted(set((*gray_plates, *occluded))))
        adjusted = []
        for card in cards:
            matches = [plate for plate in occluded
                       if card is not selected and 105 <= card.box[2] <= 205 and card.box[3] > 0
                       and min(card.box[1]+card.box[3],1134)-max(card.box[1],884)
                           >= max(250*.75,card.box[3]*.85)
                       and -8 <= card.box[0]-(plate[0]-2) <= 190-card.box[2]+8]
            if len(matches) > 1:
                return None
            if matches:
                card = replace(card,box=(matches[0][0]-2,884,190,250),
                               detector_label='hif_gray_observation',clickable=False)
            adjusted.append(card)
        cards = tuple(adjusted)
    # Count all independently proven grey neighbours before requiring neural
    # anchors. A selected halo can be the only detected body beside one other
    # card, while its bordered title exposes the remaining grey neighbour.
    # Keep the original minimum and the original number of neural detections;
    # neither reconstructed observations nor merged fragments increase it.
    if detected_count < max(1, hand_count - 1 - len(gray_plates)):
        return None
    supported = {selected_slot}
    for gx, _, _, _ in gray_plates:
        supported.update(i for i, x in enumerate(anchors) if abs(gx - 2 - x) <= 3)
    for card in cards:
        ordinary = 870 <= card.box[1] <= 892 and card.box[2] >= 105
        full_halo = (card.detector_label == 'suggestions' and 866 <= card.box[1] <= 874
                     and 180 <= card.box[2] <= 210 and 270 <= card.box[3] <= 286)
        if ordinary or full_halo:
            supported.update(i for i, x in enumerate(anchors)
                             if abs(card.box[0] - x) <= (16 if hand_count == 5 and i == 4 else 8))
    if len(supported) < min(3,hand_count):
        return None
    invalid = ((), selected_slot)
    strip = image[1100:1142].astype(np.int16)
    white = ((strip.min(axis=2) >= 210) & (np.ptp(strip, axis=2) <= 24)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(white)
    plates = [(int(x), int(w)) for x, y, w, h, area in stats[1:]
              if y == 0 and 65 <= w <= 190 and 30 <= h <= 34 and area / (w * h) >= .45]
    covered_body = None
    disabled_covered = [p for p in disabled_occluded if covered_slot is not None
                        and abs(p[0]-2-anchors[covered_slot]) <= 3]
    if len(disabled_covered) == 1:
        p = disabled_covered[0]
        covered_body = (anchors[covered_slot], p[1]+p[3]+2-250, 190, 250)
    right_plates = [px-2 for px,_py,pw,_ph in unselected_plates
                    if covered_slot is not None and covered_slot < hand_count-1
                    and abs(px-2-anchors[covered_slot+1]) <= 8 and 180 <= pw <= 190]
    exposed_right_plates = []
    if hand_count == 5 and covered_slot is not None and covered_slot < hand_count-2:
        # 1170: the next card is itself a middle card, so its complete visible
        # title ends at the following card instead of spanning 190 pixels.
        # Both calibrated boundaries must be present on this same frame.
        exposed_right_plates = [px-2 for px,_py,pw,_ph in unselected_plates
            if abs(px-2-anchors[covered_slot+1]) <= 3 and 105 <= pw < 180
            and abs(px+pw-anchors[covered_slot+2]-2) <= 3]
        right_plates.extend(exposed_right_plates)
    if (hand_count == 5 and selected.box in bordered_selected_plates
            and covered_slot is not None and covered_slot < hand_count-1
            and len(right_plates) == 1):
        # The raised card hides the left part of its right neighbour. Measure
        # only the exposure bounded by that proven body and the next title.
        left = selected.box[0]+selected.box[2]
        right = right_plates[0]
        if left < right:
            _,_,exposed_stats,_ = cv2.connectedComponentsWithStats(white[:,left:right])
            exposed = [(int(h),int(area)) for x,y,w,h,area in exposed_stats[1:]
                       if x == y == 0 and w == right-left and 30 <= h <= 34
                       and area/(w*h) >= .45]
            if len(exposed) == 1:
                height,_ = exposed[0]
                top = 1100+height+2-250
                # The exposed middle-card outline has an outer and inner
                # edge. Require every column within that five-pixel band;
                # do not infer a missing border from detector confidence.
                edge_bottom = top+3
                if np.all(np.any(edge[top-2:edge_bottom,left+2:right-2] > 0,axis=0)):
                    matches = [(int(x),int(w)) for x,y,w,h,_area in stats[1:]
                               if y == 0 and h == height and 65 <= w <= 190
                               and x <= left and x+w >= right
                               and 0 <= x-anchors[covered_slot] <= 20]
                    if len(matches) == 1:
                        if matches[0] not in plates:
                            plates.append(matches[0])
                        covered_body = (anchors[covered_slot],top,190,250)
    if (hand_count == 5 and covered_slot == hand_count-1
            and selected.box in bordered_selected_plates and covered_body is None):
        # A final neighbour has no following card to supply its right boundary.
        # Its own title edge and exposed outer rim provide that same evidence;
        # a detector clipped at the selected card's edge still needs one owner.
        titles = [p for p in unselected_plates if 0 <= p[0]-anchors[covered_slot] <= 20
            and abs(p[0]+p[2]+2-anchors[covered_slot]-190) <= 3
            and 105 <= p[2] <= 190 and 30 <= p[3] <= 34]
        if len(titles) == 1:
            px,py,pw,ph = titles[0]
            left, right = selected.box[0]+selected.box[2], px+pw
            top = py+ph+2-250
            if left < right:
                _,_,exposed_stats,_ = cv2.connectedComponentsWithStats(white[:,left:right])
                exposed = [row for row in exposed_stats[1:]
                    if row[0] == row[1] == 0 and row[2] == right-left and row[3] == ph
                    and row[4]/(row[2]*row[3]) >= .45]
                if (len(exposed) == 1
                        and np.all(np.any(edge[top-2:top+3,left+2:right-2] > 0,axis=0))
                        and np.count_nonzero(np.any(edge[top:py,right-1:right+5] > 0,axis=1)) >= 190):
                    covered_body = (anchors[covered_slot],top,190,250)
    plates.extend((x,w) for x,_,w,_ in gray_plates)
    if hand_count in (2,3) and selected.box in bordered_selected_plates:
        # In a separated row, complete titles with both physical sides are
        # stronger evidence than background-white fragments below a raised
        # card. Every neural detection must still acquire one body below.
        complete_titles = [p for p in unselected_plates if 180 <= p[2] <= 190
            and (p in separated_disabled_plates or _hif_exposed_unraised_body_edges(
                edge,p[0]-2,p[0]+173,both_sides=True,
                body_top=p[1]+p[3]+2-250 if separated_disabled_plates else None))]
        if (len(complete_titles) == hand_count-1 and len(complete_titles) == len(unselected_plates)
                and all(sum(abs(p[0]-2-anchor)<=8 for p in complete_titles)==1
                        for slot,anchor in enumerate(anchors) if slot != selected_slot)):
            plates = [(p[0],p[2]) for p in complete_titles]
    # This region's SELECT overlay cuts into title text. Its lower occupancy
    # gate does not admit identities; the complete four-plate mapping and
    # independent detector anchors must both agree.
    assignments = {}
    for x, width in plates:
        matches = [i for i, anchor in enumerate(anchors) if i != selected_slot
                   and 0 <= x - anchor <= 20 and x + width >= anchor + 100]
        if len(matches) != 1 or matches[0] in assignments:
            return invalid
        assignments[matches[0]] = (x, width)
    if set(assignments) != set(range(hand_count)) - {selected_slot}:
        return invalid
    if hand_count == 4 and selected.box in bordered_selected_plates:
        # 1230: an unselected recommendation halo can extend left of its
        # detector anchor while the next ordinary card clips its title.
        # The complete row above owns every title; the two adjacent title
        # boundaries and the exposed top/outer border prove this body.
        # Keep the existing envelope/edge gates and all raw detector scores.
        adjusted = []
        for card in cards:
            x,y,w,h = card.box
            matches = []
            if card.detector_label == 'suggestions' and 180 <= w <= 210 and 866 <= y <= 874 and 270 <= h <= 286:
                for slot,(px,pw) in assignments.items():
                    if slot == covered_slot or slot+1 not in assignments or not 105 <= pw < 180:
                        continue
                    next_x,_next_width = assignments[slot+1]
                    if (abs(px-2-anchors[slot]) > 3 or abs(next_x-2-anchors[slot+1]) > 3
                            or abs(px+pw-(next_x-2)) > 3):
                        continue
                    title = [p for p in unselected_plates if p[0] == px and p[2] == pw]
                    if len(title) != 1:
                        continue
                    _,py,_,ph = title[0]
                    left,top = px-2,py+ph+2-250
                    start,end = left+15,min(left+175,next_x-4)
                    overlap = max(0,min(x+w,left+190)-max(x,left)) * max(0,min(y+h,top+250)-max(y,top))
                    if (overlap >= w*h*.65 and end-start >= 100
                            and np.any(edge[882:889,start:end] > 0,axis=0).mean() >= .95
                            and any(np.count_nonzero(np.any(edge[880:1095,bx:bx+2] > 0,axis=1)) >= 190
                                    for bx in range(left-3,left+2))):
                        matches.append((left,top,190,250))
            if len(matches) > 1:
                return invalid
            adjusted.append(replace(card,box=matches[0]) if matches else card)
        cards = tuple(adjusted)
    if hand_count == 4 and selected.box in bordered_selected_plates and 0 in assignments:
        # 1223: the same screen-edge decoration as 1168 can accompany a
        # selected row. Discard it only after all titles have unique owners
        # and the remaining detections independently prove the complete row.
        edge_fragments = [card for card in cards
            if card.detector_label == 'useless' and card.box[0] == 0
            and 0 < card.box[2] <= assignments[0][1] / 3
            and 878 <= card.box[1] <= 890 and 240 <= card.box[3] <= 260
            and 1124 <= card.box[1] + card.box[3] <= 1142]
        if len(edge_fragments) == 1:
            bodies = tuple(card for card in cards if card is not edge_fragments[0])
            complete = hif_selected_row_proposal(image, bodies, hand_count=hand_count, title_cache=title_cache)
            return complete if complete and len(complete[0]) == hand_count and complete[1] == selected_slot else invalid
    if covered_body is not None:
        bx,by,bw,bh = covered_body
        adjusted = []
        for card in cards:
            x,y,w,h = card.box
            if (card.detector_label == 'cards' and bx <= x <= left and right <= x+w <= bx+bw
                    and selected_top <= y < by and 1100 <= y+h <= by+bh
                    and 105 <= w <= 210 and 240 <= h <= 268):
                card = replace(card,box=(x,by,w,bh))
            adjusted.append(card)
        cards = tuple(adjusted)
    gray_slots={slot for slot,plate in assignments.items() if plate in {(x,w) for x,_,w,_ in gray_plates}}
    ordinary_hosts = {}
    bordered_ordinary_hosts = set()
    if hand_count in (2, 3, 4, 5):
        for slot, anchor in enumerate(anchors):
            if slot == selected_slot:
                continue
            if hand_count in (4, 5) and (slot != hand_count-1 or slot == covered_slot or not any(
                    abs(px-2-anchor) <= 8 and 180 <= pw <= 190
                    for px, _py, pw, _ph in unselected_plates)):
                # Native 1052/1067 split the fully exposed rightmost card. Only
                # its own complete title may enable the existing short-hand
                # overlap rule; covered or partial titles retain the old gate.
                continue
            separated_halo = hand_count in (2,3) and any(
                c.detector_label == 'suggestions' and c.box[2] > 0
                and anchor-8 <= c.box[0] and c.box[0]+c.box[2] <= anchor+198
                for c in cards)
            # Keep the gray-selected host rule (1348) and every existing
            # detector host. Overlapping fragments can supply no unique host;
            # their complete title and physical borders can own the one body.
            host_minimum = 105 if selected_gray_body is not None else 140
            hosts = [c for c in cards if abs(c.box[0] - anchor) <= 8
                     and host_minimum <= c.box[2] <= 210 and 866 <= c.box[1] <= 892]
            unresolved_separated_host = (hand_count in (2,3)
                and selected_gray_body is None and len(hosts) != 1)
            if (slot != covered_slot and selected.box in bordered_selected_plates
                    and (separated_halo or unresolved_separated_host
                         or hand_count in (4,5) and slot == hand_count-1)):
                bodies = [(px-2,py+ph+2-250,190,250)
                          for px,py,pw,ph in unselected_plates
                          if abs(px-2-anchor) <= 8 and 180 <= pw <= 190
                          and _hif_exposed_unraised_body_edges(
                              edge,px-2,px+173,both_sides=True,body_top=py+ph+2-250)]
                if len(bodies) > 1:
                    return invalid
                if bodies:
                    # A halo or narrow left fragment cannot define the physical
                    # body's size. Keep only its independently proven
                    # geometry here; every original detection still needs one
                    # unique slot below, and an empty group remains invalid.
                    ordinary_hosts[slot] = bodies[0]
                    bordered_ordinary_hosts.add(slot)
                    continue
            # In the separated gray-selected row, keep the original neighbour
            # fragments and use the unselected-row host rule. Its full title
            # already proves the body; expanding a fragment first would create
            # two apparent hosts (1348). Every fragment still needs one owner.
            if not hosts and slot in gray_slots:
                continue
            if len(hosts) != 1:
                return invalid
            ordinary_hosts[slot] = hosts[0].box
    def physical_owners(card):
        x, y, width, height = card.box
        bodies = [selected.box if slot == selected_slot else
                  covered_body if slot == covered_slot else (anchor, 884, 190, 250)
                  for slot, anchor in enumerate(anchors)]
        return [slot for slot, (bx, by, bw, bh) in enumerate(bodies)
                if max(0, min(x+width, bx+bw)-max(x, bx))
                * max(0, min(y+height, by+bh)-max(y, by)) >= width*height*.65]

    groups = [[] for _ in anchors]
    edge_fragments = []
    for card in cards:
        x, y, width, height = card.box
        matches = []
        for slot, anchor in enumerate(anchors):
            if slot in bordered_ordinary_hosts:
                bx,by,bw,bh = ordinary_hosts[slot]
                overlap = (max(0,min(x+width,bx+bw)-max(x,bx))
                           * max(0,min(y+height,by+bh)-max(y,by)))
                if (width <= 0 or height <= 0 or not bx-8 <= x or x+width > bx+bw+8
                        or overlap < width*height*.65):
                    continue
                if card.detector_label in ('cards','useless') and any(
                        np.count_nonzero(np.any(edge[880:1095,b:b+2] > 0,axis=1)) >= 190
                        for boundary in (x,x+width)
                        if min(abs(boundary-bx),abs(boundary-bx-bw)) > 4
                        for b in range(boundary-3,boundary+2)):
                    continue
                if (card.detector_label == 'suggestions' and 866 <= y <= 874
                        and 270 <= height <= 286):
                    # A recommendation glow may be a contained fragment rather
                    # than a left-anchored detector. The complete title, both
                    # sides and envelope above must own it uniquely, like the
                    # other detections of this same unraised physical body.
                    matches.append(slot)
                    continue
            if slot == selected_slot:
                host = card is selected
                fragment = (0 < x - anchor < selected.box[2] and 60 <= width <= 130
                            and 180 <= max(x + width, selected.box[0] + selected.box[2]) - anchor <= 210
                            and (abs(y-selected_top)<=6 or hand_count==4
                                 and 852<=y<=866 and abs(y-selected_top)<=10
                                 # Native 933: a split right edge starts below
                                 # the selected body, while its complete title
                                 # independently proves the full footprint.
                                 or hand_count in (4,5) and len(selected_plates)==1
                                 and selected.box==selected_plates[0]
                                 and 0<=y-selected_top<=14 and 240<=height<=262
                                 and x+width<=anchor+194
                                 and (abs(y+height-selected_top-250)<=6
                                      # 1688: a split selected edge includes
                                      # lower padding. Its complete title and
                                      # both borders, plus the independently
                                      # exposed neighbour, establish ownership
                                      # within the existing outer envelope.
                                      or selected.box in bordered_selected_plates
                                      and (covered_body is not None
                                           # 1957: the partial neighbour title cannot
                                           # prove a full body. A contained raised
                                           # fragment starting left of that neighbour
                                           # still belongs to the fully bordered
                                           # selected card; every neighbour must pass
                                           # its own unchanged row proof below.
                                           or covered_slot is not None
                                           and x < anchors[covered_slot])
                                      and x+width<=selected.box[0]+selected.box[2]))
                            and abs(y + height - selected_top - 250) <= 18
                            # Complete neighbour titles and unique anchors
                            # disambiguate this contained fragment. Native
                            # 1182 also proves the selected title and both
                            # borders: its fragment may score above its body.
                            # 1904/1905's separated two-card row has the same
                            # complete title/border proof and unique owner.
                            and (hand_count in (1,3,4)
                                 or hand_count in (2,5) and selected.box in bordered_selected_plates
                                 or card.detector_score < selected.detector_score))
                if host or fragment:
                    matches.append(slot)
            else:
                x_valid = (-8 <= x - anchor <= (100 if hand_count == 4 and slot == covered_slot
                                               else 60 if slot == covered_slot
                                               else 16 if hand_count == 5 and slot == 4 else 8))
                # The partly covered right neighbour's top can track the
                # selected card by four pixels; its independent title plate
                # and unique slot mapping above still prove its presence.
                ordinary = (866 if slot == covered_slot else 870) <= y <= (900 if slot in gray_slots else 892) and 240 <= height <= (268 if hand_count == 5 and slot == covered_slot else 262)
                halo = card.detector_label == 'suggestions' and 866 <= y <= 874 and 270 <= height <= 286
                minimum_width = 75 if hand_count == 4 and slot == covered_slot else 105
                if x_valid and minimum_width <= width <= 210 and (ordinary or halo):
                    matches.append(slot)
                elif slot in ordinary_hosts:
                    hx, hy, hw, hh = ordinary_hosts[slot]
                    # Same bounded overlap rule as the unselected-row repair:
                    # preserve a complete title and account for every fragment.
                    if (60 <= width <= 190 and hx < x < hx + hw
                            and abs(y-hy) <= 8 and abs(y+height-hy-hh) <= 8
                            and 180 <= max(hx+hw, x+width)-anchor <= 200):
                        matches.append(slot)
        if (not matches and covered_body is not None
                and card.detector_label in ('cards', 'useless')
                and (105 <= width <= 210
                     # 1688 also crops this independently proven neighbour
                     # below the ordinary-body minimum. Reuse the existing
                     # fragment width only for its sole detector source,
                     # inside its actual horizontal body; never add a fragment
                     # beside another already-owned neighbour detection.
                     or selected.box in bordered_selected_plates and 60 <= width < 105
                     and covered_body[0] <= x and x+width <= covered_body[0]+covered_body[2]
                     and (sum(physical_owners(other) == [covered_slot] for other in cards) == 1
                          # A half-dark title has its own exposed outer side,
                          # top and bottom, beyond the selected body's edge.
                          # Its split detections still need unique physical
                          # ownership and the bounded group check below.
                          or len(disabled_covered) == 1))
                and min(866, selected.box[1]) <= y <= 892 and 240 <= height <= 268):
            # The complete exposed title and top border already prove this
            # neighbour's body. A displaced neural fragment must belong to
            # that body alone, using the existing bounded-overlap rule.
            if physical_owners(card) == [covered_slot]:
                matches.append(covered_slot)
        if (not matches and hand_count == 5 and selected_slot != 0
                and x == 0 and 0 < width <= 45 and 860 <= y <= 874 and 250 <= height <= 268
                and card.detector_label in ('cards','useless')):
            # A screen-clipped background fragment can straddle the first
            # card's outer rim. It supplies no card evidence; all five real
            # slots and that independent rim must still be proved below.
            edge_fragments.append(card)
            continue
        if len(matches) != 1:
            return invalid
        groups[matches[0]].append(card)
    if edge_fragments:
        first_titles = [(px,py,pw,ph) for px,py,pw,ph in unselected_plates
                        if abs(px-2-anchors[0]) <= 3 and 110 <= pw <= 130
                        and py+ph+2 == 1134]
        if (len(edge_fragments) != 1 or len(groups[0]) != 1 or len(first_titles) != 1
                or not _hif_exposed_unraised_body_edges(edge,anchors[0],anchors[1]-2,
                    both_sides=False,body_top=884)):
            return invalid
    if any((not 1 <= len(group) <= 3 if hand_count in (2, 3) or i in ordinary_hosts else len(group) != 1)
           for i, group in enumerate(groups) if i not in {selected_slot, covered_slot, *gray_slots}):
        return invalid
    bordered_covered_body = False
    if covered_slot is not None and len(groups[covered_slot]) > 1:
        fragments = groups[covered_slot]
        if (hand_count == 4 and covered_slot == hand_count-1
                and selected.box in bordered_selected_plates):
            # 1281: the partly hidden final body may yield three overlapping
            # crops. Its own full title, exposed top and outer edge establish
            # the entity; every crop has already acquired a unique slot above.
            titles = [plate for plate in unselected_plates
                      if abs(plate[0]-2-anchors[covered_slot]) <= 3
                      and 180 <= plate[2] <= 190]
            if len(titles) == 1:
                px,py,_pw,ph = titles[0]
                bx,by = px-2,py+ph+2-250
                selected_right = selected.box[0]+selected.box[2]
                start,end = max(bx+15,selected_right+3),bx+175
                bordered_covered_body = bool(
                    end-start >= 100
                    and np.any(edge[882:889,start:end] > 0,axis=0).mean() >= .95
                    # The antialiased outer border can move between adjacent
                    # columns in 2006. Count distinct rows within the same
                    # calibrated five-pixel border band, not one fixed pair
                    # of columns; title, top and fragment ownership still
                    # independently establish the complete body.
                    and np.count_nonzero(np.any(edge[880:1095,bx+187:bx+192] > 0,axis=1)) >= 190
                    and all(bx-8 <= c.box[0] and c.box[0]+c.box[2] <= bx+198
                            and c.box[2] > 0 and c.box[3] > 0
                            and max(0,min(c.box[0]+c.box[2],bx+190)-max(c.box[0],bx))
                                *max(0,min(c.box[1]+c.box[3],by+250)-max(c.box[1],by))
                                >= c.box[2]*c.box[3]*.65
                            and not any(np.count_nonzero(np.any(edge[880:1095,b:b+2] > 0,axis=1)) >= 190
                                        for boundary in (c.box[0],c.box[0]+c.box[2])
                                        if min(abs(boundary-bx),abs(boundary-bx-190),
                                               abs(boundary-selected_right)) > 4
                                        for b in range(boundary-3,boundary+2))
                            for c in fragments))
        if hand_count == 5 and covered_body is not None:
            # The independently bounded body owns all detections, including
            # recommendation glow; their union cannot redefine its footprint.
            if not 2 <= len(fragments) <= 3 or any(
                    physical_owners(card) != [covered_slot] for card in fragments):
                return invalid
            bordered_covered_body = True
        else:
            if hand_count != 4 or not 2 <= len(fragments) <= (3 if bordered_covered_body else 2):
                return invalid
            ordered = sorted(fragments, key=lambda c: c.box[0])
            right = max(c.box[0] + c.box[2] for c in fragments)
            if (not 180 <= right - anchors[covered_slot] <= 210
                    or any(min(a.box[0]+a.box[2],b.box[0]+b.box[2])-b.box[0]
                           < min(a.box[2],b.box[2])*.25 or abs(a.box[1]-b.box[1]) > 6
                           for a,b in zip(ordered,ordered[1:]))):
                return invalid
    output = []
    for slot, group in enumerate(groups):
        if not group:
            if slot != covered_slot and slot not in gray_slots:
                return invalid
            host = replace(selected, detector_label='hif_visible_title', detector_score=0.0, clickable=False)
        else:
            host = max(group, key=lambda c: c.detector_score)
            if slot in bordered_ordinary_hosts or slot == covered_slot and bordered_covered_body:
                host = replace(host,clickable=all(c.clickable for c in group))
            if any(c.detector_label == 'useless' for c in group):
                host = replace(host, detector_label='useless')
        if slot in gray_slots:
            host=replace(host,detector_label='useless',clickable=False)
        top = selected.box[1] + (14 if selected.box[3] >= 270 else 0) if slot == selected_slot else 884
        left = anchors[slot]
        if (hand_count == 4 or separated_disabled_plates) and slot != selected_slot:
            # Preserve the physical top already proved by this title's full
            # height. A fixed 884 origin moves 33px-title artwork up one pixel.
            physical_tops = {py+ph+2-250 for px,py,pw,ph in (*unselected_plates, *gray_plates)
                             if abs(px-2-anchors[slot]) <= 3}
            if len(physical_tops) == 1:
                top = physical_tops.pop()
        if (hand_count == 4 and abs(slot-selected_slot) == 1 and slot in gray_slots and top == 885
                and any(abs(px-(left+2)) <= (1 if slot > selected_slot else 0) and py+ph+2-250 == top
                        for px,py,_pw,ph in gray_fragments)):
            # A recovered gray fragment can include one extra title row (1087).
            # Prefer the independently aligned outer and inner top contours.
            start, end = left+15, left+175
            if slot < selected_slot:
                end = min(end, selected.box[0]-2)
            else:
                start = max(start, selected.box[0]+selected.box[2]+1)
            if (end-start >= 100 and (edge[883,start:end] > 0).mean() >= .95
                    and np.any(edge[885:887,start:end] > 0, axis=0).mean() >= .95):
                top = 884
        if separated_disabled_plates:
            if slot == selected_slot:
                left = selected.box[0]
            else:
                physical_lefts = {px-2 for px,py,pw,ph in unselected_plates
                                  if abs(px-2-anchors[slot]) <= 3}
                if len(physical_lefts) != 1:
                    return invalid
                left = physical_lefts.pop()
        output.append(replace(host, slot=slot, box=(left, top, 190, 250)))
    return tuple(output), selected_slot


def _hif_selected_gray_fragments(image, anchors, selected_slot, selected_box, *, unselected_plates=(), title_cache=None):
    """Prove exposed grey neighbours using three independent pixel features."""
    import cv2
    import numpy as np

    strip = image[1100:1142].astype(np.int16)
    dark = ((strip.min(2)>=50)&(strip.max(2)<=105)&(np.ptp(strip,axis=2)<=24)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    edge = _hif_canny_edges(image, title_cache)
    body = image[1068:1077].astype(np.int16)
    bright = (body.min(2)>=230)&(np.ptp(body,axis=2)<=25)
    output = []
    for label, (x,y,width,height,area) in enumerate(stats[1:], 1):
        # Reuse the existing grey-title rule for sparse <=2px attachments.
        if y==0 and 180<=width<=192 and 35<=height<=36:
            component = dark[y:y+height,x:x+width]
            rows = np.flatnonzero(component.sum(1)>=width*.5)
            columns = np.flatnonzero(component.sum(0)>=height*.5)
            if (len(rows) and len(columns) and rows[0]==0
                    and 1<=height-rows[-1]-1<=2 and columns[0]<=2 and width-columns[-1]-1<=2):
                left,right,bottom = int(columns[0]),int(columns[-1]+1),int(rows[-1]+1)
                area = int(component[:bottom,left:right].sum())
                x,width,height = x+left,right-left,bottom
        if not (y==0 and 105<=width<=190 and 30<=height<=34):
            continue
        slots = [i for i,a in enumerate(anchors) if abs(int(x)-2-a)<=3 and abs(i-selected_slot)==1]
        if len(slots)!=1:
            continue
        slot = slots[0]
        if area/(width*height)<.75:
            # 1253: SELECT glow cuts the part of this grey title behind the
            # independently bordered selected card. Measure only its exposed
            # connected component, bounded by that card's actual left edge.
            visible_width = min(int(width), selected_box[0]-2-int(x))
            if not (slot == selected_slot-1 and 105<=visible_width<width
                    and np.count_nonzero(labels[:height,x:x+visible_width]==label)
                        /(visible_width*height)>=.75):
                continue
            width = visible_width
        left, right = anchors[slot]+15, min(anchors[slot]+175,anchors[slot+1] if slot+1<len(anchors) else 720)
        if slot < selected_slot:
            right = min(right,selected_box[0]-2)
        else:
            left = max(left,selected_box[0]+selected_box[2]+1)
        bottom_rows = np.flatnonzero(bright[:,left:right].mean(axis=1)>=.75)
        thin_bordered_bottom = False
        if slot < selected_slot and len(bottom_rows) == 1:
            # 1223's exposed inner bottom border is one bright pixel thick.
            # Require both surrounding edge contours and the independent
            # full left inner border; one bright background row is not proof.
            bottom_y = 1068 + int(bottom_rows[0])
            inner = image[912:1073,left-3:left+3].astype(np.int16)
            inner_bright = (inner.min(2)>=230)&(np.ptp(inner,axis=2)<=25)
            thin_bordered_bottom = (inner_bright.sum(axis=0).max() >= 115
                and all((edge[y,left:right]>0).mean()>=.75 for y in (bottom_y-1,bottom_y+1)))
            if not thin_bordered_bottom and inner_bright.sum(axis=0).max() >= 115:
                # 1394's single white inset row has only one Canny contour.
                # Reuse the complete title's separate outer bottom rim, as
                # for an overlapped unselected gray title. Its exposed dark
                # region, left inset and top still need independent proof;
                # the caller has already proved both selected-body borders.
                visible_right = min(int(x+width), selected_box[0]-2)
                visible_width = visible_right-int(x)
                bottom = 1100+int(height)+2
                thin_bordered_bottom = (visible_width >= 105
                    and np.count_nonzero(labels[:height,x:visible_right]==label)/(visible_width*height)>=.75
                    and np.any(edge[bottom-2:bottom+3,left:right]>0,axis=0).mean()>=.95)
        if (right-left<100 or np.any(edge[882:889,left:right]>0,axis=0).mean()<.95
                or len(bottom_rows)<2 and not thin_bordered_bottom):
            continue
        output.append((int(x),1100,int(width),int(height)))
    # A background can join the exposed title, while a one-pixel inner rim
    # has only one Canny contour. Reuse the gray card's complete connected
    # inset frame: the left side and bottom must belong to the same L shape.
    # The caller has independently proved both selected-body borders.
    slot = selected_slot-1
    if slot >= 0 and not any(abs(x-2-anchors[slot])<=3 for x,_,_,_ in output):
        inset = image[908:1080].astype(np.int16)
        inset_bright = ((inset.min(2)>=230)&(np.ptp(inset,axis=2)<=25)).astype(np.uint8)
        _, inset_labels, inset_stats, _ = cv2.connectedComponentsWithStats(inset_bright)
        spans = np.flatnonzero(np.diff(np.r_[0,dark[0],0]))
        proven = []
        for x, end in zip(spans[::2],spans[1::2]):
            right = min(int(end),selected_box[0]-2)
            width = right-int(x)
            if abs(int(x)-2-anchors[slot])>3 or not 105<=width<=190:
                continue
            title = labels[:,x:right]==labels[0,x]
            rows = np.flatnonzero(title.sum(1)>=width*.5)
            height = len(rows)
            if not 30<=height<=34 or not np.array_equal(rows,np.arange(height)):
                continue
            side = np.flatnonzero(title[:,0])
            closed_height = max(height,int(side[-1])+1 if len(side) else height)
            if closed_height-height>2 or closed_height>34:
                continue
            height = closed_height
            if title[:height].mean()<.75:
                continue
            borders = [label for label,(bx,by,bw,bh,area) in enumerate(inset_stats[1:],1)
                if 11<=bx-x<=15 and 2<=by<=7 and 156<=bh<=164
                # The same halo can truncate the title and inset differently.
                # Reuse the gray-title relation, while the independently
                # proved selected body still strictly bounds the inset.
                and (abs(right-(bx+bw))<=8 or 0<=right-(bx+bw)<=15)
                # The connected inset may include its final bright border
                # pixel; it must stay outside the proven selected body,
                # rather than outside the title's two-pixel inset margin.
                and bx+bw<=selected_box[0] and 350<=area<=1000 and area/(bw*bh)<=.10
                and (inset_labels[12:142,bx:bx+3]==label).sum(0).max()>=115
                and np.any(inset_labels[by+bh-2:by+bh,bx:bx+bw]==label,axis=0).mean()>=.75]
            left = anchors[slot]+15
            if (len(borders)!=1 or right-left<100
                    or np.any(edge[882:889,left:right]>0,axis=0).mean()<.95):
                continue
            proven.append((int(x),1100,int(width),int(height)))
        if len(proven)==1:
            output.extend(proven)
    slot = selected_slot+1
    if slot+1 < len(anchors) and not any(abs(x-2-anchors[slot])<=3 for x,_,_,_ in output):
        # A raised body and the next complete title bound the exposed part
        # of a middle grey card, just as for a covered white neighbour.
        neighbours = [p for p in unselected_plates
                      if abs(p[0]-2-anchors[slot+1])<=3 and 180<=p[2]<=190]
        if len(anchors) in (4, 5) and slot+2 < len(anchors):
            # 1606/1982: the next middle card is also overlapped. Its exposed
            # title and the following card's observed left boundary prove
            # the same right edge used by the covered-white-card path.
            # The gray exposure still needs all independent pixel checks.
            neighbours.extend(p for p in unselected_plates
                if abs(p[0]-2-anchors[slot+1])<=3 and 105<=p[2]<180
                and any(abs(q[0]-2-anchors[slot+2])<=3
                        # Component ends are exclusive; the last title pixel
                        # may occupy the rim band's inclusive rightmost column.
                        and q[0]-2-3 <= p[0]+p[2] <= q[0]-2+3+1
                        for q in unselected_plates))
        if len(neighbours)==1:
            px,_,_,height = neighbours[0]
            left,right = selected_box[0]+selected_box[2],px-2
            if left < right and 30<=height<=34:
                owner = labels[0,left]
                # 1690: the selected rim's antialiased final pixel may occupy
                # this geometric boundary. Reuse the inner-border region's
                # first interior pixel only when that boundary has no owner;
                # never skip a contradictory connected title component.
                title_left = left if owner else left+1
                owner = labels[0,title_left]
                middle = (left+right)//2
                # Text and neighbour glow need not be dark. The connected
                # grey background must still span both title margins across
                # the geometric left half of this independently bounded view.
                title_margins = (owner and np.all(labels[0,title_left:middle]==owner)
                                 and np.all(labels[height-1,title_left:middle]==owner))
                if not title_margins:
                    # The halo is bounded by the independently observed body
                    # and next title, not by a fixed pixel width or hand size.
                    # One connected title must span the lower margin and the
                    # remaining upper margin before the exposed midpoint.
                    # Full-width views retain their >=105px/.75 fill proof.
                    lower_owner = labels[height-1,left]
                    visible = np.flatnonzero(labels[0,left:middle] == lower_owner) if lower_owner else ()
                    if len(visible):
                        start = left+int(visible[0])
                        rim_owner = labels[0,left]
                        glow_start = left
                        if rim_owner and rim_owner != lower_owner:
                            rx,ry,rw,rh,ra = stats[rim_owner]
                            # Only the antialiased corner of the already proven
                            # raised rim may precede the glow, never another
                            # connected title or a conflicting card body.
                            if (rx >= left-1 and rx+rw == left+1 and ry == 0
                                    and rw <= 2 and rh <= 3 and ra <= 3):
                                glow_start += 1
                        glow = image[1100,glow_start:start]
                        title_margins = (glow_start < start < middle and len(glow) > 0
                            and np.all(glow.min(1) >= 50) and np.all(glow.max(1) > 105)
                            and np.all(labels[0,start:middle] == lower_owner)
                            and np.all(labels[height-1,left:middle] == lower_owner)
                            and (right-left < 105 or right-start >= 105
                                and (labels[:height,start:right-2] == lower_owner).mean() >= .75))
                bottom_rows = np.flatnonzero(bright[:,left+1:right].mean(1)>=.75)
                # A moving backdrop can erase the high-gradient outer contour
                # without changing the gray title's connected lower boundary.
                # Use that same current component's closed extent as an
                # alternative; the selected/next body, title margins, physical
                # top and bright inner rim remain independently required.
                title_owner = labels[height-1,middle-1]
                connected_bottom = bool(title_owner and np.mean(
                    (labels[height-1,left+2:right-2] == title_owner)
                    & (labels[height+4,left+2:right-2] != title_owner)) >= .95)
                if (title_margins and len(bottom_rows)
                        and np.any(edge[882:889,left+2:right-2]>0,axis=0).mean()>=.95
                        and (connected_bottom or np.any(
                            edge[1100+height-1:1100+height+5,left+2:right-2]>0,axis=0).mean()>=.95)
                        and (any((edge[1068+int(y)+1,left+1:right-2]>0).mean()>=.95
                                 for y in bottom_rows)
                             # The thin inner rim's contour may change rows.
                             # Keep the same observed bright band and .95
                             # coverage, as for the independent outer rims.
                             or np.any(edge[1068:1077,left+1:right-2]>0,axis=0).mean()>=.95)):
                    output.append((anchors[slot]+2,1100,right-anchors[slot]-2,height))
    return tuple(output)


def _hif_selected_disabled_title_plates(image, *, title_cache=None):
    """Complete raised half-dark body, using the selected white body's rim gate.

    SELECT's glow can cover the bottom rim. The complete title, top rim and
    both 190-row side rims independently locate the body, as for white cards.
    This establishes geometry only; colour normalization and identity follow.
    """
    import cv2
    import numpy as np

    low, high, spread = _bgr_channel_bounds(image[1073:1115])
    mask = ((low >= 105) & (high <= 128) & (spread <= 24)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    edge = _hif_canny_edges(image, title_cache)
    output = []
    for x, y, w, h, area in stats[1:]:
        if not (y == 0 and 180 <= w <= 190 and 30 <= h <= 34 and area/(w*h) >= .65):
            continue
        x, y, w, h = int(x), int(y)+1073, int(w), int(h)
        left, top = x-2, y+h+2-250
        if (not 852 <= top <= 866 or left < 3 or left+193 > image.shape[1]
                or np.any(edge[top-2:top+3,left+15:left+175] > 0,axis=0).mean() < .95
                or not all(np.count_nonzero(np.any(edge[top:y,side-3:side+3] > 0,axis=1)) >= 190
                           for side in (left,x+w+2))):
            continue
        output.append((x,y,w,h))
    return tuple(output)


def _hif_selected_disabled_fragments(image, anchors, selected_slot,
                                     selected_box, neighbours, *, candidates=(), title_cache=None):
    """Prove half-dark unavailable neighbours outside a proven selected body.

    Unlike the darker selected-neighbour style, these cards also darken their
    inset frame. Reuse the unavailable-title colour, fill and physical edges;
    the selected body's edge only bounds occlusion, never proves a gray side.
    """
    import cv2
    import numpy as np

    low, high, spread = _bgr_channel_bounds(image[1100:1142])
    mask = ((low >= 105) & (high <= 128) & (spread <= 24)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    edge = _hif_canny_edges(image, title_cache)
    # 1840: the covered neighbour is followed by another unavailable card.
    # Reuse the independently bordered, strong-owner proof from the ordinary
    # row before using that gray title as the covered card's right boundary.
    output = list(_hif_disabled_title_plates(
        image, candidates, neighbours, title_cache=title_cache))
    neighbours = tuple(sorted(set((*neighbours, *output))))
    for slot in (selected_slot-1, selected_slot+1):
        if not 0 <= slot < len(anchors):
            continue
        anchor = anchors[slot]
        if slot < selected_slot:
            # The unraised card's own left rim is exposed. Locate its title
            # start in pixels; the selected card supplies only the right cut.
            starts = [int(x) for x in np.flatnonzero(np.diff(np.r_[0, mask[0], 0]) == 1)
                      if abs(int(x)-2-anchor) <= 3]
            if len(starts) != 1:
                continue
            x = starts[0]
            left, right, side = x, selected_box[0]-2, x-2
            if not 105 <= right-left < 180:
                continue
            plate_x, plate_right = x, selected_box[0]
        else:
            # The next complete white title independently fixes the exposed
            # right rim; no title beyond the known row means no reconstruction.
            if slot+1 >= len(anchors):
                continue
            next_titles = [p for p in neighbours if abs(p[0]-2-anchors[slot+1]) <= 3
                           and (180 <= p[2] <= 190 or 105 <= p[2] < 180
                                and slot+2 < len(anchors)
                                and any(abs(q[0]-2-anchors[slot+2]) <= 3
                                        and abs(p[0]+p[2]-(q[0]-2)) <= 3
                                        for q in neighbours))]
            if len(next_titles) != 1:
                continue
            left, right = selected_box[0]+selected_box[2], next_titles[0][0]-2
            side = right
            plate_x, plate_right = anchor+2, right
            # Four-card spacing exposes 139px here (1840); five cards expose
            # less than 105px. Both ends still come from proven card bodies.
            if not 30 <= right-left < (180 if len(anchors) == 4 else 105):
                continue
        # Measure just the connected exposure. Glow/text may cross behind the
        # selected body, so the entire component's area is not a visible fill.
        titles = [(label, int(h)) for label, (_x, y, _w, h, _area) in enumerate(stats[1:], 1)
                  if y == 0 and 30 <= h <= 34
                  and (labels[:h, left:right] == label).mean() >= .65]
        if len(titles) != 1:
            continue
        _, height = titles[0]
        top, bottom = 1100+height+2-250, 1100+height+2
        start, end = left+2, right-2
        if (np.any(edge[top-2:top+3, start:end] > 0, axis=0).mean() < .95
                or np.any(edge[bottom-3:bottom+3, start:end] > 0, axis=0).mean() < .95
                or not any(np.count_nonzero(np.any(edge[top:1100, b:b+2] > 0, axis=1)) >= 190
                           for b in range(side-3, side+2))):
            continue
        output.append((plate_x, 1100, plate_right-plate_x, height))
    return tuple(sorted(set(output)))


def _bgr_channel_bounds(image):
    """Exact channel bounds/range, without repeated three-channel reductions."""
    import cv2
    import numpy as np

    if image.dtype != np.uint8:
        # Preserve the original signed conversion for non-capture inputs.
        signed = image.astype(np.int16)
        return signed.min(2), signed.max(2), np.ptp(signed, axis=2)
    blue, green, red = cv2.split(image)
    low = cv2.min(cv2.min(blue, green), red)
    high = cv2.max(cv2.max(blue, green), red)
    return low, high, cv2.subtract(high, low)


def _hif_gray_title_plates(image, *, selected=False, title_cache=None):
    """Corroborate gray titles with their separate thin, bright body borders.

    This yields observation geometry only. Card identity still comes from the
    shared embedding reader, and these geometry proposals never grant input.
    """
    import cv2
    import numpy as np

    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return ()
    offset = 27 if selected else 0
    strip = image[1100-offset:1142-offset]
    strip_low, strip_high, strip_range = _bgr_channel_bounds(strip)
    dark = ((strip_low >= 50) & (strip_high <= 105)
            & (strip_range <= 24)).astype(np.uint8)
    _, title_labels, title_stats, _ = cv2.connectedComponentsWithStats(dark)
    body = image[908-offset:1080-offset]
    body_low, _body_high, body_range = _bgr_channel_bounds(body)
    bright = ((body_low >= 230) & (body_range <= 25)).astype(np.uint8)
    if not selected:
        # Resampling a neutral rim can alternate its darkest channel across
        # 230 while its luminance and physical edge remain continuous. Only
        # existing Canny edges may corroborate those bright, neutral pixels;
        # the complete connected border and title checks below still apply.
        edge = _hif_canny_edges(image, title_cache)[908:1080]
        # The antialiased rim centre and its outer contour can be two pixels apart.
        near_edge = cv2.dilate(edge, np.ones((5,5),np.uint8)) > 0
        luminance = cv2.cvtColor(body.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        bright |= ((luminance >= 230) & (body_range <= 25) & near_edge).astype(np.uint8)
    _, border_labels, border_stats, _ = cv2.connectedComponentsWithStats(bright)
    plates = {}
    for title_label, (x, y, width, height, area) in enumerate(title_stats[1:], 1):
        if y == 0 and 180 <= width <= 192 and 35 <= height <= 36:
            # 897's middle frame connects thirteen outer-border pixels to
            # the same 188x33 title. Remove only <=2px sparse attachment on
            # each edge; the dense title and both independent body borders
            # must still satisfy their original checks below.
            component = dark[y:y+height, x:x+width]
            rows = np.flatnonzero(component.sum(1) >= width*.5)
            columns = np.flatnonzero(component.sum(0) >= height*.5)
            if (len(rows) and len(columns) and rows[0] == 0
                    and 1 <= height-rows[-1]-1 <= 2
                    and columns[0] <= 2 and width-columns[-1]-1 <= 2):
                left, right, bottom = int(columns[0]), int(columns[-1]+1), int(rows[-1]+1)
                area = int(component[:bottom,left:right].sum())
                x, width, height = x+left, right-left, bottom
        if not (y == 0 and 105 <= width <= 190 and 30 <= height <= 34
                and area / (width * height) >= .75):
            continue
        borders = [(label, bx, by, bw, bh)
                   for label, (bx, by, bw, bh, ba) in enumerate(border_stats[1:], 1)
                   if 11 <= bx-x <= 15 and 2 <= by <= 7 and 156 <= bh <= 164
                   # A neighbour's recommendation glow can truncate the
                   # title and body at slightly different horizontal edges.
                   # Their independent left edges and near-coincident right
                   # edges still describe the same visible card fragment.
                   and ((abs((bx+bw)-(x+width)) <= 8 and 350 <= ba <= 1000)
                        or (180<=width<=190 and 11<=(x+width)-(bx+bw)<=15
                            and 150<=bw<=170 and 350<=ba<=1400
                            and max(bright[12:142,bx+bw-3:bx+bw].sum(axis=0))>=115))
                   and ba / (bw*bh) <= .10
                   and max(bright[12:142, bx:bx+3].sum(axis=0)) >= 115]
        if len(borders) == 1:
            plates[(title_label, borders[0][0])] = (int(x), 1100-offset, int(width), int(height))
    # A moving background can join the dark title below its bottom edge
    # (1323 unselected, 1349 selected). Locate a complete title from its own
    # top span and the two independently visible inset body borders, not the
    # connected background's bounding box. The selected row uses the same
    # pixel proof at its already-calibrated offset; no pixels are erased.
    edges = np.flatnonzero(np.diff(np.r_[0, dark[0], 0]))
    spans = [(int(left),int(right),None) for left,right in zip(edges[::2],edges[1::2])]
    if not selected:
        # 1400's sparkle cuts the first row inside one complete component.
        # This alternative only proposes its bounds; it must pass the full
        # independent physical-body proof, never the title-only shortcut.
        top_spans = {(left,right) for left,right,_ in spans}
        spans.extend((int(x),int(x+width),label)
                     for label,(x,y,width,height,_area) in enumerate(title_stats[1:],1)
                     if y==0 and 180<=width<=190 and height>34
                     and (int(x),int(x+width)) not in top_spans)
    for left, right, component_label in spans:
        width = int(right-left)
        if not 180 <= width <= 190:
            continue
        matching = [(label, bx, by, bw, bh)
                    for label, (bx, by, bw, bh, area) in enumerate(border_stats[1:], 1)
                    if 11 <= bx-left <= 15 and 11 <= right-(bx+bw) <= 15
                    and 2 <= by <= 7 and 150 <= bw <= 170 and 156 <= bh <= 164
                    and 350 <= area <= 1400 and area/(bw*bh) <= .10
                    and max(bright[12:142,bx:bx+3].sum(axis=0)) >= 115
                    and max(bright[12:142,bx+bw-3:bx+bw].sum(axis=0)) >= 115]
        if len(matching) != 1:
            continue
        title = (dark[:, left:right] if component_label is None
                 else title_labels[:,left:right]==component_label)
        rows = np.flatnonzero(title.sum(axis=1) >= width*.5)
        height = len(rows)
        valid_title = (component_label is None and 30 <= height <= 34
                       and np.array_equal(rows, np.arange(height)))
        if selected:
            if not valid_title:
                continue
            # 1349's last title row is sparse but still touches its left side.
            # Retain that closed row using the existing <=2px sparse-edge
            # bound, total 30..34 height and full-title density. A continuing
            # sidewall or a distant background pixel cannot extend the title.
            side_rows = np.flatnonzero(np.any(title[:, [0, -1]], axis=1))
            closed_height = max(height, int(side_rows[-1])+1 if len(side_rows) else height)
            if closed_height-height > 2 or closed_height > 34:
                continue
            height = closed_height
        valid_title = (valid_title and title[:height].sum()/(width*height) >= .75
                # Both complete side margins must end at this bottom.
                # An extending sidewall or a second dense title is not
                # merely background attached to an otherwise closed title.
                and not title[height:, 0].any() and not title[height:, -1].any())
        owner = (int(title_labels[0, left]) if component_label is None else component_label,
                 matching[0][0])
        if not valid_title:
            if selected or owner in plates:
                continue
            # 1397/1399: clothing below a complete card adds dark rows or
            # side pixels. Establish its physical bottom independently of
            # that background: one full top, both outer sides, and the full
            # bottom at the calibrated 250px body height. The complete inset
            # frame must also be one connected component, with both sides.
            label,bx,by,bw,bh = matching[0]
            if (any((border_labels[12:142,x:x+3]==label).sum(0).max()<115
                    for x in (bx,bx+bw-3))
                    or np.any(border_labels[by+bh-2:by+bh,bx:bx+bw]==label,axis=0).mean()<.75):
                continue
            edge = _hif_canny_edges(image, title_cache)
            top_min,top_end = 1100+30+2-250,1100+34+2-250+1
            tops = np.flatnonzero((edge[top_min:top_end,bx:bx+bw]>0).mean(1)>=.95)
            if (len(tops)==2 and 2<=tops[1]-tops[0]<=3
                    and 30<=len(rows)<=34 and np.array_equal(rows,np.arange(len(rows)))
                    and top_min+(int(tops[0])+int(tops[1])+1)//2==1100+len(rows)+2-250):
                # A thin rim has two Canny contours (1512). At 540-to-720
                # resampling, a two-source-pixel rim can span three pixels
                # (1582). Its rounded centre must still match the complete
                # title, with the independent sides/bottom checked below.
                top = top_min+(int(tops[0])+int(tops[1])+1)//2
            elif len(tops)==1:
                top = top_min+int(tops[0])
            else:
                continue
            bottom = top+250
            # As in the overlapped-title proof below, the inner rim contour
            # can lie on the final title row (bottom-3) after resampling.
            if (np.any(edge[bottom-3:bottom+3,bx:bx+bw]>0,axis=0).mean()<.95
                    or any(not any(np.count_nonzero(np.any(edge[880:1095,x:x+2]>0,axis=1))>=190
                                   for x in range(side-3,side+2))
                           for side in (int(left)-2,int(left)-2+190))):
                continue
            height = bottom-1100-2
            interior_rows = rows[rows<height]
            if (not 30<=len(interior_rows)<=height or height-len(interior_rows)>2
                    or not np.array_equal(interior_rows,np.arange(len(interior_rows)))
                    or title[:height].mean()<.75):
                continue
        # Both proofs may describe one physical title (1324). Replace
        # its component bounds only when the title AND independent body
        # border have the same labels; nearby other cards remain distinct.
        if not selected or owner not in plates:
            # Preserve the already-proven selected footprint (899/901/1348).
            # The border-bounded path only fills missing selected evidence.
            plates[owner] = (int(left), 1100-offset, width, height)
    if not selected:
        # 1379/1380: an overlapped grey title joins the moving background
        # below its bottom. Its exposed inset frame and the next card's real
        # left edge prove the shared bottom independently of that component.
        # The full-width/two-sided proof above remains unchanged.
        pale = ((strip_low >= 210) & (strip_range <= 24)).astype(np.uint8)
        _, _, pale_stats, _ = cv2.connectedComponentsWithStats(pale)
        # Text can separate white islands inside one complete neighbouring
        # title. Measure its whole observed rectangle; the independent L frame,
        # neighbour side and physical bottom below still prove its ownership.
        neighbours = [(int(px), int(ph)) for px, py, pw, ph, _pa in pale_stats[1:]
                      if py == 0 and 105 <= pw <= 190 and 30 <= ph <= 34
                      and pale[py:py+ph, px:px+pw].mean() >= .65]
        edge = None
        partials = [(label, x, y, width, height, False)
                    for label, (x, y, width, height, _area) in enumerate(title_stats[1:], 1)]
        # 1387/1388: the rounded outer pixel and recommendation glow can
        # join background below a partial title. Its uninterrupted top span
        # still belongs to that component; the inset frame, real neighbour
        # and complete bottom below must independently prove its ownership.
        partials.extend((int(title_labels[0, left]), int(left), 0,
                         int(right-left), int(title_stats[title_labels[0, left], 3]), True)
                        for left, right in zip(edges[::2], edges[1::2])
                        if 105 <= right-left < 180)
        for label, x, y, width, height, top_span in partials:
            if not (y == 0 and width >= 105 and height > 34):
                continue
            matches = [(i, bx, by, bw, bh) for i, (bx, by, bw, bh, ba)
                       in enumerate(border_stats[1:], 1)
                       if 11 <= bx-x <= 15 and 2 <= by <= 7 and 156 <= bh <= 164
                       and (top_span or width >= 180 or abs(bx+bw-x-width) <= 8)
                       and 350 <= ba <= 1000
                       and ba/(bw*bh) <= .10
                       and max(bright[12:142, bx:bx+3].sum(axis=0)) >= 115
                       and np.any(bright[by+bh-2:by+bh, bx:bx+bw], axis=0).mean() >= .75]
            if len(matches) != 1:
                continue
            border_label, bx, _by, bw, _bh = matches[0]
            adjacent = [(px, ph) for px, ph in neighbours
                        if (105 <= px-x < 180 if top_span or width >= 180
                            else -2 <= px-2-x-width <= 15)
                        # The partial title can extend past its inset frame
                        # (1390/1393), but never through the independently
                        # visible outer edge of its overlapping neighbour.
                        and (not top_span or x+width <= px-2)
                        and 0 <= px-2-bx-bw <= 15]
            if len(adjacent) != 1:
                continue
            px, observed_height = adjacent[0]
            # 1383's lower background joins the component to another card.
            # The separately proven neighbour edge bounds this exposed title;
            # the connected component's far-right extent is not card geometry.
            if width >= 180:
                width = px-x
            title = dark[:, x:x+width]
            dense = np.flatnonzero(title.sum(axis=1) >= width*.5)
            if (not 30 <= len(dense) <= 34 or not np.array_equal(dense, np.arange(len(dense)))
                    or title[:len(dense)].mean() < .75):
                continue
            if title[observed_height:, 0].any():
                # A resampled grey title may differ from its white neighbour
                # or join background below. Its own complete physical top
                # must then prove the measured title height and 250px body;
                # the independent bottom and neighbour side below still apply.
                # Disconnected background below is not this title's tail.
                tail = np.flatnonzero(title_labels[len(dense):, x] == label)
                if (title[len(dense)-1, 0] or len(tail) > 2
                        or (len(tail) and tail[-1] >= 2)):
                    continue
                if edge is None:
                    edge = _hif_canny_edges(image, title_cache)
                top = 1100+len(dense)+2-250
                if max((edge[y, bx:bx+bw] > 0).mean() for y in range(top-1,top+2)) < .95:
                    continue
                observed_height = len(dense)
            if title[:observed_height].sum()/(width*observed_height) < .75:
                continue
            if edge is None:
                edge = _hif_canny_edges(image, title_cache)
            bottom, left, right = 1100+observed_height+2, int(bx), px-6
            if (right-left < 100
                    or not any(np.count_nonzero(np.any(edge[880:1095, b:b+2] > 0, axis=1)) >= 190
                               for b in range(px-5, px))):
                continue
            # Include the final title row (bottom-3): Canny may place the
            # inner transition there after resampling.
            if np.any(edge[bottom-3:bottom+3, left:right] > 0, axis=0).mean() < .95:
                # The neighbouring white title does not fix this gray card's
                # physical height after resampling (1620). Its complete top,
                # left side and bottom must independently establish exactly
                # one 250px body. Two rim contours are not two bodies: retain
                # only a top whose own bottom also passes the original proof.
                if not any(np.count_nonzero(np.any(edge[880:1095, b:b+2] > 0, axis=1)) >= 190
                           for b in range(int(x)-5, int(x))):
                    continue
                physical_heights = [h for h in range(30,35)
                    if (edge[1100+h+2-250, bx:bx+bw] > 0).mean() >= .95
                    and np.any(edge[1100+h-1:1100+h+5, left:right] > 0, axis=0).mean() >= .95
                    and 0 <= h-len(dense) <= 2 and title[:h].mean() >= .75]
                if len(physical_heights) != 1:
                    continue
                observed_height = physical_heights[0]
            plates.setdefault((label, border_label), (int(x), 1100, int(width), observed_height))
    return tuple(sorted(plates.values()))


def _hif_unselected_title_plates(image, *, gray_plates=None):
    """Independent visible title plates corroborate detector row completeness."""
    import cv2
    import numpy as np

    strip = image[1100:1142, :720, :3].astype(np.int16)
    mask = ((strip.min(axis=2) >= 210) & (strip.max(axis=2) - strip.min(axis=2) <= 24)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    plates = sorted((int(x), int(y)+1100, int(w), int(h)) for x,y,w,h,area in stats[1:]
                    if y == 0 and 105 <= w <= 190 and 30 <= h <= 34 and area/(w*h) >= .65)
    if gray_plates is None:
        gray_plates = _hif_gray_title_plates(image)
    plates = sorted((*plates, *gray_plates))
    return tuple(plates) if 1 <= len(plates) <= 5 else ()


def _hif_disabled_title_plates(image, candidates, neighbours, *, title_cache=None):
    """Prove a half-darkened unusable card separately from selected neighbours."""
    import cv2
    import numpy as np

    disabled = [card for card in candidates if card.detector_label == 'useless'
                and 0 < card.detector_score <= 1]
    if not disabled or not neighbours:
        return ()
    # 1801's unavailable cards darken the entire body, including the inset
    # frame. Their title is the half-intensity white plate, not the existing
    # dark selected-neighbour title with an independently bright inset frame.
    low, high, spread = _bgr_channel_bounds(image[1100:1142])
    mask = ((low >= 105) & (high <= 128) & (spread <= 24)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    edge = _hif_canny_edges(image, title_cache)
    result = []
    # A proven right-hand gray title can anchor the next gray card. Sorting
    # right to left makes this finite and never promotes an unverified island.
    for x, y, width, height, area in sorted(stats[1:], key=lambda row: int(row[0]), reverse=True):
        if not (y == 0 and 0 < width <= 190 and 30 <= height <= 34
                and area/(width*height) >= .65):
            continue
        left, top, bottom = int(x)-2, 1100+int(height)+2-250, 1100+int(height)+2
        owners = [card for card in disabled
                  if abs(card.box[0]-left) <= 8 and abs(card.box[1]-top) <= 4
                  and 105 <= card.box[2] <= 190 and 240 <= card.box[3] <= 260
                  and abs(card.box[1]+card.box[3]-bottom) <= 4]
        if width >= 180:
            # A short hand exposes the whole half-dark title and both body
            # rims. It needs no adjacent card to supply its right boundary.
            right = int(x)+int(width)+2
            start, end = left+15, left+175
            if (len(owners) == 1
                    and np.any(edge[top-2:top+3,start:end] > 0,axis=0).mean() >= .95
                    and np.any(edge[bottom-3:bottom+3,start:end] > 0,axis=0).mean() >= .95
                    # Count the complete proven body, including its title
                    # sides. An inset Support badge can cover the art-side
                    # rim, while the independently bounded title remains.
                    and all(any(np.count_nonzero(np.any(edge[top+5:bottom-5,b:b+2] > 0,axis=1)) >= 190
                                for b in range(side-3,side+2)) for side in (left,right))):
                result.append((int(x),1100,int(width),int(height)))
            continue
        # Complete bodies above prove their own four rims. An occluded body
        # still needs the original strong detector and neighbour evidence.
        owners = [card for card in owners if card.detector_score >= .95]
        next_titles = [plate for plate in sorted(set((*neighbours, *result)))
                       if 105 <= plate[0]-2-int(x) < 180
                       and int(x)+int(width) <= plate[0]+1
                       and plate[3] == height]
        if len(owners) != 1 or len(next_titles) != 1:
            continue
        right = next_titles[0][0]-2
        start, end = left+15, right-4
        # A neighbour's glow can split the gray background into narrow islands.
        # Its physical edge still proves the complete >=105px title; the final
        # two pixels belong to the neighbouring border, not this title's fill.
        # The title fixes the actual body top. A fixed screen band can count
        # moving background above it while omitting its last real edge rows.
        if (end-start < 100 or mask[:height,x:right-2].mean() < .65
                or np.any(edge[top-2:top+3,start:end] > 0,axis=0).mean() < .95
                or np.any(edge[bottom-3:bottom+3,start:end] > 0,axis=0).mean() < .95
                or any(not any(np.count_nonzero(np.any(edge[top:1100,b:b+2] > 0,axis=1)) >= 190
                               for b in range(side-3,side+2)) for side in (left,right))):
            continue
        result.append((int(x),1100,right-int(x),int(height)))
    return tuple(sorted(set(result)))


def hif_unselected_row_candidates(frame_shape: tuple[int, ...], candidates: Any, *, image=None, title_cache=None):
    """Reconcile split detections only when a complete unraised row is unique.

    A narrow right fragment can otherwise become an extra physical slot even
    in two/three-card hands. Each source detection must belong to exactly one
    singleton or overlapping group. Missing gray detections require both a
    visible title plate and a separate body border, never row spacing alone.
    The centered HIF row spacing disambiguates fragments from real neighbours.
    Production callers supply image pixels to corroborate every visible title
    plate, including cards missed by the detector. Omitting image is only for
    geometry diagnostics and cannot establish row completeness.
    Selected rows retain their independent SELECT-gated path.
    """
    if frame_shape[:2] != (1280, 720):
        return None
    cards = tuple(sorted((c for c in candidates if c.box[1] >= 704), key=lambda c: c.box[0]))
    gray_plates = (() if image is None else _hif_gray_title_plates(image) if title_cache is None
                   else title_cache.gray(image))
    if len(gray_plates) == 1:
        plate = gray_plates[0]
        if 180 <= plate[2] <= 190 and abs(plate[0] - 2 - 265) <= 8:
            # 898 detects a 122x128 character-region patch above the physical
            # singleton. Only short detections wholly above its independently
            # proven body may be discarded; overlaps still reject the row.
            body_top = plate[1] + plate[3] + 2 - 250
            cards = tuple(card for card in cards
                          if not (card.box[3] < 190
                                  and card.box[1] + card.box[3] < body_top))
    if not cards and len(gray_plates) == 1:
        plate = gray_plates[0]
        # A complete, centered gray singleton can be missed entirely by the
        # detector (887). Its independent title and body borders still permit
        # observation. The complete-row check below rejects other title plates;
        # identity remains subject to the shared embedding reader.
        if 180 <= plate[2] <= 190 and abs(plate[0] - 2 - 265) <= 8:
            try:
                from agent.card_selection import CandidateBox
            except ModuleNotFoundError:
                from card_selection import CandidateBox
            cards = (CandidateBox(
                slot=0, box=(plate[0]-2, plate[1]+plate[3]+2-250, 190, 250),
                detector_label='hif_gray_observation', detector_score=0., clickable=False,
            ),)
    if not 1 <= len(cards) <= 10:
        return None
    if gray_plates:
        adjusted_gray = []
        matched_plates = {}
        for card in cards:
            x, y, width, height = card.box
            matches = [plate for plate in gray_plates
                       if height > 0
                       and (min(y+height, plate[1]+plate[3]+2)
                            - max(y, plate[1]+plate[3]+2-250) >= max(250*.75, height*.85)
                            # 1605's dim card is detected as an interior body
                            # crop. Its independently proven gray title/body
                            # may own a portrait crop anchored at the same top,
                            # horizontally inside the body and above the title.
                            # Detector localization may straddle the measured
                            # top: apply the existing 8px tolerance both ways.
                            or (width <= height
                                and abs(y-(plate[1]+plate[3]+2-250)) <= 8
                                and y+height <= plate[1]
                                and plate[0]-2 <= x <= x+width <= plate[0]-2+190))
                       and 105 <= width <= 205
                       and -8 <= x-(plate[0]-2) <= max(8, 190-width+8)]
            if len(matches) > 1:
                return ()
            if matches:
                plate = matches[0]
                if plate in matched_plates:
                    previous, index = matched_plates[plate]
                    if previous is None:
                        return ()
                    # 1608 detects both the gray body and its interior crop.
                    # The independently proven body may own this one pair;
                    # duplicate full bodies or two short crops remain unknown.
                    body_bottom = plate[1]+plate[3]+2
                    full = [c for c in (previous, card)
                            if 878 <= c.box[1] <= 890 and 240 <= c.box[3] <= 260
                            and (1124 <= c.box[1]+c.box[3] <= 1142
                                 # Apply the existing title-proven bottom
                                 # padding rule before duplicate ownership;
                                 # otherwise it is reached too late (1620).
                                 or (252 < c.box[3] <= 260
                                     and abs(c.box[1]-(body_bottom-250)) <= 4
                                     and 4 <= c.box[1]+c.box[3]-body_bottom <= 18))]
                    if len(full) != 1:
                        return ()
                    host = full[0]
                    crop = card if host is previous else previous
                    cx, cy, cw, ch = crop.box
                    if (host.detector_label not in ('cards', 'useless')
                            or crop.detector_label not in ('cards', 'useless')
                            or not (cw <= ch < 240
                                    and abs(cy-(plate[1]+plate[3]+2-250)) <= 8
                                    and cy+ch <= plate[1]
                                    and plate[0]-2 <= cx <= cx+cw <= plate[0]-2+190
                                    and host.box[0] <= cx <= cx+cw <= host.box[0]+host.box[2])):
                        return ()
                    adjusted_gray[index] = replace(host,
                        box=(plate[0]-2, plate[1]+plate[3]+2-250, 190, 250),
                        detector_label='hif_gray_observation', clickable=False)
                    matched_plates[plate] = (None, index)
                    continue
                matched_plates[plate] = (card, len(adjusted_gray))
                card = replace(card, box=(plate[0]-2, plate[1]+plate[3]+2-250, 190, 250),
                               detector_label='hif_gray_observation', clickable=False)
            adjusted_gray.append(card)
        for plate in gray_plates:
            if plate not in matched_plates:
                adjusted_gray.append(replace(cards[0],
                    box=(plate[0]-2, plate[1]+plate[3]+2-250, 190, 250),
                    detector_label='hif_gray_observation', detector_score=0., clickable=False))
        cards = tuple(sorted(adjusted_gray, key=lambda card: card.box[0]))
    plates = (None if image is None else _hif_unselected_title_plates(image) if title_cache is None
              else title_cache.unselected(image))
    disabled_plates = (() if image is None else _hif_disabled_title_plates(
        image, cards, plates, title_cache=title_cache))
    if disabled_plates:
        plates = tuple(sorted((*plates, *disabled_plates)))
        cards = tuple(replace(card, clickable=False) if card.detector_label == 'useless'
                      and any(abs(card.box[0]-(plate[0]-2)) <= 8 for plate in disabled_plates)
                      else card for card in cards)
    if plates and len(plates) in (3,4,5):
        # 1213's fresh status preflight sees a recommendation glow two pixels
        # beyond the old envelope. Its complete title, top and both side edges
        # still prove the same physical card. 1274's five-card final halo is
        # shifted inside that body. Reuse the same independent body proof
        # before the shared row/SELECT checks; never use the glow as a baseline.
        edge = None
        adjusted_halos = []
        for card in cards:
            if (card.detector_label == 'suggestions' and card.box[3] >= 270
                    and not _hif_complete_halo_title_matches(card, plates)):
                if edge is None:
                    import cv2

                    edge = cv2.Canny(image,80,160)
                anchors = ((48,265,482) if len(plates) == 3 else
                           tuple(round(18+i*494/(len(plates)-1)) for i in range(len(plates))))
                matches = _hif_bordered_neighbour_halo_matches(
                    card, plates, anchors, None, edge)
                if len(matches) > 1:
                    return ()
                if matches:
                    card = replace(card, box=matches[0])
                # With no edge match, the separate-body proof below still applies.
            adjusted_halos.append(card)
        if len(plates) in (3,4,5) and len(cards) > len(plates):
            px,py,_pw,ph = plates[-1]
            body = (px-2,py+ph+2-250,190,250)
            left,top,_,_ = body
            # 1276 has two independently owned halos plus a right-side card
            # fragment, with no separate left-aligned body detection. The
            # complete title/top/both sides prove that body; every detection
            # consumed here must independently belong to exactly that body.
            # 1282's separated three-card row has the same full final body,
            # with a narrow halo and a body crop missing its first ten pixels.
            has_body = any(c.detector_label in ('cards','useless')
                           and abs(c.box[0]-left) <= 8 and abs(c.box[1]-top) <= 4
                           and 105 <= c.box[2] <= 190 and 240 <= c.box[3] <= 260
                           for c in cards)
            if not has_body and any(c.detector_label == 'suggestions'
                                    and c.box == body for c in adjusted_halos):
                import numpy as np

                if edge is None:
                    import cv2

                    edge = cv2.Canny(image,80,160)
                anchors = ((48,265,482) if len(plates) == 3 else
                           tuple(round(18+i*494/(len(plates)-1)) for i in range(len(plates))))
                owned = []
                for index,card in enumerate(cards):
                    x,y,w,h = card.box
                    fragment = (card.detector_label in ('cards','useless')
                                and 60 <= w <= 190 and 878 <= y <= 890
                                and 240 <= h <= 260 and 1124 <= y+h <= 1142)
                    halo = (card.detector_label == 'suggestions' and w > 0
                            and 866 <= y <= 874 and 270 <= h <= 286)
                    if (not (fragment or halo)
                            or not left-8 <= x or x+w > left+198
                            or not 866 <= y or y+h > top+270):
                        continue
                    if fragment and any(
                            np.count_nonzero(np.any(edge[880:1095,b:b+2] > 0,axis=1)) >= 190
                            for boundary in (x,x+w)
                            if min(abs(boundary-left),abs(boundary-left-190)) > 4
                            for b in range(boundary-3,boundary+2)):
                        continue
                    matches = _hif_bordered_neighbour_halo_matches(
                        card,plates,anchors,None,edge)
                    if matches == [body]:
                        owned.append(index)
                if len(owned) > 1:
                    first = owned[0]
                    grouped = replace(adjusted_halos[first],box=body,
                                      clickable=all(cards[i].clickable for i in owned))
                    adjusted_halos = [grouped if i == first else card
                                      for i,card in enumerate(adjusted_halos)
                                      if i == first or i not in owned]
        cards = tuple(adjusted_halos)
    if plates and len(plates) in (4, 5) and len(cards) == len(plates)+1:
        # Screen-clipped decoration can overlap the first real card. Reuse
        # one rule for the overlapping layouts: every title/body must exist
        # on this frame and the fragment must have no independent inner rim.
        import cv2
        import numpy as np

        edge = cv2.Canny(image, 80, 160)
        left = plates[0][0]-2
        top = plates[0][1]+plates[0][3]+2-250
        fragments = [card for card in cards
            if card.detector_label == 'useless' and card.box[0] == 0
            and left < card.box[2] <= plates[0][2]/2
            and card.box[2] < 105
            and 878 <= card.box[1] <= 890 and 240 <= card.box[3] <= 260
            and 1124 <= card.box[1]+card.box[3] <= 1142]
        if (len(fragments) == 1 and plates[0] not in (*gray_plates, *disabled_plates)
                and _hif_exposed_unraised_body_edges(edge, left, plates[1][0]-4,
                    both_sides=False, body_top=top)
                and not any(np.count_nonzero(np.any(edge[top:top+215,b:b+2] > 0,axis=1)) >= 190
                    for b in range(fragments[0].box[2]-3, fragments[0].box[2]+2))):
            remaining = tuple(card for card in cards if card is not fragments[0])
            complete = hif_unselected_row_candidates(frame_shape, remaining, image=image, title_cache=title_cache)
            return complete if complete and len(complete) == len(plates) else ()
    if plates and len(plates) == 4:
        if len(cards) == 5:
            # 1828: a narrow left piece precedes the final white body. Both
            # detections must belong to its independently complete physical
            # title/rims, with no separate inner edge. Preserve the ordinary
            # body's label/score; neither fragment nor row spacing proves it.
            import cv2
            import numpy as np

            edge = cv2.Canny(image, 80, 160)
            anchors = tuple(round(18+i*494/3) for i in range(4))
            px, py, pw, ph = plates[-1]
            body = (px-2, py+ph+2-250, 190, 250)
            merges = []
            if 180 <= pw <= 190 and plates[-1] not in (*gray_plates,*disabled_plates):
                for index in range(len(cards)-1):
                    fragment, host = cards[index:index+2]
                    x,y,w,h = fragment.box
                    hx,hy,hw,hh = host.box
                    if (fragment.detector_label not in ('cards','useless') or host.detector_label != 'cards'
                            or not 60 <= w < 105 or not 105 <= hw <= 190
                            or not body[0]-8 <= x < hx < x+w or hx+hw > body[0]+198
                            or not 180 <= max(x+w,hx+hw)-x <= 200
                            or any(not (878 <= c.box[1] <= 890 and 240 <= c.box[3] <= 260
                                        and abs(c.box[1]-body[1]) <= 4
                                        and abs(c.box[1]+c.box[3]-body[1]-250) <= 4)
                                   for c in (fragment,host))
                            or any(_hif_bordered_neighbour_halo_matches(c,plates,anchors,None,edge,
                                    use_title_body_top=True) != [body] for c in (fragment,host))
                            or any(np.count_nonzero(np.any(edge[body[1]:body[1]+215,b:b+2] > 0,axis=1)) >= 190
                                   for c in (fragment,host) for boundary in (c.box[0],c.box[0]+c.box[2])
                                   if min(abs(boundary-body[0]),abs(boundary-body[0]-190)) > 4
                                   for b in range(boundary-3,boundary+2))):
                        continue
                    combined = replace(host,box=body)
                    remaining = (*cards[:index],combined,*cards[index+2:])
                    complete = hif_unselected_row_candidates(frame_shape,remaining,image=image,title_cache=title_cache)
                    if complete and len(complete) == 4:
                        merges.append(complete)
            if merges:
                return merges[0] if len(merges) == 1 else ()
            # 1364's right-edge decoration overlaps the final body. Reuse
            # its complete title/top/both-side proof; a score or label alone
            # cannot remove an extra detection or supply the remaining row.
            import cv2
            import numpy as np

            edge = cv2.Canny(image, 80, 160)
            anchors = tuple(round(18+i*494/3) for i in range(4))
            px, py, _pw, ph = plates[-1]
            body = (px-2, py+ph+2-250, 190, 250)
            clipped = []
            for card in cards:
                x, y, w, h = card.box
                if (card.detector_label != 'useless' or not 0 < w <= plates[-1][2]/3
                        or not x < body[0]+190 < x+w or not 718 <= x+w <= 720
                        or not 878 <= y <= 890 or not 240 <= h <= 260
                        or not 1124 <= y+h <= 1142
                        or _hif_bordered_neighbour_halo_matches(card, plates, anchors, None, edge) != [body]
                        or any(np.count_nonzero(np.any(edge[880:1095,b:b+2] > 0,axis=1)) >= 190
                               for boundary in (x,x+w) for b in range(boundary-3,min(boundary+2,719)))):
                    continue
                clipped.append(card)
            if len(clipped) == 1:
                remaining = tuple(card for card in cards if card is not clipped[0])
                complete = hif_unselected_row_candidates(frame_shape, remaining, image=image, title_cache=title_cache)
                return complete if complete and len(complete) == 4 else ()
        # Native 937 detects the lower menu control as a fifth card. Ignore
        # only boxes wholly below the independently visible four-card row,
        # and only after that complete row passes all existing geometry gates.
        bottom = max(1134,max(py+ph+2 for _px,py,_pw,ph in plates))
        above_controls = tuple(card for card in cards if card.box[1] < bottom)
        if len(above_controls) != len(cards):
            complete = hif_unselected_row_candidates(frame_shape,above_controls,image=image,title_cache=title_cache)
            return complete if complete and len(complete) == 4 else ()
        if len(cards) in (5,6):
            # 1238/1240 add one or two interior fragments across overlapping
            # cards. Require one unique four-body row after their simultaneous
            # removal, then prove each fragment's ownership independently.
            # Neither detector confidence nor a wider union establishes this.
            from itertools import combinations

            import cv2
            import numpy as np

            edge = cv2.Canny(image,80,160)
            fragment_indices = []
            for index, fragment in enumerate(cards):
                x,y,w,h = fragment.box
                if (fragment.detector_label != 'cards' or not 60 <= w <= 130
                        or not 878 <= y <= 890 or not 240 <= h <= 260
                        or not 1124 <= y+h <= 1142
                        or any(abs(x-(p[0]-2)) <= 8 for p in plates)):
                    continue
                # A real additional body's independently visible side must
                # remain a contradiction, even if four other titles exist.
                if any(np.count_nonzero(np.any(edge[880:1095,b:b+2] > 0,axis=1)) >= 190
                       for boundary in (x,x+w) for b in range(boundary-3,boundary+2)):
                    continue
                fragment_indices.append(index)

            def has_independent_owner(fragment, complete):
                x,y,w,h = fragment.box
                owners = []
                for slot, body in enumerate(complete):
                    left,top,_,_ = body.box
                    overlap = (max(0,min(x+w,left+190)-max(x,left))
                        * max(0,min(y+h,top+250)-max(y,top)))
                    if overlap >= w*h*.65:
                        owners.append(slot)
                if len(owners) != 1 or owners[0] == 3:
                    return False
                slot = owners[0]
                px,py,pw,ph = plates[slot]
                next_x = plates[slot+1][0]
                left = px-2
                start,end = left+15,min(left+175,next_x-4)
                return (105 <= pw < 180 and abs(px+pw-(next_x-2)) <= 3
                        and end-start >= 100
                        and np.any(edge[882:889,start:end] > 0,axis=0).mean() >= .95
                        and any(np.count_nonzero(np.any(edge[880:1095,b:b+2] > 0,axis=1)) >= 190
                                for b in range(left-3,left+2)))

            redundant = []
            for indices in combinations(fragment_indices,len(cards)-4):
                remaining = tuple(card for index,card in enumerate(cards) if index not in indices)
                complete = hif_unselected_row_candidates(
                    frame_shape,remaining,image=image,title_cache=title_cache)
                if not complete or len(complete) != 4:
                    continue
                if all(has_independent_owner(cards[index],complete) for index in indices):
                    redundant.append(complete)
            if redundant:
                return redundant[0] if len(redundant) == 1 else ()
    if plates:
        adjusted_padding = []
        for card in cards:
            card = _hif_restore_cropped_unraised_body(image, card, plates, title_cache)
            x,y,w,h = card.box
            if card.detector_label in ('cards','useless') and 252 < h <= 268:
                matched = [py+ph+2 for px,py,pw,ph in plates
                    if abs(y-(py+ph+2-250)) <= 4
                    and -8 <= x-(px-2) <= max(8,190-w+8)
                    and 105 <= w <= 190 and 4 <= y+h-(py+ph+2) <= 18]
                if len(matched)==1:
                    card=replace(card,box=(x,y,w,matched[0]-y))
            adjusted_padding.append(card)
        cards=tuple(adjusted_padding)
    # Selected cards are outside this row. A recommendation halo is taller
    # than the physical card and must be corroborated by its visible plate.
    if any(not (878 <= c.box[1] <= 890 and 240 <= c.box[3] <= 260
                and 1124 <= c.box[1] + c.box[3] <= 1142)
           and not (c.detector_label == 'suggestions' and 866 <= c.box[1] <= 874
                    and 270 <= c.box[3] <= 286) for c in cards):
        return None
    if plates == ():
        return ()
    adjusted = []
    for card in cards:
        if card.detector_label == 'suggestions' and card.box[3] >= 270:
            if plates is None:
                return ()
            x,y,w,h = card.box
            matches = _hif_complete_halo_title_matches(card, plates)
            if not matches:
                # A pulsing recommendation can be a cropped duplicate of an
                # independently detected body. Require that body and its full
                # title footprint; a cropped halo alone cannot create a card.
                matches = [(px-2,py+ph+2-250,190,250) for px,py,pw,ph in plates
                    if ((8 < x-(px-2) < 120 and 60 <= w <= 190)
                        # A middle card's visible title is cropped by its
                        # neighbour, while its own halo can span more of the
                        # body. The unique stronger ordinary detection below
                        # must still establish that same physical entity.
                        or (-8 <= x-(px-2) <= 8 and 160 <= w <= 190
                            and (w < pw or 105 <= pw < 180)))
                    and x+w <= px-2+200 and 8 <= py+ph+2-250-y <= 20
                    and 8 <= y+h-(py+ph+2) <= 20
                    and sum(other.detector_label in ('cards','useless')
                        and other.detector_score >= .80 and other.detector_score > card.detector_score
                        and abs(other.box[0]-(px-2)) <= 8
                        and abs(other.box[1]-(py+ph+2-250)) <= 4
                        and 105 <= other.box[2] <= 190 and 240 <= other.box[3] <= 260
                        and abs(other.box[1]+other.box[3]-(py+ph+2)) <= 4
                        for other in cards) == 1]
            if len(matches) != 1:
                return ()
            card = replace(card, box=matches[0])
        adjusted.append(card)
    # The detector may emit both a recommendation halo and the card inside
    # it. Their independent title already fixes one physical footprint.
    replacements, consumed = {}, set()
    for index, card in enumerate(adjusted):
        if index in consumed:
            continue
        if card.detector_label == 'suggestions' and card.box[2:] == (190, 250):
            x, y, _, _ = card.box
            bodies = [(j, other) for j, other in enumerate(adjusted)
                      if j != index and j not in consumed and other.detector_label in ('cards', 'useless')
                      and abs(other.box[0]-x) <= 8 and abs(other.box[1]-y) <= 4
                      and 105 <= other.box[2] <= 190 and 240 <= other.box[3] <= 260
                      and abs(other.box[1]+other.box[3]-y-250) <= 4]
            if len(bodies) > 1:
                return ()
            if bodies:
                j, body = bodies[0]
                consumed.add(j)
                # 1161's pulsing recommendation produces both a full halo
                # and a cropped halo over the same independently seen body.
                # Each halo above already matched exactly one visible title;
                # merge only identical calibrated footprints with this unique
                # body. A second body or an unowned halo still rejects the row.
                consumed.update(j for j, other in enumerate(adjusted)
                                if j != index and other.detector_label == 'suggestions'
                                and other.box == card.box)
                card = replace(card, detector_label=body.detector_label,
                               detector_score=max(card.detector_score, body.detector_score),
                               clickable=card.clickable and body.clickable)
        replacements[index] = card
    cards = tuple(sorted((card for index, card in replacements.items() if index not in consumed),
                         key=lambda card: card.box[0]))
    solutions = []
    short_fragment_edge = edge if plates and len(plates) == 3 else None

    def visit(index, groups):
        nonlocal short_fragment_edge
        if len(groups) > 5:
            return
        if index == len(cards):
            count = len(groups)
            if not 1 <= count <= 5:
                return
            # 190px cards are centered with 28px gaps for short hands;
            # four/five cards share the same 18px side margins and overlap.
            step = 218.0 if count <= 3 else (684.0 - 190.0) / (count - 1)
            origin = (720.0 - 190.0 - (count - 1) * step) / 2
            if plates is None:
                if any(abs(c.box[0] - (origin + slot * step)) > 8
                       for slot, c in enumerate(groups)):
                    return
            else:
                if len(plates) != count:
                    return
                for slot, (card, plate) in enumerate(zip(groups, plates, strict=True)):
                    anchor = plate[0] - 2
                    # A detector can crop the left edge as well as the right.
                    # The independent complete title fixes the physical anchor;
                    # permit only the missing width inside that same footprint.
                    if (abs(anchor - (origin + slot * step)) > 8
                            or not -8 <= card.box[0] - anchor <= max(8, 190 - card.box[2] + 8)):
                        return
            solutions.append(tuple(
                replace(c, slot=slot, box=(plates[slot][0]-2, plates[slot][1]+plates[slot][3]+2-250, 190, 250))
                if plates is not None else replace(c, slot=slot)
                for slot, c in enumerate(groups)))
            return
        host = cards[index]
        x, y, w, h = host.box
        if 105 <= w <= 205:
            visit(index + 1, (*groups, host))
        for size in (2, 3):
            fragments = cards[index+1:index+size]
            if (host.detector_label == 'suggestions' or len(fragments) != size-1 or not 105 <= w <= 190
                    or (plates is None and (w < 140 or host.detector_score < .80))):
                continue
            if any(not (60 <= f.box[2] <= 190 and x < f.box[0] < x+w
                        and abs(y-f.box[1]) <= 8 and abs(y+h-f.box[1]-f.box[3]) <= 8
                        and (plates is not None or (f.box[2] < w and f.detector_score < host.detector_score))) for f in fragments):
                continue
            members = (host, *fragments)
            right = max(c.box[0]+c.box[2] for c in members)
            top = min(c.box[1] for c in members)
            bottom = max(c.box[1]+c.box[3] for c in members)
            label = 'useless' if any(c.detector_label == 'useless' for c in members) else host.detector_label
            combined = replace(host, box=(x,top,right-x,bottom-top), detector_label=label)
            if not 180 <= right-x <= 200:
                # 1243 detects two overlapping right-side pieces of one full
                # three-card body. A narrow union alone cannot prove a card;
                # reuse the complete title/top/both-side proof for every piece.
                if not plates or len(plates) != 3 or not 105 <= right-x < 180:
                    continue
                if short_fragment_edge is None:
                    import cv2

                    short_fragment_edge = cv2.Canny(image,80,160)
                proofs = [_hif_bordered_neighbour_halo_matches(
                    member,plates,(48,265,482),None,short_fragment_edge) for member in members]
                if (any(len(proof) != 1 for proof in proofs)
                        or any(proof[0] != proofs[0][0] for proof in proofs[1:])):
                    continue
                combined = replace(combined,box=proofs[0][0])
            visit(index + size, (*groups, combined))

    visit(0, ())
    return solutions[0] if len(solutions) == 1 else ()


def hif_select_label_rois(frame_shape: tuple[int, ...], candidates: Any):
    """Cover the label below the HIF row, including a raised selected card.

    The old card-relative crop ended inside SELECT when the card was raised.
    Use the unraised row baseline and the reviewed 190px full-card footprint;
    detected boxes can be narrower because their neighbours overlap them.
    """
    if frame_shape[:2] != (1280, 720) or len(candidates) not in (1, 2, 3, 4, 5):
        raise ValueError("HIF SELECT requires a calibrated three/four/five-card row")
    if tuple(card.slot for card in candidates) != tuple(range(len(candidates))):
        raise ValueError("HIF SELECT slots are not consecutive")
    if len(candidates)==1 and abs(candidates[0].box[0]-265)>8:
        raise ValueError('HIF one-card SELECT anchor is uncalibrated')
    if len(candidates) == 4 and any(abs(card.box[0] - round(18 + i * 494 / 3)) > 12
                                    for i, card in enumerate(candidates)):
        raise ValueError("HIF four-card SELECT anchors are uncalibrated")
    if len(candidates) == 3 and any(abs(card.box[0] - anchor) > 8
                                   for card, anchor in zip(candidates, (48, 265, 482))):
        raise ValueError("HIF three-card SELECT anchors are uncalibrated")
    if len(candidates) == 2 and any(abs(card.box[0] - anchor) > 8
                                   for card, anchor in zip(candidates, (157, 374))):
        raise ValueError("HIF two-card SELECT anchors are uncalibrated")
    bottoms = sorted(int(card.box[1] + card.box[3]) for card in candidates)
    baseline = bottoms[min(2, len(bottoms) - 1)]
    if (len(candidates)==1 and 852 <= candidates[0].box[1] <= 866
            and 240 <= candidates[0].box[3] <= 262):
        # The only card has no unraised neighbour. Use the same calibrated
        # center-slot label footprint; exact SELECT still proves selection.
        baseline=1134
    if not 1124 <= baseline <= 1140:
        raise ValueError("HIF SELECT row baseline is uncalibrated")
    # Recognition-only OCR needs the glyph area: wider crops include the
    # adjacent Japanese card title during the SELECT label's vertical pulse.
    rois = [(int(card.box[0]) + 45, baseline - 30, 96, 48) for card in candidates]
    if any(x < 0 or x + w > 720 for x, _y, w, _h in rois):
        raise ValueError("HIF SELECT crop exceeds frame")
    return rois


@dataclass(frozen=True)
class HifLiveRunContext:
    run_id: str
    revision: int
    idol: str
    plan: str
    max_stamina: int

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("HIF run identity is missing")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("HIF run revision is invalid")
        from .hif_profiles import supported_hif_identity

        supported_hif_identity(self.idol, self.plan)
        if type(self.max_stamina) is not int or not 1 <= self.max_stamina <= 100:
            raise ValueError("HIF maximum stamina is invalid")


def _hif_fourth_score_bottom_evidence(header, white, *, x_offset=0):
    """Exclude only independently outlined score glyphs from the bottom frame."""
    import cv2
    import numpy as np

    # Complete white interiors have a common baseline and an independently
    # dark outline around their full boundary. Neither the predicted score nor its digit
    # count is supplied to this geometric proof.
    roi = header[120:170, 434+x_offset:569+x_offset]
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        white[120:170, 434+x_offset:569+x_offset].astype(np.uint8))
    glyphs = [(index, row) for index, row in enumerate(stats) if index
              and 8 <= row[2] <= 30 and 24 <= row[3] <= 32
              and 10 <= row[1] <= 18 and 39 <= row[1]+row[3] <= 43 and row[4] >= 100]
    glyphs.sort(key=lambda pair: pair[1][0])
    if (not 1 <= len(glyphs) <= 9 or not 4 <= glyphs[0][1][0] <= 13
            or np.ptp([row[1] for _, row in glyphs]) > 2
            or np.ptp([row[1]+row[3] for _, row in glyphs]) > 1
            or any(not 1 <= right[0]-(left[0]+left[2]) <= 10
                   for (_, left), (_, right) in zip(glyphs, glyphs[1:]))):
        return None
    dark = roi.max(2) < 160
    outlines = []
    for index, _ in glyphs:
        mask = (labels == index).astype(np.uint8)
        boundary = (mask > 0) & ~(cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0)
        # Native 479x852 repeats source pixels unevenly when normalized. Its
        # complete outline therefore cannot occupy one fixed-distance ring.
        nearby_outline = cv2.dilate((dark & (mask == 0)).astype(np.uint8),
                                    np.ones((7, 7), np.uint8)) > 0
        coverage = float(nearby_outline[boundary].mean())
        if coverage < .95:
            return None
        outlines.append(coverage)
    # Bound exclusion to actual dark outline pixels near the verified glyphs,
    # then the two normalized pixels of their rasterized antialiasing. Measure
    # every remaining bottom pixel, never search for a convenient white part.
    text = np.isin(labels, [index for index, _ in glyphs]).astype(np.uint8)
    outline = dark & (cv2.dilate(text, np.ones((7, 7), np.uint8)) > 0)
    covered = cv2.dilate(((text > 0) | outline).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    visible = ~covered[43, 34:120]
    measured = white[163, 468+x_offset:554+x_offset]
    additional_roi = None
    if int(visible.sum()) < 16:
        # Include the omitted straight left part of the same bottom border,
        # beyond its rounded corner. Preserve every original sample and mask
        # only the independently outlined glyphs above, never a score value.
        visible = ~covered[43, 12:120]
        measured = white[163, 446+x_offset:554+x_offset]
        additional_roi = [446+x_offset, 163, 22, 1]
        if int(visible.sum()) < 16:
            return None
    return dict(coverage=float(measured[visible].mean()),
                visible_pixels=int(visible.sum()), roi=[434+x_offset, 120, 135, 50],
                additional_bottom_roi=additional_roi,
                glyph_boxes=[[int(x)+434+x_offset, int(y)+120, int(w), int(h)]
                             for _, (x, y, w, h, _) in glyphs],
                outline_coverage=outlines)


def _hif_four_player_panel_geometry(image, *, self_rank=4):
    """1708's fourth physical column, independent of score or rank text.

    This character uses a white expanded frame, not the old cyan theme. Both
    long sides, top/bottom segments and all three narrow opponents are needed.
    Only this observed physical layout is calibrated here.
    """
    import cv2
    import numpy as np

    if getattr(image, 'shape', ()) not in ((170, 720, 3), (1280, 720, 3)):
        return None
    if self_rank not in (2,4):
        return None
    second = self_rank == 2
    self_sides = (293,413) if second else (435,555)
    # The second-place badge overlaps the left top edge; its full ordinal is
    # read independently below. Require the complete unobscured right segment.
    top_left,top_right = (374,412) if second else (437,554)
    offset = -140 if second else 0
    order = ('opponent-1','self','opponent-2','opponent-3') if second else (
        'opponent-1','opponent-2','opponent-3','self')
    columns = ((225,285),(293,374),(423,483),(493,553)) if second else (
        (225,285),(295,355),(365,425),(435,515))
    header = image[:170]
    white = (header.min(2) > 235) & (np.ptp(header, axis=2) < 20)
    # These are the observed panel's opaque sides. Stage lights can create
    # arbitrary white columns outside it and erase its outer contrast without
    # changing the UI. Bind the known frame, not every white background stripe.
    white_sides = [float(white[45:125, x].mean()) for x in self_sides]
    if min(white_sides) < .90:
        return None
    # The supported 479x852 and 480x854 captures normalize to the same panel,
    # but rasterize its upper white edge into three and two rows respectively.
    # Require the complete observed band; averaging a fixed three-row crop
    # incorrectly counts the second raster's background as a missing border.
    top_rows = tuple(int(y) for y in np.flatnonzero(white[6:9, top_left:top_right].mean(1) >= .90) + 6)
    if top_rows not in ((6, 7, 8), (7, 8)):
        return None
    top = float(white[list(top_rows), top_left:top_right].mean())
    # y=164 is the antialiased outer edge and includes the moving backdrop.
    # The opaque y=163 span to the right of the score is the actual bottom
    # frame: require its complete visible core, not background-dependent alpha.
    bottom_roi = [468+offset, 163, 86, 1]
    bottom = float(white[163, 468+offset:554+offset].mean())
    bottom_text = None
    if bottom < .95:
        bottom_text = _hif_fourth_score_bottom_evidence(header, white, x_offset=offset)
        if bottom_text is None:
            return None
        bottom = bottom_text['coverage']
    if top < .90 or bottom < .95:
        return None
    edges = cv2.Canny(header, 80, 160) > 0
    panels = []
    for identity, (left, right) in zip(order,columns,strict=True):
        sides = [float(edges[48:135, x-2:x+3].any(1).mean()) for x in (left, right)]
        boundary_evidence = {}
        if identity == 'self':
            # Its left opaque frame is already directly observed above; an
            # identical white backdrop cannot supply a contrasting Canny edge.
            # The separate portrait/meter boundary at x=515 must still exist.
            sides[0] = white_sides[0]
        elif sides[1] < .90:
            # The opponent's translucent meter can match the moving backdrop.
            # Its independent portrait/meter boundary still proves this column,
            # just as the self column above uses its internal portrait edge.
            inner_right = right - 20
            boundary_evidence = dict(outer_right_coverage=sides[1],
                                     right_boundary_x=inner_right)
            sides[1] = float(edges[48:135, inner_right-2:inner_right+3].any(1).mean())
        elif .70 <= sides[0] < .90:
            # 1918's moving backdrop briefly hides part of the left edge.
            # Keep its observed majority and require BOTH separate right
            # boundaries. Never use the same inner edge for two missing sides.
            inner_right = right - 20
            boundary_evidence = dict(outer_left_coverage=sides[0],left_boundary_x=inner_right)
            sides[0] = float(edges[48:135, inner_right-2:inner_right+3].any(1).mean())
        if min(sides) < .90:
            return None
        panels.append(dict(physical_identity=identity, left=left, right=right,
                           side_coverage=sides, **boundary_evidence))
    return dict(physical_panels=panels, white_side_columns=[[x] for x in self_sides],
                white_side_coverage=white_sides, self_meter_width=40,
                top_coverage=top, bottom_coverage=bottom, top_rows=list(top_rows),
                bottom_roi=bottom_roi, bottom_text_occlusion=bottom_text)


@lru_cache(maxsize=4)
def _hif_fourth_rank_outline(raster_reference='1708'):
    import cv2

    filename = {'1708': 'hif_fourth_rank_1708.png', '1727': 'hif_fourth_rank_1727.png',
                '1817': 'hif_fourth_rank_1817.png', '1866': 'hif_fourth_rank_1866.png'}[raster_reference]
    image = cv2.imread(str(Path(__file__).with_name('assets') / filename))
    if image is None or image.shape != (52, 76, 3):
        raise ValueError('fourth rank reference unavailable')
    return image.max(2) < 100


def _locate_hif_fourth_rank(image):
    import cv2
    import numpy as np

    geometry = _hif_four_player_panel_geometry(image)
    if geometry is None:
        return None
    # The complete observed "4th" outline rejects an uncalibrated selected
    # rank preview or a reordered physical row. Never infer its rank from 0.
    outline = image[:52, 430:506].max(2) < 100
    if not outline.any():
        return None
    dilation = np.ones((3, 3), np.uint8)
    expanded_outline = cv2.dilate(outline.astype(np.uint8), dilation) > 0
    matches = []
    # All references mean the same complete 4th glyph. Frame-edge thickness
    # must not select a text reference; each full outline passes independently.
    for reference in ('1708', '1727', '1817', '1866'):
        template = _hif_fourth_rank_outline(reference)
        recall = float((template & expanded_outline).sum() / template.sum())
        precision = float((outline & (cv2.dilate(template.astype(np.uint8), dilation) > 0)).sum()
                          / outline.sum())
        matches.append((min(recall, precision), recall, precision, reference))
    _, recall, precision, reference = max(matches, key=lambda match: match[0])
    if min(recall, precision) < .975:
        return None
    return dict(accepted=True, value=['opponent-1', 'opponent-2', 'opponent-3', 'self'],
                confidence=min(recall, precision), self_rank=4, anchor_x=514,
                score_roi=[434, 120, 135, 50], reason='hif_fourth_rank_calibrated',
                layout='hif-four-player-fourth-v1', geometry=geometry,
                rank_glyph=dict(roi=[430, 0, 76, 52], recall=recall, precision=precision,
                                raster_reference=reference))


def _locate_hif_four_player_second_rank(context, image):
    """Observe the complete settled second-place layout, never a score guess."""
    if context is None:
        return None
    geometry = _hif_four_player_panel_geometry(image,self_rank=2)
    if geometry is None:
        return None
    from .inventory import _inventory_ocr

    roi = [286,0,88,52]
    rows = _inventory_ocr(context,image[:52,286:374].copy(),'HifFourPlayerSecondRank',only_rec=True)
    if (len(rows) != 1 or rows[0].text != '2nd' or not .95 <= rows[0].score <= 1
            or tuple(rows[0].box) != (0,0,88,52)):
        return None
    return dict(accepted=True,value=['opponent-1','self','opponent-2','opponent-3'],
        confidence=float(rows[0].score),self_rank=2,anchor_x=374,
        score_roi=[294,120,126,50],reason='hif_four_player_second_rank_observed',
        layout='hif-four-player-second-v1',geometry=geometry,
        rank_glyph=dict(roi=roi,text=rows[0].text,confidence=float(rows[0].score)))


def locate_hif_opening_self_rank(image: Any, *, context=None) -> dict[str, Any]:
    """Recognize reviewed three-player second/third and four-player fourth.

    The cyan expanded player's top band is wider than opponent strips. Require
    its unique calibrated band and both portrait-side borders. Only observed
    physical layouts are accepted; selected rank previews remain separate.
    """
    rejected = {"accepted": False, "value": None, "confidence": 0.0, "reason": "hif_rank_layout_unverified"}
    try:
        import numpy as np

        if image.shape[0] < 170 or image.shape[1] != 720:
            return rejected
        second = _locate_hif_four_player_second_rank(context,image)
        if second is not None:
            return second
        def cyan_region(top, bottom, left, right):
            # HSV is pixel-local; convert only the pixels consumed below,
            # retaining the exact original float formula and color gates.
            hue, saturation, value = _bgr_to_opencv_hsv(image[top:bottom, left:right])
            return (hue >= 85) & (hue <= 120) & (saturation >= 160) & (value >= 150)

        columns = np.flatnonzero(cyan_region(8, 20, 210, 490).sum(axis=0) >= 9) + 210
        bands: list[list[int]] = []
        for column in columns:
            if bands and int(column) == bands[-1][-1] + 1:
                bands[-1].append(int(column))
            else:
                bands.append([int(column)])
        expanded = [band for band in bands if len(band) >= 35]
        if len(expanded) != 1:
            return _locate_hif_fourth_rank(image) or rejected
        band = expanded[0]
        if 390 <= band[0] <= 396 and 442 <= band[-1] <= 446:
            rank, anchor, left, right = 3, 444, (367, 373), (443, 445)
            order, score_roi = ["opponent-1", "opponent-2", "self"], [365, 120, 135, 50]
        elif (320 <= band[0] <= 326 and 372 <= band[-1] <= 376
                # The unchanged first opponent anchors the physical row;
                # shifting a third-place crop is not a second-place layout.
                and len([part for part in bands if 264 <= part[0] <= 266
                         and 283 <= part[-1] <= 286 and 18 <= len(part) <= 22]) == 1):
            rank, anchor, left, right = 2, 374, (297, 303), (373, 375)
            # Keep the entire expanded score, ending before the third
            # opponent's independent narrow score column at x=423.
            order, score_roi = ["opponent-1", "self", "opponent-2"], [294, 120, 126, 50]
        else:
            return _locate_hif_fourth_rank(image) or rejected
        # Portrait overlaps interrupt the sides; the four reviewed frames
        # retain over 70% cyan coverage on each independently checked side.
        border_coverage = min(float(cyan_region(70, 125, left[0], left[1]).mean()),
                              float(cyan_region(70, 125, right[0], right[1]).mean()))
        if border_coverage < 0.70:
            return rejected
        return {
            "accepted": True,
            "value": order,
            "confidence": min(1.0, border_coverage / 0.9),
            "self_rank": rank,
            "anchor_x": anchor,
            "score_roi": score_roi,
            "reason": "hif_opening_third_rank_calibrated" if rank == 3 else "hif_second_rank_calibrated",
            "layout": "hif-three-player-opening-v1",
        }
    except (AttributeError, ImportError, TypeError, ValueError, IndexError):
        return rejected


def hif_blue_resource_delta(image, roi):
    """Isolate the complete blue plus/digits, without supplying a numeric value."""
    import cv2
    import numpy as np

    x,y,w,h=roi
    if getattr(image,'shape',())!=(1280,720,3) or roi not in (
            [664,200,56,44],[648,200,72,44],[544,240,88,45]):
        return None,None
    pixels=image[y:y+h+1,x:x+w].astype(np.int16)
    b,g,r=pixels.transpose(2,0,1)
    mask=(b>=220)&(g>=100)&(g<=230)&(b-g>=15)&(g-r>=35)
    count=int(mask[:-1].sum())
    if count<12:
        return None,None
    evidence=dict(accepted=False,color='blue',raw_red_pixel_count=count,red_present=True,
        reason='resource_delta_blue_evidence_invalid',components=[],component_count=0,roi=roi)
    # Stage blue can enter this ROI outside the floating white value bubble.
    # Require that independent bubble and keep only glyphs wholly inside it.
    white=((pixels.min(2)>=230)&(np.ptp(pixels,axis=2)<=20)).astype(np.uint8)
    _,_,white_stats,_=cv2.connectedComponentsWithStats(white)
    plates=[row for row in white_stats[1:] if 40<=row[2]<=w and 26<=row[3]<=40
            and row[4]>=500 and row[4]/(row[2]*row[3])>=.50]
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask[:-1].astype(np.uint8))
    complete = []
    for bx,by,bw,bh,_ in plates:
        parts=sorted([(i,row) for i,row in enumerate(stats) if i and row[4]>=8
            and bx<row[0] and by<row[1] and row[0]+row[2]<bx+bw
            and row[1]+row[3]<by+bh],key=lambda pair:pair[1][0])
        if not 2<=len(parts)<=4 or any(a==0 or c==0 or a+b>=w or c+d>=h
                for _,(a,c,b,d,_) in parts):
            continue
        index,(px,py,pw,ph,area)=parts[0]
        plus=labels[py:py+ph,px:px+pw]==index
        if (not 8<=pw<=18 or not 8<=ph<=18 or not 25<=area<=140
                or plus.sum(0).max()<ph*.8 or plus.sum(1).max()<pw*.8
                or plus.mean()>.65 or min(row[3] for _,row in parts[1:])<10):
            continue
        complete.append((bx,by,bw,bh,parts))
    # Stage lights can form a second white region. Only a unique independent
    # bubble enclosing the complete plus and digits supplies delta evidence.
    if len(complete)!=1:
        return None,evidence
    bx,by,bw,bh,parts=complete[0]
    output=np.full_like(image,255)
    keep=np.isin(labels,[i for i,_ in parts])
    output[y:y+h,x:x+w][keep]=0
    evidence.update(accepted=True,reason='resource_delta_blue_pixels_isolated',
        white_bubble=[int(bx+x),int(by+y),int(bw),int(bh)],
        component_count=len(parts),components=[dict(box=[x+int(a),y+int(c),int(b),int(d)],area=int(area))
            for _,(a,c,b,d,area) in parts])
    return output,evidence


def _hif_slot_title_plate(candidate, plates):
    x, y, width, height = candidate.box
    matches = [p for p in plates if abs(p[0] - (x + 2)) <= 1 and p[1] >= y + height - 36 and p[1] + p[3] <= y + height and p[2] <= width - 2]
    return matches[0] if len(matches) == 1 else None


def _hif_base_title(text):
    return "".join(text.split()).rstrip("+＋⁺")


def _hif_title_correction_view(prediction, hand_count):
    return (
        hand_count in (4, 5)
        and prediction.recognizer == "card_art_embedding_top5_hif_masked_v1"
        and prediction.view_mode != "full"
        or hand_count in (2, 3)
        and prediction.recognizer == "card_art_embedding_top5_live_partial_v1"
        and prediction.view_mode == "full"
        and prediction.visible_ratio == 1.0
    )


def _hif_weak_title_candidate(candidate, prediction, *, frame_id, minimum_margin, hand_count):
    """Only a current, calibrated weak artwork ranking earns title OCR."""
    scores, groups = prediction.top_k_scores, prediction.top_k_card_ids
    limit = min(0.05, minimum_margin)
    return (
        _hif_title_correction_view(prediction, hand_count)
        and candidate.visible
        and prediction.visible
        and candidate.box[2:] == (190, 250)
        and prediction.box == candidate.box
        and prediction.slot == candidate.slot
        and prediction.frame_id == frame_id
        and prediction.source_frames == (frame_id,)
        and prediction.accepted is False
        and prediction.reason in ("below_acceptance_threshold", "below_margin_threshold")
        and 2 <= len(scores) == len(groups) <= 5
        and all(groups)
        and all(math.isfinite(s) and -1 <= s <= 1 for s in scores)
        and all(a >= b for a, b in zip(scores, scores[1:]))
        and 0.70 <= scores[0] == prediction.confidence <= 1
        and 0 <= prediction.margin < limit
        and math.isclose(prediction.margin, scores[0] - scores[1], abs_tol=1e-6)
    )


def _hif_title_visual_match(prediction, family, minimum_margin):
    groups, scores = prediction.top_k_card_ids, prediction.top_k_scores
    matches = [i for i, group in enumerate(groups) if family.intersection(group)]
    rank = matches[0] if len(matches) == 1 else None
    compatible = (
        rank is not None and set(groups[rank]) == family and scores[rank] >= 0.70 and scores[0] - scores[rank] <= min(0.05, minimum_margin)
    )
    return rank, compatible


def hif_current_title_family(read_ocr, frame, row, prediction, *, frame_id, plan, minimum_margin, plates, catalog):
    """Read current same-slot family evidence without granting selection or resolving upgrade."""
    slot = prediction.slot
    if (
        getattr(frame, "shape", None) != (1280, 720, 3)
        or type(slot) is not int
        or not 0 <= slot < len(row)
        or tuple(card.slot for card in row) != tuple(range(len(row)))
        or not _hif_weak_title_candidate(row[slot], prediction, frame_id=frame_id, minimum_margin=minimum_margin, hand_count=len(row))
    ):
        return None, None
    plate = _hif_slot_title_plate(row[slot], plates)
    if plate is None:
        return None, None
    x, y, width, height = plate
    right = min(x + width, row[slot + 1].box[0]) if slot + 1 < len(row) else x + width
    width = right - x
    if not (0 <= x < right <= frame.shape[1] and width >= 105 and 0 <= y < y + height <= frame.shape[0]):
        return None, None
    crop = frame[y : y + height, x:right].copy()
    roi = (x, y, width, height)
    title_pixels = (slot, roi, crop.tobytes())
    results = read_ocr(crop, "HifInventoryCompleteFamilyTitle", only_rec=False)
    if not results:
        return None, title_pixels
    if len(results) != 1 or not _valid_ocr_result(results[0]) or not isinstance(getattr(results[0], "text", None), str):
        return None, None
    result = results[0]
    if result.score < 0.95:
        return None, title_pixels
    tx, ty, tw, th = result.box
    # A detected word touching the crop edges can be a truncated name.
    if not (0 < tx < tx + tw < width and 0 < ty < ty + th < height):
        return None, None
    title = _hif_base_title(result.text)
    names = {key: _hif_base_title(entry.name) for key, entry in catalog.entries.items() if entry.plan in ("free", plan)}
    matching_ids = {key for key, name in names.items() if title == name}
    if (
        not title
        or not catalog.resolve_family(tuple(matching_ids), plan=plan)
        or (width < 180 and any(name != title and name.startswith(title) for name in names.values()))
    ):
        return None, None
    groups, scores = prediction.top_k_card_ids, prediction.top_k_scores
    rank, compatible = _hif_title_visual_match(prediction, matching_ids, minimum_margin)
    return dict(
        scope="current_complete_title_family",
        accepted=compatible,
        reason="complete_base_title_and_visual_family_agree" if compatible else "current_title_visual_family_conflict",
        frame_id=prediction.frame_id,
        slot=slot,
        roi=roi,
        text=result.text,
        confidence=result.score,
        family=tuple(groups[rank]) if compatible else tuple(sorted(matching_ids)),
        upgrade_verified=False,
        visual_candidate_rank=rank,
        visual_score_gap=scores[0] - scores[rank] if rank is not None else None,
        original_prediction=prediction.to_dict(),
    ), title_pixels
