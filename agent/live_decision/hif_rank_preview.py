"""Selected 3→2 rank preview; physical HUD identities do not move.

The fixed digit reference is the complete 1/3/2 glyphs in native 1284 frame
cd487e1e1d8ccbec. It is recognition evidence, never a source of score values.
"""

import re
import math
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

import cv2
import numpy as np

PHYSICAL_ORDER = ('opponent-1', 'opponent-2', 'self')
DIGIT_ROIS = ((221, 0, 24, 37), (293, 4, 23, 37), (357, 0, 43, 51))
SCORE_ROIS = ((222, 132, 74, 33), (294, 132, 70, 33), (365, 120, 135, 50))


@dataclass(frozen=True)
class RankPreviewBinding:
    bundle: object
    selected_slot: int
    card_instance_id: str
    preview_score: int

    def matches(self, bundle, run):
        return (bundle is not None and run is not None
            and (bundle.run_id, bundle.revision) == (run.run_id, run.revision)
            and (bundle.run_id, bundle.revision, bundle.state)
                == (self.bundle.run_id, self.bundle.revision, self.bundle.state)
            and bundle.kernel is self.bundle.kernel)


def bind_rank_preview(bundle, contract, physical_order):
    """Only consume an already bound policy's deterministic HUD contract."""
    if bundle is None or not isinstance(contract, dict) or tuple(physical_order) != PHYSICAL_ORDER:
        return None
    slot = contract.get('selected_slot')
    score = contract.get('preview_score')
    if (type(slot) is not int or not 0 <= slot < len(bundle.state.deck.hand)
            or type(score) is not int or score <= bundle.state.score
            or (contract.get('run_id'), contract.get('revision')) != (bundle.run_id, bundle.revision)):
        return None
    ref = bundle.state.deck.hand[slot]
    definition = bundle.kernel.card_definition(bundle.state, ref)
    if contract.get('card_id') != (definition.canonical_card_id or definition.card_id):
        return None
    return RankPreviewBinding(bundle, slot, ref.instance_id, score)


@lru_cache(maxsize=2)
def _digit_templates(method):
    image = cv2.imread(str(Path(__file__).with_name('assets') / 'hif_rank_preview_132.png'))
    if image is None or image.shape != (52, 235, 3):
        raise ValueError('rank preview digit reference unavailable')
    return tuple(_digit_mask(image[y:y+h, x-215:x-215+w], method)
        for x, y, w, h in DIGIT_ROIS)


def _digit_mask(crop, method):
    if method == 'complete_edges':
        return cv2.Canny(crop, 30, 60) > 0
    # The badge is translucent: stage lighting adds edges but is not part of
    # the digit. Closing isolates its narrow dark outline from the surrounding
    # lighter fill/background. Keep the whole crop and compare both directions;
    # no observed mark is discarded merely because the reference lacks it.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((5, 5), np.uint8)) >= 12


def _digit_proof(image, method):
    glyphs = []
    dilation = np.ones((3, 3), np.uint8)
    for digit, roi, template in zip((1, 3, 2), DIGIT_ROIS, _digit_templates(method), strict=True):
        x, y, w, h = roi
        edges = _digit_mask(image[y:y+h, x:x+w], method)
        if not edges.any():
            return None
        recall = float((template & (cv2.dilate(edges.astype(np.uint8), dilation) > 0)).sum()/template.sum())
        precision = float((edges & (cv2.dilate(template.astype(np.uint8), dilation) > 0)).sum()/edges.sum())
        if min(recall, precision) < .975:
            return None
        glyphs.append(dict(digit=digit, roi=list(roi), edge_recall=recall, edge_precision=precision))
    return glyphs


def _arrow(image, roi, direction, *, large=None):
    x, y, w, h = roi
    pixels = image[y:y+h, x:x+w].astype(np.int16)
    white = ((pixels.min(2) >= 200) & (pixels.max(2)-pixels.min(2) <= 35)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    bodies, outside = [], []
    for index, row in enumerate(stats[1:], 1):
        bx, by, bw, bh, area = (int(value) for value in row)
        borders = [name for name, hit in (('left', bx == 0), ('top', by == 0),
            ('right', bx+bw == w), ('bottom', by+bh == h)) if hit]
        if borders:
            # A component clipped by the ROI is outside evidence, not a second
            # complete arrow. Connected-components separation is essential:
            # if it touches the actual arrow, no complete internal body remains.
            outside.append(dict(box=[x+bx, y+by, bw, bh], pixels=area, borders=borders))
        else:
            bodies.append((index, row))
    if len(bodies) != 1:
        return None
    index, (bx, by, bw, bh, area) = bodies[0]
    if not (0 < bx < bx+bw < w and 0 < by < by+bh < h):
        return None
    mask = labels[by:by+bh, bx:bx+bw] == index
    if large is None:
        large = direction == 'up'
    if large:
        if not (16 <= bw <= 20 and 21 <= bh <= 25):
            return None
    elif not (9 <= bw <= 13 and 11 <= bh <= 16):
        return None
    if direction == 'up':
        mask = mask[::-1]
    # In the down-oriented image, a narrow centered shaft precedes both
    # widening wings and a centered taper. An inverted arrow cannot pass.
    widths = mask.sum(1)
    half = bh//2
    wing_span = max(int(np.flatnonzero(row)[-1]-np.flatnonzero(row)[0]+1)
        for row in mask[half:] if row.any())
    if (not np.all((widths[:half] >= 1) & (widths[:half] <= max(4, bw//3)))
            or wing_span < bw*.9
            or widths[-1] > bw*.65
            or not mask[:half, bw//2-1:bw//2+2].any(axis=1).all()):
        return None
    return dict(direction=direction, box=[int(x+bx), int(y+by), int(bw), int(bh)],
        pixels=int(area), outside_components=outside)


def _arrow_diagnostic(image, roi, expected):
    # Size belongs to the physical ROI, not the requested direction. Otherwise
    # a clear small upward arrow could never contradict the small downward ROI.
    opposite = 'up' if expected == 'down' else 'down'
    forward = _arrow(image, roi, expected, large=expected == 'up')
    backward = _arrow(image, roi, opposite, large=expected == 'up')
    if forward is not None and backward is not None:
        return dict(expected_direction=expected, direction=None, status='ambiguous',
            candidate_proofs=[forward, backward])
    proof = forward or backward
    if proof is None:
        return dict(expected_direction=expected, direction=None, status='unresolved')
    return dict(proof, expected_direction=expected,
        status='matched' if forward is not None else 'contradictory')


def _physical_panels(image):
    """Prove the fixed portrait columns, independently of their rank text.

    The self portrait is 80px wide; its separate right-hand meter is not used
    as an empty-region anchor. Each fixed column is located by two long sides
    at its calibrated width. The bottom overlaps score text and translucent
    stage pixels; record that decorative edge without using it for identity.
    """
    edges = cv2.Canny(image[:170], 80, 160) > 0
    panels = []
    for label, (left, right) in zip(PHYSICAL_ORDER, ((225, 285), (294, 355), (365, 445)), strict=True):
        sides = [float(edges[48:155, x-2:x+3].any(1).mean()) for x in (left, right)]
        if min(sides) < .8:
            return None
        bottom = max((float(edges[y:y+2, left+3:right-3].any(0).mean()), y)
            for y in range(156, 164))
        panels.append(dict(physical_identity=label, left_boundary_band=[left-2, 48, 5, 107],
            right_boundary_band=[right-2, 48, 5, 107], width=right-left,
            side_coverage=sides, bottom_edge_coverage=bottom[0], bottom_edge_y=bottom[1]))
    return panels


def rank_preview_geometry(image):
    if getattr(image, 'shape', ()) not in ((170, 720, 3), (1280, 720, 3)):
        return None
    from .hif_visual import _hif_four_player_panel_geometry

    # The 1708 fourth-column frame is calibrated only without rank movement.
    # Its three opponent scores may never be consumed as this three-player
    # preview, even if a changed badge or stage color resembles the old theme.
    if _hif_four_player_panel_geometry(image) is not None:
        return None
    hsv = cv2.cvtColor(image[:170], cv2.COLOR_BGR2HSV)
    cyan = ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 120)
        & (hsv[:, :, 1] >= 160) & (hsv[:, :, 2] >= 150))
    coverage = min(float(cyan[70:125, 367:373].mean()), float(cyan[70:125, 443:445].mean()))
    if coverage < .70:
        return None
    panels = _physical_panels(image)
    if panels is None:
        return None
    method = 'complete_edges'
    glyphs = _digit_proof(image, method)
    if glyphs is None:
        # Preserve the original complete proof for faint outlines. The second
        # proof must independently establish every digit, never combine partial
        # strokes or scores from the two methods.
        method = 'complete_local_dark_outline'
        glyphs = _digit_proof(image, method)
    # Arrows describe rank movement, not physical HUD identity. The three
    # spatial columns and fresh scores establish that identity and preview rank.
    # An unreadable decoration is unknown; a complete opposite arrow remains
    # contradictory evidence and is rejected after checking the score relation.
    down = _arrow_diagnostic(image, (342, 0, 18, 27), 'down')
    up = _arrow_diagnostic(image, (416, 0, 26, 30), 'up')
    return dict(border_coverage=coverage, physical_panels=panels, digit_glyphs=glyphs,
        glyph_method=method, glyph_diagnostic=dict(matched=glyphs is not None,
            authorizes_identity=False, reason='reference_glyph_match' if glyphs is not None
            else 'rank_glyph_unverified_or_changed'), arrows=[down, up])


def _read_scores(context, image):
    mosaic = np.full((100, 720, 3), 255, np.uint8)
    for index, (x, y, w, h) in enumerate(SCORE_ROIS):
        mosaic[20:20+h, index*240+20:index*240+20+w] = image[y:y+h, x:x+w]
    node = 'Task275HifRankPreviewScores'
    detail = context.run_recognition(node, mosaic, pipeline_override={node: dict(
        recognition='OCR', roi=[0, 0, 720, 100], expected=r'^\d{1,9}$')})
    rows = getattr(detail, 'all_results', None)
    if not detail or detail.hit is not True or not isinstance(rows, (tuple, list)) or len(rows) != 3:
        return None
    scores = {}
    for row in rows:
        text, confidence, box = getattr(row, 'text', None), getattr(row, 'score', None), getattr(row, 'box', None)
        if (not isinstance(text, str) or re.fullmatch(r'\d{1,9}', text) is None
                or type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not .95 <= confidence <= 1 or not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(v) is not int for v in box)):
            return None
        x, y, w, h = box
        index = x//240
        if index not in (0, 1, 2) or index in scores or not (
                w > 0 and h > 0 and x+w <= (index+1)*240 and 0 <= y < y+h <= 100):
            return None
        scores[index] = dict(value=int(text), confidence=float(confidence), box=list(box))
    return [scores[i] for i in range(3)]


def read_rank_preview(context, image, *, binding, frame_id, selected_slots):
    rejected = dict(accepted=False, value=None, confidence=0., reason='hif_rank_preview_unverified')
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    if (not isinstance(binding, RankPreviewBinding) or not binding.matches(bundle, run)
            or selected_slots != (binding.selected_slot,)):
        return rejected
    # Recognition reuse is limited to identical frame evidence and the same
    # immutable business binding; it never supplies capture time or input proof.
    key = (frame_id, binding)
    cached = getattr(context, 'gkh_hif_rank_preview_cache', ())
    for old_key, result in cached:
        if old_key == key:
            return dict(result)
    geometry = rank_preview_geometry(image)
    if geometry is None:
        return rejected
    scores = _read_scores(context, image)
    if scores is None:
        return dict(rejected, reason='hif_rank_preview_scores_unresolved', geometry=geometry)
    first, other, own = (row['value'] for row in scores)
    if not bundle.state.score < other < own < first or own != binding.preview_score:
        return dict(rejected, reason='hif_rank_preview_score_order_or_exact_preview_conflict',
            scores=scores, baseline_score=bundle.state.score,
            exact_preview_score=binding.preview_score, geometry=geometry)
    if any(arrow['status'] == 'contradictory' for arrow in geometry['arrows']):
        return dict(rejected, reason='hif_rank_preview_arrow_score_contradiction',
            scores=scores, baseline_score=bundle.state.score,
            exact_preview_score=binding.preview_score, geometry=geometry)
    result = dict(accepted=True, value=list(PHYSICAL_ORDER), confidence=min(r['confidence'] for r in scores),
        physical_order=list(PHYSICAL_ORDER), self_rank=2, baseline_self_rank=3,
        rank_preview=True, anchor_x=444, score_roi=list(SCORE_ROIS[2]),
        reason='hif_selected_rank_preview_physical_order_verified',
        layout='hif-selected-third-to-second-preview-v1', frame_id=frame_id,
        selected_slot=binding.selected_slot, card_instance_id=binding.card_instance_id,
        baseline_score=bundle.state.score, exact_preview_score=binding.preview_score,
        scores=scores, geometry=geometry)
    context.gkh_hif_rank_preview_cache = (*cached[-2:], (key, result))
    return result
