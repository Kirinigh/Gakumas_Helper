"""Read an already-open HIF detail's count from its own green plate only.

This is auxiliary evidence for an otherwise complete, non-unique detail body.
It does not navigate, identify a card, or infer a count from candidate rules.
"""
import math

import cv2
import numpy as np

try:
    from agent.arena_winrate.customization_badge import measure_customization_badge
except ModuleNotFoundError:
    from arena_winrate.customization_badge import measure_customization_badge

DETAIL_CARD_BOX = (38, 118, 142, 142)
SCHEMA = 'hif-inventory-detail-badge-count-v1'


def _glyph_canvases(descriptor):
    """Same four generic OCR views as arena_reader._badge_glyph_ocr_canvases.

    Keep this small pure rendering helper local to avoid importing the custom
    action registration package into an observer. Plate extraction is reused
    directly from arena_winrate; no digit templates or artwork are introduced.
    """
    values = np.asarray(descriptor)
    if (values.shape != (216,) or values.dtype.kind not in 'iu'
            or np.any(values < 0) or np.any(values > 15)):
        raise ValueError('inventory detail badge descriptor is invalid')
    mask = values.astype(np.uint8).reshape(18, 12) * 17
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('inventory detail badge descriptor is empty')
    crop = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    canvases = []
    for interpolation in (cv2.INTER_NEAREST, cv2.INTER_CUBIC):
        enlarged = cv2.resize(crop, None, fx=8., fy=8., interpolation=interpolation)
        for invert in (False, True):
            foreground = 255 - enlarged if invert else enlarged
            canvas = np.full((foreground.shape[0]+32, foreground.shape[1]+32),
                255 if invert else 0, np.uint8)
            canvas[16:-16, 16:-16] = foreground
            canvases.append(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    return tuple(canvases)


def _ids(frame_ids):
    if (not isinstance(frame_ids, (list, tuple)) or not frame_ids
            or any(type(value) is not str or not value for value in frame_ids)
            or len(set(frame_ids)) != len(frame_ids)):
        raise ValueError('inventory detail badge requires observed source frames')
    return list(frame_ids)


def _number(value, lower, upper):
    return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper


def _plate(observation):
    internal = observation.get('internal_features', {})
    if (observation.get('presence_state') != 'DETAIL_CANDIDATE'
            or observation.get('box') != list(DETAIL_CARD_BOX)
            or internal.get('status') != 'MEASURED'
            or internal.get('seeded_plate_candidate') is not True
            or internal.get('center_seed_component_count') != 1
            or not _number(internal.get('center_seed_ring_support'), 1, 96*96)
            or not _number(internal.get('badge_interior_area'), 150, 96*96)
            or not _number(internal.get('center_component_green_fill'), .70, 1)
            or not _number(internal.get('center_core_green_fill'), .15, 1)
            or not _number(internal.get('center_component_extent_ratio'), .085, 1)
            or internal.get('glyph_component_count') != 1
            or internal.get('glyph_non_badge_art_candidate') is not False):
        raise ValueError('inventory detail lacks one bounded green-plate glyph')
    box = internal.get('glyph_component_box')
    if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
            or not 2 <= box[2] <= 13 or not 5 <= box[3] <= 17
            or box[0] > 51 or box[0]+box[2]-1 < 44
            or min(box[:2]) < 0 or box[0]+box[2] > 96 or box[1]+box[3] > 96
            or not _number(internal.get('glyph_component_area'), 4, 110)):
        raise ValueError('inventory detail badge glyph geometry is unresolved')
    canvases = _glyph_canvases(internal.get('glyph_descriptor_12x18_q4'))
    return tuple(box), canvases


def _profile_canvases(canvases, profile):
    if profile not in ('standard', 'nonbattle_white'):
        raise ValueError('inventory detail badge render profile is uncalibrated')
    return [(index, canvas) for index, canvas in enumerate(canvases)
        if profile == 'standard' or index in (1, 3)]


def _frame_digit(observation):
    box, canvases = _plate(observation)
    canvases = _profile_canvases(canvases, observation.get('render_profile', 'standard'))
    variants = observation.get('ocr_variants')
    if not isinstance(variants, list) or len(variants) != len(canvases):
        raise ValueError('inventory detail badge OCR variants are incomplete')
    digits, accepted_variants = set(), set()
    for variant, (index, canvas) in zip(variants, canvases, strict=True):
        if (variant.get('index') != index or variant.get('shape') != list(canvas.shape)
                or not isinstance(variant.get('rows'), list)):
            raise ValueError('inventory detail badge OCR geometry is inconsistent')
        rows = variant['rows']
        for row in rows:
            text, score, region = row.get('text'), row.get('score'), row.get('box')
            if (text not in {'1', '2', '3'} or not _number(score, .95, 1)
                    or not isinstance(region, list) or len(region) != 4
                    or any(type(v) is not int for v in region) or min(region[:2]) < 0
                    or min(region[2:]) <= 0 or region[0]+region[2] > canvas.shape[1]
                    or region[1]+region[3] > canvas.shape[0]):
                raise ValueError('inventory detail badge OCR text or confidence is unresolved')
            digits.add(int(text))
        if len(rows) > 1:
            raise ValueError('inventory detail badge OCR has multiple digits')
        if rows:
            accepted_variants.add(index // 2)
    if len(digits) != 1 or accepted_variants != {0, 1}:
        raise ValueError('inventory detail badge lacks agreeing OCR in both renderings')
    return next(iter(digits)), box


def read_inventory_detail_badge_count(context, frames, frame_ids, *, render_profile='standard'):
    """Read the badge from the supplied inventory detail images."""
    from .inventory import _inventory_ocr

    ids = _ids(frame_ids)
    if not frames:
        raise ValueError('inventory detail badge requires an image')
    observations = []
    for image, frame_id in zip(frames, ids, strict=True):
        if not isinstance(image, np.ndarray) or image.shape != (1280, 720, 3):
            raise ValueError('inventory detail badge image is outside calibrated geometry')
        observed = measure_customization_badge(image, DETAIL_CARD_BOX)
        _, canvases = _plate(observed)
        observed.update(frame_id=frame_id, image_shape=list(image.shape), ocr_variants=[])
        if render_profile != 'standard':
            observed['render_profile'] = render_profile
        for index, canvas in _profile_canvases(canvases, render_profile):
            rows = _inventory_ocr(context, canvas, 'LiveInventoryDetailBadgeGlyph', expected=r'^[1-3]$')
            observed['ocr_variants'].append(dict(index=index, shape=list(canvas.shape), rows=[
                dict(text=row.text.strip(), score=row.score, box=list(row.box)) for row in rows]))
        observed['count'], _ = _frame_digit(observed)
        observations.append(observed)
    evidence = dict(schema=SCHEMA, frame_ids=ids, card_box=list(DETAIL_CARD_BOX),
        observations=observations, count=observations[0]['count'])
    if render_profile != 'standard':
        evidence['render_profile'] = render_profile
    verify_inventory_detail_badge_count(evidence, ids)
    return evidence


def verify_inventory_detail_badge_count(evidence, frame_ids):
    """Recheck raw glyph/geometry/OCR receipts against the same detail frames."""
    ids = _ids(frame_ids)
    if (not isinstance(evidence, dict) or evidence.get('schema') != SCHEMA
            or evidence.get('frame_ids') != ids or evidence.get('card_box') != list(DETAIL_CARD_BOX)
            or type(evidence.get('count')) is not int or evidence['count'] not in (1, 2, 3)
            or not isinstance(evidence.get('observations'), list)
            or len(evidence['observations']) != len(frame_ids)):
        raise ValueError('inventory detail badge evidence is not bound to the supplied frames')
    results = []
    for observation, frame_id in zip(evidence['observations'], ids, strict=True):
        if (observation.get('frame_id') != frame_id or observation.get('image_shape') != [1280, 720, 3]):
            raise ValueError('inventory detail badge frame geometry or identity differs')
        if observation.get('render_profile', 'standard') != evidence.get('render_profile', 'standard'):
            raise ValueError('inventory detail badge rendering differs across frames')
        digit, box = _frame_digit(observation)
        if type(observation.get('count')) is not int or observation['count'] != digit:
            raise ValueError('inventory detail badge declared count differs from OCR')
        results.append((digit, box))
    if len(set(results)) != 1 or results[0][0] != evidence['count']:
        raise ValueError('inventory detail badge count or glyph geometry changes across frames')
    return evidence['count']
