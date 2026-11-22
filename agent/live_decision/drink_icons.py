"""Recognize drink identities from calibrated HUD frames and local icon assets."""

import json
from functools import lru_cache

import cv2
import numpy as np

from .layout import calibrated_live_client_size

ICON_SIZE = 56
EXAM_CENTERS = ((62, 1200), (147, 1200), (233, 1200), (318, 1200))
# Bottom-bar layout and score rules from hiro-gakumas-hero at 85e22b4e.
MIN_SCORE = 0.50
MIN_MARGIN = 0.15
LOW_SCORE = 0.42
LOW_MARGIN = 0.22
EMPTY_MAX = 0.35
GREY_SAT_MAX = 50
ROW_OFFSETS = (0, 13)
EVENT_GREEN_JUICE_ID = 'pdrink_00-3-013'
EVENT_GREEN_JUICE_NAME = '初星スペシャル青汁X'


def _detail_pixels(image):
    pixels = image.astype(np.float32)
    return pixels - cv2.GaussianBlur(pixels, (0, 0), 4)


class DrinkIconRecognizer:
    """Compare bottle interiors by masked high-pass correlation, per physical slot."""

    def __init__(self, references, *, catalog_ids):
        # References map catalog IDs to aligned BGRA icons. A second ID is
        # needed for the margin to have meaning.
        self._catalog_ids = frozenset(catalog_ids)
        if len(references) < 2 or len(self._catalog_ids) < 2 or not all(
                type(drink_id) is str and drink_id for drink_id in self._catalog_ids):
            raise ValueError('drink icon library needs competing catalog IDs')
        self._coverage_complete = self._catalog_ids <= references.keys()
        self._ids = []
        masks = []
        templates = []
        for drink_id, image in references.items():
            if (type(drink_id) is not str or not drink_id or not isinstance(image, np.ndarray)
                    or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4 or image.size == 0):
                raise ValueError('drink icon reference lacks a valid catalog ID or BGRA pixels')
            icon = cv2.resize(image, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
            mask = icon[:, :, 3] > 128
            values = _detail_pixels(icon[:, :, :3])[mask].ravel()
            if values.size < 3 * ICON_SIZE:
                raise ValueError('drink icon reference has insufficient foreground')
            values -= values.mean()
            norm = np.linalg.norm(values)
            if norm <= 1e-6:
                raise ValueError('drink icon reference has insufficient contrast')
            masked_template = np.zeros((ICON_SIZE, ICON_SIZE, 3), dtype=np.float32)
            masked_template[mask] = (values / norm).reshape(-1, 3)
            self._ids.append(drink_id)
            templates.append(masked_template.ravel())
            masks.append(np.repeat(mask.ravel(), 3))
        self._templates = np.stack(templates)
        self._masks = np.stack(masks).astype(np.float32)
        self._counts = self._masks.sum(axis=1, keepdims=True)
        self._last_region = None

    def _rank_crops(self, crops):
        pixels = np.stack([_detail_pixels(crop).ravel() for crop in crops], axis=1)
        masked_sum = self._masks @ pixels
        masked_power = self._masks @ (pixels * pixels)
        variance = np.maximum(masked_power - masked_sum * masked_sum / self._counts, 1e-6)
        scores = np.clip((self._templates @ pixels) / np.sqrt(variance), -1, 1).T
        return [tuple((self._ids[index], float(row[index])) for index in np.argsort(-row, kind='stable')[:2])
                for row in scores]

    def rank_crop(self, image):
        """Return the leading and runner-up catalog scores for one aligned crop."""
        if not isinstance(image, np.ndarray) or image.shape != (ICON_SIZE, ICON_SIZE, 3) or image.dtype != np.uint8:
            raise ValueError('drink icon crop is uncalibrated')
        return list(self._rank_crops([image])[0])

    def read_slots(self, frame, *, client_size):
        """Read the four HUD slots with hiro's two row layouts and score rules."""
        if (calibrated_live_client_size(client_size) is None or not isinstance(frame, np.ndarray)
                or frame.shape != (1280, 720, 3) or frame.dtype != np.uint8):
            raise ValueError('drink icons require one normalized HUD frame')
        if not self._coverage_complete:
            return [dict(state='UNKNOWN', drink_id=None, reason='incomplete_icon_catalog') for _ in EXAM_CENTERS]

        half = ICON_SIZE // 2
        left, right = EXAM_CENTERS[0][0] - half, EXAM_CENTERS[-1][0] + half
        y = EXAM_CENTERS[0][1]
        region = frame[y-half:y+half+max(ROW_OFFSETS), left:right]
        cached = self._last_region
        if cached is None or not np.array_equal(cached[0], region):
            crops = [region[dy:dy+ICON_SIZE, x-half-left:x+half-left]
                     for dy in ROW_OFFSETS for x, _ in EXAM_CENTERS]
            count = len(EXAM_CENTERS)
            scores = self._rank_crops(crops)
            row = max(range(len(ROW_OFFSETS)), key=lambda i: sum(pair[0][1] for pair in scores[i*count:(i+1)*count]))
            start = row * count
            saturations = [float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
                           for crop in crops[start:start+count]]
            cached = (region.copy(), row, scores[start:start+count], saturations)
            self._last_region = cached

        _, row, rankings, saturations = cached
        slots = []
        for (x, y), ((candidate, score), (runner_up, second)), saturation in zip(
                EXAM_CENTERS, rankings, saturations, strict=True):
            margin = score - second
            result = dict(state='UNKNOWN', drink_id=None, available=None, center=(x, y + ROW_OFFSETS[row]),
                confidence=None, min_margin=margin,
                readings=[dict(candidate=candidate, score=score, runner_up=runner_up, margin=margin)])
            if (score >= MIN_SCORE and margin >= MIN_MARGIN) or (score >= LOW_SCORE and margin >= LOW_MARGIN):
                result.update(state='OCCUPIED', drink_id=candidate, available=saturation >= GREY_SAT_MAX,
                              confidence=score, saturation=saturation)
            elif score < EMPTY_MAX:
                result.update(state='EMPTY', confidence=1 - score)
            else:
                result['reason'] = 'weak_icon'
            slots.append(result)
        return slots

    def __call__(self, frames, occupied, *, client_size):
        if (calibrated_live_client_size(client_size) is None or len(frames) not in (1, 3)
                or any(not isinstance(frame, np.ndarray) or frame.shape != (1280, 720, 3)
                       or frame.dtype != np.uint8 for frame in frames)):
            raise ValueError('drink icons require one normalized HUD frame or a three-frame diagnostic batch')
        if any(type(slot) is not int or not 0 <= slot < len(EXAM_CENTERS) for slot in occupied):
            raise ValueError('drink icon slot is invalid')

        result = {}
        half = ICON_SIZE // 2
        for slot in occupied:
            x, y = EXAM_CENTERS[slot]
            readings = []
            for frame in frames:
                best, second = self.rank_crop(frame[y-half:y+half, x-half:x+half])
                readings.append(dict(candidate=best[0], score=best[1], runner_up=second[0],
                                     margin=best[1] - second[1]))
            ids = {reading['candidate'] for reading in readings}
            score = min(reading['score'] for reading in readings)
            margin = min(reading['margin'] for reading in readings)
            candidate = readings[0]['candidate']
            accepted = (self._coverage_complete and candidate in self._catalog_ids
                        and len(ids) == 1 and score >= MIN_SCORE and margin >= MIN_MARGIN)
            result[slot] = dict(drink_id=candidate if accepted else None,
                                confidence=score if accepted else None, min_margin=margin, readings=readings)
            if not accepted:
                result[slot]['reason'] = ('incomplete_icon_catalog' if not self._coverage_complete
                                          else 'weak_or_inconsistent_icon')
        return result


@lru_cache(maxsize=8)
def load_default_drink_icons(directory, catalog_ids):
    """Load a versioned local pack of aligned, masked BGRA icons named by ID."""
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_bytes())
    if (not isinstance(manifest, dict) or manifest.get('schema') != 1
            or not isinstance(manifest.get('source'), str) or not manifest['source'].strip()
            or not isinstance(manifest.get('ids'), list)):
        raise ValueError('default drink icon manifest is invalid')
    ids = manifest['ids']
    if (not all(type(item) is str and item in catalog_ids
                and (item.isascii() and item.isdecimal() or item == EVENT_GREEN_JUICE_ID) for item in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('default drink icon manifest contains duplicate or unknown IDs')
    references = {item: cv2.imdecode(np.frombuffer((directory / f'{item}.png').read_bytes(), dtype=np.uint8),
                                   cv2.IMREAD_UNCHANGED) for item in ids}
    return DrinkIconRecognizer(references, catalog_ids=catalog_ids) if len(references) >= 2 else None
