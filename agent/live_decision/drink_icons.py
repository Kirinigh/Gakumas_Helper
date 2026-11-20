"""Offline HUD drink identity matching with an explicitly supplied icon library."""

import cv2
import numpy as np

from .layout import calibrated_live_client_size

ICON_SIZE = 56
EXAM_CENTERS = ((62, 1200), (147, 1200), (233, 1200), (318, 1200))
MIN_SCORE = 0.50
MIN_MARGIN = 0.15


def _detail_pixels(image):
    pixels = image.astype(np.float32)
    return pixels - cv2.GaussianBlur(pixels, (0, 0), 4)


class DrinkIconRecognizer:
    """Compare bottle interiors by masked high-pass correlation, per physical slot."""

    def __init__(self, references, *, catalog_ids):
        # References map catalog IDs to aligned BGRA icons. A second ID is
        # needed for the margin to have meaning.
        self._catalog_ids = frozenset(catalog_ids)
        if len(references) < 2 or len(self._catalog_ids) < 2 or not all(type(drink_id) is str and drink_id for drink_id in self._catalog_ids):
            raise ValueError("drink icon library needs competing catalog IDs")
        self._coverage_complete = self._catalog_ids <= references.keys()
        self._templates = []
        for drink_id, image in references.items():
            if (
                type(drink_id) is not str
                or not drink_id
                or not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 4
                or image.size == 0
            ):
                raise ValueError("drink icon reference lacks a valid catalog ID or BGRA pixels")
            icon = cv2.resize(image, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
            mask = icon[:, :, 3] > 128
            values = _detail_pixels(icon[:, :, :3])[mask].ravel()
            if values.size < 3 * ICON_SIZE:
                raise ValueError("drink icon reference has insufficient foreground")
            values -= values.mean()
            norm = np.linalg.norm(values)
            if norm <= 1e-6:
                raise ValueError("drink icon reference has insufficient contrast")
            self._templates.append((drink_id, mask, values / norm))

    def rank_crop(self, image):
        """Return the leading and runner-up catalog scores for one aligned crop."""
        if not isinstance(image, np.ndarray) or image.shape != (ICON_SIZE, ICON_SIZE, 3) or image.dtype != np.uint8:
            raise ValueError("drink icon crop is uncalibrated")
        details = _detail_pixels(image)
        scores = []
        for drink_id, mask, reference in self._templates:
            values = details[mask].ravel()
            values -= values.mean()
            norm = np.linalg.norm(values)
            score = float(np.clip(np.dot(reference, values) / norm, -1, 1)) if norm > 1e-6 else 0.0
            scores.append((drink_id, score))
        return sorted(scores, key=lambda candidate: candidate[1], reverse=True)[:2]

    def __call__(self, frames, occupied, *, client_size):
        if (
            calibrated_live_client_size(client_size) is None
            or len(frames) != 3
            or any(not isinstance(frame, np.ndarray) or frame.shape != (1280, 720, 3) or frame.dtype != np.uint8 for frame in frames)
        ):
            raise ValueError("drink icons require three normalized HUD frames")
        if any(type(slot) is not int or not 0 <= slot < len(EXAM_CENTERS) for slot in occupied):
            raise ValueError("drink icon slot is invalid")

        result = {}
        half = ICON_SIZE // 2
        for slot in occupied:
            x, y = EXAM_CENTERS[slot]
            readings = []
            for frame in frames:
                best, second = self.rank_crop(frame[y - half : y + half, x - half : x + half])
                readings.append(dict(candidate=best[0], score=best[1], runner_up=second[0], margin=best[1] - second[1]))
            ids = {reading["candidate"] for reading in readings}
            score = min(reading["score"] for reading in readings)
            margin = min(reading["margin"] for reading in readings)
            candidate = readings[0]["candidate"]
            accepted = (
                self._coverage_complete and candidate in self._catalog_ids and len(ids) == 1 and score >= MIN_SCORE and margin >= MIN_MARGIN
            )
            result[slot] = dict(
                drink_id=candidate if accepted else None, confidence=score if accepted else None, min_margin=margin, readings=readings
            )
            if not accepted:
                result[slot]["reason"] = "incomplete_icon_catalog" if not self._coverage_complete else "weak_or_inconsistent_icon"
        return result
