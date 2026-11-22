"""Pure-image popup peaks for the controlled host's immutable resource mirror.

No Maa context or recognition RPC may enter a worker. Native recognition still
owns matrices with peaks at least .5: it collects and suppresses multiple boxes,
so eight peaks alone are not its complete result in that case.
"""

from time import perf_counter_ns
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .popup_guard import ROI, TEMPLATES, popup_configuration


class HifPopupMatcher:
    """One context-owned mirror; no global cache or OpenCV thread changes."""

    def __init__(self, params, templates):
        try:
            self.params = popup_configuration({
                'recognition': {'type': 'TemplateMatch', 'param': params},
            })
        except (TypeError, KeyError, AttributeError) as error:
            raise ValueError('invalid popup mirror configuration') from error
        if not isinstance(templates, (tuple, list)) or len(templates) != len(TEMPLATES):
            raise ValueError('popup mirror must contain all eight templates')
        for pixels in templates:
            if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
                    or pixels.ndim != 3 or pixels.shape[2] != 3
                    or not 0 < pixels.shape[0] <= ROI[3]
                    or not 0 < pixels.shape[1] <= ROI[2]
                    or pixels.flags.writeable):
                raise ValueError('popup mirror needs immutable uint8 BGR template pixels')
        self._templates = tuple(templates)

    @property
    def template_sizes(self):
        """Dimensions of the same immutable pixels installed in this context."""
        return tuple((pixels.shape[1], pixels.shape[0]) for pixels in self._templates)

    def match(self, image, *, part=None):
        """Measure all eight templates, or one ordered four-template half.

        ``needs_native`` forbids treating peak-only results as native results.
        Invalid computation raises instead of producing a negative observation.
        """
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.shape != (1280, 720, 3)):
            raise ValueError('popup mirror needs a 720x1280 uint8 BGR frame')
        if part is not None and (type(part) is not int or part not in (0, 1)):
            raise ValueError('popup mirror part must be 0, 1 or None')
        start = perf_counter_ns()
        x, y, width, height = ROI
        roi = image[y:y + height, x:x + width]
        indices = range(8) if part is None else range(part * 4, part * 4 + 4)

        def measure(index):
            pixels = self._templates[index]
            try:
                scores = cv2.matchTemplate(roi, pixels, cv2.TM_CCOEFF_NORMED)
                expected = (height - pixels.shape[0] + 1, width - pixels.shape[1] + 1)
                if (not isinstance(scores, np.ndarray) or scores.shape != expected
                        or not np.isfinite(scores).all()):
                    raise ValueError('popup template computation returned invalid scores')
                # Native scans columns before rows and keeps the first equal max.
                # Transposing only the result matrix reproduces that tie order.
                _, score, _, point = cv2.minMaxLoc(scores.T)
            except cv2.error as error:
                raise ValueError('popup template computation failed') from error
            if not np.isfinite(score):
                raise ValueError('popup template computation returned a nonfinite peak')
            threshold = .7  # The constructor validates both supported threshold forms.
            return {
                'template': TEMPLATES[index], 'score': float(score),
                'box': [point[1] + x, point[0] + y, pixels.shape[1], pixels.shape[0]],
                'threshold': threshold, 'hit': bool(score > threshold),
            }

        # Construction, dispatch, completion and shutdown all belong to this call.
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(measure, indices))
        return {
            'hit': any(row['hit'] for row in rows),
            # Nonpositive peaks can involve native's default closest-result box.
            'needs_native': any(not 0 < row['score'] < .5 for row in rows),
            'rows': rows, 'elapsed_ms': (perf_counter_ns() - start) / 1e6,
        }
