"""Arena-only context adaptation for the shared upstream card detector."""

from __future__ import annotations

import cv2
import numpy as np
from arena_winrate._detail_title_region import _white_detail_panels


def prepare_card_detector_input(image: np.ndarray) -> tuple[np.ndarray, int | None]:
    """Keep the existing coordinate transform and one inference per frame.

    A proven member panel keeps all its pixels and the stage separator. Only
    Main/Sub artwork below it is replaced with Maa's neutral padding colour.
    Unknown/obstructed pages retain the original full-frame detector input;
    this helper never supplies card positions or authorizes a page/identity.
    """
    detector_image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    bottom = _member_panel_context_bottom(detector_image)
    if bottom is not None:
        detector_image[bottom:] = 114
    return detector_image, bottom


def _member_panel_context_bottom(image: np.ndarray) -> int | None:
    """Locate the unique stage rubric bar independently of cards.onnx.

    Require ordered solid pink/blue/yellow segments, dark neutral shoulders,
    and both light member-panel margins above. Card artwork and the miniature
    Main/Sub bars cannot independently satisfy this topology. A tooltip or
    ambiguous separator abstains; no fixed source-pixel crop is assumed.
    """
    if _white_detail_panels(image):
        return None
    height, width = image.shape[:2]
    top, left = round(height * .20), round(width * .25)
    region = image[top:round(height * .75), left:round(width * .75)]
    colours = (
        ((90, 0, 190), (200, 130, 255)),
        ((180, 95, 0), (255, 200, 100)),
        ((0, 140, 210), (95, 215, 255)),
    )
    segments = []
    for lower, upper in colours:
        mask = cv2.inRange(region, lower, upper)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        segments.append([
            (int(x) + left, int(y) + top, int(w), int(h))
            for x, y, w, h, area in stats[1:]
            if width * .009 <= w <= width * .28
            and height * .006 <= h <= height * .027
            and area >= w * h * .90
        ])

    boundaries = []
    for pink in segments[0]:
        for blue in segments[1]:
            for yellow in segments[2]:
                strip = (pink, blue, yellow)
                if not _aligned_stage_segments(strip, width):
                    continue
                if not _enclosed_stage_segments(image, strip):
                    continue
                y = min(part[1] for part in strip)
                bar_height = max(part[3] for part in strip)
                # The neutral band must surround the rubric, with the light
                # member panel directly above it. Margins avoid card pixels,
                # stage labels, plan icons and the right-hand scrollbar.
                dark = image[max(0, y - bar_height):max(0, y - bar_height + 4)]
                light = image[max(0, y - 4 * bar_height):max(0, y - 3 * bar_height)]
                if not (
                    _neutral_margins(dark, width, 20, 100, .90)
                    and _neutral_margins(light, width, 225, 255, .90)
                ):
                    continue
                # Preserve the entire separator and a margin below the bar.
                bottom = max(part[1] + part[3] for part in strip) + max(12, bar_height + 2)
                boundaries.append(bottom)
    return boundaries.pop() if len(boundaries) == 1 else None


def _aligned_stage_segments(parts: tuple[tuple[int, int, int, int], ...], width: int) -> bool:
    if max(part[1] for part in parts) - min(part[1] for part in parts) > 1:
        return False
    if max(part[3] for part in parts) - min(part[3] for part in parts) > 2:
        return False
    if not width * .22 <= parts[-1][0] + parts[-1][2] - parts[0][0] <= width * .34:
        return False
    return all(
        1 <= right[0] - (left[0] + left[2]) <= width * .014
        for left, right in zip(parts, parts[1:])
    )


def _enclosed_stage_segments(image: np.ndarray, parts: tuple[tuple[int, int, int, int], ...]) -> bool:
    left, right = parts[0][0], parts[-1][0] + parts[-1][2]
    top = min(part[1] for part in parts)
    bottom = max(part[1] + part[3] for part in parts)
    # Require the enclosing horizontal edge above and below the full strip.
    # A partly rendered/cropped strip must not acquire a smaller valid height.
    edge_depth = max(3, round(max(part[3] for part in parts) * .4))
    for first, last in ((max(0, top - 2), top), (bottom, min(image.shape[0], bottom + edge_depth))):
        edge = image[first:last, left:right]
        if not edge.size:
            return False
        low, high = edge.min(axis=2), edge.max(axis=2)
        # Resizing a small window blends adjacent segment colour into its
        # one-pixel dark edge; the separate band margins remain strictly neutral.
        neutral = (high <= 130) & (high - low <= 35)
        if not np.any(neutral.mean(axis=1) >= .90):
            return False
    return True


def _neutral_margins(image: np.ndarray, width: int, minimum: int, maximum: int, fraction: float) -> bool:
    if not image.size:
        return False
    for left, right in ((.018, .030), (.963, .969)):
        margin = image[:, round(width * left):round(width * right)]
        low, high = margin.min(axis=2), margin.max(axis=2)
        if ((low >= minimum) & (high <= maximum) & (high - low <= 15)).mean() < fraction:
            return False
    return True
