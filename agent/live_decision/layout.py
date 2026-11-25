from __future__ import annotations

from typing import Any
from dataclasses import dataclass

DMM_TOTAL_TURNS = 9
DMM_LATER_AUDITION_TURNS = 12
DMM_SELF_RANK_ANCHORS = (294, 367, 450, 520, 591, 662)
DMM_STAMINA_BAR_BOX = (506, 183, 142, 15)
DMM_STAMINA_BAR_INNER_WIDTH = DMM_STAMINA_BAR_BOX[2]


def calibrated_live_client_size(size):
    """Resolve a native size to its fixed rendering profile, without drift.

    The small HIF window can round either client edge by one physical pixel.
    Maa still captures/clicks in the existing 720x1280 normalized coordinates.
    This profile only chooses recognition assets; it never resizes a window.
    """
    if not isinstance(size, tuple) or len(size) != 2 or any(type(v) is not int for v in size):
        return None
    # MaaCLI observed a 719x1279 client at 96 DPI and normalized it to
    # the same complete 720x1280 frame as the nominal client.
    if size in ((720, 1280), (719, 1279)):
        return (720, 1280)
    # Native 474: at 144 DPI, Maa's logical client is 721x1282 while
    # the physical client is 1082x1923. Its own long-side normalization
    # produces the same 720x1280 layout. Bind input to the observed logical
    # dimensions; this mapping never converts or sends native coordinates.
    if all(abs(v-base) <= 1 for v,base in zip(size, (721, 1282), strict=True)):
        return (720, 1280)
    if all(abs(v-base) <= 1 for v,base in zip(size, (480, 853), strict=True)):
        return (480, 853)
    # Both observed small clients deliver a 720x1280 Maa screenshot. Use
    # small-window recognition assets; Maa maps clicks to its own client.
    if size in ((359, 638), (360, 640)):
        return (480, 853)
    # A disconnected display can move the existing 540x960 window between
    # DPI contexts, rounding both client edges together. Native 1302 proves
    # the existing 720 glyph profile after full-frame HIF normalization;
    # the capture adapter separately binds and maps Maa's 721/722px image.
    # Native 1920 also records 541x962 at 144 DPI with a complete 720x1280
    # capture. Admit that observed rounding, not arbitrary aspect-ratio drift.
    if size in ((540, 960), (541, 961), (541, 962), (542, 962)):
        return (720, 1280)
    return None


def _bgr_to_opencv_hsv(image: Any) -> tuple[Any, Any, Any]:
    """Return OpenCV-scale HSV channels without requiring the cv2 wheel."""

    import numpy as np

    bgr = np.asarray(image, dtype=np.float32)[..., :3] / 255.0
    blue, green, red = (bgr[..., index] for index in range(3))
    maximum = np.max(bgr, axis=2)
    minimum = np.min(bgr, axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green)
    blue_max = nonzero & (maximum == blue)
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6.0
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0
    hue *= 30.0
    saturation = np.zeros_like(maximum)
    positive = maximum > 1e-6
    saturation[positive] = delta[positive] / maximum[positive] * 255.0
    return hue, saturation, maximum * 255.0


@dataclass
class DmmTurnTracker:
    """Track the 9-turn first audition and 12-turn later auditions."""

    stage_gap_seconds: float = 90.0
    total_turns: int | None = None
    last_observed_at: float | None = None

    def observe(self, turns_remaining: int, observed_at: float) -> dict[str, Any]:
        if not 1 <= turns_remaining <= DMM_LATER_AUDITION_TURNS:
            return {"accepted": False, "value": None, "confidence": 0.0}
        new_stage = (
            self.last_observed_at is None
            or observed_at - self.last_observed_at > self.stage_gap_seconds
        )
        if new_stage:
            self.total_turns = (
                DMM_LATER_AUDITION_TURNS
                if turns_remaining > DMM_TOTAL_TURNS
                else DMM_TOTAL_TURNS
            )
        elif turns_remaining > DMM_TOTAL_TURNS:
            self.total_turns = DMM_LATER_AUDITION_TURNS
        self.last_observed_at = observed_at
        return infer_dmm_turn(turns_remaining, self.total_turns or DMM_TOTAL_TURNS)


def infer_dmm_turn(turns_remaining: int, total_turns: int = DMM_TOTAL_TURNS) -> dict[str, Any]:
    """Map an NIA audition countdown to a one-based turn."""

    if total_turns not in {DMM_TOTAL_TURNS, DMM_LATER_AUDITION_TURNS}:
        return {"accepted": False, "value": None, "confidence": 0.0}
    if not 1 <= turns_remaining <= total_turns:
        return {"accepted": False, "value": None, "confidence": 0.0}
    return {
        "accepted": True,
        "value": total_turns - turns_remaining + 1,
        "confidence": 1.0,
        "total_turns": total_turns,
    }


def locate_dmm_self_rank(image: Any) -> dict[str, Any]:
    """Locate the expanded pink player column in the DMM audition ranking strip."""

    try:
        import numpy as np

        if image.shape[0] < 70 or image.shape[1] < 720:
            raise ValueError("frame is smaller than the calibrated DMM layout")
        hue, saturation, value = _bgr_to_opencv_hsv(
            np.ascontiguousarray(image[:70, :720])
        )
        mask = (
            (hue >= 155)
            & (hue <= 179)
            & (saturation >= 80)
            & (value >= 120)
        )
        counts = mask.sum(axis=0)
        anchor_x = int(counts.argmax())
        peak = int(counts[anchor_x])
        rank_index = min(
            range(len(DMM_SELF_RANK_ANCHORS)),
            key=lambda index: abs(anchor_x - DMM_SELF_RANK_ANCHORS[index]),
        )
        distance = abs(anchor_x - DMM_SELF_RANK_ANCHORS[rank_index])
        if peak < 50 or distance > 12:
            raise ValueError("player ranking anchor did not meet the calibrated geometry")
        rank = rank_index + 1
        confidence = min(1.0, peak / 61.0) * max(0.0, 1.0 - distance / 16.0)
        return {
            "accepted": True,
            "value": [
                "self" if position == rank else f"opponent-{position}"
                for position in range(1, 7)
            ],
            "confidence": confidence,
            "self_rank": rank,
            "anchor_x": anchor_x,
            "score_roi": [anchor_x - 74, 90, 125, 80],
        }
    except (AttributeError, ImportError, TypeError, ValueError):
        return {"accepted": False, "value": None, "confidence": 0.0}


def infer_dmm_max_stamina(image: Any, stamina: int) -> dict[str, Any]:
    """Diagnose maximum stamina from the bar's saturated green/blue extent.

    The blue shield layer is rendered over the green stamina bar.  This helper
    intentionally measures their saturated union and remains diagnostic until
    partial-stamina frames with a positive shield prove that the overlay never
    extends the stamina boundary.
    """

    try:
        import numpy as np

        x, y, width, height = DMM_STAMINA_BAR_BOX
        if (
            stamina <= 0
            or image.shape[0] < y + height
            or image.shape[1] < x + width
        ):
            raise ValueError("stamina or frame is outside the calibrated layout")
        _hue, saturation, value = _bgr_to_opencv_hsv(
            np.ascontiguousarray(image[y : y + height, x : x + width])
        )
        mask = (saturation >= 80) & (value >= 100)
        columns = mask.sum(axis=0)
        if int(columns[0]) < 5:
            raise ValueError("stamina fill did not start at the calibrated anchor")
        fill_end = 0
        while fill_end + 1 < len(columns) and int(columns[fill_end + 1]) >= 7:
            fill_end += 1
        fill_width = fill_end + 1
        candidate = round(stamina * DMM_STAMINA_BAR_INNER_WIDTH / fill_width)
        expected_width = stamina * DMM_STAMINA_BAR_INNER_WIDTH / candidate
        residual = abs(fill_width - expected_width)
        if candidate < stamina or candidate > 100 or residual > 2.0:
            raise ValueError("stamina fill did not produce a stable integer maximum")
        return {
            "accepted": True,
            "value": candidate,
            "confidence": max(0.0, 1.0 - residual / 2.0),
            "fill_width": fill_width,
        }
    except (AttributeError, ImportError, TypeError, ValueError, ZeroDivisionError):
        return {"accepted": False, "value": None, "confidence": 0.0}
