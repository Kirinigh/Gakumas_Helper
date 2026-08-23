"""Fail-closed presence recognition for arena skill-card customization badges.

The contest member-detail screen renders a green count badge only when a card
has one or more customizations.  The fixed-centre green plate first authorizes
the authoritative detail read.  Only when detail semantics remain non-unique
does the reader use an aspect-preserving glyph cut from that same plate to
resolve the count; card art and legacy digit templates never participate.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from functools import lru_cache
from dataclasses import dataclass
from collections.abc import Mapping, Sequence


class CustomizationBadgeState(StrEnum):
    """Fail-closed result of the card-face shortlist.

    Card pixels are intentionally unable to emit a positive customization
    verdict. A stable visual candidate only authorizes the authoritative
    detail transaction.
    """

    DETAIL_CANDIDATE = "DETAIL_CANDIDATE"
    CONFIDENT_ZERO = "CONFIDENT_ZERO"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class CustomizationBadgeDecision:
    state: CustomizationBadgeState
    count: int | None
    reason: str


def duplicate_marker_visual_features(card_image: Any) -> dict[str, float]:
    """Return cheap chroma features for shortlisting grey ``重複`` cards."""

    import cv2
    import numpy as np

    if not isinstance(card_image, np.ndarray) or card_image.ndim != 3 or card_image.shape[2] < 3:
        raise ValueError("duplicate-marker card image must be a BGR array")
    height, width = card_image.shape[:2]
    inner = card_image[
        max(1, int(height * 0.08)) : max(2, int(height * 0.72)),
        max(1, int(width * 0.08)) : max(2, int(width * 0.92)),
        :3,
    ]
    if inner.size == 0:
        raise ValueError("duplicate-marker card image is too small")
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    channel_range = inner.max(axis=2) - inner.min(axis=2)
    return {
        "mean_saturation": round(float(saturation.mean()), 6),
        "median_saturation": round(float(np.median(saturation)), 6),
        "p90_saturation": round(float(np.percentile(saturation, 90)), 6),
        "mean_channel_range": round(float(channel_range.mean()), 6),
        "low_chroma_fraction": round(float((channel_range <= 12).mean()), 6),
    }


def is_duplicate_marker_visual_candidate(
    repeated_features: Sequence[Mapping[str, float]],
) -> bool:
    """Shortlist stable grey cards; OCR remains the authoritative marker gate."""

    if len(repeated_features) != 3:
        raise ValueError("exactly three duplicate-marker visual observations are required")
    try:
        mean_ranges = tuple(float(item["mean_channel_range"]) for item in repeated_features)
        p90_saturations = tuple(float(item["p90_saturation"]) for item in repeated_features)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("duplicate-marker visual observations are incomplete") from error
    return max(mean_ranges) <= 45.0 or max(p90_saturations) <= 145.0


@lru_cache(maxsize=1)
def _centered_badge_geometry() -> tuple[Any, ...]:
    """Precompute the only fixed geometry used by the shortlist detector."""

    import numpy as np

    yy, xx = np.indices((96, 96))
    peak_x, peak_y = 46, 72
    distance = np.sqrt(
        (xx - peak_x).astype(np.float32) ** 2
        + (yy - peak_y).astype(np.float32) ** 2
    )
    return yy, xx, distance


def _measure_customization_badge_internal(
    crop: Any,
) -> dict[str, Any]:
    """Extract badge-presence features from one centre-seeded green plate.

    The only accepted structure is one connected green plate at the fixed
    lower-centre badge locus.  Green card art and the card frame cannot move
    the centre or independently authorize the plate.  No pixel mask leaves
    this function or becomes a second consumer.
    """

    import cv2
    import numpy as np

    array = np.asarray(crop)
    if array.ndim != 3 or array.shape[0] < 8 or array.shape[1] < 8 or array.shape[2] < 3:
        return {
            "status": "INVALID_ROI",
            "normalization_size": [96, 96],
            "seeded_plate_candidate": False,
        }

    normalized = cv2.resize(
        np.ascontiguousarray(array[:, :, :3]),
        (96, 96),
        interpolation=cv2.INTER_AREA,
    )
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(normalized.astype(np.int16))
    hsv_green = cv2.inRange(hsv, (30, 55, 45), (100, 255, 255)) > 0
    dominant_green = (g >= 65) & ((g - np.maximum(r, b)) >= 14)
    green_union = hsv_green | dominant_green

    # Card boxes are normalized before this point.  The centre is therefore a
    # semantic constant, never a peak selected from the card artwork.
    yy, xx, distance_from_peak = _centered_badge_geometry()
    peak_x, peak_y = 46, 72
    local_domain = distance_from_peak <= 12.5
    ideal_disk = distance_from_peak <= 11.5
    local_green = green_union & local_domain
    _, component_labels, _, _ = (
        cv2.connectedComponentsWithStats(local_green.astype(np.uint8), connectivity=8)
    )
    # A white digit can cover the exact centre pixel, so the seed is a tiny
    # centre neighbourhood rather than one pixel.  There must nevertheless be
    # exactly one green connected component touching that seed.  A detached
    # ring plus an unrelated central fleck must never be fused into a badge.
    component_seed = distance_from_peak <= 4.0
    component_ring = (distance_from_peak >= 5.0) & (distance_from_peak <= 10.0)
    seeded_component_ids = tuple(
        int(value)
        for value in np.unique(component_labels[component_seed & local_green])
        if int(value) != 0
    )
    selected_component = np.zeros((96, 96), dtype=bool)
    plate_component = np.zeros((96, 96), dtype=bool)
    badge_interior = np.zeros((96, 96), dtype=bool)
    badge_hull_points: list[list[int]] = []
    center_seed_ring_support = 0
    discarded_outer_component_pixels = 0
    if len(seeded_component_ids) == 1:
        selected_index = seeded_component_ids[0]
        selected_component = component_labels == selected_index
        center_seed_ring_support = int((selected_component & component_ring).sum())
        # A badge may physically touch green frame/art outside its circular
        # boundary.  Keep connectivity for selecting the seeded component, but
        # never let its external branch contribute points to the interior hull.
        plate_component = selected_component & (distance_from_peak <= 10.75)
        discarded_outer_component_pixels = int(
            (selected_component & ~plate_component).sum()
        )
        component_points_yx = np.column_stack(np.nonzero(plate_component))
        if center_seed_ring_support and len(component_points_yx) >= 3:
            component_points_xy = component_points_yx[:, ::-1].astype(np.int32)
            hull = cv2.convexHull(component_points_xy)
            hull_mask = np.zeros((96, 96), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)
            badge_interior = (hull_mask > 0) & ideal_disk
            badge_hull_points = [
                [int(point[0][0]), int(point[0][1])] for point in hull
            ]

    interior_area = int(badge_interior.sum())
    component_green_fill = (
        float(plate_component[badge_interior].mean()) if interior_area else 0.0
    )
    center_core = badge_interior & component_seed
    center_core_green_fill = (
        float(plate_component[center_core].mean())
        if bool(center_core.any())
        else 0.0
    )
    component_extent_ratio = (
        float(distance_from_peak[plate_component].max()) / 96.0
        if bool(plate_component.any())
        else 0.0
    )
    # This is deliberately a high-recall shortlist, not a positive verdict.
    # A card-art rectangle may enter the shortlist; the clicked detail is the
    # sole authority that returns zero or a concrete customization set. Do not
    # require all four fixed-centre quadrants: a correctly detected card can
    # still shift the rendered plate a few normalized pixels, and the old
    # quadrant veto created another systematic false-negative lock.
    seeded_plate_candidate = bool(
        len(seeded_component_ids) == 1
        and center_seed_ring_support > 0
        and interior_area >= 150
        and component_green_fill >= 0.70
        and center_core_green_fill >= 0.15
        and component_extent_ratio >= 0.085
    )
    glyph_descriptor: list[int] | None = None
    glyph_component_box: list[int] | None = None
    glyph_component_area: int | None = None
    glyph_component_count = 0
    glyph_foreground_pixel_count = 0
    glyph_raw_components: list[dict[str, Any]] = []
    glyph_non_badge_art_candidate = False
    if seeded_plate_candidate:
        # Count is never inferred from arbitrary white pixels in the card box.
        # The auxiliary glyph is cut only from the bounded interior generated
        # by the exact same centre-seeded green component that authorized the
        # detail click.  It is used later only when the already-open detail is
        # semantically non-unique without the displayed badge total.
        # Preserve the original white connectivity. Subtracting pale green
        # pixels first can shatter one large card-art stroke into a small
        # digit-like fragment and produce a silent false positive.
        white = (hsv[:, :, 1] <= 135) & (hsv[:, :, 2] >= 145)
        # The badge plate is allowed the same bounded card-box displacement as
        # the presence shortlist.  Keep the domain fixed around the semantic
        # locus, but do not assume the rendered digit is centred on the seed:
        # the live 720x1280 glyph sits about seven normalized pixels below it.
        glyph_domain = badge_interior & (distance_from_peak <= 10.25)
        glyph_pixels = white & glyph_domain
        glyph_foreground_pixel_count = int(glyph_pixels.sum())
        component_count, glyph_labels, glyph_stats, glyph_centroids = (
            cv2.connectedComponentsWithStats(
                glyph_pixels.astype(np.uint8),
                connectivity=8,
            )
        )
        glyph_candidates: list[int] = []
        for component_index in range(1, component_count):
            gx, gy, gw, gh, area = (
                int(value) for value in glyph_stats[component_index]
            )
            centroid_x, centroid_y = glyph_centroids[component_index]
            centroid_distance = float(
                np.hypot(centroid_x - peak_x, centroid_y - peak_y)
            )
            # A rendered one-digit count crosses the lower-centre lane of the
            # same green plate.  Merely lying somewhere inside the plate is
            # insufficient: live card art can leave a second white component
            # near the lower-left edge.  Requiring geometric intersection with
            # the centre lane keeps that art out without merging or selecting
            # components by OCR outcome.
            intersects_digit_lane = bool(
                gx <= peak_x + 5
                and gx + gw - 1 >= peak_x - 2
            )
            glyph_raw_components.append(
                {
                    "box": [gx, gy, gw, gh],
                    "area": area,
                    "centroid": [
                        round(float(centroid_x), 6),
                        round(float(centroid_y), 6),
                    ],
                    "centroid_distance": round(centroid_distance, 6),
                    "intersects_digit_lane": intersects_digit_lane,
                }
            )
            if (
                4 <= area <= 110
                and 2 <= gw <= 13
                and 5 <= gh <= 17
                and centroid_distance <= 9.5
                and centroid_y >= peak_y + 0.5
                and intersects_digit_lane
            ):
                glyph_candidates.append(component_index)
        glyph_component_count = len(glyph_candidates)
        if not glyph_candidates:
            oversized_components = []
            for raw in glyph_raw_components:
                _, _, raw_width, raw_height = raw["box"]
                if (
                    bool(raw["intersects_digit_lane"])
                    and float(raw["centroid_distance"]) <= 9.5
                    and (
                        int(raw["area"]) > 110
                        or int(raw_width) > 13
                        or int(raw_height) > 17
                    )
                ):
                    oversized_components.append(raw)
            if len(glyph_raw_components) == 1:
                glyph_non_badge_art_candidate = len(oversized_components) == 1
            elif len(oversized_components) == 1:
                dominant = oversized_components[0]
                # A live false plate can contain one large white art block plus
                # a few interpolation specks at its outer edge.  Do not let
                # those specks disable the contradiction, but require the main
                # block to sit above the digit baseline.  Off-lane remnants
                # remain bounded to eight pixels; at most one isolated
                # one-pixel interpolation remnant may touch the digit lane,
                # because it cannot encode a rendered count.  Any larger lane
                # fragment remains a detail candidate.  A missing or
                # fragmented real digit has no such dominant component and
                # therefore also remains a detail candidate.
                other_components = tuple(
                    raw for raw in glyph_raw_components if raw is not dominant
                )
                lane_single_pixel_specks = sum(
                    int(raw["area"]) == 1
                    and bool(raw["intersects_digit_lane"])
                    for raw in other_components
                )
                glyph_non_badge_art_candidate = bool(
                    float(dominant["centroid"][1]) <= peak_y
                    and lane_single_pixel_specks <= 1
                    and all(
                        int(raw["area"]) <= 8
                        and (
                            not bool(raw["intersects_digit_lane"])
                            or int(raw["area"]) == 1
                        )
                        for raw in other_components
                    )
                )
        if len(glyph_candidates) == 1:
            component_mask = glyph_labels == glyph_candidates[0]
            glyph_y, glyph_x = np.nonzero(component_mask)
            gx = int(glyph_x.min())
            gy = int(glyph_y.min())
            gw = int(glyph_x.max() - gx + 1)
            gh = int(glyph_y.max() - gy + 1)
            area = int(component_mask.sum())
            glyph_crop = component_mask[gy : gy + gh, gx : gx + gw]
            if area <= 140 and gw <= 15 and gh <= 17:
                scale = min(12.0 / gw, 18.0 / gh)
                descriptor_width = max(1, min(12, int(round(gw * scale))))
                descriptor_height = max(1, min(18, int(round(gh * scale))))
                resized_glyph = cv2.resize(
                    glyph_crop.astype(np.uint8) * 255,
                    (descriptor_width, descriptor_height),
                    interpolation=cv2.INTER_AREA,
                )
                descriptor = np.zeros((18, 12), dtype=np.uint8)
                descriptor_x = (12 - descriptor_width) // 2
                descriptor_y = (18 - descriptor_height) // 2
                descriptor[
                    descriptor_y : descriptor_y + descriptor_height,
                    descriptor_x : descriptor_x + descriptor_width,
                ] = resized_glyph
                glyph_descriptor = [
                    int(round(float(value) / 255.0 * 15.0))
                    for value in descriptor.reshape(-1)
                ]
                glyph_component_box = [gx, gy, gw, gh]
                glyph_component_area = area
    features = {
        "status": "MEASURED",
        "normalization_size": [96, 96],
        "anchor_box": [22, 48, 52, 46],
        "peak_center": [int(peak_x), int(peak_y)],
        "badge_center": [int(peak_x), int(peak_y)],
        "badge_hull_points": badge_hull_points,
        "badge_mask_radius": 11.5,
        "badge_interior_area": interior_area,
        "center_seed_component_count": len(seeded_component_ids),
        "center_seed_ring_support": center_seed_ring_support,
        "discarded_outer_component_pixels": discarded_outer_component_pixels,
        "center_component_green_fill": round(component_green_fill, 6),
        "center_core_green_fill": round(center_core_green_fill, 6),
        "center_component_extent_ratio": round(component_extent_ratio, 6),
        "seeded_plate_candidate": seeded_plate_candidate,
        "glyph_component_count": glyph_component_count,
        "glyph_component_box": glyph_component_box,
        "glyph_component_area": glyph_component_area,
        "glyph_foreground_pixel_count": glyph_foreground_pixel_count,
        "glyph_raw_components": glyph_raw_components,
        "glyph_non_badge_art_candidate": glyph_non_badge_art_candidate,
        "glyph_descriptor_12x18_q4": glyph_descriptor,
    }
    return features


def customization_badge_internal_features(crop: Any) -> dict[str, Any]:
    """Return serializable badge evidence without retaining pixels or masks."""

    return _measure_customization_badge_internal(crop)


def measure_customization_badge(
    image: Any,
    box: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Measure badge presence; card-face pixels never classify its count."""

    import cv2
    import numpy as np

    x, y, width, height = box
    crop = image[y : y + height, x : x + width, :3]
    if crop.shape[:2] != (height, width):
        return {
            "status": "INVALID_ROI",
            "count": None,
            "presence_state": CustomizationBadgeState.AMBIGUOUS.value,
            "presence_reason": "invalid_roi",
        }
    badge_box = (
        int(width * 0.27),
        int(height * 0.52),
        max(8, int(width * 0.47)),
        max(8, int(height * 0.46)),
    )
    internal_features = _measure_customization_badge_internal(crop)
    internal_candidate = internal_features.get("seeded_plate_candidate") is True
    if internal_candidate:
        presence_state = CustomizationBadgeState.DETAIL_CANDIDATE
        presence_reason = "seeded_plate_requires_detail"
    elif internal_features.get("status") == "MEASURED":
        presence_state = CustomizationBadgeState.CONFIDENT_ZERO
        presence_reason = "normalized_seeded_plate_absence"
    else:
        presence_state = CustomizationBadgeState.AMBIGUOUS
        presence_reason = "conflicting_or_incomplete_internal_evidence"
    if presence_state is CustomizationBadgeState.CONFIDENT_ZERO:
        return {
            "status": "CONFIDENT_ZERO",
            "count": 0,
            "presence_state": presence_state.value,
            "presence_reason": presence_reason,
            "box": list(box),
            "badge_box_within_card": list(badge_box),
            "internal_features": internal_features,
            "crop_mean": round(float(np.mean(crop)), 6),
        }
    return {
        "status": presence_state.value,
        "count": None,
        "presence_state": presence_state.value,
        "presence_reason": presence_reason,
        "box": list(box),
        "badge_box_within_card": list(badge_box),
        "internal_features": internal_features,
        "crop_mean": round(float(np.mean(crop)), 6),
    }


def stable_customization_badge_presence(
    observations: Sequence[Mapping[str, Any]],
    *,
    required_frames: int = 3,
) -> CustomizationBadgeDecision:
    """Fuse only badge-presence evidence without requiring a digit result."""

    if len(observations) != required_frames:
        return CustomizationBadgeDecision(
            CustomizationBadgeState.AMBIGUOUS,
            None,
            "incomplete_frame_set",
        )
    valid = tuple(
        observation.get("internal_features", {}).get("status") == "MEASURED"
        for observation in observations
    )
    if not all(valid):
        return CustomizationBadgeDecision(
            CustomizationBadgeState.AMBIGUOUS,
            None,
            "invalid_seeded_plate_frame",
        )
    seeded_plate_states = tuple(
        observation["internal_features"].get("seeded_plate_candidate") is True
        for observation in observations
    )
    if all(seeded_plate_states):
        oversized_foreground_contradictions = tuple(
            observation["internal_features"].get(
                "glyph_non_badge_art_candidate"
            )
            is True
            for observation in observations
        )
        if all(oversized_foreground_contradictions):
            # This is deliberately narrower than a missing-glyph veto.  A real
            # badge whose digit is fragmented or absent remains a detail
            # candidate.  Only one stable, centre-colocated foreground component
            # that is physically too large to be a one-digit badge is negative
            # evidence; bounded off-lane interpolation specks do not turn that
            # art block into a digit.
            return CustomizationBadgeDecision(
                CustomizationBadgeState.CONFIDENT_ZERO,
                0,
                "all_frames_oversized_center_foreground_contradiction",
            )
        return CustomizationBadgeDecision(
            CustomizationBadgeState.DETAIL_CANDIDATE,
            None,
            "stable_seeded_plate_requires_detail",
        )
    if not any(seeded_plate_states):
        return CustomizationBadgeDecision(
            CustomizationBadgeState.CONFIDENT_ZERO,
            0,
            "all_frames_seeded_plate_absence",
        )
    return CustomizationBadgeDecision(
        CustomizationBadgeState.AMBIGUOUS,
        None,
        "seeded_plate_frame_disagreement",
    )
