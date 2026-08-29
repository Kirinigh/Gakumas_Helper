"""Three-frame card-face evidence for generic-cost customizations."""

from __future__ import annotations

import json
import hashlib
from typing import Any
from pathlib import Path
from dataclasses import replace, dataclass
from collections.abc import Sequence


class GenericCostReferenceError(RuntimeError):
    """The fixed cost references or one live observation are inconclusive."""


@dataclass(frozen=True)
class GenericCostPrediction:
    status: str
    value: int | None = None
    candidate_value: int | None = None
    best_error: float | None = None
    class_margin: float | None = None
    pixel_count: int = 0
    component_boxes: tuple[tuple[int, int, int, int, int], ...] = ()
    cutout_candidate_value: int | None = None
    cutout_best_error: float | None = None
    cutout_class_margin: float | None = None
    cutout_pixel_count: int = 0
    carrier_descriptor_hex: str = ""
    symbol_candidate_value: int | None = None
    symbol_best_error: float | None = None
    symbol_class_margin: float | None = None
    symbol_descriptor_hex: str = ""
    digit_candidate_value: int | None = None
    digit_best_error: float | None = None
    digit_class_margin: float | None = None
    digit_descriptor_hex: str = ""
    evidence_mode: str = ""
    zero_hole_box: tuple[int, int, int, int, int] | None = None
    zero_alternative_error: float | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class GenericCostReferenceGallery:
    """Classify only the two costs allowed by one card's fixed DSL."""

    NORMALIZED_SIZE = 96
    ROI = (64, 62, 32, 34)
    GLYPH_SIZE = 24
    ZERO_MIN_OUTER_SOLIDITY = 0.85
    ZERO_MAX_HULL_BBOX_FILL = 0.98

    def __init__(
        self,
        *,
        generic_card_ids: Any,
        base_values: Any,
        cost_kinds: Any,
        base_masks: Any,
        template_values: Any,
        template_kinds: Any,
        template_masks: Any,
        base_symbols: Any,
        template_symbols: Any,
        maximum_mask_error: float,
        minimum_class_margin: float,
        maximum_medium_error: float,
        minimum_medium_margin: float,
        maximum_high_error: float,
        maximum_cutout_error: float,
        minimum_cutout_margin: float,
        maximum_digit_error: float,
        minimum_digit_margin: float,
    ) -> None:
        import numpy as np

        self.generic_card_ids = np.asarray(generic_card_ids, dtype=np.int16)
        self.base_values = np.asarray(base_values, dtype=np.int8)
        self.cost_kinds = tuple(str(value) for value in cost_kinds)
        self.base_masks = np.asarray(base_masks, dtype=np.uint8)
        self.template_values = np.asarray(template_values, dtype=np.int8)
        self.template_kinds = tuple(str(value) for value in template_kinds)
        self.template_masks = np.asarray(template_masks, dtype=np.uint8)
        self.base_symbols = np.asarray(base_symbols, dtype=np.uint8)
        self.template_symbols = np.asarray(template_symbols, dtype=np.uint8)
        card_count = len(self.generic_card_ids)
        template_count = len(self.template_values)
        if (
            card_count < 1
            or len(set(int(value) for value in self.generic_card_ids)) != card_count
            or self.base_values.shape != (card_count,)
            or len(self.cost_kinds) != card_count
            or self.base_masks.shape != (card_count, 24, 24)
            or self.base_symbols.shape != (card_count, 2, 24, 24)
            or template_count < 1
            or len(self.template_kinds) != template_count
            or self.template_masks.shape != (template_count, 24, 24)
            or self.template_symbols.shape != (template_count, 2, 24, 24)
        ):
            raise GenericCostReferenceError("generic-cost reference arrays are inconsistent")
        if (
            maximum_mask_error <= 0
            or minimum_class_margin < 0
            or maximum_medium_error < maximum_mask_error
            or minimum_medium_margin < minimum_class_margin
            or maximum_high_error < maximum_medium_error
            or maximum_cutout_error <= 0
            or minimum_cutout_margin < minimum_class_margin
            or maximum_digit_error <= 0
            or minimum_digit_margin < minimum_class_margin
        ):
            raise GenericCostReferenceError("generic-cost thresholds are invalid")
        self.maximum_mask_error = float(maximum_mask_error)
        self.minimum_class_margin = float(minimum_class_margin)
        self.maximum_medium_error = float(maximum_medium_error)
        self.minimum_medium_margin = float(minimum_medium_margin)
        self.maximum_high_error = float(maximum_high_error)
        self.maximum_cutout_error = float(maximum_cutout_error)
        self.minimum_cutout_margin = float(minimum_cutout_margin)
        self.maximum_digit_error = float(maximum_digit_error)
        self.minimum_digit_margin = float(minimum_digit_margin)
        self._card_indices = {
            int(card_id): index
            for index, card_id in enumerate(self.generic_card_ids)
        }
        self._template_indices = {
            (kind, int(value)): index
            for index, (kind, value) in enumerate(
                zip(self.template_kinds, self.template_values, strict=True)
            )
        }
        if len(self._template_indices) != template_count:
            raise GenericCostReferenceError("generic-cost templates repeat a kind/value pair")
        digit_reference_pools: dict[tuple[str, int], list[Any]] = {}
        for kind, value, symbol in zip(
            self.cost_kinds,
            self.base_values,
            self.base_symbols,
            strict=True,
        ):
            digit = self._digit_descriptor(symbol)
            if digit is not None:
                digit_reference_pools.setdefault((kind, int(value)), []).append(digit)
        for kind, value, symbol in zip(
            self.template_kinds,
            self.template_values,
            self.template_symbols,
            strict=True,
        ):
            digit = self._digit_descriptor(symbol)
            if digit is not None:
                digit_reference_pools.setdefault((kind, int(value)), []).append(digit)
        self._digit_reference_pools = {
            key: tuple(references)
            for key, references in digit_reference_pools.items()
        }

    @classmethod
    def load(cls, root: str | Path) -> "GenericCostReferenceGallery":
        import numpy as np

        model_root = Path(root)
        manifest = json.loads(
            (model_root / "manifest.json").read_text(encoding="utf-8-sig")
        )
        if manifest.get("schema_version") != 1:
            raise GenericCostReferenceError("unsupported generic-cost manifest schema")
        gallery_info = manifest.get("gallery", {})
        gallery_path = model_root / str(gallery_info.get("path", ""))
        expected_hash = str(gallery_info.get("sha256", "")).upper()
        if not expected_hash or _sha256_file(gallery_path) != expected_hash:
            raise GenericCostReferenceError("generic-cost gallery SHA-256 mismatch")
        runtime = manifest.get("runtime", {})
        with np.load(gallery_path, allow_pickle=False) as payload:
            return cls(
                generic_card_ids=payload["generic_card_ids"],
                base_values=payload["base_values"],
                cost_kinds=payload["cost_kinds"],
                base_masks=payload["base_masks"],
                template_values=payload["template_values"],
                template_kinds=payload["template_kinds"],
                template_masks=payload["template_masks"],
                base_symbols=payload["base_symbols"],
                template_symbols=payload["template_symbols"],
                maximum_mask_error=float(runtime["maximum_mask_error"]),
                minimum_class_margin=float(runtime["minimum_class_margin"]),
                maximum_medium_error=float(runtime["maximum_medium_error"]),
                minimum_medium_margin=float(runtime["minimum_medium_margin"]),
                maximum_high_error=float(runtime["maximum_high_error"]),
                maximum_cutout_error=float(runtime["maximum_cutout_error"]),
                minimum_cutout_margin=float(runtime["minimum_cutout_margin"]),
                maximum_digit_error=float(runtime["maximum_digit_error"]),
                minimum_digit_margin=float(runtime["minimum_digit_margin"]),
            )

    @staticmethod
    def _cost_color_mask(image: Any, kind: str) -> Any:
        import cv2
        import numpy as np

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(image.astype(np.int16))
        if kind == "cost":
            hsv_mask = cv2.inRange(hsv, (30, 55, 45), (100, 255, 255)) > 0
            dominant = (g >= 65) & ((g - np.maximum(r, b)) >= 14)
        elif kind == "stamina":
            hsv_mask = (
                (cv2.inRange(hsv, (0, 45, 50), (15, 255, 255)) > 0)
                | (cv2.inRange(hsv, (165, 45, 50), (179, 255, 255)) > 0)
            )
            dominant = (r >= 65) & ((r - np.maximum(g, b)) >= 14)
        else:
            raise GenericCostReferenceError(f"unsupported card-face cost kind: {kind!r}")
        return hsv_mask & dominant

    @staticmethod
    def _shifted_error(query: Any, reference: Any) -> float:
        import numpy as np

        best = float("inf")
        height, width = query.shape
        for shift_y in (-1, 0, 1):
            for shift_x in (-1, 0, 1):
                shifted = np.zeros_like(reference)
                source_y0 = max(0, -shift_y)
                source_y1 = min(height, height - shift_y)
                source_x0 = max(0, -shift_x)
                source_x1 = min(width, width - shift_x)
                target_y0 = max(0, shift_y)
                target_y1 = target_y0 + (source_y1 - source_y0)
                target_x0 = max(0, shift_x)
                target_x1 = target_x0 + (source_x1 - source_x0)
                shifted[target_y0:target_y1, target_x0:target_x1] = reference[
                    source_y0:source_y1,
                    source_x0:source_x1,
                ]
                best = min(best, float(np.mean(query != shifted)))
        return best

    @classmethod
    def _normalise_glyph_component(cls, mask: Any) -> Any | None:
        """Normalize the anchored cost glyph independently of card-box scale."""

        import cv2
        import numpy as np

        binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            8,
        )
        if component_count <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        label = int(np.argmax(areas)) + 1
        x, y, width, height, area = (
            int(value) for value in stats[label]
        )
        if width < 8 or height < 8 or area < 32:
            return None
        component = (labels[y : y + height, x : x + width] == label).astype(
            np.uint8
        )
        return cv2.resize(
            component,
            (cls.GLYPH_SIZE, cls.GLYPH_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

    @classmethod
    def _anchored_carrier(
        cls,
        mask: Any,
    ) -> tuple[Any | None, tuple[Any, ...], tuple[int, int, int, int] | None]:
        """Group the complete lower-right cost carrier before normalization."""

        import cv2
        import numpy as np

        binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        if binary.shape != (cls.NORMALIZED_SIZE, cls.NORMALIZED_SIZE):
            return None, (), None
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            8,
        )
        boxes = tuple(
            tuple(int(value) for value in row) for row in stats[1:]
        )
        if component_count <= 1:
            return None, boxes, None
        search_x, search_y, search_width, search_height = cls.ROI
        search_right = search_x + search_width
        search_bottom = search_y + search_height
        eligible: list[int] = []
        seeds: list[int] = []
        for label in range(1, component_count):
            x, y, width, height, area = (int(value) for value in stats[label])
            right = x + width
            bottom = y + height
            if (
                area < 2
                or x < search_x
                or y < search_y
                or right <= search_x
                or bottom <= search_y
                or x >= search_right
                or y >= search_bottom
            ):
                continue
            eligible.append(label)
            if (right >= 84 and bottom >= 84) or (
                right >= 82 and bottom >= 88
            ):
                seeds.append(label)
        if not seeds:
            return None, boxes, None
        selected = np.isin(labels, seeds).astype(np.uint8)
        remaining = set(eligible) - set(seeds)
        kernel = np.ones((3, 3), dtype=np.uint8)
        for _ in range(2):
            if not remaining:
                break
            neighbourhood = cv2.dilate(selected, kernel, iterations=1) > 0
            adjacent = {
                label
                for label in remaining
                if bool(np.any(neighbourhood & (labels == label)))
            }
            if not adjacent:
                break
            selected |= np.isin(labels, tuple(adjacent)).astype(np.uint8)
            remaining -= adjacent
        points_y, points_x = np.where(selected > 0)
        if points_x.size < 32:
            return None, boxes, None
        x0, x1 = int(points_x.min()), int(points_x.max()) + 1
        y0, y1 = int(points_y.min()), int(points_y.max()) + 1
        if x0 < search_x or x1 > search_right or y0 < search_y or y1 > search_bottom:
            return None, boxes, None
        carrier = selected[y0:y1, x0:x1]
        if carrier.shape[0] < 8 or carrier.shape[1] < 8:
            return None, boxes, None
        return (
            cv2.resize(
                carrier,
                (cls.GLYPH_SIZE, cls.GLYPH_SIZE),
                interpolation=cv2.INTER_NEAREST,
            ),
            boxes,
            (x0, y0, x1 - x0, y1 - y0),
        )

    @staticmethod
    def _glyph_cutout(component: Any) -> Any | None:
        """Keep only the symbol cut out of the coloured cost carrier.

        The live card-box scale changes the coloured carrier much more than it
        changes the rendered ``-N`` symbol.  Comparing the carrier therefore
        confuses values such as 7 and 5.  Its convex support is scale-stable;
        subtracting the coloured pixels leaves the low-dimensional symbol and
        discards the shared carrier fill.
        """

        import cv2
        import numpy as np

        binary = (np.asarray(component, dtype=np.uint8) > 0).astype(np.uint8)
        points_y, points_x = np.where(binary > 0)
        if points_x.size < 8:
            return None
        hull = cv2.convexHull(
            np.stack((points_x, points_y), axis=1).astype(np.int32)
        )
        support = np.zeros_like(binary)
        cv2.fillConvexPoly(support, hull, 1)
        cutout = ((support > 0) & (binary == 0)).astype(np.uint8)
        if int(cutout.sum()) < 4:
            return None
        return cutout

    @classmethod
    def _symbol_descriptor(
        cls,
        image: Any,
        color_mask: Any,
        carrier_box: tuple[int, int, int, int],
    ) -> Any | None:
        """Extract achromatic light fill and dark outline of the ``-N`` text."""

        import cv2
        import numpy as np

        array = np.asarray(image)
        mask = (np.asarray(color_mask, dtype=np.uint8) > 0).astype(np.uint8)
        x, y, width, height = carrier_box
        crop = array[y : y + height, x : x + width, :3]
        mask_crop = mask[y : y + height, x : x + width]
        if crop.shape[:2] != (height, width) or min(width, height) < 8:
            return None
        normalized = cv2.resize(
            np.ascontiguousarray(crop),
            (cls.GLYPH_SIZE, cls.GLYPH_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        normalized_color = cv2.resize(
            mask_crop,
            (cls.GLYPH_SIZE, cls.GLYPH_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        points_y, points_x = np.where(normalized_color > 0)
        if points_x.size < 32:
            return None
        hull = cv2.convexHull(
            np.stack((points_x, points_y), axis=1).astype(np.int32)
        )
        support = np.zeros_like(normalized_color)
        cv2.fillConvexPoly(support, hull, 1)
        support = cv2.erode(
            support,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ) > 0
        hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
        outside_color = normalized_color == 0
        light = (
            support
            & outside_color
            & (hsv[:, :, 1] <= 105)
            & (hsv[:, :, 2] >= 135)
        )
        dark = support & outside_color & (hsv[:, :, 2] <= 115)
        descriptor = np.stack((light, dark)).astype(np.uint8)
        if int(descriptor.sum()) < 8:
            return None
        return descriptor

    @staticmethod
    def _shifted_descriptor_error(query: Any, reference: Any) -> float:
        import numpy as np

        query_array = np.asarray(query, dtype=np.uint8)
        reference_array = np.asarray(reference, dtype=np.uint8)
        if query_array.shape != (2, 24, 24) or reference_array.shape != (2, 24, 24):
            return float("inf")
        best = float("inf")
        for shift_y in (-1, 0, 1):
            for shift_x in (-1, 0, 1):
                shifted = np.zeros_like(reference_array)
                source_y0 = max(0, -shift_y)
                source_y1 = min(24, 24 - shift_y)
                source_x0 = max(0, -shift_x)
                source_x1 = min(24, 24 - shift_x)
                target_y0 = max(0, shift_y)
                target_y1 = target_y0 + (source_y1 - source_y0)
                target_x0 = max(0, shift_x)
                target_x1 = target_x0 + (source_x1 - source_x0)
                shifted[:, target_y0:target_y1, target_x0:target_x1] = (
                    reference_array[:, source_y0:source_y1, source_x0:source_x1]
                )
                best = min(best, float(np.mean(query_array != shifted)))
        return best

    @classmethod
    def _digit_descriptor(cls, symbol: Any) -> Any | None:
        """Remove the minus sign and normalize only the rendered numeral."""

        import cv2
        import numpy as np

        descriptor = np.asarray(symbol, dtype=np.uint8)
        if descriptor.shape != (2, 24, 24):
            return None
        light = (descriptor[0] > 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            light,
            8,
        )
        keep: list[int] = []
        for label in range(1, component_count):
            x, _, width, height, area = (int(value) for value in stats[label])
            if area < 3:
                continue
            if x < 10 and width >= height * 1.5:
                continue
            keep.append(label)
        if not keep:
            return None
        numeral = np.isin(labels, keep).astype(np.uint8)
        points_y, points_x = np.where(numeral > 0)
        crop = numeral[
            int(points_y.min()) : int(points_y.max()) + 1,
            int(points_x.min()) : int(points_x.max()) + 1,
        ]
        # Antialiasing can leave a few connected edge pixels on one render but
        # not another.  If those pixels define the resize box, the same digit
        # is stretched to a different aspect ratio.  Bound the glyph by its
        # row/column support before resizing while retaining every pixel inside
        # that robust core.
        column_threshold = max(2, int(np.ceil(crop.shape[0] * 0.15)))
        row_threshold = max(2, int(np.ceil(crop.shape[1] * 0.15)))
        supported_columns = np.where(crop.sum(axis=0) >= column_threshold)[0]
        supported_rows = np.where(crop.sum(axis=1) >= row_threshold)[0]
        if supported_columns.size >= 2 and supported_rows.size >= 2:
            crop = crop[
                int(supported_rows[0]) : int(supported_rows[-1]) + 1,
                int(supported_columns[0]) : int(supported_columns[-1]) + 1,
            ]
        if min(crop.shape) < 2:
            return None
        return cv2.resize(
            crop,
            (cls.GLYPH_SIZE, cls.GLYPH_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

    @classmethod
    def _explicit_zero_hole(
        cls,
        digit: Any,
    ) -> tuple[int, int, int, int, int] | None:
        """Return the single enclosed counter of an explicit rendered ``0``.

        The live contest renderer retains the coloured lower-right carrier for
        a zero generic cost and draws an explicit ``0`` inside it.  Treating a
        missing carrier or an empty ROI as zero would turn failed extraction
        into false positive customization evidence.

        This uses a topological core plus a coarse outer-geometry gate rather
        than a looser template threshold.  It is reached only for the fixed
        ``{0, 2}`` hypotheses; the normalized numeral must contain exactly one
        sizeable background component that is fully enclosed by one dominant,
        convex numeral-like foreground glyph rather than an axis-aligned UI
        frame.
        """

        import cv2
        import numpy as np

        binary = (np.asarray(digit, dtype=np.uint8) > 0).astype(np.uint8)
        if binary.shape != (cls.GLYPH_SIZE, cls.GLYPH_SIZE):
            return None
        foreground_count, foreground_labels, foreground_stats, _ = (
            cv2.connectedComponentsWithStats(binary, 8)
        )
        foreground_areas = tuple(
            int(foreground_stats[label, cv2.CC_STAT_AREA])
            for label in range(1, foreground_count)
        )
        foreground_pixels = int(binary.sum())
        if (
            foreground_pixels < 64
            or not foreground_areas
            or max(foreground_areas) < int(foreground_pixels * 0.9)
        ):
            return None
        dominant_label = int(np.argmax(foreground_areas)) + 1
        dominant = (foreground_labels == dominant_label).astype(np.uint8)
        _, _, foreground_width, foreground_height, _ = (
            int(value) for value in foreground_stats[dominant_label]
        )
        contours, _ = cv2.findContours(
            dominant,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        points = cv2.findNonZero(dominant)
        if len(contours) != 1 or points is None:
            return None
        outer = np.zeros_like(dominant)
        cv2.drawContours(outer, contours, -1, 1, cv2.FILLED)
        hull = cv2.convexHull(points)
        hull_mask = np.zeros_like(dominant)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        outer_support = int(outer.sum())
        hull_support = int(hull_mask.sum())
        foreground_box_area = foreground_width * foreground_height
        if (
            hull_support < 1
            or foreground_box_area < 1
            or outer_support / float(hull_support)
            < cls.ZERO_MIN_OUTER_SOLIDITY
            or hull_support / float(foreground_box_area)
            >= cls.ZERO_MAX_HULL_BBOX_FILL
        ):
            return None

        background = (binary == 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            background,
            8,
        )
        holes: list[tuple[int, int, int, int, int, int]] = []
        for label in range(1, component_count):
            x, y, width, height, area = (
                int(value) for value in stats[label]
            )
            if (
                x == 0
                or y == 0
                or x + width == cls.GLYPH_SIZE
                or y + height == cls.GLYPH_SIZE
                or width < 7
                or height < 10
                or area < 60
            ):
                continue
            # Corner occupancy and fill ratio inside the counter's own bounding
            # box are rasterisation details, not topology.  Real zero counters
            # can touch a corner or fill more than 90% of that box after
            # nearest-neighbour normalisation.  The outer-glyph gate above owns
            # UI-frame rejection instead.
            holes.append((label, x, y, width, height, area))
        if len(holes) != 1:
            return None
        label, x, y, width, height, area = holes[0]
        if labels[cls.GLYPH_SIZE // 2, cls.GLYPH_SIZE // 2] != label:
            return None
        return (x, y, width, height, area)

    def classify_mask(
        self,
        mask: Any,
        *,
        card_id: int,
        allowed_values: Sequence[int],
        symbol_descriptor: Any | None = None,
    ) -> GenericCostPrediction:
        import numpy as np

        card_index = self._card_indices.get(card_id)
        if card_index is None:
            return GenericCostPrediction("REFERENCE_CARD_MISSING")
        allowed = tuple(dict.fromkeys(int(value) for value in allowed_values))
        if len(allowed) != 2 or any(value < 0 for value in allowed):
            return GenericCostPrediction("INVALID_HYPOTHESES")
        query = np.asarray(mask, dtype=np.uint8)
        if query.shape != (24, 24):
            return GenericCostPrediction("INVALID_ROI")
        query = query > 0
        pixel_count = int(query.sum())
        kind = self.cost_kinds[card_index]
        base_value = int(self.base_values[card_index])
        normalized_query = (query > 0).astype(np.uint8)
        if int(normalized_query.sum()) < 32:
            return GenericCostPrediction(
                "GLYPH_COMPONENT_MISSING",
                pixel_count=pixel_count,
            )
        ranked: list[tuple[float, int]] = []
        cutout_query = self._glyph_cutout(normalized_query)
        cutout_ranked: list[tuple[float, int]] = []
        symbol_query = (
            np.asarray(symbol_descriptor, dtype=np.uint8)
            if symbol_descriptor is not None
            else None
        )
        if symbol_query is not None and symbol_query.shape != (2, 24, 24):
            return GenericCostPrediction("INVALID_SYMBOL_DESCRIPTOR")
        symbol_ranked: list[tuple[float, int]] = []
        digit_query = (
            self._digit_descriptor(symbol_query)
            if symbol_query is not None
            else None
        )
        zero_hole = (
            self._explicit_zero_hole(digit_query)
            if digit_query is not None and set(allowed) == {0, 2}
            else None
        )
        digit_ranked: list[tuple[float, int]] = []
        for value in allowed:
            if value == 0:
                continue
            references = []
            if value == base_value:
                references.append(self.base_masks[card_index] > 0)
            template_index = self._template_indices.get((kind, value))
            if template_index is not None:
                references.append(self.template_masks[template_index] > 0)
            if not references:
                return GenericCostPrediction("REFERENCE_VALUE_MISSING", pixel_count=pixel_count)
            normalized_references = tuple(
                (reference > 0).astype(np.uint8) for reference in references
            )
            if not normalized_references:
                return GenericCostPrediction(
                    "REFERENCE_GLYPH_MISSING",
                    pixel_count=pixel_count,
                )
            ranked.append(
                (
                    min(
                        self._shifted_error(normalized_query, reference)
                        for reference in normalized_references
                    ),
                    value,
                )
            )
            cutout_references = tuple(
                cutout
                for reference in normalized_references
                if (cutout := self._glyph_cutout(reference)) is not None
            )
            if cutout_query is not None and cutout_references:
                cutout_ranked.append(
                    (
                        min(
                            self._shifted_error(cutout_query, reference)
                            for reference in cutout_references
                        ),
                        value,
                    )
                )
            symbol_references = []
            if value == base_value:
                symbol_references.append(self.base_symbols[card_index])
            if template_index is not None:
                symbol_references.append(self.template_symbols[template_index])
            if symbol_query is not None and symbol_references:
                symbol_ranked.append(
                    (
                        min(
                            self._shifted_descriptor_error(symbol_query, reference)
                            for reference in symbol_references
                        ),
                        value,
                    )
                )
            digit_references = tuple(
                digit
                for reference in symbol_references
                if (digit := self._digit_descriptor(reference)) is not None
            )
            if digit_query is not None and digit_references:
                digit_ranked.append(
                    (
                        min(
                            self._shifted_error(digit_query, reference)
                            for reference in digit_references
                        ),
                        value,
                    )
                )
        ranked.sort()
        best_error, value = ranked[0]
        margin = (
            ranked[1][0] - best_error
            if len(ranked) > 1
            else float("inf")
        )
        candidate_value = value
        cutout_ranked.sort()
        if cutout_ranked:
            cutout_best_error, cutout_candidate_value = cutout_ranked[0]
            cutout_class_margin = (
                cutout_ranked[1][0] - cutout_best_error
                if len(cutout_ranked) > 1
                else float("inf")
            )
        else:
            cutout_best_error = None
            cutout_candidate_value = None
            cutout_class_margin = None
        symbol_ranked.sort()
        if symbol_ranked:
            symbol_best_error, symbol_candidate_value = symbol_ranked[0]
            symbol_class_margin = (
                symbol_ranked[1][0] - symbol_best_error
                if len(symbol_ranked) > 1
                else float("inf")
            )
        else:
            symbol_best_error = None
            symbol_candidate_value = None
            symbol_class_margin = None
        digit_ranked.sort()
        if digit_ranked:
            digit_best_error, digit_candidate_value = digit_ranked[0]
            digit_class_margin = (
                digit_ranked[1][0] - digit_best_error
                if len(digit_ranked) > 1
                else float("inf")
            )
        else:
            digit_best_error = None
            digit_candidate_value = None
            digit_class_margin = None
        if zero_hole is not None:
            alternative_references = self._digit_reference_pools.get((kind, 2), ())
            alternative_error = (
                min(
                    self._shifted_error(digit_query, reference)
                    for reference in alternative_references
                )
                if alternative_references
                else None
            )
            zero_certified = (
                alternative_error is not None
                and alternative_error > self.maximum_digit_error
            )
            return GenericCostPrediction(
                "MEASURED" if zero_certified else "ZERO_TOPOLOGY_CONFLICT",
                value=0 if zero_certified else None,
                candidate_value=0,
                pixel_count=pixel_count,
                carrier_descriptor_hex=np.packbits(
                    normalized_query.reshape(-1)
                ).tobytes().hex(),
                symbol_candidate_value=0,
                symbol_descriptor_hex=np.packbits(
                    symbol_query.reshape(-1)
                ).tobytes().hex(),
                digit_candidate_value=0,
                digit_descriptor_hex=np.packbits(
                    digit_query.reshape(-1)
                ).tobytes().hex(),
                evidence_mode=(
                    "explicit_zero_hole"
                    if zero_certified
                    else "explicit_zero_hole_conflict"
                ),
                zero_hole_box=zero_hole,
                zero_alternative_error=(
                    round(alternative_error, 6)
                    if alternative_error is not None
                    else None
                ),
            )
        cutout_certified = (
            cutout_candidate_value == candidate_value
            and cutout_best_error is not None
            and cutout_best_error <= self.maximum_cutout_error
            and cutout_class_margin is not None
            and cutout_class_margin >= self.minimum_cutout_margin
        )
        digit_certified = (
            digit_candidate_value is not None
            and digit_best_error is not None
            and digit_best_error <= self.maximum_digit_error
            and digit_class_margin is not None
            and digit_class_margin >= self.minimum_digit_margin
        )
        if best_error > self.maximum_high_error:
            status = "MASK_ERROR_TOO_HIGH"
            value = None
        elif symbol_query is not None and not digit_certified:
            status = "DIGIT_EVIDENCE_NOT_CERTIFIED"
            value = None
        elif symbol_query is not None:
            status = "MEASURED"
            value = digit_candidate_value
        elif best_error > self.maximum_medium_error and not cutout_certified:
            status = "HIGH_ERROR_CUTOUT_NOT_CERTIFIED"
            value = None
        elif best_error > self.maximum_medium_error:
            status = "MEASURED"
        elif (
            best_error > self.maximum_mask_error
            and margin < self.minimum_medium_margin
        ):
            status = "MEDIUM_ERROR_MARGIN_TOO_SMALL"
            value = None
        elif margin < self.minimum_class_margin:
            status = "CLASS_MARGIN_TOO_SMALL"
            value = None
        else:
            status = "MEASURED"
        return GenericCostPrediction(
            status,
            value=value,
            candidate_value=candidate_value,
            best_error=round(best_error, 6),
            class_margin=round(margin, 6),
            pixel_count=pixel_count,
            cutout_candidate_value=cutout_candidate_value,
            cutout_best_error=(
                round(cutout_best_error, 6)
                if cutout_best_error is not None
                else None
            ),
            cutout_class_margin=(
                round(cutout_class_margin, 6)
                if cutout_class_margin is not None
                else None
            ),
            cutout_pixel_count=(
                int(cutout_query.sum()) if cutout_query is not None else 0
            ),
            carrier_descriptor_hex=np.packbits(
                normalized_query.reshape(-1)
            ).tobytes().hex(),
            symbol_candidate_value=symbol_candidate_value,
            symbol_best_error=(
                round(symbol_best_error, 6)
                if symbol_best_error is not None
                else None
            ),
            symbol_class_margin=(
                round(symbol_class_margin, 6)
                if symbol_class_margin is not None
                else None
            ),
            symbol_descriptor_hex=(
                np.packbits(symbol_query.reshape(-1)).tobytes().hex()
                if symbol_query is not None
                else ""
            ),
            digit_candidate_value=digit_candidate_value,
            digit_best_error=(
                round(digit_best_error, 6)
                if digit_best_error is not None
                else None
            ),
            digit_class_margin=(
                round(digit_class_margin, 6)
                if digit_class_margin is not None
                else None
            ),
            digit_descriptor_hex=(
                np.packbits(digit_query.reshape(-1)).tobytes().hex()
                if digit_query is not None
                else ""
            ),
        )

    def classify_symbol_descriptor(
        self,
        symbol_descriptor: Any,
        *,
        card_id: int,
        allowed_values: Sequence[int],
    ) -> GenericCostPrediction:
        """Classify only the achromatic numeral inside the fixed cost ROI."""

        import numpy as np

        card_index = self._card_indices.get(card_id)
        if card_index is None:
            return GenericCostPrediction("REFERENCE_CARD_MISSING")
        allowed = tuple(dict.fromkeys(int(value) for value in allowed_values))
        if len(allowed) != 2 or any(value < 0 for value in allowed):
            return GenericCostPrediction("INVALID_HYPOTHESES")
        symbol = np.asarray(symbol_descriptor, dtype=np.uint8)
        if symbol.shape != (2, 24, 24):
            return GenericCostPrediction("INVALID_SYMBOL_DESCRIPTOR")
        digit = self._digit_descriptor(symbol)
        if digit is None:
            return GenericCostPrediction("DIGIT_EVIDENCE_MISSING")
        zero_hole = (
            self._explicit_zero_hole(digit)
            if set(allowed) == {0, 2}
            else None
        )
        kind = self.cost_kinds[card_index]
        base_value = int(self.base_values[card_index])
        ranked: list[tuple[float, int]] = []
        for value in allowed:
            if value == 0:
                continue
            template_index = self._template_indices.get((kind, value))
            symbol_references = []
            if value == base_value:
                symbol_references.append(self.base_symbols[card_index])
            if template_index is not None:
                symbol_references.append(self.template_symbols[template_index])
            digit_references = tuple(
                reference_digit
                for reference in symbol_references
                if (
                    reference_digit := self._digit_descriptor(reference)
                )
                is not None
            )
            if digit_references:
                ranked.append(
                    (
                        min(
                            self._shifted_error(digit, reference)
                            for reference in digit_references
                        ),
                        value,
                    )
                )
        ranked.sort()
        if not ranked:
            return GenericCostPrediction("REFERENCE_VALUE_MISSING")
        if zero_hole is not None:
            alternative_references = self._digit_reference_pools.get((kind, 2), ())
            alternative_error = (
                min(
                    self._shifted_error(digit, reference)
                    for reference in alternative_references
                )
                if alternative_references
                else None
            )
            zero_certified = (
                alternative_error is not None
                and alternative_error > self.maximum_digit_error
            )
            return GenericCostPrediction(
                "MEASURED" if zero_certified else "ZERO_TOPOLOGY_CONFLICT",
                value=0 if zero_certified else None,
                candidate_value=0,
                symbol_candidate_value=0,
                symbol_descriptor_hex=np.packbits(
                    symbol.reshape(-1)
                ).tobytes().hex(),
                digit_candidate_value=0,
                digit_descriptor_hex=np.packbits(
                    digit.reshape(-1)
                ).tobytes().hex(),
                evidence_mode=(
                    "explicit_zero_hole"
                    if zero_certified
                    else "explicit_zero_hole_conflict"
                ),
                zero_hole_box=zero_hole,
                zero_alternative_error=(
                    round(alternative_error, 6)
                    if alternative_error is not None
                    else None
                ),
            )
        best_error, candidate_value = ranked[0]
        class_margin = (
            ranked[1][0] - best_error
            if len(ranked) > 1
            else float("inf")
        )
        measured = (
            best_error <= self.maximum_digit_error
            and class_margin >= self.minimum_digit_margin
        )
        return GenericCostPrediction(
            "MEASURED" if measured else "DIGIT_EVIDENCE_NOT_CERTIFIED",
            value=candidate_value if measured else None,
            candidate_value=candidate_value,
            symbol_descriptor_hex=np.packbits(symbol.reshape(-1)).tobytes().hex(),
            digit_candidate_value=candidate_value,
            digit_best_error=round(best_error, 6),
            digit_class_margin=round(class_margin, 6),
            digit_descriptor_hex=np.packbits(digit.reshape(-1)).tobytes().hex(),
        )

    def measure(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        *,
        card_id: int,
        allowed_values: Sequence[int],
    ) -> GenericCostPrediction:
        import cv2
        import numpy as np

        card_index = self._card_indices.get(card_id)
        if card_index is None:
            return GenericCostPrediction("REFERENCE_CARD_MISSING")
        array = np.asarray(image)
        x, y, width, height = box
        crop = array[y : y + height, x : x + width, :3]
        if crop.shape[:2] != (height, width) or width < 8 or height < 8:
            return GenericCostPrediction("INVALID_ROI")
        normalized = cv2.resize(
            np.ascontiguousarray(crop),
            (self.NORMALIZED_SIZE, self.NORMALIZED_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        full_mask = self._cost_color_mask(
            normalized,
            self.cost_kinds[card_index],
        ).astype(np.uint8)
        carrier, component_boxes, carrier_box = self._anchored_carrier(full_mask)
        if carrier is None:
            crosses_cost_roi = any(
                component_x < self.ROI[0]
                and component_x + component_width > 74
                and component_y + component_height > 74
                for component_x, component_y, component_width, component_height, _
                in component_boxes
            )
            if crosses_cost_roi:
                fixed_symbol = self._symbol_descriptor(
                    normalized,
                    full_mask,
                    self.ROI,
                )
                if fixed_symbol is not None:
                    digit_prediction = self.classify_symbol_descriptor(
                        fixed_symbol,
                        card_id=card_id,
                        allowed_values=allowed_values,
                    )
                    if digit_prediction.status == "MEASURED":
                        return replace(
                            digit_prediction,
                            component_boxes=component_boxes,
                        )
                return GenericCostPrediction(
                    "BADGE_INTRUSION",
                    component_boxes=component_boxes,
                    symbol_descriptor_hex=(
                        digit_prediction.symbol_descriptor_hex
                        if fixed_symbol is not None
                        else ""
                    ),
                    digit_candidate_value=(
                        digit_prediction.digit_candidate_value
                        if fixed_symbol is not None
                        else None
                    ),
                    digit_best_error=(
                        digit_prediction.digit_best_error
                        if fixed_symbol is not None
                        else None
                    ),
                    digit_class_margin=(
                        digit_prediction.digit_class_margin
                        if fixed_symbol is not None
                        else None
                    ),
                    digit_descriptor_hex=(
                        digit_prediction.digit_descriptor_hex
                        if fixed_symbol is not None
                        else ""
                    ),
                )
            roi_x, roi_y, roi_width, roi_height = self.ROI
            residue = int(
                full_mask[
                    roi_y : roi_y + roi_height,
                    roi_x : roi_x + roi_width,
                ].sum()
            )
            return GenericCostPrediction(
                "COST_CARRIER_MISSING",
                pixel_count=residue,
                component_boxes=component_boxes,
            )
        if carrier_box is None:
            return GenericCostPrediction(
                "COST_CARRIER_BOX_MISSING",
                component_boxes=component_boxes,
            )
        symbol_descriptor = self._symbol_descriptor(
            normalized,
            full_mask,
            carrier_box,
        )
        if symbol_descriptor is None:
            return GenericCostPrediction(
                "COST_SYMBOL_MISSING",
                component_boxes=component_boxes,
            )
        return replace(
            self.classify_mask(
                carrier,
                card_id=card_id,
                allowed_values=allowed_values,
                symbol_descriptor=symbol_descriptor,
            ),
            component_boxes=component_boxes,
        )


def stable_generic_cost_value(
    predictions: Sequence[GenericCostPrediction],
) -> int:
    """Return one value only after all three stable frames agree."""

    if len(predictions) != 3:
        raise GenericCostReferenceError(
            f"generic-cost evidence requires three frames, found {len(predictions)}"
        )
    if any(prediction.status != "MEASURED" for prediction in predictions):
        raise GenericCostReferenceError(
            "generic-cost frame was inconclusive: " + repr(tuple(predictions))
        )
    values = tuple(prediction.value for prediction in predictions)
    if None in values or len(set(values)) != 1:
        raise GenericCostReferenceError(
            f"generic-cost values disagreed across stable frames: {values!r}"
        )
    return int(values[0])
