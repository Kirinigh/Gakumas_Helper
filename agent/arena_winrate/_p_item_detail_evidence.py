"""Per-opening P-item source evidence; no input or retry budget is owned here."""

from __future__ import annotations

import re
import statistics
import unicodedata
from typing import Any
from collections.abc import Sequence

from .reader import ArenaReaderError
from ._reader_visual import _box, _text
from ._reader_evidence import _NormalizedPItemRow, _PItemOcrObservation


class PItemDetailEvidence:
    """Source and title evidence owned by exactly one P-item detail opening."""

    def __init__(
        self,
        backend,
        candidate_ids,
        plan,
        source_images,
        slot_scope,
        source_boxes,
        global_title_scope,
        visual_tiebreak_ids,
        unrepresented_ids,
        clock,
    ):
        self.backend = backend
        self.candidate_ids = candidate_ids
        self.plan = plan
        self.source_images = source_images
        self.slot_scope = slot_scope
        self.source_boxes = source_boxes
        self.global_title_scope = global_title_scope
        self.visual_tiebreak_ids = visual_tiebreak_ids
        self.unrepresented_ids = unrepresented_ids
        self.clock = clock

    def prepare_source(self):
        self.stable_source_title_signatures: frozenset[tuple[str, ...]] = frozenset()

        self.stable_source_title_rows: tuple[
            tuple[str, tuple[float, float, float, float]],
            ...,
        ] = ()

        self.stable_source_row_frames: tuple[
            tuple[Any, tuple[tuple[float, float, float, float], ...]],
            ...,
        ] = ()

        self.stable_source_background_frames: tuple[tuple[int, tuple[_NormalizedPItemRow, ...]], ...] = ()

        self.background_majority_required = 2

        if self.source_images:
            cache = getattr(self.backend, "_p_item_source_ocr_cache", None)
            if cache is None:
                cache = {}
                self.backend._p_item_source_ocr_cache = cache
            # ``read_p_item_ids`` clears this bounded cache once per member.
            # The frozen frame objects stay alive for all four slot
            # transactions, so their in-process identities are sufficient and
            # avoid hashing three full captures again for every slot.
            cache_key = tuple(id(image) for image in self.source_images)
            self.source_observations = cache.get(cache_key)
            if self.source_observations is None:
                ocr_started = self.clock.perf_counter()
                try:
                    self.source_observations = tuple(self.observe_p_item(source_image) for source_image in self.source_images)
                    self.backend._assert_p_item_source_generation_stable(
                        self.source_images,
                        self.source_boxes,
                    )
                except Exception as error:
                    raise ArenaReaderError(
                        "p_item_source_title_evidence_invalid",
                        "P-item detail recovery could not freeze source-page catalog-title evidence before clicking",
                    ) from error
                self.backend._add_timing(
                    "p_item_source_title_ocr",
                    self.clock.perf_counter() - ocr_started,
                )
                cache[cache_key] = self.source_observations
            signature_counts: dict[tuple[str, ...], int] = {}
            for _, _, _, source_rows, _, _ in self.source_observations:
                signature = tuple(row[1] for row in source_rows)
                if signature:
                    signature_counts[signature] = signature_counts.get(signature, 0) + 1
            strict_majority = len(self.source_observations) // 2 + 1
            self.background_majority_required = max(2, strict_majority)
            if len(self.source_observations) >= 2:
                self.stable_source_title_signatures = frozenset(
                    signature for signature, count in signature_counts.items() if count >= strict_majority
                )
                if self.stable_source_title_signatures:
                    stable_signature = next(iter(self.stable_source_title_signatures))
                    agreeing_rows = tuple(
                        source_rows
                        for _, _, _, source_rows, _, _ in self.source_observations
                        if tuple(row[1] for row in source_rows) == stable_signature
                    )
                    self.stable_source_row_frames = tuple(
                        (source_image, tuple(row[2] for row in source_rows))
                        for source_image, (_, _, _, source_rows, _, _) in zip(
                            self.source_images,
                            self.source_observations,
                            strict=True,
                        )
                        if tuple(row[1] for row in source_rows) == stable_signature
                    )
                    self.stable_source_title_rows = tuple(
                        (
                            normalized,
                            tuple(statistics.median(rows[index][2][coordinate] for rows in agreeing_rows) for coordinate in range(4)),
                        )
                        for index, normalized in enumerate(stable_signature)
                    )
                    # Extra rows never alter the old source signature or its
                    # pixel-row indexes. Only its agreeing source frames may
                    # contribute exact background evidence.
                    self.stable_source_background_frames = tuple(
                        (id(source_image), background_rows)
                        for source_image, (_, _, _, source_rows, background_rows, _) in zip(
                            self.source_images, self.source_observations, strict=True
                        )
                        if tuple(row[1] for row in source_rows) == stable_signature
                    )
                else:
                    raise ArenaReaderError(
                        "p_item_source_title_evidence_unstable",
                        "P-item detail recovery could not establish a strict-majority spatial source-row signature before clicking",
                    )

    def match_title_rows(self, text: str) -> tuple[int, ...]:
        matches: list[int] = []
        for title_row in str(text or "").splitlines():
            if not title_row.strip():
                continue
            family_matcher = getattr(self.backend.catalog, "p_item_detail_title_family_ids", None)
            if callable(family_matcher):
                row_matches = family_matcher(title_row, plan=self.plan, **self.slot_scope)
                if not row_matches:
                    recovered = family_matcher(
                        title_row,
                        plan=self.plan,
                        allow_one_substitution=True,
                        **self.slot_scope,
                    )
                    # Preserve the existing visual boundary only for a
                    # one-character OCR repair. An exact complete title
                    # always searches its full catalog family.
                    old_matches = (
                        self.backend.catalog.clicked_p_item_candidate_matches(
                            title_row,
                            candidate_p_item_ids=self.candidate_ids,
                        )
                        if recovered
                        else ()
                    )
                    if set(recovered).intersection(old_matches):
                        row_matches = recovered
            elif self.global_title_scope:
                row_matches = self.backend.catalog.clicked_p_item_global_detail_matches(
                    title_row,
                    candidate_p_item_ids=self.candidate_ids,
                    visual_tiebreak_p_item_ids=self.visual_tiebreak_ids,
                    unrepresented_p_item_ids=self.unrepresented_ids,
                    allow_one_substitution=True,
                )
            else:
                row_matches = self.backend.catalog.clicked_p_item_candidate_matches(
                    title_row,
                    candidate_p_item_ids=self.candidate_ids,
                )
            matches.extend(row_matches)
        return tuple(dict.fromkeys(matches))

    def match_title_rows_exact(self, text: str) -> tuple[int, ...]:
        matches: list[int] = []
        for title_row in str(text or "").splitlines():
            if not title_row.strip():
                continue
            family_matcher = getattr(self.backend.catalog, "p_item_detail_title_family_ids", None)
            if callable(family_matcher):
                matches.extend(family_matcher(title_row, plan=self.plan, **self.slot_scope))
            else:
                matches.extend(
                    self.backend.catalog.clicked_p_item_global_detail_matches(
                        title_row,
                        candidate_p_item_ids=self.candidate_ids,
                        visual_tiebreak_p_item_ids=self.visual_tiebreak_ids,
                        unrepresented_p_item_ids=self.unrepresented_ids,
                        allow_one_substitution=False,
                    )
                )
        return tuple(dict.fromkeys(matches))

    def normalize_title_row(self, text: str) -> str:
        return re.sub(
            r"\s+",
            "",
            unicodedata.normalize("NFKC", str(text or "")),
        )

    def title_rows(self, text: str) -> tuple[tuple[str, str], ...]:
        rows: list[tuple[str, str]] = []
        for raw_row in str(text or "").splitlines():
            normalized = self.normalize_title_row(raw_row)
            if normalized:
                rows.append((raw_row.strip(), normalized))
        return tuple(rows)

    def observe_p_item(
        self, image: Any
    ) -> tuple[
        str,
        str,
        bool,
        tuple[_NormalizedPItemRow, ...],
        tuple[_NormalizedPItemRow, ...],
        tuple[dict[str, Any], ...],
    ]:
        observation = self.backend._p_item_ocr_observation(image)

        def normalize_rows(geometric_rows: Sequence[Any]) -> tuple[_NormalizedPItemRow, ...]:
            normalized_rows_list = []
            for geometric_row in geometric_rows:
                if len(geometric_row) == 3:
                    raw, row_box, components = geometric_row
                elif len(geometric_row) == 2:
                    raw, row_box = geometric_row
                    components = ((raw, row_box),)
                else:
                    raise ArenaReaderError(
                        "p_item_ocr_panel_row_invalid",
                        "P-item OCR panel row has an unsupported shape",
                    )
                normalized = self.normalize_title_row(raw)
                if not normalized:
                    continue
                normalized_components = tuple(
                    (
                        str(component_raw).strip(),
                        self.normalize_title_row(component_raw),
                        tuple(float(value) for value in component_box),
                    )
                    for component_raw, component_box in components
                    if self.normalize_title_row(component_raw)
                )
                normalized_rows_list.append(
                    (
                        str(raw).strip(),
                        normalized,
                        tuple(float(value) for value in row_box),
                        normalized_components,
                    )
                )
            return tuple(normalized_rows_list)

        background_rows: tuple[_NormalizedPItemRow, ...] = ()
        atoms: tuple[dict[str, Any], ...] = ()
        if isinstance(observation, _PItemOcrObservation):
            full_text = observation.full_text
            panel_text = observation.panel_text
            anchors_visible = observation.anchors_visible
            normalized_rows = normalize_rows(observation.panel_rows)
            background_rows = normalize_rows(observation.background_rows)
            if observation.all_items:
                height, width = image.shape[:2]
                atoms = tuple(
                    {
                        "text": _text(item),
                        "box": tuple(
                            value / scale
                            for value, scale in zip(
                                _box(item),
                                (width / 720.0, height / 1280.0) * 2,
                                strict=True,
                            )
                        ),
                    }
                    for item in observation.all_items
                )
        elif len(observation) == 4:
            full_text, panel_text, anchors_visible, geometric_rows = observation
            normalized_rows = normalize_rows(geometric_rows)
        elif len(observation) == 3:
            # Isolated tests and injected diagnostic backends predate row
            # geometry. Give their ordered rows stable synthetic positions;
            # production always supplies normalized OCR boxes.
            full_text, panel_text, anchors_visible = observation
            normalized_rows = tuple(
                (
                    raw,
                    normalized,
                    (18.0, 16.0 + index * 48.0, 320.0, 24.0),
                    (
                        (
                            raw,
                            normalized,
                            (18.0, 16.0 + index * 48.0, 320.0, 24.0),
                        ),
                    ),
                )
                for index, (raw, normalized) in enumerate(self.title_rows(panel_text))
            )
        else:
            raise ArenaReaderError(
                "p_item_ocr_observation_invalid",
                "P-item OCR observation has an unsupported shape",
            )
        return (
            str(full_text),
            str(panel_text),
            bool(anchors_visible),
            normalized_rows,
            background_rows,
            atoms,
        )

    def one_substitution(self, left: str, right: str) -> bool:
        return len(left) >= 3 and len(left) == len(right) and sum(a != b for a, b in zip(left, right)) == 1

    def one_insertion_or_deletion(self, left: str, right: str) -> bool:
        if abs(len(left) - len(right)) != 1:
            return False
        shorter, longer = sorted((left, right), key=len)
        first_difference = next(
            (index for index, value in enumerate(shorter) if value != longer[index]),
            len(shorter),
        )
        return shorter[first_difference:] == longer[first_difference + 1 :]

    def numeric_source_row_extension(
        self,
        current_row: str,
        source_row: str,
    ) -> bool:
        if current_row == source_row or source_row not in current_row:
            return False
        residue = current_row.replace(source_row, "", 1)
        return bool(residue and any(character.isdigit() for character in residue) and re.fullmatch(r"[0-9A-Za-z.,%+\-]+", residue))

    def verified_numeric_source_row_extension(
        self,
        components: tuple[
            tuple[str, str, tuple[float, float, float, float]],
            ...,
        ],
        source_row: str,
        source_box: tuple[float, float, float, float],
    ) -> bool:
        source_components = tuple(
            index
            for index, (_, normalized, component_box) in enumerate(components)
            if normalized == source_row and self.same_source_position(component_box, source_box)
        )
        for source_component in source_components:
            remaining = tuple(normalized for index, (_, normalized, _) in enumerate(components) if index != source_component)
            if remaining and all(
                any(character.isdigit() for character in value) and re.fullmatch(r"[0-9A-Za-z.,%+\-]+", value) for value in remaining
            ):
                return True
        return False

    def same_source_position(
        self,
        current_box: tuple[float, float, float, float],
        source_box: tuple[float, float, float, float],
    ) -> bool:
        current_x, current_y, current_width, current_height = current_box
        source_x, source_y, source_width, source_height = source_box
        intersection_width = max(
            0.0,
            min(current_x + current_width, source_x + source_width) - max(current_x, source_x),
        )
        intersection_height = max(
            0.0,
            min(current_y + current_height, source_y + source_height) - max(current_y, source_y),
        )
        minimum_area = min(
            current_width * current_height,
            source_width * source_height,
        )
        if minimum_area <= 0:
            return False
        current_centre_y = current_y + current_height / 2.0
        source_centre_y = source_y + source_height / 2.0
        return intersection_width * intersection_height / minimum_area >= 0.50 and abs(current_centre_y - source_centre_y) <= 8.0

    def source_row_pixels_unchanged(
        self,
        current_image: Any,
        current_box: tuple[float, float, float, float],
        source_index: int,
    ) -> bool:
        import math

        import numpy as np

        comparison_started = self.clock.perf_counter()
        try:
            if not isinstance(current_image, np.ndarray) or current_image.ndim != 3 or current_image.shape[2] != 3:
                return False
            height, width = current_image.shape[:2]
            for source_image, source_row_boxes in self.stable_source_row_frames:
                if (
                    not isinstance(source_image, np.ndarray)
                    or source_image.shape != current_image.shape
                    or source_image.dtype != current_image.dtype
                ):
                    continue
                source_box = source_row_boxes[source_index]
                if not self.same_source_position(current_box, source_box):
                    continue
                current_x, current_y, current_width, current_height = current_box
                source_x, source_y, source_width, source_height = source_box
                left = math.floor(min(current_x, source_x) * width / 720.0)
                top = math.floor(min(current_y, source_y) * height / 1280.0)
                right = math.ceil(max(current_x + current_width, source_x + source_width) * width / 720.0)
                bottom = math.ceil(max(current_y + current_height, source_y + source_height) * height / 1280.0)
                # Compare the complete union at its original coordinates;
                # clipping, resizing or mixing pixels from several source
                # frames would no longer prove this source row unchanged.
                if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                    continue
                if np.array_equal(
                    current_image[top:bottom, left:right],
                    source_image[top:bottom, left:right],
                ):
                    return True
            return False
        finally:
            self.backend._add_timing(
                "p_item_source_row_pixel_identity",
                self.clock.perf_counter() - comparison_started,
            )
