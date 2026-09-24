"""Same-frame effect OCR views and error-only semantic text recovery.

The port retains all source, transaction and metric state. Recovery functions
are explicit call dependencies, preserving the facade's per-call overrides.
"""

from __future__ import annotations

import re
import json
from typing import Any, Protocol
from collections.abc import Callable, Sequence

from .reader import ClickedSkillCard
from .catalog import ArenaCatalogError, ArenaEntityCatalog
from ._reader_errors import ArenaReaderError
from ._reader_visual import _box, _text
from ._reader_evidence import _TitleBoundEffectRoiText
from ._detail_text_layout import ErrorNumericLayoutRecovery, ErrorWrappedSignedRecovery
from ._detail_capture_evidence import _FullFrameOcrEvidence
from ._skill_card_effect_recovery import EffectBodyRecovery


class _EvidenceClock(Protocol):
    def perf_counter(self) -> float: ...


class _EvidenceLogger(Protocol):
    def debug(self, message: str) -> None: ...


class DetailEffectOcrPort(Protocol):
    """Only the state views and operations used by this responsibility."""

    def _add_timing(self, name: str, elapsed: float) -> None: ...

    def _auxiliary_badge_glyph_count(
        self, key: tuple[int, int], *, maximum_count: int, admissible_counts: Sequence[int] | None = None, allow_ocr: bool = True
    ) -> int: ...

    def _background_parameter_column_left(
        self, frame_width: int, frame_height: int, title_box: tuple[int, int, int, int], items: Sequence[Any]
    ) -> int | None: ...

    def _cached_full_frame_ocr_evidence(self, image: Any) -> _FullFrameOcrEvidence | None: ...

    _card_detail_images: dict[tuple[int, int], Any]

    _card_detail_open_ids: dict[tuple[int, int], int]

    _card_face_cost_optional_errors: dict[tuple[int, int], str]

    def _full_ocr_text(self, image: Any, *, spatial_reading_order: bool = False) -> str: ...

    def _increment(self, name: str) -> None: ...

    def _isolated_skill_card_title_atoms(self, image: Any, title_pattern: str) -> Any: ...

    def _measure_optional_card_face_generic_cost(self, key: tuple[int, int], card_id: int) -> tuple[int, int, int] | None: ...

    def _merge_skill_card_effect_detail_views(self, card_id: int, detail_texts: Sequence[str]) -> str: ...

    def _record_duration_sample(self, name: str, elapsed: float) -> None: ...

    _runtime_counts: dict[str, int]

    _runtime_timing_seconds: dict[str, float]

    def _skill_card_frame_title_pattern(self, image: Any, card_id: int) -> str: ...

    def _skill_card_title_anchor_evidence(self, image: Any, title_pattern: str) -> tuple[list[Any], list[Any]]: ...

    def _spatial_ocr_rows(
        self, items: Sequence[Any]
    ) -> tuple[tuple[str, tuple[int, int, int, int], tuple[tuple[str, tuple[int, int, int, int]], ...]], ...]: ...

    def _spatial_ocr_text(self, items: Sequence[Any]) -> str: ...

    def _title_anchored_effect_roi(
        self, frame_width: int, frame_height: int, title_box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]: ...

    def _title_anchored_effect_roi_without_background_parameters(
        self, frame_width: int, frame_height: int, title_box: tuple[int, int, int, int], items: Sequence[Any]
    ) -> tuple[int, int, int, int]: ...

    def _with_full_frame_ocr_kind(self, kind: str, operation: Any, *args: Any, **kwargs: Any) -> Any: ...

    catalog: ArenaEntityCatalog


class DetailEffectOcr:
    def __init__(
        self,
        port: DetailEffectOcrPort,
        *,
        clock: _EvidenceClock,
        logger: _EvidenceLogger,
        recover_distant_number_text: Callable[..., ErrorNumericLayoutRecovery],
        recover_wrapped_signed_text: Callable[..., ErrorWrappedSignedRecovery],
        recover_skill_effect_body: Callable[..., EffectBodyRecovery],
    ) -> None:
        self.port = port
        self.clock = clock
        self.logger = logger
        self.recover_distant_number_text = recover_distant_number_text
        self.recover_wrapped_signed_text = recover_wrapped_signed_text
        self.recover_skill_effect_body = recover_skill_effect_body

    def _recover_failed_skill_card_effect_text(
        self,
        key: tuple[int, int],
        card_id: int,
        error: ArenaReaderError,
        *,
        expected_customization_count: int | None,
        allow_zero_without_badge_count: bool,
    ) -> tuple[str, ClickedSkillCard] | None:
        """Repair cached native atoms only after the existing views failed.

        Geometry cannot select a catalog answer. The repaired text must still
        have explicit positive evidence. Resolve its count from the detail
        before considering existing auxiliary evidence; never start extra OCR.
        The caller retains all cost, identity and fresh-frame confirmation.
        """
        if error.code not in {
            "skill_card_detail_ambiguous",
            "skill_card_badge_glyph_domain_invalid",
            "skill_card_badge_glyph_ambiguous",
        }:
            return None
        started = self.clock.perf_counter()
        self.port._increment("skill_card_error_numeric_attempts")
        try:
            image = self.port._card_detail_images.get(key)
            evidence = self.port._cached_full_frame_ocr_evidence(image)
            if evidence is None:
                return None
            pattern = self.port._skill_card_frame_title_pattern(image, card_id)
            isolated = self.port._isolated_skill_card_title_atoms(image, pattern)
            native_items = evidence.all_items if isolated is None else isolated[1]
            atoms = tuple((_text(item), _box(item)) for item in native_items)
            titles = tuple(box for value, box in atoms if re.search(pattern, value))
            if len(titles) != 1:
                return None
            height, width = image.shape[:2]
            roi = self.port._title_anchored_effect_roi(width, height, titles[0])
            wrapped = self.recover_wrapped_signed_text(atoms, titles[0], roi)
            if wrapped.status == "recovered":
                self.port._increment("skill_card_error_wrapped_row_repairs")
                self.logger.debug(
                    f"skill-card error wrapped row recovery: key={key} card={card_id} "
                    f"order={wrapped.reordered_indices} reason={wrapped.reason}",
                )
                clean_text = wrapped.protected_text
            else:
                if wrapped.status == "unrecoverable":
                    return None
                layout = self.recover_distant_number_text(atoms, titles[0], roi)
                if layout.status != "recovered":
                    self.port._increment("skill_card_error_numeric_no_repair")
                    return None
                self.port._increment("skill_card_error_numeric_layout_repairs")
                self.logger.debug(
                    f"skill-card error numeric isolation: key={key} card={card_id} "
                    f"far={layout.far_indices} ambiguous={layout.ambiguous_indices}",
                )
                clean_text = layout.protected_text
            if getattr(self.port, "_card_face_cost_optional_errors", {}).get(key):
                return None
            generic_values = self.port._measure_optional_card_face_generic_cost(key, card_id)
            if wrapped.status == "recovered":
                if expected_customization_count is None and not allow_zero_without_badge_count:
                    return None
                customizations = self.port.catalog._resolve_failed_detail_unique_customizations(
                    card_id,
                    clean_text,
                )
                count = sum(customizations.values())
                if expected_customization_count is not None and count != expected_customization_count:
                    return None
                if generic_values is not None:
                    self.port.catalog.certify_card_face_generic_cost(
                        card_id,
                        resolved=customizations,
                        frame_values=generic_values,
                    )
                source = "detail_card_face_unique" if generic_values is not None else "detail_unconstrained"
                mode = self.port.catalog.effective_customization_evidence_mode(
                    card_id,
                    clean_text,
                    expected_count=count,
                    resolved=customizations,
                )
            elif expected_customization_count is None:
                if not allow_zero_without_badge_count:
                    return None
                try:
                    if generic_values is None:
                        customizations = self.port.catalog.resolve_effective_customizations_without_badge_count(
                            card_id,
                            clean_text,
                        )
                        source = "detail_unconstrained"
                    else:
                        customizations = self.port.catalog.resolve_effective_customizations_with_generic_cost_evidence(
                            card_id,
                            clean_text,
                            generic_cost_frame_values=generic_values,
                        )
                        source = "detail_card_face_unique"
                    mode = self.port.catalog.effective_customization_evidence_mode(
                        card_id,
                        clean_text,
                        expected_count=sum(customizations.values()),
                        resolved=customizations,
                    )
                except ArenaCatalogError:
                    maximum = self.port.catalog.maximum_customization_count(card_id)
                    # Reuse each resolution if auxiliary evidence is still
                    # needed for a genuinely non-unique detail combination.
                    by_count = {}
                    for observed_count in range(1, maximum + 1):
                        try:
                            candidate = self.port.catalog.resolve_clicked_customizations(
                                card_id,
                                clean_text,
                                observed_badge_count=observed_count,
                                generic_cost_frame_values=generic_values,
                            )
                        except ArenaCatalogError:
                            continue
                        if candidate.badge_count_match and candidate.resolved_count == observed_count:
                            by_count[observed_count] = candidate
                    count = self.port._auxiliary_badge_glyph_count(
                        key,
                        maximum_count=maximum,
                        admissible_counts=tuple(by_count),
                        allow_ocr=False,
                    )
                    if count == 0:
                        return None
                    resolution = by_count[count]
                    customizations = resolution.customizations
                    source = f"{resolution.source}_auxiliary_badge_glyph"
                    mode = resolution.evidence_mode
            else:
                if expected_customization_count == 0:
                    return None
                resolution = self.port.catalog.resolve_clicked_customizations(
                    card_id,
                    clean_text,
                    observed_badge_count=expected_customization_count,
                    generic_cost_frame_values=generic_values,
                )
                customizations = resolution.customizations
                source, mode = resolution.source, resolution.evidence_mode
            if not customizations or mode != "positive_unique":
                return None
            self.port._increment("skill_card_error_numeric_reparse_successes")
            return clean_text, ClickedSkillCard(
                card_id,
                customizations,
                resolution_source=source,
                detail_evidence_mode=mode,
            )
        except (ArenaCatalogError, ArenaReaderError) as recovery_error:
            self.port._increment("skill_card_error_numeric_reparse_failures")
            self.logger.debug(f"skill-card error numeric recovery unresolved: {recovery_error}")
            return None
        finally:
            self.port._record_duration_sample(
                "skill_card_error_numeric_recovery",
                self.clock.perf_counter() - started,
            )

    def _recover_failed_skill_card_body_text(self, key: tuple[int, int]) -> str | None:
        """Repair cached body atoms after a failure, without observing the game."""
        card_id = getattr(self.port, "_card_detail_open_ids", {}).get(key)
        if card_id is None:
            return None
        image = self.port._card_detail_images.get(key)
        evidence = self.port._cached_full_frame_ocr_evidence(image)
        if evidence is None:
            return None
        started = self.clock.perf_counter()
        self.port._increment("skill_card_error_body_attempts")
        try:
            pattern = self.port._skill_card_frame_title_pattern(image, card_id)
            isolated = self.port._isolated_skill_card_title_atoms(image, pattern)
            native_items = evidence.all_items if isolated is None else isolated[1]
            atoms = tuple((_text(item), _box(item)) for item in native_items)
            titles = tuple(box for value, box in atoms if re.search(pattern, value))
            if len(titles) != 1:
                return None
            height, width = image.shape[:2]
            repaired = self.recover_skill_effect_body(
                atoms,
                titles[0],
                self.port._title_anchored_effect_roi(width, height, titles[0]),
                self.port._spatial_ocr_rows(native_items),
            )
            if not repaired.changed:
                return None
            self.port._increment("skill_card_error_body_repairs")
            self.logger.debug(
                json.dumps(
                    {
                        "event": "arena_skill_body_text_repaired",
                        "key": key,
                        "card_id": card_id,
                        "reordered_indices": repaired.reordered_indices,
                        "replacements": repaired.replacements,
                    },
                    ensure_ascii=False,
                )
            )
            return repaired.text
        except (ArenaCatalogError, ArenaReaderError, ValueError, TypeError) as error:
            self.port._increment("skill_card_error_body_failures")
            self.logger.debug(f"skill-card cached body repair unavailable: {error}")
            return None
        finally:
            self.port._record_duration_sample("skill_card_error_body_recovery", self.clock.perf_counter() - started)

    def _skill_card_effect_roi_text(self, image: Any) -> str:
        """OCR an enlarged detail-effect panel without taking another screenshot."""

        import cv2

        height, width = image.shape[:2]
        left = int(round(width * 0.055))
        top = int(round(height * 0.025))
        right = int(round(width * 0.61))
        bottom = int(round(height * 0.32))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            raise ArenaReaderError(
                "skill_card_detail_effect_roi_invalid",
                f"detail effect ROI is empty for frame {width}x{height}",
            )
        enlarged = cv2.resize(
            crop,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        started = self.clock.perf_counter()
        try:
            return self.port._with_full_frame_ocr_kind(
                "derived_roi",
                self.port._full_ocr_text,
                enlarged,
            )
        finally:
            self.port._increment("skill_card_detail_effect_roi_reads")
            self.port._add_timing(
                "skill_card_detail_effect_roi_ocr",
                self.clock.perf_counter() - started,
            )

    def _skill_card_detail_ocr_text(self, image: Any, *, phase: str) -> str:
        """Time full detail OCR by transaction phase without changing its result."""

        if phase not in {"open", "reread"}:
            raise ValueError("skill-card detail OCR phase must be open or reread")
        if not hasattr(self.port, "_runtime_timing_seconds"):
            self.port._runtime_timing_seconds = {}
        started = self.clock.perf_counter()
        try:
            return self.port._with_full_frame_ocr_kind(
                "detail_capture",
                self.port._full_ocr_text,
                image,
            )
        finally:
            self.port._increment(f"skill_card_detail_{phase}_ocr_reads")
            self.port._add_timing(
                f"skill_card_detail_{phase}_ocr",
                self.clock.perf_counter() - started,
            )

    def _title_anchored_effect_roi_without_background_parameters(
        self,
        frame_width: int,
        frame_height: int,
        title_box: tuple[int, int, int, int],
        items: Sequence[Any],
    ) -> tuple[int, int, int, int]:
        """Clip only a geometrically proven right-side member-stat column."""

        left, top, width, height = self.port._title_anchored_effect_roi(
            frame_width,
            frame_height,
            title_box,
        )
        boundary = self.port._background_parameter_column_left(
            frame_width,
            frame_height,
            title_box,
            items,
        )
        if boundary is None or boundary >= left + width:
            return left, top, width, height
        clipped_width = boundary - left
        if clipped_width < 32 or boundary < title_box[0] + title_box[2]:
            return left, top, width, height
        if hasattr(self.port, "_runtime_counts"):
            self.port._increment("skill_card_detail_title_anchored_background_parameter_clips")
        return left, top, clipped_width, height

    def _skill_card_title_anchored_effect_roi_text(
        self,
        image: Any,
        card_id: int,
    ) -> str:
        """Retry effect OCR in the panel geometrically bound to its fixed title."""

        import cv2

        try:
            title_pattern = self.port._skill_card_frame_title_pattern(image, card_id)
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_detail_title_catalog_missing",
                str(error),
            ) from error
        matches, native_items = self.port._skill_card_title_anchor_evidence(
            image,
            title_pattern,
        )
        if len(matches) != 1:
            self.port._increment("skill_card_detail_title_anchor_transition_retries")
            raise ArenaReaderError(
                "skill_card_detail_title_anchor_ambiguous",
                f"card {card_id} has {len(matches)} title anchors on the retry frame",
            )
        height, width = image.shape[:2]
        title_box = _box(matches[0])
        left, top, roi_width, roi_height = self.port._title_anchored_effect_roi_without_background_parameters(
            width,
            height,
            title_box,
            native_items,
        )
        crop = image[top : top + roi_height, left : left + roi_width]
        if crop.size == 0:
            raise ArenaReaderError(
                "skill_card_detail_effect_roi_invalid",
                f"title-anchored detail effect ROI is empty for frame {width}x{height}",
            )
        # Preserve the native full-frame OCR boxes from the same recognition
        # that proved the title. The enlarged crop is valuable for small
        # glyphs, but interpolation can also erase a small standalone value.
        # Spatially rebuild both same-frame views and combine their evidence;
        # catalog resolution remains fail-closed when the views conflict.
        native_roi_items = []
        for item in native_items:
            item_x, item_y, item_width, item_height = _box(item)
            centre_x = item_x + item_width / 2.0
            centre_y = item_y + item_height / 2.0
            if left <= centre_x <= left + roi_width and top <= centre_y <= top + roi_height:
                native_roi_items.append(item)
        native_text = self.port._spatial_ocr_text(native_roi_items)
        enlarged = cv2.resize(
            crop,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        started = self.clock.perf_counter()
        try:
            enlarged_text = self.port._with_full_frame_ocr_kind(
                "derived_roi",
                self.port._full_ocr_text,
                enlarged,
                spatial_reading_order=True,
            )
            observations = tuple(value for value in (native_text, enlarged_text) if value.strip())
            texts = tuple(dict.fromkeys(observations))
            if not texts:
                raise ArenaReaderError(
                    "skill_card_detail_effect_roi_empty",
                    "title-bound native and enlarged OCR views were both empty",
                )
            return _TitleBoundEffectRoiText(
                self.port._merge_skill_card_effect_detail_views(card_id, texts),
                len(observations),
            )
        finally:
            self.port._increment("skill_card_detail_title_anchored_effect_roi_reads")
            self.port._increment("skill_card_detail_title_anchored_native_spatial_reads")
            self.port._increment("skill_card_detail_title_anchored_enlarged_spatial_reads")
            self.port._increment("skill_card_detail_effect_roi_reads")
            self.port._add_timing(
                "skill_card_detail_effect_roi_ocr",
                self.clock.perf_counter() - started,
            )

    def _skill_card_title_anchored_enhanced_effect_roi_text(
        self,
        image: Any,
        card_id: int,
    ) -> str:
        """Contrast one title-bound effect panel on the already-captured frame."""

        import cv2

        self.port._increment("skill_card_detail_enhanced_effect_roi_attempts")
        started = self.clock.perf_counter()
        try:
            try:
                title_pattern = self.port._skill_card_frame_title_pattern(image, card_id)
            except ArenaCatalogError as error:
                raise ArenaReaderError(
                    "skill_card_detail_title_catalog_missing",
                    str(error),
                ) from error
            matches, native_items = self.port._skill_card_title_anchor_evidence(
                image,
                title_pattern,
            )
            if len(matches) != 1:
                self.port._increment("skill_card_detail_enhanced_effect_roi_title_ambiguous")
                raise ArenaReaderError(
                    "skill_card_detail_title_anchor_ambiguous",
                    f"card {card_id} has {len(matches)} title anchors on the enhanced frame",
                )
            height, width = image.shape[:2]
            left, top, roi_width, roi_height = self.port._title_anchored_effect_roi_without_background_parameters(
                width,
                height,
                _box(matches[0]),
                native_items,
            )
            crop = image[top : top + roi_height, left : left + roi_width]
            if crop.size == 0:
                raise ArenaReaderError(
                    "skill_card_detail_effect_roi_invalid",
                    f"enhanced title-bound effect ROI is empty for frame {width}x{height}",
                )

            # This is a catalog-agnostic recognition view: local contrast makes
            # light effect labels legible without interpreting isolated digits.
            # The active catalog's existing effect matchers merge and certify
            # the result, so conflicting levels still fail closed.
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            contrasted = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            ).apply(gray)
            enlarged = cv2.resize(
                contrasted,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_LANCZOS4,
            )
            enhanced = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)
            return self.port._with_full_frame_ocr_kind(
                "derived_roi",
                self.port._full_ocr_text,
                enhanced,
                spatial_reading_order=True,
            )
        finally:
            self.port._add_timing(
                "skill_card_detail_enhanced_effect_roi_ocr",
                self.clock.perf_counter() - started,
            )

    def _merge_skill_card_effect_detail_views(
        self,
        card_id: int,
        detail_texts: Sequence[str],
    ) -> str:
        """Apply the catalog-wide cross-view evidence gate before OCR fusion."""

        try:
            return self.port.catalog.merge_compatible_effect_detail_views(
                card_id,
                detail_texts,
            )
        except ArenaCatalogError as error:
            self.port._increment("skill_card_detail_effect_view_conflicts")
            raise ArenaReaderError(
                "skill_card_detail_effect_view_conflict",
                str(error),
            ) from error
