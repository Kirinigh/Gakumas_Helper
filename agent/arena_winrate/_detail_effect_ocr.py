"""Same-frame effect OCR views and error-only semantic text recovery.

The port retains all source, transaction and metric state. Recovery functions
are explicit call dependencies, preserving the facade's per-call overrides.
"""

from __future__ import annotations

import re
import json
import unicodedata
from typing import Any, Protocol
from collections.abc import Callable, Sequence

from .reader import ClickedSkillCard
from .catalog import ArenaCatalogError, ArenaEntityCatalog
from ._reader_errors import ArenaReaderError
from ._reader_visual import _box, _text
from ._reader_evidence import _TitleBoundEffectRoiText
from ._detail_text_layout import ErrorNumericLayoutRecovery, ErrorWrappedSignedRecovery
from ._skill_detail_domain import SkillDetailText, SkillDetailDomain, reobserved_unknowns, project_domain_items, attribute_skill_detail
from ._detail_capture_evidence import _EvidenceClock, _FullFrameOcrEvidence
from ._skill_card_effect_recovery import EffectBodyRecovery, recover_wrapped_effect_decoration


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

    _card_count_frames: dict[int, tuple[Any, Any, Any]]

    _full_frame_ocr_evidence: dict[int, _FullFrameOcrEvidence]

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
            domain = self._domain_for_frame(image, key=key, card_id=card_id)
            if not domain.complete:
                return None
            pattern = self.port._skill_card_frame_title_pattern(image, card_id)
            native_items = domain.owned
            atoms = tuple((_text(item), _box(item)) for item in native_items)
            titles = tuple(box for value, box in atoms if re.search(pattern, value))
            if len(titles) != 1:
                return None
            height, width = image.shape[:2]
            roi = domain.panel
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
            domain = self._domain_for_frame(image, key=key, card_id=card_id)
            if not domain.complete:
                return None
            pattern = self.port._skill_card_frame_title_pattern(image, card_id)
            native_items = domain.owned
            atoms = tuple((_text(item), _box(item)) for item in native_items)
            titles = tuple(box for value, box in atoms if re.search(pattern, value))
            if len(titles) != 1:
                return None
            height, width = image.shape[:2]
            repaired = self.recover_skill_effect_body(
                atoms,
                titles[0],
                domain.panel,
                self.port._spatial_ocr_rows(native_items),
            )
            if not repaired.changed:
                return None
            current = self.port._cached_full_frame_ocr_evidence(image)
            if isinstance(current, _FullFrameOcrEvidence) and current.image is image:
                self.port._full_frame_ocr_evidence[id(image)] = current._replace(skill_body_repair=repaired)
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
        keys = [key for key, frame in self.port._card_detail_images.items() if frame is image]
        if len(keys) != 1 or keys[0] not in self.port._card_detail_open_ids:
            raise ArenaReaderError("skill_card_detail_domain_unconfirmed", "detail identity is missing")
        return self._skill_card_title_anchored_effect_roi_text(image, self.port._card_detail_open_ids[keys[0]])

    def _domain_for_frame(self, image: Any, *, key: tuple[int, int] | None = None, card_id: int | None = None):
        evidence = self.port._cached_full_frame_ocr_evidence(image)
        if not isinstance(evidence, _FullFrameOcrEvidence) or evidence.image is not image or not evidence.hit:
            return SkillDetailDomain(None, None, reason="frame_evidence_unavailable")
        if key is None:
            keys = [k for k, frame in getattr(self.port, "_card_detail_images", {}).items()
                    if frame is image and (card_id is None or self.port._card_detail_open_ids.get(k) == card_id)]
            if len(keys) != 1:
                return SkillDetailDomain(None, None, reason="detail_transaction_not_unique")
            key = keys[0]
        if card_id is None:
            card_id = getattr(self.port, "_card_detail_open_ids", {}).get(key)
        sources = tuple(getattr(self.port, "_card_count_frames", {}).get(key[0], ()))
        token = getattr(self.port, "_card_transaction_tokens", {}).get(key)
        binding = (key, token, card_id, sources)
        cached = evidence.skill_domain_binding if isinstance(evidence, _FullFrameOcrEvidence) else None
        if (cached is not None and cached[:3] == binding[:3] and len(cached[3]) == len(sources)
                and all(a is b for a, b in zip(cached[3], sources, strict=True))):
            self.port._increment("skill_card_detail_domain_cache_hits")
            return evidence.skill_domain
        same_context = (cached is not None and cached[:2] == binding[:2] and len(cached[3]) == len(sources)
                        and all(a is b for a, b in zip(cached[3], sources, strict=True)))
        if same_context and card_id is not None and evidence.skill_domain.complete:
            pattern = self.port._skill_card_frame_title_pattern(image, card_id)
            anchors, _ = self.port._skill_card_title_anchor_evidence(image, pattern)
            if len(anchors) == 1 and _box(anchors[0]) == evidence.skill_domain.title:
                # Opening learned the ID after this same frame was attributed.
                # Its unique, unchanged header can bind that existing domain.
                self.port._full_frame_ocr_evidence[id(image)] = evidence._replace(skill_domain_binding=binding)
                self.port._increment("skill_card_detail_domain_cache_hits")
                return evidence.skill_domain
        items = evidence.all_items
        title_box = None
        if card_id is not None:
            pattern = self.port._skill_card_frame_title_pattern(image, card_id)
            matches, items = self.port._skill_card_title_anchor_evidence(image, pattern)
            if len(matches) != 1:
                return SkillDetailDomain(None, None, reason="known_title_not_unique")
            title_box = _box(matches[0])
        started = self.clock.perf_counter()
        try:
            domain = attribute_skill_detail(image, sources, items, title_box=title_box)
        finally:
            self.port._increment("skill_card_detail_domain_scans")
            self.port._add_timing("skill_card_detail_domain_attribution", self.clock.perf_counter() - started)
        if isinstance(evidence, _FullFrameOcrEvidence):
            self.port._full_frame_ocr_evidence[id(image)] = evidence._replace(
                skill_domain=domain, skill_domain_binding=binding, skill_body_repair=None,
            )
        if not domain.complete:
            self.logger.debug(json.dumps({
                "event": "arena_skill_detail_domain_unconfirmed", "key": key,
                "card_id": card_id, "reason": domain.reason, "blocked": domain.blocked,
                "unattributed": [(_text(item), _box(item)) for item in domain.unknown],
            }, ensure_ascii=False))
        return domain

    def _skill_card_detail_ocr_text(self, image: Any, *, phase: str, key: tuple[int, int] | None = None) -> str:
        """Keep full OCR for diagnostics; serialize only the attributed tooltip."""
        if phase not in {"open", "reread"}:
            raise ValueError("skill-card detail OCR phase must be open or reread")
        if key is None:
            raise ArenaReaderError("skill_card_detail_domain_unconfirmed", "detail transaction key is missing")
        started = self.clock.perf_counter()
        try:
            self.port._with_full_frame_ocr_kind("detail_capture", self.port._full_ocr_text, image)
            domain = self._domain_for_frame(image, key=key)
            current = self.port._cached_full_frame_ocr_evidence(image)
            repair = current.skill_body_repair if isinstance(current, _FullFrameOcrEvidence) else None
            return SkillDetailText(repair.text if repair is not None else self.port._spatial_ocr_text(domain.owned), domain)
        finally:
            self.port._increment(f"skill_card_detail_{phase}_ocr_reads")
            self.port._add_timing(f"skill_card_detail_{phase}_ocr", self.clock.perf_counter() - started)

    def _title_anchored_effect_roi_without_background_parameters(
        self,
        frame_width: int,
        frame_height: int,
        title_box: tuple[int, int, int, int],
        items: Sequence[Any],
    ) -> tuple[int, int, int, int]:
        """Legacy candidate geometry only; semantic reads use a proved domain."""

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

    def _skill_card_title_anchored_effect_roi_text(self, image: Any, card_id: int) -> str:
        """Native and 2x observations share one source-proved tooltip domain."""
        import cv2

        domain = self._domain_for_frame(image, card_id=card_id)
        if domain.panel is None or domain.blocked:
            raise ArenaReaderError("skill_card_detail_domain_unconfirmed", domain.reason)
        left, top, width, height = domain.panel
        crop = image[top:top + height, left:left + width]
        enlarged = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        started = self.clock.perf_counter()
        try:
            self.port._with_full_frame_ocr_kind("derived_roi", self.port._full_ocr_text, enlarged)
            observed = self.port._cached_full_frame_ocr_evidence(enlarged)
            if observed is None or observed.image is not enlarged or not observed.hit or observed.kind != "derived_roi":
                raise ArenaReaderError("skill_card_detail_effect_roi_empty", "bounded ROI has no OCR evidence")
            projected = project_domain_items(observed.all_items, domain.panel)
            if projected is None:
                raise ArenaReaderError("skill_card_detail_domain_unconfirmed", "derived OCR crosses its input crop")
            if not reobserved_unknowns(domain, projected):
                raise ArenaReaderError("skill_card_detail_domain_unconfirmed", "crossing text was not completely re-observed")
            current = self.port._cached_full_frame_ocr_evidence(image)
            repair = current.skill_body_repair if isinstance(current, _FullFrameOcrEvidence) else None
            if repair is None:
                proposed = self.recover_skill_effect_body(
                    tuple((_text(item), _box(item)) for item in domain.owned), domain.title, domain.panel,
                    self.port._spatial_ocr_rows(domain.owned),
                )
                if proposed.changed:
                    repair = proposed
                    if isinstance(current, _FullFrameOcrEvidence):
                        self.port._full_frame_ocr_evidence[id(image)] = current._replace(skill_body_repair=repair)
                    self.port._increment("skill_card_detail_domain_body_repairs")
            native_items = list(domain.owned)
            if repair is not None:
                for index, _, replacement in repair.replacements:
                    native_items[index] = {"text": replacement, "box": _box(native_items[index])}
            native_text = repair.text if repair is not None else self.port._spatial_ocr_text(native_items)
            enlarged_text = self.port._spatial_ocr_text(projected)
            observations = tuple(text for text in (native_text, enlarged_text) if text.strip())
            if not observations:
                raise ArenaReaderError("skill_card_detail_effect_roi_empty", "bounded views are empty")
            full_text = native_text if domain.complete else enlarged_text
            try:
                merged = self.port._merge_skill_card_effect_detail_views(card_id, observations)
            except ArenaReaderError as error:
                if error.code != "skill_card_detail_effect_view_conflict":
                    raise
                recovered = recover_wrapped_effect_decoration(
                    tuple((_text(item), _box(item)) for item in native_items), domain.title, domain.panel,
                    self.port._spatial_ocr_rows(native_items), image=image,
                )
                if not recovered.changed:
                    raise
                filtered = tuple(item for index, item in enumerate(native_items) if index not in recovered.excluded_indices)
                if repair is not None:
                    composed = self.recover_skill_effect_body(
                        tuple((_text(item), _box(item)) for item in filtered), domain.title, domain.panel,
                        self.port._spatial_ocr_rows(filtered),
                    )
                    recovered_text = composed.text if composed.changed else self.port._spatial_ocr_text(filtered)
                else:
                    recovered_text = self.port._spatial_ocr_text(filtered)
                merged = self.port._merge_skill_card_effect_detail_views(card_id, (recovered_text, enlarged_text))
                full_text = recovered_text if domain.complete else enlarged_text
                self.port._increment("skill_card_detail_wrapped_decoration_repairs")
            count = len(observations) if domain.complete else int(bool(enlarged_text.strip()))
            return _TitleBoundEffectRoiText(merged, count, recovered_full_text=full_text)
        finally:
            self.port._increment("skill_card_detail_title_anchored_effect_roi_reads")
            self.port._increment("skill_card_detail_title_anchored_native_spatial_reads")
            self.port._increment("skill_card_detail_title_anchored_enlarged_spatial_reads")
            self.port._increment("skill_card_detail_effect_roi_reads")
            self.port._add_timing("skill_card_detail_effect_roi_ocr", self.clock.perf_counter() - started)

    def _skill_card_title_anchored_enhanced_effect_roi_text(self, image: Any, card_id: int) -> str:
        """Contrast the same attributed domain, never re-detect enhanced borders."""
        import cv2

        domain = self._domain_for_frame(image, card_id=card_id)
        if domain.panel is None or domain.blocked:
            raise ArenaReaderError("skill_card_detail_domain_unconfirmed", domain.reason)
        self.port._increment("skill_card_detail_enhanced_effect_roi_attempts")
        started = self.clock.perf_counter()
        try:
            left, top, width, height = domain.panel
            gray = cv2.cvtColor(image[top:top + height, left:left + width], cv2.COLOR_BGR2GRAY)
            contrasted = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
            enlarged = cv2.resize(contrasted, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            enhanced = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)
            self.port._with_full_frame_ocr_kind("derived_roi", self.port._full_ocr_text, enhanced)
            observed = self.port._cached_full_frame_ocr_evidence(enhanced)
            projected = None if observed is None or observed.image is not enhanced or not observed.hit or observed.kind != "derived_roi" else project_domain_items(observed.all_items, domain.panel)
            if projected is None or not reobserved_unknowns(domain, projected):
                raise ArenaReaderError("skill_card_detail_domain_unconfirmed", "enhanced OCR has no complete bounded evidence")
            return self.port._spatial_ocr_text(projected)
        finally:
            self.port._add_timing("skill_card_detail_enhanced_effect_roi_ocr", self.clock.perf_counter() - started)

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
