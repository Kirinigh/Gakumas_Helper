"""Same-frame effect OCR views and error-only semantic text recovery.

The port retains all source, transaction and metric state. Recovery functions
are explicit call dependencies, preserving the facade's per-call overrides.
"""

from __future__ import annotations

import re
import json
import unicodedata
from typing import Any, Protocol
from collections import Counter
from collections.abc import Mapping, Callable, Sequence

from agent.live_decision.ocr_nodes import OcrRows, read_ocr_node

from .reader import ClickedSkillCard
from .catalog import (
    _NUMERIC_EFFECT_LABELS,
    _OCR_EFFECT_VIEW_BOUNDARY,
    _ADDED_NUMERIC_EFFECT_MARKERS,
    ArenaCatalogError,
    ArenaEntityCatalog,
    StructuralOmissionFieldConflict,
    _normalise_effect_text,
)
from ._reader_errors import ArenaReaderError
from ._reader_visual import _box, _text
from ._reader_evidence import _EffectRoiSourceBinding, _TitleBoundEffectRoiText, _SourceBackgroundOcrBudget
from .recognition_probe import RecognitionProbe
from ._detail_text_layout import NUMERIC_TOKEN_BOUNDARY, ErrorNumericLayoutRecovery, ErrorWrappedSignedRecovery, _inside_roi
from ._skill_detail_domain import SkillDetailDomain, project_domain_items, attribute_skill_detail
from ._detail_capture_evidence import _EvidenceClock, _FullFrameOcrEvidence, read_skill_detail_capture_text
from ._skill_card_effect_recovery import (
    _VALUE_LABELS,
    EffectBodyRecovery,
    recover_wrapped_effect_decoration,
    recover_overlapping_enthusiasm_operand,
)

_EFFECT_OPERAND_LABEL = "|".join(map(re.escape, dict.fromkeys((
    *_NUMERIC_EFFECT_LABELS, *_VALUE_LABELS, "コスト値減少", "コスト値増加", "消費",
))))
_LABELED_EFFECT_OPERAND = re.compile(rf"(?P<label>{_EFFECT_OPERAND_LABEL})[+\-−]?[0-9]+\Z")
_SPLIT_EFFECT_OPERAND_LABEL = re.compile(rf"(?:{_EFFECT_OPERAND_LABEL})[+\-−]?\Z")
_DETACHED_EFFECT_OPERAND = re.compile(r"[+\-−]?[0-9]+\Z")
_EFFECT_OPERAND_FRAGMENT = re.compile(r"不|未|非|無|なし|[0-9.%+\-−]+|回|囘|ターン|倍|以上|以下")
_ONCE_PER_STAGE_LABELS = ("試験・ステージ中", "試験·ステージ中", "試験ステージ中")


def _normalise_operand_atom(text: str) -> str:
    return _normalise_effect_text(unicodedata.normalize("NFKC", text))


def _is_symbol_fragment(text: str) -> bool:
    return bool(text) and all(unicodedata.category(char)[0] in {"P", "S"} for char in text)


class _EvidenceLogger(Protocol):
    def debug(self, message: str) -> None: ...


class _SkillDetailTextWithNumericFragments(str):
    """Keep a native title readable until detail identity can bind the boxes."""

    def __new__(cls, text: str, validate: Callable[[int, tuple[int, int] | None], None]):
        value = super().__new__(cls, text)
        value.validate = validate
        return value


class DetailEffectOcrPort(Protocol):
    """Only the state views and operations used by this responsibility."""

    context: Any

    def _add_timing(self, name: str, elapsed: float) -> None: ...

    def _check_cancelled(self) -> None: ...

    def _auxiliary_badge_glyph_count(
        self, key: tuple[int, int], *, maximum_count: int, admissible_counts: Sequence[int] | None = None, allow_ocr: bool | None = None
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

    def _full_ocr_text(self, image: Any, *, spatial_reading_order: bool = False, accept: Callable[[Sequence[Any]], bool] | None = ...) -> str: ...

    def _increment(self, name: str) -> None: ...

    def _isolated_skill_card_title_atoms(self, image: Any, title_pattern: str) -> Any: ...

    def _measure_optional_card_face_generic_cost(self, key: tuple[int, int], card_id: int) -> tuple[int | None, ...] | None: ...

    def _merge_skill_card_effect_detail_views(self, card_id: int, detail_texts: Sequence[str]) -> str: ...

    def _record_duration_sample(self, name: str, elapsed: float) -> None: ...

    _runtime_counts: dict[str, int]

    _runtime_timing_seconds: dict[str, float]

    def _skill_card_frame_title_pattern(self, image: Any, card_id: int) -> str: ...
    def _skill_detail_visual_ids(self, key: tuple[int, int]) -> tuple[int, ...]: ...

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
            roi = self.port._title_anchored_effect_roi_without_background_parameters(width, height, titles[0], native_items)
            self._assert_native_view_numeric_fragments(native_items, image, titles[0], roi, evidence)
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
            if self._numeric_fragment_pairs(evidence.all_items) or self._numeric_fragment_pairs(native_items):
                roi = self.port._title_anchored_effect_roi_without_background_parameters(width, height, titles[0], native_items)
                self._assert_native_view_numeric_fragments(native_items, image, titles[0], roi, evidence)
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

    def _native_skill_card_omission_text(self, image: Any, card_id: int, *, preserve_body_boundaries: bool = False) -> str:
        """Use cached OCR boxes scoped to the opened title without another read."""
        self._assert_detail_recovery_binding(image, card_id)
        evidence = self.port._cached_full_frame_ocr_evidence(image)
        if (image is None or evidence is None or evidence.image is not image
                or not evidence.hit or evidence.kind != "detail_capture"):
            raise ArenaReaderError("skill_card_detail_positive_evidence_missing", "no cached native detail boxes")
        pattern = self.port._skill_card_frame_title_pattern(image, card_id)
        titles = [item for item in evidence.all_items if re.fullmatch(pattern, _text(item).strip())]
        if len(titles) != 1:
            raise ArenaReaderError("skill_card_detail_positive_evidence_missing", "no unique native detail title")
        title = _box(titles[0])
        height, width = image.shape[:2]
        roi = self.port._title_anchored_effect_roi_without_background_parameters(
            width, height, title, evidence.all_items,
        )
        local = tuple(item for item in evidence.all_items if _inside_roi(_box(item), roi)
                      and (item is titles[0] or _box(item)[1] + _box(item)[3] / 2 >= title[1] + title[3]))
        self._assert_native_view_numeric_fragments(local, image, title, roi, evidence)
        body = tuple(item for item in local if item is not titles[0])
        local = self._coalesce_omission_field_atoms(local, body, card_id)
        atoms = tuple((_text(item), _box(item)) for item in local)
        rows = self.port._spatial_ocr_rows(local)
        repair = self.recover_skill_effect_body(atoms, title, roi, rows)
        replacements = {index: value for index, _, value in repair.replacements}
        repaired_atoms = tuple(
            (replacements.get(index, text), box) for index, (text, box) in enumerate(atoms)
        )
        if self._native_omission_operand_conflicts(repaired_atoms):
            raise self._native_operand_effect_view_conflict()
        if preserve_body_boundaries:
            body_items = tuple({"text": text, "box": box} for text, box in repaired_atoms)
            return "\n".join((_text(titles[0]), self._scoped_omission_body_text(body_items, title, roi)))
        return repair.text if repair.changed else self._ordered_effect_atom_text(local)

    def _coalesce_omission_field_atoms(self, items, local_items, card_id):
        """Preserve a split cost or footer as one observed field on its own row."""
        card = getattr(self.port.catalog, "_cards_by_id", {}).get(card_id)
        if not isinstance(card, Mapping):
            return items
        labels = _ONCE_PER_STAGE_LABELS if card.get("limit") == 1 else ()
        cost = self.port.catalog._typed_cost_descriptor(card)
        if cost is not None:
            labels = (*labels, cost[0])
        if not labels:
            return items

        keys = tuple((_text(item).strip(), _box(item)) for item in items)
        counts = Counter(keys)
        replacements = {}
        for _, _, row in self.port._spatial_ocr_rows(local_items):
            start = 0
            while start < len(row):
                end = self._joined_omission_field_end(row, start, labels)
                group = row[start:end]
                start = end
                if len(group) < 2 or any(counts[atom] != 1 for atom in group):
                    continue
                boxes = [box for _, box in group]
                left = min(x for x, _, _, _ in boxes)
                top = min(y for _, y, _, _ in boxes)
                right = max(x + width for x, _, width, _ in boxes)
                bottom = max(y + height for _, y, _, height in boxes)
                replacements[group[0]] = {
                    "text": "".join(text for text, _ in group),
                    "box": (left, top, right - left, bottom - top),
                }
                replacements.update((atom, None) for atom in group[1:])
        if not replacements:
            return items
        return tuple(value for item, key in zip(items, keys, strict=True)
                     if (value := replacements.get(key, item)) is not None)

    @staticmethod
    def _joined_omission_field_end(row, start, labels):
        # Begin inside the label. Once a digit is read, further bare digits
        # must stay separate, even when their boxes touch.
        partial = _normalise_operand_atom(row[start][0]).lstrip("(")
        if not partial or not any(label.startswith(partial) for label in labels):
            return start + 1
        end = start + 1
        for index in range(start + 1, len(row)):
            text, box = row[index]
            if not DetailEffectOcr._neighboring_effect_atoms(row[index - 1][1], box):
                break
            token = _normalise_operand_atom(text)
            candidate = partial + token
            prefix = next((label for label in labels if candidate.startswith(label)), None)
            if prefix is not None:
                payload = candidate[len(prefix):]
                if not payload:
                    end = index + 1
                elif re.fullmatch(r"[+-]?[0-9]+", payload) and prefix in _ONCE_PER_STAGE_LABELS:
                    # The recognised footer count can also have a separate unit.
                    end = index + 1
                elif re.match(r"[+-]?[0-9]+", payload):
                    return index + 1
                else:
                    break
                if payload and (index + 1 == len(row) or _normalise_operand_atom(row[index + 1][0]) not in {"回", "囘"}):
                    return end
            elif not any(label.startswith(candidate) for label in labels):
                break
            partial = candidate
        return end

    @staticmethod
    def _native_operand_effect_view_conflict() -> ArenaReaderError:
        return ArenaReaderError(
            "skill_card_detail_effect_view_conflict",
            "native detail has an adjacent sign, number or qualifier at a protected value",
        )

    @staticmethod
    def _native_omission_operand_conflicts(atoms, *, include_symbols=False):
        """Find adjoining modifiers only when a body label owns the observed value."""
        normalized = tuple((_normalise_operand_atom(text), box) for text, box in atoms)
        pairs = []
        for index, (value, value_box) in enumerate(normalized):
            x, y, width, height = value_box
            if _LABELED_EFFECT_OPERAND.search(value) is None:
                if not _DETACHED_EFFECT_OPERAND.fullmatch(value):
                    continue
                # Bare numbers are common on the underlying screen. A detached
                # operand needs a label directly to its left on the same row.
                if not any(
                    _SPLIT_EFFECT_OPERAND_LABEL.search(label)
                    and min(y + height, ly + lh) - max(y, ly) >= .5 * min(height, lh)
                    and 0 <= x - lx - lw <= .5 * min(height, lh)
                    for label, (lx, ly, lw, lh) in normalized
                ):
                    continue
            for other, (fragment, (fx, fy, fw, fh)) in enumerate(normalized):
                if other == index or fy >= y + height or fy + fh <= y:
                    continue
                gap = fx - x - width if fx > x else x - fx - fw
                if (-.25 * min(height, fh) <= gap <= .5 * min(height, fh)
                        and (_EFFECT_OPERAND_FRAGMENT.fullmatch(fragment)
                             or (include_symbols and _is_symbol_fragment(fragment)))):
                    pairs.append((index, other))
        return tuple(pairs)

    def _skill_card_effect_roi_text(self, image: Any) -> str:
        keys = [key for key, frame in self.port._card_detail_images.items() if frame is image]
        if len(keys) != 1 or keys[0] not in self.port._card_detail_open_ids:
            raise ArenaReaderError("skill_card_detail_domain_unconfirmed", "detail identity is missing")
        return self._skill_card_title_anchored_effect_roi_text(image, self.port._card_detail_open_ids[keys[0]])

    def _domain_for_frame(self, image: Any, *, key: tuple[int, int] | None = None, card_id: int | None = None):
        """Legacy pixel-attribution diagnostic; never a production text gate."""
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
        transaction_sources = getattr(self.port, "_card_detail_source_frames", None)
        sources = (tuple(transaction_sources.get(key, ())) if transaction_sources is not None
                   else tuple(getattr(self.port, "_card_count_frames", {}).get(key[0], ())))
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
            domain = attribute_skill_detail(
                image, sources, items, title_box=title_box, title_verified=card_id is not None,
            )
        finally:
            self.port._increment("skill_card_detail_domain_scans")
            self.port._add_timing("skill_card_detail_domain_attribution", self.clock.perf_counter() - started)
        if isinstance(evidence, _FullFrameOcrEvidence):
            self.port._full_frame_ocr_evidence[id(image)] = evidence._replace(
                skill_domain=domain, skill_domain_binding=binding, skill_body_repair=None,
            )
        if not domain.complete:
            try:
                RecognitionProbe.observe_domain_uncertainty(
                    getattr(self.port, "_detail_failure_frames", None), self.port._full_frame_ocr_evidence[id(image)],
                )
            except Exception:
                # Optional evidence must not change recognition or recovery.
                probe = getattr(self.port, "_recognition_probe", None)
                if probe is not None:
                    probe.disabled = True
            self.logger.debug(json.dumps({
                "event": "arena_skill_detail_domain_unconfirmed", "key": key,
                "card_id": card_id, "reason": domain.reason, "blocked": domain.blocked,
                "unattributed": [(_text(item), _box(item)) for item in domain.unknown],
            }, ensure_ascii=False))
        return domain

    def _skill_card_detail_ocr_text(self, image: Any, *, phase: str, key: tuple[int, int] | None = None) -> str:
        """Read a frozen card frame with the original local recovery on errors.

        Whole-screen spatial concatenation loses token boundaries between
        tooltip values and game-page digits. Background pixels are not a
        prerequisite for catalog evidence; the session retains identity,
        fresh-frame confirmation and genuine cross-view value conflicts.
        """
        if phase not in {"open", "reread"}:
            raise ValueError("skill-card detail OCR phase must be open or reread")
        if key is None:
            raise ArenaReaderError("skill_card_detail_domain_unconfirmed", "detail transaction key is missing")
        started = self.clock.perf_counter()
        try:
            text = read_skill_detail_capture_text(self.port, image, phase, key, self._simd_detail_view_complete)
            lookup = getattr(self.port, "_cached_full_frame_ocr_evidence", None)
            evidence = lookup(image) if lookup is not None else None
            if evidence is not None and evidence.image is image and evidence.hit:
                pairs = self._numeric_fragment_pairs(evidence.all_items)
                if pairs:
                    return _SkillDetailTextWithNumericFragments(
                        text, lambda card_id, current_key: self._validate_native_numeric_fragments(
                            image, evidence.all_items, pairs, card_id, key, current_key,
                        ),
                    )
            return text
        finally:
            self.port._increment(f"skill_card_detail_{phase}_ocr_reads")
            self.port._add_timing(f"skill_card_detail_{phase}_ocr", self.clock.perf_counter() - started)

    def _simd_detail_view_complete(self, image, rows, candidate_ids):
        try:
            patterns = {
                card_id: re.compile(self.port.catalog.skill_card_title_anchor_pattern(card_id))
                for card_id in candidate_ids
            }
            owners = [
                (card_id, row) for row in rows for card_id, pattern in patterns.items()
                if pattern.fullmatch(_text(row).strip())
            ]
            if len(owners) != 1:
                return False
            card_id, title = owners[0]
            title_box = _box(title)
            height, width = image.shape[:2]
            roi = self.port._title_anchored_effect_roi_without_background_parameters(width, height, title_box, rows)
            local = tuple(row for row in rows if self._inside_detail_body(_box(row), title_box, roi))
            if not local or self._native_omission_operand_conflicts(tuple((_text(row), _box(row)) for row in local)):
                return False
            if self._numeric_fragment_pairs(local):
                return False
            body = self._scoped_omission_body_text(local, title_box, roi)
            return self.port.catalog.has_complete_skill_card_effect_view(card_id, body)
        except (ArenaCatalogError, ArenaReaderError):
            return False

    def _title_anchored_effect_roi_without_background_parameters(
        self,
        frame_width: int,
        frame_height: int,
        title_box: tuple[int, int, int, int],
        items: Sequence[Any],
    ) -> tuple[int, int, int, int]:
        """Bound error-only local OCR by its unique title and known stat column."""

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

        self._assert_detail_recovery_binding(image, card_id)
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
        roi = (left, top, roi_width, roi_height)
        evidence = self.port._cached_full_frame_ocr_evidence(image)
        native_conflicts = self._native_view_numeric_fragment_conflicts(native_items, image, title_box, roi, evidence)
        if any(not all(self._inside_detail_body(_box(item), title_box, roi) for item in pair)
               for pair in native_conflicts):
            raise self._numeric_boundary_error()
        crop = image[top : top + roi_height, left : left + roi_width]
        if crop.size == 0:
            raise ArenaReaderError(
                "skill_card_detail_effect_roi_invalid",
                f"title-anchored detail effect ROI is empty for frame {width}x{height}",
            )
        # Preserve the native full-frame OCR boxes from the same recognition
        # that proved the title. The enlarged crop is valuable for small
        # glyphs, but interpolation can also erase a small standalone value.
        # Keep native atom boundaries in both same-frame views;
        # catalog resolution remains fail-closed when the views conflict.
        native_roi_items = []
        for item in native_items:
            item_x, item_y, item_width, item_height = _box(item)
            centre_x = item_x + item_width / 2.0
            centre_y = item_y + item_height / 2.0
            if left <= centre_x <= left + roi_width and top <= centre_y <= top + roi_height:
                native_roi_items.append(item)
        body_items = tuple(item for item in native_roi_items if _box(item)[1] + _box(item)[3] / 2 >= title_box[1] + title_box[3])
        body_atoms = tuple((_text(item), _box(item)) for item in body_items)
        omission_conflicts = tuple(
            (body_items[field], body_items[fragment])
            for field, fragment in self._native_omission_operand_conflicts(body_atoms, include_symbols=True)
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
            return self._read_derived_effect_roi(
                enlarged, roi, title_pattern, title_box,
                lambda projected: self._merge_title_anchored_effect_observations(
                    image, card_id, matches[0], title_pattern, roi, native_items,
                    native_roi_items, (*native_conflicts, *omission_conflicts), projected,
                ),
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

    def _read_derived_effect_roi(self, image, roi, title_pattern, native_title_box, build):
        """Use the shared ROI order and keep rejected backend observations separate."""
        context = getattr(self.port, "context", None)
        selected = None

        def project(rows):
            projected = project_domain_items(rows, roi)
            if projected is None:
                raise ArenaReaderError("skill_card_detail_domain_unconfirmed", "derived OCR crosses its input crop")
            return projected

        def accept(rows):
            nonlocal selected
            self.port._check_cancelled()
            try:
                projected = project(rows)
                titles = tuple(item for item in projected if re.fullmatch(title_pattern, _text(item).strip()))
                if len(titles) != 1 or not self._overlaps_operand(_box(titles[0]), native_title_box):
                    return False
                title_box = _box(titles[0])
                if not self._scoped_omission_body_text(projected, title_box, roi):
                    return False
                selected = build(projected)
            except ArenaReaderError:
                self.port._increment("derived_roi_simd_ocr_rejections")
                return False
            self.port._increment("derived_roi_simd_ocr_acceptances")
            return True

        def native_read():
            self.port._with_full_frame_ocr_kind(
                "derived_roi", self.port._full_ocr_text, image, spatial_reading_order=False,
            )
            observed = self.port._cached_full_frame_ocr_evidence(image)
            if observed is None or observed.image is not image or not observed.hit or observed.kind != "derived_roi":
                raise ArenaReaderError("skill_card_detail_effect_roi_empty", "bounded ROI has no matching OCR evidence")
            return OcrRows(observed.all_items)

        if context is not None:
            self.port._check_cancelled()
        result = read_ocr_node(
            context, image, "ArenaReaderOCR",
            override={"recognition": "OCR", "expected": r".+", "order_by": "Vertical"},
            all_results=True, accept=accept, prefer_gpu=True, native_read=native_read,
        )
        return selected if selected is not None else build(project(result.rows))

    def _merge_title_anchored_effect_observations(
        self, image, card_id, title, title_pattern, roi, native_items, native_roi_items, disputed, projected,
    ) -> str:
        title_box = _box(title)
        if self._numeric_fragment_pairs(projected):
            raise self._native_operand_effect_view_conflict() if disputed else self._numeric_boundary_error()
        excluded = ()
        deferred = ()
        if disputed:
            derived_titles = tuple(item for item in projected if re.fullmatch(title_pattern, _text(item).strip()))
            if len(derived_titles) != 1 or not self._overlaps_operand(_box(derived_titles[0]), title_box):
                raise self._native_operand_effect_view_conflict()
            signatures = Counter((_text(item), _box(item)) for item in native_items)
            omitted_keys, separated_keys = set(), set()
            for field, fragment in disputed:
                fragment_key = (_text(fragment), _box(fragment))
                decision = self._compare_derived_numeric_fragment(field, fragment, projected)
                if decision == "separate":
                    separated_keys.add(fragment_key)
                elif (
                    decision == "absent"
                    and signatures[fragment_key] == 1
                    and self._inside_detail_body(_box(fragment), title_box, roi)
                ):
                    omitted_keys.add(fragment_key)
                else:
                    raise self._native_operand_effect_view_conflict()
            excluded = tuple(item for item in native_items if (_text(item), _box(item)) in omitted_keys)
            deferred = tuple(item for item in native_items if (_text(item), _box(item)) in separated_keys)
            affected = omitted_keys | separated_keys
            native_roi_items = [item for item in native_roi_items if (_text(item), _box(item)) not in affected]
            if excluded:
                self.port._increment("skill_card_detail_enlarged_fragment_omissions")
        original_items = native_items
        native_body = tuple(item for item in native_roi_items if self._inside_detail_body(_box(item), title_box, roi))
        native_items = self._coalesce_omission_field_atoms(native_items, native_body, card_id)
        if native_items is not original_items:
            native_roi_items = [item for item in native_items if _inside_roi(_box(item), roi)
                                and all(item is not fragment for fragment in (*deferred, *excluded))]
        repair = self.recover_skill_effect_body(
            tuple((_text(item), _box(item)) for item in native_roi_items), title_box, roi,
            self.port._spatial_ocr_rows(native_roi_items),
        )
        native_text = repair.text if repair.changed else self._ordered_effect_atom_text(native_roi_items)
        if deferred:
            native_text = "\n".join((native_text, *self._isolated_numeric_atom_lines(deferred)))
        full_text = self._repaired_full_effect_text(
            native_items, native_roi_items, repair, excluded=excluded, deferred=deferred,
        )
        derived_body = tuple(item for item in projected if self._inside_detail_body(_box(item), title_box, roi))
        derived_items = self._coalesce_omission_field_atoms(projected, derived_body, card_id)
        validate_cost = getattr(self.port.catalog, "_assert_typed_detail_cost_atoms", None)
        if validate_cost is not None:
            try:
                validate_cost(card_id, tuple(
                    _text(item)
                    for view in (native_roi_items, derived_items)
                    for item in view
                    if self._inside_detail_body(_box(item), title_box, roi)
                ))
            except StructuralOmissionFieldConflict as error:
                raise ArenaReaderError("skill_card_detail_structural_field_conflict", str(error)) from error
        enlarged_text = self._repair_derived_effect_text(
            derived_items, title_pattern, roi,
            native_items=native_roi_items, native_title=title_box, card_id=card_id,
        )
        observations = tuple(value for value in (native_text, enlarged_text) if value.strip())
        texts = tuple(dict.fromkeys(observations))
        if not texts:
            raise ArenaReaderError(
                "skill_card_detail_effect_roi_empty",
                "title-bound native and enlarged OCR views were both empty",
            )
        try:
            merged = self.port._merge_skill_card_effect_detail_views(card_id, (*texts, full_text))
        except ArenaReaderError as error:
            if error.code != "skill_card_detail_effect_view_conflict":
                raise
            recovered = recover_wrapped_effect_decoration(
                tuple((_text(item), _box(item)) for item in native_roi_items),
                title_box, roi,
                self.port._spatial_ocr_rows(native_roi_items),
                corroborating_atoms=tuple((_text(item), _box(item)) for item in projected),
            )
            if not recovered.changed:
                raise
            deferred = (*deferred, *(native_roi_items[index] for index in recovered.deferred_indices))
            filtered = tuple(item for index, item in enumerate(native_roi_items)
                             if index not in (*recovered.excluded_indices, *recovered.deferred_indices))
            composed = self.recover_skill_effect_body(
                tuple((_text(item), _box(item)) for item in filtered), title_box,
                roi, self.port._spatial_ocr_rows(filtered),
            )
            native_text = composed.text if composed.changed else self._ordered_effect_atom_text(filtered)
            if deferred:
                native_text = "\n".join((native_text, *self._isolated_numeric_atom_lines(deferred)))
            full_text = self._repaired_full_effect_text(
                native_items, filtered, composed,
                excluded=(*excluded, *(native_roi_items[index] for index in recovered.excluded_indices)),
                deferred=deferred,
            )
            merged = self.port._merge_skill_card_effect_detail_views(card_id, (native_text, enlarged_text, full_text))
            self.port._increment("skill_card_detail_wrapped_decoration_repairs")
        return _TitleBoundEffectRoiText(
            merged, len(observations), recovered_full_text=full_text,
            scoped_body_text=self._scoped_omission_body_text(derived_items, title_box, roi),
            scoped_native_text="\n".join((
                _text(title), self._scoped_omission_body_text(native_roi_items, title_box, roi),
            )),
            source_binding=self._effect_roi_source_binding(
                image, card_id, roi, native_items, full_text,
            ),
        )

    def _skill_card_title_anchored_enhanced_effect_roi_text(
        self,
        image: Any,
        card_id: int,
    ) -> str:
        """Contrast one title-bound effect panel on the already-captured frame."""

        import cv2

        self._assert_detail_recovery_binding(image, card_id)
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
            roi = (left, top, roi_width, roi_height)

            def build(projected):
                self._assert_effect_numeric_atom_boundaries(projected)
                return _TitleBoundEffectRoiText(
                    self._repair_derived_effect_text(projected, title_pattern, roi), 1,
                    scoped_body_text=self._scoped_omission_body_text(projected, _box(matches[0]), roi),
                )

            return self._read_derived_effect_roi(enhanced, roi, title_pattern, _box(matches[0]), build)
        finally:
            self.port._add_timing(
                "skill_card_detail_enhanced_effect_roi_ocr",
                self.clock.perf_counter() - started,
            )

    @staticmethod
    def _neighboring_effect_atoms(first, second):
        x, y, width, height = first
        next_x, next_y, _, next_height = second
        scale = min(height, next_height)
        overlap = min(y + height, next_y + next_height) - max(y, next_y)
        return scale > 0 and overlap >= .5 * scale and -.25 * scale <= next_x - x - width <= .5 * scale

    def _scoped_omission_body_text(self, items, title_box, roi):
        in_body = tuple(item for item in items if self._inside_detail_body(_box(item), title_box, roi))
        spans = []
        for _, _, row in self.port._spatial_ocr_rows(in_body):
            previous = None
            for text, box in row:
                if previous is not None and self._neighboring_effect_atoms(previous, box):
                    spans[-1] += text
                else:
                    spans.append(text)
                previous = box
        # Compaction must not turn a detached digit into an operand, including
        # when both OCR boxes were captured inside the title crop.
        return f"\n{NUMERIC_TOKEN_BOUNDARY}\n".join(spans)

    def _repair_derived_effect_text(
        self, items, title_pattern, roi, *, native_items=(), native_title=None, card_id=None,
    ):
        titles = [item for item in items if re.search(title_pattern, _text(item))]
        if len(titles) != 1:
            return self._ordered_effect_atom_text(items)
        if native_title is not None and card_id is not None:
            operand = recover_overlapping_enthusiasm_operand(
                tuple((_text(item), _box(item)) for item in native_items),
                tuple((_text(item), _box(item)) for item in items),
                native_title, _box(titles[0]),
            )
            if operand is not None and int(operand[1]) in self.port.catalog._fixed_enthusiasm_effect_values(card_id):
                index, value = operand
                items = tuple({"text": value, "box": _box(item)} if i == index else item
                              for i, item in enumerate(items))
                self.port._increment("skill_card_detail_overlapping_enthusiasm_operands_recovered")
        repair = self.recover_skill_effect_body(
            tuple((_text(item), _box(item)) for item in items), _box(titles[0]), roi,
            self.port._spatial_ocr_rows(items),
        )
        if not repair.changed:
            return self._ordered_effect_atom_text(items)
        replacements = {index: value for index, _, value in repair.replacements}
        repaired = tuple({"text": replacements.get(index, _text(item)), "box": _box(item)}
                         for index, item in enumerate(items))
        self._assert_effect_numeric_atom_boundaries(repaired)
        return repair.text

    def _effect_roi_source_binding(self, image, card_id, roi, native_items, full_text):
        keys = [key for key, frame in getattr(self.port, "_card_detail_images", {}).items() if frame is image]
        if len(keys) != 1:
            return None
        key = keys[0]
        token = getattr(self.port, "_card_transaction_tokens", {}).get(key)
        frozen = getattr(self.port, "_card_detail_source_frames", None)
        sources = tuple(frozen.get(key, ())) if frozen is not None else ()
        if (type(token) is not int or getattr(self.port, "_card_detail_open_ids", {}).get(key) != card_id
                or len(sources) != 3 or len({id(frame) for frame in sources}) != 3
                or any(frame is image for frame in sources)):
            return None
        return _EffectRoiSourceBinding(key, token, card_id, image, sources, roi, tuple(native_items), full_text)

    def _effect_roi_source_binding_is_current(self, binding):
        check = getattr(self.port, "_check_cancelled", None)
        if check is not None:
            check()
        current = self._effect_roi_source_binding(
            binding.detail_image, binding.card_id, binding.effect_roi, binding.native_items, binding.full_text,
        )
        return bool(current is not None and current.key == binding.key
                    and current.transaction_token == binding.transaction_token
                    and all(a is b for a, b in zip(current.source_frames, binding.source_frames, strict=True)))

    def _recover_source_background_effect_roi(
        self, image, card_id, view, *, budget: _SourceBackgroundOcrBudget,
    ):
        """Exclude proven old nonnumeric labels only after coverage failed.

        The caller retains the original raw text and all numeric/conflicting
        atoms. Source OCR never counts as another detail observation. At most
        two missing source observations may be read in the entire session.
        """
        from ._effect_source_background import certified_source_background_lines

        binding = getattr(view, "source_binding", None)
        if (not isinstance(view, _TitleBoundEffectRoiText) or binding is None
                or binding.detail_image is not image or binding.card_id != card_id
                or view.recovered_full_text != binding.full_text
                or not self._effect_roi_source_binding_is_current(binding)):
            return None
        atomic_views = tuple(part.strip() for part in view.split(_OCR_EFFECT_VIEW_BOUNDARY) if part.strip())
        if binding.full_text.strip() not in atomic_views:
            return None
        # There is nothing to certify if every native atom is already local.
        left, top, width, height = binding.effect_roi
        outer_labels = [item for item in binding.native_items
                        if not any(character.isnumeric() or character in "+＋-−－%％*/=<>" for character in _text(item))
                        and (_box(item)[0] + _box(item)[2] <= left or _box(item)[0] >= left + width
                             or _box(item)[1] + _box(item)[3] <= top or _box(item)[1] >= top + height)]
        if not outer_labels:
            return None
        sources = []
        for source in binding.source_frames:
            if not self._effect_roi_source_binding_is_current(binding):
                return None
            evidence = self.port._cached_full_frame_ocr_evidence(source)
            if evidence is None and budget.remaining > 0:
                budget.remaining -= 1
                self.port._increment("skill_card_detail_source_background_ocr_reads")
                try:
                    self.port._with_full_frame_ocr_kind("source_reference", self.port._full_ocr_text, source)
                except ArenaReaderError:
                    # Optional source attribution must not turn a failed
                    # source OCR into a new terminal detail failure. Stop and
                    # supersession exceptions are separate and propagate.
                    if not self._effect_roi_source_binding_is_current(binding):
                        return None
                    self.port._increment("skill_card_detail_source_background_ocr_failures")
                    return None
                evidence = self.port._cached_full_frame_ocr_evidence(source)
            if not self._effect_roi_source_binding_is_current(binding):
                return None
            sources.append(tuple({"text": _text(item), "box": _box(item)} for item in evidence.all_items)
                           if isinstance(evidence, _FullFrameOcrEvidence) and evidence.image is source
                           and evidence.hit and evidence.kind != "derived_roi" else ())
        removed = certified_source_background_lines(
            tuple({"text": _text(item), "box": _box(item)} for item in binding.native_items),
            effect_roi=binding.effect_roi, full_text=binding.full_text, source_frames=tuple(sources),
        )
        if not removed:
            return None
        scoped_full = "\n".join(line for line in binding.full_text.splitlines() if line not in removed)
        try:
            merged = self.port._merge_skill_card_effect_detail_views(
                card_id, tuple(scoped_full if part == binding.full_text.strip() else part for part in atomic_views),
            )
        except ArenaReaderError:
            # Keep the original unmodified views and their existing recovery
            # path if removing old labels cannot preserve semantic agreement.
            return None
        if not self._effect_roi_source_binding_is_current(binding):
            return None
        self.port._increment("skill_card_detail_source_background_recoveries")
        self.logger.debug(json.dumps({"event": "arena_skill_detail_source_background_recovered",
                                     "key": binding.key, "card_id": card_id,
                                     "removed_old_labels": sorted(removed)}, ensure_ascii=False))
        return _TitleBoundEffectRoiText(
            merged, view.observation_count, recovered_full_text=scoped_full,
            scoped_body_text=view.scoped_body_text,
            scoped_native_text=view.scoped_native_text,
            source_binding=binding._replace(full_text=scoped_full),
        )

    def _assert_detail_recovery_binding(self, image: Any, card_id: int) -> None:
        frames = getattr(self.port, "_card_detail_images", None)
        if frames is None:  # Legacy pure helper ports have no session state.
            return
        keys = [key for key, frame in frames.items() if frame is image]
        if len(keys) != 1:
            raise ArenaReaderError("skill_card_detail_domain_unconfirmed", "recovery frame has no unique transaction")
        expected = getattr(self.port, "_card_detail_open_ids", {}).get(keys[0])
        if expected is not None and expected != card_id:
            raise ArenaReaderError("skill_card_detail_id_changed", "recovery card differs from the opened detail")

    @staticmethod
    def _numeric_fragment_pairs(items: Sequence[Any]) -> tuple[tuple[Any, Any], ...]:
        """Find possible contiguous suffixes without interpreting either box."""
        effects = []
        for item in items:
            value = _text(item).strip()
            if _ADDED_NUMERIC_EFFECT_MARKERS.search(value) and re.search(r"[0-9]+$", value):
                effects.append((item, _box(item)))
        if not effects:
            return ()
        numbers = [(item, _box(item)) for item in items if re.fullmatch(r"[0-9]+", _text(item).strip())]
        pairs = []
        for item, (x, y, width, height) in effects:
            for other, (a, b, _, h) in numbers:
                scale = min(height, h)
                overlap = min(y + height, b + h) - max(y, b)
                if (scale > 0 and max(height, h) <= 1.7 * scale
                        and overlap >= 0.5 * scale and 0 <= a - x - width <= 0.75 * scale):
                    pairs.append((item, other))
        return tuple(pairs)

    @staticmethod
    def _numeric_boundary_error() -> ArenaReaderError:
        return ArenaReaderError(
            "skill_card_detail_numeric_boundary_unconfirmed", "adjacent OCR atoms may split one numeric effect value",
        )

    def _assert_effect_numeric_atom_boundaries(self, items: Sequence[Any]) -> None:
        if self._numeric_fragment_pairs(items):
            raise self._numeric_boundary_error()

    def _validate_native_numeric_fragments(self, image, items, pairs, card_id, key, current_key):
        if (key != current_key or self.port._card_detail_images.get(key) is not image
                or self.port._card_detail_open_ids.get(key) != card_id):
            raise self._numeric_boundary_error()
        pattern = self.port._skill_card_frame_title_pattern(image, card_id)
        titles = [_box(item) for item in items if re.fullmatch(pattern, _text(item).strip())]
        if len(titles) != 1:
            raise self._numeric_boundary_error()
        height, width = image.shape[:2]
        title = titles[0]
        roi = self.port._title_anchored_effect_roi_without_background_parameters(width, height, title, items)
        self._assert_scoped_numeric_fragments(items, image, title, roi, pairs=pairs)

    def _assert_native_view_numeric_fragments(self, items, image, title, roi, evidence):
        if self._native_view_numeric_fragment_conflicts(items, image, title, roi, evidence):
            raise self._numeric_boundary_error()

    def _native_view_numeric_fragment_conflicts(self, items, image, title, roi, evidence):
        conflicts = self._scoped_numeric_fragment_conflicts(items, image, title, roi)
        if (isinstance(evidence, _FullFrameOcrEvidence) and evidence.image is image and evidence.hit and evidence.kind != "derived_roi"
                and items is not evidence.all_items):
            conflicts += self._scoped_numeric_fragment_conflicts(evidence.all_items, image, title, roi)
        return conflicts

    def _assert_scoped_numeric_fragments(self, items, image, title, roi, *, pairs=None):
        if self._scoped_numeric_fragment_conflicts(items, image, title, roi, pairs=pairs):
            raise self._numeric_boundary_error()

    @staticmethod
    def _inside_detail_body(box, title, roi):
        x, y, w, h = box
        left, top, width, height = roi
        return left <= x and max(top, title[1] + title[3]) <= y and x + w <= left + width and y + h <= top + height

    def _scoped_numeric_fragment_conflicts(self, items, image, title, roi, *, pairs=None):
        pairs = self._numeric_fragment_pairs(items) if pairs is None else pairs
        if not pairs:
            return ()
        height, width = image.shape[:2]
        right = roi[0] + roi[2]
        crop_x, _, crop_width, _ = self.port._title_anchored_effect_roi(width, height, title)
        proven_stat_column = right < crop_x + crop_width
        conflicts = []
        for effect, digit in pairs:
            # The exact digit must take part in the independent, title-adjacent
            # stat-column proof. The crop edge alone does not classify a digit.
            if (proven_stat_column and self._inside_detail_body(_box(effect), title, roi) and _box(digit)[0] >= right
                    and self.port._background_parameter_column_left(
                        width, height, title, tuple(item for item in items if item is not digit),
                    ) is None and self._cached_source_stat_matches(image, digit)):
                continue
            conflicts.append((effect, digit))
        return tuple(conflicts)

    @staticmethod
    def _overlaps_operand(first, second):
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        return (min(ax + aw, bx + bw) - max(ax, bx) >= .5 * max(aw, bw)
                and min(ay + ah, by + bh) - max(ay, by) >= .5 * max(ah, bh))

    def _compare_derived_numeric_fragment(self, field, fragment, projected):
        """Keep corroborated fragments, or omit an isolated number or symbol."""
        text, fragment_text = (_normalise_operand_atom(_text(item)) for item in (field, fragment))
        match = _LABELED_EFFECT_OPERAND.search(text)
        symbol_fragment = _is_symbol_fragment(fragment_text)
        if match is None or not (
            _DETACHED_EFFECT_OPERAND.fullmatch(fragment_text) or symbol_fragment
        ):
            return None
        atoms = tuple((_normalise_operand_atom(_text(item)), _box(item)) for item in projected)
        overlapping_fields = tuple(
            index for index, (value, box) in enumerate(atoms)
            if self._overlaps_operand(box, _box(field))
            and (derived_field := _LABELED_EFFECT_OPERAND.search(value)) is not None
            and derived_field["label"] == match["label"]
        )
        fields = tuple(index for index in overlapping_fields if atoms[index][0] == text)
        fragments = tuple(
            value for value, box in atoms
            if value == fragment_text and self._overlaps_operand(box, _box(fragment))
        )
        if len(fields) != 1 or len(overlapping_fields) != 1 or any(
            index == fields[0] for index, _ in self._native_omission_operand_conflicts(atoms, include_symbols=True)
        ):
            return None
        if len(fragments) == 1:
            return "separate"
        if not any(
            (_EFFECT_OPERAND_FRAGMENT.fullmatch(value) or _is_symbol_fragment(value) or re.search(r"[0-9]", value)
             or (symbol_fragment and any(char in value for char in fragment_text)))
            and self._overlaps_fragment_area(box, _box(fragment))
            for value, box in atoms
        ):
            return "absent"
        return None

    @staticmethod
    def _overlaps_fragment_area(box, fragment):
        x, y, w, h = box
        a, b, iw, ih = fragment
        overlap_x = max(0, min(x + w, a + iw) - max(x, a))
        overlap_y = max(0, min(y + h, b + ih) - max(y, b))
        # A sharper crop can box a real character much more tightly.
        smaller = min(w * h, iw * ih)
        return smaller > 0 and overlap_x * overlap_y >= .5 * smaller

    def _cached_source_stat_matches(self, detail_image: Any, digit: Any) -> bool:
        keys = [key for key, frame in self.port._card_detail_images.items() if frame is detail_image]
        if len(keys) != 1:
            return False
        sources = self.port._card_count_frames.get(keys[0][0], ())
        signature = (_text(digit).strip(), _box(digit))
        matches = set()
        for source in sources:
            if source is detail_image or id(source) in matches:
                continue
            evidence = self.port._cached_full_frame_ocr_evidence(source)
            if (isinstance(evidence, _FullFrameOcrEvidence) and evidence.image is source and evidence.hit
                    and evidence.kind != "derived_roi"
                    and sum((_text(item).strip(), _box(item)) == signature for item in evidence.all_items) == 1):
                matches.add(id(source))
        return len(matches) >= 2

    def _repaired_full_effect_text(
        self, full_items: Sequence[Any], local_items: Sequence[Any], repair: EffectBodyRecovery,
        *, excluded: Sequence[Any] = (), deferred: Sequence[Any] = (),
    ) -> str:
        """Preserve outer semantic evidence when replacing a failed local view.

        Only exact, unique local atoms licensed by the existing error recovery
        may change. Cropping must not erase a conflicting effect elsewhere.
        """
        signatures = [(_text(item), _box(item)) for item in full_items]
        replacements = {(_text(local_items[index]), _box(local_items[index])): value
                        for index, _, value in repair.replacements}
        separated = {(_text(item), _box(item)) for item in (*excluded, *deferred)}
        local_signatures = {(_text(item), _box(item)) for item in local_items}
        local, outer = [], []
        for item, signature in zip(full_items, signatures, strict=True):
            unique = signatures.count(signature) == 1
            if unique and signature in separated:
                continue
            (local if signature in local_signatures else outer).append(
                {"text": replacements[signature], "box": _box(item)}
                if unique and signature in replacements else item)
        # Determine the reading region before ordering: a remote attribute
        # label must not be inserted between a local effect and its operand.
        # Outer atoms remain available to the same catalog conflict check.
        return "\n".join(value for value in (self._ordered_effect_atom_text(local),
                                              self._ordered_effect_atom_text(outer),
                                              *self._isolated_numeric_atom_lines(deferred)) if value)

    @staticmethod
    def _isolated_numeric_atom_lines(items: Sequence[Any]) -> tuple[str, ...]:
        return tuple(f"{NUMERIC_TOKEN_BOUNDARY}{_text(item)}{NUMERIC_TOKEN_BOUNDARY}" for item in items)

    def _ordered_effect_atom_text(self, items: Sequence[Any]) -> str:
        """Restore local reading order without concatenating numeric atoms.

        This is restricted to failed detail views already bound to one title.
        Keeping each atom on its own line lets the catalog retain its existing
        digit-to-digit lexical boundary protection.
        """
        self._assert_effect_numeric_atom_boundaries(items)
        return "\n".join(text for _, _, atoms in self.port._spatial_ocr_rows(items) for text, _ in atoms)

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
