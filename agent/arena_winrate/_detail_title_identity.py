"""Resolve one clicked title from the existing capture and frozen source.

This component owns no title cache or transaction state. Every read, write and
cross-helper call goes through its explicit port to the current backend view.
"""

from __future__ import annotations

import re
from typing import Any, Protocol
from collections.abc import Callable, Sequence

from .catalog import ArenaCatalogError, ArenaEntityCatalog
from ._reader_errors import ArenaReaderError
from ._reader_visual import _box, _text
from ._detail_title_region import isolated_title_region
from ._detail_capture_evidence import _TrustedSkillCardTitleRowsEvidence


class _EvidenceClock(Protocol):
    def perf_counter(self) -> float: ...


class _EvidenceLogger(Protocol):
    def debug(self, message: str) -> None: ...


class DetailTitleIdentityPort(Protocol):
    """Only the state views and operations used by this responsibility."""

    def _add_timing(self, name: str, elapsed: float) -> None: ...

    def _assert_skill_detail_open_id(self, key: tuple[int, int] | None, card_id: int) -> None: ...

    def _authoritative_skill_card_title_text(
        self,
        group_index: int | None,
        detail_image: Any | None,
        *,
        candidate_card_ids: Sequence[int] = (),
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> str | None: ...

    _card_count_frames: dict[int, tuple[Any, Any, Any]]

    _card_detail_open_ids: dict[tuple[int, int], int]

    _card_rows: dict[int, tuple[tuple[int, int, int, int], ...]]

    _card_transaction_contact_counts: dict[tuple[int, int], int]

    _card_transaction_serial: int

    def _confirm_clicked_skill_card_id(
        self,
        detail_text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = None,
        detail_image: Any | None = None,
        source_card_box: tuple[int, int, int, int] | None = None,
        source_card_slot: int | None = None,
    ) -> int: ...

    def _confirm_clicked_skill_card_id_from_title(
        self,
        detail_text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = None,
        detail_image: Any | None = None,
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> int: ...

    def _increment(self, name: str) -> None: ...

    def _isolated_skill_card_title_entry(self, image: Any, group: int, box: Any) -> Any: ...

    def _isolated_skill_card_title_key(self, image: Any, group: int, box: Any) -> tuple: ...

    _isolated_title_evidence: dict[tuple, tuple[Any, ...]]

    def _normalized_skill_card_ocr_lines(self, text: str) -> tuple[str, ...]: ...

    def _ocr(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = None, only_rec: bool = False) -> list[Any]: ...

    def _recover_isolated_skill_card_title(self, group: int, image: Any, box: Any) -> str | None: ...

    def _recover_missing_upgrade_title(
        self, image: Any, key: tuple[int, int], source_box: Any, base_id: int, expected_id: int
    ) -> int | None: ...

    def _recover_unmatched_skill_card_title(
        self, group_index: int, image: Any, candidate_ids: Sequence[int], source_card_box: tuple[int, int, int, int]
    ) -> str | None: ...

    def _resolve_proven_skill_card_title(self, title_text: str, *, allow_one_character: bool = True) -> int: ...

    _runtime_counts: dict[str, int]

    _runtime_timing_seconds: dict[str, float]

    _skill_card_recovered_title_frames: dict[tuple[int, int], tuple[Any, str]]

    def _skill_card_scope_kwargs(self, slot_index: int, *, plan: str | None = None) -> dict[str, Any]: ...

    def _skill_card_title_ocr_segments(self, items: Sequence[Any]) -> tuple[tuple[str, tuple[int, int, int, int]], ...]: ...

    def _skill_card_title_row_has_neutral_ink(self, image: Any, box: tuple[int, int, int, int]) -> bool: ...

    def _spatial_ocr_rows(
        self, items: Sequence[Any]
    ) -> tuple[tuple[str, tuple[int, int, int, int], tuple[tuple[str, tuple[int, int, int, int]], ...]], ...]: ...

    def _stable_skill_card_source_ocr_counts(self, group_index: int) -> dict[str, int]: ...

    def _trusted_skill_card_title_rows(
        self, image: Any, *, candidate_card_ids: Sequence[int] = (), source_card_box: tuple[int, int, int, int] | None = None
    ) -> tuple[str, ...]: ...

    _trusted_skill_card_title_rows_cache: dict[tuple, _TrustedSkillCardTitleRowsEvidence]

    _upgrade_title_roi_evidence: dict[tuple, tuple[Any, str | None, int | None]]

    def _with_full_frame_ocr_kind(self, kind: str, operation: Any, *args: Any, **kwargs: Any) -> Any: ...

    catalog: ArenaEntityCatalog


class DetailTitleIdentity:
    def __init__(self, port: DetailTitleIdentityPort, *, clock: _EvidenceClock, logger: _EvidenceLogger) -> None:
        self.port = port
        self.clock = clock
        self.logger = logger

    def _confirm_clicked_skill_card_id(
        self,
        detail_text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = None,
        detail_image: Any | None = None,
        source_card_box: tuple[int, int, int, int] | None = None,
        source_card_slot: int | None = None,
    ) -> int:
        """A full title may exceed visual Top-K, but must belong to the legal slot."""
        scope = self.port._skill_card_scope_kwargs(source_card_slot - 1).get("eligible_card_ids") if source_card_slot is not None else None
        if scope is not None:
            candidate_ids = tuple(value for value in candidate_ids if value in scope)
            if not candidate_ids:
                candidate_ids = tuple(sorted(scope))
        resolved = self.port._confirm_clicked_skill_card_id_from_title(
            detail_text,
            candidate_ids,
            expected_customization_count=expected_customization_count,
            source_group_index=source_group_index,
            detail_image=detail_image,
            source_card_box=source_card_box,
        )
        if source_group_index is not None and source_card_slot is not None:
            key = (source_group_index, source_card_slot)
            expected = getattr(self.port, "_card_detail_open_ids", {}).get(key)
            pair = getattr(self.port.catalog, "is_skill_card_upgrade_pair", None)
            if expected is not None and pair is not None and pair(resolved, expected):
                repaired = self.port._recover_missing_upgrade_title(
                    detail_image,
                    key,
                    source_card_box,
                    resolved,
                    expected,
                )
                if repaired is not None:
                    resolved = repaired
        if scope is not None and resolved not in scope:
            raise ArenaCatalogError(
                f"detail card {resolved} is outside the stage/slot domain for group {source_group_index}/slot {source_card_slot}"
            )
        if source_group_index is not None and source_card_slot is not None:
            self.port._assert_skill_detail_open_id((source_group_index, source_card_slot), resolved)
        return resolved

    def _recover_missing_upgrade_title(
        self,
        image: Any,
        key: tuple[int, int],
        source_box: Any,
        base_id: int,
        expected_id: int,
    ) -> int | None:
        """One same-frame title crop; a prior upgraded ID alone never repairs '+'."""
        if image is None or not hasattr(image, "shape") or source_box is None:
            return None
        import cv2

        cache = getattr(self.port, "_upgrade_title_roi_evidence", None)
        if cache is None:
            cache = self.port._upgrade_title_roi_evidence = {}
        cache_key = (id(image), key[0], tuple(source_box), base_id)
        cached = cache.get(cache_key)
        if cached is not None and cached[0] is image:
            return cached[2]
        # Cache unsuccessful observations too: repeated parsing of this frame
        # cannot spend another recognition attempt or count as a fresh vote.
        cache[cache_key] = (image, None, None)
        while len(cache) > 16:
            cache.pop(next(iter(cache)))
        base_title = self.port.catalog.skill_card_title(base_id)
        normalize = self.port.catalog.normalize_skill_card_title_text
        items = self.port._ocr(image, r".+")
        # Match the title resolver's cross-column splitting. A visual row can
        # also contain background stats (e.g. "08 100%") to the right of the
        # popup; merging that whole row prevents the failure-only crop running.
        candidates = tuple((_text(item).strip(), _box(item)) for item in items)
        candidates += self.port._skill_card_title_ocr_segments(items)
        anchors = sorted({box for text, box in candidates if normalize(text) == normalize(base_title)})
        if len(anchors) != 1:
            return None
        x, y, width, height = anchors[0]
        ih, iw = image.shape[:2]
        # The full-frame resolver already proved this row belongs to the
        # opened detail. Keep the crop within that row, never the effect '+1'.
        pad = max(1, round(height * 0.15))
        left, top = max(0, x - pad), max(0, y - pad)
        right, bottom = min(iw, x + width + height), min(ih, y + height + pad)
        if width <= 0 or height <= 0 or right <= left or bottom <= top:
            return None
        crop = cv2.resize(image[top:bottom, left:right], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        self.port._increment("skill_card_upgrade_title_roi_reads")
        items = self.port._with_full_frame_ocr_kind("derived_roi", self.port._ocr, crop, r".+")
        crop_rows = self.port._spatial_ocr_rows(items)
        if len(crop_rows) != 1:
            return None
        title = crop_rows[0][0]
        try:
            actual = self.port.catalog.confirm_clicked_skill_card_by_exact_title(title)
        except ArenaCatalogError:
            return None
        if actual != expected_id:
            return None
        cache[cache_key] = (image, title, actual)
        recovered = getattr(self.port, "_skill_card_recovered_title_frames", None)
        if recovered is None:
            recovered = self.port._skill_card_recovered_title_frames = {}
        recovered[(id(image), actual)] = (image, base_title)
        while len(recovered) > 32:
            recovered.pop(next(iter(recovered)))
        self.port._increment("skill_card_upgrade_title_roi_recoveries")
        return actual

    def _confirm_clicked_skill_card_id_from_title(
        self,
        detail_text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = None,
        detail_image: Any | None = None,
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> int:
        """Bind a clicked detail to one business ID under bounded fallbacks.

        A complete exact title line from a full member-detail frame may rebind
        outside the visual family when the active RIS catalog is newer than
        the fixed gallery.  Existing bounded title recovery remains available
        for historical OCR cases; the positive-count fallback is never called
        for an explicit zero.
        """

        candidate_ids = tuple(dict.fromkeys(candidate_ids))
        if not candidate_ids or any(isinstance(card_id, bool) or not isinstance(card_id, int) or card_id < 1 for card_id in candidate_ids):
            raise ArenaCatalogError("visual identity contains no valid skill-card candidate IDs")

        known_ids_resolver = getattr(
            self.port.catalog,
            "skill_card_business_ids",
            None,
        )
        if known_ids_resolver is not None:
            known_ids = frozenset(known_ids_resolver())
            unknown_ids = tuple(card_id for card_id in candidate_ids if card_id not in known_ids)
            if unknown_ids:
                raise ArenaCatalogError(f"visual identity references unknown skill-card IDs {unknown_ids!r}")

        title_text = self.port._authoritative_skill_card_title_text(
            source_group_index,
            detail_image,
            candidate_card_ids=candidate_ids,
            source_card_box=source_card_box,
        )
        if title_text is None and detail_image is not None and source_card_box is not None:
            isolated = self.port._isolated_skill_card_title_entry(detail_image, source_group_index, source_card_box)
            opened = getattr(self.port, "_card_detail_open_ids", {})
            if (
                isolated is not None
                and isolated[3] is not None
                and any(
                    (source_group_index, slot + 1) in opened and tuple(row) == tuple(source_card_box)
                    for slot, row in enumerate(getattr(self.port, "_card_rows", {}).get(source_group_index, ()))
                )
            ):
                raise ArenaReaderError(
                    "skill_card_detail_title_unresolved",
                    "the isolated detail title could not be uniquely read",
                )
        # Production detail transactions always carry their frame. When that
        # frame exists, every identity resolver is restricted to one dynamically
        # proven title row. A missing or ambiguous title therefore fails closed
        # instead of letting an effect row such as ``眠気`` masquerade as another
        # card name.
        identity_text = detail_text if detail_image is None else "" if title_text is None else title_text
        exact_title_error: ArenaCatalogError | None = None
        exact_resolver = getattr(
            self.port.catalog,
            "confirm_clicked_skill_card_by_exact_title",
            None,
        )
        if exact_resolver is not None and title_text is not None:
            try:
                return self.port._resolve_proven_skill_card_title(title_text)
            except ArenaCatalogError as error:
                exact_title_error = error

        candidate_error: ArenaCatalogError | None = None
        try:
            return self.port.catalog.confirm_clicked_skill_card_candidates(
                identity_text,
                candidate_card_ids=candidate_ids,
            )
        except ArenaCatalogError as error:
            candidate_error = error
        if title_text is not None:
            note_resolver = getattr(
                self.port.catalog,
                "confirm_clicked_skill_card_by_terminal_note_title",
                None,
            )
            if note_resolver is not None:
                try:
                    recovered_id = note_resolver(title_text)
                except ArenaCatalogError:
                    pass
                else:
                    self.port._increment("skill_card_terminal_note_title_recoveries")
                    return recovered_id
        try:
            return self.port.catalog.confirm_clicked_customizable_skill_card_without_badge_count(identity_text)
        except ArenaCatalogError:
            if expected_customization_count is not None and expected_customization_count > 0:
                return self.port.catalog.confirm_clicked_customizable_skill_card(
                    identity_text,
                    expected_count=expected_customization_count,
                )
            if exact_title_error is not None:
                raise exact_title_error
            assert candidate_error is not None
            raise candidate_error

    def _resolve_proven_skill_card_title(
        self,
        title_text: str,
        *,
        allow_one_character: bool = True,
    ) -> int:
        """Resolve a geometrically proven title without widening other aliases."""
        exact_resolver = getattr(self.port.catalog, "confirm_clicked_skill_card_by_exact_title", None)
        if exact_resolver is None:
            raise ArenaCatalogError("catalog has no exact title resolver")
        try:
            return exact_resolver(title_text)
        except ArenaCatalogError as exact_error:
            # A real display title (including duplicate owners) is never
            # reinterpreted through a lossy correction.
            normalizer = getattr(self.port.catalog, "normalize_skill_card_title_text", str)
            if normalizer(title_text) in getattr(self.port.catalog, "_cards_by_display_title", {}):
                raise
            note_resolver = getattr(
                self.port.catalog,
                "confirm_clicked_skill_card_by_terminal_note_title",
                None,
            )
            if note_resolver is not None:
                try:
                    return note_resolver(title_text)
                except ArenaCatalogError:
                    pass
            recovery = getattr(self.port.catalog, "recover_skill_card_title_one_character", None)
            if allow_one_character and recovery is not None:
                return recovery(title_text)
            raise exact_error

    def _same_proven_skill_card_title(self, first: str, second: str, card_id: int) -> bool:
        if first == second:
            return True
        exact = getattr(self.port.catalog, "confirm_clicked_skill_card_by_exact_title", None)
        if exact is not None:
            try:
                exact(first)
                exact(second)
            except ArenaCatalogError:
                pass
            else:
                # Distinct exact titles are an identity conflict, never a
                # correction (even if a caller supplies a faulty resolver).
                return False
        try:
            return self.port._resolve_proven_skill_card_title(first) == self.port._resolve_proven_skill_card_title(second) == card_id
        except ArenaCatalogError:
            return False

    def _normalized_skill_card_ocr_lines(self, text: str) -> tuple[str, ...]:
        normalizer = getattr(
            self.port.catalog,
            "normalize_skill_card_title_text",
            None,
        )
        return tuple(
            normalized
            for line in str(text or "").splitlines()
            if (normalized := (normalizer(line) if normalizer is not None else re.sub(r"\s+", "", line)))
        )

    def _trusted_skill_card_title_rows(
        self,
        image: Any,
        *,
        candidate_card_ids: Sequence[int] = (),
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> tuple[str, ...]:
        """Return authoritative catalog titles at the popover's dynamic title anchor."""

        candidate_ids = tuple(sorted(set(candidate_card_ids)))
        normalized_source_card_box = None if source_card_box is None else tuple(int(value) for value in source_card_box)
        if hasattr(self.port, "_runtime_counts"):
            self.port._increment("skill_card_trusted_title_row_requests")
        cache = getattr(self.port, "_trusted_skill_card_title_rows_cache", None)
        if cache is None:
            cache = {}
            self.port._trusted_skill_card_title_rows_cache = cache
        cache_key = (id(image), candidate_ids, normalized_source_card_box)
        cached = cache.get(cache_key)
        if cached is not None and cached.image is image:
            if hasattr(self.port, "_runtime_counts"):
                self.port._increment("skill_card_trusted_title_row_object_cache_hits")
            return cached.rows

        height, width = image.shape[:2]
        if width < 320 or height < 568:
            raise ArenaReaderError(
                "skill_card_detail_title_frame_invalid",
                f"capture {width}x{height} is too small for title evidence",
            )
        exact_resolver = getattr(
            self.port.catalog,
            "confirm_clicked_skill_card_by_exact_title",
            None,
        )
        if exact_resolver is None:
            return ()
        anchor_resolver = getattr(
            self.port.catalog,
            "skill_card_title_anchor_pattern",
            None,
        )

        def title_row_is_authoritative(line: str) -> bool:
            if hasattr(self.port, "_runtime_counts"):
                self.port._increment("skill_card_exact_title_index_queries")
            try:
                self.port._resolve_proven_skill_card_title(line, allow_one_character=False)
                return True
            except ArenaCatalogError:
                pass
            if not candidate_ids or anchor_resolver is None:
                return False
            matches = []
            for card_id in candidate_ids:
                try:
                    pattern = anchor_resolver(card_id)
                except ArenaCatalogError:
                    continue
                if re.fullmatch(pattern, line) is not None:
                    matches.append(card_id)
            return len(matches) == 1

        if hasattr(self.port, "_runtime_counts"):
            self.port._increment("skill_card_trusted_title_row_parse_calls")
        started = self.clock.perf_counter()
        try:
            rows: list[tuple[int, int, str, tuple[int, int, int, int]]] = []
            unmatched_rows: list[tuple[str, tuple[int, int, int, int]]] = []
            items = self.port._ocr(image, r".+")
            original_candidates = tuple((_text(item).strip(), _box(item)) for item in items if _text(item).strip())
            candidates = original_candidates + self.port._skill_card_title_ocr_segments(items)
            original_candidate_keys = set(original_candidates)
            seen_candidates: set[tuple[str, tuple[int, int, int, int]]] = set()
            for raw_text, box in candidates:
                candidate_key = (raw_text, box)
                if candidate_key in seen_candidates:
                    continue
                seen_candidates.add(candidate_key)
                normalized = self.port._normalized_skill_card_ocr_lines(raw_text)
                if len(normalized) != 1:
                    continue
                line = normalized[0]
                left, top, row_width, row_height = box
                in_legacy_title_band = top <= int(round(height * 0.36))
                in_lower_popover_title_band = False
                if normalized_source_card_box is not None:
                    _, card_top, _, card_height = normalized_source_card_box
                    card_bottom = card_top + card_height
                    # The adaptive panel can flip below a bottom-row card.  Do
                    # not widen the global title band: an effect row may itself
                    # be an exact catalog card name.  Bind this extra band to
                    # the clicked card and keep it inside the observed overlay
                    # envelope so the first effect row remains out of domain.
                    in_lower_popover_title_band = bool(
                        card_height > 0
                        and 0 <= card_top < card_bottom <= height
                        and card_bottom + int(round(card_height * 0.25)) <= top <= card_bottom + int(round(card_height * 0.75))
                        and top + row_height <= int(round(height * 0.48))
                    )
                if not (
                    0 <= left < width
                    and 0 <= top
                    and (in_legacy_title_band or in_lower_popover_title_band)
                    and row_width >= int(round(width * 0.04))
                    and int(round(height * 0.012)) <= row_height <= int(round(height * 0.05))
                ):
                    continue
                if not title_row_is_authoritative(line):
                    # Keep whole original atoms for a failure-only second
                    # pass. Do not repair a substring manufactured by split
                    # title processing, or run fuzzy lookup on normal frames.
                    if candidate_key in original_candidate_keys:
                        unmatched_rows.append((raw_text, box))
                    continue
                if not self.port._skill_card_title_row_has_neutral_ink(image, box):
                    continue
                rows.append((top, left, line, box))
            maximal_rows = tuple(
                row
                for row in rows
                if not any(
                    other is not row
                    and len(other[2]) > len(row[2])
                    and other[2].startswith(row[2])
                    and other[3][0] <= row[3][0]
                    and other[3][1] <= row[3][1]
                    and other[3][0] + other[3][2] >= row[3][0] + row[3][2]
                    and other[3][1] + other[3][3] >= row[3][1] + row[3][3]
                    for other in rows
                )
            )
            trusted_rows = tuple(line for _, _, line, _ in sorted(maximal_rows))
        finally:
            if hasattr(self.port, "_runtime_timing_seconds"):
                self.port._add_timing(
                    "skill_card_trusted_title_row_resolution",
                    self.clock.perf_counter() - started,
                )
        cache[cache_key] = _TrustedSkillCardTitleRowsEvidence(
            image,
            trusted_rows,
            tuple(unmatched_rows),
        )
        return trusted_rows

    def _authoritative_skill_card_title_text(
        self,
        group_index: int | None,
        detail_image: Any | None,
        *,
        candidate_card_ids: Sequence[int] = (),
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> str | None:
        """Return exactly one new, structurally proven detail-title row."""

        if group_index is None or detail_image is None:
            return None
        try:
            remaining_source = self.port._stable_skill_card_source_ocr_counts(group_index)
        except ArenaCatalogError:
            if hasattr(self.port, "_runtime_counts"):
                self.port._increment("skill_card_source_title_ocr_unavailable")
            return None
        try:
            detail_title_rows = self.port._trusted_skill_card_title_rows(
                detail_image,
                candidate_card_ids=candidate_card_ids,
                source_card_box=source_card_box,
            )
        except ArenaReaderError:
            self.port._increment("skill_card_detail_title_ocr_unavailable")
            return None
        new_rows: list[str] = []
        for line in detail_title_rows:
            remaining = remaining_source.get(line, 0)
            if remaining > 0:
                remaining_source[line] = remaining - 1
                continue
            new_rows.append(line)
        if len(new_rows) == 1:
            if source_card_box is not None and getattr(self.port, "_upgrade_title_roi_evidence", None):
                try:
                    base_id = self.port.catalog.confirm_clicked_skill_card_by_exact_title(new_rows[0])
                except ArenaCatalogError:
                    base_id = None
                cached = getattr(self.port, "_upgrade_title_roi_evidence", {}).get(
                    (id(detail_image), group_index, tuple(source_card_box), base_id),
                )
                if cached is not None and cached[0] is detail_image and cached[1] is not None:
                    return cached[1]
            return new_rows[0]
        if not new_rows and source_card_box is not None:
            recovered = self.port._recover_unmatched_skill_card_title(
                group_index,
                detail_image,
                candidate_card_ids,
                source_card_box,
            )
            if recovered is not None:
                return recovered
        if new_rows:
            self.port._increment("skill_card_detail_title_ambiguous")
        else:
            self.port._increment("skill_card_detail_title_missing")
        return None

    def _recover_unmatched_skill_card_title(
        self,
        group_index: int,
        image: Any,
        candidate_ids: Sequence[int],
        source_card_box: tuple[int, int, int, int],
    ) -> str | None:
        """Reuse whole native title atoms only after normal title selection fails."""
        cache_key = (id(image), tuple(sorted(set(candidate_ids))), tuple(source_card_box))
        cached = getattr(self.port, "_trusted_skill_card_title_rows_cache", {}).get(cache_key)
        if cached is None or cached.image is not image:
            return None
        recovery = getattr(self.port.catalog, "recover_skill_card_title_one_character", None)
        if recovery is None:
            return None
        source_counts = self.port._stable_skill_card_source_ocr_counts(group_index)
        candidates = []
        ambiguous = False
        for raw_text, box in cached.unmatched_rows:
            (line,) = self.port._normalized_skill_card_ocr_lines(raw_text)
            if source_counts.get(line, 0) or not self.port._skill_card_title_row_has_neutral_ink(image, box):
                continue
            self.port._increment("skill_card_title_one_character_queries")
            try:
                card_id = recovery(line)
            except ArenaCatalogError as error:
                ambiguous |= "skill_card_title_ambiguous:" in str(error)
                continue
            candidates.append((line, box, card_id, raw_text))
        if ambiguous or len(candidates) > 1:
            return None
        if not candidates:
            return self.port._recover_isolated_skill_card_title(group_index, image, source_card_box)
        line, box, card_id, raw_text = candidates[0]
        # This is frame-local title evidence, not an accepted card result.
        # Existing effect, slot-domain and fresh-frame checks still follow.
        evidence = getattr(self.port, "_skill_card_recovered_title_frames", None)
        if evidence is None:
            evidence = self.port._skill_card_recovered_title_frames = {}
        key = (id(image), card_id)
        if key not in evidence:
            self.port._increment("skill_card_title_one_character_recoveries")
            self.logger.debug(
                f"skill-card title one-character recovery: group={group_index} "
                f"source_box={source_card_box} title_box={box} observed={raw_text!r} "
                f"resolved={self.port.catalog.skill_card_title(card_id)!r} card_id={card_id}",
            )
        evidence[key] = (image, raw_text)
        while len(evidence) > 32:
            evidence.pop(next(iter(evidence)))
        return line

    def _isolated_skill_card_title_key(self, image: Any, group: int, box: Any) -> tuple:
        return (
            id(image),
            group,
            tuple(box),
            getattr(self.port, "_card_transaction_serial", 0),
            tuple(id(frame) for frame in getattr(self.port, "_card_count_frames", {}).get(group, ())),
        )

    def _isolated_skill_card_title_entry(self, image: Any, group: int, box: Any) -> Any:
        entry = getattr(self.port, "_isolated_title_evidence", {}).get(
            self.port._isolated_skill_card_title_key(image, group, box),
        )
        return entry if entry is not None and entry[0] is image else None

    def _recover_isolated_skill_card_title(self, group: int, image: Any, box: Any) -> str | None:
        """One crop on a proven new panel; never repair a background substring."""
        rows = getattr(self.port, "_card_rows", {}).get(group, ())
        slots = [index + 1 for index, row in enumerate(rows) if tuple(row) == tuple(box)]
        if len(slots) != 1 or getattr(self.port, "_card_transaction_contact_counts", {}).get((group, slots[0])) not in (1, 2):
            return None
        frames = getattr(self.port, "_card_count_frames", {}).get(group, ())
        if len(frames) != 3 or any(not hasattr(frame, "shape") for frame in frames):
            return None
        cached = self.port._isolated_skill_card_title_entry(image, group, box)
        if cached is not None:
            return cached[1]
        cache = getattr(self.port, "_isolated_title_evidence", None)
        if cache is None:
            cache = self.port._isolated_title_evidence = {}
        key = self.port._isolated_skill_card_title_key(image, group, box)
        cache[key] = (image, None, None, None, tuple(frames))
        while len(cache) > 16:
            cache.pop(next(iter(cache)))
        items = self.port._ocr(image, r".+")
        region = isolated_title_region(image, frames, tuple(box), [(_text(v), _box(v)) for v in items])
        if region is None:
            return None
        cache[key] = (image, None, None, region, tuple(frames))
        import cv2

        x, y, w, h = region
        crop = cv2.resize(image[y : y + h, x : x + w], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        self.port._increment("skill_card_isolated_title_roi_reads")
        cropped = self.port._with_full_frame_ocr_kind("derived_roi", self.port._ocr, crop, r".+")
        title_rows = self.port._spatial_ocr_rows(cropped)
        if len(title_rows) != 1:
            return None
        title, (tx, ty, tw, th), _ = title_rows[0]
        # Touching a crop boundary could hide a leading glyph or terminal '+'.
        if min(tx, ty, 2 * w - tx - tw, 2 * h - ty - th) < 3:
            return None
        mapped = (x + tx // 2, y + ty // 2, (tw + 1) // 2, (th + 1) // 2)
        if not self.port._skill_card_title_row_has_neutral_ink(image, mapped):
            return None
        if any(
            self.port._stable_skill_card_source_ocr_counts(group).get(line, 0) for line in self.port._normalized_skill_card_ocr_lines(title)
        ):
            return None
        try:
            # Resolve against the complete catalog before the existing scope
            # and identity guards. Top-K must not manufacture uniqueness.
            card_id = self.port._resolve_proven_skill_card_title(title)
        except ArenaCatalogError:
            return None
        item = {"text": title, "box": mapped}
        cache[key] = (image, title, item, region, tuple(frames))
        recovered = getattr(self.port, "_skill_card_recovered_title_frames", None)
        if recovered is None:
            recovered = self.port._skill_card_recovered_title_frames = {}
        recovered[(id(image), card_id)] = (image, title)
        while len(recovered) > 32:
            recovered.pop(next(iter(recovered)))
        self.port._increment("skill_card_isolated_title_roi_recoveries")
        return title

    def _skill_card_frame_title_pattern(self, image: Any, card_id: int) -> str:
        pattern = self.port.catalog.skill_card_title_anchor_pattern(card_id)
        recovered = getattr(self.port, "_skill_card_recovered_title_frames", {}).get((id(image), card_id))
        if recovered is not None and recovered[0] is image:
            # Quote only this frame's proven original row. Never add a global
            # wildcard to the catalog or rewrite the effect text.
            pattern = f"(?:{pattern}|{re.escape(recovered[1])})"
        return pattern

    def _confirm_skill_card_detail_identity(
        self,
        text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = None,
        detail_image: Any | None = None,
        source_card_box: tuple[int, int, int, int] | None = None,
        source_card_slot: int | None = None,
    ) -> int:
        """Resolve the already-open detail title under the existing bounded gates."""

        try:
            return self.port._confirm_clicked_skill_card_id(
                text,
                candidate_ids,
                expected_customization_count=expected_customization_count,
                source_group_index=source_group_index,
                detail_image=detail_image,
                source_card_box=source_card_box,
                source_card_slot=source_card_slot,
            )
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_detail_ambiguous",
                str(error),
            ) from error

    def _isolated_skill_card_title_atoms(self, image: Any, title_pattern: str) -> Any:
        for key, isolated in getattr(self.port, "_isolated_title_evidence", {}).items():
            if (
                isolated[0] is image
                and isolated[2] is not None
                and key == self.port._isolated_skill_card_title_key(image, key[1], key[2])
                and re.fullmatch(title_pattern, isolated[1]) is not None
            ):
                # Return the corrected geometry, never the original merged
                # title/background atom. Retain unchanged body observations.
                x, y, w, h = isolated[3]
                native_items = []
                for item in self.port._ocr(image, r".+"):
                    bx, by, bw, bh = _box(item)
                    if not (bx < x + w and bx + bw > x and by < y + h and by + bh > y):
                        native_items.append(item)
                return [isolated[2]], [*native_items, isolated[2]]
        return None


def skill_card_title_ocr_segments(
    items: Sequence[Any],
    *,
    spatial_ocr_rows: Callable[[Sequence[Any]], Sequence[Any]],
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    """Split full-screen OCR rows at cross-column horizontal gaps."""

    segments: list[tuple[str, tuple[int, int, int, int]]] = []
    for _, _, components in spatial_ocr_rows(items):
        current: list[tuple[str, tuple[int, int, int, int]]] = []

        def flush() -> None:
            if not current:
                return
            left = min(box[0] for _, box in current)
            top = min(box[1] for _, box in current)
            right = max(box[0] + box[2] for _, box in current)
            bottom = max(box[1] + box[3] for _, box in current)
            segments.append(
                (
                    "".join(text for text, _ in current),
                    (left, top, right - left, bottom - top),
                )
            )
            current.clear()

        for text, box in components:
            if current:
                previous_right = max(previous_box[0] + previous_box[2] for _, previous_box in current)
                previous_height = max(previous_box[3] for _, previous_box in current)
                gap = box[0] - previous_right
                maximum_inline_gap = max(
                    12,
                    int(round(max(previous_height, box[3]) * 1.5)),
                )
                if gap > maximum_inline_gap:
                    flush()
            current.append((text, box))
        flush()
    return tuple(segments)
