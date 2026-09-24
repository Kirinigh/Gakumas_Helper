"""Capture-bound OCR observations and title-anchor cache coordination.

The port is the sole owner of cache dictionaries. Calls retain the backend's
dynamic hooks, so replacing a hook or a cache view remains visible immediately.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, NamedTuple
from collections import Counter
from collections.abc import Sequence

from .catalog import ArenaCatalogError
from ._reader_visual import _text


class _FullFrameOcrEvidence(NamedTuple):
    """One immutable OCR observation bound to one frozen capture object."""

    image: Any
    kind: str
    hit: bool
    filtered_items: tuple[Any, ...]
    all_items: tuple[Any, ...]
    reco_id: int | None = None


class _TitleAnchorOcrEvidence(NamedTuple):
    """One title-specific fallback bound to the same frozen capture object."""

    image: Any
    matches: tuple[Any, ...]
    native_items: tuple[Any, ...]


class _TrustedSkillCardTitleRowsEvidence(NamedTuple):
    """One semantic title result bound to one frozen capture object."""

    image: Any
    rows: tuple[str, ...]
    unmatched_rows: tuple[tuple[str, tuple[int, int, int, int]], ...] = ()


class _EvidenceClock(Protocol):
    def perf_counter(self) -> float: ...


class DetailCaptureEvidencePort(Protocol):
    """Only the state views and operations used by this responsibility."""

    def _add_timing(self, name: str, elapsed: float) -> None: ...

    def _cached_full_frame_ocr_evidence(self, image: Any) -> _FullFrameOcrEvidence | None: ...

    _card_count_frames: dict[int, tuple[Any, Any, Any]]

    _card_source_ocr_counts: dict[int, dict[str, int]]

    _full_frame_ocr_evidence: dict[int, _FullFrameOcrEvidence]

    _full_frame_ocr_kind_hint: str | None

    def _increment(self, name: str) -> None: ...

    def _isolated_skill_card_title_atoms(self, image: Any, title_pattern: str) -> Any: ...

    def _observe_recognition_probe(self, image: Any, items: Any, *, reco_id: Any = None) -> None: ...

    def _ocr_detail(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = None, only_rec: bool = False) -> Any: ...

    def _record_full_frame_ocr_cache_hit(self, evidence: _FullFrameOcrEvidence) -> None: ...

    def _run_recognition(self, *args: Any, **kwargs: Any) -> Any: ...

    _runtime_counts: dict[str, int]

    _runtime_timing_seconds: dict[str, float]

    _title_anchor_ocr_evidence: dict[tuple[int, str], _TitleAnchorOcrEvidence]

    def _trusted_skill_card_title_rows(
        self, image: Any, *, candidate_card_ids: Sequence[int] = (), source_card_box: tuple[int, int, int, int] | None = None
    ) -> tuple[str, ...]: ...

    def _with_full_frame_ocr_kind(self, kind: str, operation: Any, *args: Any, **kwargs: Any) -> Any: ...


class DetailCaptureEvidence:
    def __init__(self, port: DetailCaptureEvidencePort, *, clock: _EvidenceClock) -> None:
        self.port = port
        self.clock = clock

    def _stable_skill_card_source_ocr_counts(
        self,
        group_index: int,
    ) -> dict[str, int]:
        cache = getattr(self.port, "_card_source_ocr_counts", None)
        if cache is None:
            cache = {}
            self.port._card_source_ocr_counts = cache
        cached = cache.get(group_index)
        if cached is not None:
            return dict(cached)
        frames = getattr(self.port, "_card_count_frames", {}).get(group_index, ())
        if len(frames) != 3:
            raise ArenaCatalogError(f"group {group_index} has no frozen three-frame OCR source")
        started = self.clock.perf_counter()
        try:
            frame_counts = tuple(
                Counter(
                    self.port._with_full_frame_ocr_kind(
                        "skill_card_source",
                        self.port._trusted_skill_card_title_rows,
                        frame,
                    )
                )
                for frame in frames
            )
        except Exception as error:
            raise ArenaCatalogError(f"group {group_index} source OCR could not be frozen") from error
        stable = {
            line: max(counts[line] for counts in frame_counts)
            for line in set().union(*(counts.keys() for counts in frame_counts))
            if max(counts[line] for counts in frame_counts) > 0
        }
        cache[group_index] = stable
        self.port._increment("skill_card_source_title_ocr_batches")
        self.port._add_timing(
            "skill_card_source_title_ocr",
            self.clock.perf_counter() - started,
        )
        return dict(stable)

    def _cached_full_frame_ocr_evidence(
        self,
        image: Any,
    ) -> _FullFrameOcrEvidence | None:
        cache = getattr(self.port, "_full_frame_ocr_evidence", None)
        if cache is None:
            return None
        evidence = cache.get(id(image))
        if evidence is None or evidence.image is not image:
            return None
        return evidence

    def _record_full_frame_ocr_cache_hit(
        self,
        evidence: _FullFrameOcrEvidence,
    ) -> None:
        if not hasattr(self.port, "_runtime_counts"):
            return
        self.port._increment("broad_ocr_cache_hits")
        self.port._increment(f"{evidence.kind}_broad_ocr_cache_hits")

    def _with_full_frame_ocr_kind(
        self,
        kind: str,
        operation: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Tag one OCR call without changing the callable's public signature."""

        had_previous = hasattr(self.port, "_full_frame_ocr_kind_hint")
        previous = getattr(self.port, "_full_frame_ocr_kind_hint", None)
        self.port._full_frame_ocr_kind_hint = kind
        try:
            return operation(*args, **kwargs)
        finally:
            if had_previous:
                self.port._full_frame_ocr_kind_hint = previous
            else:
                del self.port._full_frame_ocr_kind_hint

    def _full_frame_ocr_evidence_for(self, image: Any) -> _FullFrameOcrEvidence:
        """Recognize a frozen frame once while keeping fresh frames isolated."""

        cached = self.port._cached_full_frame_ocr_evidence(image)
        if cached is not None:
            self.port._record_full_frame_ocr_cache_hit(cached)
            return cached

        cache = getattr(self.port, "_full_frame_ocr_evidence", None)
        if cache is None:
            cache = {}
            self.port._full_frame_ocr_evidence = cache
        started = self.clock.perf_counter()
        detail = self.port._run_recognition(
            "ArenaReaderOCR",
            image,
            pipeline_override={
                "ArenaReaderOCR": {
                    "recognition": "OCR",
                    "expected": r".+",
                    "order_by": "Vertical",
                }
            },
        )
        hit = bool(detail and detail.hit)
        filtered_items = tuple(detail.filtered_results or detail.all_results or ()) if hit else ()
        all_items = tuple(detail.all_results or detail.filtered_results or ()) if hit else ()
        evidence = _FullFrameOcrEvidence(
            image,
            str(getattr(self.port, "_full_frame_ocr_kind_hint", "generic")),
            hit,
            filtered_items,
            all_items,
            getattr(detail, "reco_id", None),
        )
        cache[id(image)] = evidence
        self.port._observe_recognition_probe(image, all_items, reco_id=getattr(detail, "reco_id", None))
        # Retain enough member-local entries for both accepted source groups,
        # their detail confirmations and transformed ROI views without holding
        # captures beyond the member transaction.
        while len(cache) > 64:
            oldest_key = next(iter(cache))
            if oldest_key == id(image):
                break
            cache.pop(oldest_key, None)
        if hasattr(self.port, "_runtime_counts"):
            self.port._increment("broad_ocr_backend_calls")
            self.port._increment(f"{evidence.kind}_broad_ocr_backend_calls")
        if hasattr(self.port, "_runtime_timing_seconds"):
            self.port._add_timing(
                "broad_ocr_backend",
                self.clock.perf_counter() - started,
            )
        return evidence

    def _skill_card_title_anchor_evidence(
        self,
        image: Any,
        title_pattern: str,
    ) -> tuple[list[Any], list[Any]]:
        """Reuse raw boxes from the exact detail frame for title geometry.

        The broad full-frame OCR is the authoritative observation for the
        accepted capture.  A title-specific backend call remains a fail-safe
        fallback only when those raw boxes do not produce one unique anchor;
        this preserves the prior failure semantics for unusual OCR layouts.
        """

        isolated = self.port._isolated_skill_card_title_atoms(image, title_pattern)
        if isolated is not None:
            return isolated
        evidence = self.port._cached_full_frame_ocr_evidence(image)
        if evidence is not None and evidence.hit:
            native_items = list(evidence.all_items)
            matches = [item for item in native_items if re.fullmatch(title_pattern, _text(item).strip()) is not None]
            if len(matches) == 1:
                if hasattr(self.port, "_runtime_counts"):
                    self.port._record_full_frame_ocr_cache_hit(evidence)
                    self.port._increment("skill_card_detail_title_anchor_cache_hits")
                return matches, native_items
        cache = getattr(self.port, "_title_anchor_ocr_evidence", None)
        cache_key = (id(image), title_pattern)
        cached = None if cache is None else cache.get(cache_key)
        if cached is not None and cached.image is image:
            if hasattr(self.port, "_runtime_counts"):
                self.port._increment("skill_card_detail_title_anchor_cache_hits")
            return list(cached.matches), list(cached.native_items)

        if hasattr(self.port, "_runtime_counts"):
            self.port._increment("skill_card_detail_title_anchor_backend_fallbacks")
        detail = self.port._ocr_detail(image, title_pattern)
        matches = [] if not detail or not detail.hit else list(detail.filtered_results or detail.all_results or [])
        native_items = [] if not detail or not detail.hit else list(detail.all_results or detail.filtered_results or [])
        if cache is None:
            cache = {}
            self.port._title_anchor_ocr_evidence = cache
        cache[cache_key] = _TitleAnchorOcrEvidence(
            image,
            tuple(matches),
            tuple(native_items),
        )
        while len(cache) > 16:
            oldest_key = next(iter(cache))
            if oldest_key == cache_key:
                break
            cache.pop(oldest_key, None)
        return matches, native_items
