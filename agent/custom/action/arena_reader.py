"""Administrator-Maa backend for the read-only arena lineup reader.

All screenshots stay in memory.  The backend performs only ordinary navigation,
long-press, detail-card clicks, scrolling and back/close actions.  It deliberately
has no operation capable of starting a contest match.
"""

from __future__ import annotations

import re
import json
import time
import unicodedata
from typing import Any, Protocol, NamedTuple
from pathlib import Path
from threading import Lock
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from maa.context import Context
from maa.pipeline import JClick, JActionType
from arena_winrate import (
    TeamTarget,
    ArenaPageState,
    MemberSlotState,
    ArenaReaderError,
    ClickedSkillCard,
    ArenaCatalogError,
    MemberSlotMetrics,
    ArenaEntityCatalog,
    BadgeReferenceError,
    BadgeWorkerSelection,
    BadgeReferenceGallery,
    BadgeWorkerCalibrator,
    ContestSeasonDefinition,
    CustomizationBadgeState,
    GenericCostReferenceError,
    CustomizationBadgeDecision,
    GenericCostReferenceGallery,
    stage_member_cap,
    classify_arena_page,
    classify_member_slot,
    fixed_member_slot_boxes,
    team_stage_total_anchors,
    canonical_skill_card_rows,
    stable_generic_cost_value,
    infer_parameter_label_boxes,
    measure_customization_badge,
    arena_page_allows_team_entry,
    duplicate_marker_visual_features,
    p_item_screen_slot_to_engine_slot,
    p_item_screen_order_to_engine_order,
    stable_customization_badge_presence,
    is_duplicate_marker_visual_candidate,
    stable_excluded_duplicate_card_flags,
    stable_reference_business_candidates,
    expected_scrolled_secondary_skill_card_row,
    canonical_scrolled_secondary_skill_card_row,
)
from card_selection import EmbeddingCardRecognizer
from p_item_recognition import (
    PItemReferenceError,
    PItemReferenceDecision,
    PItemRenderedReferenceGallery,
    measure_p_item_content_generation,
)
from card_selection.model import frame_identifier, isolate_card_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _normalize_latin_anchor(value: str) -> str:
    """Normalize OCR-only Latin diacritics without weakening anchor identity."""

    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    ).strip().upper()


def _grade_ocr_value(value: str) -> int | None:
    """Map only proven single-glyph Grade OCR variants to their value."""

    normalized = _normalize_latin_anchor(value)
    if re.fullmatch(r"[1-7]", normalized) is not None:
        return int(normalized)
    # The stylized Grade-7 glyph is alternately emitted as 7 and V by the
    # same OCR model. This alias is consumed only inside the unique Grade
    # geometry below; it is never a global OCR replacement.
    if normalized == "V":
        return 7
    return None


def _resource_model_root(*parts: str) -> Path:
    """Resolve both source-tree and installed Maa resource layouts."""

    candidates = (
        PROJECT_ROOT / "assets" / "resource" / "base" / "model" / Path(*parts),
        PROJECT_ROOT / "resource" / "base" / "model" / Path(*parts),
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


CARD_MODEL_ROOT = _resource_model_root("classify", "card_embedding")
ARENA_CARD_MODEL_ROOT = _resource_model_root("embedding", "arena_card")
CARD_REFERENCE_ROOT = _resource_model_root("embedding", "arena_badge_reference")
CARD_COST_REFERENCE_ROOT = _resource_model_root(
    "embedding",
    "arena_card_cost_reference",
)
P_ITEM_REFERENCE_ROOT = _resource_model_root("embedding", "p_item_reference")
CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR = 0.75


class _BadgeShortlistResult(NamedTuple):
    """Complete output of the sole card-face badge shortlist route."""

    decision: CustomizationBadgeDecision
    observations: tuple[dict[str, Any], ...]


class PItemReader(Protocol):
    @property
    def content_generation_max_mean_abs_error(self) -> float: ...

    @property
    def maximum_detail_fallbacks_per_member(self) -> int: ...

    def content_generation_errors(
        self,
        images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
    ) -> Sequence[float]: ...

    def read(
        self,
        images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
        *,
        plan: str,
    ) -> Sequence[PItemReferenceDecision]: ...


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _box(item: Any) -> tuple[int, int, int, int]:
    value = _value(item, "box")
    if value is None or len(value) != 4:
        raise ArenaReaderError("recognition_box_missing", "recognition result has no four-value box")
    return tuple(int(round(number)) for number in value)


def _text(item: Any) -> str:
    return str(_value(item, "text", ""))


class Task085PItemReader:
    """Use the fixed rendered gallery with calibrated coarse/fine acceleration."""

    def __init__(self, gallery: PItemRenderedReferenceGallery) -> None:
        self.gallery = gallery

    @property
    def maximum_detail_fallbacks_per_member(self) -> int:
        return self.gallery.runtime.maximum_detail_fallbacks_per_member

    @property
    def content_generation_max_mean_abs_error(self) -> float:
        return self.gallery.runtime.content_generation_max_mean_abs_error

    def content_generation_errors(
        self,
        images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
    ) -> Sequence[float]:
        return measure_p_item_content_generation(images, boxes)

    @classmethod
    def from_model_root(
        cls,
        model_root: str | Path = P_ITEM_REFERENCE_ROOT,
    ) -> "Task085PItemReader":
        root = Path(model_root)
        try:
            gallery = PItemRenderedReferenceGallery.load(root)
        except (OSError, KeyError, TypeError, ValueError, PItemReferenceError) as error:
            raise ArenaReaderError(
                "p_item_runtime_not_approved",
                "the fixed P-item rendered-reference runtime is unavailable or invalid",
            ) from error
        return cls(gallery)

    def read(
        self,
        images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
        *,
        plan: str,
    ) -> Sequence[PItemReferenceDecision]:
        def classify(box: tuple[int, int, int, int]) -> PItemReferenceDecision:
            return self.gallery.classify(
                images,
                box,
                plan=plan,
            )

        with ThreadPoolExecutor(max_workers=min(2, len(boxes))) as executor:
            return tuple(executor.map(classify, boxes))


class ArenaReaderDiagnosticPort:
    """Explicit development-only access to low-level reader diagnostics.

    Production observations never use this port.  Keeping calibration access
    behind one named public object prevents probes from silently depending on
    backend-private cache layout.
    """

    def __init__(self, backend: "MaaArenaReaderBackend") -> None:
        self._backend = backend

    @property
    def grade(self) -> int | None:
        return self._backend._grade

    @property
    def member_boxes(self) -> Any:
        return self._backend._member_boxes

    @property
    def member_slot_observations(self) -> Any:
        return self._backend._member_slot_observations

    @property
    def card_rows(self) -> Any:
        return self._backend._card_rows

    @property
    def card_predictions(self) -> Any:
        return self._backend._card_predictions

    @property
    def card_candidate_groups(self) -> Any:
        return self._backend._card_candidate_groups

    @property
    def card_images(self) -> Any:
        return self._backend._card_images

    @property
    def card_count_frames(self) -> Any:
        return self._backend._card_count_frames

    @property
    def card_excluded_duplicate_flags(self) -> Any:
        return self._backend._card_excluded_duplicate_flags

    @property
    def card_empty_flags(self) -> Any:
        return self._backend._card_empty_flags

    @property
    def card_empty_diagnostics(self) -> Any:
        return self._backend._card_empty_diagnostics

    @property
    def card_duplicate_marker_diagnostics(self) -> Any:
        return self._backend._card_duplicate_marker_diagnostics

    @property
    def badge_count_diagnostics(self) -> Any:
        return self._backend._badge_count_diagnostics

    @property
    def badge_candidate_detail_checks(self) -> Any:
        return self._backend._badge_candidate_detail_checks

    @property
    def card_content_generation_diagnostics(self) -> Any:
        return self._backend._card_content_generation_diagnostics

    @property
    def inferred_clicked_cards(self) -> Any:
        return self._backend._inferred_clicked_cards

    def capture(self) -> Any:
        return self._backend._capture()

    def ocr(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._ocr(*args, **kwargs)

    def click(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._click(*args, **kwargs)

    def back(self) -> Any:
        return self._backend._back()

    def swipe(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._swipe(*args, **kwargs)

    def long_press(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._long_press(*args, **kwargs)

    def full_ocr_text(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._full_ocr_text(*args, **kwargs)

    def stage_total_anchors(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._stage_total_anchors(*args, **kwargs)

    def team_stage_total_anchors(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._team_stage_total_anchors(*args, **kwargs)

    def assert_member_detail(self) -> Any:
        return self._backend._assert_member_detail()

    def assert_card_group_visible(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._assert_card_group_visible(*args, **kwargs)

    def dismiss_skill_card_detail(self) -> Any:
        return self._backend._dismiss_skill_card_detail()

    def dismiss_known_blocking_overlay(self, image: Any) -> bool:
        return self._backend._dismiss_known_blocking_overlay(image)

    def detect_card_rows(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._detect_card_rows(*args, **kwargs)

    def raw_card_candidate_boxes(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._raw_card_candidate_boxes(*args, **kwargs)

    def targeted_duplicate_marker_observations(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        return self._backend._targeted_duplicate_marker_observations(
            *args, **kwargs
        )

    def card_recognizer(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._card_recognizer(*args, **kwargs)


class MaaArenaReaderBackend:
    """Arena screen backend intended to run inside the elevated Maa agent."""

    _member_long_press_seconds = 0.7
    _source_restore_poll_seconds = 0.08

    def __init__(
        self,
        context: Context,
        season: ContestSeasonDefinition,
        bundle_dir: str | Path,
        *,
        p_item_reader: PItemReader | None = None,
        card_swipe_duration_ms: int = 100,
    ) -> None:
        if card_swipe_duration_ms not in (50, 100, 150, 200):
            raise ValueError("card_swipe_duration_ms must be 50, 100, 150, or 200")
        self.context = context
        self.season = season
        self.catalog = ArenaEntityCatalog.from_bundle(bundle_dir)
        self.p_item_reader = p_item_reader
        self._card_swipe_duration_ms = card_swipe_duration_ms
        self._grade: int | None = None
        self._member_boxes: dict[tuple[str, int], dict[int, tuple[int, int, int, int]]] = {}
        self._member_slot_observations: dict[
            tuple[str, int],
            dict[int, tuple[MemberSlotState, MemberSlotMetrics, tuple[int, int, int, int]]],
        ] = {}
        self._card_rows: dict[int, tuple[tuple[int, int, int, int], ...]] = {}
        self._card_predictions: dict[tuple[int, int], int] = {}
        self._card_candidate_groups: dict[tuple[int, int], tuple[int, ...]] = {}
        self._card_images: dict[int, Any] = {}
        self._card_count_frames: dict[int, tuple[Any, Any, Any]] = {}
        self._card_identity_frames: dict[
            int, tuple[tuple[dict[str, Any], ...], ...]
        ] = {}
        self._card_restoration_signatures: dict[
            int,
            tuple[tuple[tuple[str, Any], ...], ...],
        ] = {}
        self._card_excluded_duplicate_flags: dict[int, tuple[bool, ...]] = {}
        self._card_empty_flags: dict[int, tuple[bool, ...]] = {}
        self._card_empty_diagnostics: dict[int, tuple[dict[str, Any], ...]] = {}
        self._card_duplicate_marker_diagnostics: dict[int, tuple[dict[str, Any], ...]] = {}
        self._badge_count_diagnostics: dict[int, tuple[dict[str, Any], ...]] = {}
        self._detail_count_overrides: dict[tuple[int, int], dict[str, Any]] = {}
        self._detail_semantic_confirmations: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        self._badge_local_results: dict[int, tuple[Any, ...]] = {}
        self._badge_glyph_observations: dict[
            tuple[int, int], tuple[dict[str, Any], ...]
        ] = {}
        self._badge_glyph_count_diagnostics: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        self._inferred_clicked_cards: dict[tuple[int, int], ClickedSkillCard] = {}
        self._active_inferred_clicked_card: tuple[int, int] | None = None
        self._card_detail_texts: dict[tuple[int, int], str] = {}
        self._card_detail_images: dict[tuple[int, int], Any] = {}
        self._p_item_diagnostics: tuple[dict[str, Any], ...] = ()
        self._p_item_generation_evidence: dict[str, Any] = {}
        self._p_item_source_frames: tuple[Any, Any, Any] | None = None
        self._card_recognizers: dict[bool, tuple[EmbeddingCardRecognizer, float, float]] = {}
        self._card_recognizer_lock = Lock()
        self._card_reference_gallery: BadgeReferenceGallery | None = None
        self._card_cost_reference_gallery: GenericCostReferenceGallery | None = None
        self._zero_card_identity_fusion_diagnostics: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        self._zero_card_embedding_detail_candidates: dict[
            tuple[int, int], tuple[int, ...]
        ] = {}
        self._zero_card_detail_candidates: dict[tuple[int, int], tuple[int, ...]] = {}
        self._badge_worker_calibrator = BadgeWorkerCalibrator()
        self._badge_worker_selection: BadgeWorkerSelection | None = None
        self._runtime_timing_seconds: dict[str, float] = {}
        self._runtime_counts: dict[str, int] = {}
        self._runtime_duration_samples: dict[str, list[float]] = {}
        self._card_transaction_started: dict[tuple[int, int], float] = {}
        self._card_transaction_kinds: dict[tuple[int, int], str] = {}
        self._badge_candidate_detail_checks: list[dict[str, Any]] = []
        self._secondary_fixed_slot_fallback_enabled = False
        self._secondary_presence_diagnostics: tuple[dict[str, Any], ...] = ()
        self._card_content_generation_diagnostics: tuple[dict[str, Any], ...] = ()
        self._card_face_cost_diagnostics: dict[tuple[int, int], dict[str, Any]] = {}
        self._card_face_cost_optional_errors: dict[tuple[int, int], str] = {}
        self._secondary_presence_cache: dict[
            int,
            tuple[Any, tuple[dict[str, Any], ...]],
        ] = {}
        self._member_metric_baseline: tuple[
            dict[str, float],
            dict[str, int],
            dict[str, int],
        ] | None = None
        self._challenge_selection_committed = False

    def _add_timing(self, name: str, elapsed: float) -> None:
        self._runtime_timing_seconds[name] = (
            self._runtime_timing_seconds.get(name, 0.0) + elapsed
        )

    def _increment(self, name: str) -> None:
        self._runtime_counts[name] = self._runtime_counts.get(name, 0) + 1

    def _record_duration_sample(self, name: str, elapsed: float) -> None:
        self._runtime_duration_samples.setdefault(name, []).append(elapsed)

    @staticmethod
    def _percentile(values: Sequence[float], quantile: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    def _finish_card_transaction(self, key: tuple[int, int]) -> None:
        started = self._card_transaction_started.pop(key, None)
        kind = self._card_transaction_kinds.pop(
            key,
            "necessary_skill_card_detail_transaction",
        )
        if started is None:
            return
        elapsed = time.perf_counter() - started
        self._record_duration_sample("skill_card_detail_transaction", elapsed)
        self._record_duration_sample(kind, elapsed)

    def runtime_metrics(self) -> dict[str, Any]:
        selection = self._badge_worker_selection
        sample_summaries = {
            name: {
                "count": len(values),
                "p50": round(self._percentile(values, 0.50), 6),
                "p95": round(self._percentile(values, 0.95), 6),
                "max": round(max(values), 6),
            }
            for name, values in sorted(self._runtime_duration_samples.items())
            if values
        }
        return {
            "timing_seconds": {
                key: round(value, 6)
                for key, value in sorted(self._runtime_timing_seconds.items())
            },
            "counts": dict(sorted(self._runtime_counts.items())),
            "duration_percentiles_seconds": sample_summaries,
            "badge_workers": None if selection is None else selection.workers,
            "badge_worker_cache_hit": None if selection is None else selection.cache_hit,
            "badge_worker_benchmarks": (
                [] if selection is None else list(selection.benchmarks)
            ),
            "card_swipe_duration_ms": self._card_swipe_duration_ms,
            "member_long_press_duration_ms": int(
                round(self._member_long_press_seconds * 1000)
            ),
            "point_click_dispatch": "controller_point_context_box",
        }

    def diagnostic_port(self) -> ArenaReaderDiagnosticPort:
        """Return the explicit non-production calibration interface."""

        return ArenaReaderDiagnosticPort(self)

    def member_observation_evidence(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
    ) -> dict[str, Any]:
        """Return only low-dimensional evidence from the active member page."""

        del target
        if self._member_metric_baseline is None:
            raise ArenaReaderError(
                "member_observation_generation_missing",
                f"stage-{stage_number}/member-{member_slot} has no active metric generation",
            )
        timing_before, counts_before, sample_lengths = self._member_metric_baseline
        timing_delta = {
            key: round(value - timing_before.get(key, 0.0), 6)
            for key, value in self._runtime_timing_seconds.items()
            if value - timing_before.get(key, 0.0) > 0
        }
        counts_delta = {
            key: value - counts_before.get(key, 0)
            for key, value in self._runtime_counts.items()
            if value - counts_before.get(key, 0) > 0
        }
        duration_samples = {
            key: [round(value, 6) for value in values[sample_lengths.get(key, 0) :]]
            for key, values in self._runtime_duration_samples.items()
            if values[sample_lengths.get(key, 0) :]
        }
        groups = []
        for group_index in (0, 1):
            groups.append(
                {
                    "group_index": group_index,
                    "badge_count_diagnostics": [
                        dict(value)
                        for value in self._badge_count_diagnostics.get(group_index, ())
                    ],
                    "detail_count_overrides": [
                        dict(value)
                        for (detail_group, _), value in sorted(
                            self._detail_count_overrides.items()
                        )
                        if detail_group == group_index
                    ],
                    "detail_semantic_confirmations": [
                        dict(value)
                        for (detail_group, _), value in sorted(
                            self._detail_semantic_confirmations.items()
                        )
                        if detail_group == group_index
                    ],
                    "auxiliary_badge_glyph_counts": [
                        dict(value)
                        for (glyph_group, _), value in sorted(
                            self._badge_glyph_count_diagnostics.items()
                        )
                        if glyph_group == group_index
                    ],
                    "card_face_cost_evidence": [
                        dict(value)
                        for (cost_group, _), value in sorted(
                            self._card_face_cost_diagnostics.items()
                        )
                        if cost_group == group_index
                    ],
                    "zero_card_identity_fusions": [
                        dict(value)
                        for (fusion_group, _), value in sorted(
                            self._zero_card_identity_fusion_diagnostics.items()
                        )
                        if fusion_group == group_index
                    ],
                    "zero_card_detail_candidates": [
                        {
                            "slot": slot,
                            "candidate_ids": list(candidate_ids),
                        }
                        for (detail_group, slot), candidate_ids in sorted(
                            self._zero_card_detail_candidates.items()
                        )
                        if detail_group == group_index
                    ],
                    "excluded_duplicate_flags": list(
                        self._card_excluded_duplicate_flags.get(group_index, ())
                    ),
                    "empty_flags": list(
                        getattr(self, "_card_empty_flags", {}).get(group_index, ())
                    ),
                    "empty_slot_diagnostics": [
                        dict(value)
                        for value in getattr(
                            self,
                            "_card_empty_diagnostics",
                            {},
                        ).get(group_index, ())
                    ],
                    "resolved_card_ids": [
                        self._card_predictions.get((group_index, slot), 0)
                        for slot in range(1, 7)
                    ],
                    "candidate_business_ids": [
                        list(self._card_candidate_groups.get((group_index, slot), ()))
                        for slot in range(1, 7)
                    ],
                }
            )
        return {
            "stage_number": stage_number,
            "member_slot": member_slot,
            "p_items": [dict(value) for value in self._p_item_diagnostics],
            "p_item_generation": dict(self._p_item_generation_evidence),
            "groups": groups,
            "secondary_presence_diagnostics": [
                dict(value) for value in self._secondary_presence_diagnostics
            ],
            "card_content_generation_diagnostics": [
                dict(value) for value in self._card_content_generation_diagnostics
            ],
            "timing_seconds": timing_delta,
            "counts": counts_delta,
            "duration_samples_seconds": duration_samples,
            "badge_candidate_detail_checks": [
                dict(value)
                for value in self._badge_candidate_detail_checks
                if value.get("stage_number") == stage_number
                and value.get("member_slot") == member_slot
            ],
            "screenshots_persisted": False,
        }

    def _card_references(self) -> BadgeReferenceGallery:
        if self._card_reference_gallery is not None:
            return self._card_reference_gallery
        try:
            self._card_reference_gallery = BadgeReferenceGallery.load(CARD_REFERENCE_ROOT)
        except (OSError, KeyError, TypeError, ValueError, BadgeReferenceError) as error:
            raise ArenaReaderError(
                "skill_card_reference_unavailable",
                "compact clean-card reference gallery is unavailable or invalid",
            ) from error
        return self._card_reference_gallery

    def _card_cost_references(self) -> GenericCostReferenceGallery:
        if self._card_cost_reference_gallery is not None:
            return self._card_cost_reference_gallery
        try:
            self._card_cost_reference_gallery = GenericCostReferenceGallery.load(
                CARD_COST_REFERENCE_ROOT
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            GenericCostReferenceError,
        ) as error:
            raise ArenaReaderError(
                "skill_card_cost_reference_unavailable",
                "compact card-face cost gallery is unavailable or invalid",
            ) from error
        return self._card_cost_reference_gallery

    @staticmethod
    def _parallel_badge_jobs(
        workers: int,
        jobs: Sequence[tuple[int, int]],
        operation: Any,
    ) -> tuple[Any, ...]:
        if workers <= 1:
            return tuple(operation(job) for job in jobs)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return tuple(executor.map(operation, jobs))

    def _local_badge_slot_decision(
        self,
        frames: Sequence[Any],
        frame_rows: Sequence[Sequence[tuple[int, int, int, int]]],
        group_index: int,
        slot_index: int,
    ) -> _BadgeShortlistResult:
        repeated = tuple(
            measure_customization_badge(image, frame_row[slot_index])
            for image, frame_row in zip(frames, frame_rows, strict=True)
        )
        presence = stable_customization_badge_presence(repeated)
        return _BadgeShortlistResult(
            decision=presence,
            observations=repeated,
        )

    def _badge_workers(
        self,
        frames: Sequence[Any],
        jobs: Sequence[tuple[int, int]],
        operation: Any,
    ) -> int:
        if self._badge_worker_selection is not None:
            return self._badge_worker_selection.workers
        signature = self._badge_worker_calibrator.signature(
            frame_shape=frames[0].shape,
            batch_slots=len(jobs),
            reference_version="seeded-plate-shortlist-v1",
            recognition_asset_signature="seeded-plate-detail-authority-v1",
        )

        def run_batch(workers: int) -> None:
            self._parallel_badge_jobs(workers, jobs, operation)

        started = time.perf_counter()
        self._badge_worker_selection = self._badge_worker_calibrator.select(
            signature=signature,
            run_batch=run_batch,
        )
        self._add_timing("badge_worker_selection", time.perf_counter() - started)
        return self._badge_worker_selection.workers

    @staticmethod
    def _badge_detail_inferable(decision: CustomizationBadgeDecision) -> bool:
        """Only the typed shortlist state may authorize a detail."""

        return decision.state is CustomizationBadgeState.DETAIL_CANDIDATE

    def _record_badge_candidate_detail(
        self,
        *,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        phase: str,
        reason: str,
    ) -> dict[str, Any]:
        """Record every authorized shortlist click before touching the UI."""

        entry = {
            "stage_number": int(stage_number),
            "member_slot": int(member_slot),
            "group_index": int(group_index),
            "card_slot": int(card_slot),
            "phase": phase,
            "reason": reason,
            "detail_click_allowed": True,
            "resolved_state": None,
        }
        self._badge_candidate_detail_checks.append(entry)
        return entry

    def _finish_badge_candidate_detail(
        self,
        entry: dict[str, Any],
        result: ClickedSkillCard,
    ) -> None:
        count = sum(int(value) for value in result.customizations.values())
        entry.update(
            {
                "resolved_state": "POSITIVE" if count else "CONFIDENT_ZERO",
                "resolved_card_id": result.card_id,
                "resolved_customization_count": count,
            }
        )
        self._increment(
            "badge_candidate_positive_details"
            if count
            else "badge_candidate_zero_details"
        )

    def _batched_badge_local_results(
        self,
        requested_group: int,
    ) -> tuple[Any, ...]:
        cached = self._badge_local_results.get(requested_group)
        if cached is not None:
            return cached
        visible_groups = tuple(
            index
            for index in (0, 1)
            if len(self._card_rows.get(index, ())) == 6
            and len(self._card_count_frames.get(index, ())) == 3
        )
        if requested_group not in visible_groups:
            raise ArenaReaderError(
                "skill_card_badge_batch_missing",
                f"group {requested_group} has no stable three-frame card observation",
            )
        frame_rows = {
            group_index: tuple(
                self._validated_card_group_row(image, group_index)
                for image in self._card_count_frames[group_index]
            )
            for group_index in visible_groups
        }
        duplicate_flags = {
            group_index: self._card_excluded_duplicate_flags.get(
                group_index,
                (False,) * 6,
            )
            for group_index in visible_groups
        }
        empty_flags = {
            group_index: getattr(self, "_card_empty_flags", {}).get(
                group_index,
                (False,) * 6,
            )
            for group_index in visible_groups
        }
        jobs = tuple(
            (group_index, slot_index)
            for group_index in visible_groups
            for slot_index in range(6)
        )

        def operation(job: tuple[int, int]) -> Any:
            group_index, slot_index = job
            if (
                duplicate_flags[group_index][slot_index]
                or empty_flags[group_index][slot_index]
            ):
                return None
            return self._local_badge_slot_decision(
                self._card_count_frames[group_index],
                frame_rows[group_index],
                group_index,
                slot_index,
            )

        workers = self._badge_workers(
            self._card_count_frames[requested_group],
            jobs,
            operation,
        )
        started = time.perf_counter()
        results = self._parallel_badge_jobs(workers, jobs, operation)
        self._add_timing("badge_local_batch", time.perf_counter() - started)
        grouped: dict[int, list[Any]] = {
            group_index: [None] * 6 for group_index in visible_groups
        }
        for (group_index, slot_index), result in zip(jobs, results, strict=True):
            grouped[group_index][slot_index] = result
        for group_index, values in grouped.items():
            self._badge_local_results[group_index] = tuple(values)
            for slot_index, result in enumerate(values, start=1):
                if result is not None:
                    self._badge_glyph_observations[(group_index, slot_index)] = (
                        result.observations
                    )
        return self._badge_local_results[requested_group]

    @staticmethod
    def _badge_glyph_ocr_canvases(descriptor: Any) -> tuple[Any, ...]:
        """Render one centre-plate glyph mask for generic OCR.

        The input is the quantized mask cut from the sole centre-seeded green
        component. No card pixels, frame, green artwork, template asset, or
        card identity participates in these canvases.
        """

        import cv2
        import numpy as np

        values = np.asarray(descriptor)
        if (
            values.shape != (216,)
            or values.dtype.kind not in "iu"
            or bool(np.any(values < 0))
            or bool(np.any(values > 15))
        ):
            raise ArenaReaderError(
                "skill_card_badge_glyph_invalid",
                "badge glyph descriptor must contain 216 integers in [0, 15]",
            )
        mask = values.astype(np.uint8).reshape(18, 12) * 17
        active = mask > 0
        ys, xs = np.nonzero(active)
        if not len(xs):
            raise ArenaReaderError(
                "skill_card_badge_glyph_invalid",
                "badge glyph descriptor has no foreground",
            )
        crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        canvases: list[Any] = []
        for interpolation in (cv2.INTER_NEAREST, cv2.INTER_CUBIC):
            enlarged = cv2.resize(
                crop,
                None,
                fx=8.0,
                fy=8.0,
                interpolation=interpolation,
            )
            for invert in (False, True):
                foreground = 255 - enlarged if invert else enlarged
                background = 255 if invert else 0
                canvas = np.full(
                    (foreground.shape[0] + 32, foreground.shape[1] + 32),
                    background,
                    dtype=np.uint8,
                )
                canvas[16:-16, 16:-16] = foreground
                canvases.append(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
        return tuple(canvases)

    def _auxiliary_badge_glyph_count(self, key: tuple[int, int]) -> int:
        """Use one stable count only after detail semantics remain non-unique.

        This never classifies badge presence. All three source frames must
        already be detail candidates, expose exactly one co-located glyph from
        the same green-plate interior, and agree through generic OCR.
        """

        cached = self._badge_glyph_count_diagnostics.get(key)
        if cached is not None:
            count = cached.get("resolved_count")
            if type(count) is int and 0 <= count <= 3:
                return count
            raise ArenaReaderError(
                "skill_card_badge_glyph_ambiguous",
                "cached badge glyph evidence is not resolvable for "
                f"{key!r}; diagnostic={cached!r}",
            )

        observations = self._badge_glyph_observations.get(key, ())
        if len(observations) != 3:
            raise ArenaReaderError(
                "skill_card_badge_glyph_frames_missing",
                f"{key!r} has no stable three-frame badge glyph evidence",
            )
        started = time.perf_counter()
        frame_digits: list[int] = []
        descriptor_digests: list[str] = []
        variant_texts: list[list[str]] = []
        try:
            import hashlib

            for observation in observations:
                internal = observation.get("internal_features", {})
                descriptor = internal.get("glyph_descriptor_12x18_q4")
                component_count = internal.get("glyph_component_count")
                if (
                    component_count == 0
                    and descriptor is None
                    and internal.get("glyph_non_badge_art_candidate") is True
                ):
                    frame_digits.append(0)
                    descriptor_digests.append("NON_BADGE_ART")
                    variant_texts.append([])
                    continue
                if (
                    type(component_count) is not int
                    or component_count < 1
                    or descriptor is None
                ):
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_ambiguous",
                        f"{key!r} lacks one bounded co-located glyph state in every source frame",
                    )
                canvases = self._badge_glyph_ocr_canvases(descriptor)
                descriptor_digests.append(
                    hashlib.sha256(bytes(int(value) for value in descriptor)).hexdigest()
                )
                texts: list[str] = []
                digits: set[int] = set()
                for canvas in canvases:
                    for item in self._ocr(canvas, r"^[1-3]$"):
                        value = _text(item).strip()
                        texts.append(value)
                        if value in {"1", "2", "3"}:
                            digits.add(int(value))
                variant_texts.append(texts)
                if len(digits) != 1:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_ambiguous",
                        f"{key!r} generic OCR did not produce one frame-local digit",
                    )
                frame_digits.append(next(iter(digits)))
            if len(set(frame_digits)) != 1:
                raise ArenaReaderError(
                    "skill_card_badge_glyph_frame_disagreement",
                    f"{key!r} badge glyph OCR disagreed across source frames: {frame_digits!r}",
                )
            count = frame_digits[0]
            self._badge_glyph_count_diagnostics[key] = {
                "group_index": key[0],
                "slot": key[1],
                "resolved_count": count,
                "frame_digits": list(frame_digits),
                "descriptor_sha256": descriptor_digests,
                "ocr_texts": variant_texts,
                "mode": (
                    "detail_ambiguity_same_plate_non_badge_art_zero"
                    if count == 0
                    else "detail_ambiguity_auxiliary_same_plate_glyph_ocr"
                ),
            }
            self._increment("skill_card_badge_auxiliary_glyph_resolutions")
            if count > 0:
                self._increment("skill_card_badge_auxiliary_glyph_ocr")
            return count
        except ArenaReaderError as error:
            self._badge_glyph_count_diagnostics[key] = {
                "group_index": key[0],
                "slot": key[1],
                "resolved_count": None,
                "frame_digits": list(frame_digits),
                "descriptor_sha256": descriptor_digests,
                "ocr_texts": variant_texts,
                "mode": "fail_closed",
                "error_code": error.code,
                "error_detail": error.detail,
            }
            raise
        finally:
            self._add_timing(
                "skill_card_badge_auxiliary_glyph_ocr",
                time.perf_counter() - started,
            )

    def _card_recognizer(self, *, customized: bool) -> tuple[EmbeddingCardRecognizer, float, float]:
        """Load a visual-family candidate source without accepting its business ID.

        Neither embedding becomes a business ID directly.  Positive cards are
        resolved by their mandatory detail page; zero cards require independent
        agreement with the fixed clean-reference gallery.  The intentionally
        null production thresholds therefore remain a ban on accepting a
        prediction ID while a permissive score supplies bounded candidates.
        """

        with self._card_recognizer_lock:
            cached = self._card_recognizers.get(customized)
            if cached is not None:
                return cached
            root = ARENA_CARD_MODEL_ROOT if customized else CARD_MODEL_ROOT
            try:
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
                recognizer = EmbeddingCardRecognizer.load(root)
                runtime = manifest["runtime"]
                if "acceptance_threshold" not in runtime or "minimum_margin" not in runtime:
                    raise KeyError("runtime candidate-source metadata")
                acceptance_threshold = 0.000001
                minimum_margin = 0.0
            except (OSError, KeyError, TypeError, ValueError, ImportError) as error:
                model_label = "arena custom-card" if customized else "base card"
                raise ArenaReaderError(
                    "skill_card_runtime_not_approved",
                    f"{model_label} embedding candidate assets are unavailable",
                ) from error
            loaded = (recognizer, acceptance_threshold, minimum_margin)
            self._card_recognizers[customized] = loaded
            return loaded

    def _card_visual_family_candidates(
        self,
        *,
        stage_number: int,
        image: Any,
        box: tuple[int, int, int, int],
        slot_index: int,
    ) -> tuple[int, ...]:
        """Union base-card and arena custom-card Top-K families before a detail click.

        A seeded green plate is intentionally only a high-recall shortlist, so
        it may come from either a real customized overlay or clean card art.
        The two embeddings only bound title verification; neither one decides
        badge presence or supplies the final business ID.
        """

        from card_selection.types import CandidateBox

        plan = self.season.stages[stage_number - 1].plan
        candidate_box = CandidateBox(
            slot=slot_index,
            box=box,
            detector_label="cards",
            detector_score=1.0,
        )
        candidates: list[int] = []
        for customized in (True, False):
            recognizer, acceptance_threshold, minimum_margin = self._card_recognizer(
                customized=customized
            )
            prediction = recognizer.classify(
                image,
                candidate_box,
                frame_id=frame_identifier(image, slot_index + (100 if customized else 200)),
                plan=plan,
                acceptance_threshold=acceptance_threshold,
                min_margin=minimum_margin,
                resolve_upgrade_state=False,
            )
            if not prediction.accepted:
                continue
            if prediction.card_id.isdigit():
                candidates.append(int(prediction.card_id))
            candidates.extend(
                int(value)
                for visual_family in prediction.top_k_card_ids
                for value in visual_family
            )
        ordered = tuple(dict.fromkeys(candidates))
        if not ordered:
            raise ArenaReaderError(
                "skill_card_badge_detail_identity_unknown",
                "neither card embedding supplied a visual family for the "
                "seeded-plate detail candidate",
            )
        return ordered

    def ensure_arena_main(self, *, require_opponents: bool = True) -> None:
        state = ArenaPageState.AMBIGUOUS
        counts = (0, 0, 0, 0, 0)
        grade_observations: tuple[Any, ...] = ()
        grade: int | None = None
        grade_error: ArenaReaderError | None = None
        unavailable_reads = 0
        ambiguous_reads = 0
        for attempt in range(4):
            image = self._capture()
            # Capture the Grade label and digit in one OCR generation before
            # the page-state recognizers reuse the shared OCR entry with
            # narrower filters. This avoids cross-override cache pollution and
            # removes a second independent Grade OCR decision.
            grade_observations = tuple(self._ocr(image, r".+"))
            state, counts = self._arena_page_state(image)
            if arena_page_allows_team_entry(
                state,
                require_opponents=require_opponents,
            ):
                try:
                    grade = self._read_grade(
                        image,
                        observations=grade_observations,
                    )
                except ArenaReaderError as error:
                    if error.code not in {
                        "arena_grade_label_ambiguous",
                        "arena_grade_ambiguous",
                    }:
                        raise
                    grade_error = error
                    if attempt < 3:
                        time.sleep(0.25)
                        continue
                    raise
                break
            if state is ArenaPageState.AMBIGUOUS:
                ambiguous_reads += 1
                if ambiguous_reads < 2:
                    time.sleep(0.5)
                    continue
                break
            if state is ArenaPageState.OPPONENTS_UNAVAILABLE:
                unavailable_reads += 1
                if unavailable_reads < 3:
                    time.sleep(0.25)
                    continue
            break
        (
            rehearsal_count,
            opponent_count,
            stage_label_count,
            stage_total_count,
            exhausted_notice_count,
        ) = counts
        if (
            state is ArenaPageState.OPPONENTS_UNAVAILABLE
            and require_opponents
            and unavailable_reads == 3
        ):
            raise ArenaReaderError(
                "arena_opponents_unavailable",
                "opponent cards are absent on three stable arena-main reads; today's five contest attempts are exhausted",
            )
        if state is ArenaPageState.TEAM_PREVIEW:
            raise ArenaReaderError(
                "arena_team_preview_visible",
                "rehearsal/team preview is still open and must be left before reading opponents",
            )
        if not arena_page_allows_team_entry(
            state,
            require_opponents=require_opponents,
        ):
            raise ArenaReaderError(
                "arena_main_ambiguous",
                "arena page anchors are inconsistent: "
                f"rehearsal={rehearsal_count}, opponents={opponent_count}, "
                f"stage_labels={stage_label_count}, stage_totals={stage_total_count}, "
                f"exhausted_notices={exhausted_notice_count}",
            )
        if grade is None:
            if grade_error is not None:
                raise grade_error
            raise ArenaReaderError(
                "arena_grade_ambiguous",
                "arena main became available without a same-frame Grade decision",
            )
        if self._grade is not None and self._grade != grade:
            raise ArenaReaderError(
                "arena_grade_changed",
                f"contest Grade changed during one read: {self._grade} -> {grade}",
            )
        self._grade = grade

    def enter_team(self, target: TeamTarget) -> None:
        self._badge_candidate_detail_checks.clear()
        image = self._capture()
        if target.is_own_team:
            matches = self._ocr(image, r"^リハーサル$")
            if len(matches) != 1:
                raise ArenaReaderError("rehearsal_anchor_ambiguous", f"found {len(matches)} rehearsal anchors")
            self._click(self._box_center_point(_box(matches[0])))
        else:
            opponents = self._recognize("ArenaReaderOpponentCards", image)
            if target.opponent_position is None or len(opponents) != 3:
                raise ArenaReaderError("opponent_count_mismatch", "three ordered opponent cards are required")
            ordered = sorted(opponents, key=lambda item: (_box(item)[1], _box(item)[0]))
            self._click(_box(ordered[target.opponent_position]))
        self._wait_for_stage_member_list((3,) if target.is_own_team else (6,))

    def select_opponent_for_challenge(self, position: int) -> dict[str, Any]:
        """Perform one lightweight same-session guard, then open the target.

        The complete lineup has already been read and simulated.  This guard
        intentionally does not repeat that expensive work: it only proves the
        arena main page and three ordered cards still exist, then confirms the
        selected preview exposes the existing challenge-start control.
        """

        if self._challenge_selection_committed:
            raise ArenaReaderError(
                "arena_challenge_already_committed",
                "this reader session already sent an opponent-selection click",
            )
        if position not in (0, 1, 2):
            raise ArenaReaderError(
                "arena_challenge_position_invalid",
                f"selected opponent position is outside 0..2: {position}",
            )
        self.ensure_arena_main(require_opponents=True)
        image = self._capture()
        state, counts = self._arena_page_state(image)
        if state is not ArenaPageState.READY or counts[1] != 3:
            raise ArenaReaderError(
                "arena_challenge_preclick_guard_failed",
                f"arena main is not stable immediately before selection: state={state}, counts={counts}",
            )
        opponents = self._recognize("ArenaReaderOpponentCards", image)
        if len(opponents) != 3:
            raise ArenaReaderError(
                "arena_challenge_preclick_guard_failed",
                f"expected three opponent cards immediately before selection, got {len(opponents)}",
            )
        ordered = sorted(opponents, key=lambda item: (_box(item)[1], _box(item)[0]))
        target_box = _box(ordered[position])
        self._click(target_box)
        self._wait_for_stage_member_list((6,))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            preview = self._capture()
            starts = self._recognize("ChallengeStart", preview)
            if len(starts) == 1:
                self._challenge_selection_committed = True
                self._increment("arena_challenge_selection_clicks")
                return {
                    "position": position,
                    "opponent_count": 3,
                    "target_box": list(target_box),
                    "challenge_start_count": 1,
                    "full_lineup_reread": False,
                }
            if len(starts) > 1:
                break
            time.sleep(0.2)
        raise ArenaReaderError(
            "arena_challenge_start_ambiguous",
            "selected opponent preview did not expose one challenge-start control",
        )

    def select_stage(self, target: TeamTarget, stage_number: int) -> None:
        if stage_number not in (1, 2, 3):
            raise ArenaReaderError("stage_number_invalid", f"stage number is outside 1..3: {stage_number}")
        self._team_stage_total_anchors(target, self._capture())

    def member_slots(self, target: TeamTarget, stage_number: int) -> Sequence[int]:
        image = self._capture()
        totals = self._team_stage_total_anchors(target, image)
        anchor = totals[stage_number - 1]
        height, width = image.shape[:2]
        if self._grade is None:
            raise ArenaReaderError("arena_grade_missing", "contest Grade was not read from the arena main screen")
        slot_cap = stage_member_cap(self._grade, stage_number)
        avatar_height = int(height * 0.0906)
        avatar_top = anchor[1] + 35
        occupied: dict[int, tuple[int, int, int, int]] = {}
        observations: dict[int, tuple[MemberSlotState, MemberSlotMetrics, tuple[int, int, int, int]]] = {}
        for slot, box in enumerate(
            fixed_member_slot_boxes(
                width,
                avatar_top,
                avatar_height,
                slot_cap,
                group_center_ratio=(0.6430555556 if target.is_own_team else 0.7361111111),
            ),
            start=1,
        ):
            x0, y0, avatar_width, slot_height = box
            crop = image[y0 : y0 + slot_height, x0 : x0 + avatar_width]
            if crop.shape[:2] != (slot_height, avatar_width):
                raise ArenaReaderError("member_slot_roi_invalid", f"member slot exceeds frame: {box}")
            state, metrics = classify_member_slot(crop)
            observations[slot] = (state, metrics, box)
            if state is MemberSlotState.OCCUPIED:
                occupied[slot] = box
            elif state is MemberSlotState.AMBIGUOUS:
                raise ArenaReaderError(
                    "member_slot_ambiguous",
                    f"{target.team_id}/stage-{stage_number}/slot-{slot} is neither an explicit black blank "
                    f"nor a confirmed portrait: {metrics}",
                )
        if not occupied:
            raise ArenaReaderError(
                "member_count_mismatch",
                f"{target.team_id}/stage-{stage_number} has no occupied member within Grade {self._grade} cap {slot_cap}",
            )
        key = (target.team_id, stage_number)
        self._member_boxes[key] = occupied
        self._member_slot_observations[key] = observations
        return tuple(occupied)

    def open_member(self, target: TeamTarget, stage_number: int, slot: int) -> None:
        boxes = self._member_boxes.get((target.team_id, stage_number), {})
        box = boxes.get(slot)
        if box is None:
            raise ArenaReaderError("member_slot_missing", f"member slot {slot} has no recognized card")
        self._reset_member_card_state()
        self._member_metric_baseline = (
            dict(self._runtime_timing_seconds),
            dict(self._runtime_counts),
            {
                key: len(values)
                for key, values in self._runtime_duration_samples.items()
            },
        )
        self._long_press(box)
        try:
            self._assert_member_detail()
        except ArenaReaderError:
            # A long-press may occasionally be dropped.  Retry it once only
            # while the complete team-preview anchors prove that the first
            # action did not navigate anywhere; never repeat on an ambiguous
            # or partially transitioned screen.
            image = self._capture()
            self._team_stage_total_anchors(target, image)
            if len(self._ocr(image, r"^ステージ\s*[123]$")) < 3:
                raise
            time.sleep(0.5)
            self._long_press(box)
            self._assert_member_detail()

    def read_support_bonus(self, target: TeamTarget) -> float:
        del target
        image = self._capture()
        height, width = image.shape[:2]
        info_box = (
            int(width * 0.955),
            int(height * 0.022),
            1,
            1,
        )
        self._click(info_box, settle_seconds=0)
        started = time.perf_counter()
        deadline = time.monotonic() + 2.0
        value: float | None = None
        last_text = ""
        last_close_count = 0
        while time.monotonic() < deadline:
            overlay = self._capture()
            try:
                last_text = self._full_ocr_text(overlay)
            except ArenaReaderError:
                last_text = ""
            last_close_count = len(self._ocr(overlay, r"^閉じる$"))
            value = self._support_bonus_from_overlay(
                last_text,
                close_count=last_close_count,
            )
            self._increment("support_bonus_overlay_reads")
            if value is not None:
                break
            time.sleep(0.08)
        self._add_timing("support_bonus_open_wait", time.perf_counter() - started)
        if value is None:
            bonuses = re.findall(
                r"(?<![0-9])\+([0-9]+(?:\.[0-9]+)?)%",
                last_text,
            )
            raise ArenaReaderError(
                "support_bonus_missing",
                "support overlay did not reach one same-frame semantic decision: "
                f"bonuses={bonuses!r}, close_count={last_close_count}",
            )
        # The same info icon is the stable toggle on both own and opponent
        # member pages.  OCR's visible 閉じる label belongs to overlay content
        # and is not a reliable hit target across these two layouts.
        self._click(info_box, settle_seconds=0)
        self._assert_member_detail()
        return value

    @staticmethod
    def _support_bonus_from_overlay(
        text: str,
        *,
        close_count: int,
    ) -> float | None:
        """Resolve only a complete same-frame support-bonus overlay."""

        bonuses = re.findall(
            r"(?<![0-9])\+([0-9]+(?:\.[0-9]+)?)%",
            text,
        )
        if "サポートボーナス" not in text or len(bonuses) != 1 or close_count != 1:
            return None
        value = float(bonuses[0]) / 100
        if not 0 <= value <= 1:
            raise ArenaReaderError(
                "support_bonus_invalid",
                f"support bonus is outside [0,1]: {value}",
            )
        return value

    def read_params(self, target: TeamTarget, stage_number: int, slot: int) -> Sequence[int]:
        del target, stage_number, slot
        failure = "parameter OCR did not run"
        previous_values: tuple[int, ...] | None = None
        for attempt in range(4):
            image = self._capture()
            height, width = image.shape[:2]
            observations = self._ocr(
                image,
                r".+",
                roi=(0, 0, width, int(height * 0.25)),
            )
            label_items: dict[str, list[Any]] = {}
            for item in observations:
                label = _text(item)
                if label in {"ボーカル", "ダンス", "ビジュアル", "体力"}:
                    label_items.setdefault(label, []).append(item)
            label_boxes: tuple[tuple[int, int, int, int], ...] = ()
            if any(len(items) != 1 for items in label_items.values()):
                failure = "parameter labels were duplicated"
            else:
                try:
                    label_boxes = infer_parameter_label_boxes(
                        {label: _box(items[0]) for label, items in label_items.items()},
                        frame_height=height,
                    )
                except ValueError as error:
                    failure = (
                        f"parameter labels yielded {sorted(label_items)!r}: {error}"
                    )
                    label_boxes = ()
            if label_boxes:
                values: list[int] = []
                failure = ""
                for label, label_box in zip(
                    ("ボーカル", "ダンス", "ビジュアル", "体力"),
                    label_boxes,
                    strict=True,
                ):
                    candidates = sorted(
                        (
                            item
                            for item in observations
                            if re.fullmatch(r"[0-9]{1,6}", _text(item))
                            and int(width * 0.55) <= _box(item)[0] < int(width * 0.68)
                            and label_box[1] + label_box[3] - 6
                            <= _box(item)[1]
                            <= label_box[1] + label_box[3] + int(height * 0.04)
                        ),
                        key=lambda item: (_box(item)[0], _box(item)[1]),
                    )
                    if not candidates:
                        value_roi = (
                            int(width * 0.55),
                            max(0, label_box[1] + label_box[3] - 8),
                            int(width * 0.16),
                            int(height * 0.055),
                        )
                        candidates = sorted(
                            self._ocr(image, r"^[0-9]{1,6}$", roi=value_roi),
                            key=lambda item: (_box(item)[0], _box(item)[1]),
                        )
                    enhanced_value: int | None = None
                    enhanced_observations: tuple[str, ...] = ()
                    if not candidates:
                        enhanced_value, enhanced_observations = self._read_enhanced_numeric_roi(
                            image,
                            value_roi,
                        )
                        if enhanced_value is not None:
                            values.append(enhanced_value)
                            continue
                    if len(candidates) != 1:
                        numeric_observations = [
                            (_text(item), _box(item))
                            for item in observations
                            if re.fullmatch(r"[0-9]{1,6}", _text(item))
                        ]
                        failure = (
                            f"{label} yielded "
                            f"{[(_text(item), _box(item)) for item in candidates]!r}; "
                            f"numeric observations={numeric_observations!r}; "
                            f"enhanced observations={enhanced_observations!r}"
                        )
                        break
                    values.append(int(_text(candidates[0])))
                if not failure and len(values) == 4:
                    current_values = tuple(values)
                    if current_values == previous_values:
                        return current_values
                    previous_values = current_values
                    failure = f"parameter values have not stabilized: {current_values!r}"
            if attempt < 3:
                time.sleep(0.25)
        raise ArenaReaderError("param_value_ambiguous", f"stable in-memory reads failed: {failure}")

    def _read_enhanced_numeric_roi(
        self,
        image: Any,
        roi: tuple[int, int, int, int],
    ) -> tuple[int | None, tuple[str, ...]]:
        from collections import Counter

        import cv2

        x, y, width, height = roi
        crop = image[y : y + height, x : x + width]
        if crop.shape[:2] != (height, width):
            return None, ()
        enlarged_2x = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        enlarged_3x = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        enlarged_4x = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        contrast = cv2.convertScaleAbs(enlarged_3x, alpha=1.5, beta=-20)
        gray = cv2.cvtColor(enlarged_3x, cv2.COLOR_BGR2GRAY)
        _, binary_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2BGR)
        observed: list[str] = []
        votes: Counter[int] = Counter()
        for name, variant in (
            ("2x", enlarged_2x),
            ("3x", enlarged_3x),
            ("4x", enlarged_4x),
            ("contrast", contrast),
            ("binary", binary),
        ):
            results = self._ocr(variant, r"^[0-9]{1,6}$")
            texts = tuple(_text(item) for item in results if re.fullmatch(r"[0-9]{1,6}", _text(item)))
            observed.append(f"{name}:{','.join(texts) or '-'}")
            if len(texts) == 1:
                votes[int(texts[0])] += 1
        ranked = votes.most_common()
        if ranked and ranked[0][1] >= 2 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            return ranked[0][0], tuple(observed)
        return None, tuple(observed)

    @staticmethod
    def _p_item_decision_evidence(
        slot: int,
        decision: PItemReferenceDecision,
        *,
        path: str,
        resolved_id: int | None = None,
        screen_slot: int | None = None,
    ) -> dict[str, Any]:
        evidence = {
            "slot": slot,
            "state": decision.status,
            "reason": decision.reason,
            "path": path,
            "p_item_id": decision.p_item_id if resolved_id is None else resolved_id,
            "candidates": list(decision.candidates),
            "similarity": (
                None if decision.similarity is None else round(decision.similarity, 6)
            ),
            "margin": None if decision.margin is None else round(decision.margin, 6),
            "content_generation_max_mean_abs_error": round(
                decision.content_generation_max_mean_abs_error,
                6,
            ),
            "frames_byte_identical": decision.frames_byte_identical,
            "ranking_route": decision.ranking_route,
            "full_fallback_used": decision.full_fallback_used,
            "completeness": [
                {
                    "foreground_fraction": round(item.foreground_fraction, 6),
                    "minimum_opposite_half_fraction": round(
                        item.minimum_opposite_half_fraction,
                        6,
                    ),
                }
                for item in decision.completeness
            ],
            "top_k": [
                {
                    "p_item_id": hit.p_item_id,
                    "similarity": round(hit.similarity, 6),
                    "size": list(hit.size),
                    "offset": list(hit.offset),
                }
                for hit in decision.top_k
            ],
        }
        if screen_slot is not None:
            evidence["screen_slot"] = screen_slot
        return evidence

    def _confirm_p_item_detail(
        self,
        box: tuple[int, int, int, int],
        *,
        candidate_ids: Sequence[int],
    ) -> int:
        if not candidate_ids:
            raise ArenaReaderError(
                "p_item_detail_candidates_missing",
                "P-item detail fallback requires fixed-reference candidates",
            )
        started = time.perf_counter()
        interaction_box = self._p_item_interaction_box(box)
        self._click(interaction_box, settle_seconds=0)
        self._increment("p_item_detail_clicks")
        self._increment("p_item_safe_region_clicks")
        transaction_states = ["SOURCE_STABLE", "CLICK_SENT"]
        deadline = time.monotonic() + 3.0
        last_text = ""
        last_error = "P-item detail OCR did not run"
        resolved: int | None = None
        candidate_title_seen = False
        detail_confirmed = False
        consecutive_source_reads = 0
        consecutive_non_source_reads = 0
        retried = False
        terminal_error: Exception | None = None
        open_wait_started = time.perf_counter()
        while time.monotonic() < deadline:
            try:
                image = self._capture()
                ocr_started = time.perf_counter()
                last_text = self._full_ocr_text(image)
                self._add_timing(
                    "p_item_detail_ocr",
                    time.perf_counter() - ocr_started,
                )
                matches = self.catalog.clicked_p_item_candidate_matches(
                    last_text,
                    candidate_p_item_ids=candidate_ids,
                )
            except Exception as error:
                terminal_error = error
                last_error = str(error)
                matches = ()
                image = None
            source_visible = False
            if terminal_error is not None:
                break
            if image is not None:
                try:
                    source_visible = (
                        "体力" in last_text and "総合力" in last_text
                    ) or bool(
                        self._ocr(image, r"^体力$")
                        and self._ocr(image, r"^総合力$")
                    )
                except Exception as error:
                    terminal_error = error
                    last_error = str(error)
                    break
            if len(matches) == 1:
                resolved = matches[0]
                detail_confirmed = True
                transaction_states.extend(("DETAIL_CONFIRMED", "RESOLVED"))
                break
            if matches:
                candidate_title_seen = True
                detail_confirmed = True
                consecutive_source_reads = 0
                transaction_states.append("DETAIL_CONFIRMED")
                last_error = (
                    "P-item detail matched multiple candidate titles "
                    f"{matches!r}"
                )
            elif source_visible:
                consecutive_source_reads += 1
                consecutive_non_source_reads = 0
                last_error = "the original member page remained stable after the click"
            else:
                consecutive_source_reads = 0
                consecutive_non_source_reads += 1
                if last_text:
                    last_error = (
                        "P-item detail exposed no fixed candidate title "
                        f"within {tuple(candidate_ids)!r}"
                    )
                if consecutive_non_source_reads >= 2 and last_text:
                    detail_confirmed = True
                    transaction_states.extend(("DETAIL_CONFIRMED", "AMBIGUOUS"))
                    break
            if (
                not retried
                and not detail_confirmed
                and consecutive_source_reads >= 3
            ):
                self._click(interaction_box, settle_seconds=0)
                self._increment("p_item_detail_click_retries")
                self._increment("p_item_safe_region_clicks")
                retried = True
                consecutive_source_reads = 0
                transaction_states.append("CLICK_SENT_RETRY")
            elif retried and consecutive_source_reads >= 3:
                transaction_states.append("OPEN_FAILED")
                break
            time.sleep(0.12)
        self._add_timing(
            "p_item_detail_open_wait",
            time.perf_counter() - open_wait_started,
        )
        recovery_started = time.perf_counter()
        try:
            if detail_confirmed or terminal_error is not None:
                self._dismiss_skill_card_detail()
                transaction_states.append("DISMISS_SENT")
            self._assert_member_detail()
            transaction_states.append("SOURCE_RESTORED")
        except ArenaReaderError as recovery_error:
            elapsed = time.perf_counter() - started
            self._add_timing("p_item_detail_transactions", elapsed)
            self._add_timing(
                "p_item_detail_close_restore",
                time.perf_counter() - recovery_started,
            )
            self._record_duration_sample("p_item_detail_failed_transaction", elapsed)
            raise ArenaReaderError(
                "p_item_detail_recovery_failed",
                "P-item detail transaction did not restore the original member page; "
                f"states={transaction_states!r}",
            ) from recovery_error
        self._add_timing(
            "p_item_detail_close_restore",
            time.perf_counter() - recovery_started,
        )
        elapsed = time.perf_counter() - started
        self._add_timing("p_item_detail_transactions", elapsed)
        if terminal_error is not None:
            self._record_duration_sample("p_item_detail_failed_transaction", elapsed)
            raise ArenaReaderError(
                "p_item_detail_parser_failed",
                "P-item detail parsing failed after the source page was restored; "
                f"states={transaction_states!r}; error={last_error}",
            ) from terminal_error
        if resolved is None:
            self._record_duration_sample("p_item_detail_failed_transaction", elapsed)
            error_code = (
                "p_item_detail_ambiguous"
                if detail_confirmed or candidate_title_seen
                else "p_item_detail_open_or_title_failed"
            )
            raise ArenaReaderError(
                error_code,
                f"P-item detail did not uniquely confirm {tuple(candidate_ids)!r}: "
                f"{last_error}; states={transaction_states!r}; "
                f"safe_box={interaction_box!r}; retried={retried}; "
                f"OCR={last_text[:400]!r}",
            )
        self._record_duration_sample("p_item_detail_transaction", elapsed)
        return resolved

    @staticmethod
    def _p_item_interaction_box(
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Keep Maa's click target inside the central, unobscured P-item artwork."""

        x, y, width, height = box
        if width < 16 or height < 16:
            raise ArenaReaderError(
                "p_item_interaction_box_invalid",
                f"P-item box is too small for a safe click: {box!r}",
            )
        inset_x = max(2, width // 4)
        inset_y = max(2, height // 4)
        return (
            x + inset_x,
            y + inset_y,
            width - inset_x * 2,
            height - inset_y * 2,
        )

    def _capture_stable_p_item_generation(
        self,
        boxes: Sequence[tuple[int, int, int, int]],
        *,
        initial_image: Any | None = None,
        timeout_seconds: float = 1.5,
        interval_seconds: float = 0.08,
    ) -> tuple[tuple[Any, Any, Any], tuple[float, ...], int]:
        """Wait for three consecutive frames from one P-item render generation."""

        if self.p_item_reader is None:
            raise ArenaReaderError(
                "p_item_reader_missing",
                "P-item content stability requires an initialized reader",
            )
        threshold = float(
            self.p_item_reader.content_generation_max_mean_abs_error
        )
        frames: list[Any] = []
        rejected_windows = 0
        last_errors: tuple[float, ...] = ()
        deadline = time.monotonic() + timeout_seconds
        started = time.perf_counter()
        pending = initial_image
        while time.monotonic() < deadline:
            image = pending if pending is not None else self._capture()
            pending = None
            frames.append(image)
            if len(frames) > 3:
                frames.pop(0)
            if len(frames) == 3:
                try:
                    last_errors = tuple(
                        float(value)
                        for value in self.p_item_reader.content_generation_errors(
                            frames,
                            boxes,
                        )
                    )
                except (PItemReferenceError, TypeError, ValueError) as error:
                    raise ArenaReaderError(
                        "p_item_content_generation_measurement_failed",
                        "P-item content-generation measurement failed",
                    ) from error
                if len(last_errors) != len(boxes):
                    raise ArenaReaderError(
                        "p_item_content_generation_count_mismatch",
                        "P-item content-generation measurement returned "
                        f"{len(last_errors)} slots for {len(boxes)} boxes",
                    )
                if all(value <= threshold for value in last_errors):
                    self._add_timing(
                        "p_item_content_stable_wait",
                        time.perf_counter() - started,
                    )
                    return (
                        (frames[0], frames[1], frames[2]),
                        last_errors,
                        rejected_windows,
                    )
                rejected_windows += 1
                self._increment("p_item_content_generation_resets")
            if interval_seconds > 0:
                time.sleep(interval_seconds)
        self._add_timing(
            "p_item_content_stable_wait",
            time.perf_counter() - started,
        )
        self._p_item_generation_evidence = {
            "frame_ids": [frame_identifier(image) for image in frames],
            "content_stable": False,
            "identity_resolved": False,
            "content_generation_errors": [
                round(value, 6) for value in last_errors
            ],
            "content_generation_threshold": threshold,
            "content_generation_rejected_windows": rejected_windows,
        }
        raise ArenaReaderError(
            "p_item_content_generation_unstable",
            "P-item slots did not yield three consecutive stable frames within "
            f"{timeout_seconds:.2f}s; threshold={threshold:.6f}; "
            f"last_errors={[round(value, 6) for value in last_errors]!r}",
        )

    def read_p_item_ids(
        self,
        target: TeamTarget,
        stage_number: int,
        slot: int,
    ) -> Sequence[int]:
        del target, slot
        self._p_item_source_frames = None
        if self.p_item_reader is None:
            self.p_item_reader = Task085PItemReader.from_model_root()
        first = self._capture()
        height, width = first.shape[:2]
        icon = int(width * 0.09)
        boxes = tuple(
            (int(width * x), int(height * 0.207), icon, icon)
            for x in (0.05, 0.147, 0.247, 0.34)
        )
        plan = self.season.stages[stage_number - 1].plan
        decisions: tuple[PItemReferenceDecision, ...] = ()
        images: tuple[Any, ...] = ()
        generation_attempt = 0
        matching_seconds = 0.0
        content_generation_errors: tuple[float, ...] = ()
        content_generation_rejected_windows = 0
        while generation_attempt < 2:
            generation_attempt += 1
            (
                images,
                content_generation_errors,
                rejected_windows,
            ) = self._capture_stable_p_item_generation(
                boxes,
                initial_image=first if generation_attempt == 1 else None,
            )
            content_generation_rejected_windows += rejected_windows
            started = time.perf_counter()
            try:
                decisions = tuple(
                    self.p_item_reader.read(images, boxes, plan=plan)
                )
            except (PItemReferenceError, OSError, KeyError, TypeError, ValueError) as error:
                raise ArenaReaderError(
                    "p_item_reference_runtime_failed",
                    "fixed P-item reference matching failed",
                ) from error
            matching_seconds += time.perf_counter() - started
            if len(decisions) != 4:
                raise ArenaReaderError(
                    "p_item_count_mismatch",
                    f"P-item reference reader returned {len(decisions)} slots",
                )
            if all(decision.accepted for decision in decisions):
                break
            if generation_attempt == 1:
                self._increment("p_item_generation_resamples")
                continue
            break
        self._add_timing("p_item_reference_matching", matching_seconds)
        ambiguous = tuple(
            (index, decision)
            for index, decision in enumerate(decisions, start=1)
            if not decision.accepted
        )
        detail_eligible = tuple(
            (index, decision)
            for index, decision in ambiguous
            if decision.reason
            in {
                "reference_margin_below_threshold",
                "reference_similarity_below_threshold",
            }
            and decision.candidates
        )
        if len(detail_eligible) != len(ambiguous):
            raise ArenaReaderError(
                "p_item_unknown",
                "P-item slots stayed ambiguous after one fresh generation: "
                + ", ".join(
                    f"slot {index}={decision.reason}"
                    "("
                    f"content_error={decision.content_generation_max_mean_abs_error:.6f},"
                    f"similarity={decision.similarity!r},"
                    f"margin={decision.margin!r},"
                    f"candidates={decision.candidates!r},"
                    "top_k="
                    f"{tuple((hit.p_item_id, round(hit.similarity, 6)) for hit in decision.top_k)!r},"
                    "minimum_opposite_half="
                    f"{tuple(round(item.minimum_opposite_half_fraction, 6) for item in decision.completeness)!r}"
                    ")"
                    for index, decision in ambiguous
                ),
            )
        if len(detail_eligible) > self.p_item_reader.maximum_detail_fallbacks_per_member:
            raise ArenaReaderError(
                "p_item_detail_budget_exceeded",
                f"P-item detail fallbacks {len(detail_eligible)} exceed the per-member budget "
                f"{self.p_item_reader.maximum_detail_fallbacks_per_member}",
            )

        screen_resolved_ids: list[int] = []
        screen_diagnostics: list[dict[str, Any]] = []
        detail_by_slot = dict(detail_eligible)
        for screen_slot, (box, decision) in enumerate(
            zip(boxes, decisions, strict=True),
            start=1,
        ):
            engine_slot = p_item_screen_slot_to_engine_slot(screen_slot)
            if screen_slot in detail_by_slot:
                resolved_id = self._confirm_p_item_detail(
                    box,
                    candidate_ids=decision.candidates,
                )
                screen_resolved_ids.append(resolved_id)
                screen_diagnostics.append(
                    self._p_item_decision_evidence(
                        engine_slot,
                        decision,
                        path="bounded_detail_confirmation",
                        resolved_id=resolved_id,
                        screen_slot=screen_slot,
                    )
                )
                continue
            if decision.p_item_id is None:
                raise ArenaReaderError(
                    "p_item_unknown",
                    f"P-item screen slot {screen_slot} has no resolved business ID",
                )
            screen_resolved_ids.append(decision.p_item_id)
            path = (
                "stable_empty_slot"
                if decision.status == "EMPTY"
                else "three_frame_rendered_reference"
            )
            self._increment(
                "p_item_empty_slots"
                if decision.status == "EMPTY"
                else "p_item_exact_reference_resolutions"
            )
            screen_diagnostics.append(
                self._p_item_decision_evidence(
                    engine_slot,
                    decision,
                    path=path,
                    screen_slot=screen_slot,
                )
            )
        gallery = getattr(self.p_item_reader, "gallery", None)
        resolved_ids = p_item_screen_order_to_engine_order(screen_resolved_ids)
        self._p_item_diagnostics = p_item_screen_order_to_engine_order(screen_diagnostics)
        self._p_item_generation_evidence = {
            "frame_ids": [frame_identifier(image) for image in images],
            "fresh_resamples": generation_attempt - 1,
            "content_stable": True,
            "identity_resolved": True,
            "content_generation_errors": [
                round(value, 6) for value in content_generation_errors
            ],
            "content_generation_threshold": (
                self.p_item_reader.content_generation_max_mean_abs_error
            ),
            "content_generation_rejected_windows": (
                content_generation_rejected_windows
            ),
            "reference_gallery_sha256": getattr(gallery, "gallery_sha256", None),
            "background_workers": 2,
            "ranking_routes": [decision.ranking_route for decision in decisions],
            "full_fallbacks": sum(
                int(decision.full_fallback_used) for decision in decisions
            ),
            "screen_slot_to_engine_slot": [1, 4, 3, 2],
            "matching_seconds": round(matching_seconds, 6),
        }
        # With no P-item detail transaction, these three frames still belong
        # to the untouched member page.  The pre-swipe upper-left safety guard
        # may reuse them, but only after independently proving card-row
        # geometry and that card's content generation.  Any P-item click makes
        # the cache ineligible because a UI transaction occurred in between.
        self._p_item_source_frames = (
            (images[0], images[1], images[2])
            if not detail_eligible and len(images) == 3
            else None
        )
        if self._p_item_source_frames is not None:
            self._increment("p_item_source_frame_generations_cached")
        return resolved_ids

    def _reusable_preswipe_upper_first_generation(
        self,
    ) -> tuple[
        tuple[Any, Any, Any],
        tuple[tuple[tuple[int, int, int, int], ...], ...],
    ] | None:
        """Reuse untouched P-item frames only when upper-left evidence is stable."""

        frames = getattr(self, "_p_item_source_frames", None)
        self._p_item_source_frames = None
        if frames is None:
            return None
        try:
            rows = tuple(
                self._validated_card_group_row(image, 0) for image in frames
            )
            signatures = tuple(
                self._card_references().content_signature_with_identity(
                    image,
                    row[0],
                )[:2]
                for image, row in zip(frames, rows, strict=True)
            )
        except (ArenaReaderError, BadgeReferenceError):
            self._increment("upper_first_preswipe_frame_reuse_rejected")
            return None
        if any(
            self._card_rows_shifted(actual, expected)
            for actual, expected in zip(rows[1:], rows[:-1], strict=True)
        ) or any(
            self._card_content_generation_shifted(
                (actual,),
                (expected,),
            )
            for actual, expected in zip(
                signatures[1:],
                signatures[:-1],
                strict=True,
            )
        ):
            self._increment("upper_first_preswipe_frame_reuse_rejected")
            return None
        self._increment("upper_first_preswipe_frame_reuses")
        return frames, rows

    def prepare_skill_card_group(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> None:
        required_groups = (0, 1) if group_index == 1 else (0,)
        if all(
            index in self._card_rows
            and index in self._card_images
            and len(self._card_count_frames.get(index, ())) == 3
            for index in required_groups
        ):
            return
        if group_index == 1:
            self._pre_resolve_upper_first_card(
                target,
                stage_number,
                member_slot,
            )
            self._swipe(vertical="up")
        first_error: ArenaReaderError | None = None
        try:
            self._capture_stable_card_groups(
                required_groups,
                allow_fixed_secondary=group_index == 1,
            )
            if group_index == 1:
                self._secondary_fixed_slot_fallback_enabled = True
            return
        except ArenaReaderError as error:
            if group_index != 1:
                raise
            first_error = error

        # A second gesture is allowed only when screen geometry proves that
        # the first gesture did not reach the calibrated position. If the
        # primary row did move, repeating the gesture cannot expose more
        # content and only adds latency; the fixed-slot presence gate has
        # already been attempted on every stable frame.
        image = self._capture()
        rows = canonical_skill_card_rows(
            image.shape,
            self._raw_card_candidate_boxes(image),
        )
        assert first_error is not None
        if rows and rows[0][0][1] <= int(image.shape[0] * 0.205):
            if first_error.code == "skill_card_layout_incomplete":
                raise ArenaReaderError(
                    "skill_card_secondary_row_missing",
                    "primary row reached the calibrated position but group 1 lacked "
                    f"six stable fixed-slot card observations: {first_error}",
                ) from first_error
            raise first_error
        self._swipe(vertical="up")
        self._increment("corrective_swipes")
        self._capture_stable_card_groups(
            required_groups,
            allow_fixed_secondary=True,
        )
        self._secondary_fixed_slot_fallback_enabled = True

    def _pre_resolve_upper_first_card(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
    ) -> None:
        """Protect the only card whose detail hitbox disappears after the swipe."""

        started = time.perf_counter()
        reused = self._reusable_preswipe_upper_first_generation()
        source_generation_cached = reused is None
        if reused is None:
            self._capture_stable_card_groups(
                (0,),
                interval_seconds=0.04,
            )
            frames = self._card_count_frames[0]
            frame_rows = tuple(
                self._validated_card_group_row(image, 0) for image in frames
            )
        else:
            frames, frame_rows = reused

        def classify_upper_first() -> tuple[bool, Any]:
            repeated_visual_features = tuple(
                duplicate_marker_visual_features(
                    image[
                        row[0][1] : row[0][1] + row[0][3],
                        row[0][0] : row[0][0] + row[0][2],
                        :3,
                    ]
                )
                for image, row in zip(frames, frame_rows, strict=True)
            )
            if is_duplicate_marker_visual_candidate(repeated_visual_features):
                duplicate_observations, diagnostics = (
                    self._targeted_duplicate_marker_observations(
                        frames,
                        frame_rows[0],
                        (0,),
                        phase="pre_swipe_upper_first",
                    )
                )
                stable_duplicate = stable_excluded_duplicate_card_flags(
                    duplicate_observations
                )[0]
                self._card_duplicate_marker_diagnostics[0] = tuple(diagnostics)
                if stable_duplicate is None:
                    raise ArenaReaderError(
                        "skill_card_upper_first_duplicate_ambiguous",
                        "upper group/slot 1 duplicate marker changed across pre-swipe frames",
                    )
                if stable_duplicate:
                    return True, None
            shortlist = self._local_badge_slot_decision(
                frames,
                frame_rows,
                0,
                0,
            )
            self._badge_glyph_observations[(0, 1)] = shortlist.observations
            return False, shortlist

        stable_duplicate, shortlist = classify_upper_first()
        if stable_duplicate:
            self._increment("upper_first_preswipe_excluded")
            self._add_timing(
                "upper_first_preswipe_guard",
                time.perf_counter() - started,
            )
            return
        decision = shortlist.decision
        if (
            decision.state is not CustomizationBadgeState.CONFIDENT_ZERO
            and not source_generation_cached
        ):
            # A positive upper-left candidate will open a UI transaction.  The
            # source-restoration gate therefore needs full six-slot signatures,
            # not only the single-card proof used by the no-click fast path.
            self._increment("upper_first_preswipe_frame_reuse_escalations")
            self._capture_stable_card_groups(
                (0,),
                interval_seconds=0.04,
            )
            frames = self._card_count_frames[0]
            frame_rows = tuple(
                self._validated_card_group_row(image, 0) for image in frames
            )
            source_generation_cached = True
            stable_duplicate, shortlist = classify_upper_first()
            if stable_duplicate:
                self._increment("upper_first_preswipe_excluded")
                self._add_timing(
                    "upper_first_preswipe_guard",
                    time.perf_counter() - started,
                )
                return
            decision = shortlist.decision
        if decision.state is not CustomizationBadgeState.CONFIDENT_ZERO:
            if not self._badge_detail_inferable(decision):
                raise ArenaReaderError(
                    "skill_card_upper_first_badge_ambiguous",
                    f"upper group/slot 1 is ambiguous before swipe: {decision.reason}",
                )
            detail_check = self._record_badge_candidate_detail(
                stage_number=stage_number,
                member_slot=member_slot,
                group_index=0,
                card_slot=1,
                phase="pre_swipe_upper_first",
                reason=decision.reason,
            )
            inferred = self._infer_badge_card_from_detail(
                target,
                stage_number,
                member_slot,
                0,
                1,
                frames[0],
                frame_rows[0][0],
                None,
                "badge_candidate_detail_transaction",
                allow_zero_without_badge_count=True,
            )
            self._finish_badge_candidate_detail(detail_check, inferred)
            inferred_count = sum(
                int(value) for value in inferred.customizations.values()
            )
            self._increment(
                "upper_first_preswipe_resolved"
                if inferred_count > 0
                else "upper_first_preswipe_zero"
            )
            self._add_timing(
                "upper_first_preswipe_guard",
                time.perf_counter() - started,
            )
            return
        else:
            self._increment("upper_first_preswipe_zero")
        self._add_timing(
            "upper_first_preswipe_guard",
            time.perf_counter() - started,
        )

    @staticmethod
    def _card_rows_shifted(
        actual: Sequence[tuple[int, int, int, int]],
        expected: Sequence[tuple[int, int, int, int]],
        *,
        tolerance: int = 3,
    ) -> bool:
        return any(
            abs(actual_value - expected_value) > tolerance
            for actual_box, expected_box in zip(actual, expected, strict=True)
            for actual_value, expected_value in zip(actual_box, expected_box, strict=True)
        )

    @staticmethod
    def _card_content_generation_deltas(
        actual: Sequence[tuple[str, Any]],
        expected: Sequence[tuple[str, Any]],
        *,
        maximum_mean_absolute_error: float = (
            CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR
        ),
    ) -> tuple[dict[str, Any], ...]:
        """Measure why a rendered-card generation differs after geometry settles."""

        import numpy as np

        if len(actual) != len(expected) or not actual:
            return (
                {
                    "slot": 0,
                    "visual_group_changed": False,
                    "shape_changed": True,
                    "mean_absolute_error": None,
                    "shifted": True,
                },
            )
        deltas: list[dict[str, Any]] = []
        for slot, ((actual_group, actual_query), (expected_group, expected_query)) in enumerate(
            zip(actual, expected, strict=True),
            start=1,
        ):
            actual_array = np.asarray(actual_query)
            expected_array = np.asarray(expected_query)
            visual_group_changed = actual_group != expected_group
            shape_changed = actual_array.shape != expected_array.shape
            motion = None
            zero_mean_motion = None
            edge_motion = None
            structural_correlation = None
            mean_color_shift = None
            if not shape_changed:
                actual_float = actual_array.astype(np.float32)
                expected_float = expected_array.astype(np.float32)
                difference = actual_float - expected_float
                motion = float(
                    np.mean(np.abs(difference))
                )
                channel_shift = np.mean(difference, axis=(0, 1), keepdims=True)
                mean_color_shift = float(np.mean(np.abs(channel_shift)))
                zero_mean_motion = float(
                    np.mean(np.abs(difference - channel_shift))
                )
                actual_gray = np.mean(actual_float, axis=2)
                expected_gray = np.mean(expected_float, axis=2)
                actual_edges = np.concatenate(
                    (
                        np.diff(actual_gray, axis=0).ravel(),
                        np.diff(actual_gray, axis=1).ravel(),
                    )
                )
                expected_edges = np.concatenate(
                    (
                        np.diff(expected_gray, axis=0).ravel(),
                        np.diff(expected_gray, axis=1).ravel(),
                    )
                )
                edge_motion = float(np.mean(np.abs(actual_edges - expected_edges)))
                actual_centered = actual_float.ravel() - float(np.mean(actual_float))
                expected_centered = expected_float.ravel() - float(
                    np.mean(expected_float)
                )
                denominator = float(
                    np.linalg.norm(actual_centered) * np.linalg.norm(expected_centered)
                )
                structural_correlation = (
                    1.0
                    if denominator == 0.0 and motion == 0.0
                    else (
                        0.0
                        if denominator == 0.0
                        else float(
                            np.dot(actual_centered, expected_centered) / denominator
                        )
                    )
                )
            deltas.append(
                {
                    "slot": slot,
                    "visual_group_changed": visual_group_changed,
                    "shape_changed": shape_changed,
                    "mean_absolute_error": motion,
                    "mean_color_shift": mean_color_shift,
                    "zero_mean_absolute_error": zero_mean_motion,
                    "edge_mean_absolute_error": edge_motion,
                    "structural_correlation": structural_correlation,
                    "shifted": (
                        visual_group_changed
                        or shape_changed
                        or motion is None
                        or motion > maximum_mean_absolute_error
                    ),
                }
            )
        return tuple(deltas)

    @classmethod
    def _card_content_generation_shifted(
        cls,
        actual: Sequence[tuple[str, Any]],
        expected: Sequence[tuple[str, Any]],
        *,
        maximum_mean_absolute_error: float = (
            CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR
        ),
    ) -> bool:
        """Detect a new rendered-card generation after geometry has settled."""

        return any(
            bool(delta["shifted"])
            for delta in cls._card_content_generation_deltas(
                actual,
                expected,
                maximum_mean_absolute_error=maximum_mean_absolute_error,
            )
        )

    @staticmethod
    def _reference_business_ids(identity: dict[str, Any]) -> tuple[int, ...]:
        """Return every high-confidence clean-reference ID in one observation."""

        raw_candidates: Sequence[dict[str, Any]]
        if identity.get("status") == "MEASURED":
            raw_candidates = (identity,)
        elif identity.get("status") == "MEASURED_CANDIDATES":
            raw_candidates = tuple(
                candidate
                for candidate in identity.get("candidates", ())
                if isinstance(candidate, dict)
            )
        else:
            return ()
        candidate_ids = {
            int(candidate["reference_business_id"])
            for candidate in raw_candidates
            if candidate.get("identity_low_confidence") is not True
            and isinstance(candidate.get("reference_business_id"), int)
            and not isinstance(candidate.get("reference_business_id"), bool)
            and int(candidate["reference_business_id"]) > 0
        }
        return tuple(sorted(candidate_ids))

    @staticmethod
    def _unique_reference_business_id(identity: dict[str, Any]) -> int | None:
        """Return one high-confidence clean-reference ID, never a shortlist."""

        candidate_ids = MaaArenaReaderBackend._reference_business_ids(identity)
        if len(candidate_ids) != 1:
            return None
        return candidate_ids[0]

    def _card_content_generation_signatures(
        self,
        image: Any,
        rows: dict[int, tuple[tuple[int, int, int, int], ...]],
    ) -> tuple[
        dict[int, tuple[tuple[str, Any], ...]],
        dict[int, tuple[dict[str, Any], ...]],
    ]:
        gallery = self._card_references()
        signatures: dict[int, tuple[tuple[str, Any], ...]] = {}
        identities: dict[int, tuple[dict[str, Any], ...]] = {}
        for group_index, row in rows.items():
            measured = tuple(
                gallery.content_signature_with_identity(image, box)
                for box in row
            )
            signatures[group_index] = tuple(
                (visual_group, query)
                for visual_group, query, _ in measured
            )
            identities[group_index] = tuple(
                identity for _, _, identity in measured
            )
        return signatures, identities

    def _capture_stable_card_groups(
        self,
        group_indices: Sequence[int],
        *,
        timeout_seconds: float = 1.5,
        maximum_timeout_seconds: float = 2.5,
        confirmation_grace_seconds: float = 0.6,
        interval_seconds: float = 0.08,
        allow_fixed_secondary: bool = False,
    ) -> None:
        """Capture one three-frame observation shared by every visible card row."""

        groups = tuple(dict.fromkeys(int(index) for index in group_indices))
        if not groups or any(index not in (0, 1) for index in groups):
            raise ValueError("group_indices must contain card groups 0 and/or 1")
        if timeout_seconds <= 0 or maximum_timeout_seconds < timeout_seconds:
            raise ValueError(
                "maximum_timeout_seconds must be at least the positive base timeout"
            )
        if confirmation_grace_seconds < 0:
            raise ValueError("confirmation_grace_seconds cannot be negative")
        frames: list[Any] = []
        rows_by_group: dict[int, list[tuple[tuple[int, int, int, int], ...]]] = {
            index: [] for index in groups
        }
        content_by_group: dict[int, list[tuple[tuple[str, Any], ...]]] = {
            index: [] for index in groups
        }
        identity_by_group: dict[int, list[tuple[dict[str, Any], ...]]] = {
            index: [] for index in groups
        }
        monotonic_started = time.monotonic()
        deadline = monotonic_started + timeout_seconds
        absolute_deadline = monotonic_started + maximum_timeout_seconds
        next_capture_not_before = monotonic_started
        capture_times: list[float] = []
        started = time.perf_counter()
        last_error = "no complete card frame"
        last_failure_code = "skill_card_layout_unstable"
        instability: dict[tuple[int, int], dict[str, Any]] = {}

        def instability_summary() -> tuple[dict[str, Any], ...]:
            return tuple(
                {
                    **value,
                    **{
                        key: round(float(value[key]), 6)
                        for key in (
                            "maximum_mean_absolute_error",
                            "maximum_mean_color_shift",
                            "maximum_zero_mean_absolute_error",
                            "maximum_edge_mean_absolute_error",
                            "minimum_structural_correlation",
                        )
                    },
                }
                for _, value in sorted(instability.items())
            )

        while time.monotonic() < deadline:
            remaining = next_capture_not_before - time.monotonic()
            while remaining > 0:
                time.sleep(remaining)
                remaining = next_capture_not_before - time.monotonic()
            if time.monotonic() >= deadline:
                break
            capture_started = time.monotonic()
            next_capture_not_before = capture_started + interval_seconds
            image = self._capture()
            try:
                observed = {
                    index: self._validated_card_group_row(
                        image,
                        index,
                        allow_fixed_secondary=allow_fixed_secondary,
                    )
                    for index in groups
                }
            except ArenaReaderError as error:
                last_error = str(error)
                last_failure_code = "skill_card_layout_incomplete"
                frames.clear()
                capture_times.clear()
                for values in rows_by_group.values():
                    values.clear()
                for values in content_by_group.values():
                    values.clear()
                for values in identity_by_group.values():
                    values.clear()
            else:
                content_started = time.perf_counter()
                content, identities = self._card_content_generation_signatures(
                    image,
                    observed,
                )
                self._add_timing(
                    "card_content_stability_check",
                    time.perf_counter() - content_started,
                )
                geometry_shifted = bool(frames) and any(
                    self._card_rows_shifted(observed[index], rows_by_group[index][-1])
                    for index in groups
                )
                content_deltas = (
                    {
                        index: self._card_content_generation_deltas(
                            content[index],
                            content_by_group[index][-1],
                        )
                        for index in groups
                    }
                    if frames
                    else {}
                )
                content_shifted = any(
                    bool(delta["shifted"])
                    for deltas in content_deltas.values()
                    for delta in deltas
                )
                if geometry_shifted or content_shifted:
                    frames.clear()
                    capture_times.clear()
                    for values in rows_by_group.values():
                        values.clear()
                    for values in content_by_group.values():
                        values.clear()
                    for values in identity_by_group.values():
                        values.clear()
                    if content_shifted:
                        last_failure_code = "skill_card_content_generation_unstable"
                        for group_index, deltas in content_deltas.items():
                            for delta in deltas:
                                if not delta["shifted"]:
                                    continue
                                key = (group_index, int(delta["slot"]))
                                aggregate = instability.setdefault(
                                    key,
                                    {
                                        "group": group_index,
                                        "slot": int(delta["slot"]),
                                        "shifts": 0,
                                        "visual_group_changes": 0,
                                        "shape_changes": 0,
                                        "maximum_mean_absolute_error": 0.0,
                                        "maximum_mean_color_shift": 0.0,
                                        "maximum_zero_mean_absolute_error": 0.0,
                                        "maximum_edge_mean_absolute_error": 0.0,
                                        "minimum_structural_correlation": 1.0,
                                    },
                                )
                                aggregate["shifts"] += 1
                                aggregate["visual_group_changes"] += int(
                                    bool(delta["visual_group_changed"])
                                )
                                aggregate["shape_changes"] += int(
                                    bool(delta["shape_changed"])
                                )
                                motion = delta["mean_absolute_error"]
                                if motion is not None:
                                    aggregate["maximum_mean_absolute_error"] = max(
                                        float(aggregate["maximum_mean_absolute_error"]),
                                        float(motion),
                                    )
                                for source, target in (
                                    ("mean_color_shift", "maximum_mean_color_shift"),
                                    (
                                        "zero_mean_absolute_error",
                                        "maximum_zero_mean_absolute_error",
                                    ),
                                    (
                                        "edge_mean_absolute_error",
                                        "maximum_edge_mean_absolute_error",
                                    ),
                                ):
                                    value = delta[source]
                                    if value is not None:
                                        aggregate[target] = max(
                                            float(aggregate[target]),
                                            float(value),
                                        )
                                correlation = delta["structural_correlation"]
                                if correlation is not None:
                                    aggregate["minimum_structural_correlation"] = min(
                                        float(
                                            aggregate[
                                                "minimum_structural_correlation"
                                            ]
                                        ),
                                        float(correlation),
                                    )
                        last_error = (
                            "card content generation changed before three-frame stability; "
                            "instability="
                            + json.dumps(
                                instability_summary(),
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                        )
                        self._increment("card_content_generation_resets")
                    elif geometry_shifted:
                        last_failure_code = "skill_card_geometry_unstable"
                        last_error = (
                            "card geometry changed before three-frame stability"
                        )
                    extension_deadline = min(
                        absolute_deadline,
                        time.monotonic() + confirmation_grace_seconds,
                    )
                    if extension_deadline > deadline:
                        deadline = extension_deadline
                        self._increment("card_stability_deadline_extensions")
                frames.append(image)
                capture_times.append(capture_started)
                for index in groups:
                    rows_by_group[index].append(observed[index])
                    content_by_group[index].append(content[index])
                    identity_by_group[index].append(identities[index])
                required_span = interval_seconds * 2
                stable_span = capture_times[-1] - capture_times[0]
                if len(frames) == 3 and stable_span + 1e-6 >= required_span:
                    stable_frames = tuple(frames)
                    self._card_content_generation_diagnostics = instability_summary()
                    self._badge_local_results.clear()
                    for index in groups:
                        self._card_rows[index] = rows_by_group[index][0]
                        self._card_images[index] = stable_frames[0]
                        self._card_count_frames[index] = stable_frames
                        self._card_identity_frames[index] = tuple(
                            identity_by_group[index]
                        )
                        # Preserve the accepted three-frame generation itself.
                        # Detail dismissal may then prove that the original
                        # source card has returned without a fixed sleep.  All
                        # three accepted signatures remain valid baselines so
                        # harmless render jitter does not create a false stop.
                        self._card_restoration_signatures[index] = tuple(
                            content_by_group[index]
                        )
                    self._add_timing(
                        "card_layout_stable_wait",
                        time.perf_counter() - started,
                    )
                    self._add_timing(
                        "card_layout_accepted_stable_span",
                        stable_span,
                    )
                    return
                if len(frames) == 3:
                    frames.pop(0)
                    capture_times.pop(0)
                    for values in rows_by_group.values():
                        values.pop(0)
                    for values in content_by_group.values():
                        values.pop(0)
                    for values in identity_by_group.values():
                        values.pop(0)
            remaining = max(0.0, next_capture_not_before - time.monotonic())
            self._add_timing(
                "card_stability_processing_overlap",
                max(0.0, interval_seconds - remaining),
            )
        self._add_timing("card_layout_stable_wait", time.perf_counter() - started)
        self._card_content_generation_diagnostics = instability_summary()
        raise ArenaReaderError(
            last_failure_code,
            f"card groups {groups!r} did not yield three consecutive stable frames: {last_error}",
        )

    def _resolve_zero_card_reference_ids(
        self,
        group_index: int,
        slot_indices: Sequence[int],
        *,
        plan: str,
    ) -> dict[int, int]:
        identity_frames = self._card_identity_frames.get(group_index, ())
        if len(identity_frames) != 3 or any(
            len(frame) != 6 for frame in identity_frames
        ):
            raise BadgeReferenceError(
                f"group {group_index} has no stable three-frame identity input"
            )
        resolved: dict[int, int] = {}
        for slot_index in slot_indices:
            references = tuple(
                frame[slot_index] for frame in identity_frames
            )
            try:
                stable_candidates = stable_reference_business_candidates(references)
                plan_candidates = self.catalog.skill_card_candidates_for_plan(
                    stable_candidates,
                    plan=plan,
                )
                embedding_candidates: tuple[int, ...] = ()
                embedding_detail_candidates: tuple[int, ...] = ()
                embedding_error: str | None = None
                fused_candidates = plan_candidates
                if len(plan_candidates) != 1:
                    try:
                        embedding_candidates = (
                            self._stable_zero_card_embedding_family_candidates(
                                group_index,
                                slot_index,
                                plan=plan,
                            )
                        )
                        embedding_detail_candidates = getattr(
                            self,
                            "_zero_card_embedding_detail_candidates",
                            {},
                        ).get(
                            (group_index, slot_index + 1),
                            embedding_candidates,
                        )
                    except BadgeReferenceError as error:
                        # The base card embedding is only an optional disambiguator for a
                        # bounded clean-reference family.  If it is unavailable,
                        # the authoritative title transaction can still resolve
                        # those candidates without guessing a business ID.
                        embedding_error = str(error)
                        self._increment(
                            "skill_card_zero_identity_embedding_title_fallbacks"
                        )
                        self._zero_card_identity_fusion_diagnostics[
                            (group_index, slot_index + 1)
                        ] = {
                            "group_index": group_index,
                            "slot": slot_index + 1,
                            "plan": plan,
                            "embedding_error": embedding_error,
                            "mode": "task075_unavailable_requires_one_title_click",
                        }
                    fused_candidates = tuple(
                        sorted(set(plan_candidates) & set(embedding_candidates))
                    )
                if len(fused_candidates) == 1:
                    resolved[slot_index] = fused_candidates[0]
                    continue
                detail_candidates = tuple(
                    dict.fromkeys(
                        (*plan_candidates, *embedding_detail_candidates)
                    )
                )
                if not detail_candidates:
                    raise BadgeReferenceError(
                        "fixed reference, stage plan, and stable embedding family "
                        "yielded no card identity candidate: "
                        f"stable={stable_candidates!r}; plan={plan!r}; "
                        f"eligible={plan_candidates!r}; "
                        f"embedding={embedding_candidates!r}"
                    )
                # A rare independent-source conflict is resolved by one
                # authoritative title click for this slot only.  It never
                # broadens into clicking every visually uncustomized card.
                self._zero_card_detail_candidates[
                    (group_index, slot_index + 1)
                ] = detail_candidates
                diagnostic = self._zero_card_identity_fusion_diagnostics.get(
                    (group_index, slot_index + 1)
                )
                if diagnostic is not None:
                    diagnostic.update(
                        {
                            "fixed_reference_candidates": list(plan_candidates),
                            "fused_candidates": list(fused_candidates),
                            "detail_candidates": list(detail_candidates),
                            "embedding_error": embedding_error,
                            "mode": "independent_identity_conflict_requires_one_title_click",
                        }
                    )
                resolved[slot_index] = 0
            except (ArenaCatalogError, BadgeReferenceError) as error:
                raise BadgeReferenceError(
                    f"slot {slot_index + 1}: {error}"
                ) from error
        return resolved

    def _stable_zero_card_embedding_family_candidates(
        self,
        group_index: int,
        slot_index: int,
        *,
        plan: str,
    ) -> tuple[int, ...]:
        """Return base-card top-family IDs stable in the same three frames.

        This is used only when the fixed clean reference remains non-unique
        after the authoritative stage-plan filter.  The embedding never
        supplies a business ID by itself: its stable top visual family must
        intersect the independent fixed-reference candidates uniquely.
        """

        from card_selection.types import CandidateBox

        frames = self._card_count_frames.get(group_index, ())
        if len(frames) != 3:
            raise BadgeReferenceError(
                f"group {group_index} has no stable three-frame embedding input"
            )
        recognizer, acceptance_threshold, minimum_margin = self._card_recognizer(
            customized=False
        )
        frame_candidates: list[set[int]] = []
        detail_candidates: set[int] = set()
        frame_diagnostics: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(frames):
            row = self._validated_card_group_row(frame, group_index)
            box = row[slot_index]
            prediction = recognizer.classify(
                frame,
                CandidateBox(
                    slot=slot_index,
                    box=box,
                    detector_label="cards",
                    detector_score=1.0,
                ),
                frame_id=frame_identifier(
                    frame,
                    300 + group_index * 10 + slot_index,
                ),
                plan=plan,
                acceptance_threshold=acceptance_threshold,
                min_margin=minimum_margin,
                resolve_upgrade_state=False,
            )
            if not prediction.accepted or not prediction.top_k_card_ids:
                raise BadgeReferenceError(
                    "the base card embedding did not yield an accepted top visual family for "
                    f"group {group_index}/slot {slot_index + 1}/frame {frame_index}: "
                    f"{prediction.reason}"
                )
            visual_families = tuple(
                tuple(dict.fromkeys(int(value) for value in family))
                for family in prediction.top_k_card_ids
            )
            eligible_families: list[tuple[int, ...]] = []
            for family in visual_families:
                try:
                    eligible_family = self.catalog.skill_card_candidates_for_plan(
                        family,
                        plan=plan,
                    )
                except ArenaCatalogError as error:
                    raise BadgeReferenceError(str(error)) from error
                eligible_families.append(eligible_family)
                detail_candidates.update(eligible_family)
            top_family = visual_families[0]
            eligible = eligible_families[0]
            frame_candidates.append(set(eligible))
            frame_diagnostics.append(
                {
                    "frame_index": frame_index,
                    "top_family": list(top_family),
                    "eligible": list(eligible),
                    "top_k_families": [list(family) for family in visual_families],
                    "top_k_eligible_families": [
                        list(family) for family in eligible_families
                    ],
                    "top_k_scores": [
                        round(float(score), 6)
                        for score in getattr(prediction, "top_k_scores", ())
                    ],
                    "confidence": round(float(prediction.confidence), 6),
                    "margin": round(float(prediction.margin), 6),
                }
            )
        stable = tuple(sorted(set.intersection(*frame_candidates)))
        stable_detail_candidates = tuple(sorted(detail_candidates))
        detail_candidate_cache = getattr(
            self,
            "_zero_card_embedding_detail_candidates",
            None,
        )
        if detail_candidate_cache is None:
            detail_candidate_cache = {}
            self._zero_card_embedding_detail_candidates = detail_candidate_cache
        detail_candidate_cache[
            (group_index, slot_index + 1)
        ] = stable_detail_candidates
        self._zero_card_identity_fusion_diagnostics[
            (group_index, slot_index + 1)
        ] = {
            "group_index": group_index,
            "slot": slot_index + 1,
            "plan": plan,
            "stable_embedding_candidates": list(stable),
            "bounded_top_k_detail_candidates": list(stable_detail_candidates),
            "frames": frame_diagnostics,
            "mode": (
                "fixed_reference_stage_plan_task075_top_family_fusion"
                if stable
                else "task075_top_family_unavailable_requires_one_title_click"
            ),
        }
        if stable:
            self._increment("skill_card_zero_identity_embedding_fusions")
        else:
            self._increment("skill_card_zero_identity_topk_title_fallbacks")
        return stable

    def _refresh_visible_card_groups(self, requested_group: int) -> tuple[Any, Any, Any]:
        self._increment("corrective_card_frame_reads")
        groups = tuple(
            index for index in (0, 1) if len(self._card_rows.get(index, ())) == 6
        )
        if requested_group not in groups:
            groups = (requested_group,)
        self._capture_stable_card_groups(groups)
        frames = self._card_count_frames.get(requested_group, ())
        if len(frames) != 3:
            raise ArenaReaderError(
                "skill_card_layout_unstable",
                f"group {requested_group} was not cached after a stable refresh",
            )
        return frames

    def open_skill_card(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        expected_customization_count: int,
    ) -> None:
        del target, stage_number, member_slot
        row = self._card_rows.get(group_index, ())
        if not 1 <= card_slot <= len(row):
            raise ArenaReaderError("skill_card_slot_missing", f"group {group_index}/slot {card_slot} is absent")
        duplicate_flags = self._card_excluded_duplicate_flags.get(group_index, ())
        if len(duplicate_flags) == 6 and duplicate_flags[card_slot - 1]:
            raise ArenaReaderError(
                "skill_card_excluded_duplicate",
                f"group {group_index}/slot {card_slot} is a non-interactive excluded duplicate",
            )
        key = (group_index, card_slot)
        inferred = self._inferred_clicked_cards.get(key)
        if inferred is not None:
            inferred_count = sum(int(value) for value in inferred.customizations.values())
            if inferred_count != expected_customization_count:
                raise ArenaReaderError(
                    "skill_card_inferred_count_changed",
                    f"group {group_index}/slot {card_slot} inferred {inferred_count} "
                    f"but the reader requested {expected_customization_count}",
                )
            self._active_inferred_clicked_card = key
            return
        candidate_ids = self._card_candidate_groups.get((group_index, card_slot))
        if candidate_ids is None:
            raise ArenaReaderError("skill_card_prediction_missing", "card icon was not classified before its click")
        card_box = row[card_slot - 1]
        primary_click_box = self._skill_card_interaction_box(card_box)
        retry_click_box = self._skill_card_retry_interaction_box(card_box)
        last_text = ""
        last_error = ""
        transaction_started = time.perf_counter()
        for attempt in range(2):
            click_box = primary_click_box if attempt == 0 else retry_click_box
            self._increment("skill_card_detail_clicks")
            self._increment("skill_card_safe_region_clicks")
            if attempt:
                self._increment("skill_card_detail_click_retries")
                self._increment("skill_card_detail_contact_retries")
            if attempt == 0:
                self._short_press(click_box, duration_ms=80)
            else:
                self._short_press(click_box, duration_ms=120)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                try:
                    detail_image = self._capture()
                    last_text = self._skill_card_detail_ocr_text(
                        detail_image,
                        phase="open",
                    )
                except ArenaReaderError:
                    last_text = ""
                try:
                    self.catalog.confirm_clicked_skill_card_candidates(
                        last_text,
                        candidate_card_ids=candidate_ids,
                    )
                    self._card_detail_texts[key] = last_text
                    self._card_detail_images[key] = detail_image
                    self._card_transaction_started[key] = transaction_started
                    self._card_transaction_kinds[key] = (
                        "necessary_skill_card_detail_transaction"
                    )
                    return
                except ArenaCatalogError as error:
                    last_error = str(error)
                try:
                    self.catalog.confirm_clicked_customizable_skill_card_without_badge_count(
                        last_text
                    )
                    self._card_detail_texts[key] = last_text
                    self._card_detail_images[key] = detail_image
                    self._card_transaction_started[key] = transaction_started
                    self._card_transaction_kinds[key] = (
                        "necessary_skill_card_detail_transaction"
                    )
                    return
                except ArenaCatalogError as error:
                    last_error = str(error)
                    try:
                        self.catalog.confirm_clicked_customizable_skill_card(
                            last_text,
                            expected_count=expected_customization_count,
                        )
                        self._card_detail_texts[key] = last_text
                        self._card_detail_images[key] = detail_image
                        self._card_transaction_started[key] = transaction_started
                        self._card_transaction_kinds[key] = (
                            "necessary_skill_card_detail_transaction"
                        )
                        return
                    except ArenaCatalogError as constrained_error:
                        last_error = str(constrained_error)
                time.sleep(0.25)
            if attempt == 0:
                # Retry only when the unchanged member-detail card grid proves
                # the first click never opened an overlay.  An unknown page or
                # a wrong overlay remains fail-closed.
                try:
                    self._assert_card_group_visible(
                        group_index,
                        card_slot=card_slot,
                    )
                except ArenaReaderError:
                    break
        try:
            self._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
            )
        except ArenaReaderError:
            terminal_code = "skill_card_detail_missing"
        else:
            # The game contract is binary on this page: ordinary cards open
            # from their body, while an excluded duplicate is non-interactive.
            # Only promote that functional evidence after one instantaneous
            # click, one bounded contact retry, and a still-valid source grid.
            terminal_code = "skill_card_detail_noninteractive"
        raise ArenaReaderError(
            terminal_code,
            f"clicked card did not confirm visual-family IDs {candidate_ids!r} "
            "after one bounded retry: "
            f"{last_error}; card_box={card_box!r}; "
            f"primary_click_box={primary_click_box!r}; "
            f"retry_click_box={retry_click_box!r}; "
            f"raw_boxes={self._raw_card_candidate_boxes(self._card_images[group_index])!r}; "
            f"OCR={last_text[:400]!r}",
        )

    @staticmethod
    def _skill_card_interaction_box(
        card_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Return the deterministic, unobscured card-body click anchor.

        Maa treats the supplied rectangle as an action target rather than a
        promise to use its centre.  A broad rectangle can therefore land on
        the cost, type, or ``+`` overlays.  Live evidence showed that the old
        upper-art point could leave the source grid unchanged while this
        central body point opened the same cards on the bounded retry.  Make
        that proven point the first contact without widening the hit region.
        """

        x, y, width, height = card_box
        if width < 8 or height < 8:
            raise ArenaReaderError(
                "skill_card_interaction_box_invalid",
                f"card box is too small for a stable interaction region: {card_box!r}",
            )
        return (
            x + int(round(width * 0.48)),
            y + int(round(height * 0.48)),
            1,
            1,
        )

    @staticmethod
    def _skill_card_retry_interaction_box(
        card_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Return the same safe body point for one longer-contact retry."""

        x, y, width, height = card_box
        if width < 8 or height < 8:
            raise ArenaReaderError(
                "skill_card_interaction_box_invalid",
                f"card box is too small for a stable interaction region: {card_box!r}",
            )
        return (
            x + int(round(width * 0.48)),
            y + int(round(height * 0.48)),
            1,
            1,
        )

    def _infer_badge_card_from_detail(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        image: Any,
        box: tuple[int, int, int, int],
        expected_customization_count: int | None,
        detail_kind: str,
        *,
        allow_zero_without_badge_count: bool = False,
    ) -> ClickedSkillCard:
        """Resolve one seeded-plate shortlist candidate from clicked detail."""

        candidate_ids = self._card_visual_family_candidates(
            stage_number=stage_number,
            image=image,
            box=box,
            slot_index=card_slot - 1,
        )
        key = (group_index, card_slot)
        self._card_candidate_groups[key] = candidate_ids
        overlay_opened = False
        try:
            self.open_skill_card(
                target,
                stage_number,
                member_slot,
                group_index,
                card_slot,
                expected_customization_count or 1,
            )
            overlay_opened = True
            self._card_transaction_kinds[key] = detail_kind
            inferred = self._read_resolved_card_detail(
                key,
                candidate_ids,
                expected_customization_count=(
                    None
                    if allow_zero_without_badge_count
                    else expected_customization_count
                ),
                allow_zero_without_badge_count=allow_zero_without_badge_count,
            )
        except ArenaReaderError as error:
            if error.code == "skill_card_detail_noninteractive":
                raise
            raise ArenaReaderError(
                "skill_card_badge_detail_inference_ambiguous",
                f"group {group_index}/slot {card_slot}: {error}",
            ) from error
        finally:
            if overlay_opened:
                self._dismiss_skill_card_detail()
        self._assert_card_group_visible(
            group_index,
            card_slot=card_slot,
            expected_card_id=inferred.card_id,
        )
        self._finish_card_transaction(key)
        self._inferred_clicked_cards[key] = inferred
        self._card_predictions[key] = inferred.card_id
        return inferred

    def _infer_zero_card_identity_from_detail(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        candidate_ids: Sequence[int],
    ) -> ClickedSkillCard:
        """Resolve one rare fixed-reference/embedding conflict by title."""

        key = (group_index, card_slot)
        self._card_candidate_groups[key] = tuple(candidate_ids)
        overlay_opened = False
        try:
            self.open_skill_card(
                target,
                stage_number,
                member_slot,
                group_index,
                card_slot,
                0,
            )
            overlay_opened = True
            self._card_transaction_kinds[key] = (
                "zero_identity_detail_transaction"
            )
            inferred = self._read_resolved_card_detail(
                key,
                candidate_ids,
                expected_customization_count=None,
                allow_zero_without_badge_count=True,
            )
            if inferred.customizations:
                raise ArenaReaderError(
                    "skill_card_zero_identity_detail_positive",
                    f"group {group_index}/slot {card_slot} was visually zero but "
                    f"detail resolved customizations {dict(inferred.customizations)!r}",
                )
        finally:
            if overlay_opened:
                self._dismiss_skill_card_detail()
        self._assert_card_group_visible(
            group_index,
            card_slot=card_slot,
            expected_card_id=inferred.card_id,
        )
        self._finish_card_transaction(key)
        self._inferred_clicked_cards[key] = inferred
        self._card_predictions[key] = inferred.card_id
        self._increment("skill_card_zero_identity_detail_resolutions")
        return inferred

    def read_skill_card_id_hints(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        customization_counts: Sequence[int],
        excluded_duplicate_flags: Sequence[bool] | None = None,
    ) -> Sequence[int]:
        row = self._card_rows.get(group_index, ())
        image = self._card_images.get(group_index)
        if len(row) != 6 or image is None or len(customization_counts) != 6:
            raise ArenaReaderError(
                "skill_card_identity_input_mismatch",
                f"group {group_index} is missing its six boxes, frame, or customization counts",
            )
        if excluded_duplicate_flags is None:
            excluded_duplicate_flags = (False,) * 6
        if len(excluded_duplicate_flags) != 6:
            raise ArenaReaderError(
                "skill_card_excluded_duplicate_count_mismatch",
                f"group {group_index} yielded duplicate flags {excluded_duplicate_flags!r}",
            )
        empty_flags = getattr(self, "_card_empty_flags", {}).get(
            group_index,
            (False,) * 6,
        )
        if len(empty_flags) != 6:
            raise ArenaReaderError(
                "skill_card_empty_slot_count_mismatch",
                f"group {group_index} yielded empty flags {empty_flags!r}",
            )
        plan = self.season.stages[stage_number - 1].plan
        predictions: list[int] = []
        from card_selection.types import CandidateBox

        zero_slot_indices = tuple(
            index
            for index, (customization_count, excluded, empty) in enumerate(
                zip(
                    customization_counts,
                    excluded_duplicate_flags,
                    empty_flags,
                    strict=True,
                )
            )
            if not excluded
            and not empty
            and customization_count == 0
            and self._inferred_clicked_cards.get((group_index, index + 1)) is None
        )
        zero_reference_ids: dict[int, int] = {}
        if zero_slot_indices:
            try:
                zero_reference_ids = self._resolve_zero_card_reference_ids(
                    group_index,
                    zero_slot_indices,
                    plan=plan,
                )
            except BadgeReferenceError as first_error:
                # A geometry-complete frame generation can still become stale
                # between badge and identity phases.  Re-acquire the complete
                # visible generation once; never vote or introduce an alias.
                try:
                    self._refresh_visible_card_groups(group_index)
                    self._increment("skill_card_identity_generation_refreshes")
                    zero_reference_ids = self._resolve_zero_card_reference_ids(
                        group_index,
                        zero_slot_indices,
                        plan=plan,
                    )
                except (ArenaReaderError, BadgeReferenceError) as retry_error:
                    raise ArenaReaderError(
                        "skill_card_reference_identity_ambiguous",
                        f"group {group_index} initial generation failed: {first_error}; "
                        f"fresh generation failed: {retry_error}",
                    ) from retry_error
                row = self._card_rows.get(group_index, ())
                image = self._card_images.get(group_index)
                if len(row) != 6 or image is None:
                    raise ArenaReaderError(
                        "skill_card_identity_input_mismatch",
                        f"group {group_index} lost its frame after identity refresh",
                    )

        for index, (box, customization_count) in enumerate(
            zip(row, customization_counts, strict=True),
            start=1,
        ):
            if excluded_duplicate_flags[index - 1] or empty_flags[index - 1]:
                self._card_candidate_groups.pop((group_index, index), None)
                self._card_predictions[(group_index, index)] = 0
                predictions.append(0)
                continue
            key = (group_index, index)
            inferred = self._inferred_clicked_cards.get(key)
            if inferred is not None:
                inferred_count = sum(
                    int(value) for value in inferred.customizations.values()
                )
                if inferred_count != customization_count:
                    raise ArenaReaderError(
                        "skill_card_inferred_count_changed",
                        f"group {group_index}/slot {index} inferred {inferred_count} "
                        f"but identity read requested {customization_count}",
                    )
                self._card_candidate_groups[key] = (inferred.card_id,)
                self._card_predictions[key] = inferred.card_id
                predictions.append(inferred.card_id)
                continue
            customized = customization_count > 0
            if customized:
                # The arena custom-card embedding supplies only bounded visual-family candidates.  A
                # positive card's mandatory detail title remains authoritative.
                recognizer, acceptance_threshold, minimum_margin = self._card_recognizer(
                    customized=True
                )
                prediction = recognizer.classify(
                    image,
                    CandidateBox(
                        slot=index - 1,
                        box=box,
                        detector_label="cards",
                        detector_score=1.0,
                    ),
                    frame_id=frame_identifier(image, index),
                    plan=plan,
                    acceptance_threshold=acceptance_threshold,
                    min_margin=minimum_margin,
                    resolve_upgrade_state=False,
                )
                if not prediction.accepted or not prediction.card_id.isdigit():
                    raise ArenaReaderError(
                        "skill_card_icon_unknown",
                        f"group {group_index}/slot {index} was rejected: {prediction.reason}",
                    )
                candidate_ids = tuple(
                    dict.fromkeys(
                        int(value)
                        for visual_family in prediction.top_k_card_ids
                        for value in visual_family
                    )
                )
                if not candidate_ids:
                    raise ArenaReaderError(
                        "skill_card_icon_unknown",
                        f"group {group_index}/slot {index} has no visual-family candidates",
                    )
                resolved_card_id = int(prediction.card_id)
            else:
                # Zero-card identities were resolved as one page generation
                # before the loop.  This prevents one ambiguous slot from
                # mixing fresh frames with already accepted sibling slots.
                resolved_card_id = zero_reference_ids[index - 1]
                if resolved_card_id:
                    candidate_ids = (resolved_card_id,)
                    self._increment("skill_card_exact_reference_resolutions")
                else:
                    candidate_ids = self._zero_card_detail_candidates.get(key, ())
                    if not candidate_ids:
                        raise ArenaReaderError(
                            "skill_card_zero_identity_candidates_missing",
                            f"group {group_index}/slot {index} has no bounded title candidates",
                        )
                    inferred = self._infer_zero_card_identity_from_detail(
                        target,
                        stage_number,
                        member_slot,
                        group_index,
                        index,
                        candidate_ids,
                    )
                    resolved_card_id = inferred.card_id
                    candidate_ids = (resolved_card_id,)
            self._card_candidate_groups[key] = candidate_ids
            self._card_predictions[key] = resolved_card_id
            predictions.append(resolved_card_id)
        if group_index == 1 and tuple(predictions) == tuple(
            self._card_predictions.get((0, index)) for index in range(1, 7)
        ):
            raise ArenaReaderError(
                "skill_card_group_unchanged",
                "upper-detail swipe did not expose a distinct second six-card group",
            )
        return tuple(predictions)

    def _targeted_duplicate_marker_observations(
        self,
        frames: Sequence[Any],
        row: Sequence[tuple[int, int, int, int]],
        target_slots: Sequence[int],
        *,
        phase: str,
    ) -> tuple[tuple[tuple[bool, ...], ...], list[dict[str, Any]]]:
        import cv2

        observations = []
        diagnostics = []
        targets = set(int(index) for index in target_slots)
        for image in frames:
            height, width = image.shape[:2]
            flags = [False] * len(row)
            frame_diagnostics = []
            for index in sorted(targets):
                card_x, card_y, card_width, card_height = row[index]
                left = max(0, int(round(card_x - card_width * 0.10)))
                top = max(0, int(round(card_y - card_height * 0.10)))
                right = min(width, int(round(card_x + card_width * 1.10)))
                # ``重複`` can follow a lower card-face value rather than sit
                # in the historical upper-half position.  Keep the horizontal
                # ROI slot-local, but include the complete card plus a bounded
                # lower margin so that marker text cannot escape the OCR crop.
                bottom = min(height, int(round(card_y + card_height * 1.10)))
                crop = image[top:bottom, left:right]
                card_crop = image[
                    card_y : card_y + card_height,
                    card_x : card_x + card_width,
                    :3,
                ]
                visual_features = duplicate_marker_visual_features(card_crop)
                texts_by_variant: dict[str, list[str]] = {}
                if crop.size:
                    enlarged = cv2.resize(
                        crop,
                        None,
                        fx=3.0,
                        fy=3.0,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    texts = [_text(item) for item in self._ocr(enlarged, r".+")]
                    texts_by_variant["3x"] = texts
                    if any(
                        marker in text
                        for text in texts
                        for marker in ("重複", "制限")
                    ):
                        flags[index] = True
                frame_diagnostic = {
                        "slot": index + 1,
                        "crop_box": [left, top, right - left, bottom - top],
                        "texts": texts_by_variant,
                        "flag": flags[index],
                        "visual_features": visual_features,
                    }
                frame_diagnostics.append(frame_diagnostic)
            observations.append(tuple(flags))
            diagnostics.append(
                {
                    "phase": phase,
                    "targeted": frame_diagnostics,
                    "flags": list(flags),
                }
            )
        return tuple(observations), diagnostics

    def read_skill_card_excluded_duplicate_flags(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[bool]:
        del target, stage_number, member_slot
        row = self._card_rows.get(group_index, ())
        frames = self._card_count_frames.get(group_index, ())
        if len(row) != 6:
            raise ArenaReaderError(
                "skill_card_duplicate_marker_input_mismatch",
                f"group {group_index} has no stable six-card row",
            )
        if len(frames) != 3:
            frames = self._refresh_visible_card_groups(group_index)
            frame_rows = tuple(
                self._validated_card_group_row(image, group_index) for image in frames
            )
            if any(
                abs(actual - expected) > 3
                for observed_row in frame_rows
                for actual_box, expected_box in zip(observed_row, row, strict=True)
                for actual, expected in zip(actual_box, expected_box, strict=True)
            ):
                raise ArenaReaderError(
                    "skill_card_duplicate_marker_frame_shifted",
                    f"group {group_index} card geometry changed during duplicate-marker read",
                )
            self._card_count_frames[group_index] = frames

        repeated_visual_features = tuple(
            tuple(
                duplicate_marker_visual_features(
                    image[y : y + height, x : x + width, :3]
                )
                for image in frames
            )
            for x, y, width, height in row
        )
        empty_flags = getattr(self, "_card_empty_flags", {}).get(
            group_index,
            (False,) * 6,
        )
        candidate_slots = tuple(
            index
            for index, features in enumerate(repeated_visual_features)
            if not empty_flags[index]
            and is_duplicate_marker_visual_candidate(features)
        )
        visual_diagnostic = {
            "phase": "local_visual_candidates",
            "candidate_slots": [index + 1 for index in candidate_slots],
            "slots": [
                {
                    "slot": index + 1,
                    "candidate": index in candidate_slots,
                    "features": list(features),
                }
                for index, features in enumerate(repeated_visual_features)
            ],
        }
        targeted, diagnostics = self._targeted_duplicate_marker_observations(
            frames,
            row,
            candidate_slots,
            phase="targeted",
        )
        diagnostics.insert(0, visual_diagnostic)
        stable = stable_excluded_duplicate_card_flags(targeted)
        unresolved = tuple(index for index, flag in enumerate(stable) if flag is None)
        if unresolved:
            corrective_frames_tuple = self._refresh_visible_card_groups(group_index)
            corrective_rows = tuple(
                self._validated_card_group_row(image, group_index)
                for image in corrective_frames_tuple
            )
            if any(
                abs(actual - expected) > 3
                for observed_row in corrective_rows
                for actual_box, expected_box in zip(observed_row, row, strict=True)
                for actual, expected in zip(actual_box, expected_box, strict=True)
            ):
                raise ArenaReaderError(
                    "skill_card_duplicate_marker_frame_shifted",
                    f"group {group_index} card geometry changed during corrective duplicate-marker read",
                )
            corrective_targeted, corrective_targeted_diagnostics = (
                self._targeted_duplicate_marker_observations(
                    corrective_frames_tuple,
                    row,
                    unresolved,
                    phase="corrective_targeted",
                )
            )
            diagnostics.extend(corrective_targeted_diagnostics)
            stable = stable_excluded_duplicate_card_flags(corrective_targeted)
            self._card_count_frames[group_index] = corrective_frames_tuple
        unknown_slots = tuple(index for index, flag in enumerate(stable, start=1) if flag is None)
        self._card_duplicate_marker_diagnostics[group_index] = tuple(diagnostics)
        if unknown_slots:
            raise ArenaReaderError(
                "skill_card_duplicate_marker_unstable",
                f"group {group_index}/slots {unknown_slots} remained ambiguous after one "
                "targeted and one corrective 重複 read",
            )
        flags = tuple(bool(flag) for flag in stable)
        self._card_excluded_duplicate_flags[group_index] = flags
        return flags

    def current_skill_card_excluded_duplicate_flags(
        self,
        group_index: int,
    ) -> Sequence[bool]:
        """Return OCR flags plus any functionally proven non-interactive slot."""

        flags = self._card_excluded_duplicate_flags.get(group_index)
        if flags is None or len(flags) != 6:
            raise ArenaReaderError(
                "skill_card_duplicate_marker_input_mismatch",
                f"group {group_index} has no current six-slot duplicate state",
            )
        return flags

    def read_skill_card_customization_counts(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[int]:
        row = self._card_rows.get(group_index, ())
        prepared = self._card_images.get(group_index)
        if len(row) != 6 or prepared is None:
            raise ArenaReaderError(
                "skill_card_customization_count_input_mismatch",
                f"group {group_index} is missing its six boxes or prepared frame",
            )
        cached_frames = self._card_count_frames.get(group_index, ())
        if len(cached_frames) == 3:
            frame_samples = list(cached_frames)
        else:
            frame_samples = [prepared]
            for _ in range(2):
                time.sleep(0.12)
                frame_samples.append(self._capture())
        frames = tuple(frame_samples)
        for attempt in range(2):
            try:
                frame_rows = tuple(
                    self._validated_card_group_row(image, group_index) for image in frames
                )
            except ArenaReaderError as error:
                if error.code != "skill_card_customization_frame_incomplete" or attempt == 1:
                    raise
                frames = self._refresh_visible_card_groups(group_index)
                continue
            baseline = row if attempt == 0 else frame_rows[0]
            shifted = any(
                abs(actual - expected) > 3
                for observed_row in frame_rows
                for actual_box, expected_box in zip(observed_row, baseline, strict=True)
                for actual, expected in zip(actual_box, expected_box, strict=True)
            )
            if shifted:
                if attempt == 1:
                    raise ArenaReaderError(
                        "skill_card_customization_frame_shifted",
                        f"group {group_index} card geometry changed after one corrective reread",
                    )
                frames = self._refresh_visible_card_groups(group_index)
                continue
            duplicate_flags = self._card_excluded_duplicate_flags.get(
                group_index,
                (False,) * 6,
            )
            empty_flags = getattr(self, "_card_empty_flags", {}).get(
                group_index,
                (False,) * 6,
            )
            local_results = self._batched_badge_local_results(group_index)
            decisions = []
            count_diagnostics = []
            for slot_index, (local_result, excluded, empty) in enumerate(
                zip(local_results, duplicate_flags, empty_flags, strict=True)
            ):
                if excluded or empty:
                    reason = "excluded_duplicate" if excluded else "empty_slot"
                    decisions.append(
                        CustomizationBadgeDecision(
                            state=CustomizationBadgeState.CONFIDENT_ZERO,
                            count=0,
                            reason=reason,
                        )
                    )
                    count_diagnostics.append(
                        {
                            "slot": slot_index + 1,
                            "local_state": CustomizationBadgeState.CONFIDENT_ZERO.value,
                            "local_reason": reason,
                            "ocr_fallback": False,
                            "resolution_path": reason,
                        }
                    )
                    continue
                assert local_result is not None
                local_decision = local_result.decision
                if (
                    local_decision.reason
                    == "all_frames_oversized_center_foreground_contradiction"
                ):
                    self._increment("skill_card_badge_oversized_art_strong_zeros")
                badge_observations = local_result.observations
                plate_diagnostics = []
                for observation in badge_observations:
                    internal = observation.get("internal_features")
                    if not isinstance(internal, dict):
                        internal = {}
                    plate_diagnostics.append(
                        {
                            key: internal.get(key)
                            for key in (
                                "peak_center",
                                "badge_center",
                                "badge_hull_points",
                                "badge_interior_area",
                                "center_seed_component_count",
                                "center_seed_ring_support",
                                "discarded_outer_component_pixels",
                                "center_component_green_fill",
                                "center_core_green_fill",
                                "center_component_extent_ratio",
                                "seeded_plate_candidate",
                                "glyph_component_count",
                                "glyph_component_box",
                                "glyph_component_area",
                                "glyph_foreground_pixel_count",
                                "glyph_raw_components",
                                "glyph_non_badge_art_candidate",
                                "glyph_descriptor_12x18_q4",
                            )
                            if key in internal
                        }
                    )
                decisions.append(local_decision)
                count_diagnostics.append(
                    {
                        "slot": slot_index + 1,
                        "local_state": local_decision.state.value,
                        "local_reason": local_decision.reason,
                        "resolution_path": (
                            "fail_closed"
                            if not self._badge_detail_inferable(local_decision)
                            else "badge_candidate_detail"
                        ),
                        "plate_diagnostics": plate_diagnostics,
                    }
                )
            decisions = tuple(decisions)
            counts = [decision.count for decision in decisions]
            for index, count in enumerate(counts, start=1):
                if count is not None:
                    continue
                inferred = self._inferred_clicked_cards.get((group_index, index))
                if inferred is None:
                    continue
                inferred_count = sum(
                    int(value) for value in inferred.customizations.values()
                )
                if inferred_count < 1:
                    raise ArenaReaderError(
                        "skill_card_cached_detail_inference_empty",
                        f"group {group_index}/slot {index} cached no positive customization",
                    )
                counts[index - 1] = inferred_count
                count_diagnostics[index - 1].update(
                    {
                        "fallback_state": "POSITIVE",
                        "fallback_reason": "preswipe_detail_unique_positive_count",
                        "fallback_mode": "preswipe_clicked_detail_cache",
                        "inferred_card_id": inferred.card_id,
                        "inferred_customizations": dict(inferred.customizations),
                    }
                )
            counts = tuple(counts)
            unknown_slots = tuple(
                index for index, count in enumerate(counts, start=1) if count is None
            )
            self._badge_count_diagnostics[group_index] = tuple(count_diagnostics)
            if unknown_slots:
                inferable = tuple(
                    index
                    for index in unknown_slots
                    if self._badge_detail_inferable(decisions[index - 1])
                )
                if inferable == unknown_slots:
                    inferred_counts = list(counts)
                    for index in inferable:
                        reason = decisions[index - 1].reason
                        detail_check = self._record_badge_candidate_detail(
                            stage_number=stage_number,
                            member_slot=member_slot,
                            group_index=group_index,
                            card_slot=index,
                            phase="post_swipe_card_group",
                            reason=reason,
                        )
                        try:
                            inferred = self._infer_badge_card_from_detail(
                                target,
                                stage_number,
                                member_slot,
                                group_index,
                                index,
                                frames[0],
                                frame_rows[0][index - 1],
                                None,
                                "badge_candidate_detail_transaction",
                                allow_zero_without_badge_count=True,
                            )
                        except ArenaReaderError as error:
                            if error.code == "skill_card_detail_noninteractive":
                                current_flags = list(
                                    self._card_excluded_duplicate_flags.get(
                                        group_index,
                                        (False,) * 6,
                                    )
                                )
                                current_flags[index - 1] = True
                                self._card_excluded_duplicate_flags[group_index] = tuple(
                                    current_flags
                                )
                                inferred_counts[index - 1] = 0
                                detail_check.update(
                                    {
                                        "resolved_state": "EXCLUDED_DUPLICATE",
                                        "functional_evidence": (
                                            "click_and_bounded_press_noninteractive"
                                        ),
                                    }
                                )
                                count_diagnostics[index - 1].update(
                                    {
                                        "fallback_state": "EXCLUDED_DUPLICATE",
                                        "fallback_reason": (
                                            "stable_card_body_noninteractive"
                                        ),
                                        "fallback_mode": (
                                            "functional_duplicate_evidence"
                                        ),
                                    }
                                )
                                self._increment(
                                    "skill_card_noninteractive_duplicate_fallbacks"
                                )
                                continue
                            raise ArenaReaderError(
                                error.code,
                                f"{error.detail}; badge_diagnostic="
                                f"{count_diagnostics[index - 1]!r}",
                            ) from error
                        self._finish_badge_candidate_detail(detail_check, inferred)
                        inferred_count = sum(
                            int(value) for value in inferred.customizations.values()
                        )
                        inferred_counts[index - 1] = inferred_count
                        count_diagnostics[index - 1].update(
                            {
                                "fallback_state": (
                                    "POSITIVE"
                                    if inferred_count > 0
                                    else CustomizationBadgeState.CONFIDENT_ZERO.value
                                ),
                                "fallback_reason": (
                                    "clicked_detail_unique_positive_count"
                                    if inferred_count > 0
                                    else "clicked_detail_unique_zero"
                                ),
                                "fallback_mode": "clicked_detail_unique_state",
                                "inferred_card_id": inferred.card_id,
                                "inferred_customizations": dict(inferred.customizations),
                            }
                        )
                    self._badge_count_diagnostics[group_index] = tuple(count_diagnostics)
                    self._card_rows[group_index] = baseline
                    self._card_images[group_index] = frames[0]
                    self._card_count_frames[group_index] = frames
                    return tuple(int(count) for count in inferred_counts)
                if attempt == 0:
                    frames = self._refresh_visible_card_groups(group_index)
                    continue
                raise ArenaReaderError(
                    "skill_card_customization_count_unknown",
                    f"group {group_index}/slots {unknown_slots} did not produce three consistent "
                    "badge reads after one corrective reread; "
                    f"reasons={tuple(decisions[index - 1].reason for index in unknown_slots)!r}; "
                    f"diagnostics={tuple(count_diagnostics[index - 1] for index in unknown_slots)!r}",
                )
            row = baseline
            self._card_rows[group_index] = row
            self._card_images[group_index] = frames[0]
            self._card_count_frames[group_index] = frames
            return tuple(int(count) for count in counts)
        raise AssertionError("customization count retry loop exhausted without a terminal result")

    def read_clicked_skill_card(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        expected_customization_count: int,
    ) -> ClickedSkillCard:
        del target, stage_number, member_slot
        key = (group_index, card_slot)
        inferred = self._inferred_clicked_cards.get(key)
        if self._active_inferred_clicked_card == key and inferred is not None:
            inferred_count = sum(int(value) for value in inferred.customizations.values())
            if inferred_count != expected_customization_count:
                raise ArenaReaderError(
                    "skill_card_inferred_count_changed",
                    f"group {group_index}/slot {card_slot} inferred {inferred_count} "
                    f"but detail read requested {expected_customization_count}",
                )
            return inferred
        candidate_ids = self._card_candidate_groups.get((group_index, card_slot))
        if candidate_ids is None:
            raise ArenaReaderError("skill_card_prediction_missing", "card icon was not classified before its click")
        resolved = self._read_resolved_card_detail(
            key,
            candidate_ids,
            expected_customization_count=expected_customization_count,
        )
        # The embedding result is only a pre-click visual-family hint.  Once
        # the detail title/effects resolve the business ID, bind that
        # authoritative identity before the close transaction verifies the
        # restored source slot.  Otherwise a valid family mismatch (for
        # example hint 752 resolving to actual card 747) is misreported as a
        # failed close even though the original grid returned unchanged.
        self._card_predictions[key] = resolved.card_id
        return resolved

    def _resolve_clicked_card_text(
        self,
        candidate_ids: Sequence[int],
        text: str,
        *,
        expected_customization_count: int | None,
        allow_zero_without_badge_count: bool = False,
        allow_auxiliary_badge_glyph: bool = True,
        key: tuple[int, int] | None = None,
    ) -> ClickedSkillCard:
        try:
            try:
                card_id = self.catalog.confirm_clicked_skill_card_candidates(
                    text,
                    candidate_card_ids=candidate_ids,
                )
            except ArenaCatalogError:
                try:
                    card_id = (
                        self.catalog.confirm_clicked_customizable_skill_card_without_badge_count(
                            text
                        )
                    )
                except ArenaCatalogError:
                    if expected_customization_count is None:
                        raise
                    card_id = self.catalog.confirm_clicked_customizable_skill_card(
                        text,
                        expected_count=expected_customization_count,
                    )
            generic_cost_frame_values = (
                self._measure_optional_card_face_generic_cost(key, card_id)
                if key is not None
                else None
            )
            if expected_customization_count is None:
                detail_evidence_mode = "unconstrained"
                customizations = None
                if generic_cost_frame_values is not None:
                    try:
                        customizations = (
                            self.catalog.resolve_effective_customizations_with_generic_cost_evidence(
                                card_id,
                                text,
                                generic_cost_frame_values=generic_cost_frame_values,
                            )
                        )
                    except ArenaCatalogError:
                        # A presence shortlist may still be an uncustomized
                        # card whose green art overlaps the semantic locus.
                        # Positive-only detail+cost evidence therefore gets
                        # first refusal, not authority to reject the later
                        # zero-inclusive catalog path.
                        customizations = None
                    else:
                        resolution_source = "detail_card_face_unique"
                        detail_evidence_mode = (
                            self.catalog.effective_customization_evidence_mode(
                                card_id,
                                text,
                                expected_count=sum(
                                    int(value) for value in customizations.values()
                                ),
                                resolved=customizations,
                            )
                        )
                if customizations is None:
                    resolver = (
                        self.catalog.resolve_effective_customizations_unconstrained
                        if allow_zero_without_badge_count
                        else self.catalog.resolve_effective_customizations_without_badge_count
                    )
                    try:
                        customizations = resolver(card_id, text)
                    except ArenaCatalogError:
                        if not allow_zero_without_badge_count or key is None:
                            raise
                        if not allow_auxiliary_badge_glyph:
                            self._increment(
                                "skill_card_detail_auxiliary_glyph_deferred"
                            )
                            raise
                        try:
                            auxiliary_count = self._auxiliary_badge_glyph_count(key)
                        except ArenaReaderError as error:
                            cost_error = getattr(
                                self,
                                "_card_face_cost_optional_errors",
                                {},
                            ).get(key)
                            if cost_error is None:
                                raise
                            raise ArenaReaderError(
                                error.code,
                                f"{error.detail}; card_face_cost={cost_error}",
                            ) from error
                        if auxiliary_count == 0:
                            customizations = self.catalog.resolve_effective_customizations(
                                card_id,
                                text,
                                expected_count=0,
                                allow_positive_completion=False,
                            )
                            if customizations:
                                raise ArenaCatalogError(
                                    "stable non-badge card art did not resolve a zero state"
                                )
                            resolution_source = (
                                "detail_zero_auxiliary_same_plate_non_badge_art"
                            )
                            detail_evidence_mode = (
                                "stable_same_plate_non_badge_art_zero"
                            )
                            resolution = None
                        else:
                            resolution_kwargs = {
                                "observed_badge_count": auxiliary_count,
                            }
                            if generic_cost_frame_values is not None:
                                resolution_kwargs["generic_cost_frame_values"] = (
                                    generic_cost_frame_values
                                )
                            resolution = self.catalog.resolve_clicked_customizations(
                                card_id,
                                text,
                                **resolution_kwargs,
                            )
                        if resolution is not None:
                            customizations = resolution.customizations
                            resolution_source = (
                                f"{resolution.source}_auxiliary_badge_glyph"
                            )
                            detail_evidence_mode = getattr(
                                resolution,
                                "evidence_mode",
                                "negative_dependent",
                            )
                    else:
                        resolution_source = "detail_unconstrained"
                        resolved_count = sum(
                            int(value) for value in customizations.values()
                        )
                        if resolved_count:
                            detail_evidence_mode = (
                                self.catalog.effective_customization_evidence_mode(
                                    card_id,
                                    text,
                                    expected_count=resolved_count,
                                    resolved=customizations,
                                )
                            )
            else:
                resolution_kwargs: dict[str, Any] = {
                    "observed_badge_count": expected_customization_count,
                }
                if generic_cost_frame_values is not None:
                    resolution_kwargs["generic_cost_frame_values"] = (
                        generic_cost_frame_values
                    )
                resolution = self.catalog.resolve_clicked_customizations(
                    card_id,
                    text,
                    **resolution_kwargs,
                )
                customizations = resolution.customizations
                resolution_source = resolution.source
                detail_evidence_mode = getattr(
                    resolution,
                    "evidence_mode",
                    "negative_dependent",
                )
        except ArenaCatalogError as error:
            raise ArenaReaderError("skill_card_detail_ambiguous", str(error)) from error
        return ClickedSkillCard(
            card_id,
            customizations,
            resolution_source=resolution_source,
            detail_evidence_mode=detail_evidence_mode,
        )

    def read_clicked_skill_card_unconstrained(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
    ) -> ClickedSkillCard:
        """Resolve a development truth click without trusting the badge count."""

        del target, stage_number, member_slot
        key = (group_index, card_slot)
        candidate_ids = self._card_candidate_groups.get(key)
        if candidate_ids is None:
            raise ArenaReaderError(
                "skill_card_prediction_missing",
                "card identity candidates were not established before its truth click",
            )
        return self._read_resolved_card_detail(
            key,
            candidate_ids,
            expected_customization_count=None,
            allow_zero_without_badge_count=True,
        )

    def _measure_card_face_generic_cost(
        self,
        key: tuple[int, int],
        card_id: int,
    ) -> tuple[int, int, int] | None:
        diagnostics = getattr(self, "_card_face_cost_diagnostics", None)
        if diagnostics is None:
            diagnostics = {}
            self._card_face_cost_diagnostics = diagnostics
        cached = diagnostics.get(key)
        hypotheses = self.catalog.generic_cost_hypotheses(card_id)
        if hypotheses is None:
            # The visual-family hint may resolve to a different authoritative
            # detail ID.  Do not leave evidence from the hinted generic-cost
            # card attached to a slot whose confirmed card has no such field.
            diagnostics.pop(key, None)
            return None
        hypotheses = tuple(int(value) for value in hypotheses)
        if cached is not None:
            cached_hypotheses = cached.get("hypotheses")
            cache_matches = (
                cached.get("card_id") == card_id
                and isinstance(cached_hypotheses, list)
                and tuple(cached_hypotheses) == hypotheses
            )
            if cache_matches:
                values = cached.get("frame_values")
                if (
                    isinstance(values, list)
                    and len(values) == 3
                    and all(type(value) is int for value in values)
                ):
                    return int(values[0]), int(values[1]), int(values[2])
                raise ArenaReaderError(
                    "skill_card_cost_evidence_invalid",
                    f"cached card-face cost evidence is invalid for {key!r}",
                )
            # Detail title resolution is authoritative over the pre-click
            # visual-family hint.  Re-evaluate the already frozen three source
            # frames against the confirmed identity/hypotheses instead of
            # reusing a semantically stale integer or taking another screenshot.
            self._increment("skill_card_face_cost_cache_rebinds")
        group_index, card_slot = key
        frames = self._card_count_frames.get(group_index, ())
        if len(frames) != 3:
            raise ArenaReaderError(
                "skill_card_cost_frames_missing",
                f"group {group_index} has no stable three-frame cost evidence",
            )
        started = time.perf_counter()
        try:
            frame_rows = tuple(
                self._validated_card_group_row(frame, group_index)
                for frame in frames
            )
            predictions = tuple(
                self._card_cost_references().measure(
                    frame,
                    row[card_slot - 1],
                    card_id=card_id,
                    allowed_values=hypotheses,
                )
                for frame, row in zip(frames, frame_rows, strict=True)
            )
            value = stable_generic_cost_value(predictions)
            values = (value, value, value)
        except (
            GenericCostReferenceError,
        ) as error:
            raise ArenaReaderError(
                "skill_card_cost_evidence_inconclusive",
                f"group {group_index}/slot {card_slot}/card {card_id}: {error}",
            ) from error
        finally:
            self._add_timing(
                "skill_card_face_cost_evidence",
                time.perf_counter() - started,
            )
        self._increment("skill_card_face_cost_measurements")
        diagnostics[key] = {
            "group_index": group_index,
            "slot": card_slot,
            "card_id": card_id,
            "hypotheses": list(hypotheses),
            "frame_values": list(values),
            "mode": "card_face_generic_cost_measured",
            "predictions": [
                {
                    "status": prediction.status,
                    "value": prediction.value,
                    "candidate_value": prediction.candidate_value,
                    "best_error": prediction.best_error,
                    "class_margin": prediction.class_margin,
                    "pixel_count": prediction.pixel_count,
                    "evidence_mode": prediction.evidence_mode,
                    "zero_hole_box": (
                        list(prediction.zero_hole_box)
                        if prediction.zero_hole_box is not None
                        else None
                    ),
                    "zero_alternative_error": prediction.zero_alternative_error,
                    "component_boxes": [
                        list(box) for box in prediction.component_boxes
                    ],
                }
                for prediction in predictions
            ],
        }
        return values

    def _measure_optional_card_face_generic_cost(
        self,
        key: tuple[int, int],
        card_id: int,
    ) -> tuple[int, int, int] | None:
        """Return optional cost evidence without blocking independent detail truth."""

        errors = getattr(self, "_card_face_cost_optional_errors", None)
        if errors is None:
            errors = {}
            self._card_face_cost_optional_errors = errors
        try:
            values = self._measure_card_face_generic_cost(key, card_id)
        except ArenaReaderError as error:
            if error.code != "skill_card_cost_evidence_inconclusive":
                raise
            self._increment("skill_card_face_cost_optional_inconclusive")
            errors[key] = str(error)
            return None
        errors.pop(key, None)
        return values

    def _certify_card_face_generic_cost(
        self,
        key: tuple[int, int],
        card_id: int,
        customizations: Any,
    ) -> tuple[int, int, int] | None:
        values = self._measure_card_face_generic_cost(key, card_id)
        if values is None:
            return None
        group_index, card_slot = key
        try:
            mode = self.catalog.certify_card_face_generic_cost(
                card_id,
                resolved=customizations,
                frame_values=values,
            )
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_cost_evidence_conflict",
                f"group {group_index}/slot {card_slot}/card {card_id}: {error}",
            ) from error
        diagnostic = self._card_face_cost_diagnostics[key]
        if diagnostic.get("mode") != mode:
            self._increment("skill_card_face_cost_certifications")
            diagnostic["mode"] = mode
        return values

    def _read_resolved_card_detail(
        self,
        key: tuple[int, int],
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        timeout_seconds: float = 1.50,
        allow_zero_without_badge_count: bool = False,
    ) -> ClickedSkillCard:
        """Resolve the accepted overlay text, rereading only while effects settle."""

        deadline = time.monotonic() + timeout_seconds
        transaction_started = getattr(self, "_card_transaction_started", {}).get(
            key,
            time.monotonic(),
        )
        zero_not_before = transaction_started + 0.60
        text = self._card_detail_texts.get(key, "")
        last_error = "accepted overlay text was not cached"
        confirmation_candidate: ClickedSkillCard | None = None
        confirmation_reason: str | None = None
        confirmation_started = False
        resolution_reads = 0
        resolution_conflicts = 0
        effect_roi_failures = 0
        zero_candidate: ClickedSkillCard | None = None
        zero_confirmation_reads = 0
        zero_settle_wait_recorded = False
        zero_confirmation_extension_applied = False
        positive_candidate: ClickedSkillCard | None = None
        positive_confirmation_reads = 0
        positive_confirmation_extension_applied = False
        terminal_error: ArenaReaderError | None = None
        same_frame_effect_roi_attempted = False
        while True:
            # Provenance is frame-local.  A title-anchored ROI may resolve an
            # otherwise ambiguous detail on the accepted frame itself; keep
            # that exact full/ROI pair so the structural-omission certifier can
            # consume it without a redundant OCR pass or a cross-frame join.
            same_frame_effect_roi_context: tuple[str, str] | None = None
            if text:
                auxiliary_badge_glyph_allowed = (
                    time.monotonic() >= zero_not_before
                )
                try:
                    try:
                        resolved = self._resolve_clicked_card_text(
                            candidate_ids,
                            text,
                            expected_customization_count=expected_customization_count,
                            allow_zero_without_badge_count=(
                                allow_zero_without_badge_count
                            ),
                            allow_auxiliary_badge_glyph=(
                                auxiliary_badge_glyph_allowed
                            ),
                            key=key,
                        )
                    except ArenaReaderError as initial_error:
                        if initial_error.code != "skill_card_detail_ambiguous":
                            raise
                        if same_frame_effect_roi_attempted:
                            raise
                        same_frame_effect_roi_attempted = True
                        detail_image = self._card_detail_images.get(key)
                        if detail_image is None:
                            raise
                        card_id = self._confirm_skill_card_detail_identity(
                            text,
                            candidate_ids,
                            expected_customization_count=(
                                expected_customization_count
                            ),
                        )
                        roi_text = (
                            self._skill_card_title_anchored_effect_roi_text(
                                detail_image,
                                card_id,
                            )
                        )
                        same_frame_effect_roi_context = (text, roi_text)
                        combined_text = f"{text}\n{roi_text}"
                        resolved = self._resolve_clicked_card_text(
                            candidate_ids,
                            combined_text,
                            expected_customization_count=(
                                expected_customization_count
                            ),
                            allow_zero_without_badge_count=(
                                allow_zero_without_badge_count
                            ),
                            allow_auxiliary_badge_glyph=(
                                auxiliary_badge_glyph_allowed
                            ),
                            key=key,
                        )
                        text = combined_text
                        self._card_detail_texts[key] = combined_text
                        self._increment(
                            "skill_card_detail_same_frame_effect_roi_recoveries"
                        )
                    resolved_count = sum(
                        int(value) for value in resolved.customizations.values()
                    )
                    if allow_zero_without_badge_count and resolved_count == 0:
                        # A detail title becomes readable before its effect rows are
                        # guaranteed to finish animating.  The prior context-action
                        # overhead happened to hide this race; direct point dispatch
                        # exposed it as a false zero.  Zero therefore crosses both a
                        # minimum render boundary and one fresh, semantically
                        # identical detail frame.
                        now = time.monotonic()
                        if now < zero_not_before:
                            if not zero_settle_wait_recorded:
                                zero_settle_wait_recorded = True
                                self._increment("skill_card_detail_zero_settle_waits")
                            raise ArenaReaderError(
                                "skill_card_detail_zero_before_settle",
                                "an unconstrained zero detail arrived before the "
                                "effect-render settle boundary",
                            )
                        zero_confirmation_reads += 1
                        self._increment("skill_card_detail_zero_confirmation_reads")
                        if self._same_clicked_card_resolution(zero_candidate, resolved):
                            resolved = ClickedSkillCard(
                                resolved.card_id,
                                dict(resolved.customizations),
                                resolution_source=(
                                    f"{resolved.resolution_source}_zero_confirmed"
                                ),
                                detail_confirmation_reads=zero_confirmation_reads,
                                detail_evidence_mode=resolved.detail_evidence_mode,
                            )
                            self._increment("skill_card_detail_zero_confirmations")
                        else:
                            if zero_candidate is not None:
                                self._increment(
                                    "skill_card_detail_zero_confirmation_conflicts"
                                )
                            zero_candidate = resolved
                            if not zero_confirmation_extension_applied:
                                # The base deadline covers positive-effect settling.
                                # Once a post-settle zero actually exists, reserve
                                # exactly one additional OCR frame for its required
                                # confirmation instead of widening every detail read.
                                deadline = max(deadline, time.monotonic() + 0.45)
                                zero_confirmation_extension_applied = True
                                self._increment(
                                    "skill_card_detail_zero_confirmation_extensions"
                                )
                            raise ArenaReaderError(
                                "skill_card_detail_zero_unconfirmed",
                                "an unconstrained zero detail requires one fresh "
                                "semantically identical post-settle frame",
                            )
                    evidence_customization_count = expected_customization_count
                    if (
                        evidence_customization_count is None
                        and resolved.resolution_source.endswith(
                            "_auxiliary_badge_glyph"
                        )
                    ):
                        auxiliary = self._badge_glyph_count_diagnostics.get(key, {})
                        auxiliary_count = auxiliary.get("resolved_count")
                        if type(auxiliary_count) is not int:
                            raise ArenaReaderError(
                                "skill_card_badge_glyph_evidence_missing",
                                "an auxiliary badge-glyph resolution has no cached count",
                            )
                        evidence_customization_count = auxiliary_count
                    count_mismatch = (
                        expected_customization_count is not None
                        and resolved_count != expected_customization_count
                    )
                    if count_mismatch and (
                        resolved.resolution_source not in {
                            "detail_unique",
                            "detail_card_face_unique",
                        }
                        or resolved_count < 1
                    ):
                        raise ArenaReaderError(
                            "skill_card_detail_count_conflict",
                            "a badge-count mismatch was not resolved by one unique positive detail",
                        )

                    try:
                        generic_cost_frame_values = (
                            self._certify_card_face_generic_cost(
                                key,
                                resolved.card_id,
                                resolved.customizations,
                            )
                        )
                    except ArenaReaderError as error:
                        if error.code != "skill_card_cost_evidence_inconclusive":
                            raise
                        # Detail semantics and the bounded auxiliary badge count
                        # may already prove a unique group.  An inconclusive
                        # optional cost carrier must not preempt those independent
                        # paths; any later branch that truly needs cost evidence
                        # still fails inside the external-evidence certifier.
                        self._increment("skill_card_face_cost_optional_inconclusive")
                        generic_cost_frame_values = None
                        generic_cost_evidence_error = str(error)
                    else:
                        generic_cost_evidence_error = None

                    if resolved.detail_evidence_mode == "negative_dependent":
                        if same_frame_effect_roi_context is not None:
                            full_detail_text, roi_text = (
                                same_frame_effect_roi_context
                            )
                            effect_roi_title_bound = True
                            combined_text = f"{full_detail_text}\n{roi_text}"
                        else:
                            detail_image = self._card_detail_images.get(key)
                            if detail_image is None:
                                raise ArenaReaderError(
                                    "skill_card_detail_positive_evidence_missing",
                                    "the accepted detail page has no cached image for an "
                                    "independent effect-ROI OCR; "
                                    f"key={key!r}; card_id={resolved.card_id}; "
                                    f"customizations={dict(resolved.customizations)!r}; "
                                    f"expected_badge_count={evidence_customization_count!r}",
                                )
                            full_detail_text = text
                            # A fixed screen column is not a valid coverage
                            # boundary: live detail panels shift horizontally,
                            # and a fresh frame cannot repair a cropped panel.
                            # Every structural-omission pass is therefore bound
                            # to the uniquely resolved title, including the
                            # first pass. A bounded fresh-frame retry remains
                            # only for genuinely incomplete effect animation.
                            effect_roi_title_bound = True
                            roi_text = (
                                self._skill_card_title_anchored_effect_roi_text(
                                    detail_image,
                                    resolved.card_id,
                                )
                            )
                            combined_text = f"{text}\n{roi_text}"
                            resolved = self._resolve_clicked_card_text(
                                candidate_ids,
                                combined_text,
                                expected_customization_count=(
                                    expected_customization_count
                                ),
                                allow_zero_without_badge_count=(
                                    allow_zero_without_badge_count
                                ),
                                allow_auxiliary_badge_glyph=(
                                    auxiliary_badge_glyph_allowed
                                ),
                                key=key,
                            )
                        resolved_count = sum(
                            int(value)
                            for value in resolved.customizations.values()
                        )
                        evidence_customization_count = expected_customization_count
                        if (
                            evidence_customization_count is None
                            and resolved.resolution_source.endswith(
                                "_auxiliary_badge_glyph"
                            )
                        ):
                            auxiliary = self._badge_glyph_count_diagnostics.get(key, {})
                            auxiliary_count = auxiliary.get("resolved_count")
                            if type(auxiliary_count) is not int:
                                raise ArenaReaderError(
                                    "skill_card_badge_glyph_evidence_missing",
                                    "an auxiliary badge-glyph reread has no cached count",
                                )
                            evidence_customization_count = auxiliary_count
                        count_mismatch = (
                            expected_customization_count is not None
                            and resolved_count != expected_customization_count
                        )
                        if resolved.detail_evidence_mode == "negative_dependent":
                            certifier = getattr(
                                self.catalog,
                                "certify_structural_omission_evidence",
                                None,
                            )
                            certified_mode = None
                            certification_error = "catalog has no structural omission certifier"
                            certification_count = evidence_customization_count
                            if certification_count is None:
                                # The coverage certifier is itself the independent
                                # evidence boundary: it accepts every selected level
                                # only when two covered detail views prove an explicit
                                # positive signature, a UI-defined typed-cost/limit
                                # omission, or a stable three-frame card-face cost.
                                # Let it certify the resolved group's total without
                                # making the fallible face-badge digit a prerequisite.
                                certification_count = resolved_count
                            if certifier is not None and certification_count is not None:
                                try:
                                    certified_mode = certifier(
                                        resolved.card_id,
                                        full_detail_text=full_detail_text,
                                        effect_roi_text=roi_text,
                                        expected_count=certification_count,
                                        resolved=resolved.customizations,
                                        generic_cost_frame_values=(
                                            generic_cost_frame_values
                                        ),
                                        effect_roi_title_bound=(
                                            effect_roi_title_bound
                                        ),
                                    )
                                except ArenaCatalogError as error:
                                    certification_error = str(error)
                            if generic_cost_evidence_error is not None:
                                certification_error = (
                                    f"{certification_error}; card_face_cost="
                                    f"{generic_cost_evidence_error}"
                                )
                            if certified_mode not in {
                                "structural_omission_unique",
                                "card_face_generic_cost_unique",
                                "external_evidence_unique",
                            }:
                                raise ArenaReaderError(
                                    "skill_card_detail_positive_evidence_missing",
                                    "full-screen and enlarged effect-ROI OCR both lacked "
                                    "positive evidence and did not certify a UI-defined "
                                    "structural omission or independent card-face cost; "
                                    f"key={key!r}; card_id={resolved.card_id}; "
                                    f"customizations={dict(resolved.customizations)!r}; "
                                    f"expected_badge_count={evidence_customization_count!r}; "
                                    f"certification={certification_error}; "
                                    f"effect_roi={roi_text[:240]!r}",
                                )
                            resolved = ClickedSkillCard(
                                resolved.card_id,
                                dict(resolved.customizations),
                                resolution_source=resolved.resolution_source,
                                detail_confirmation_reads=resolved.detail_confirmation_reads,
                                detail_evidence_mode=certified_mode,
                            )
                            self._increment(
                                "skill_card_detail_external_evidence_certifications"
                            )
                            if certified_mode in {
                                "structural_omission_unique",
                                "external_evidence_unique",
                            }:
                                self._increment(
                                    "skill_card_detail_structural_omission_certifications"
                                )
                        text = combined_text
                        self._card_detail_texts[key] = combined_text
                    if (
                        allow_zero_without_badge_count
                        and expected_customization_count is None
                        and resolved_count > 0
                        and resolved.detail_evidence_mode == "positive_unique"
                    ):
                        # This branch supplies the badge count as well as the
                        # concrete customization IDs.  A single transitional OCR
                        # frame must not become self-authenticating positive
                        # evidence, so require the same business resolution once
                        # more from a fresh frame.  Structural-omission and
                        # card-face-cost modes already carry independent channels.
                        # This adds no navigation or broad retry.
                        positive_confirmation_reads += 1
                        self._increment(
                            "skill_card_detail_positive_confirmation_reads"
                        )
                        if self._same_clicked_card_resolution(
                            positive_candidate,
                            resolved,
                        ):
                            resolved = ClickedSkillCard(
                                resolved.card_id,
                                dict(resolved.customizations),
                                resolution_source=(
                                    f"{resolved.resolution_source}_positive_confirmed"
                                ),
                                detail_confirmation_reads=(
                                    positive_confirmation_reads
                                ),
                                detail_evidence_mode=resolved.detail_evidence_mode,
                            )
                            self._increment(
                                "skill_card_detail_positive_confirmations"
                            )
                        else:
                            if positive_candidate is not None:
                                self._increment(
                                    "skill_card_detail_positive_confirmation_conflicts"
                                )
                            positive_candidate = resolved
                            if not positive_confirmation_extension_applied:
                                deadline = max(
                                    deadline,
                                    time.monotonic() + 0.25,
                                )
                                positive_confirmation_extension_applied = True
                                self._increment(
                                    "skill_card_detail_positive_confirmation_extensions"
                                )
                            raise ArenaReaderError(
                                "skill_card_detail_positive_unconfirmed",
                                "an unconstrained positive detail requires one fresh "
                                "semantically identical frame",
                            )
                    reason = (
                        "badge_count_override"
                        if count_mismatch
                        else None
                    )
                    if reason is None:
                        return resolved

                    resolution_reads += 1
                    if not confirmation_started:
                        confirmation_started = True
                        self._increment("skill_card_detail_semantic_confirmations")
                    if self._same_clicked_card_resolution(
                        confirmation_candidate,
                        resolved,
                    ):
                        confirmed = ClickedSkillCard(
                            resolved.card_id,
                            dict(resolved.customizations),
                            resolution_source=f"{resolved.resolution_source}_confirmed",
                            detail_confirmation_reads=resolution_reads,
                            detail_evidence_mode=resolved.detail_evidence_mode,
                        )
                        self._record_detail_semantic_confirmation(
                            key,
                            confirmed,
                            reason=reason,
                            resolution_conflicts=resolution_conflicts,
                        )
                        if count_mismatch:
                            assert expected_customization_count is not None
                            self._record_detail_count_override(
                                key,
                                expected_customization_count,
                                confirmed,
                            )
                        return confirmed
                    if confirmation_candidate is not None:
                        resolution_conflicts += 1
                        self._increment("skill_card_detail_semantic_conflicts")
                    confirmation_candidate = resolved
                    confirmation_reason = reason
                    last_error = self._detail_confirmation_wait_message(
                        reason,
                        expected_customization_count=expected_customization_count,
                        resolved_count=resolved_count,
                    )
                except ArenaReaderError as error:
                    confirmation_candidate = None
                    confirmation_reason = None
                    last_error = str(error)
                    if error.code == "skill_card_detail_positive_evidence_missing":
                        if effect_roi_failures < 1:
                            # The title can become readable before the effect rows
                            # finish animating. Retry the complete evidence pair on
                            # one fresh frame; never join full/ROI text across frames.
                            effect_roi_failures += 1
                            self._increment("skill_card_detail_effect_roi_retries")
                        else:
                            terminal_error = error
                            break
            if terminal_error is not None:
                break
            if time.monotonic() >= deadline:
                break
            # The title is intentionally not a readiness signal: live details
            # can hold a title-only plateau for longer than the former 0.75 s
            # transaction budget.  Poll semantic evidence at a short cadence
            # and return immediately on the first complete result; the 1.50 s
            # deadline is a failure bound, not a per-card fixed wait.
            time.sleep(0.08)
            self._increment("skill_card_detail_ocr_rereads")
            try:
                detail_image = self._capture()
                text = self._skill_card_detail_ocr_text(
                    detail_image,
                    phase="reread",
                )
            except ArenaReaderError as error:
                text = ""
                last_error = str(error)
            else:
                self._card_detail_texts[key] = text
                self._card_detail_images[key] = detail_image
                same_frame_effect_roi_attempted = False
        if terminal_error is not None:
            raise terminal_error
        raise ArenaReaderError(
            "skill_card_detail_ambiguous",
            "card detail did not become uniquely resolvable within the bounded settle window: "
            f"{last_error}; pending_confirmation={confirmation_reason!r}; "
            f"OCR={text[:400]!r}",
        )

    @staticmethod
    def _same_clicked_card_resolution(
        left: ClickedSkillCard | None,
        right: ClickedSkillCard,
    ) -> bool:
        return (
            left is not None
            and left.card_id == right.card_id
            and dict(left.customizations) == dict(right.customizations)
        )

    @staticmethod
    def _detail_confirmation_wait_message(
        reason: str,
        *,
        expected_customization_count: int | None,
        resolved_count: int,
    ) -> str:
        if reason == "badge_count_override":
            return (
                f"card-face count {expected_customization_count} conflicts with "
                f"unique detail count {resolved_count}; awaiting one fresh confirmation"
            )
        return (
            f"badge count {resolved_count} admits multiple legal customization groups; "
            "awaiting the same final-effect resolution from one fresh frame"
        )

    def _record_detail_semantic_confirmation(
        self,
        key: tuple[int, int],
        resolved: ClickedSkillCard,
        *,
        reason: str,
        resolution_conflicts: int,
    ) -> None:
        self._detail_semantic_confirmations[key] = {
            "slot": key[1],
            "card_id": resolved.card_id,
            "resolved_detail_count": sum(
                int(value) for value in resolved.customizations.values()
            ),
            "customizations": dict(resolved.customizations),
            "reason": reason,
            "detail_confirmation_reads": resolved.detail_confirmation_reads,
            "resolution_conflicts": resolution_conflicts,
            "evidence_mode": resolved.detail_evidence_mode,
        }

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
        started = time.perf_counter()
        try:
            return self._full_ocr_text(enlarged)
        finally:
            self._increment("skill_card_detail_effect_roi_reads")
            self._add_timing(
                "skill_card_detail_effect_roi_ocr",
                time.perf_counter() - started,
            )

    def _confirm_skill_card_detail_identity(
        self,
        text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
    ) -> int:
        """Resolve the already-open detail title under the existing bounded gates."""

        try:
            try:
                return self.catalog.confirm_clicked_skill_card_candidates(
                    text,
                    candidate_card_ids=candidate_ids,
                )
            except ArenaCatalogError:
                try:
                    return (
                        self.catalog.confirm_clicked_customizable_skill_card_without_badge_count(
                            text
                        )
                    )
                except ArenaCatalogError:
                    if expected_customization_count is None:
                        raise
                    return self.catalog.confirm_clicked_customizable_skill_card(
                        text,
                        expected_count=expected_customization_count,
                    )
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_detail_ambiguous",
                str(error),
            ) from error

    def _skill_card_detail_ocr_text(self, image: Any, *, phase: str) -> str:
        """Time full detail OCR by transaction phase without changing its result."""

        if phase not in {"open", "reread"}:
            raise ValueError("skill-card detail OCR phase must be open or reread")
        if not hasattr(self, "_runtime_timing_seconds"):
            self._runtime_timing_seconds = {}
        started = time.perf_counter()
        try:
            return self._full_ocr_text(image)
        finally:
            self._increment(f"skill_card_detail_{phase}_ocr_reads")
            self._add_timing(
                f"skill_card_detail_{phase}_ocr",
                time.perf_counter() - started,
            )

    @staticmethod
    def _title_anchored_effect_roi(
        frame_width: int,
        frame_height: int,
        title_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Cover one effect panel from its title instead of a fixed screen column."""

        title_x, title_y, title_width, title_height = title_box
        # OCR boxes are intentionally tight and may cover only the first title
        # glyphs.  The effect rows can extend far to either side of that box.
        # Keep the retry bounded to the title's panel, but guarantee at least
        # two thirds of the normalized screen width so a shifted panel cannot
        # truncate its invariant action rows.
        minimum_width = int(round(frame_width * 0.66))
        left = max(0, title_x - int(round(frame_width * 0.12)))
        top = max(0, title_y - int(round(frame_height * 0.015)))
        right = min(
            frame_width,
            title_x + title_width + int(round(frame_width * 0.55)),
        )
        if right - left < minimum_width:
            left = max(0, min(left, frame_width - minimum_width))
            right = min(frame_width, max(right, left + minimum_width))
            if right - left < minimum_width:
                left = max(0, right - minimum_width)
        bottom = min(
            frame_height,
            max(
                title_y + title_height + int(round(frame_height * 0.25)),
                top + int(round(frame_height * 0.295)),
            ),
        )
        if right - left < 32 or bottom - top < 32:
            raise ArenaReaderError(
                "skill_card_detail_effect_roi_invalid",
                "title-anchored detail effect ROI is too small: "
                f"frame={frame_width}x{frame_height}, title={title_box!r}",
            )
        return left, top, right - left, bottom - top

    def _skill_card_title_anchored_effect_roi_text(
        self,
        image: Any,
        card_id: int,
    ) -> str:
        """Retry effect OCR in the panel geometrically bound to its fixed title."""

        import cv2

        try:
            title_pattern = self.catalog.skill_card_title_anchor_pattern(card_id)
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_detail_title_catalog_missing",
                str(error),
            ) from error
        matches = self._ocr(image, title_pattern)
        if len(matches) != 1:
            self._increment("skill_card_detail_title_anchor_transition_retries")
            raise ArenaReaderError(
                "skill_card_detail_title_anchor_ambiguous",
                f"card {card_id} has {len(matches)} title anchors on the retry frame",
            )
        height, width = image.shape[:2]
        left, top, roi_width, roi_height = self._title_anchored_effect_roi(
            width,
            height,
            _box(matches[0]),
        )
        crop = image[top : top + roi_height, left : left + roi_width]
        if crop.size == 0:
            raise ArenaReaderError(
                "skill_card_detail_effect_roi_invalid",
                f"title-anchored detail effect ROI is empty for frame {width}x{height}",
            )
        enlarged = cv2.resize(
            crop,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        started = time.perf_counter()
        try:
            return self._full_ocr_text(enlarged)
        finally:
            self._increment("skill_card_detail_title_anchored_effect_roi_reads")
            self._increment("skill_card_detail_effect_roi_reads")
            self._add_timing(
                "skill_card_detail_effect_roi_ocr",
                time.perf_counter() - started,
            )

    def _record_detail_count_override(
        self,
        key: tuple[int, int],
        observed_badge_count: int,
        resolved: ClickedSkillCard,
    ) -> None:
        resolved_count = sum(int(value) for value in resolved.customizations.values())
        self._detail_count_overrides[key] = {
            "slot": key[1],
            "observed_badge_count": observed_badge_count,
            "resolved_detail_count": resolved_count,
            "card_id": resolved.card_id,
            "customizations": dict(resolved.customizations),
            "detail_confirmation_reads": resolved.detail_confirmation_reads,
        }
        self._increment("skill_card_detail_count_overrides")

    def close_skill_card(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
    ) -> None:
        del target, stage_number, member_slot
        key = (group_index, card_slot)
        if self._active_inferred_clicked_card == key:
            # The candidate-detail inference already dismissed the overlay and
            # proved this exact slot against the accepted source generation.
            # Reusing the cached result performs no UI action, so a second
            # screenshot/signature guard here cannot add state evidence.
            self._active_inferred_clicked_card = None
            self._inferred_clicked_cards.pop(key, None)
            self._increment("skill_card_inferred_detail_reuses")
            return
        self._dismiss_skill_card_detail()
        try:
            self._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
                expected_card_id=self._card_predictions.get(key),
            )
        except ArenaReaderError:
            image = self._capture()
            if len(self._ocr(image, r"^ステージ\s*[123]$")) >= 3:
                raise ArenaReaderError(
                    "skill_card_close_left_member",
                    "closing a skill card unexpectedly left the member detail",
                )
            self._dismiss_skill_card_detail()
            self._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
                expected_card_id=self._card_predictions.get(key),
            )
        self._finish_card_transaction(key)

    def close_member(self, target: TeamTarget, stage_number: int) -> None:
        del stage_number
        self._reset_member_card_state()
        self._back()
        expected_totals = (1, 2, 3) if target.is_own_team else (6,)
        try:
            self._wait_for_stage_member_list(
                expected_totals,
                timeout_seconds=2.0,
            )
        except ArenaReaderError:
            # A single retry is allowed only when the unchanged 体力/総合力
            # anchors prove that the first back action was dropped. Any other
            # page remains fail-closed rather than issuing another blind back.
            image = self._capture()
            if not self._ocr(image, r"^体力$") or not self._ocr(image, r"^総合力$"):
                raise
            self._increment("close_member_retries")
            self._back()
            self._wait_for_stage_member_list(
                expected_totals,
                timeout_seconds=2.0,
            )
        self._member_metric_baseline = None

    def _reset_member_card_state(self) -> None:
        """Discard card geometry, hints and frames that belong to the prior member."""

        self._card_rows.clear()
        self._card_predictions.clear()
        self._card_candidate_groups.clear()
        self._card_images.clear()
        self._card_count_frames.clear()
        self._card_identity_frames.clear()
        self._card_restoration_signatures.clear()
        self._card_excluded_duplicate_flags.clear()
        self._card_empty_flags.clear()
        self._card_empty_diagnostics.clear()
        self._card_duplicate_marker_diagnostics.clear()
        self._badge_count_diagnostics.clear()
        self._detail_count_overrides.clear()
        self._detail_semantic_confirmations.clear()
        self._badge_local_results.clear()
        self._badge_glyph_observations.clear()
        self._badge_glyph_count_diagnostics.clear()
        self._inferred_clicked_cards.clear()
        self._active_inferred_clicked_card = None
        self._card_detail_texts.clear()
        self._card_detail_images.clear()
        self._p_item_diagnostics = ()
        self._p_item_generation_evidence.clear()
        self._p_item_source_frames = None
        self._card_transaction_started.clear()
        self._card_transaction_kinds.clear()
        self._secondary_fixed_slot_fallback_enabled = False
        self._secondary_presence_diagnostics = ()
        self._secondary_presence_cache.clear()
        self._card_content_generation_diagnostics = ()
        self._card_face_cost_diagnostics.clear()
        self._card_face_cost_optional_errors.clear()
        self._zero_card_identity_fusion_diagnostics.clear()
        self._zero_card_embedding_detail_candidates.clear()
        self._zero_card_detail_candidates.clear()

    def leave_team(self, target: TeamTarget) -> None:
        self._recover_arena_main(
            maximum_steps=3,
            error_code="team_return_failed",
            require_opponents=not target.is_own_team,
        )

    def recover_to_arena_main(self, *, require_opponents: bool = True) -> None:
        self._recover_arena_main(
            maximum_steps=6,
            error_code="arena_recovery_failed",
            require_opponents=require_opponents,
        )

    def _recover_arena_main(
        self,
        *,
        maximum_steps: int,
        error_code: str,
        require_opponents: bool,
    ) -> None:
        unavailable_reads = 0
        for _ in range(maximum_steps):
            image = self._capture()
            state, _ = self._arena_page_state(image)
            if arena_page_allows_team_entry(
                state,
                require_opponents=require_opponents,
            ):
                return
            if state is ArenaPageState.OPPONENTS_UNAVAILABLE:
                unavailable_reads += 1
                if unavailable_reads >= 3:
                    raise ArenaReaderError(
                        "arena_opponents_unavailable",
                        "opponent cards stayed absent on the arena main page; recovery will not navigate away",
                    )
                time.sleep(0.25)
                continue
            unavailable_reads = 0
            if self._dismiss_known_blocking_overlay(image):
                time.sleep(0.25)
                continue
            self._back()
        required_state = "with three opponents" if require_opponents else "with the rehearsal anchor"
        raise ArenaReaderError(
            error_code,
            f"back navigation did not reach an arena main page {required_state}",
        )

    def _dismiss_known_blocking_overlay(self, image: Any) -> bool:
        """Dismiss only overlays proven by independent page-specific anchors."""

        menu_profile = self._ocr(image, r"^プロフィール$")
        menu_settings = self._ocr(image, r"^設定$")
        if len(menu_profile) == 1 and len(menu_settings) == 1:
            height, width = image.shape[:2]
            # The upper-centre backdrop is outside the menu's functional grid
            # at every supported aspect ratio.  A click there dismisses the
            # proven menu without relying on a scale-sensitive close template.
            self._click((int(width * 0.50), int(height * 0.16), 1, 1))
            self._increment("arena_menu_overlays_dismissed")
            return True

        error_titles = self._ocr(image, r"^通信エラ(?:ー)?$")
        transient_details = self._ocr(
            image,
            r"^通信中にエラーが発生しました$",
        )
        retry_buttons = self._ocr(image, r"^リトライ$")
        title_buttons = self._ocr(image, r"^タイトルへ$")
        if (
            len(error_titles) == 1
            and len(transient_details) == 1
            and len(retry_buttons) == 1
            and len(title_buttons) == 1
        ):
            if getattr(self, "_arena_communication_retry_attempted", False):
                raise ArenaReaderError(
                    "arena_communication_retry_exhausted",
                    "the proven communication-error overlay remained after one retry",
                )
            self._arena_communication_retry_attempted = True
            self._click(_box(retry_buttons[0]), settle_seconds=1.5)
            self._increment("arena_communication_retries")
            return True

        failure_details = self._ocr(image, r"^アセット取得に失敗$")
        if len(error_titles) != 1 or len(failure_details) != 1:
            return False
        close_buttons = self._recognize("CloseRoundButton", image)
        if len(close_buttons) != 1:
            raise ArenaReaderError(
                "arena_blocking_overlay_close_ambiguous",
                "a communication-error overlay is proven but its close button is not unique",
            )
        self._click(_box(close_buttons[0]))
        self._increment("arena_blocking_overlays_dismissed")
        return True

    def _capture(self) -> Any:
        started = time.perf_counter()
        try:
            return self.context.tasker.controller.post_screencap().wait().get()
        finally:
            self._increment("screenshots")
            self._add_timing("screenshots", time.perf_counter() - started)

    def _recognize(self, entry: str, image: Any) -> list[Any]:
        detail = self.context.run_recognition(entry, image)
        if not detail or not detail.hit:
            return []
        return list(detail.filtered_results or detail.all_results or [])

    def _ocr(
        self,
        image: Any,
        expected: str,
        *,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[Any]:
        override: dict[str, Any] = {
            "ArenaReaderOCR": {
                "recognition": "OCR",
                "expected": expected,
                "order_by": "Vertical",
            }
        }
        if roi is not None:
            override["ArenaReaderOCR"]["roi"] = list(roi)
        detail = self.context.run_recognition("ArenaReaderOCR", image, pipeline_override=override)
        if not detail or not detail.hit:
            return []
        return list(detail.filtered_results or detail.all_results or [])

    def _full_ocr_text(self, image: Any) -> str:
        detail = self.context.run_recognition(
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
        if not detail or not detail.hit:
            raise ArenaReaderError("ocr_empty", "full-screen OCR returned no text")
        return "\n".join(_text(item) for item in (detail.all_results or detail.filtered_results or []))

    def _click(
        self,
        box: tuple[int, int, int, int],
        *,
        settle_seconds: float = 0.25,
    ) -> None:
        started = time.perf_counter()
        dispatch = "context_box"
        if box[2:] == (1, 1):
            dispatch = "controller_direct"
            job = self.context.tasker.controller.post_click(box[0], box[1]).wait()
            succeeded = bool(job.succeeded)
        else:
            result = self.context.run_action_direct(JActionType.Click, JClick(), box)
            succeeded = bool(result is not None and result.success)
        elapsed = time.perf_counter() - started
        self._add_timing("click_actions", elapsed)
        self._add_timing(f"{dispatch}_click_actions", elapsed)
        self._increment("click_actions")
        self._increment(f"{dispatch}_click_actions")
        if not succeeded:
            raise ArenaReaderError("maa_click_failed", f"Maa could not click {box}")
        if settle_seconds > 0:
            time.sleep(settle_seconds)

    @staticmethod
    def _box_center_point(
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Collapse one uniquely recognized control to its deterministic centre."""

        x, y, width, height = box
        if width < 1 or height < 1:
            raise ArenaReaderError(
                "maa_click_box_invalid",
                f"recognized control has an invalid box: {box!r}",
            )
        return (x + width // 2, y + height // 2, 1, 1)

    def _long_press(self, box: tuple[int, int, int, int]) -> None:
        x, y, width, height = box
        point = (x + width // 2, y + height // 2)
        started = time.perf_counter()
        controller = self.context.tasker.controller
        down_succeeded = False
        up_succeeded = False
        interaction_error: Exception | None = None
        try:
            down_succeeded = bool(
                controller.post_touch_down(*point).wait().succeeded
            )
            if down_succeeded:
                time.sleep(self._member_long_press_seconds)
        except Exception as error:  # pragma: no cover - native Maa boundary
            interaction_error = error
        finally:
            try:
                up_succeeded = bool(controller.post_touch_up().wait().succeeded)
            except Exception as error:  # pragma: no cover - native Maa boundary
                if interaction_error is None:
                    interaction_error = error
        elapsed = time.perf_counter() - started
        self._add_timing("long_press_actions", elapsed)
        self._add_timing("controller_direct_long_press_actions", elapsed)
        self._increment("long_press_actions")
        self._increment("controller_direct_long_press_actions")
        if interaction_error is not None or not down_succeeded or not up_succeeded:
            detail = (
                f"Maa could not long-press {box} at {point}; "
                f"down={down_succeeded}, up={up_succeeded}"
            )
            if interaction_error is not None:
                detail += f", error={interaction_error}"
            raise ArenaReaderError("maa_long_press_failed", detail)

    def _short_press(
        self,
        box: tuple[int, int, int, int],
        *,
        duration_ms: int,
    ) -> None:
        """Send one bounded tap-like contact when an instantaneous click drops."""

        if not 50 <= duration_ms <= 200:
            raise ArenaReaderError(
                "maa_short_press_duration_invalid",
                f"short press duration is outside 50..200 ms: {duration_ms}",
            )
        x, y, width, height = box
        point = (x + width // 2, y + height // 2)
        started = time.perf_counter()
        controller = self.context.tasker.controller
        down_succeeded = False
        up_succeeded = False
        interaction_error: Exception | None = None
        try:
            down_succeeded = bool(
                controller.post_touch_down(*point).wait().succeeded
            )
            if down_succeeded:
                time.sleep(duration_ms / 1000.0)
        except Exception as error:  # pragma: no cover - native Maa boundary
            interaction_error = error
        finally:
            try:
                up_succeeded = bool(controller.post_touch_up().wait().succeeded)
            except Exception as error:  # pragma: no cover - native Maa boundary
                if interaction_error is None:
                    interaction_error = error
        elapsed = time.perf_counter() - started
        self._add_timing("short_press_actions", elapsed)
        self._add_timing("controller_direct_short_press_actions", elapsed)
        self._increment("short_press_actions")
        self._increment("controller_direct_short_press_actions")
        if interaction_error is not None or not down_succeeded or not up_succeeded:
            detail = (
                f"Maa could not short-press {box} at {point}; "
                f"down={down_succeeded}, up={up_succeeded}"
            )
            if interaction_error is not None:
                detail += f", error={interaction_error}"
            raise ArenaReaderError(
                "maa_short_press_failed",
                detail,
            )

    def _swipe(
        self,
        *,
        vertical: str,
        distance_ratio: float = 0.0625,
        distance_pixels: int | None = None,
    ) -> None:
        image = self._capture()
        height, width = image.shape[:2]
        if not 0.0 < distance_ratio <= 0.20:
            raise ValueError("distance_ratio must be in (0, 0.20]")
        if distance_pixels is None:
            distance_pixels = int(round(height * distance_ratio))
        if not 1 <= distance_pixels <= int(height * 0.20):
            raise ValueError("distance_pixels must be in [1, 20% of frame height]")
        lower_y = int(height * 0.34)
        upper_y = lower_y - distance_pixels
        if vertical == "up":
            begin, end = (int(width * 0.90), lower_y, 1, 1), (
                int(width * 0.90),
                upper_y,
                1,
                1,
            )
        else:
            begin, end = (int(width * 0.90), upper_y, 1, 1), (
                int(width * 0.90),
                lower_y,
                1,
                1,
            )
        started = time.perf_counter()
        job = self.context.tasker.controller.post_swipe(
            begin[0],
            begin[1],
            end[0],
            end[1],
            duration=self._card_swipe_duration_ms,
        ).wait()
        elapsed = time.perf_counter() - started
        self._add_timing("swipe_actions", elapsed)
        self._add_timing("controller_direct_swipe_actions", elapsed)
        self._increment("swipe_actions")
        self._increment("controller_direct_swipe_actions")
        if not job.succeeded:
            raise ArenaReaderError("maa_swipe_failed", f"Maa could not swipe {vertical}")

    def _back(self) -> None:
        image = self._capture()
        height, width = image.shape[:2]
        detail = self.context.run_recognition(
            "ChallengeBack",
            image,
            pipeline_override={
                "ChallengeBack": {
                    "recognition": "TemplateMatch",
                    "template": "back.png",
                    "roi": [0, int(height * 0.8), int(width * 0.5), int(height * 0.2)],
                    "threshold": 0.7,
                    "order_by": "Score",
                }
            },
        )
        results = list((detail.filtered_results or detail.all_results or [])) if detail and detail.hit else []
        if len(results) != 1:
            raise ArenaReaderError("maa_back_anchor_ambiguous", f"found {len(results)} back-button anchors")
        self._click(self._box_center_point(_box(results[0])))

    def _close_overlay(self) -> None:
        result = self.context.run_task("CloseButton")
        if not result or not result.status.succeeded:
            image = self._capture()
            height, width = image.shape[:2]
            self._click((int(width * 0.91), int(height * 0.02), int(width * 0.07), int(height * 0.07)))

    def _dismiss_skill_card_detail(self) -> None:
        """Close a skill-card detail by tapping the inert upper-left backdrop."""

        image = self._capture()
        height, width = image.shape[:2]
        self._increment("skill_card_detail_event_driven_dismissals")
        self._click(
            (
                max(1, int(width * 0.015)),
                max(1, int(height * 0.04)),
                1,
                1,
            ),
            settle_seconds=0,
        )

    def _assert_stage_navigation(self) -> None:
        results = self._ocr(self._capture(), r"^ステージ\s*[123]$")
        if len(results) < 3:
            raise ArenaReaderError("stage_navigation_missing", f"found only {len(results)} stage anchors")

    def _wait_for_stage_member_list(
        self,
        expected_total_counts: tuple[int, ...],
        timeout_seconds: float = 3.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_stage_count = 0
        last_total_count = 0
        while time.monotonic() < deadline:
            image = self._capture()
            last_stage_count = len(self._ocr(image, r"^ステージ\s*[123]$"))
            last_total_count = len(self._stage_total_anchors(image))
            if last_stage_count >= 3 and last_total_count in expected_total_counts:
                return
            time.sleep(0.25)
        raise ArenaReaderError(
            "stage_member_list_timeout",
            f"stage anchors={last_stage_count}, total anchors={last_total_count}, "
            f"expected totals={expected_total_counts!r}",
        )

    def _stage_total_anchors(self, image: Any) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            sorted(
                (_box(item) for item in self._ocr(image, r"^総合力$")),
                key=lambda box: (box[1], box[0]),
            )
        )

    def _team_stage_total_anchors(
        self,
        target: TeamTarget,
        image: Any,
    ) -> tuple[tuple[int, int, int, int], ...]:
        totals = self._stage_total_anchors(image)
        try:
            return team_stage_total_anchors(totals, own_team=target.is_own_team)
        except ValueError as error:
            raise ArenaReaderError(
                "stage_total_count_mismatch",
                f"{target.team_id} stage totals are invalid: {error}",
            ) from error

    def _arena_main_visible(self) -> bool:
        image = self._capture()
        state, _ = self._arena_page_state(image)
        return state is ArenaPageState.READY

    def _arena_page_state(
        self,
        image: Any,
    ) -> tuple[ArenaPageState, tuple[int, int, int, int, int]]:
        rehearsal_count = len(self._ocr(image, r"^リハーサル$"))
        opponent_count = len(self._recognize("ArenaReaderOpponentCards", image))
        stage_label_count = len(self._ocr(image, r"^ステージ\s*[123]$"))
        stage_total_count = len(self._stage_total_anchors(image))
        exhausted_notice_count = len(
            self._ocr(image, r"^本日の挑戦権を消費しました$")
        )
        counts = (
            rehearsal_count,
            opponent_count,
            stage_label_count,
            stage_total_count,
            exhausted_notice_count,
        )
        return (
            classify_arena_page(
                rehearsal_count=rehearsal_count,
                opponent_count=opponent_count,
                stage_label_count=stage_label_count,
                stage_total_count=stage_total_count,
                exhausted_notice_count=exhausted_notice_count,
            ),
            counts,
        )

    def _read_grade(
        self,
        image: Any,
        *,
        observations: Sequence[Any] | None = None,
    ) -> int:
        height, width = image.shape[:2]
        observations = tuple(
            self._ocr(image, r".+")
            if observations is None
            else observations
        )
        labels = tuple(
            item
            for item in observations
            if _normalize_latin_anchor(_text(item)) == "GRADE"
        )
        if len(labels) != 1:
            raise ArenaReaderError(
                "arena_grade_label_ambiguous",
                f"found {len(labels)} GRADE labels on the arena main screen",
            )
        label_x, label_y, label_width, label_height = _box(labels[0])
        left = max(0, label_x - label_width)
        top = min(height - 1, label_y + label_height)
        roi = (
            left,
            top,
            min(width - left, label_width * 3),
            min(
                height - top,
                max(label_height * 8, int(height * 0.14)),
            ),
        )
        right = left + roi[2]
        bottom = top + roi[3]
        matches = tuple(
            (item, grade_value)
            for item in observations
            if (grade_value := _grade_ocr_value(_text(item))) is not None
            and (
                left
                <= _box(item)[0] + _box(item)[2] / 2
                <= right
                and top
                <= _box(item)[1] + _box(item)[3] / 2
                <= bottom
            )
        )
        targeted: tuple[tuple[Any, int], ...] = ()
        if not matches:
            # Full-screen OCR can consistently omit the large stylised Grade
            # digit while still resolving the surrounding small menu text.
            # Reuse the same screenshot and label-derived ROI for one focused
            # pass; never widen the ROI toward the unrelated single-digit
            # notification badge on the right side of this page.
            targeted = tuple(
                (item, grade_value)
                for item in self._ocr(image, r"^[1-7V]$", roi=roi)
                if (grade_value := _grade_ocr_value(_text(item))) is not None
            )
            if len(targeted) == 1:
                return targeted[0][1]
        if len(matches) != 1:
            nearby = tuple(
                (_text(item), _box(item))
                for item in observations
                if (
                    left
                    <= _box(item)[0] + _box(item)[2] / 2
                    <= right
                    and label_y
                    <= _box(item)[1] + _box(item)[3] / 2
                    <= bottom
                )
            )
            raise ArenaReaderError(
                "arena_grade_ambiguous",
                f"found {len(matches)} geometrically valid Grade digits below the "
                f"GRADE anchor in {roi}; targeted_ocr="
                f"{tuple((_text(item), _box(item)) for item, _ in targeted)!r}; "
                f"nearby_ocr={nearby!r}",
            )
        return matches[0][1]

    def _assert_member_detail(self, timeout_seconds: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            image = self._capture()
            if self._ocr(image, r"^体力$") and self._ocr(image, r"^総合力$"):
                return
            time.sleep(0.25)
        raise ArenaReaderError(
            "member_detail_anchor_missing",
            "体力/総合力 anchors did not become visible",
        )

    def _assert_card_group_visible(
        self,
        group_index: int,
        timeout_seconds: float = 2.0,
        *,
        card_slot: int | None = None,
        expected_card_id: int | None = None,
    ) -> None:
        """Prove the original card generation is restored after a detail action.

        Geometry alone is insufficient because a fading detail overlay can
        expose the source row before it is ready for the next interaction.
        When the caller identifies the clicked slot, require that slot to
        match one of the three signatures from the already accepted source
        generation. A semantically identical card can nevertheless acquire a
        different 16x16 signature after the overlay closes (for example from
        sub-pixel row-box jitter). In that case an already resolved detail ID
        may authorize one stricter fallback: the same unique clean-reference
        identity must remain a high-confidence candidate in three consecutive
        source frames. The exact detail title has already resolved the business
        ID, so a stable clean-reference shortlist may confirm that ID without
        pretending the shortlist itself is unique. Pixel
        stability is deliberately not reused here: the fallback exists because
        the 16x16 content signature itself is unstable under sub-pixel row-box
        jitter. This replaces the former unconditional 250 ms sleep without
        accepting geometry alone; the timeout and failure semantics remain
        bounded and fail closed.
        """

        if card_slot is not None and not 1 <= card_slot <= 6:
            raise ArenaReaderError(
                "skill_card_source_slot_invalid",
                f"source restoration slot is outside 1..6: {card_slot}",
            )
        restoration_signatures = None
        if card_slot is not None:
            restoration_signatures = getattr(
                self,
                "_card_restoration_signatures",
                {},
            ).get(group_index)
            if (
                restoration_signatures is None
                or len(restoration_signatures) != 3
                or any(len(frame) != 6 for frame in restoration_signatures)
            ):
                raise ArenaReaderError(
                    "skill_card_source_generation_missing",
                    f"group {group_index} has no accepted three-frame restoration generation",
                )
        if expected_card_id is not None and (
            isinstance(expected_card_id, bool)
            or not isinstance(expected_card_id, int)
            or expected_card_id < 1
        ):
            raise ArenaReaderError(
                "skill_card_source_identity_invalid",
                f"expected source card ID is invalid: {expected_card_id!r}",
            )
        deadline = time.monotonic() + timeout_seconds
        started = time.perf_counter()
        last_error = ""
        reads = 0
        semantic_stable_frames = 0
        semantic_identity_frames: list[dict[str, Any]] = []
        accepted_row = tuple(
            getattr(self, "_card_rows", {}).get(group_index, ())
        )
        while time.monotonic() < deadline:
            try:
                image = self._capture()
                detected_row = self._validated_card_group_row(image, group_index)
                row = detected_row
                if len(accepted_row) == 6:
                    if self._card_rows_shifted(detected_row, accepted_row):
                        raise ArenaReaderError(
                            "skill_card_source_geometry_changed",
                            f"group {group_index} returned with geometry outside the "
                            f"accepted source row; accepted={accepted_row!r}; "
                            f"detected={detected_row!r}",
                        )
                    # Detector NMS may retain a dense set before the click and
                    # only two edge anchors after the overlay closes.  Those
                    # equivalent observations can differ by a pixel or two.
                    # Once the current frame proves the same fixed row within
                    # the existing geometry tolerance, crop the clicked slot
                    # with the pre-action row itself.  This keeps both the
                    # 16x16 generation signature and clean-reference identity
                    # bound to one physical ROI instead of letting detector
                    # sparsity redefine the transaction mid-close.
                    row = accepted_row
                    if detected_row != accepted_row:
                        self._increment(
                            "skill_card_source_restore_frozen_geometry"
                        )
                reads += 1
                self._increment("skill_card_source_restore_reads")
                if restoration_signatures is not None:
                    signature_started = time.perf_counter()
                    try:
                        visual_group, query, identity = (
                            self._card_references().content_signature_with_identity(
                                image,
                                row[card_slot - 1],
                            )
                        )
                    except BadgeReferenceError as error:
                        last_error = (
                            "source card signature was not measurable after detail close: "
                            f"{error}"
                        )
                        self._increment(
                            "skill_card_source_restore_signature_failures"
                        )
                        time.sleep(self._source_restore_poll_seconds)
                        continue
                    finally:
                        self._add_timing(
                            "skill_card_source_restore_signature",
                            time.perf_counter() - signature_started,
                        )
                    current = ((visual_group, query),)
                    source_matches = any(
                        not self._card_content_generation_shifted(
                            current,
                            (frame[card_slot - 1],),
                        )
                        for frame in restoration_signatures
                    )
                    if not source_matches:
                        measured_business_ids = self._reference_business_ids(identity)
                        semantic_identity_matches = bool(
                            expected_card_id is not None
                            and expected_card_id in measured_business_ids
                        )
                        if semantic_identity_matches:
                            semantic_stable_frames += 1
                            semantic_identity_frames.append(identity)
                            del semantic_identity_frames[:-3]
                        else:
                            semantic_stable_frames = 0
                            semantic_identity_frames.clear()
                        stable_business_ids: tuple[int, ...] = ()
                        if len(semantic_identity_frames) == 3:
                            try:
                                stable_business_ids = (
                                    stable_reference_business_candidates(
                                        tuple(semantic_identity_frames)
                                    )
                                )
                            except BadgeReferenceError:
                                stable_business_ids = ()
                        semantic_source_settled = bool(
                            expected_card_id is not None
                            and expected_card_id in stable_business_ids
                        )
                        if semantic_source_settled:
                            self._increment(
                                "skill_card_source_restore_semantic_settled_fallbacks"
                            )
                        else:
                            last_error = (
                                f"group {group_index}/slot {card_slot} is visible but "
                                "does not match the accepted source generation; "
                                f"expected_card_id={expected_card_id!r}; "
                                f"measured_card_ids={measured_business_ids!r}; "
                                f"semantic_consecutive_frames={semantic_stable_frames}; "
                                f"stable_card_ids={stable_business_ids!r}; "
                                f"identity_status={identity.get('status')!r}"
                            )
                            self._increment(
                                "skill_card_source_restore_generation_mismatches"
                            )
                            time.sleep(self._source_restore_poll_seconds)
                            continue
                self._card_rows[group_index] = row
                self._card_images[group_index] = image
                other_group = 1 - group_index
                if other_group in self._card_rows:
                    try:
                        other_row = self._validated_card_group_row(image, other_group)
                    except ArenaReaderError:
                        pass
                    else:
                        self._card_rows[other_group] = other_row
                        self._card_images[other_group] = image
                self._add_timing(
                    "skill_card_source_restore_wait",
                    time.perf_counter() - started,
                )
                if card_slot is not None and reads == 1:
                    self._increment("skill_card_source_restore_first_read")
                return
            except ArenaReaderError as error:
                last_error = str(error)
            time.sleep(self._source_restore_poll_seconds)
        self._add_timing(
            "skill_card_source_restore_wait",
            time.perf_counter() - started,
        )
        raise ArenaReaderError(
            "skill_card_close_failed",
            f"group {group_index} did not return after closing the card detail: {last_error}",
        )

    def _detect_card_rows(self, image: Any) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
        return canonical_skill_card_rows(image.shape, self._raw_card_candidate_boxes(image))

    def _validated_card_group_row(
        self,
        image: Any,
        group_index: int,
        *,
        allow_fixed_secondary: bool = False,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Require complete fixed-row anchors before treating badge absence as zero."""

        allow_fixed_secondary = bool(
            allow_fixed_secondary
            or (
                group_index == 1
                and getattr(
                    self,
                    "_secondary_fixed_slot_fallback_enabled",
                    False,
                )
            )
        )

        raw_boxes = self._raw_card_candidate_boxes(image)
        rows = canonical_skill_card_rows(image.shape, raw_boxes)
        selected = rows[0] if rows and group_index == 0 else ()
        if rows and group_index == 1:
            selected = canonical_scrolled_secondary_skill_card_row(
                image.shape,
                raw_boxes,
                rows[0],
            )
        if group_index == 1 and len(selected) != 6 and allow_fixed_secondary:
            expected_row = expected_scrolled_secondary_skill_card_row(
                image.shape,
                rows[0] if rows else (),
            )
            cache = getattr(self, "_secondary_presence_cache", None)
            if cache is None:
                cache = {}
                self._secondary_presence_cache = cache
            cached = cache.get(id(image))
            diagnostics = (
                cached[1]
                if cached is not None and cached[0]() is image
                else ()
            )
            if diagnostics:
                self._increment("secondary_presence_cache_hits")
            else:
                started = time.perf_counter()
                diagnostics = tuple(
                    self._secondary_fixed_slot_presence(image, box, slot)
                    for slot, box in enumerate(expected_row, start=1)
                )
                self._add_timing(
                    "secondary_fixed_slot_presence",
                    time.perf_counter() - started,
                )
                try:
                    from weakref import ref

                    cache[id(image)] = (ref(image), diagnostics)
                except TypeError:
                    pass
            self._secondary_presence_diagnostics = diagnostics
            if len(expected_row) == 6 and all(
                item["present"] is True or item["empty"] is True
                for item in diagnostics
            ):
                selected = expected_row
                self._increment("secondary_reference_presence_frames")
        if len(selected) != 6:
            suffix = (
                ""
                if group_index != 1 or not self._secondary_presence_diagnostics
                else f"; fixed-slot presence={self._secondary_presence_diagnostics!r}"
            )
            raise ArenaReaderError(
                "skill_card_customization_frame_incomplete",
                f"group {group_index} lacks complete card anchors on a repeated frame"
                f"{suffix}",
            )
        return selected

    def _secondary_fixed_slot_presence(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        slot: int,
    ) -> dict[str, Any]:
        """Prove one fixed second-row slot contains a card, not its identity.

        The ordinary detector remains the first route. This fallback runs only
        after the shifted row is geometrically implied by a fully scrolled
        primary row. The fixed clean-reference gallery must find a card-like
        image; this gate never accepts an ID, badge, or customization.
        """

        empty_observation = self._skill_card_empty_observation(image, box)
        if empty_observation["empty"]:
            return {
                "slot": slot,
                "present": False,
                "presence_path": "stable_blank_candidate",
                "reference_status": None,
                "top_error": None,
                "top_group": None,
                **empty_observation,
            }
        measurement = self._card_references().measure_identity(image, box)
        status = measurement.get("status")
        reference_like = status in {"MEASURED", "MEASURED_CANDIDATES"}
        return {
            "slot": slot,
            "present": reference_like,
            "empty": False,
            "presence_path": "clean_reference" if reference_like else "unresolved",
            "reference_status": status,
            "top_error": measurement.get(
                "identity_top_error",
                measurement.get("top_error"),
            ),
            "top_group": measurement.get(
                "reference_visual_group",
                measurement.get("top_group"),
            ),
        }

    @staticmethod
    def _skill_card_empty_observation(
        image: Any,
        box: tuple[int, int, int, int],
    ) -> dict[str, Any]:
        """Detect only a nearly uniform neutral/dark blank, never card art."""

        import cv2
        import numpy as np

        x, y, width, height = box
        crop = image[y : y + height, x : x + width, :3]
        if crop.size == 0 or width < 16 or height < 16:
            raise ArenaReaderError(
                "skill_card_empty_slot_input_invalid",
                f"skill-card slot crop is empty or too small: {box!r}",
            )
        inset = max(2, int(round(min(width, height) * 0.06)))
        inner = crop[inset:-inset, inset:-inset]
        if inner.size == 0:
            inner = crop
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 32, 96)
        gray_mean = float(np.mean(gray))
        gray_stddev = float(np.std(gray))
        gray_p95 = float(np.percentile(gray, 95))
        dark_pixel_ratio = float(np.mean(gray <= 24))
        edge_density = float(np.mean(edges > 0))
        channel_range = np.max(inner, axis=2) - np.min(inner, axis=2)
        mean_channel_range = float(np.mean(channel_range))
        uniform_neutral_placeholder = bool(
            mean_channel_range <= 8.0
            and gray_stddev <= 8.0
            and edge_density <= 0.01
        )
        uniform_dark_placeholder = bool(
            gray_mean <= 12.0
            and gray_stddev <= 6.0
            and gray_p95 <= 24.0
            and dark_pixel_ratio >= 0.97
            and edge_density <= 0.015
        )
        empty = bool(
            uniform_neutral_placeholder or uniform_dark_placeholder
        )
        return {
            "empty": empty,
            "uniform_neutral_placeholder": uniform_neutral_placeholder,
            "uniform_dark_placeholder": uniform_dark_placeholder,
            "gray_mean": round(gray_mean, 6),
            "gray_stddev": round(gray_stddev, 6),
            "gray_p95": round(gray_p95, 6),
            "dark_pixel_ratio": round(dark_pixel_ratio, 6),
            "edge_density": round(edge_density, 6),
            "mean_channel_range": round(mean_channel_range, 6),
        }

    def read_skill_card_empty_flags(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[bool]:
        """Classify stable enabled blank slots before duplicate or badge work."""

        del target, stage_number, member_slot
        row = self._card_rows.get(group_index, ())
        frames = self._card_count_frames.get(group_index, ())
        if len(row) != 6 or len(frames) != 3:
            raise ArenaReaderError(
                "skill_card_empty_slot_input_mismatch",
                f"group {group_index} has no stable six-slot three-frame input",
            )
        per_frame = tuple(
            tuple(
                self._skill_card_empty_observation(image, box)
                for box in row
            )
            for image in frames
        )
        flags: list[bool] = []
        diagnostics: list[dict[str, Any]] = []
        for slot_index in range(6):
            observations = tuple(frame[slot_index] for frame in per_frame)
            values = tuple(bool(item["empty"]) for item in observations)
            if len(set(values)) != 1:
                raise ArenaReaderError(
                    "skill_card_empty_slot_unstable",
                    f"group {group_index}/slot {slot_index + 1} blank state changed "
                    "inside one stable card generation",
                )
            flags.append(values[0])
            diagnostics.append(
                {
                    "slot": slot_index + 1,
                    "empty": values[0],
                    "frames": [dict(item) for item in observations],
                }
            )
        result = tuple(flags)
        empty_cache = getattr(self, "_card_empty_flags", None)
        if empty_cache is None:
            empty_cache = {}
            self._card_empty_flags = empty_cache
        diagnostic_cache = getattr(self, "_card_empty_diagnostics", None)
        if diagnostic_cache is None:
            diagnostic_cache = {}
            self._card_empty_diagnostics = diagnostic_cache
        empty_cache[group_index] = result
        diagnostic_cache[group_index] = tuple(diagnostics)
        for _ in range(sum(result)):
            self._increment("skill_card_empty_slots")
        return result

    def _raw_card_candidate_boxes(
        self,
        image: Any,
    ) -> tuple[tuple[int, int, int, int], ...]:
        detail = self.context.run_recognition(
            "ProduceRecognitionCards",
            image,
        )
        results = (detail.all_results or []) if detail and detail.hit else []
        return tuple(candidate.box for candidate in isolate_card_candidates(results))
