"""Card-face badge capture, worker calibration and task-domain binding."""

from __future__ import annotations

import json
from typing import Any, Protocol
from pathlib import Path
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from arena_winrate import (
    ArenaReaderError,
    ClickedSkillCard,
    BadgeReferenceError,
    BadgeReferenceGallery,
    CustomizationBadgeState,
    GenericCostReferenceError,
    CustomizationBadgeDecision,
    GenericCostReferenceGallery,
    measure_customization_badge,
    stable_customization_badge_presence,
)
from arena_winrate.badge_glyph_pool import BadgeGlyphDomain


class BadgeShortlist(Protocol):
    decision: CustomizationBadgeDecision
    observations: tuple[dict[str, Any], ...]


class BadgeCapturePort(Protocol):
    """Live reader state and callbacks used by this component; never copied."""

    def _add_timing(self, name: str, elapsed: float) -> None: ...

    _badge_candidate_detail_checks: Any

    _badge_glyph_count_diagnostics: Any

    _badge_glyph_domain: Any

    _badge_glyph_exemplars: Any

    @classmethod
    def _badge_glyph_frames_are_bounded_raster_settling(cls, observations: Sequence[dict[str, Any]]) -> bool: ...

    _badge_glyph_observations: Any

    @classmethod
    def _badge_glyph_observations_are_temporally_stable(cls, observations: Sequence[dict[str, Any]]) -> bool: ...

    @classmethod
    def _badge_glyph_observations_need_fresh_group(cls, observations: Sequence[dict[str, Any]]) -> bool: ...

    _badge_glyph_polluted_descriptors: Any

    _badge_glyph_pool_size: Any

    _badge_glyph_runtime_labels: Any

    _badge_glyph_task_pool: Any

    _badge_local_results: Any

    _badge_worker_calibrator: Any

    _badge_worker_selection: Any

    def _badge_workers(self, frames: Sequence[Any], jobs: Sequence[tuple[int, int]], operation: Any) -> int: ...

    def _bind_badge_glyph_task_domain(self, frames: Sequence[Any]) -> None: ...

    def _capture(self) -> Any: ...

    def _capture_one_fresh_badge_glyph_group(
        self, visible_groups: Sequence[int], related_jobs: Sequence[tuple[int, int]], *, workers: int
    ) -> tuple[Any, ...] | None: ...

    @classmethod
    def _card_content_generation_shifted(
        cls, actual: Sequence[tuple[str, Any]], expected: Sequence[tuple[str, Any]], *, maximum_mean_absolute_error: float = ...
    ) -> bool: ...

    def _card_content_generation_signatures(
        self, image: Any, rows: dict[int, tuple[tuple[int, int, int, int], ...]]
    ) -> tuple[dict[int, tuple[tuple[str, Any], ...]], dict[int, tuple[dict[str, Any], ...]]]: ...

    _card_cost_reference_gallery: Any

    _card_count_frames: Any

    _card_empty_flags: Any

    _card_excluded_duplicate_flags: Any

    _card_predictions: Any

    _card_reference_gallery: Any

    _card_restoration_signatures: Any

    _card_rows: Any

    @staticmethod
    def _card_rows_shifted(
        actual: Sequence[tuple[int, int, int, int]], expected: Sequence[tuple[int, int, int, int]], *, tolerance: int = ...
    ) -> bool: ...

    def _increment(self, name: str) -> None: ...

    def _load_static_resource(self, name: str, root: str | Path, factory: Callable[[], Any]) -> Any: ...

    def _local_badge_slot_decision(
        self, frames: Sequence[Any], frame_rows: Sequence[Sequence[tuple[int, int, int, int]]], group_index: int, slot_index: int
    ) -> BadgeShortlist: ...

    _member_failure_frames: Any

    @staticmethod
    def _parallel_badge_jobs(workers: int, jobs: Sequence[tuple[int, int]], operation: Any) -> tuple[Any, ...]: ...

    def _sleep(self, seconds: float) -> None: ...

    @staticmethod
    def _stable_badge_glyph_exemplar_signature(
        observations: Sequence[dict[str, Any]],
    ) -> tuple[tuple[int, ...], tuple[int, int, int, int, int], int] | None: ...

    def _validated_card_group_row(
        self, image: Any, group_index: int, *, allow_fixed_secondary: bool = ...
    ) -> tuple[tuple[int, int, int, int], ...]: ...

    def _validated_card_group_rows(self, image: Any, groups: Sequence[int], **kwargs): ...


def _parallel_badge_jobs(
    workers: int,
    jobs: Sequence[tuple[int, int]],
    operation: Any,
) -> tuple[Any, ...]:
    if workers <= 1:
        return tuple(operation(job) for job in jobs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return tuple(executor.map(operation, jobs))


def _badge_detail_inferable(decision: CustomizationBadgeDecision) -> bool:
    """Only the typed shortlist state may authorize a detail."""

    return decision.state is CustomizationBadgeState.DETAIL_CANDIDATE


def _badge_glyph_observations_are_temporally_stable(
    cls: BadgeCapturePort,
    observations: Sequence[dict[str, Any]],
) -> bool:
    return bool(
        cls._stable_badge_glyph_exemplar_signature(observations) is not None
        or cls._badge_glyph_frames_are_bounded_raster_settling(observations)
    )


def _badge_glyph_observations_need_fresh_group(
    cls: BadgeCapturePort,
    observations: Sequence[dict[str, Any]],
) -> bool:
    """Retry only three individually valid positive glyphs that disagree.

    Repeating each observation through the strict signature gate reuses its
    complete typed/geometry validation without turning a missing or
    fragmented glyph into a new prerequisite for opening card detail.
    """

    if len(observations) != 3:
        return False
    if not all(cls._stable_badge_glyph_exemplar_signature((observation, observation, observation)) is not None for observation in observations):
        return False
    return not cls._badge_glyph_observations_are_temporally_stable(observations)


def _badge_glyph_ocr_canvases(
    descriptor: Any,
) -> tuple[tuple[int, str, Any], ...]:
    """Render one centre-plate glyph mask for recognition-only OCR.

    The input is the quantized mask cut from the sole centre-seeded green
    component. No card pixels, frame, green artwork, template asset, or
    card identity participates in these canvases.  The recognizer needs a
    much wider quiet zone than ordinary scene OCR: the former fixed
    16-pixel border was only two cells at the 8x glyph scale and caused a
    real ``1`` to be decoded as CJK strokes.  Three fixed quiet zones and
    both polarities form six deterministic recognition views of the same
    bounded glyph; they are robustness transforms, not independent source
    observations.
    """

    import cv2
    import numpy as np

    values = np.asarray(descriptor)
    if values.shape != (216,) or values.dtype.kind not in "iu" or bool(np.any(values < 0)) or bool(np.any(values > 15)):
        raise ArenaReaderError(
            "skill_card_badge_glyph_invalid",
            "badge glyph descriptor must contain 216 integers in [0, 15]",
        )
    mask = (values.reshape(18, 12) > 0).astype(np.uint8) * 255
    active = mask > 0
    ys, xs = np.nonzero(active)
    if not len(xs):
        raise ArenaReaderError(
            "skill_card_badge_glyph_invalid",
            "badge glyph descriptor has no foreground",
        )
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    canvases: list[tuple[int, str, Any]] = []
    for quiet_cells in (8, 16, 24):
        padded = np.zeros(
            (
                crop.shape[0] + 2 * quiet_cells,
                crop.shape[1] + 2 * quiet_cells,
            ),
            dtype=np.uint8,
        )
        padded[
            quiet_cells : quiet_cells + crop.shape[0],
            quiet_cells : quiet_cells + crop.shape[1],
        ] = crop
        target_width = max(
            1,
            int(round(padded.shape[1] * 48 / padded.shape[0])),
        )
        normal = cv2.resize(
            padded,
            (target_width, 48),
            interpolation=cv2.INTER_CUBIC,
        )
        for polarity, canvas in (
            ("normal", normal),
            ("inverted", 255 - normal),
        ):
            canvases.append(
                (
                    quiet_cells,
                    polarity,
                    cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR),
                )
            )
    return tuple(canvases)


class BadgeCapture:
    """Stateless operation component over the reader's explicit port."""

    def __init__(
        self,
        port: BadgeCapturePort,
        *,
        clock: Any,
        logger: Any,
        reference_root: Path,
        cost_reference_root: Path,
        shortlist_result_type: Callable[..., BadgeShortlist],
        fresh_group_timeout_seconds: float,
        fresh_group_interval_seconds: float,
        fresh_group_max_frames: int,
    ) -> None:
        self.port = port
        self.clock = clock
        self.logger = logger
        self.reference_root = reference_root
        self.cost_reference_root = cost_reference_root
        self.shortlist_result_type = shortlist_result_type
        self.fresh_group_timeout_seconds = fresh_group_timeout_seconds
        self.fresh_group_interval_seconds = fresh_group_interval_seconds
        self.fresh_group_max_frames = fresh_group_max_frames

    def _card_references(self) -> BadgeReferenceGallery:
        if self.port._card_reference_gallery is not None:
            return self.port._card_reference_gallery
        try:
            self.port._card_reference_gallery = self.port._load_static_resource(
                "card_reference",
                self.reference_root,
                lambda: BadgeReferenceGallery.load(self.reference_root),
            )
        except (OSError, KeyError, TypeError, ValueError, BadgeReferenceError) as error:
            raise ArenaReaderError(
                "skill_card_reference_unavailable",
                "compact clean-card reference gallery is unavailable or invalid",
            ) from error
        return self.port._card_reference_gallery

    def _card_cost_references(self) -> GenericCostReferenceGallery:
        if self.port._card_cost_reference_gallery is not None:
            return self.port._card_cost_reference_gallery
        try:
            self.port._card_cost_reference_gallery = self.port._load_static_resource(
                "card_cost_reference",
                self.cost_reference_root,
                lambda: GenericCostReferenceGallery.load(self.cost_reference_root),
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
        return self.port._card_cost_reference_gallery

    def _local_badge_slot_decision(
        self,
        frames: Sequence[Any],
        frame_rows: Sequence[Sequence[tuple[int, int, int, int]]],
        group_index: int,
        slot_index: int,
    ) -> BadgeShortlist:
        repeated = tuple(measure_customization_badge(image, frame_row[slot_index]) for image, frame_row in zip(frames, frame_rows, strict=True))
        presence = stable_customization_badge_presence(repeated)
        return self.shortlist_result_type(
            decision=presence,
            observations=repeated,
        )

    def _badge_workers(
        self,
        frames: Sequence[Any],
        jobs: Sequence[tuple[int, int]],
        operation: Any,
    ) -> int:
        if self.port._badge_worker_selection is not None:
            return self.port._badge_worker_selection.workers
        signature = self.port._badge_worker_calibrator.signature(
            frame_shape=frames[0].shape,
            batch_slots=len(jobs),
            reference_version="seeded-plate-shortlist-v1",
            recognition_asset_signature="seeded-plate-detail-authority-v1",
        )

        def run_batch(workers: int) -> None:
            self.port._parallel_badge_jobs(workers, jobs, operation)

        started = self.clock.perf_counter()
        self.port._badge_worker_selection = self.port._badge_worker_calibrator.select(
            signature=signature,
            run_batch=run_batch,
        )
        self.port._add_timing("badge_worker_selection", self.clock.perf_counter() - started)
        return self.port._badge_worker_selection.workers

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
        self.port._badge_candidate_detail_checks.append(entry)
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
        self.port._increment("badge_candidate_positive_details" if count else "badge_candidate_zero_details")

    def _capture_one_fresh_badge_glyph_group(
        self,
        visible_groups: Sequence[int],
        related_jobs: Sequence[tuple[int, int]],
        *,
        workers: int,
    ) -> tuple[Any, ...] | None:
        """Return one atomic stable fresh window, or preserve the original one.

        The first eligible window is entirely separate from the initially
        accepted card frames.  Every additional frame must still match the
        frozen six-slot geometry and one complete source-generation signature
        for every visible row.  Unstable-but-individually-valid windows slide
        by one frame until the single deadline expires; the deadline is never
        extended and slots are never combined across different windows.
        """

        groups = tuple(visible_groups)
        jobs = tuple(related_jobs)
        if not groups or not jobs:
            return None
        accepted_rows = {group_index: tuple(self.port._card_rows.get(group_index, ())) for group_index in groups}
        frozen_signatures = {
            group_index: tuple(
                getattr(self.port, "_card_restoration_signatures", {}).get(
                    group_index,
                    (),
                )
            )
            for group_index in groups
        }
        initial_frame_ids = {id(image) for group_index in groups for image in self.port._card_count_frames.get(group_index, ())}
        if any(
            len(accepted_rows[group_index]) != 6
            or len(frozen_signatures[group_index]) != 3
            or any(len(frame) != 6 for frame in frozen_signatures[group_index])
            for group_index in groups
        ):
            return None

        self.port._increment("skill_card_badge_glyph_fresh_group_attempts")
        monotonic_started = self.clock.monotonic()
        deadline = monotonic_started + self.fresh_group_timeout_seconds
        next_capture_not_before = monotonic_started
        capture_times: list[float] = []
        frames: list[Any] = []
        frame_ids: set[int] = set()
        rows_by_group: dict[
            int,
            list[tuple[tuple[int, int, int, int], ...]],
        ] = {group_index: [] for group_index in groups}
        timing_started = self.clock.perf_counter()
        fresh_outcome, fresh_reason = "interrupted", None

        def reject(reason: str) -> None:
            nonlocal fresh_outcome, fresh_reason
            fresh_outcome, fresh_reason = "rejected", reason
            self.port._increment("skill_card_badge_glyph_fresh_group_rejections")
            self.port._increment(f"skill_card_badge_glyph_fresh_group_{reason}_rejections")
            return None

        try:
            for _ in range(self.fresh_group_max_frames):
                remaining = next_capture_not_before - self.clock.monotonic()
                if remaining > 0:
                    if self.clock.monotonic() + remaining >= deadline:
                        return reject("deadline")
                    self.port._sleep(remaining)
                if self.clock.monotonic() >= deadline:
                    return reject("deadline")
                capture_started = self.clock.monotonic()
                next_capture_not_before = capture_started + self.fresh_group_interval_seconds
                try:
                    image = self.port._capture()
                except ArenaReaderError:
                    return reject("capture")
                self.port._increment("skill_card_badge_glyph_fresh_group_reads")
                if self.clock.monotonic() >= deadline:
                    return reject("deadline")
                image_id = id(image)
                if image_id in initial_frame_ids or image_id in frame_ids:
                    return reject("duplicate")
                frame_ids.add(image_id)
                frames.append(image)
                capture_times.append(capture_started)
                try:
                    observed = self.port._validated_card_group_rows(image, groups)
                except ArenaReaderError:
                    return reject("source")
                if any(
                    self.port._card_rows_shifted(
                        observed[group_index],
                        accepted_rows[group_index],
                    )
                    for group_index in groups
                ):
                    return reject("source")
                if self.clock.monotonic() >= deadline:
                    return reject("deadline")
                try:
                    current_signatures, _ = self.port._card_content_generation_signatures(
                        image,
                        observed,
                    )
                except (ArenaReaderError, BadgeReferenceError):
                    return reject("source")
                if any(
                    not any(
                        not self.port._card_content_generation_shifted(
                            current_signatures[group_index],
                            frozen_frame,
                        )
                        for frozen_frame in frozen_signatures[group_index]
                    )
                    for group_index in groups
                ):
                    return reject("source")
                if self.clock.monotonic() >= deadline:
                    return reject("deadline")
                for group_index in groups:
                    rows_by_group[group_index].append(observed[group_index])

                if len(frames) < 3:
                    continue
                required_span = self.fresh_group_interval_seconds * 2
                if capture_times[-1] - capture_times[-3] + 1e-6 < required_span:
                    return reject("deadline")
                stable_frames = tuple(frames[-3:])
                stable_rows_by_group = {group_index: tuple(rows_by_group[group_index][-3:]) for group_index in groups}

                def operation(job: tuple[int, int]) -> Any:
                    group_index, slot_index = job
                    return self.port._local_badge_slot_decision(
                        stable_frames,
                        stable_rows_by_group[group_index],
                        group_index,
                        slot_index,
                    )

                try:
                    fresh_results = self.port._parallel_badge_jobs(
                        workers,
                        jobs,
                        operation,
                    )
                except (ArenaReaderError, ValueError):
                    return reject("measurement")
                if self.clock.monotonic() >= deadline:
                    return reject("deadline")
                self.port._increment("skill_card_badge_glyph_fresh_group_windows")
                if any(result.decision.state is not CustomizationBadgeState.DETAIL_CANDIDATE for result in fresh_results):
                    return reject("temporal")
                stable_results = tuple(
                    self.port._badge_glyph_observations_are_temporally_stable(result.observations) for result in fresh_results
                )
                if all(stable_results):
                    self.port._increment("skill_card_badge_glyph_fresh_group_acceptances")
                    fresh_outcome = "accepted"
                    return fresh_results
                if any(
                    not stable and not self.port._badge_glyph_observations_need_fresh_group(result.observations)
                    for stable, result in zip(
                        stable_results,
                        fresh_results,
                        strict=True,
                    )
                ):
                    return reject("temporal")
                self.port._increment("skill_card_badge_glyph_fresh_group_unstable_windows")
            return reject("deadline")
        finally:
            self.port._add_timing(
                "badge_glyph_fresh_group_wait",
                self.clock.perf_counter() - timing_started,
            )
            member = getattr(self.port, "_member_failure_frames", None) or {}
            self.logger.info(
                json.dumps(
                    {
                        "event": "arena_badge_fresh_group_result",
                        **member.get("position", {}),
                        "member_read_id": member.get("read_id"),
                        "outcome": fresh_outcome,
                        "reason": fresh_reason,
                        "duration_seconds": round(self.clock.perf_counter() - timing_started, 6),
                        "related_slots": [
                            {
                                "group_index": group,
                                "card_slot": slot + 1,
                                "candidate_card_id": getattr(self.port, "_card_predictions", {}).get((group, slot + 1)),
                            }
                            for group, slot in jobs
                        ],
                    },
                    ensure_ascii=False,
                )
            )

    def _batched_badge_local_results(
        self,
        requested_group: int,
    ) -> tuple[Any, ...]:
        self.port._bind_badge_glyph_task_domain(self.port._card_count_frames.get(requested_group, ()))
        cached = self.port._badge_local_results.get(requested_group)
        if cached is not None:
            return cached
        visible_groups = tuple(
            index for index in (0, 1) if len(self.port._card_rows.get(index, ())) == 6 and len(self.port._card_count_frames.get(index, ())) == 3
        )
        if requested_group not in visible_groups:
            raise ArenaReaderError(
                "skill_card_badge_batch_missing",
                f"group {requested_group} has no stable three-frame card observation",
            )
        frame_rows = {
            group_index: tuple(self.port._validated_card_group_row(image, group_index) for image in self.port._card_count_frames[group_index])
            for group_index in visible_groups
        }
        duplicate_flags = {
            group_index: self.port._card_excluded_duplicate_flags.get(
                group_index,
                (False,) * 6,
            )
            for group_index in visible_groups
        }
        empty_flags = {
            group_index: getattr(self.port, "_card_empty_flags", {}).get(
                group_index,
                (False,) * 6,
            )
            for group_index in visible_groups
        }
        jobs = tuple((group_index, slot_index) for group_index in visible_groups for slot_index in range(6))

        def operation(job: tuple[int, int]) -> Any:
            group_index, slot_index = job
            if duplicate_flags[group_index][slot_index] or empty_flags[group_index][slot_index]:
                return None
            return self.port._local_badge_slot_decision(
                self.port._card_count_frames[group_index],
                frame_rows[group_index],
                group_index,
                slot_index,
            )

        workers = self.port._badge_workers(
            self.port._card_count_frames[requested_group],
            jobs,
            operation,
        )
        started = self.clock.perf_counter()
        results = self.port._parallel_badge_jobs(workers, jobs, operation)
        self.port._add_timing("badge_local_batch", self.clock.perf_counter() - started)
        grouped: dict[int, list[Any]] = {group_index: [None] * 6 for group_index in visible_groups}
        for (group_index, slot_index), result in zip(jobs, results, strict=True):
            grouped[group_index][slot_index] = result
        related_jobs = tuple(
            (group_index, slot_index)
            for group_index, slot_index in jobs
            if (
                (result := grouped[group_index][slot_index]) is not None
                and result.decision.state is CustomizationBadgeState.DETAIL_CANDIDATE
                and self.port._badge_glyph_observations_need_fresh_group(result.observations)
            )
        )
        if related_jobs:
            fresh_results = self.port._capture_one_fresh_badge_glyph_group(
                visible_groups,
                related_jobs,
                workers=workers,
            )
            if fresh_results is not None:
                for (group_index, slot_index), result in zip(
                    related_jobs,
                    fresh_results,
                    strict=True,
                ):
                    grouped[group_index][slot_index] = result
        for group_index, values in grouped.items():
            self.port._badge_local_results[group_index] = tuple(values)
            for slot_index, result in enumerate(values, start=1):
                if result is not None:
                    self.port._badge_glyph_observations[(group_index, slot_index)] = result.observations
        return self.port._badge_local_results[requested_group]

    def _bind_badge_glyph_task_domain(self, frames: Sequence[Any]) -> None:
        """Select a task pool with existing frame metadata, without new input."""

        pool = getattr(self.port, "_badge_glyph_task_pool", None)
        if pool is None or not frames:
            return
        sizes = {tuple(getattr(image, "shape", ())[:2]) for image in frames}
        if len(sizes) != 1:
            size = None
        else:
            size = next(iter(sizes))
            if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
                size = None
        if size is not None and size == self.port._badge_glyph_pool_size and pool.active:
            return
        domain = pool.domain(size) if size is not None else BadgeGlyphDomain()
        self.port._badge_glyph_pool_size = size
        self.port._badge_glyph_domain = domain
        self.port._badge_glyph_exemplars = domain.exemplars
        self.port._badge_glyph_polluted_descriptors = domain.polluted
        # These are decisions about the old captured member, never task truth.
        self.port._badge_glyph_runtime_labels.clear()
        self.port._badge_glyph_count_diagnostics.clear()
        self.port._badge_local_results.clear()
        self.port._badge_glyph_observations.clear()
