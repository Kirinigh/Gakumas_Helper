"""Card layout component; no imports from the Maa facade."""

from __future__ import annotations

import json
from typing import Any, Protocol
from collections.abc import Callable, Sequence

from arena_winrate import (
    TeamTarget,
    ArenaReaderError,
    BadgeReferenceGallery,
    canonical_skill_card_rows,
    stable_excluded_duplicate_card_flags,
    expected_scrolled_secondary_skill_card_row,
    canonical_scrolled_secondary_skill_card_row,
)
from card_selection.model import isolate_card_candidates
from arena_winrate._reader_visual import _text


class CardLayoutReaderPort(Protocol):
    """Only the state and callbacks needed by this responsibility."""

    def _add_timing(self, name: str, elapsed: float) -> None: ...

    _badge_local_results: Any

    def _capture(self) -> Any: ...
    def _capture_stable_card_groups(
        self,
        group_indices: Sequence[int],
        *,
        timeout_seconds: float = ...,
        maximum_timeout_seconds: float = ...,
        confirmation_grace_seconds: float = ...,
        interval_seconds: float = ...,
        allow_fixed_secondary: bool = ...,
    ) -> None: ...

    _card_content_generation_deltas: Any
    _card_content_generation_diagnostics: Any

    def _card_content_generation_signatures(
        self, image: Any, rows: dict[int, tuple[tuple[int, int, int, int], ...]]
    ) -> tuple[dict[int, tuple[tuple[str, Any], ...]], dict[int, tuple[dict[str, Any], ...]]]: ...

    _card_count_frames: Any
    _card_detection_observation: Any
    _card_duplicate_marker_diagnostics: Any
    _card_empty_diagnostics: Any
    _card_empty_flags: Any
    _card_excluded_duplicate_flags: Any
    _card_identity_frames: Any
    _card_images: Any

    def _card_references(self) -> BadgeReferenceGallery: ...

    _card_restoration_signatures: Any
    _card_rows: Any
    _card_rows_shifted: Any
    _card_source_guard_frames: Any
    _card_source_guard_signature_cache: Any
    _card_source_ocr_counts: Any

    def _check_cancelled(self) -> None: ...
    def _detect_raw_card_candidate_boxes(self, image: Any) -> tuple[tuple[int, int, int, int], ...]: ...
    def _fixed_gallery_visual_groups_for_business_id(self, business_id: int | None) -> tuple[str, ...] | None: ...
    def _increment(self, name: str) -> None: ...
    def _ocr(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = ..., only_rec: bool = ...) -> list[Any]: ...
    def _raw_card_candidate_boxes(self, image: Any) -> tuple[tuple[int, int, int, int], ...]: ...
    def _reader_compute(self, metric: str, operation: Callable[[], Any]) -> Any: ...
    def _refresh_visible_card_groups(self, requested_group: int) -> tuple[Any, Any, Any]: ...
    def _run_recognition(self, *args, **kwargs): ...

    _runtime_counts: Any
    _runtime_timing_seconds: Any
    _secondary_fixed_slot_fallback_enabled: Any

    def _secondary_fixed_slot_presence(self, image: Any, box: tuple[int, int, int, int], slot: int) -> dict[str, Any]: ...

    _secondary_presence_cache: Any
    _secondary_presence_diagnostics: Any
    _skill_card_empty_observation: Any

    def _skill_card_scope_kwargs(self, slot_index: int, *, plan: str | None = ...) -> dict[str, Any]: ...
    def _sleep(self, seconds: float) -> None: ...
    def _swipe(self, *, vertical: str, distance_ratio: float = ..., distance_pixels: int | None = ...) -> None: ...
    def _targeted_duplicate_marker_observations(
        self,
        frames: Sequence[Any],
        row: Sequence[tuple[int, int, int, int]],
        target_slots: Sequence[int],
        *,
        phase: str,
        visual_features_by_slot: Sequence[Sequence[dict[str, Any]]] | None = ...,
    ) -> tuple[tuple[tuple[bool, ...], ...], list[dict[str, Any]]]: ...
    def _validated_card_group_row(
        self, image: Any, group_index: int, *, allow_fixed_secondary: bool = ...
    ) -> tuple[tuple[int, int, int, int], ...]: ...
    def _validated_card_group_rows(self, image: Any, groups: Sequence[int], **kwargs): ...


class CardLayoutReader:
    """Card layout with live, reader-owned state."""

    def __init__(
        self, port: CardLayoutReaderPort, *, duplicate_marker_visual_features: Any, is_duplicate_marker_visual_candidate: Any, clock: Any
    ) -> None:
        self.port = port
        self.duplicate_marker_visual_features = duplicate_marker_visual_features
        self.is_duplicate_marker_visual_candidate = is_duplicate_marker_visual_candidate
        self.clock = clock

    def prepare_skill_card_group(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> None:
        required_groups = (0, 1) if group_index == 1 else (0,)
        if all(
            index in self.port._card_rows and index in self.port._card_images and len(self.port._card_count_frames.get(index, ())) == 3
            for index in required_groups
        ):
            return
        if group_index == 1:
            self.port._swipe(vertical="up")
        first_error: ArenaReaderError | None = None
        try:
            self.port._capture_stable_card_groups(
                required_groups,
                allow_fixed_secondary=group_index == 1,
            )
            if group_index == 1:
                self.port._secondary_fixed_slot_fallback_enabled = True
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
        image = self.port._capture()
        rows = canonical_skill_card_rows(
            image.shape,
            self.port._raw_card_candidate_boxes(image),
        )
        assert first_error is not None
        if rows and rows[0][0][1] <= int(image.shape[0] * 0.205):
            if first_error.code == "skill_card_layout_incomplete":
                raise ArenaReaderError(
                    "skill_card_secondary_row_missing",
                    f"primary row reached the calibrated position but group 1 lacked six stable fixed-slot card observations: {first_error}",
                ) from first_error
            raise first_error
        self.port._swipe(vertical="up")
        self.port._increment("corrective_swipes")
        self.port._capture_stable_card_groups(
            required_groups,
            allow_fixed_secondary=True,
        )
        self.port._secondary_fixed_slot_fallback_enabled = True

    def _source_visual_group_matches_expected_card(
        self,
        source_visual_group: str,
        expected_card_id: int | None,
        source_business_ids: Sequence[int],
    ) -> bool:
        """Gate visual-only restore to the expected gallery family or a new ID."""

        if expected_card_id is None:
            return True
        expected_groups = self.port._fixed_gallery_visual_groups_for_business_id(expected_card_id)
        if expected_groups is None:
            return expected_card_id in source_business_ids
        if not expected_groups:
            # The active RIS catalog may legitimately be newer than the fixed
            # visual gallery. Exact title evidence is then the business-ID
            # authority and the frozen source family remains the close guard.
            return True
        return expected_groups == (source_visual_group,)

    def _fixed_gallery_visual_groups_for_business_id(
        self,
        business_id: int | None,
    ) -> tuple[str, ...] | None:
        """Return the fixed-gallery families for one ID, or ``None`` if unknown."""

        if business_id is None:
            return ()
        gallery = self.port._card_references()
        raw_business_ids = getattr(gallery, "business_ids", None)
        raw_visual_groups = getattr(gallery, "visual_group_ids", None)
        if raw_business_ids is None or raw_visual_groups is None:
            return None
        try:
            business_ids = tuple(int(value) for value in raw_business_ids)
            visual_groups = tuple(str(value) for value in raw_visual_groups)
        except (TypeError, ValueError):
            return None
        if len(business_ids) != len(visual_groups):
            return None
        return tuple(
            sorted(
                {
                    visual_group
                    for candidate_id, visual_group in zip(
                        business_ids,
                        visual_groups,
                        strict=True,
                    )
                    if candidate_id == business_id and visual_group.strip()
                }
            )
        )

    def _card_content_generation_signatures(
        self,
        image: Any,
        rows: dict[int, tuple[tuple[int, int, int, int], ...]],
    ) -> tuple[
        dict[int, tuple[tuple[str, Any], ...]],
        dict[int, tuple[dict[str, Any], ...]],
    ]:
        gallery = self.port._card_references()
        signatures: dict[int, tuple[tuple[str, Any], ...]] = {}
        identities: dict[int, tuple[dict[str, Any], ...]] = {}
        for group_index, row in rows.items():
            measured = tuple(
                gallery.content_signature_with_identity(
                    image,
                    box,
                    **self.port._skill_card_scope_kwargs(slot_index),
                )
                for slot_index, box in enumerate(row)
            )
            signatures[group_index] = tuple((visual_group, query) for visual_group, query, _ in measured)
            identities[group_index] = tuple(identity for _, _, identity in measured)
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
            raise ValueError("maximum_timeout_seconds must be at least the positive base timeout")
        if confirmation_grace_seconds < 0:
            raise ValueError("confirmation_grace_seconds cannot be negative")
        frames: list[Any] = []
        rows_by_group: dict[int, list[tuple[tuple[int, int, int, int], ...]]] = {index: [] for index in groups}
        content_by_group: dict[int, list[tuple[tuple[str, Any], ...]]] = {index: [] for index in groups}
        identity_by_group: dict[int, list[tuple[dict[str, Any], ...]]] = {index: [] for index in groups}
        monotonic_started = self.clock.monotonic()
        deadline = monotonic_started + timeout_seconds
        absolute_deadline = monotonic_started + maximum_timeout_seconds
        next_capture_not_before = monotonic_started
        capture_times: list[float] = []
        started = self.clock.perf_counter()
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

        while self.clock.monotonic() < deadline:
            remaining = next_capture_not_before - self.clock.monotonic()
            while remaining > 0:
                self.port._sleep(remaining)
                remaining = next_capture_not_before - self.clock.monotonic()
            if self.clock.monotonic() >= deadline:
                break
            capture_started = self.clock.monotonic()
            next_capture_not_before = capture_started + interval_seconds
            image = self.port._capture()
            try:
                observed = self.port._validated_card_group_rows(
                    image,
                    groups,
                    allow_fixed_secondary=allow_fixed_secondary,
                )
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
                content_started = self.clock.perf_counter()
                content, identities = self.port._card_content_generation_signatures(
                    image,
                    observed,
                )
                self.port._add_timing(
                    "card_content_stability_check",
                    self.clock.perf_counter() - content_started,
                )
                geometry_shifted = bool(frames) and any(
                    self.port._card_rows_shifted(observed[index], previous_row) for index in groups for previous_row in rows_by_group[index]
                )
                content_deltas = (
                    {
                        index: self.port._card_content_generation_deltas(
                            content[index],
                            content_by_group[index][-1],
                        )
                        for index in groups
                    }
                    if frames
                    else {}
                )
                content_shifted = any(bool(delta["shifted"]) for deltas in content_deltas.values() for delta in deltas)
                if geometry_shifted or content_shifted:
                    if geometry_shifted:
                        self.port._increment("skill_card_layout_geometry_drift_resets")
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
                                aggregate["visual_group_changes"] += int(bool(delta["visual_group_changed"]))
                                aggregate["shape_changes"] += int(bool(delta["shape_changed"]))
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
                                        float(aggregate["minimum_structural_correlation"]),
                                        float(correlation),
                                    )
                        last_error = "card content generation changed before three-frame stability; instability=" + json.dumps(
                            instability_summary(),
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        self.port._increment("card_content_generation_resets")
                    elif geometry_shifted:
                        last_failure_code = "skill_card_geometry_unstable"
                        last_error = "card geometry changed before three-frame stability"
                    extension_deadline = min(
                        absolute_deadline,
                        self.clock.monotonic() + confirmation_grace_seconds,
                    )
                    if extension_deadline > deadline:
                        deadline = extension_deadline
                        self.port._increment("card_stability_deadline_extensions")
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
                    self.port._card_content_generation_diagnostics = instability_summary()
                    self.port._badge_local_results.clear()
                    for index in groups:
                        self.port._card_rows[index] = rows_by_group[index][0]
                        self.port._card_images[index] = stable_frames[0]
                        self.port._card_count_frames[index] = stable_frames
                        guard_store = getattr(
                            self.port,
                            "_card_source_guard_frames",
                            None,
                        )
                        if guard_store is None:
                            guard_store = {}
                            self.port._card_source_guard_frames = guard_store
                        guard_store[index] = stable_frames
                        # A new accepted source generation invalidates every
                        # cached source-side overlay signature.  Clearing here
                        # also prevents Python object-ID reuse from ever
                        # binding a later generation to stale signatures.
                        getattr(
                            self.port,
                            "_card_source_guard_signature_cache",
                            {},
                        ).clear()
                        self.port._card_identity_frames[index] = tuple(identity_by_group[index])
                        getattr(
                            self.port,
                            "_card_source_ocr_counts",
                            {},
                        ).pop(index, None)
                        # Preserve the accepted three-frame generation itself.
                        # Detail dismissal may then prove that the original
                        # source card has returned without a fixed sleep.  All
                        # three accepted signatures remain valid baselines so
                        # harmless render jitter does not create a false stop.
                        self.port._card_restoration_signatures[index] = tuple(content_by_group[index])
                    self.port._add_timing(
                        "card_layout_stable_wait",
                        self.clock.perf_counter() - started,
                    )
                    self.port._add_timing(
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
            remaining = max(0.0, next_capture_not_before - self.clock.monotonic())
            self.port._add_timing(
                "card_stability_processing_overlap",
                max(0.0, interval_seconds - remaining),
            )
        self.port._add_timing("card_layout_stable_wait", self.clock.perf_counter() - started)
        self.port._card_content_generation_diagnostics = instability_summary()
        raise ArenaReaderError(
            last_failure_code,
            f"card groups {groups!r} did not yield three consecutive stable frames: {last_error}",
        )

    def _refresh_visible_card_groups(self, requested_group: int) -> tuple[Any, Any, Any]:
        self.port._increment("corrective_card_frame_reads")
        groups = tuple(index for index in (0, 1) if len(self.port._card_rows.get(index, ())) == 6)
        if requested_group not in groups:
            groups = (requested_group,)
        self.port._capture_stable_card_groups(groups)
        frames = self.port._card_count_frames.get(requested_group, ())
        if len(frames) != 3:
            raise ArenaReaderError(
                "skill_card_layout_unstable",
                f"group {requested_group} was not cached after a stable refresh",
            )
        return frames

    def _targeted_duplicate_marker_observations(
        self,
        frames: Sequence[Any],
        row: Sequence[tuple[int, int, int, int]],
        target_slots: Sequence[int],
        *,
        phase: str,
        visual_features_by_slot: Sequence[Sequence[dict[str, Any]]] | None = None,
    ) -> tuple[tuple[tuple[bool, ...], ...], list[dict[str, Any]]]:
        import cv2

        observations = []
        diagnostics = []
        targets = set(int(index) for index in target_slots)
        for frame_index, image in enumerate(frames):
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
                visual_features = (
                    visual_features_by_slot[index][frame_index]
                    if visual_features_by_slot is not None
                    else self.duplicate_marker_visual_features(card_crop)
                )
                texts_by_variant: dict[str, list[str]] = {}
                if crop.size:
                    enlarged = cv2.resize(
                        crop,
                        None,
                        fx=3.0,
                        fy=3.0,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    if hasattr(self.port, "_runtime_counts"):
                        self.port._increment("skill_card_duplicate_marker_ocr_requests")
                        self.port._increment("skill_card_duplicate_marker_ocr_backend_calls")
                    started = self.clock.perf_counter()
                    try:
                        texts = [_text(item) for item in self.port._ocr(enlarged, r".+")]
                    finally:
                        if hasattr(self.port, "_runtime_timing_seconds"):
                            self.port._add_timing(
                                "skill_card_duplicate_marker_ocr",
                                self.clock.perf_counter() - started,
                            )
                    texts_by_variant["3x"] = texts
                    if any(marker in text for text in texts for marker in ("重複", "制限")):
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
        row = self.port._card_rows.get(group_index, ())
        frames = self.port._card_count_frames.get(group_index, ())
        if len(row) != 6:
            raise ArenaReaderError(
                "skill_card_duplicate_marker_input_mismatch",
                f"group {group_index} has no stable six-card row",
            )
        if len(frames) != 3:
            frames = self.port._refresh_visible_card_groups(group_index)
            frame_rows = tuple(self.port._validated_card_group_row(image, group_index) for image in frames)
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
            self.port._card_count_frames[group_index] = frames

        repeated_visual_features = tuple(
            tuple(self.duplicate_marker_visual_features(image[y : y + height, x : x + width, :3]) for image in frames)
            for x, y, width, height in row
        )
        empty_flags = getattr(self.port, "_card_empty_flags", {}).get(
            group_index,
            (False,) * 6,
        )
        candidate_slots = tuple(
            index
            for index, features in enumerate(repeated_visual_features)
            if not empty_flags[index] and self.is_duplicate_marker_visual_candidate(features)
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
        targeted, diagnostics = self.port._targeted_duplicate_marker_observations(
            frames,
            row,
            candidate_slots,
            phase="targeted",
            visual_features_by_slot=repeated_visual_features,
        )
        diagnostics.insert(0, visual_diagnostic)
        stable = stable_excluded_duplicate_card_flags(targeted)
        unresolved = tuple(index for index, flag in enumerate(stable) if flag is None)
        if unresolved:
            corrective_frames_tuple = self.port._refresh_visible_card_groups(group_index)
            corrective_rows = tuple(self.port._validated_card_group_row(image, group_index) for image in corrective_frames_tuple)
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
            corrective_targeted, corrective_targeted_diagnostics = self.port._targeted_duplicate_marker_observations(
                corrective_frames_tuple,
                row,
                unresolved,
                phase="corrective_targeted",
            )
            diagnostics.extend(corrective_targeted_diagnostics)
            stable = stable_excluded_duplicate_card_flags(corrective_targeted)
            self.port._card_count_frames[group_index] = corrective_frames_tuple
        unknown_slots = tuple(index for index, flag in enumerate(stable, start=1) if flag is None)
        self.port._card_duplicate_marker_diagnostics[group_index] = tuple(diagnostics)
        if unknown_slots:
            raise ArenaReaderError(
                "skill_card_duplicate_marker_unstable",
                f"group {group_index}/slots {unknown_slots} remained ambiguous after one targeted and one corrective 重複 read",
            )
        flags = tuple(bool(flag) for flag in stable)
        self.port._card_excluded_duplicate_flags[group_index] = flags
        return flags

    def current_skill_card_excluded_duplicate_flags(
        self,
        group_index: int,
    ) -> Sequence[bool]:
        """Return OCR flags plus any functionally proven non-interactive slot."""

        flags = self.port._card_excluded_duplicate_flags.get(group_index)
        if flags is None or len(flags) != 6:
            raise ArenaReaderError(
                "skill_card_duplicate_marker_input_mismatch",
                f"group {group_index} has no current six-slot duplicate state",
            )
        return flags

    def _detect_card_rows(self, image: Any) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
        return canonical_skill_card_rows(image.shape, self.port._raw_card_candidate_boxes(image))

    def _validated_card_group_rows(self, image: Any, groups: Sequence[int], **kwargs):
        """Share detector output only during this synchronous frame observation."""
        previous = getattr(self.port, "_card_detection_observation", None)
        self.port._card_detection_observation = [image, None]
        try:
            return {index: self.port._validated_card_group_row(image, index, **kwargs) for index in groups}
        finally:
            self.port._card_detection_observation = previous

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
                    self.port,
                    "_secondary_fixed_slot_fallback_enabled",
                    False,
                )
            )
        )

        raw_boxes = self.port._raw_card_candidate_boxes(image)
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
            cache = getattr(self.port, "_secondary_presence_cache", None)
            if cache is None:
                cache = {}
                self.port._secondary_presence_cache = cache
            cached = cache.get(id(image))
            diagnostics = cached[1] if cached is not None and cached[0]() is image else ()
            if diagnostics:
                self.port._increment("secondary_presence_cache_hits")
            else:
                started = self.clock.perf_counter()
                diagnostics = tuple(
                    self.port._secondary_fixed_slot_presence(image, box, slot) for slot, box in enumerate(expected_row, start=1)
                )
                self.port._add_timing(
                    "secondary_fixed_slot_presence",
                    self.clock.perf_counter() - started,
                )
                try:
                    from weakref import ref

                    cache[id(image)] = (ref(image), diagnostics)
                except TypeError:
                    pass
            self.port._secondary_presence_diagnostics = diagnostics
            if len(expected_row) == 6 and all(item["present"] is True or item["empty"] is True for item in diagnostics):
                selected = expected_row
                self.port._increment("secondary_reference_presence_frames")
        if len(selected) != 6:
            suffix = (
                ""
                if group_index != 1 or not self.port._secondary_presence_diagnostics
                else f"; fixed-slot presence={self.port._secondary_presence_diagnostics!r}"
            )
            raise ArenaReaderError(
                "skill_card_customization_frame_incomplete",
                f"group {group_index} lacks complete card anchors on a repeated frame{suffix}",
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

        empty_observation = self.port._skill_card_empty_observation(image, box)
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
        measurement = self.port._card_references().measure_identity(image, box)
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

    def read_skill_card_empty_flags(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[bool]:
        """Classify stable enabled blank slots before duplicate or badge work."""

        del target, stage_number, member_slot
        row = self.port._card_rows.get(group_index, ())
        frames = self.port._card_count_frames.get(group_index, ())
        if len(row) != 6 or len(frames) != 3:
            raise ArenaReaderError(
                "skill_card_empty_slot_input_mismatch",
                f"group {group_index} has no stable six-slot three-frame input",
            )
        per_frame = tuple(tuple(self.port._skill_card_empty_observation(image, box) for box in row) for image in frames)
        flags: list[bool] = []
        diagnostics: list[dict[str, Any]] = []
        for slot_index in range(6):
            observations = tuple(frame[slot_index] for frame in per_frame)
            values = tuple(bool(item["empty"]) for item in observations)
            if len(set(values)) != 1:
                raise ArenaReaderError(
                    "skill_card_empty_slot_unstable",
                    f"group {group_index}/slot {slot_index + 1} blank state changed inside one stable card generation",
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
        empty_cache = getattr(self.port, "_card_empty_flags", None)
        if empty_cache is None:
            empty_cache = {}
            self.port._card_empty_flags = empty_cache
        diagnostic_cache = getattr(self.port, "_card_empty_diagnostics", None)
        if diagnostic_cache is None:
            diagnostic_cache = {}
            self.port._card_empty_diagnostics = diagnostic_cache
        empty_cache[group_index] = result
        diagnostic_cache[group_index] = tuple(diagnostics)
        for _ in range(sum(result)):
            self.port._increment("skill_card_empty_slots")
        return result

    def _raw_card_candidate_boxes(
        self,
        image: Any,
    ) -> tuple[tuple[int, int, int, int], ...]:
        self.port._check_cancelled()
        observation = getattr(self.port, "_card_detection_observation", None)
        if observation is not None and observation[0] is image and observation[1] is not None:
            self.port._increment("card_detection_reuse_hits")
            return observation[1]
        boxes = self.port._reader_compute(
            "card_detection",
            lambda: self.port._detect_raw_card_candidate_boxes(image),
        )
        if observation is not None and observation[0] is image:
            observation[1] = boxes
        return boxes

    def _detect_raw_card_candidate_boxes(
        self,
        image: Any,
    ) -> tuple[tuple[int, int, int, int], ...]:
        import cv2

        # cards.onnx was validated with a stretched 640-square input. Maa 5.13
        # now letterboxes non-square inputs, so preserve that model contract
        # here without changing the shared detector or taking another frame.
        height, width = image.shape[:2]
        detector_image = cv2.resize(
            image,
            (640, 640),
            interpolation=cv2.INTER_AREA,
        )
        detail = self.port._run_recognition(
            "ProduceRecognitionCards",
            detector_image,
        )
        results = (detail.all_results or []) if detail and detail.hit else []
        # Match Maa's integer rectangle conversion: truncate each coordinate
        # and extent independently. Keep the source frame's coordinate system
        # for the existing row geometry and subsequent card crops/clicks.
        scale_x, scale_y = width / 640, height / 640
        return tuple(
            (
                int(candidate.box[0] * scale_x),
                int(candidate.box[1] * scale_y),
                int(candidate.box[2] * scale_x),
                int(candidate.box[3] * scale_y),
            )
            for candidate in isolate_card_candidates(results)
        )


def _card_content_generation_shifted(
    cls, actual: Sequence[tuple[str, Any]], expected: Sequence[tuple[str, Any]], *, maximum_mean_absolute_error: float
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


def _fixed_slot_aligned_content_proof(
    cls, source_frames: Sequence[Any], current_image: Any, box: tuple[int, int, int, int], visual_group: str
) -> dict[str, Any]:
    """Prove one current raw slot matches 2/3 frozen frames at one shift.

    A clean-reference family is only a routing hint here.  Acceptance uses
    the existing production content query and unchanged 0.75 MAE gate.
    Exactly one native-pixel alignment must reach quorum, which rejects
    flat or repetitive content that cannot identify one physical render.
    """

    if len(source_frames) != 3 or not visual_group:
        return {
            "matched": False,
            "reason": "source_generation_invalid",
            "passing_shifts": [],
        }
    shift_diagnostics: list[dict[str, Any]] = []
    passing_shifts: list[dict[str, Any]] = []
    try:
        for shift_y in (-1, 0, 1):
            for shift_x in (-1, 0, 1):
                current_query = cls._aligned_card_content_query(
                    current_image,
                    box,
                    shift_x,
                    shift_y,
                    current_side=True,
                )
                errors: list[float | None] = []
                support = 0
                for source_image in source_frames:
                    source_query = cls._aligned_card_content_query(
                        source_image,
                        box,
                        shift_x,
                        shift_y,
                        current_side=False,
                    )
                    delta = cls._card_content_generation_deltas(
                        ((visual_group, current_query),),
                        ((visual_group, source_query),),
                    )[0]
                    error = delta["mean_absolute_error"]
                    errors.append(None if error is None else round(float(error), 6))
                    if not bool(delta["shifted"]):
                        support += 1
                diagnostic = {
                    "shift": [shift_x, shift_y],
                    "source_support": support,
                    "source_errors": errors,
                }
                shift_diagnostics.append(diagnostic)
                if support >= 2:
                    passing_shifts.append(diagnostic)
    except ArenaReaderError as error:
        return {
            "matched": False,
            "reason": error.code,
            "passing_shifts": [],
        }
    matched = len(passing_shifts) == 1
    return {
        "matched": matched,
        "reason": ("unique_alignment" if matched else ("no_alignment" if not passing_shifts else "alignment_not_unique")),
        "shift": passing_shifts[0]["shift"] if matched else None,
        "source_support": (passing_shifts[0]["source_support"] if matched else 0),
        "source_errors": (passing_shifts[0]["source_errors"] if matched else []),
        "passing_shifts": [
            {
                "shift": list(value["shift"]),
                "source_support": value["source_support"],
            }
            for value in passing_shifts
        ],
        "best_source_error": min(
            (error for value in shift_diagnostics for error in value["source_errors"] if error is not None),
            default=None,
        ),
    }


def _reference_identity_projection_is_strict(cls, identity: dict[str, Any]) -> bool:
    """Require the legacy ID/group projections to describe every exact pair."""

    candidates = cls._reference_identity_candidates(identity)
    if not candidates:
        return False
    candidate_business_ids = tuple(sorted({business_id for business_id, _visual_group in candidates}))
    candidate_visual_groups = tuple(sorted({visual_group for _business_id, visual_group in candidates}))
    return bool(
        candidate_business_ids == cls._reference_business_ids(identity) and candidate_visual_groups == cls._reference_visual_groups(identity)
    )


def _stable_reference_identity_candidates(cls, identities: Sequence[dict[str, Any]]) -> tuple[tuple[int, str], ...]:
    """Return one non-empty candidate set reproduced exactly in every frame."""

    if len(identities) != 3:
        return ()
    observed = tuple(cls._reference_identity_candidates(identity) for identity in identities)
    if not observed or any(not candidates for candidates in observed):
        return ()
    first = observed[0]
    if any(candidates != first for candidates in observed[1:]):
        return ()
    return first


def _stable_reference_visual_group(cls, identities: Sequence[dict[str, Any]]) -> str | None:
    """Return one visual family present without ambiguity in every frame."""

    observed = tuple(cls._reference_visual_groups(value) for value in identities)
    if not observed or any(len(groups) != 1 for groups in observed):
        return None
    stable = {groups[0] for groups in observed}
    if len(stable) != 1:
        return None
    return stable.pop()
