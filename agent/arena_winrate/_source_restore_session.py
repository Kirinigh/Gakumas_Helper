"""Bounded return-to-source proof owned by one detail closing operation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from p_item_recognition import PItemReferenceError

from .reader import ArenaReaderError
from ._reader_visual import CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR
from .badge_reference import BadgeReferenceError, stable_reference_business_candidates


class SourceRestoreSession:
    def __init__(self, backend, group_index, timeout_seconds, card_slot, expected_card_id, clock, *, signatures, signature_errors):
        self.backend = backend
        self.clock = clock
        self.signatures = signatures
        self.signature_errors = signature_errors
        self.card_slot = card_slot
        self.expected_card_id = expected_card_id
        self.group_index = group_index
        self.timeout_seconds = timeout_seconds

    def run(self):
        """Prove the original card generation is restored after a detail action.

        Geometry alone is insufficient because a fading detail overlay can
        expose the source row before it is ready for the next interaction.
        When the caller identifies the clicked slot, require that slot to
        match one of the three signatures from the already accepted source
        generation. A semantically identical card can nevertheless acquire a
        different 16x16 signature after the overlay closes (for example from
        sub-pixel row-box jitter or a top-family flip inside one stable
        shortlist). In that case an already resolved detail ID may authorize
        one stricter fallback: either the same single clean-reference family or
        the ambiguous business-ID/visual-family pair set must remain exact or
        narrow to a subset retaining the detail's exact pair. That set must repeat in
        three consecutive frames, and each frame's raw clicked-slot content
        must match 2/3 frozen source frames at one unique +/-1 native-pixel
        alignment under the unchanged 0.75 MAE gate.
        If the fixed gallery maps that family to another business ID, the
        fallback remains available only to the active transaction whose exact
        title and complete semantic detail already resolved the expected ID,
        while the frozen page guard and six-slot geometry also match. The
        conflict is then reported explicitly instead of letting the weaker
        gallery hint override the clicked detail. Pixel thresholds and all
        failure bounds remain unchanged.
        """

        if self.card_slot is not None and not 1 <= self.card_slot <= 6:
            raise ArenaReaderError(
                "skill_card_source_slot_invalid",
                f"source restoration slot is outside 1..6: {self.card_slot}",
            )

        self.restoration_signatures = None

        self.source_guard_frames: tuple[Any, ...] | None = None

        self.source_stable_business_ids: tuple[int, ...] = ()

        self.source_stable_visual_group: str | None = None

        self.source_stable_identity_candidates: tuple[tuple[int, str], ...] = ()

        source_identity_projection_is_strict = False

        if self.card_slot is not None:
            self.restoration_signatures = getattr(
                self.backend,
                "_card_restoration_signatures",
                {},
            ).get(self.group_index)
            if (
                self.restoration_signatures is None
                or len(self.restoration_signatures) != 3
                or any(len(frame) != 6 for frame in self.restoration_signatures)
            ):
                raise ArenaReaderError(
                    "skill_card_source_generation_missing",
                    f"group {self.group_index} has no accepted three-frame restoration generation",
                )
            guard_store = getattr(self.backend, "_card_source_guard_frames", None)
            if guard_store is not None:
                self.source_guard_frames = tuple(guard_store.get(self.group_index, ()))
                if len(self.source_guard_frames) != 3:
                    raise ArenaReaderError(
                        "skill_card_source_guard_missing",
                        f"group {self.group_index} has no accepted three-frame overlay guard",
                    )
            identity_frames = getattr(
                self.backend,
                "_card_identity_frames",
                {},
            ).get(self.group_index, ())
            if len(identity_frames) == 3 and all(len(frame) == 6 for frame in identity_frames):
                source_identities = tuple(frame[self.card_slot - 1] for frame in identity_frames)
                try:
                    self.source_stable_business_ids = stable_reference_business_candidates(source_identities)
                except BadgeReferenceError:
                    self.source_stable_business_ids = ()
                self.source_stable_visual_group = self.backend._stable_reference_visual_group(source_identities)
                self.source_stable_identity_candidates = self.backend._stable_reference_identity_candidates(source_identities)
                source_identity_projection_is_strict = bool(
                    self.source_stable_identity_candidates
                    and all(self.backend._reference_identity_projection_is_strict(identity) for identity in source_identities)
                )

        if self.expected_card_id is not None and (
            isinstance(self.expected_card_id, bool) or not isinstance(self.expected_card_id, int) or self.expected_card_id < 1
        ):
            raise ArenaReaderError(
                "skill_card_source_identity_invalid",
                f"expected source card ID is invalid: {self.expected_card_id!r}",
            )

        self.accepted_row = tuple(getattr(self.backend, "_card_rows", {}).get(self.group_index, ()))

        expected_fixed_visual_groups = (
            self.backend._fixed_gallery_visual_groups_for_business_id(self.expected_card_id) if self.expected_card_id is not None else ()
        )

        self.expected_fixed_visual_group = (
            expected_fixed_visual_groups[0] if expected_fixed_visual_groups is not None and len(expected_fixed_visual_groups) == 1 else None
        )

        source_visual_group_allowed = bool(
            source_identity_projection_is_strict
            and self.source_stable_visual_group is not None
            and self.backend._source_visual_group_matches_expected_card(
                self.source_stable_visual_group,
                self.expected_card_id,
                self.source_stable_business_ids,
            )
        )

        detail_identity_proof_matches = bool(
            self.card_slot is not None
            and len(self.accepted_row) == 6
            and self.backend._detail_identity_proof_matches(
                (self.group_index, self.card_slot),
                self.expected_card_id,
                self.accepted_row[self.card_slot - 1],
            )
        )

        single_family_detail_title_override_candidate = bool(
            self.card_slot is not None
            and len(self.accepted_row) == 6
            and self.source_guard_frames is not None
            and source_identity_projection_is_strict
            and self.source_stable_visual_group is not None
            and self.expected_fixed_visual_group is not None
            and self.source_stable_visual_group != self.expected_fixed_visual_group
            and not source_visual_group_allowed
            and detail_identity_proof_matches
        )

        source_stable_identity_business_ids = tuple(
            sorted({business_id for business_id, _visual_group in self.source_stable_identity_candidates})
        )

        source_stable_identity_visual_groups = tuple(
            sorted({visual_group for _business_id, visual_group in self.source_stable_identity_candidates})
        )

        expected_source_identity_present = bool(
            self.expected_card_id is not None
            and self.expected_fixed_visual_group is not None
            and (self.expected_card_id, self.expected_fixed_visual_group) in self.source_stable_identity_candidates
        )

        self.multi_family_detail_title_override_candidate = bool(
            self.card_slot is not None
            and len(self.accepted_row) == 6
            and self.source_guard_frames is not None
            and source_identity_projection_is_strict
            and self.source_stable_visual_group is None
            and len(source_stable_identity_business_ids) > 1
            and len(source_stable_identity_visual_groups) > 1
            and source_stable_identity_business_ids == self.source_stable_business_ids
            and expected_source_identity_present
            and detail_identity_proof_matches
        )

        self.detail_title_override_candidate = bool(
            single_family_detail_title_override_candidate or self.multi_family_detail_title_override_candidate
        )

        self.source_visual_group_authorized = bool(source_visual_group_allowed or single_family_detail_title_override_candidate)

        deadline = self.clock.monotonic() + self.timeout_seconds

        self.started = self.clock.perf_counter()

        self.last_error = ""

        self.reads = 0

        self.previous_capture_started_at: float | None = None

        self.semantic_stable_frames = 0

        self.semantic_identity_frames: list[dict[str, Any]] = []

        self.aligned_content_frames: list[dict[str, Any]] = []

        self.source_guard_consecutive = 0

        while self.clock.monotonic() < deadline:
            if self._observe_frame():
                return
            # One wait per rejected observation. Inner evidence stages only
            # report readiness so the normal first guard frame cannot wait twice.
            self.backend._sleep(self.backend._source_restore_poll_seconds)
        self._raise_failure()

    def _observe_frame(self):
        try:
            capture_started_at = self.clock.monotonic()
            if self.previous_capture_started_at is not None:
                self.backend._record_duration_sample(
                    "skill_card_source_restore_capture_start_span",
                    capture_started_at - self.previous_capture_started_at,
                )
            self.previous_capture_started_at = capture_started_at
            self.image = self.backend._capture()
            detected_row = self.backend._validated_card_group_row(self.image, self.group_index)
            self.row = detected_row
            if len(self.accepted_row) == 6:
                if self.backend._card_rows_shifted(detected_row, self.accepted_row):
                    raise ArenaReaderError(
                        "skill_card_source_geometry_changed",
                        f"group {self.group_index} returned with geometry outside the "
                        f"accepted source row; accepted={self.accepted_row!r}; "
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
                self.row = self.accepted_row
                if detected_row != self.accepted_row:
                    self.backend._increment("skill_card_source_restore_frozen_geometry")
            self.reads += 1
            self.backend._increment("skill_card_source_restore_reads")
            if not self._confirm_page_guard():
                return False
            if not self._confirm_slot_identity():
                return False
            self.backend._card_rows[self.group_index] = self.row
            self.backend._card_images[self.group_index] = self.image
            other_group = 1 - self.group_index
            if other_group in self.backend._card_rows:
                try:
                    other_row = self.backend._validated_card_group_row(self.image, other_group)
                except ArenaReaderError:
                    pass
                else:
                    self.backend._card_rows[other_group] = other_row
                    self.backend._card_images[other_group] = self.image
            self.restore_elapsed = self.clock.perf_counter() - self.started
            self.backend._add_timing(
                "skill_card_source_restore_wait",
                self.restore_elapsed,
            )
            self.backend._record_duration_sample(
                "skill_card_source_restore_phase",
                self.restore_elapsed,
            )
            if self.card_slot is not None and self.reads == 1:
                self.backend._increment("skill_card_source_restore_first_read")
            return True
        except ArenaReaderError as error:
            self.reset_consecutive_evidence()
            self.last_error = str(error)
        return False

    def _raise_failure(self):
        self.restore_elapsed = self.clock.perf_counter() - self.started
        self.backend._add_timing(
            "skill_card_source_restore_wait",
            self.restore_elapsed,
        )
        self.backend._record_duration_sample(
            "skill_card_source_restore_phase",
            self.restore_elapsed,
        )
        raise ArenaReaderError(
            "skill_card_close_failed",
            f"group {self.group_index} did not return after closing the card detail: {self.last_error}",
        )

    def matches_source_candidates(self, candidates: tuple[tuple[int, str], ...]) -> bool:
        # A disappearing alternative is not a changed card. This only
        # routes to raw-content proof; it never proves restoration itself.
        return bool(
            len(candidates) > 1
            and (self.expected_card_id, self.expected_fixed_visual_group) in candidates
            and set(candidates).issubset(self.source_stable_identity_candidates)
        )

    def reset_consecutive_evidence(self) -> None:
        self.semantic_stable_frames = 0
        self.semantic_identity_frames.clear()
        self.aligned_content_frames.clear()
        self.source_guard_consecutive = 0

    def _confirm_page_guard(self):
        if self.source_guard_frames is not None:
            guard_started = self.clock.perf_counter()
            try:
                guard_boxes = self.backend._skill_card_detail_overlay_guard_boxes(self.image)
                signature_cache = getattr(
                    self.backend,
                    "_card_source_guard_signature_cache",
                    None,
                )
                if signature_cache is None:
                    signature_cache = {}
                    self.backend._card_source_guard_signature_cache = signature_cache
                signature_cache_key = (
                    tuple(id(frame) for frame in self.source_guard_frames),
                    guard_boxes,
                )
                source_guard_signatures = signature_cache.get(signature_cache_key)
                if source_guard_signatures is None:
                    source_guard_signatures = tuple(
                        self.signatures(
                            source_image,
                            guard_boxes,
                        )
                        for source_image in self.source_guard_frames
                    )
                    signature_cache[signature_cache_key] = source_guard_signatures
                    self.backend._increment("skill_card_source_guard_signature_cache_builds")
                else:
                    self.backend._increment("skill_card_source_guard_signature_cache_hits")
                current_guard_signatures = self.signatures(
                    self.image,
                    guard_boxes,
                )
                guard_error_sets = tuple(
                    self.signature_errors(
                        source_signatures,
                        current_guard_signatures,
                    )
                    for source_signatures in source_guard_signatures
                )
            except (PItemReferenceError, ArenaReaderError) as error:
                raise ArenaReaderError(
                    "skill_card_source_guard_invalid",
                    "source/detail overlay guard could not be measured",
                ) from error
            finally:
                self.backend._add_timing(
                    "skill_card_source_restore_overlay_guard",
                    self.clock.perf_counter() - guard_started,
                )
            guard_matches_source = any(errors and max(errors) <= CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR for errors in guard_error_sets)
            if not guard_matches_source:
                self.reset_consecutive_evidence()
                self.last_error = (
                    "detail-overlay guard still differs from the frozen "
                    "source page; errors="
                    f"{tuple(tuple(round(value, 6) for value in errors) for errors in guard_error_sets)!r}"
                )
                self.backend._increment("skill_card_source_restore_overlay_guard_mismatches")
                return False
            self.source_guard_consecutive += 1
            if self.source_guard_consecutive < 2:
                return False
        return True

    def _confirm_slot_identity(self):
        if self.restoration_signatures is not None:
            signature_started = self.clock.perf_counter()
            try:
                visual_group, query, identity = self.backend._card_references().content_signature_with_identity(
                    self.image,
                    self.row[self.card_slot - 1],
                    **self.backend._skill_card_scope_kwargs(self.card_slot - 1),
                )
            except BadgeReferenceError as error:
                self.reset_consecutive_evidence()
                self.last_error = f"source card signature was not measurable after detail close: {error}"
                self.backend._increment("skill_card_source_restore_signature_failures")
                return False
            finally:
                self.backend._add_timing(
                    "skill_card_source_restore_signature",
                    self.clock.perf_counter() - signature_started,
                )
            current = ((visual_group, query),)
            source_matches = any(
                not self.backend._card_content_generation_shifted(
                    current,
                    (frame[self.card_slot - 1],),
                )
                for frame in self.restoration_signatures
            )
            if not source_matches:
                source_generation_deltas = tuple(
                    self.backend._card_content_generation_deltas(
                        current,
                        (frame[self.card_slot - 1],),
                    )[0]
                    for frame in self.restoration_signatures
                )
                best_source_delta = min(
                    source_generation_deltas,
                    key=lambda delta: float("inf") if delta["mean_absolute_error"] is None else float(delta["mean_absolute_error"]),
                )
            else:
                best_source_delta = None
            if not source_matches:
                measured_business_ids = self.backend._reference_business_ids(identity)
                measured_visual_groups = self.backend._reference_visual_groups(identity)
                measured_identity_candidates = self.backend._reference_identity_candidates(identity)
                measured_identity_projection_is_strict = self.backend._reference_identity_projection_is_strict(identity)
                expected_identity_matches = bool(
                    measured_identity_projection_is_strict
                    and self.expected_card_id is not None
                    and self.expected_card_id in measured_business_ids
                    and self.expected_fixed_visual_group is not None
                    and measured_visual_groups == (self.expected_fixed_visual_group,)
                )
                source_visual_group_matches = bool(
                    measured_identity_projection_is_strict
                    and self.source_visual_group_authorized
                    and measured_visual_groups == (self.source_stable_visual_group,)
                )
                multi_family_identity_matches = bool(
                    measured_identity_projection_is_strict
                    and self.multi_family_detail_title_override_candidate
                    and self.matches_source_candidates(measured_identity_candidates)
                )
                aligned_content_proof: dict[str, Any] | None = None
                if (
                    (source_visual_group_matches or multi_family_identity_matches)
                    and self.detail_title_override_candidate
                    and self.source_guard_frames is not None
                ):
                    aligned_started = self.clock.perf_counter()
                    try:
                        aligned_content_proof = self.backend._fixed_slot_aligned_content_proof(
                            self.source_guard_frames,
                            self.image,
                            self.accepted_row[self.card_slot - 1],
                            (self.source_stable_visual_group or "detail-title-multi-family"),
                        )
                    finally:
                        self.backend._add_timing(
                            "skill_card_source_restore_aligned_content",
                            self.clock.perf_counter() - aligned_started,
                        )
                    self.backend._increment("skill_card_source_restore_aligned_content_checks")
                    if aligned_content_proof["matched"]:
                        self.backend._increment("skill_card_source_restore_aligned_content_matches")
                    else:
                        self.backend._increment("skill_card_source_restore_aligned_content_mismatches")
                        source_visual_group_matches = False
                        multi_family_identity_matches = False
                semantic_identity_matches = bool(expected_identity_matches or source_visual_group_matches or multi_family_identity_matches)
                if semantic_identity_matches:
                    if (
                        multi_family_identity_matches
                        and self.semantic_identity_frames
                        and self.backend._reference_identity_candidates(self.semantic_identity_frames[-1]) != measured_identity_candidates
                    ):
                        # Do not combine differing subsets into a vote.
                        self.semantic_stable_frames = 0
                        self.semantic_identity_frames.clear()
                        self.aligned_content_frames.clear()
                    self.semantic_stable_frames += 1
                    self.semantic_identity_frames.append(identity)
                    del self.semantic_identity_frames[:-3]
                    if aligned_content_proof is not None:
                        self.aligned_content_frames.append(aligned_content_proof)
                        del self.aligned_content_frames[:-3]
                else:
                    self.reset_consecutive_evidence()
                stable_business_ids: tuple[int, ...] = ()
                if len(self.semantic_identity_frames) == 3:
                    try:
                        stable_business_ids = stable_reference_business_candidates(tuple(self.semantic_identity_frames))
                    except BadgeReferenceError:
                        stable_business_ids = ()
                stable_visual_group = (
                    self.backend._stable_reference_visual_group(self.semantic_identity_frames)
                    if len(self.semantic_identity_frames) == 3
                    else None
                )
                stable_identity_candidates = (
                    self.backend._stable_reference_identity_candidates(self.semantic_identity_frames)
                    if len(self.semantic_identity_frames) == 3
                    else ()
                )
                source_visual_group_settled = bool(
                    self.source_visual_group_authorized
                    and self.source_stable_visual_group is not None
                    and stable_visual_group == self.source_stable_visual_group
                )
                multi_family_identity_settled = bool(
                    self.multi_family_detail_title_override_candidate
                    and self.matches_source_candidates(stable_identity_candidates)
                    and self.semantic_stable_frames >= 3
                    and len(self.aligned_content_frames) == 3
                    and all(bool(proof.get("matched")) for proof in self.aligned_content_frames)
                )
                semantic_source_settled = bool(
                    (
                        self.expected_card_id is not None
                        and self.expected_card_id in stable_business_ids
                        and self.expected_fixed_visual_group is not None
                        and stable_visual_group == self.expected_fixed_visual_group
                    )
                    or source_visual_group_settled
                    or multi_family_identity_settled
                )
                if semantic_source_settled:
                    if multi_family_identity_settled and stable_identity_candidates != self.source_stable_identity_candidates:
                        self.backend._increment("skill_card_source_restore_candidate_contractions")
                    self.backend._increment("skill_card_source_restore_semantic_settled_fallbacks")
                    if source_visual_group_settled:
                        self.backend._increment("skill_card_source_restore_source_family_fallbacks")
                    if self.detail_title_override_candidate and (source_visual_group_settled or multi_family_identity_settled):
                        disambiguation = {
                            "reason_code": (
                                "exact_detail_title_overrode_ambiguous_card_face"
                                if multi_family_identity_settled
                                else "exact_detail_title_overrode_card_face"
                            ),
                            "status": "resolved",
                            "severity": "info",
                            "group_index": self.group_index,
                            "card_slot": self.card_slot,
                            "expected_card_id": self.expected_card_id,
                            "expected_fixed_visual_group": (self.expected_fixed_visual_group),
                            "source_stable_card_ids": list(self.source_stable_business_ids),
                            "source_stable_visual_group": (self.source_stable_visual_group),
                            "source_stable_identity_candidates": [
                                [business_id, visual_group] for business_id, visual_group in self.source_stable_identity_candidates
                            ],
                            "measured_card_ids": list(measured_business_ids),
                            "measured_visual_groups": list(measured_visual_groups),
                            "measured_identity_candidates": [
                                [business_id, visual_group] for business_id, visual_group in measured_identity_candidates
                            ],
                            "semantic_consecutive_frames": (self.semantic_stable_frames),
                            "aligned_content_frames": deepcopy(self.aligned_content_frames),
                            "best_source_delta": dict(best_source_delta),
                            "detail_identity_proof": (
                                self.backend._detail_identity_proof_diagnostic(
                                    self.backend._detail_identity_proofs[(self.group_index, self.card_slot)]
                                )
                            ),
                        }
                        runtime_disambiguations = getattr(
                            self.backend,
                            "_runtime_detail_title_disambiguations",
                            None,
                        )
                        if runtime_disambiguations is None:
                            runtime_disambiguations = []
                            self.backend._runtime_detail_title_disambiguations = runtime_disambiguations
                        runtime_disambiguations.append(disambiguation)
                        member_disambiguations = getattr(
                            self.backend,
                            "_detail_title_disambiguations",
                            None,
                        )
                        if member_disambiguations is None:
                            member_disambiguations = []
                            self.backend._detail_title_disambiguations = member_disambiguations
                        member_disambiguations.append(deepcopy(disambiguation))
                        self.backend._increment("skill_card_source_restore_detail_title_disambiguations")
                else:
                    self.last_error = (
                        f"group {self.group_index}/slot {self.card_slot} is visible but "
                        "does not match the accepted source generation; "
                        f"expected_card_id={self.expected_card_id!r}; "
                        "source_stable_card_ids="
                        f"{self.source_stable_business_ids!r}; "
                        "source_stable_visual_group="
                        f"{self.source_stable_visual_group!r}; "
                        "source_stable_identity_candidates="
                        f"{self.source_stable_identity_candidates!r}; "
                        f"measured_card_ids={measured_business_ids!r}; "
                        "measured_visual_groups="
                        f"{measured_visual_groups!r}; "
                        "measured_identity_candidates="
                        f"{measured_identity_candidates!r}; "
                        f"semantic_consecutive_frames={self.semantic_stable_frames}; "
                        f"stable_card_ids={stable_business_ids!r}; "
                        f"stable_visual_group={stable_visual_group!r}; "
                        f"identity_status={identity.get('status')!r}; "
                        "aligned_content_proof="
                        f"{aligned_content_proof!r}; "
                        f"best_source_delta={best_source_delta!r}"
                    )
                    self.backend._increment("skill_card_source_restore_generation_mismatches")
                    if len(self.semantic_identity_frames) == 3:
                        self.reset_consecutive_evidence()
                    return False
        return True
