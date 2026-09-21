"""One skill-detail resolution session; evidence and budgets never cross openings."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any
from collections import deque

from .reader import ArenaReaderError, ClickedSkillCard


class _FrameStep(Enum):
    NEXT = auto()
    REPEAT = auto()
    STOP = auto()


from .catalog import ArenaCatalogError
from ._reader_evidence import _TitleBoundEffectRoiText, _DetailIdentityObservation, _effect_roi_observation_count


class SkillDetailSession:
    def __init__(
        self,
        backend,
        key,
        candidate_ids,
        *,
        expected_customization_count,
        timeout_seconds,
        allow_zero_without_badge_count,
        target,
        stage_number,
        member_slot,
        clock,
    ):
        self.backend = backend
        self.clock = clock
        self.allow_zero_without_badge_count = allow_zero_without_badge_count
        self.candidate_ids = candidate_ids
        self.expected_customization_count = expected_customization_count
        self.key = key
        self.member_slot = member_slot
        self.stage_number = stage_number
        self.target = target
        self.timeout_seconds = timeout_seconds

    def run(self) -> ClickedSkillCard:
        """Resolve the accepted overlay text, rereading only while effects settle."""

        diagnostic = getattr(self.backend, "_detail_failure_frames", None)

        if diagnostic is not None:
            diagnostic["position"]["phase"] = "body"

        self.resolution_started = self.clock.perf_counter()

        self.detail_identity_observations: list[_DetailIdentityObservation] = []

        self.opened_image = self.backend._card_detail_images.get(self.key)

        self.opened_capture_time = getattr(self.backend, "_card_detail_capture_started_at", {}).get(self.key)

        self.transaction_token = getattr(self.backend, "_card_transaction_tokens", {}).get(self.key)

        self.source_guard_frames = getattr(self.backend, "_card_source_guard_frames", {}).get(self.key[0])

        self.recent_detail_frames: deque[tuple[float, Any]] = deque(maxlen=2)

        self.deadline = self.clock.monotonic() + self.timeout_seconds

        self.last_contact_released_at = getattr(
            self.backend,
            "_card_detail_last_contact_released_at",
            {},
        ).get(self.key)

        self.frame_capture_started_at = getattr(
            self.backend,
            "_card_detail_capture_started_at",
            {},
        ).get(self.key)

        self.frame_is_settled = (
            self.last_contact_released_at is not None
            and self.frame_capture_started_at is not None
            and self.frame_capture_started_at >= self.last_contact_released_at + 0.60
        )

        self.text = self.backend._card_detail_texts.get(self.key, "")

        self.last_error = "accepted overlay text was not cached"

        self.confirmation_candidate: ClickedSkillCard | None = None

        self.confirmation_reason: str | None = None

        self.confirmation_started = False

        self.resolution_reads = 0

        self.resolution_conflicts = 0

        self.effect_roi_failures = 0

        self.zero_candidate: ClickedSkillCard | None = None

        self.zero_confirmation_reads = 0

        self.zero_settle_wait_recorded = False

        self.zero_confirmation_extension_applied = False

        self.positive_candidate: ClickedSkillCard | None = None

        self.positive_confirmation_reads = 0

        self.positive_confirmation_extension_applied = False

        self.positive_confirmation_started_at: float | None = None

        self.cost_fallback_candidate: ClickedSkillCard | None = None

        self.cost_fallback_confirmation_reads = 0

        self.cost_fallback_confirmation_extension_applied = False

        self.terminal_error: ArenaReaderError | None = None

        self.pending_title_error: ArenaReaderError | None = None

        self.numeric_recovery_started = False

        self.body_recovery_active = False

        self.same_frame_body_recovery_attempted = False

        self.frame_confirmation_reads = (0, 0, 0, 0)

        self.frame_confirmation_counter_names = (
            "skill_card_detail_zero_confirmation_reads",
            "skill_card_detail_zero_confirmations",
            "skill_card_detail_positive_confirmation_reads",
            "skill_card_detail_positive_confirmations",
            "skill_card_cost_fallback_confirmation_reads",
            "skill_card_cost_fallback_confirmations",
        )

        self.frame_confirmation_counts = {name: self.backend._runtime_counts.get(name) for name in self.frame_confirmation_counter_names}

        self.same_frame_effect_roi_attempted = False

        self.same_frame_enhanced_effect_roi_attempted = False

        self.generic_cost_fallback_policy = (
            None if self.target is None else "assume_unenhanced" if self.target.is_own_team else "assume_enhanced"
        )

        self.same_frame_effect_roi_context: tuple[str, str, int] | None = None

        while True:
            if self.text:
                step = self._resolve_frame()
                if step is _FrameStep.REPEAT:
                    continue
                if step is _FrameStep.STOP:
                    break
                if step is not _FrameStep.NEXT:
                    return step
            if self.terminal_error is not None:
                break
            if self.clock.monotonic() >= self.deadline:
                break
            self._read_next_frame()
        return self._raise_failure()

    def _resolve_frame(self):
        self.auxiliary_badge_glyph_allowed = self.frame_is_settled
        try:
            step = self._resolve_current_text()
            if step is not _FrameStep.NEXT:
                return step
            self.conservative_cost_assumption = getattr(
                self.resolved,
                "conservative_cost_assumption",
                None,
            )
            step = self._read_cost_effect_evidence()
            if step is not _FrameStep.NEXT:
                return step
            self.pending_title_error = None
            self.resolved_count = sum(int(value) for value in self.resolved.customizations.values())
            self.conservative_cost_assumption = getattr(
                self.resolved,
                "conservative_cost_assumption",
                None,
            )
            if self.cost_fallback_candidate is not None and self.conservative_cost_assumption is None:
                raise ArenaReaderError(
                    "skill_card_cost_fallback_frame_conflict",
                    "a settled fallback frame was followed by a different non-fallback resolution",
                )
            step = self._confirm_cost_assumption()
            if step is not _FrameStep.NEXT:
                return step
            step = self._confirm_zero_count()
            if step is not _FrameStep.NEXT:
                return step
            self.evidence_customization_count = self.expected_customization_count
            if self.evidence_customization_count is None and self.resolved.resolution_source.endswith("_auxiliary_badge_glyph"):
                self.auxiliary = self.backend._badge_glyph_count_diagnostics.get(self.key, {})
                self.auxiliary_count = self.auxiliary.get("resolved_count")
                if type(self.auxiliary_count) is not int:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_evidence_missing",
                        "an auxiliary badge-glyph resolution has no cached count",
                    )
                self.evidence_customization_count = self.auxiliary_count
            self.count_mismatch = self.expected_customization_count is not None and self.resolved_count != self.expected_customization_count
            cost_fallback_confirmed = (
                getattr(
                    self.resolved,
                    "conservative_cost_assumption",
                    None,
                )
                is not None
                and self.resolved.resolution_source == "generic_cost_conservative_fallback_confirmed"
            )
            if (
                self.count_mismatch
                and not cost_fallback_confirmed
                and (
                    self.resolved.resolution_source
                    not in {
                        "detail_unique",
                        "detail_card_face_unique",
                    }
                    or self.resolved_count < 1
                )
            ):
                raise ArenaReaderError(
                    "skill_card_detail_count_conflict",
                    "a badge-count mismatch was not resolved by one unique positive detail",
                )

            try:
                self.generic_cost_frame_values = self.backend._certify_card_face_generic_cost(
                    self.key,
                    self.resolved.card_id,
                    self.resolved.customizations,
                )
            except ArenaReaderError as error:
                if error.code != "skill_card_cost_evidence_inconclusive":
                    raise
                # Detail semantics and the bounded auxiliary badge count
                # may already prove a unique group.  An inconclusive
                # optional cost carrier must not preempt those independent
                # paths; any later branch that truly needs cost evidence
                # still fails inside the external-evidence certifier.
                self.backend._increment("skill_card_face_cost_optional_inconclusive")
                self.generic_cost_frame_values = None
                self.generic_cost_evidence_error = str(error)
            else:
                self.generic_cost_evidence_error = None

            step = self._confirm_negative_evidence()
            if step is not _FrameStep.NEXT:
                return step
            step = self._confirm_positive_evidence()
            if step is not _FrameStep.NEXT:
                return step
            if self.count_mismatch and cost_fallback_confirmed:
                assert self.expected_customization_count is not None
                self.backend._record_detail_count_override(
                    self.key,
                    self.expected_customization_count,
                    self.resolved,
                )
            reason = "badge_count_override" if self.count_mismatch and not cost_fallback_confirmed else None
            if reason is None:
                self.backend._record_cost_customization_fallback(
                    self.key,
                    self.resolved,
                    target=self.target,
                    stage_number=self.stage_number,
                    member_slot=self.member_slot,
                )
                self.backend._maybe_register_badge_glyph_exemplar(
                    self.key,
                    self.resolved,
                    target=self.target,
                    stage_number=self.stage_number,
                    member_slot=self.member_slot,
                )
                return self.finish_resolution(self.resolved)

            self.resolution_reads += 1
            if not self.confirmation_started:
                self.confirmation_started = True
                self.backend._increment("skill_card_detail_semantic_confirmations")
            if self.backend._same_clicked_card_resolution(
                self.confirmation_candidate,
                self.resolved,
            ):
                confirmed = ClickedSkillCard(
                    self.resolved.card_id,
                    dict(self.resolved.customizations),
                    resolution_source=f"{self.resolved.resolution_source}_confirmed",
                    detail_confirmation_reads=self.resolution_reads,
                    detail_evidence_mode=self.resolved.detail_evidence_mode,
                    conservative_cost_assumption=getattr(
                        self.resolved,
                        "conservative_cost_assumption",
                        None,
                    ),
                )
                self.backend._record_detail_semantic_confirmation(
                    self.key,
                    confirmed,
                    reason=reason,
                    resolution_conflicts=self.resolution_conflicts,
                )
                if self.count_mismatch:
                    assert self.expected_customization_count is not None
                    self.backend._record_detail_count_override(
                        self.key,
                        self.expected_customization_count,
                        confirmed,
                    )
                self.backend._record_cost_customization_fallback(
                    self.key,
                    confirmed,
                    target=self.target,
                    stage_number=self.stage_number,
                    member_slot=self.member_slot,
                )
                self.backend._maybe_register_badge_glyph_exemplar(
                    self.key,
                    confirmed,
                    target=self.target,
                    stage_number=self.stage_number,
                    member_slot=self.member_slot,
                )
                return self.finish_resolution(confirmed)
            if self.confirmation_candidate is not None:
                self.resolution_conflicts += 1
                self.backend._increment("skill_card_detail_semantic_conflicts")
            self.confirmation_candidate = self.resolved
            self.confirmation_reason = reason
            self.last_error = self.backend._detail_confirmation_wait_message(
                reason,
                expected_customization_count=self.expected_customization_count,
                resolved_count=self.resolved_count,
            )
        except ArenaReaderError as error:
            self.confirmation_candidate = None
            self.confirmation_reason = None
            self.last_error = str(error)
            self.pending_title_error = (
                error if error.code in {"skill_card_detail_upgrade_mark_missing", "skill_card_detail_title_unresolved"} else None
            )
            if self.pending_title_error is not None:
                # Do not bridge confirmation votes across an uncertain
                # identity. Retry only within the existing deadline;
                # no ROI/body repair or extra budget can turn it valid.
                self.zero_candidate = self.positive_candidate = self.cost_fallback_candidate = None
                self.detail_identity_observations.clear()
            if (
                self.frame_is_settled
                and not self.same_frame_body_recovery_attempted
                and error.code
                in {
                    "skill_card_detail_ambiguous",
                    "skill_card_badge_glyph_domain_invalid",
                    "skill_card_badge_glyph_ambiguous",
                    "skill_card_badge_glyph_exemplar_transition_conflict",
                }
            ):
                self.same_frame_body_recovery_attempted = True
                self.repaired_text = self.backend._recover_failed_skill_card_body_text(self.key)
                if self.repaired_text is not None:
                    # Re-run this frame through the ordinary parser and
                    # validators. A prior vote from its unrepaired text
                    # cannot act as an independent confirmation frame.
                    self.zero_candidate = self.positive_candidate = self.cost_fallback_candidate = None
                    (
                        self.resolution_reads,
                        self.zero_confirmation_reads,
                        self.positive_confirmation_reads,
                        self.cost_fallback_confirmation_reads,
                    ) = self.frame_confirmation_reads
                    for name, previous_count in self.frame_confirmation_counts.items():
                        if previous_count is None:
                            self.backend._runtime_counts.pop(name, None)
                        else:
                            self.backend._runtime_counts[name] = previous_count
                    self.body_recovery_active = True
                    self.text = self.repaired_text
                    self.backend._card_detail_texts[self.key] = self.text
                    if self.same_frame_effect_roi_context is not None:
                        self.same_frame_effect_roi_context = (
                            self.text,
                            self.same_frame_effect_roi_context[1],
                            self.same_frame_effect_roi_context[2],
                        )
                    return _FrameStep.REPEAT
            image_evidence = self.backend._cached_full_frame_ocr_evidence(self.backend._card_detail_images.get(self.key))
            if image_evidence is not None and self.backend._retry_transient_communication_items(image_evidence.filtered_items):
                self.text = ""
                self.zero_candidate = self.positive_candidate = self.cost_fallback_candidate = None
                return _FrameStep.REPEAT
            if error.code in {
                "skill_card_detail_id_changed",
                "skill_card_cost_fallback_changed",
                "skill_card_cost_fallback_contract_invalid",
                "skill_card_cost_fallback_frame_conflict",
                "skill_card_cost_fallback_location_missing",
                "skill_card_cost_fallback_roi_conflict",
                "skill_card_detail_effect_view_conflict",
            }:
                self.terminal_error = error
                return _FrameStep.STOP
            if error.code == "skill_card_detail_positive_evidence_missing":
                if self.effect_roi_failures < 1:
                    # The title can become readable before the effect rows
                    # finish animating. Retry the complete evidence pair on
                    # one fresh frame; never join full/ROI text across frames.
                    self.effect_roi_failures += 1
                    self.backend._increment("skill_card_detail_effect_roi_retries")
                else:
                    self.terminal_error = error
                    return _FrameStep.STOP
        return _FrameStep.NEXT

    def _read_next_frame(self):
        self.backend._sleep(0.08)
        self.backend._increment("skill_card_detail_ocr_rereads")
        try:
            previous_capture_started_at = self.frame_capture_started_at
            detail_capture_started_at = self.clock.monotonic()
            if previous_capture_started_at is not None:
                self.backend._record_duration_sample(
                    "skill_card_detail_capture_start_span",
                    detail_capture_started_at - previous_capture_started_at,
                )
            self.detail_image = self.backend._capture()
            self.text = self.backend._skill_card_detail_ocr_text(
                self.detail_image,
                phase="reread",
            )
        except ArenaReaderError as error:
            self.text = ""
            self.last_error = str(error)
        else:
            self.backend._card_detail_texts[self.key] = self.text
            self.backend._card_detail_images[self.key] = self.detail_image
            self.recent_detail_frames.append((detail_capture_started_at, self.detail_image))
            capture_times = getattr(
                self.backend,
                "_card_detail_capture_started_at",
                None,
            )
            if capture_times is None:
                capture_times = {}
                self.backend._card_detail_capture_started_at = capture_times
            capture_times[self.key] = detail_capture_started_at
            self.frame_capture_started_at = detail_capture_started_at
            self.frame_is_settled = (
                self.last_contact_released_at is not None and self.frame_capture_started_at >= self.last_contact_released_at + 0.60
            )
            self.same_frame_body_recovery_attempted = False
            self.frame_confirmation_reads = (
                self.resolution_reads,
                self.zero_confirmation_reads,
                self.positive_confirmation_reads,
                self.cost_fallback_confirmation_reads,
            )
            self.frame_confirmation_counts = {name: self.backend._runtime_counts.get(name) for name in self.frame_confirmation_counter_names}
            if self.body_recovery_active and self.frame_is_settled:
                self.same_frame_body_recovery_attempted = True
                self.repaired_text = self.backend._recover_failed_skill_card_body_text(self.key)
                if self.repaired_text is not None:
                    self.text = self.repaired_text
                    self.backend._card_detail_texts[self.key] = self.text
            self.same_frame_effect_roi_attempted = False
            self.same_frame_enhanced_effect_roi_attempted = False
            self.same_frame_effect_roi_context = None

    def _raise_failure(self):
        self.backend._record_duration_sample(
            "skill_card_detail_failed_resolution_phase",
            self.clock.perf_counter() - self.resolution_started,
        )
        self.backend._increment("skill_card_detail_failed_resolutions")
        if self.terminal_error is not None:
            # The detail transaction owns this terminal decision. Propagate it
            # through both direct and badge paths without renewing the budget.
            self.terminal_error.retry_whole_read = False
            raise self.terminal_error
        if self.pending_title_error is not None:
            self.pending_title_error.retry_whole_read = False
            raise self.pending_title_error
        if self.backend._skill_card_detail_disappeared(
            self.key,
            self.opened_image,
            self.opened_capture_time,
            self.transaction_token,
            self.source_guard_frames,
            tuple(self.recent_detail_frames),
        ):
            self.backend._increment("skill_card_detail_disappeared_before_confirmation")
            raise ArenaReaderError(
                "skill_card_detail_disappeared",
                f"group {self.key[0]}/slot {self.key[1]}: the accepted detail disappeared "
                "before confirmation; the last two captured frames match the "
                "frozen member page; no extra observation was taken; "
                f"original_error={self.last_error}",
            )
        raise ArenaReaderError(
            "skill_card_detail_ambiguous",
            "card detail did not become uniquely resolvable within the bounded settle window: "
            f"{self.last_error}; pending_confirmation={self.confirmation_reason!r}; "
            f"OCR={self.text[:400]!r}",
        )

    def finish_resolution(self, value: ClickedSkillCard) -> ClickedSkillCard:
        # Exact title proves only the business ID.  Reuse the already
        # accepted detail frame for every customization evidence mode;
        # never add a click, capture, or reread solely to create this proof.
        identity_observation = self.backend._detail_identity_observation(
            self.key,
            self.candidate_ids,
            value,
        )
        if identity_observation is not None:
            self.detail_identity_observations.append(identity_observation)
            del self.detail_identity_observations[:-2]
        self.backend._record_detail_identity_proof(
            self.key,
            value,
            self.detail_identity_observations,
        )
        self.backend._record_duration_sample(
            "skill_card_detail_resolution_phase",
            self.clock.perf_counter() - self.resolution_started,
        )
        return value

    def _resolve_current_text(self):
        try:
            self.resolved = self.backend._resolve_clicked_card_text(
                self.candidate_ids,
                self.text,
                expected_customization_count=self.expected_customization_count,
                allow_zero_without_badge_count=(self.allow_zero_without_badge_count),
                # Full-detail semantics and the title-bound effect
                # panel both get first refusal. The fallible card-
                # face glyph is only a final disambiguator after the
                # same frame has exhausted those two text views.
                allow_auxiliary_badge_glyph=False,
                generic_cost_fallback_policy=(self.generic_cost_fallback_policy),
                allow_generic_cost_fallback=(self.auxiliary_badge_glyph_allowed),
                key=self.key,
            )
        except ArenaReaderError as initial_error:
            if initial_error.code != "skill_card_detail_ambiguous":
                raise
            if not self.frame_is_settled:
                self.backend._increment("skill_card_detail_render_incomplete_rereads")
                raise ArenaReaderError(
                    "skill_card_detail_render_incomplete",
                    "the captured detail frame predates the semantic settle boundary and has no unique full-detail result",
                ) from initial_error
            if self.same_frame_effect_roi_attempted:
                raise
            self.same_frame_effect_roi_attempted = True
            self.detail_image = self.backend._card_detail_images.get(self.key)
            if self.detail_image is None:
                raise
            card_id = self.backend._confirm_skill_card_detail_identity(
                self.text,
                self.candidate_ids,
                expected_customization_count=(self.expected_customization_count),
                source_group_index=self.key[0],
                detail_image=self.detail_image,
                source_card_box=self.backend._skill_card_source_box(self.key),
                source_card_slot=self.key[1],
            )
            self.roi_text = self.backend._skill_card_title_anchored_effect_roi_text(
                self.detail_image,
                card_id,
            )
            self.same_frame_effect_roi_context = (
                self.text,
                self.roi_text,
                _effect_roi_observation_count(self.roi_text),
            )
            self.combined_text = self.backend._merge_skill_card_effect_detail_views(
                card_id,
                (self.text, self.roi_text),
            )
            try:
                self.resolved = self.backend._resolve_clicked_card_text(
                    self.candidate_ids,
                    self.combined_text,
                    expected_customization_count=(self.expected_customization_count),
                    allow_zero_without_badge_count=(self.allow_zero_without_badge_count),
                    allow_auxiliary_badge_glyph=False,
                    generic_cost_fallback_policy=(self.generic_cost_fallback_policy),
                    allow_generic_cost_fallback=(self.auxiliary_badge_glyph_allowed),
                    generic_cost_coverage_context=(self.same_frame_effect_roi_context),
                    key=self.key,
                )
            except ArenaReaderError as combined_error:
                if combined_error.code != "skill_card_detail_ambiguous":
                    raise
                enhanced_resolved: ClickedSkillCard | None = None
                unresolved_error = combined_error
                if not self.same_frame_enhanced_effect_roi_attempted:
                    self.same_frame_enhanced_effect_roi_attempted = True
                    try:
                        enhanced_roi_text = self.backend._skill_card_title_anchored_enhanced_effect_roi_text(
                            self.detail_image,
                            card_id,
                        )
                    except ArenaReaderError as enhancement_error:
                        if enhancement_error.code not in {
                            "ocr_empty",
                            "skill_card_detail_effect_roi_empty",
                        }:
                            raise
                        self.backend._increment("skill_card_detail_enhanced_effect_roi_empty")
                    else:
                        roi_observation_count = _effect_roi_observation_count(self.roi_text) + int(bool(enhanced_roi_text.strip()))
                        self.roi_text = _TitleBoundEffectRoiText(
                            self.backend._merge_skill_card_effect_detail_views(
                                card_id,
                                (self.roi_text, enhanced_roi_text),
                            ),
                            roi_observation_count,
                        )
                        self.combined_text = self.backend._merge_skill_card_effect_detail_views(
                            card_id,
                            (self.text, self.roi_text),
                        )
                        self.same_frame_effect_roi_context = (
                            self.text,
                            self.roi_text,
                            _effect_roi_observation_count(self.roi_text),
                        )
                        try:
                            enhanced_resolved = self.backend._resolve_clicked_card_text(
                                self.candidate_ids,
                                self.combined_text,
                                expected_customization_count=(self.expected_customization_count),
                                allow_zero_without_badge_count=(self.allow_zero_without_badge_count),
                                allow_auxiliary_badge_glyph=False,
                                generic_cost_fallback_policy=(self.generic_cost_fallback_policy),
                                allow_generic_cost_fallback=(self.auxiliary_badge_glyph_allowed),
                                generic_cost_coverage_context=(self.same_frame_effect_roi_context),
                                key=self.key,
                            )
                        except ArenaReaderError as enhancement_error:
                            if enhancement_error.code != "skill_card_detail_ambiguous":
                                raise
                            unresolved_error = enhancement_error
                        else:
                            self.backend._increment("skill_card_detail_enhanced_effect_roi_recoveries")
                if enhanced_resolved is not None:
                    self.resolved = enhanced_resolved
                else:
                    if not self.auxiliary_badge_glyph_allowed:
                        raise unresolved_error
                    try:
                        self.resolved = self.backend._resolve_clicked_card_text(
                            self.candidate_ids,
                            self.combined_text,
                            expected_customization_count=(self.expected_customization_count),
                            allow_zero_without_badge_count=(self.allow_zero_without_badge_count),
                            allow_auxiliary_badge_glyph=True,
                            generic_cost_fallback_policy=(self.generic_cost_fallback_policy),
                            allow_generic_cost_fallback=True,
                            generic_cost_coverage_context=(self.same_frame_effect_roi_context),
                            key=self.key,
                        )
                    except ArenaReaderError as exhausted_error:
                        if not self.numeric_recovery_started and exhausted_error.code in {
                            "skill_card_detail_ambiguous",
                            "skill_card_badge_glyph_domain_invalid",
                            "skill_card_badge_glyph_ambiguous",
                        }:
                            self.numeric_recovery_started = True
                            self.backend._increment("skill_card_error_numeric_transactions")
                        recovery = self.backend._recover_failed_skill_card_effect_text(
                            self.key,
                            card_id,
                            exhausted_error,
                            expected_customization_count=(self.expected_customization_count),
                            allow_zero_without_badge_count=(self.allow_zero_without_badge_count),
                        )
                        if recovery is None:
                            raise
                        self.combined_text, self.resolved = recovery
                        # Recovery admits positive signatures only.
                        # Discard the failed views instead of feeding
                        # them into later coverage/cost certificates.
                        self.same_frame_effect_roi_context = None
            self.text = self.combined_text
            self.backend._card_detail_texts[self.key] = self.combined_text
            self.backend._increment("skill_card_detail_same_frame_effect_roi_recoveries")
        return _FrameStep.NEXT

    def _confirm_negative_evidence(self):
        if self.resolved.detail_evidence_mode == "negative_dependent":
            if self.same_frame_effect_roi_context is not None:
                full_detail_text, self.roi_text, _ = self.same_frame_effect_roi_context
                effect_roi_title_bound = True
                self.combined_text = self.backend._merge_skill_card_effect_detail_views(
                    self.resolved.card_id,
                    (full_detail_text, self.roi_text),
                )
            else:
                self.detail_image = self.backend._card_detail_images.get(self.key)
                if self.detail_image is None:
                    raise ArenaReaderError(
                        "skill_card_detail_positive_evidence_missing",
                        "the accepted detail page has no cached image for an "
                        "independent effect-ROI OCR; "
                        f"key={self.key!r}; card_id={self.resolved.card_id}; "
                        f"customizations={dict(self.resolved.customizations)!r}; "
                        f"expected_badge_count={self.evidence_customization_count!r}",
                    )
                full_detail_text = self.text
                # A fixed screen column is not a valid coverage
                # boundary: live detail panels shift horizontally,
                # and a fresh frame cannot repair a cropped panel.
                # Every structural-omission pass is therefore bound
                # to the uniquely resolved title, including the
                # first pass. A bounded fresh-frame retry remains
                # only for genuinely incomplete effect animation.
                effect_roi_title_bound = True
                self.roi_text = self.backend._skill_card_title_anchored_effect_roi_text(
                    self.detail_image,
                    self.resolved.card_id,
                )
                self.combined_text = self.backend._merge_skill_card_effect_detail_views(
                    self.resolved.card_id,
                    (self.text, self.roi_text),
                )
                try:
                    self.resolved = self.backend._resolve_clicked_card_text(
                        self.candidate_ids,
                        self.combined_text,
                        expected_customization_count=(self.expected_customization_count),
                        allow_zero_without_badge_count=(self.allow_zero_without_badge_count),
                        allow_auxiliary_badge_glyph=False,
                        generic_cost_fallback_policy=(self.generic_cost_fallback_policy),
                        allow_generic_cost_fallback=(self.auxiliary_badge_glyph_allowed),
                        generic_cost_coverage_context=(
                            full_detail_text,
                            self.roi_text,
                            _effect_roi_observation_count(self.roi_text),
                        ),
                        key=self.key,
                    )
                except ArenaReaderError as combined_error:
                    if combined_error.code != "skill_card_detail_ambiguous" or not self.auxiliary_badge_glyph_allowed:
                        raise
                    self.resolved = self.backend._resolve_clicked_card_text(
                        self.candidate_ids,
                        self.combined_text,
                        expected_customization_count=(self.expected_customization_count),
                        allow_zero_without_badge_count=(self.allow_zero_without_badge_count),
                        allow_auxiliary_badge_glyph=True,
                        generic_cost_fallback_policy=(self.generic_cost_fallback_policy),
                        allow_generic_cost_fallback=True,
                        generic_cost_coverage_context=(
                            full_detail_text,
                            self.roi_text,
                            _effect_roi_observation_count(self.roi_text),
                        ),
                        key=self.key,
                    )
            self.resolved_count = sum(int(value) for value in self.resolved.customizations.values())
            self.evidence_customization_count = self.expected_customization_count
            if self.evidence_customization_count is None and self.resolved.resolution_source.endswith("_auxiliary_badge_glyph"):
                self.auxiliary = self.backend._badge_glyph_count_diagnostics.get(self.key, {})
                self.auxiliary_count = self.auxiliary.get("resolved_count")
                if type(self.auxiliary_count) is not int:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_evidence_missing",
                        "an auxiliary badge-glyph reread has no cached count",
                    )
                self.evidence_customization_count = self.auxiliary_count
            self.count_mismatch = self.expected_customization_count is not None and self.resolved_count != self.expected_customization_count
            if self.resolved.detail_evidence_mode == "negative_dependent":
                certifier = getattr(
                    self.backend.catalog,
                    "certify_structural_omission_evidence",
                    None,
                )
                certified_mode = None
                certification_error = "catalog has no structural omission certifier"
                certification_count = self.evidence_customization_count
                if certification_count is None:
                    # The coverage certifier is itself the independent
                    # evidence boundary: it accepts every selected level
                    # only when two covered detail views prove an explicit
                    # positive signature, a UI-defined typed-cost/limit
                    # omission, or a stable three-frame card-face cost.
                    # Let it certify the resolved group's total without
                    # making the fallible face-badge digit a prerequisite.
                    certification_count = self.resolved_count
                if certifier is not None and certification_count is not None:
                    try:
                        certified_mode = certifier(
                            self.resolved.card_id,
                            full_detail_text=full_detail_text,
                            effect_roi_text=self.roi_text,
                            expected_count=certification_count,
                            resolved=self.resolved.customizations,
                            generic_cost_frame_values=(self.generic_cost_frame_values),
                            effect_roi_title_bound=(effect_roi_title_bound),
                        )
                    except ArenaCatalogError as error:
                        certification_error = str(error)
                if self.generic_cost_evidence_error is not None:
                    certification_error = f"{certification_error}; card_face_cost={self.generic_cost_evidence_error}"
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
                        f"key={self.key!r}; card_id={self.resolved.card_id}; "
                        f"customizations={dict(self.resolved.customizations)!r}; "
                        f"expected_badge_count={self.evidence_customization_count!r}; "
                        f"certification={certification_error}; "
                        f"effect_roi={self.roi_text[:240]!r}",
                    )
                self.resolved = ClickedSkillCard(
                    self.resolved.card_id,
                    dict(self.resolved.customizations),
                    resolution_source=self.resolved.resolution_source,
                    detail_confirmation_reads=self.resolved.detail_confirmation_reads,
                    detail_evidence_mode=certified_mode,
                    conservative_cost_assumption=getattr(
                        self.resolved,
                        "conservative_cost_assumption",
                        None,
                    ),
                )
                self.backend._increment("skill_card_detail_external_evidence_certifications")
                if certified_mode in {
                    "structural_omission_unique",
                    "external_evidence_unique",
                }:
                    self.backend._increment("skill_card_detail_structural_omission_certifications")
            self.text = self.combined_text
            self.backend._card_detail_texts[self.key] = self.combined_text
        return _FrameStep.NEXT

    def _confirm_positive_evidence(self):
        if (
            self.allow_zero_without_badge_count
            and self.expected_customization_count is None
            and self.resolved_count > 0
            and self.resolved.detail_evidence_mode == "positive_unique"
        ):
            # This branch supplies the badge count as well as the
            # concrete customization IDs.  A single transitional OCR
            # frame must not become self-authenticating positive
            # evidence, so require the same business resolution once
            # more from a fresh frame.  Structural-omission and
            # card-face-cost modes already carry independent channels.
            # This adds no navigation or broad retry.
            same_positive_resolution = self.backend._same_clicked_card_resolution(
                self.positive_candidate,
                self.resolved,
            )
            self.positive_confirmation_reads += 1
            self.backend._increment("skill_card_detail_positive_confirmation_reads")
            if same_positive_resolution:
                self.resolved = ClickedSkillCard(
                    self.resolved.card_id,
                    dict(self.resolved.customizations),
                    resolution_source=(f"{self.resolved.resolution_source}_positive_confirmed"),
                    detail_confirmation_reads=(self.positive_confirmation_reads),
                    detail_evidence_mode=self.resolved.detail_evidence_mode,
                    conservative_cost_assumption=getattr(
                        self.resolved,
                        "conservative_cost_assumption",
                        None,
                    ),
                )
                self.backend._increment("skill_card_detail_positive_confirmations")
                if self.positive_confirmation_started_at is not None:
                    self.backend._record_duration_sample(
                        "skill_card_detail_positive_confirmation_phase",
                        self.clock.perf_counter() - self.positive_confirmation_started_at,
                    )
            else:
                if self.positive_candidate is not None:
                    self.backend._increment("skill_card_detail_positive_confirmation_conflicts")
                self.positive_candidate = self.resolved
                if self.positive_confirmation_started_at is None:
                    self.positive_confirmation_started_at = self.clock.perf_counter()
                if not self.positive_confirmation_extension_applied:
                    self.deadline = max(
                        self.deadline,
                        self.clock.monotonic() + 0.25,
                    )
                    self.positive_confirmation_extension_applied = True
                    self.backend._increment("skill_card_detail_positive_confirmation_extensions")
                raise ArenaReaderError(
                    "skill_card_detail_positive_unconfirmed",
                    "an unconstrained positive detail requires one fresh semantically identical frame",
                )
        return _FrameStep.NEXT

    def _read_cost_effect_evidence(self):
        if self.conservative_cost_assumption is not None and self.same_frame_effect_roi_context is None:
            self.detail_image = self.backend._card_detail_images.get(self.key)
            if self.detail_image is None:
                raise ArenaReaderError(
                    "skill_card_cost_fallback_detail_image_missing",
                    "generic-cost fallback has no accepted detail image",
                )
            self.roi_text = self.backend._skill_card_title_anchored_effect_roi_text(
                self.detail_image,
                self.resolved.card_id,
            )
            self.same_frame_effect_roi_context = (
                self.text,
                self.roi_text,
                _effect_roi_observation_count(self.roi_text),
            )
            self.combined_text = self.backend._merge_skill_card_effect_detail_views(
                self.resolved.card_id,
                (self.text, self.roi_text),
            )
            combined_resolved = self.backend._resolve_clicked_card_text(
                self.candidate_ids,
                self.combined_text,
                expected_customization_count=(self.expected_customization_count),
                allow_zero_without_badge_count=(self.allow_zero_without_badge_count),
                allow_auxiliary_badge_glyph=(self.auxiliary_badge_glyph_allowed),
                generic_cost_fallback_policy=(self.generic_cost_fallback_policy),
                allow_generic_cost_fallback=True,
                generic_cost_coverage_context=(self.same_frame_effect_roi_context),
                key=self.key,
            )
            if not self.backend._same_clicked_card_resolution(
                self.resolved,
                combined_resolved,
            ):
                raise ArenaReaderError(
                    "skill_card_cost_fallback_roi_conflict",
                    "full detail and title-bound effect ROI do not leave the same generic-cost pair",
                )
            self.resolved = combined_resolved
            self.text = self.combined_text
            self.backend._card_detail_texts[self.key] = self.combined_text
            self.backend._increment("skill_card_cost_fallback_title_bound_roi_confirmations")
        return _FrameStep.NEXT

    def _confirm_cost_assumption(self):
        if self.conservative_cost_assumption is not None:
            if any(
                candidate is not None
                for candidate in (
                    self.confirmation_candidate,
                    self.zero_candidate,
                    self.positive_candidate,
                )
            ):
                raise ArenaReaderError(
                    "skill_card_cost_fallback_frame_conflict",
                    "a non-fallback confirmation candidate was followed by a generic-cost fallback resolution",
                )
            self.cost_fallback_confirmation_reads += 1
            self.backend._increment("skill_card_cost_fallback_confirmation_reads")
            if self.backend._same_clicked_card_resolution(
                self.cost_fallback_candidate,
                self.resolved,
            ):
                self.resolved = ClickedSkillCard(
                    self.resolved.card_id,
                    dict(self.resolved.customizations),
                    resolution_source=("generic_cost_conservative_fallback_confirmed"),
                    detail_confirmation_reads=(self.cost_fallback_confirmation_reads),
                    detail_evidence_mode=(self.resolved.detail_evidence_mode),
                    conservative_cost_assumption=dict(self.conservative_cost_assumption),
                )
                self.backend._increment("skill_card_cost_fallback_confirmations")
            else:
                if self.cost_fallback_candidate is not None:
                    self.backend._increment("skill_card_cost_fallback_confirmation_conflicts")
                    raise ArenaReaderError(
                        "skill_card_cost_fallback_frame_conflict",
                        "settled detail frames leave different generic-cost bounds",
                    )
                self.cost_fallback_candidate = self.resolved
                if not self.cost_fallback_confirmation_extension_applied:
                    self.deadline = max(
                        self.deadline,
                        self.clock.monotonic() + 0.45,
                    )
                    self.cost_fallback_confirmation_extension_applied = True
                    self.backend._increment("skill_card_cost_fallback_confirmation_extensions")
                raise ArenaReaderError(
                    "skill_card_cost_fallback_unconfirmed",
                    "generic-cost fallback requires one fresh settled detail frame with the same bounded pair",
                )
        return _FrameStep.NEXT

    def _confirm_zero_count(self):
        if self.allow_zero_without_badge_count and self.resolved_count == 0 and self.conservative_cost_assumption is None:
            # A detail title becomes readable before its effect rows are
            # guaranteed to finish animating.  The prior context-action
            # overhead happened to hide this race; direct point dispatch
            # exposed it as a false zero.  Zero therefore crosses both a
            # minimum render boundary and one fresh, semantically
            # identical detail frame.
            if not self.frame_is_settled:
                if not self.zero_settle_wait_recorded:
                    self.zero_settle_wait_recorded = True
                    self.backend._increment("skill_card_detail_zero_settle_waits")
                raise ArenaReaderError(
                    "skill_card_detail_zero_before_settle",
                    "an unconstrained zero detail arrived before the effect-render settle boundary",
                )
            same_zero_resolution = self.backend._same_clicked_card_resolution(
                self.zero_candidate,
                self.resolved,
            )
            self.zero_confirmation_reads += 1
            self.backend._increment("skill_card_detail_zero_confirmation_reads")
            if same_zero_resolution:
                self.resolved = ClickedSkillCard(
                    self.resolved.card_id,
                    dict(self.resolved.customizations),
                    resolution_source=(f"{self.resolved.resolution_source}_zero_confirmed"),
                    detail_confirmation_reads=self.zero_confirmation_reads,
                    detail_evidence_mode=self.resolved.detail_evidence_mode,
                    conservative_cost_assumption=getattr(
                        self.resolved,
                        "conservative_cost_assumption",
                        None,
                    ),
                )
                self.backend._increment("skill_card_detail_zero_confirmations")
            else:
                if self.zero_candidate is not None:
                    self.backend._increment("skill_card_detail_zero_confirmation_conflicts")
                self.zero_candidate = self.resolved
                if not self.zero_confirmation_extension_applied:
                    # The base deadline covers positive-effect settling.
                    # Once a post-settle zero actually exists, reserve
                    # exactly one additional OCR frame for its required
                    # confirmation instead of widening every detail read.
                    self.deadline = max(self.deadline, self.clock.monotonic() + 0.45)
                    self.zero_confirmation_extension_applied = True
                    self.backend._increment("skill_card_detail_zero_confirmation_extensions")
                raise ArenaReaderError(
                    "skill_card_detail_zero_unconfirmed",
                    "an unconstrained zero detail requires one fresh semantically identical post-settle frame",
                )
        return _FrameStep.NEXT
