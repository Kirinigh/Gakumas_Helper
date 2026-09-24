"""Card customizations component; no imports from the Maa facade."""

from __future__ import annotations

import json
from typing import Any, Protocol
from dataclasses import asdict
from collections.abc import Callable, Sequence

from arena_winrate import (
    TeamTarget,
    ArenaReaderError,
    ClickedSkillCard,
    ArenaCatalogError,
    CustomizationBadgeState,
    GenericCostReferenceError,
    CustomizationBadgeDecision,
    GenericCostReferenceGallery,
    stable_generic_cost_value,
)


class CardCustomizationsReaderPort(Protocol):
    """Only the state and callbacks needed by this responsibility."""

    _active_inferred_clicked_card: Any

    def _add_timing(self, name: str, elapsed: float) -> None: ...
    def _auxiliary_badge_glyph_count(
        self, key: tuple[int, int], *, maximum_count: int, admissible_counts: Sequence[int] | None = ..., allow_ocr: bool = ...
    ) -> int: ...

    _badge_count_diagnostics: Any

    @staticmethod
    def _badge_detail_inferable(decision: CustomizationBadgeDecision) -> bool: ...
    def _batched_badge_local_results(self, requested_group: int) -> tuple[Any, ...]: ...
    def _capture(self) -> Any: ...

    _card_candidate_groups: Any

    def _card_cost_references(self) -> GenericCostReferenceGallery: ...

    _card_count_frames: Any
    _card_detail_images: Any
    _card_empty_flags: Any
    _card_excluded_duplicate_flags: Any
    _card_face_cost_diagnostics: Any
    _card_face_cost_failed_sources: Any
    _card_face_cost_optional_errors: Any
    _card_images: Any
    _card_predictions: Any
    _card_rows: Any

    def _confirm_clicked_skill_card_id(
        self,
        detail_text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = ...,
        detail_image: Any | None = ...,
        source_card_box: tuple[int, int, int, int] | None = ...,
        source_card_slot: int | None = ...,
    ) -> int: ...

    _cost_customization_fallbacks: Any
    _detail_count_overrides: Any

    def _finish_badge_candidate_detail(self, entry: dict[str, Any], result: ClickedSkillCard) -> None: ...
    def _increment(self, name: str) -> None: ...
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
        allow_zero_without_badge_count: bool = ...,
        candidate_ids_override: Sequence[int] | None = ...,
    ) -> ClickedSkillCard: ...

    _inferred_clicked_cards: Any

    def _measure_card_face_generic_cost(self, key: tuple[int, int], card_id: int) -> tuple[int, int, int] | None: ...
    def _measure_optional_card_face_generic_cost(self, key: tuple[int, int], card_id: int) -> tuple[int, int, int] | None: ...

    _member_failure_frames: Any

    def _read_resolved_card_detail(
        self,
        key: tuple[int, int],
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        timeout_seconds: float = ...,
        allow_zero_without_badge_count: bool = ...,
        target: TeamTarget | None = ...,
        stage_number: int | None = ...,
        member_slot: int | None = ...,
    ) -> ClickedSkillCard: ...
    def _record_badge_candidate_detail(
        self, *, stage_number: int, member_slot: int, group_index: int, card_slot: int, phase: str, reason: str
    ) -> dict[str, Any]: ...
    def _refresh_visible_card_groups(self, requested_group: int) -> tuple[Any, Any, Any]: ...
    def _run_skill_card_gallery_detail(
        self,
        operation: Callable[[], ClickedSkillCard],
        *,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        additional_zero_detail: bool = ...,
    ) -> ClickedSkillCard: ...

    _runtime_cost_customization_fallbacks: Any

    def _skill_card_catalog_candidates_for_plan(self, plan: str, *, slot_index: int | None = ...) -> tuple[int, ...]: ...
    def _skill_card_forward_drift_for_plan(self, plan: str) -> tuple[int, ...]: ...
    def _skill_card_forward_drift_slot_indices(self, missing_card_ids: Sequence[int], *, plan: str | None = ...) -> tuple[int, ...]: ...
    def _skill_card_source_box(self, key: tuple[int, int] | None) -> tuple[int, int, int, int] | None: ...
    def _sleep(self, seconds: float) -> None: ...
    def _validated_card_group_row(
        self, image: Any, group_index: int, *, allow_fixed_secondary: bool = ...
    ) -> tuple[tuple[int, int, int, int], ...]: ...

    catalog: Any
    season: Any


class CardCustomizationsReader:
    """Card customizations with live, reader-owned state."""

    def __init__(self, port: CardCustomizationsReaderPort, *, logger: Any, clock: Any, validate_cost_fallback: Any) -> None:
        self.port = port
        self.logger = logger
        self.clock = clock
        self.validate_cost_fallback = validate_cost_fallback

    def read_skill_card_customization_counts(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[int]:
        row = self.port._card_rows.get(group_index, ())
        prepared = self.port._card_images.get(group_index)
        if len(row) != 6 or prepared is None:
            raise ArenaReaderError(
                "skill_card_customization_count_input_mismatch",
                f"group {group_index} is missing its six boxes or prepared frame",
            )
        plan = self.port.season.stages[stage_number - 1].plan
        forward_drift_ids = self.port._skill_card_forward_drift_for_plan(plan)
        drift_slot_indices = self.port._skill_card_forward_drift_slot_indices(
            forward_drift_ids,
            plan=plan,
        )
        ordinary_drift = any(slot >= 2 for slot in drift_slot_indices)
        drift_title_candidates = {slot: self.port._skill_card_catalog_candidates_for_plan(plan, slot_index=slot) for slot in drift_slot_indices}
        cached_frames = self.port._card_count_frames.get(group_index, ())
        if len(cached_frames) == 3:
            frame_samples = list(cached_frames)
        else:
            frame_samples = [prepared]
            for _ in range(2):
                self.port._sleep(0.12)
                frame_samples.append(self.port._capture())
        frames = tuple(frame_samples)
        for attempt in range(2):
            try:
                frame_rows = tuple(self.port._validated_card_group_row(image, group_index) for image in frames)
            except ArenaReaderError as error:
                if error.code != "skill_card_customization_frame_incomplete" or attempt == 1:
                    raise
                frames = self.port._refresh_visible_card_groups(group_index)
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
                frames = self.port._refresh_visible_card_groups(group_index)
                continue
            duplicate_flags = self.port._card_excluded_duplicate_flags.get(
                group_index,
                (False,) * 6,
            )
            empty_flags = getattr(self.port, "_card_empty_flags", {}).get(
                group_index,
                (False,) * 6,
            )
            local_results = self.port._batched_badge_local_results(group_index)
            decisions = []
            count_diagnostics = []
            for slot_index, (local_result, excluded, empty) in enumerate(zip(local_results, duplicate_flags, empty_flags, strict=True)):
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
                if local_decision.reason == "all_frames_oversized_center_foreground_contradiction":
                    self.port._increment("skill_card_badge_oversized_art_strong_zeros")
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
                            "fail_closed" if not self.port._badge_detail_inferable(local_decision) else "badge_candidate_detail"
                        ),
                        "plate_diagnostics": plate_diagnostics,
                    }
                )
            decisions = tuple(decisions)
            counts = [decision.count for decision in decisions]
            for index, count in enumerate(counts, start=1):
                if count is not None:
                    continue
                inferred = self.port._inferred_clicked_cards.get((group_index, index))
                if inferred is None:
                    continue
                inferred_count = sum(int(value) for value in inferred.customizations.values())
                if inferred_count < 1:
                    fallback_claimed = (
                        inferred.conservative_cost_assumption is not None
                        or inferred.resolution_source.startswith("generic_cost_conservative_fallback")
                        or inferred.detail_evidence_mode == "conservative_generic_cost_bound"
                    )
                    if fallback_claimed:
                        self.validate_cost_fallback(
                            inferred,
                            catalog=self.port.catalog,
                            expected_policy=("assume_unenhanced" if target.is_own_team else "assume_enhanced"),
                            expected_count=None,
                            expected_count_kind="observed",
                        )
                    elif not (
                        not inferred.customizations
                        and (
                            (inferred.resolution_source == "detail_known_zero" and inferred.detail_evidence_mode == "known_zero")
                            or (
                                inferred.resolution_source.endswith("_zero_confirmed")
                                and not isinstance(
                                    inferred.detail_confirmation_reads,
                                    bool,
                                )
                                and isinstance(
                                    inferred.detail_confirmation_reads,
                                    int,
                                )
                                and inferred.detail_confirmation_reads >= 2
                            )
                        )
                    ):
                        raise ArenaReaderError(
                            "skill_card_zero_cache_contract_invalid",
                            f"cached zero detail for group {group_index}/slot {index} is neither known-zero nor freshly confirmed",
                        )
                counts[index - 1] = inferred_count
                count_diagnostics[index - 1].update(
                    {
                        "fallback_state": ("POSITIVE" if inferred_count > 0 else CustomizationBadgeState.CONFIDENT_ZERO.value),
                        "fallback_reason": (
                            "cached_detail_unique_positive_count"
                            if inferred_count > 0
                            else (
                                "cached_generic_cost_conservative_zero"
                                if inferred.conservative_cost_assumption is not None
                                else "cached_detail_confirmed_zero"
                            )
                        ),
                        "fallback_mode": "clicked_detail_cache",
                        "inferred_card_id": inferred.card_id,
                        "inferred_customizations": dict(inferred.customizations),
                    }
                )
            counts = tuple(counts)
            unknown_slots = tuple(index for index, count in enumerate(counts, start=1) if count is None)
            self.port._badge_count_diagnostics[group_index] = tuple(count_diagnostics)
            if unknown_slots:
                inferable = tuple(index for index in unknown_slots if self.port._badge_detail_inferable(decisions[index - 1]))
                if inferable == unknown_slots:
                    inferred_counts = list(counts)
                    for index in inferable:
                        reason = decisions[index - 1].reason
                        detail_check = self.port._record_badge_candidate_detail(
                            stage_number=stage_number,
                            member_slot=member_slot,
                            group_index=group_index,
                            card_slot=index,
                            phase="post_swipe_card_group",
                            reason=reason,
                        )
                        try:

                            def infer_badge():
                                return self.port._infer_badge_card_from_detail(
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
                                    candidate_ids_override=(drift_title_candidates[index - 1] if index - 1 in drift_slot_indices else None),
                                )

                            inferred = (
                                self.port._run_skill_card_gallery_detail(
                                    infer_badge,
                                    target=target,
                                    stage_number=stage_number,
                                    member_slot=member_slot,
                                    group_index=group_index,
                                    card_slot=index,
                                )
                                if ordinary_drift and index - 1 in drift_slot_indices
                                else infer_badge()
                            )
                        except ArenaReaderError as error:
                            if error.code == "skill_card_detail_noninteractive":
                                current_flags = list(
                                    self.port._card_excluded_duplicate_flags.get(
                                        group_index,
                                        (False,) * 6,
                                    )
                                )
                                current_flags[index - 1] = True
                                self.port._card_excluded_duplicate_flags[group_index] = tuple(current_flags)
                                inferred_counts[index - 1] = 0
                                detail_check.update(
                                    {
                                        "resolved_state": "EXCLUDED_DUPLICATE",
                                        "functional_evidence": ("click_and_bounded_press_noninteractive"),
                                    }
                                )
                                count_diagnostics[index - 1].update(
                                    {
                                        "fallback_state": "EXCLUDED_DUPLICATE",
                                        "fallback_reason": ("stable_card_body_noninteractive"),
                                        "fallback_mode": ("functional_duplicate_evidence"),
                                    }
                                )
                                self.port._increment("skill_card_noninteractive_duplicate_fallbacks")
                                continue
                            raise ArenaReaderError(
                                error.code,
                                f"{error.detail}; badge_diagnostic={count_diagnostics[index - 1]!r}",
                            ) from error
                        self.port._finish_badge_candidate_detail(detail_check, inferred)
                        inferred_count = sum(int(value) for value in inferred.customizations.values())
                        inferred_counts[index - 1] = inferred_count
                        count_diagnostics[index - 1].update(
                            {
                                "fallback_state": ("POSITIVE" if inferred_count > 0 else CustomizationBadgeState.CONFIDENT_ZERO.value),
                                "fallback_reason": (
                                    "clicked_detail_unique_positive_count" if inferred_count > 0 else "clicked_detail_unique_zero"
                                ),
                                "fallback_mode": "clicked_detail_unique_state",
                                "inferred_card_id": inferred.card_id,
                                "inferred_customizations": dict(inferred.customizations),
                            }
                        )
                    self.port._badge_count_diagnostics[group_index] = tuple(count_diagnostics)
                    self.port._card_rows[group_index] = baseline
                    self.port._card_images[group_index] = frames[0]
                    self.port._card_count_frames[group_index] = frames
                    return tuple(int(count) for count in inferred_counts)
                if attempt == 0:
                    frames = self.port._refresh_visible_card_groups(group_index)
                    continue
                raise ArenaReaderError(
                    "skill_card_customization_count_unknown",
                    f"group {group_index}/slots {unknown_slots} did not produce three consistent "
                    "badge reads after one corrective reread; "
                    f"reasons={tuple(decisions[index - 1].reason for index in unknown_slots)!r}; "
                    f"diagnostics={tuple(count_diagnostics[index - 1] for index in unknown_slots)!r}",
                )
            row = baseline
            self.port._card_rows[group_index] = row
            self.port._card_images[group_index] = frames[0]
            self.port._card_count_frames[group_index] = frames
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
        key = (group_index, card_slot)
        inferred = self.port._inferred_clicked_cards.get(key)
        if self.port._active_inferred_clicked_card == key and inferred is not None:
            inferred_count = sum(int(value) for value in inferred.customizations.values())
            if inferred_count != expected_customization_count:
                raise ArenaReaderError(
                    "skill_card_inferred_count_changed",
                    f"group {group_index}/slot {card_slot} inferred {inferred_count} but detail read requested {expected_customization_count}",
                )
            return inferred
        candidate_ids = self.port._card_candidate_groups.get((group_index, card_slot))
        if candidate_ids is None:
            raise ArenaReaderError(
                "skill_card_prediction_missing",
                "card icon was not classified before its click",
            )
        resolved = self.port._read_resolved_card_detail(
            key,
            candidate_ids,
            expected_customization_count=expected_customization_count,
            target=target,
            stage_number=stage_number,
            member_slot=member_slot,
        )
        # The embedding result is only a pre-click visual-family hint.  Once
        # the detail title/effects resolve the business ID, bind that
        # authoritative identity before the close transaction verifies the
        # restored source slot.  Otherwise a valid family mismatch (for
        # example hint 752 resolving to actual card 747) is misreported as a
        # failed close even though the original grid returned unchanged.
        self.port._card_predictions[key] = resolved.card_id
        return resolved

    def _resolve_clicked_card_text(
        self,
        candidate_ids: Sequence[int],
        text: str,
        *,
        expected_customization_count: int | None,
        allow_zero_without_badge_count: bool = False,
        allow_auxiliary_badge_glyph: bool = True,
        generic_cost_fallback_policy: str | None = None,
        allow_generic_cost_fallback: bool = False,
        generic_cost_coverage_context: tuple[str, str, int] | None = None,
        key: tuple[int, int] | None = None,
    ) -> ClickedSkillCard:
        conservative_cost_assumption: dict[str, Any] | None = None
        try:
            card_id = self.port._confirm_clicked_skill_card_id(
                text,
                candidate_ids,
                expected_customization_count=expected_customization_count,
                source_group_index=(None if key is None else key[0]),
                detail_image=(None if key is None else getattr(self.port, "_card_detail_images", {}).get(key)),
                source_card_box=self.port._skill_card_source_box(key),
                source_card_slot=None if key is None else key[1],
            )
            generic_cost_frame_values = self.port._measure_optional_card_face_generic_cost(key, card_id) if key is not None else None
            cost_error = getattr(self.port, "_card_face_cost_optional_errors", {}).get(key) if key is not None else None
            if expected_customization_count == 0:
                customizations = self.port.catalog.resolve_effective_customizations(
                    card_id,
                    text,
                    expected_count=0,
                    allow_positive_completion=False,
                )
                if customizations:
                    raise ArenaCatalogError("a known-zero card resolved a positive customization state")
                resolution_source = "detail_known_zero"
                detail_evidence_mode = "known_zero"
            elif generic_cost_fallback_policy is not None and cost_error is not None:
                if generic_cost_fallback_policy not in {
                    "assume_unenhanced",
                    "assume_enhanced",
                }:
                    raise ArenaReaderError(
                        "skill_card_cost_fallback_policy_invalid",
                        f"unsupported generic-cost fallback policy {generic_cost_fallback_policy!r}",
                    )
                if not allow_generic_cost_fallback:
                    self.port._increment("skill_card_cost_fallback_settle_waits")
                    raise ArenaReaderError(
                        "skill_card_cost_fallback_before_settle",
                        "generic-cost fallback is deferred until settled detail evidence",
                    )
                coverage_kwargs: dict[str, Any] = {}
                if generic_cost_coverage_context is not None:
                    full_detail_text, effect_roi_text, observation_count = generic_cost_coverage_context
                    coverage_kwargs = {
                        "full_detail_text": full_detail_text,
                        "effect_roi_text": effect_roi_text,
                        "effect_roi_title_bound": True,
                        "effect_roi_observation_count": observation_count,
                    }
                pair = self.port.catalog.resolve_generic_cost_ambiguity(
                    card_id,
                    text,
                    observed_badge_count=expected_customization_count,
                    **coverage_kwargs,
                )
                unenhanced = dict(pair.unenhanced_customizations)
                enhanced = dict(pair.enhanced_customizations)
                customizations = enhanced if generic_cost_fallback_policy == "assume_enhanced" else unenhanced
                hypotheses = self.port.catalog.generic_cost_hypotheses(card_id)
                if hypotheses is None:
                    raise ArenaCatalogError(f"card {card_id} lost its generic-cost hypotheses")
                conservative_cost_assumption = {
                    "reason_code": "skill_card_cost_evidence_inconclusive",
                    "policy": generic_cost_fallback_policy,
                    "card_id": card_id,
                    "generic_customization_id": pair.generic_customization_id,
                    "observed_badge_count": expected_customization_count,
                    "cost_hypotheses": list(hypotheses),
                    "non_cost_customizations": {key: value for key, value in unenhanced.items() if key != str(pair.generic_customization_id)},
                    "candidate_customizations": {
                        "unenhanced": unenhanced,
                        "enhanced": enhanced,
                    },
                    "applied_customizations": dict(customizations),
                }
                resolution_source = "generic_cost_conservative_fallback"
                detail_evidence_mode = "conservative_generic_cost_bound"
            elif expected_customization_count is None:
                detail_evidence_mode = "unconstrained"
                customizations = None
                if generic_cost_frame_values is not None:
                    try:
                        customizations = self.port.catalog.resolve_effective_customizations_with_generic_cost_evidence(
                            card_id,
                            text,
                            generic_cost_frame_values=generic_cost_frame_values,
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
                        detail_evidence_mode = self.port.catalog.effective_customization_evidence_mode(
                            card_id,
                            text,
                            expected_count=sum(int(value) for value in customizations.values()),
                            resolved=customizations,
                        )
                if customizations is None:
                    resolver = (
                        self.port.catalog.resolve_effective_customizations_unconstrained
                        if allow_zero_without_badge_count
                        else self.port.catalog.resolve_effective_customizations_without_badge_count
                    )
                    try:
                        customizations = resolver(card_id, text)
                    except ArenaCatalogError:
                        if not allow_zero_without_badge_count or key is None:
                            raise
                        if not allow_auxiliary_badge_glyph:
                            self.port._increment("skill_card_detail_auxiliary_glyph_deferred")
                            raise
                        try:
                            admissible_counts = self.port.catalog.admissible_clicked_customization_counts(
                                card_id,
                                text,
                                generic_cost_frame_values=(generic_cost_frame_values),
                            )
                            auxiliary_count = self.port._auxiliary_badge_glyph_count(
                                key,
                                maximum_count=(self.port.catalog.maximum_customization_count(card_id)),
                                admissible_counts=admissible_counts,
                            )
                        except ArenaReaderError as error:
                            cost_error = getattr(
                                self.port,
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
                            customizations = self.port.catalog.resolve_effective_customizations(
                                card_id,
                                text,
                                expected_count=0,
                                allow_positive_completion=False,
                            )
                            if customizations:
                                raise ArenaCatalogError("stable non-badge card art did not resolve a zero state")
                            resolution_source = "detail_zero_auxiliary_same_plate_non_badge_art"
                            detail_evidence_mode = "stable_same_plate_non_badge_art_zero"
                            resolution = None
                        else:
                            resolution_kwargs = {
                                "observed_badge_count": auxiliary_count,
                            }
                            if generic_cost_frame_values is not None:
                                resolution_kwargs["generic_cost_frame_values"] = generic_cost_frame_values
                            resolution = self.port.catalog.resolve_clicked_customizations(
                                card_id,
                                text,
                                **resolution_kwargs,
                            )
                        if resolution is not None:
                            customizations = resolution.customizations
                            resolution_source = f"{resolution.source}_auxiliary_badge_glyph"
                            detail_evidence_mode = getattr(
                                resolution,
                                "evidence_mode",
                                "negative_dependent",
                            )
                    else:
                        resolution_source = "detail_unconstrained"
                        resolved_count = sum(int(value) for value in customizations.values())
                        if resolved_count:
                            detail_evidence_mode = self.port.catalog.effective_customization_evidence_mode(
                                card_id,
                                text,
                                expected_count=resolved_count,
                                resolved=customizations,
                            )
            else:
                resolution_kwargs: dict[str, Any] = {
                    "observed_badge_count": expected_customization_count,
                }
                if generic_cost_frame_values is not None:
                    resolution_kwargs["generic_cost_frame_values"] = generic_cost_frame_values
                resolution = self.port.catalog.resolve_clicked_customizations(
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
            conservative_cost_assumption=conservative_cost_assumption,
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
        candidate_ids = self.port._card_candidate_groups.get(key)
        if candidate_ids is None:
            raise ArenaReaderError(
                "skill_card_prediction_missing",
                "card identity candidates were not established before its truth click",
            )
        return self.port._read_resolved_card_detail(
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
        diagnostics = getattr(self.port, "_card_face_cost_diagnostics", None)
        if diagnostics is None:
            diagnostics = {}
            self.port._card_face_cost_diagnostics = diagnostics
        cached = diagnostics.get(key)
        hypotheses = self.port.catalog.generic_cost_hypotheses(card_id)
        if hypotheses is None:
            # The visual-family hint may resolve to a different authoritative
            # detail ID.  Do not leave evidence from the hinted generic-cost
            # card attached to a slot whose confirmed card has no such field.
            diagnostics.pop(key, None)
            getattr(self.port, "_card_face_cost_failed_sources", {}).pop(key, None)
            return None
        hypotheses = tuple(int(value) for value in hypotheses)
        if cached is not None:
            cached_hypotheses = cached.get("hypotheses")
            cache_matches = cached.get("card_id") == card_id and isinstance(cached_hypotheses, list) and tuple(cached_hypotheses) == hypotheses
            if cache_matches:
                values = cached.get("frame_values")
                if isinstance(values, list) and len(values) == 3 and all(type(value) is int for value in values):
                    return int(values[0]), int(values[1]), int(values[2])
                raise ArenaReaderError(
                    "skill_card_cost_evidence_invalid",
                    f"cached card-face cost evidence is invalid for {key!r}",
                )
            # Detail title resolution is authoritative over the pre-click
            # visual-family hint.  Re-evaluate the already frozen three source
            # frames against the confirmed identity/hypotheses instead of
            # reusing a semantically stale integer or taking another screenshot.
            self.port._increment("skill_card_face_cost_cache_rebinds")
        group_index, card_slot = key
        frames = self.port._card_count_frames.get(group_index, ())
        if len(frames) != 3:
            raise ArenaReaderError(
                "skill_card_cost_frames_missing",
                f"group {group_index} has no stable three-frame cost evidence",
            )
        started = self.clock.perf_counter()
        failed_sources = getattr(self.port, "_card_face_cost_failed_sources", None)
        if failed_sources is None:
            failed_sources = self.port._card_face_cost_failed_sources = {}
        frame_rows = tuple(self.port._validated_card_group_row(frame, group_index) for frame in frames)
        boxes = tuple(tuple(row[card_slot - 1]) for row in frame_rows)
        prior_failure = failed_sources.get(key)
        if (
            prior_failure is not None
            and prior_failure["card_id"] == card_id
            and prior_failure["hypotheses"] == hypotheses
            and prior_failure["boxes"] == boxes
            and all(old is new for old, new in zip(prior_failure["frames"], frames, strict=True))
        ):
            self.port._increment("skill_card_face_cost_inconclusive_cache_hits")
            raise ArenaReaderError("skill_card_cost_evidence_inconclusive", prior_failure["error"])
        failed_sources.pop(key, None)
        predictions = ()
        try:
            predictions = tuple(
                self.port._card_cost_references().measure(
                    frame,
                    row[card_slot - 1],
                    card_id=card_id,
                    allowed_values=hypotheses,
                )
                for frame, row in zip(frames, frame_rows, strict=True)
            )
            value = stable_generic_cost_value(predictions)
            values = (value, value, value)
        except (GenericCostReferenceError,) as error:
            detail = f"group {group_index}/slot {card_slot}/card {card_id}: {error}"
            failed_sources[key] = {
                "card_id": card_id,
                "hypotheses": hypotheses,
                "frames": tuple(frames),
                "boxes": boxes,
                "error": detail,
                "predictions": [asdict(prediction) for prediction in predictions],
            }
            # Keep rejected evidence too; a later abort may prevent the lineup
            # summary from being emitted. No new image or OCR is acquired here.
            self.logger.info(
                json.dumps(
                    {
                        "event": "arena_card_cost_inconclusive",
                        **((getattr(self.port, "_member_failure_frames", None) or {}).get("position", {})),
                        "member_read_id": (getattr(self.port, "_member_failure_frames", None) or {}).get("read_id"),
                        "group_index": group_index,
                        "card_slot": card_slot,
                        "card_id": card_id,
                        "hypotheses": hypotheses,
                        "boxes": boxes,
                        "reason": detail,
                        "predictions": failed_sources[key]["predictions"],
                    },
                    ensure_ascii=False,
                )
            )
            raise ArenaReaderError(
                "skill_card_cost_evidence_inconclusive",
                detail,
            ) from error
        finally:
            self.port._add_timing(
                "skill_card_face_cost_evidence",
                self.clock.perf_counter() - started,
            )
        self.port._increment("skill_card_face_cost_measurements")
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
                    "zero_hole_box": (list(prediction.zero_hole_box) if prediction.zero_hole_box is not None else None),
                    "zero_alternative_error": prediction.zero_alternative_error,
                    "component_boxes": [list(box) for box in prediction.component_boxes],
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

        errors = getattr(self.port, "_card_face_cost_optional_errors", None)
        if errors is None:
            errors = {}
            self.port._card_face_cost_optional_errors = errors
        try:
            values = self.port._measure_card_face_generic_cost(key, card_id)
        except ArenaReaderError as error:
            if error.code != "skill_card_cost_evidence_inconclusive":
                raise
            self.port._increment("skill_card_face_cost_optional_inconclusive")
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
        values = self.port._measure_card_face_generic_cost(key, card_id)
        if values is None:
            return None
        group_index, card_slot = key
        try:
            mode = self.port.catalog.certify_card_face_generic_cost(
                card_id,
                resolved=customizations,
                frame_values=values,
            )
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_cost_evidence_conflict",
                f"group {group_index}/slot {card_slot}/card {card_id}: {error}",
            ) from error
        diagnostic = self.port._card_face_cost_diagnostics[key]
        if diagnostic.get("mode") != mode:
            self.port._increment("skill_card_face_cost_certifications")
            diagnostic["mode"] = mode
        return values

    def _record_detail_count_override(
        self,
        key: tuple[int, int],
        observed_badge_count: int,
        resolved: ClickedSkillCard,
    ) -> None:
        resolved_count = sum(int(value) for value in resolved.customizations.values())
        self.port._detail_count_overrides[key] = {
            "slot": key[1],
            "observed_badge_count": observed_badge_count,
            "resolved_detail_count": resolved_count,
            "card_id": resolved.card_id,
            "customizations": dict(resolved.customizations),
            "detail_confirmation_reads": resolved.detail_confirmation_reads,
        }
        self.port._increment("skill_card_detail_count_overrides")

    def _record_cost_customization_fallback(
        self,
        key: tuple[int, int],
        resolved: ClickedSkillCard,
        *,
        target: TeamTarget | None,
        stage_number: int | None,
        member_slot: int | None,
    ) -> None:
        assumption = getattr(
            resolved,
            "conservative_cost_assumption",
            None,
        )
        if assumption is None:
            return
        if target is None or stage_number not in (1, 2, 3) or member_slot not in (1, 2, 3):
            raise ArenaReaderError(
                "skill_card_cost_fallback_location_missing",
                "a conservative generic-cost fallback lacks its team/member location",
            )
        self.validate_cost_fallback(
            resolved,
            catalog=self.port.catalog,
            expected_policy=("assume_unenhanced" if target.is_own_team else "assume_enhanced"),
            expected_count=sum(int(value) for value in resolved.customizations.values()),
            expected_count_kind="effective",
        )
        record = {
            **dict(assumption),
            "side": "own" if target.is_own_team else "opponent",
            "team_id": target.team_id,
            "opponent_position": target.opponent_position,
            "stage_number": stage_number,
            "member_slot": member_slot,
            "group_index": key[0],
            "card_slot": key[1],
            "detail_confirmation_reads": resolved.detail_confirmation_reads,
            "resolution_source": resolved.resolution_source,
            "detail_evidence_mode": resolved.detail_evidence_mode,
        }
        per_member = getattr(self.port, "_cost_customization_fallbacks", None)
        if per_member is None:
            per_member = {}
            self.port._cost_customization_fallbacks = per_member
        prior = per_member.get(key)
        if prior is not None:
            if prior != record:
                raise ArenaReaderError(
                    "skill_card_cost_fallback_changed",
                    f"generic-cost fallback for {key!r} changed within one member",
                )
            return
        per_member[key] = record
        runtime = getattr(self.port, "_runtime_cost_customization_fallbacks", None)
        if runtime is None:
            runtime = []
            self.port._runtime_cost_customization_fallbacks = runtime
        runtime.append(dict(record))
        self.logger.info(
            json.dumps(
                {
                    "event": "arena_cost_customization_fallback_recorded",
                    **record,
                    "member_read_id": (getattr(self.port, "_member_failure_frames", None) or {}).get("read_id"),
                    "cost_evidence_error": getattr(self.port, "_card_face_cost_optional_errors", {}).get(key),
                },
                ensure_ascii=False,
            )
        )
        self.port._increment("skill_card_cost_customization_fallbacks")
        self.port._increment(
            "skill_card_cost_customization_fallbacks_own" if target.is_own_team else "skill_card_cost_customization_fallbacks_opponent"
        )
