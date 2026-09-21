"""One P-item opening, bounded observation, and mandatory source restoration."""

from __future__ import annotations

from enum import Enum, auto

from .reader import ArenaReaderError
from ._p_item_detail_evidence import PItemDetailEvidence


class _FrameStep(Enum):
    NEXT = auto()
    REPEAT = auto()
    STOP = auto()


class PItemDetailSession:
    def __init__(
        self,
        backend,
        clock,
        *,
        box,
        candidate_ids,
        global_title_scope,
        plan,
        slot_index,
        source_boxes,
        source_images,
        unrepresented_ids,
        visual_tiebreak_ids,
    ):
        self.backend = backend
        self.clock = clock
        self.box = box
        self.candidate_ids = candidate_ids
        self.global_title_scope = global_title_scope
        self.plan = plan
        self.slot_index = slot_index
        self.source_boxes = source_boxes
        self.source_images = source_images
        self.unrepresented_ids = unrepresented_ids
        self.visual_tiebreak_ids = visual_tiebreak_ids

    def run(self):
        self.slot_scope = {} if self.slot_index is None else {"slot_index": self.slot_index}
        if not self.candidate_ids:
            raise ArenaReaderError(
                "p_item_detail_candidates_missing",
                "P-item detail fallback requires fixed-reference candidates",
            )
        started = self.clock.perf_counter()

        self.evidence = PItemDetailEvidence(
            self.backend,
            self.candidate_ids,
            self.plan,
            self.source_images,
            self.slot_scope,
            self.source_boxes,
            self.global_title_scope,
            self.visual_tiebreak_ids,
            self.unrepresented_ids,
            self.clock,
        )
        self.evidence.prepare_source()
        safe_region = self.backend._p_item_interaction_box(self.box)
        self.interaction_point = self.backend._box_center_point(safe_region)
        self.backend._click(self.interaction_point, settle_seconds=0)
        self.backend._increment("p_item_detail_clicks")
        self.backend._increment("p_item_safe_region_clicks")
        self.transaction_states = ["SOURCE_STABLE", "CLICK_SENT"]
        self.deadline = self.clock.monotonic() + 3.0
        self.last_text = ""
        self.last_panel_text = ""
        self.last_title_text = ""
        self.last_error = "P-item detail OCR did not run"
        self.last_text_resolution = ""
        self.resolved: int | None = None
        self.candidate_title_seen = False
        self.detail_confirmed = False
        self.consecutive_source_reads = 0
        self.consecutive_non_source_reads = 0
        self.last_unique_match: int | None = None
        self.consecutive_unique_matches = 0
        self.unique_match_used_effect = False
        self.retried = False
        self.terminal_error: Exception | None = None
        self.last_source_visual_errors: tuple[float, ...] = ()
        self.last_source_match_route = "none"
        self.last_source_row_uncertain = False
        self.consecutive_source_transition_reads = 0
        self.last_frame_may_have_overlay = False
        open_wait_started = self.clock.perf_counter()
        self._wait_for_detail()
        self.backend._add_timing(
            "p_item_detail_open_wait",
            self.clock.perf_counter() - open_wait_started,
        )
        recovery_started = self.clock.perf_counter()
        try:
            if self.detail_confirmed or self.terminal_error is not None or self.last_frame_may_have_overlay:
                self.backend._dismiss_skill_card_detail()
                self.transaction_states.append("DISMISS_SENT")
            if self.source_images and self.source_boxes:
                self.backend._assert_p_item_source_restored(
                    self.source_images,
                    self.source_boxes,
                )
            else:
                self.backend._assert_member_detail()
            self.transaction_states.append("SOURCE_RESTORED")
        except ArenaReaderError as recovery_error:
            elapsed = self.clock.perf_counter() - started
            self.backend._add_timing("p_item_detail_transactions", elapsed)
            self.backend._add_timing(
                "p_item_detail_close_restore",
                self.clock.perf_counter() - recovery_started,
            )
            self.backend._record_duration_sample("p_item_detail_failed_transaction", elapsed)
            raise ArenaReaderError(
                "p_item_detail_recovery_failed",
                "P-item detail transaction did not restore the original member page; "
                f"states={self.transaction_states!r}; cause={recovery_error}",
            ) from recovery_error
        self.backend._add_timing(
            "p_item_detail_close_restore",
            self.clock.perf_counter() - recovery_started,
        )
        elapsed = self.clock.perf_counter() - started
        self.backend._add_timing("p_item_detail_transactions", elapsed)
        if self.terminal_error is not None:
            self.backend._record_duration_sample("p_item_detail_failed_transaction", elapsed)
            raise ArenaReaderError(
                "p_item_detail_parser_failed",
                f"P-item detail parsing failed after the source page was restored; states={self.transaction_states!r}; error={self.last_error}",
            ) from self.terminal_error
        if self.resolved is None:
            self.backend._record_duration_sample("p_item_detail_failed_transaction", elapsed)
            error_code = (
                "p_item_detail_ambiguous" if self.detail_confirmed or self.candidate_title_seen else "p_item_detail_open_or_title_failed"
            )
            raise ArenaReaderError(
                error_code,
                f"P-item detail did not uniquely confirm {tuple(self.candidate_ids)!r}: "
                f"{self.last_error}; states={self.transaction_states!r}; "
                f"text_resolution={self.last_text_resolution!r}; "
                f"safe_box={safe_region!r}; click_point={self.interaction_point!r}; "
                f"retried={self.retried}; "
                f"source_match_route={self.last_source_match_route!r}; "
                f"source_row_uncertain={self.last_source_row_uncertain}; "
                f"source_visual_errors="
                f"{tuple(round(value, 6) for value in self.last_source_visual_errors)!r}; "
                f"title_OCR={self.last_title_text[:160]!r}; "
                f"panel_OCR={self.last_panel_text[:240]!r}; OCR={self.last_text[:400]!r}",
            )
        self.backend._record_duration_sample("p_item_detail_transaction", elapsed)
        return self.resolved

    def _wait_for_detail(self):
        while self.clock.monotonic() < self.deadline:
            self.last_text_resolution = ""
            self.diagnostic = getattr(self.backend, "_detail_failure_frames", None)
            if self.diagnostic is not None:
                self.diagnostic.pop("p_item_text", None)
            step = self._observe_detail()
            if step is _FrameStep.STOP:
                break
            if step is _FrameStep.REPEAT:
                continue
            source_visible = False
            source_transition = False
            if self.terminal_error is not None:
                break
            if self.image is not None:
                self.last_frame_may_have_overlay = True
                try:
                    if self.source_images and self.source_boxes:
                        source_visible, self.last_source_visual_errors = self.backend._p_item_source_frame_matches(
                            self.source_images,
                            self.image,
                            self.source_boxes,
                            self.member_anchors_visible,
                        )
                        if source_visible:
                            self.last_source_match_route = "visual_frozen_generation"
                        elif (
                            self.member_anchors_visible
                            and self.current_title_signature
                            and self.current_title_signature in self.evidence.stable_source_title_signatures
                        ):
                            source_transition = True
                            self.last_source_match_route = "semantic_source_transition"
                        else:
                            self.last_source_match_route = "none"
                    else:
                        source_visible = ("体力" in self.last_text and "総合力" in self.last_text) or bool(
                            self.backend._ocr(self.image, r"^体力$") and self.backend._ocr(self.image, r"^総合力$")
                        )
                except Exception as error:
                    self.terminal_error = error
                    self.last_error = str(error)
                    break
            if source_visible:
                self.last_frame_may_have_overlay = False
                self.last_unique_match = None
                self.consecutive_unique_matches = 0
                self.unique_match_used_effect = False
                self.consecutive_source_reads += 1
                self.consecutive_non_source_reads = 0
                self.consecutive_source_transition_reads = 0
                self.last_error = "the original member page remained stable after the click"
            elif source_transition:
                self.last_unique_match = None
                self.consecutive_unique_matches = 0
                self.unique_match_used_effect = False
                self.consecutive_source_reads = 0
                self.consecutive_non_source_reads = 0
                self.consecutive_source_transition_reads += 1
                self.backend._increment("p_item_source_transition_frames")
                self.last_error = "the frozen source title generation remained visible while source pixels were transitioning"
                if self.consecutive_source_transition_reads >= 8:
                    self.detail_confirmed = True
                    self.transaction_states.extend(("SOURCE_TRANSITION_EXHAUSTED", "AMBIGUOUS"))
                    break
            elif len(self.matches) == 1:
                self.consecutive_source_transition_reads = 0
                current_match = self.matches[0]
                if self.last_unique_match == current_match:
                    self.consecutive_unique_matches += 1
                    self.unique_match_used_effect = self.unique_match_used_effect or self.effect_disambiguated
                else:
                    self.consecutive_unique_matches = 1
                    self.unique_match_used_effect = self.effect_disambiguated
                self.last_unique_match = current_match
                if self.consecutive_unique_matches < 2:
                    self.candidate_title_seen = True
                    self.detail_confirmed = True
                    self.consecutive_source_reads = 0
                    self.last_error = f"P-item title needs a second consecutive targeted OCR confirmation for {current_match}"
                    self.backend._sleep(0.12)
                    continue
                self.resolved = self.matches[0]
                if self.diagnostic is not None:
                    self.diagnostic["position"]["p_item_name"] = self.last_title_text
                if self.unique_match_used_effect:
                    self.backend._increment("p_item_detail_effect_disambiguations")
                self.detail_confirmed = True
                self.transaction_states.extend(("DETAIL_CONFIRMED", "RESOLVED"))
                break
            elif self.matches:
                self.consecutive_source_transition_reads = 0
                self.last_unique_match = None
                self.consecutive_unique_matches = 0
                self.unique_match_used_effect = False
                self.candidate_title_seen = True
                self.detail_confirmed = True
                self.consecutive_source_reads = 0
                self.transaction_states.append("DETAIL_CONFIRMED")
                self.last_error = f"P-item detail matched multiple candidate titles {self.matches!r}"
            else:
                self.consecutive_source_transition_reads = 0
                self.last_unique_match = None
                self.consecutive_unique_matches = 0
                self.unique_match_used_effect = False
                self.consecutive_source_reads = 0
                self.consecutive_non_source_reads += 1
                if self.last_text:
                    if self.raw_matches and self.last_text_resolution:
                        self.last_error = f"P-item title recognized; body unresolved: {self.last_text_resolution}"
                    elif self.last_source_row_uncertain:
                        self.last_error = "P-item detail first changed row remained one OCR substitution from a frozen source row"
                    else:
                        self.last_error = f"P-item detail exposed no fixed candidate title within {tuple(self.candidate_ids)!r}"
                if self.consecutive_non_source_reads >= 2 and self.last_text:
                    self.detail_confirmed = True
                    self.transaction_states.extend(("DETAIL_CONFIRMED", "AMBIGUOUS"))
                    break
            if not self.retried and not self.detail_confirmed and self.consecutive_source_reads >= 3:
                self.backend._click(self.interaction_point, settle_seconds=0)
                self.backend._increment("p_item_detail_click_retries")
                self.backend._increment("p_item_safe_region_clicks")
                self.retried = True
                self.consecutive_source_reads = 0
                self.transaction_states.append("CLICK_SENT_RETRY")
            elif self.retried and self.consecutive_source_reads >= 3:
                self.transaction_states.append("OPEN_FAILED")
                break
            self.backend._sleep(0.12)

    def _observe_detail(self):
        try:
            self.image = self.backend._capture()
            ocr_started = self.clock.perf_counter()
            (
                self.last_text,
                self.last_panel_text,
                self.member_anchors_visible,
                current_rows,
                _,
                current_atoms,
            ) = self.evidence.observe_p_item(self.image)
            self.backend._add_timing(
                "p_item_detail_ocr",
                self.clock.perf_counter() - ocr_started,
            )
            self.current_title_signature = tuple(normalized for _, normalized, _, _ in current_rows)
            self.last_source_row_uncertain = False
            used_source_rows: set[int] = set()
            used_background_rows: set[tuple[int, int]] = set()
            self.last_title_text = ""
            for raw, normalized, row_box, components in current_rows:
                exact_sources = tuple(
                    (index,)
                    for index, (source_row, source_box) in enumerate(self.evidence.stable_source_title_rows)
                    if index not in used_source_rows and normalized == source_row and self.evidence.same_source_position(row_box, source_box)
                )
                if exact_sources:
                    (source_index,) = min(exact_sources)
                    used_source_rows.add(source_index)
                    continue
                extended_sources = tuple(
                    (index,)
                    for index, (source_row, source_box) in enumerate(self.evidence.stable_source_title_rows)
                    if index not in used_source_rows
                    and self.evidence.same_source_position(row_box, source_box)
                    and self.evidence.numeric_source_row_extension(normalized, source_row)
                )
                verified_extended_sources = tuple(
                    (index,)
                    for (index,) in extended_sources
                    if self.evidence.verified_numeric_source_row_extension(
                        components,
                        self.evidence.stable_source_title_rows[index][0],
                        self.evidence.stable_source_title_rows[index][1],
                    )
                )
                if verified_extended_sources:
                    if self.evidence.match_title_rows_exact(raw):
                        self.last_title_text = raw
                        break
                    # A numeric value can be misread as one Latin letter and
                    # joined to an unchanged source label (for example,
                    # ``総合力`` + ``15Z``). The independently positioned
                    # source label still proves this is a source-row
                    # extension, not the detail title.
                    (source_index,) = min(verified_extended_sources)
                    used_source_rows.add(source_index)
                    continue
                if extended_sources:
                    self.last_source_row_uncertain = True
                    break
                edited_sources = tuple(
                    index
                    for index, (source_row, source_box) in enumerate(self.evidence.stable_source_title_rows)
                    if index not in used_source_rows
                    and self.evidence.same_source_position(row_box, source_box)
                    and self.evidence.one_insertion_or_deletion(normalized, source_row)
                )
                if edited_sources:
                    # A genuine catalog title (including a new '+') wins
                    # over source-row reuse, even with identical pixels.
                    if self.evidence.match_title_rows_exact(raw):
                        self.last_title_text = raw
                        break
                    unchanged_source = next(
                        (index for index in edited_sources if self.evidence.source_row_pixels_unchanged(self.image, row_box, index)),
                        None,
                    )
                    if unchanged_source is not None:
                        used_source_rows.add(unchanged_source)
                        self.backend._increment("p_item_source_row_pixel_identity_reuses")
                        continue
                approximate_source = any(
                    index not in used_source_rows
                    and self.evidence.same_source_position(row_box, source_box)
                    and self.evidence.one_substitution(normalized, source_row)
                    for index, (source_row, source_box) in enumerate(self.evidence.stable_source_title_rows)
                )
                if approximate_source:
                    # A one-character difference at a frozen source
                    # position is not removable evidence: doing so could
                    # promote effect prose into the title slot.
                    self.last_source_row_uncertain = True
                    break
                extra_sources: set[tuple[int, int]] = set()
                for frame_id, background_rows in self.evidence.stable_source_background_frames:
                    candidates = tuple(
                        index
                        for index, (_, source_row, source_box, _) in enumerate(background_rows)
                        if normalized == source_row and self.evidence.same_source_position(row_box, source_box)
                    )
                    if len(candidates) != 1:
                        continue
                    source_index = candidates[0]
                    source_key = (frame_id, source_index)
                    source_box = background_rows[source_index][2]
                    if (
                        source_key in used_background_rows
                        or sum(
                            current_normalized == normalized and self.evidence.same_source_position(current_box, source_box)
                            for _, current_normalized, current_box, _ in current_rows
                        )
                        != 1
                        or any(
                            index in used_source_rows and source_row == normalized and self.evidence.same_source_position(source_box, old_box)
                            for index, (source_row, old_box) in enumerate(self.evidence.stable_source_title_rows)
                        )
                    ):
                        continue
                    extra_sources.add(source_key)
                if len(
                    {frame_id for frame_id, _ in extra_sources}
                ) >= self.evidence.background_majority_required and not self.evidence.match_title_rows_exact(raw):
                    used_background_rows.update(extra_sources)
                    self.backend._increment("p_item_source_background_row_reuses")
                    continue
                self.last_title_text = raw
                break
            # The first row newly introduced by the detail panel is the
            # identity row. Later rows are effect prose and can contain
            # other catalog titles, so they are never matching evidence.
            self.raw_matches = self.evidence.match_title_rows(self.last_title_text)
            if not self.raw_matches and self.source_images and current_atoms:
                isolated_row = self.backend._isolated_p_item_title_row(self.image, self.source_images, current_atoms)
                if isolated_row is not None:
                    self.last_title_text, row_box, isolated_components = isolated_row
                    components = tuple((text, self.evidence.normalize_title_row(text), bounds) for text, bounds in isolated_components)
                    self.raw_matches = self.evidence.match_title_rows(self.last_title_text)
                    self.backend._increment("p_item_isolated_title_reuses")
            self.matches = self.raw_matches
            self.effect_disambiguated = False
            text_resolver = getattr(self.backend.catalog, "resolve_clicked_p_item_text", None)
            if callable(text_resolver) and self.last_title_text and self.raw_matches:
                # The first new row was fixed above without consulting
                # effect prose. Merge only its original title components;
                # every other same-frame OCR atom retains its own boundary.
                title_components = {(value, tuple(box)) for value, _, box in components}
                text_atoms = [atom for atom in current_atoms if (str(atom["text"]).strip(), tuple(atom["box"])) not in title_components]
                title_index = len(text_atoms)
                if current_atoms:
                    text_atoms.append({"text": self.last_title_text, "box": row_box})
                text_started = self.clock.perf_counter()
                text_result = text_resolver(
                    text_atoms,
                    title_index=title_index,
                    source_frames=tuple(value[5] for value in self.evidence.source_observations) if self.source_images else (),
                    plan=self.plan,
                    candidate_p_item_ids=self.candidate_ids,
                    **self.slot_scope,
                    visual_tiebreak_p_item_ids=self.visual_tiebreak_ids,
                    title_text=self.last_title_text,
                    detail_text=self.last_text,
                    allow_one_substitution=not bool(self.evidence.match_title_rows_exact(self.last_title_text)),
                )
                if text_result.status != "unique" and current_atoms and self.source_images:
                    from arena_winrate._detail_title_region import isolated_p_item_body_region

                    panel_box = isolated_p_item_body_region(self.image, self.source_images, tuple(row_box))
                    if panel_box is not None:
                        text_result = text_resolver(
                            text_atoms,
                            title_index=title_index,
                            source_frames=tuple(value[5] for value in self.evidence.source_observations),
                            plan=self.plan,
                            candidate_p_item_ids=self.candidate_ids,
                            **self.slot_scope,
                            visual_tiebreak_p_item_ids=self.visual_tiebreak_ids,
                            title_text=self.last_title_text,
                            detail_text=self.last_text,
                            allow_one_substitution=not bool(self.evidence.match_title_rows_exact(self.last_title_text)),
                            panel_box=panel_box,
                        )
                        self.backend._increment("p_item_detail_panel_reuses")
                self.backend._add_timing("p_item_detail_text", self.clock.perf_counter() - text_started)
                self.backend._increment(f"p_item_detail_text_{text_result.status}")
                self.last_text_resolution = f"{text_result.status}: {text_result.reason}"
                if self.diagnostic is not None:
                    self.diagnostic["p_item_text"] = {
                        "status": text_result.status,
                        "reason": text_result.reason,
                        "ids": text_result.ids,
                        **text_result.diagnostics,
                    }
                self.matches = text_result.ids if text_result.status == "unique" else self.raw_matches
                if text_result.status != "unique" and len(self.matches) == 1:
                    self.matches = ()
                self.effect_disambiguated = text_result.status == "unique" and (
                    len(self.raw_matches) != 1 or text_result.ids != self.raw_matches
                )
            elif len(self.raw_matches) > 1:
                effect_matcher = getattr(
                    self.backend.catalog,
                    "clicked_p_item_effect_detail_matches",
                    None,
                )
                if callable(effect_matcher):
                    effect_matches = effect_matcher(
                        self.last_text,
                        candidate_p_item_ids=self.raw_matches,
                    )
                    if len(effect_matches) == 1 and effect_matches[0] in self.raw_matches:
                        self.matches = effect_matches
                        self.effect_disambiguated = True
        except Exception as error:
            self.terminal_error = error
            self.last_error = str(error)
            self.raw_matches = ()
            self.matches = ()
            self.effect_disambiguated = False
            self.image = None
            self.current_title_signature = ()
            self.last_source_row_uncertain = False
        return _FrameStep.NEXT
