"""Detail lifecycle component; no imports from the Maa facade."""

from __future__ import annotations

import json
from typing import Any, Protocol
from collections.abc import Sequence

from arena_winrate import (
    TeamTarget,
    ArenaReaderError,
    ClickedSkillCard,
    ArenaCatalogError,
    BadgeReferenceError,
    stable_reference_business_candidates,
)
from p_item_recognition import PItemReferenceError
from arena_winrate.cancellation import ArenaTaskCancelled, ArenaReadSuperseded
from arena_winrate._detail_identity import (
    DetailIdentityTransaction,
    build_detail_identity_proof,
    detail_identity_proof_matches,
    matches_detail_identity_transaction,
)
from arena_winrate._reader_evidence import _DetailIdentityProof, _DetailIdentityObservation
from arena_winrate.recognition_probe import RecognitionProbe
from arena_winrate._card_detail_state import card_detail_state
from arena_winrate._detail_capture_evidence import _FullFrameOcrEvidence


class DetailLifecycleReaderPort(Protocol):
    """Only the state and callbacks needed by this responsibility."""

    def _accept_skill_detail_open_id(self, key: tuple[int, int], card_id: int, attempt: int) -> bool: ...

    _active_inferred_clicked_card: Any

    def _assert_card_group_visible(
        self, group_index: int, timeout_seconds: float = ..., *, card_slot: int | None = ..., expected_card_id: int | None = ...
    ) -> None: ...
    def _assert_inferred_card_group_visible_after_dismiss(self, group_index: int, *, card_slot: int, expected_card_id: int) -> None: ...
    def _authoritative_skill_card_title_text(
        self,
        group_index: int | None,
        detail_image: Any | None,
        *,
        candidate_card_ids: Sequence[int] = ...,
        source_card_box: tuple[int, int, int, int] | None = ...,
    ) -> str | None: ...
    def _begin_detail_diagnostics(self, *, source: Any = ..., **position: Any) -> None: ...
    def _cached_full_frame_ocr_evidence(self, image: Any) -> _FullFrameOcrEvidence | None: ...
    def _capture(self) -> Any: ...

    _card_candidate_groups: Any
    _card_detail_capture_started_at: Any
    _card_detail_images: Any
    _card_detail_last_contact_released_at: Any
    _card_detail_open_ids: Any
    _card_detail_rebind_ids: Any
    _card_detail_texts: Any
    _card_excluded_duplicate_flags: Any
    _card_identity_frames: Any
    _card_images: Any
    _card_predictions: Any
    _card_restoration_signatures: Any
    _card_rows: Any
    _card_source_guard_frames: Any
    _card_transaction_interaction_boxes: Any
    _card_transaction_kinds: Any
    _card_transaction_source_boxes: Any
    _card_transaction_started: Any
    _card_transaction_tokens: Any

    def _card_visual_family_candidates(
        self, *, stage_number: int, image: Any, box: tuple[int, int, int, int], slot_index: int
    ) -> tuple[int, ...]: ...
    def _click(self, box: tuple[int, int, int, int], *, settle_seconds: float = ...) -> None: ...
    def _close_skill_card_once(self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int) -> None: ...
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

    _detail_failure_frames: Any
    _detail_identity_proofs: Any

    def _detail_identity_transaction(self, key: tuple[int, int]) -> DetailIdentityTransaction: ...

    _detail_kind_hint: Any
    _detail_semantic_confirmations: Any

    def _dismiss_skill_card_detail(self) -> None: ...
    def _finish_card_transaction(
        self, key: tuple[int, int], *, failed: bool = ..., error: str | None = ..., superseded: bool = ..., cancelled: bool = ...
    ) -> None: ...
    def _increment(self, name: str) -> None: ...
    def _infer_zero_card_identity_from_detail_once(
        self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int, candidate_ids: Sequence[int]
    ) -> ClickedSkillCard: ...

    _inferred_clicked_cards: Any
    _matching_ocr_items: Any
    _member_failure_frames: Any

    def _note_confirmed_detail_name(self, card_id: int) -> None: ...
    def _ocr(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = ..., only_rec: bool = ...) -> list[Any]: ...
    def _open_detail_with_kind(self, kind: str, *args: Any) -> None: ...
    def _open_skill_card_once(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        expected_customization_count: int,
        *,
        contact_start_index: int = ...,
    ) -> None: ...
    def _persist_detail_failure(self, error: str) -> None: ...
    def _raw_card_candidate_boxes(self, image: Any) -> tuple[tuple[int, int, int, int], ...]: ...
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
    def _record_duration_sample(self, name: str, elapsed: float) -> None: ...
    def _resolve_proven_skill_card_title(self, title_text: str, *, allow_one_character: bool = ...) -> int: ...
    def _retry_transient_communication_items(self, items: Sequence[Any]) -> bool: ...

    _runtime_counts: Any

    def _same_proven_skill_card_title(self, first: str, second: str, card_id: int) -> bool: ...
    def _short_press(self, box: tuple[int, int, int, int], *, duration_ms: int) -> None: ...
    def _skill_card_detail_close_retry_proven(
        self, group_index: int, expected_card_id: int, source_card_box: tuple[int, int, int, int] | None = ...
    ) -> bool: ...
    def _skill_card_detail_ocr_text(self, image: Any, *, phase: str) -> str: ...

    _skill_card_detail_overlay_guard_boxes: Any
    _skill_card_interaction_box: Any
    _skill_card_retry_interaction_box: Any

    def _skill_card_source_box(self, key: tuple[int, int] | None) -> tuple[int, int, int, int] | None: ...
    def _skill_detail_other_source_slots(self, key: tuple[int, int], card_id: int) -> list[tuple[int, int]]: ...
    def _skill_detail_visual_ids(self, key: tuple[int, int]) -> tuple[int, ...]: ...
    def _sleep(self, seconds: float) -> None: ...

    _source_restore_poll_seconds: Any
    catalog: Any

    def open_skill_card(
        self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int, expected_customization_count: int
    ) -> None: ...


class DetailLifecycleReader:
    """Detail lifecycle with live, reader-owned state."""

    def __init__(
        self,
        port: DetailLifecycleReaderPort,
        *,
        content_error_limit: Any,
        logger: Any,
        source_signature_errors: Any,
        source_signatures: Any,
        clock: Any,
    ) -> None:
        self.port = port
        self.content_error_limit = content_error_limit
        self.logger = logger
        self.source_signature_errors = source_signature_errors
        self.source_signatures = source_signatures
        self.clock = clock

    def _skill_detail_visual_ids(self, key: tuple[int, int]) -> tuple[int, ...]:
        # Embeddings only supply candidates. A single first-place ID is not
        # reliable enough to turn a normal upgraded card into a retry.
        return getattr(self.port, "_card_candidate_groups", {}).get(key, ())

    def _skill_detail_other_source_slots(self, key: tuple[int, int], card_id: int) -> list[tuple[int, int]]:
        """Failure-only lookup of already frozen identities; never re-recognize."""
        matches = set()
        for other_key, predicted in getattr(self.port, "_card_predictions", {}).items():
            if other_key != key and predicted == card_id:
                matches.add(other_key)
        for group, frames in getattr(self.port, "_card_identity_frames", {}).items():
            if len(frames) != 3 or any(len(frame) != 6 for frame in frames):
                continue
            for slot in range(6):
                other_key = (group, slot + 1)
                if other_key == key:
                    continue
                try:
                    ids = stable_reference_business_candidates(tuple(frame[slot] for frame in frames))
                except BadgeReferenceError:
                    continue
                if card_id in ids:
                    matches.add(other_key)
        return sorted(matches)

    def _accept_skill_detail_open_id(self, key: tuple[int, int], card_id: int, attempt: int) -> bool:
        visual_ids = self.port._skill_detail_visual_ids(key)
        if visual_ids and card_id not in visual_ids:
            self.port._increment("skill_card_detail_id_mismatches")
            rebind = getattr(self.port, "_card_detail_rebind_ids", None)
            if rebind is None:
                rebind = self.port._card_detail_rebind_ids = {}
            other_slots = self.port._skill_detail_other_source_slots(key, card_id)
            # The same ID may legitimately appear in both six-card groups.
            # Other slots are diagnostic context, not proof of a wrong click.
            accepted = attempt == 1 and rebind.get(key) == card_id
            diagnostic = getattr(self.port, "_detail_failure_frames", None)
            if diagnostic is not None:
                diagnostic["position"].update(
                    expected_visual_ids=list(visual_ids),
                    actual_detail_id=card_id,
                    other_source_slots=other_slots,
                )
                self.port._note_confirmed_detail_name(card_id)
            self.logger.info(
                json.dumps(
                    {
                        "event": "arena_skill_detail_id_check",
                        "target_slot": key,
                        **((getattr(self.port, "_member_failure_frames", None) or {}).get("position", {})),
                        "group_index": key[0],
                        "card_slot": key[1],
                        "member_read_id": (getattr(self.port, "_member_failure_frames", None) or {}).get("read_id"),
                        "transaction_id": (diagnostic or {}).get("transaction_id"),
                        "source_card_box": self.port._skill_card_source_box(key),
                        "expected_visual_ids": visual_ids,
                        "actual_detail_id": card_id,
                        "other_source_slots": other_slots,
                        "contact": attempt + 1,
                        "outcome": "repeated_title_rebind" if accepted else "reopen" if attempt == 0 else "exhausted",
                    },
                    ensure_ascii=False,
                )
            )
            if not accepted:
                rebind[key] = card_id
                return False
            # Two independent openings plus the existing source-return and full
            # body confirmation retain detail priority over an incorrect image ID.
            self.port._increment("skill_card_detail_reopened_title_rebinds")
        if attempt == 1 and key in getattr(self.port, "_card_detail_rebind_ids", {}):
            self.port._increment("skill_card_detail_id_reopen_successes")
        opened = getattr(self.port, "_card_detail_open_ids", None)
        if opened is None:
            opened = self.port._card_detail_open_ids = {}
        opened[key] = card_id
        return True

    def _assert_skill_detail_open_id(self, key: tuple[int, int] | None, card_id: int) -> None:
        expected = getattr(self.port, "_card_detail_open_ids", {}).get(key)
        if expected is not None and expected != card_id:
            upgrade_pair = getattr(self.port.catalog, "is_skill_card_upgrade_pair", None)
            if upgrade_pair is not None and upgrade_pair(card_id, expected):
                # A missing '+' is also a valid base title. Do not parse its
                # body as either ID or count it as a confirmation observation.
                self.port._increment("skill_card_detail_upgrade_mark_missing")
                raise ArenaReaderError(
                    "skill_card_detail_upgrade_mark_missing",
                    f"target {key!r}: opened detail ID {expected}, actual detail ID {card_id}; "
                    "upgrade mark is unconfirmed; waiting for a complete title",
                )
            self.port._increment("skill_card_detail_id_changes")
            raise ArenaReaderError(
                "skill_card_detail_id_changed",
                f"target {key!r}: opened detail ID {expected}, actual detail ID {card_id}",
            )

    def _skill_card_source_box(
        self,
        key: tuple[int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        """Return the frozen source-card geometry for one detail transaction."""

        if key is None:
            return None
        group_index, card_slot = key
        row = getattr(self.port, "_card_rows", {}).get(group_index, ())
        if not 1 <= card_slot <= len(row):
            return None
        return row[card_slot - 1]

    def open_skill_card(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        expected_customization_count: int,
    ) -> None:
        started = self.clock.perf_counter()
        clicks_before = self.port._runtime_counts.get("skill_card_detail_clicks", 0)
        self.port._begin_detail_diagnostics(
            target=target,
            stage_number=stage_number,
            member_slot=member_slot,
            kind="skill_card",
            group_index=group_index,
            card_slot=card_slot,
            phase="open",
            source=getattr(self.port, "_card_images", {}).get(group_index),
        )
        try:
            self.port._open_skill_card_once(
                target,
                stage_number,
                member_slot,
                group_index,
                card_slot,
                expected_customization_count,
            )
        except Exception as error:
            if self.port._runtime_counts.get("skill_card_detail_clicks", 0) > clicks_before:
                elapsed = self.clock.perf_counter() - started
                self.port._record_duration_sample("skill_card_detail_transaction", elapsed)
                self.port._record_duration_sample(getattr(self.port, "_detail_kind_hint", "necessary_skill_card_detail_transaction"), elapsed)
                self.port._record_duration_sample("skill_card_detail_failed_transaction", elapsed)
                self.port._record_duration_sample("skill_card_detail_open_failed", elapsed)
                self.port._increment("skill_card_detail_failed_transactions")
                self.port._persist_detail_failure(str(error))
            else:
                self.port._detail_failure_frames = None
            raise

    def _open_skill_card_once(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        expected_customization_count: int,
        *,
        contact_start_index: int = 0,
    ) -> None:
        del target, stage_number, member_slot
        row = self.port._card_rows.get(group_index, ())
        if not 1 <= card_slot <= len(row):
            raise ArenaReaderError("skill_card_slot_missing", f"group {group_index}/slot {card_slot} is absent")
        duplicate_flags = self.port._card_excluded_duplicate_flags.get(group_index, ())
        if len(duplicate_flags) == 6 and duplicate_flags[card_slot - 1]:
            raise ArenaReaderError(
                "skill_card_excluded_duplicate",
                f"group {group_index}/slot {card_slot} is a non-interactive excluded duplicate",
            )
        key = (group_index, card_slot)
        inferred = self.port._inferred_clicked_cards.get(key)
        if inferred is not None:
            inferred_count = sum(int(value) for value in inferred.customizations.values())
            if inferred_count != expected_customization_count:
                raise ArenaReaderError(
                    "skill_card_inferred_count_changed",
                    f"group {group_index}/slot {card_slot} inferred {inferred_count} but the reader requested {expected_customization_count}",
                )
            self.port._active_inferred_clicked_card = key
            return
        candidate_ids = self.port._card_candidate_groups.get((group_index, card_slot))
        if candidate_ids is None:
            raise ArenaReaderError("skill_card_prediction_missing", "card icon was not classified before its click")
        card_box = row[card_slot - 1]
        primary_click_box = self.port._skill_card_interaction_box(card_box)
        retry_click_box = self.port._skill_card_retry_interaction_box(card_box)
        last_text = ""
        last_error = ""
        open_started = self.clock.perf_counter()
        detail_state = card_detail_state(self.port)
        opening = detail_state.prepare_open(key, open_started, self.port._runtime_counts)
        for attempt in range(contact_start_index, 2):
            detail_state.record_contact(key, attempt + 1)
            click_box = primary_click_box if attempt == 0 else retry_click_box
            self.port._increment("skill_card_detail_clicks")
            self.port._increment("skill_card_safe_region_clicks")
            if attempt:
                self.port._increment("skill_card_detail_click_retries")
                self.port._increment("skill_card_detail_contact_retries")
            if attempt == 0:
                self.port._short_press(click_box, duration_ms=80)
            else:
                self.port._short_press(click_box, duration_ms=120)
            last_contact_released_at = self.clock.monotonic()
            deadline = self.clock.monotonic() + 1.5
            while self.clock.monotonic() < deadline:
                try:
                    detail_capture_started_at = self.clock.monotonic()
                    detail_image = self.port._capture()
                    last_text = self.port._skill_card_detail_ocr_text(
                        detail_image,
                        phase="open",
                    )
                except ArenaReaderError:
                    last_text = ""
                try:
                    confirmed_card_id = self.port._confirm_clicked_skill_card_id(
                        last_text,
                        candidate_ids,
                        expected_customization_count=expected_customization_count,
                        source_group_index=group_index,
                        detail_image=detail_image,
                        source_card_box=card_box,
                        source_card_slot=card_slot,
                    )
                    accepted_open = self.port._accept_skill_detail_open_id(key, confirmed_card_id, attempt)
                    RecognitionProbe.observe_identity_check(
                        getattr(self.port, "_detail_failure_frames", None),
                        detail_image,
                        card_id=confirmed_card_id,
                        contact=attempt + 1,
                        accepted=accepted_open,
                    )
                    if not accepted_open:
                        last_error = (
                            f"skill_card_detail_id_mismatch: target {key!r}; "
                            f"visual_ids={self.port._skill_detail_visual_ids(key)!r}; "
                            f"actual_detail_id={confirmed_card_id}"
                        )
                        break
                    self.port._card_detail_texts[key] = last_text
                    self.port._note_confirmed_detail_name(confirmed_card_id)
                    detail_state.accept_open(
                        key,
                        opening,
                        image=detail_image,
                        contact_released_at=last_contact_released_at,
                        capture_started_at=detail_capture_started_at,
                        source_box=card_box,
                        interaction_box=click_box,
                    )
                    diagnostic = getattr(self.port, "_detail_failure_frames", None)
                    if diagnostic is not None and "confirmed_open" not in diagnostic:
                        for captured in reversed(diagnostic["frames"]):
                            if captured[1] is detail_image:
                                diagnostic["confirmed_open"] = captured
                                break
                    self.port._record_duration_sample(
                        "skill_card_detail_open_phase",
                        self.clock.perf_counter() - open_started,
                    )
                    return
                except ArenaCatalogError as error:
                    last_error = str(error)
                    evidence = self.port._cached_full_frame_ocr_evidence(detail_image)
                    if evidence is not None and self.port._retry_transient_communication_items(evidence.filtered_items):
                        self.port._sleep(0.25)
                        continue
                self.port._sleep(0.25)
            # A failed OCR/identity read does not prove that the overlay stayed
            # closed. The same card point is a toggle while a detail is open,
            # so every retry first performs an inert backdrop dismissal and
            # proves the original source generation has returned. This reset
            # is safe when the first click was non-interactive as well.
            try:
                self.port._dismiss_skill_card_detail()
                self.port._assert_card_group_visible(
                    group_index,
                    card_slot=card_slot,
                )
            except ArenaReaderError as restore_error:
                raise ArenaReaderError(
                    "skill_card_detail_missing",
                    "clicked card detail could not be confirmed and the source "
                    "page could not be restored before another contact: "
                    f"identity_error={last_error}; "
                    f"restore_error={restore_error}",
                ) from restore_error
            self.port._increment("skill_card_detail_source_resets")
        # A restored source page proves only that this transaction was safely
        # reset. It cannot distinguish a genuinely non-interactive duplicate
        # from an ordinary card whose opened detail never yielded a title.
        # Preserve that uncertainty so callers cannot silently mark the slot as
        # an excluded duplicate.
        terminal_code = "skill_card_detail_ambiguous"
        raise ArenaReaderError(
            terminal_code,
            f"clicked card did not confirm visual-family IDs {candidate_ids!r} "
            "after one bounded retry: "
            f"{last_error}; card_box={card_box!r}; "
            f"primary_click_box={primary_click_box!r}; "
            f"retry_click_box={retry_click_box!r}; "
            f"raw_boxes={self.port._raw_card_candidate_boxes(self.port._card_images[group_index])!r}; "
            f"OCR={last_text[:400]!r}",
        )

    def _open_detail_with_kind(self, kind: str, *args: Any) -> None:
        previous = getattr(self.port, "_detail_kind_hint", "necessary_skill_card_detail_transaction")
        self.port._detail_kind_hint = kind
        try:
            self.port.open_skill_card(*args)
        finally:
            self.port._detail_kind_hint = previous

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
        candidate_ids_override: Sequence[int] | None = None,
    ) -> ClickedSkillCard:
        """Resolve one seeded-plate shortlist candidate from clicked detail."""

        candidate_ids = tuple(
            dict.fromkeys(
                candidate_ids_override
                if candidate_ids_override is not None
                else self.port._card_visual_family_candidates(
                    stage_number=stage_number,
                    image=image,
                    box=box,
                    slot_index=card_slot - 1,
                )
            )
        )
        if not candidate_ids:
            raise ArenaReaderError(
                "skill_card_badge_detail_identity_unknown",
                "detail recovery has no active-catalog title candidate",
            )
        key = (group_index, card_slot)
        self.port._card_candidate_groups[key] = candidate_ids
        overlay_opened = False
        inferred = None
        detail_failure: ArenaReaderError | None = None
        interruption: BaseException | None = None
        try:
            self.port._open_detail_with_kind(
                detail_kind,
                target,
                stage_number,
                member_slot,
                group_index,
                card_slot,
                (1 if expected_customization_count is None else expected_customization_count),
            )
            overlay_opened = True
            self.port._card_transaction_kinds[key] = detail_kind
            inferred = self.port._read_resolved_card_detail(
                key,
                candidate_ids,
                expected_customization_count=expected_customization_count,
                allow_zero_without_badge_count=allow_zero_without_badge_count,
                target=target,
                stage_number=stage_number,
                member_slot=member_slot,
            )
        except (ArenaReadSuperseded, ArenaTaskCancelled) as error:
            interruption = error
            raise
        except ArenaReaderError as error:
            detail_failure = error
            if error.code == "skill_card_detail_noninteractive":
                raise
            raise ArenaReaderError(
                "skill_card_badge_detail_inference_ambiguous",
                f"group {group_index}/slot {card_slot}: {error}",
            ) from error
        finally:
            try:
                if overlay_opened and interruption is None:
                    self.port._dismiss_skill_card_detail()
            except (ArenaReadSuperseded, ArenaTaskCancelled) as error:
                interruption = error
                raise
            except Exception as error:
                if key in getattr(self.port, "_card_transaction_started", {}):
                    self.port._finish_card_transaction(key, failed=True, error=str(error))
                raise
            finally:
                if key in getattr(self.port, "_card_transaction_started", {}):
                    if interruption is not None:
                        self.port._finish_card_transaction(
                            key,
                            superseded=isinstance(interruption, ArenaReadSuperseded),
                            cancelled=isinstance(interruption, ArenaTaskCancelled),
                        )
                    elif inferred is None:
                        self.port._finish_card_transaction(
                            key,
                            failed=True,
                            error=None if detail_failure is None else str(detail_failure),
                        )
        try:
            self.port._assert_inferred_card_group_visible_after_dismiss(
                group_index,
                card_slot=card_slot,
                expected_card_id=inferred.card_id,
            )
        except Exception as error:
            if key in getattr(self.port, "_card_transaction_started", {}):
                self.port._finish_card_transaction(key, failed=True, error=str(error))
            raise
        self.port._finish_card_transaction(key)
        self.port._inferred_clicked_cards[key] = inferred
        self.port._card_predictions[key] = inferred.card_id
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
        try:
            return self.port._infer_zero_card_identity_from_detail_once(
                target,
                stage_number,
                member_slot,
                group_index,
                card_slot,
                candidate_ids,
            )
        except Exception as error:
            self.port._finish_card_transaction((group_index, card_slot), failed=True, error=str(error))
            raise

    def _infer_zero_card_identity_from_detail_once(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        candidate_ids: Sequence[int],
    ) -> ClickedSkillCard:
        """Resolve one zero-customization card by its exact title and full effects."""

        key = (group_index, card_slot)
        self.port._card_candidate_groups[key] = tuple(candidate_ids)
        overlay_opened = False
        try:
            self.port._open_detail_with_kind(
                "zero_identity_detail_transaction",
                target,
                stage_number,
                member_slot,
                group_index,
                card_slot,
                0,
            )
            overlay_opened = True
            self.port._card_transaction_kinds[key] = "zero_identity_detail_transaction"
            inferred = self.port._read_resolved_card_detail(
                key,
                candidate_ids,
                expected_customization_count=0,
                allow_zero_without_badge_count=True,
                target=target,
                stage_number=stage_number,
                member_slot=member_slot,
            )
            if inferred.customizations:
                raise ArenaReaderError(
                    "skill_card_zero_identity_detail_positive",
                    f"group {group_index}/slot {card_slot} was visually zero but "
                    f"detail resolved customizations {dict(inferred.customizations)!r}",
                )
        finally:
            if overlay_opened:
                self.port._dismiss_skill_card_detail()
        self.port._assert_inferred_card_group_visible_after_dismiss(
            group_index,
            card_slot=card_slot,
            expected_card_id=inferred.card_id,
        )
        self.port._finish_card_transaction(key)
        self.port._inferred_clicked_cards[key] = inferred
        self.port._card_predictions[key] = inferred.card_id
        self.port._increment("skill_card_zero_identity_detail_resolutions")
        self.logger.info(
            json.dumps(
                {
                    "event": "arena_zero_identity_detail_resolved",
                    "team_id": target.team_id,
                    "opponent_position": target.opponent_position,
                    "stage_number": stage_number,
                    "member_slot": member_slot,
                    "member_read_id": (getattr(self.port, "_member_failure_frames", None) or {}).get("read_id"),
                    "group_index": group_index,
                    "card_slot": card_slot,
                    "card_id": inferred.card_id,
                },
                ensure_ascii=False,
            )
        )
        return inferred

    def _detail_identity_observation(
        self,
        key: tuple[int, int],
        candidate_ids: Sequence[int],
        resolved: ClickedSkillCard,
    ) -> _DetailIdentityObservation | None:
        """Return exact-title evidence bound to the current post-contact frame."""

        transaction_started = getattr(self.port, "_card_transaction_started", {}).get(key)
        transaction_token = getattr(self.port, "_card_transaction_tokens", {}).get(key)
        source_card_box = getattr(
            self.port,
            "_card_transaction_source_boxes",
            {},
        ).get(key)
        interaction_box = getattr(
            self.port,
            "_card_transaction_interaction_boxes",
            {},
        ).get(key)
        contact_released_at = getattr(
            self.port,
            "_card_detail_last_contact_released_at",
            {},
        ).get(key)
        capture_started_at = getattr(
            self.port,
            "_card_detail_capture_started_at",
            {},
        ).get(key)
        detail_image = getattr(self.port, "_card_detail_images", {}).get(key)
        if (
            not isinstance(transaction_token, int)
            or isinstance(transaction_token, bool)
            or not isinstance(transaction_started, (int, float))
            or isinstance(transaction_started, bool)
            or source_card_box is None
            or interaction_box is None
            or not isinstance(contact_released_at, (int, float))
            or isinstance(contact_released_at, bool)
            or not isinstance(capture_started_at, (int, float))
            or isinstance(capture_started_at, bool)
            or capture_started_at <= contact_released_at
            or detail_image is None
            or tuple(source_card_box) != self.port._skill_card_source_box(key)
        ):
            return None
        source_card_box = tuple(source_card_box)
        interaction_box = tuple(interaction_box)
        if len(source_card_box) != 4 or len(interaction_box) != 4:
            return None
        source_x, source_y, source_width, source_height = source_card_box
        click_x, click_y, click_width, click_height = interaction_box
        if not (
            source_width > 0
            and source_height > 0
            and click_width > 0
            and click_height > 0
            and source_x <= click_x
            and source_y <= click_y
            and click_x + click_width <= source_x + source_width
            and click_y + click_height <= source_y + source_height
        ):
            return None
        source_guard_frames = getattr(self.port, "_card_source_guard_frames", {}).get(key[0])
        restoration_signatures = getattr(
            self.port,
            "_card_restoration_signatures",
            {},
        ).get(key[0])
        identity_frames = getattr(self.port, "_card_identity_frames", {}).get(key[0])
        if source_guard_frames is None or restoration_signatures is None or identity_frames is None:
            return None
        exact_resolver = getattr(
            getattr(self.port, "catalog", None),
            "confirm_clicked_skill_card_by_exact_title",
            None,
        )
        if exact_resolver is None:
            return None
        title_text = self.port._authoritative_skill_card_title_text(
            key[0],
            detail_image,
            candidate_card_ids=candidate_ids,
            source_card_box=source_card_box,
        )
        if title_text is None:
            return None
        try:
            exact_card_id = self.port._resolve_proven_skill_card_title(title_text)
        except ArenaCatalogError:
            return None
        if exact_card_id != resolved.card_id:
            return None
        return _DetailIdentityObservation(
            transaction_token=transaction_token,
            transaction_started=float(transaction_started),
            source_card_box=source_card_box,
            interaction_box=interaction_box,
            contact_released_at=float(contact_released_at),
            capture_started_at=float(capture_started_at),
            detail_image=detail_image,
            source_guard_frames=source_guard_frames,
            restoration_signatures=restoration_signatures,
            identity_frames=identity_frames,
            title=title_text,
            card_id=resolved.card_id,
            customizations=tuple(sorted((str(customization_id), int(count)) for customization_id, count in resolved.customizations.items())),
            evidence_mode=resolved.detail_evidence_mode,
        )

    def _detail_identity_transaction(self, key: tuple[int, int]) -> DetailIdentityTransaction:
        """Read current evidence references without copying frames or renewing them."""

        return DetailIdentityTransaction(
            transaction_started=getattr(self.port, "_card_transaction_started", {}).get(key),
            transaction_token=getattr(self.port, "_card_transaction_tokens", {}).get(key),
            source_card_box=getattr(self.port, "_card_transaction_source_boxes", {}).get(key),
            interaction_box=getattr(self.port, "_card_transaction_interaction_boxes", {}).get(key),
            contact_released_at=getattr(self.port, "_card_detail_last_contact_released_at", {}).get(key),
            capture_started_at=getattr(self.port, "_card_detail_capture_started_at", {}).get(key),
            detail_image=getattr(self.port, "_card_detail_images", {}).get(key),
            source_guard_frames=getattr(self.port, "_card_source_guard_frames", {}).get(key[0]),
            restoration_signatures=getattr(self.port, "_card_restoration_signatures", {}).get(key[0]),
            identity_frames=getattr(self.port, "_card_identity_frames", {}).get(key[0]),
        )

    def _record_detail_identity_proof(
        self,
        key: tuple[int, int],
        resolved: ClickedSkillCard,
        observations: Sequence[_DetailIdentityObservation],
    ) -> None:
        """Publish independent identity only after checking the current opening."""

        proofs = getattr(self.port, "_detail_identity_proofs", None)
        if proofs is None:
            proofs = {}
            self.port._detail_identity_proofs = proofs
        proofs.pop(key, None)
        proof = build_detail_identity_proof(
            resolved,
            observations,
            self.port._same_proven_skill_card_title,
        )
        # Read the active transaction AFTER the title comparisons, as before.
        if proof is not None and matches_detail_identity_transaction(
            proof,
            self.port._detail_identity_transaction(key),
        ):
            proofs[key] = proof

    def _detail_identity_proof_matches(
        self,
        key: tuple[int, int],
        expected_card_id: int | None,
        source_card_box: tuple[int, int, int, int] | None,
    ) -> bool:
        """Return whether an active transaction carries exact semantic identity."""

        if expected_card_id is None or source_card_box is None or key not in getattr(self.port, "_card_transaction_started", {}):
            return False
        proof = getattr(self.port, "_detail_identity_proofs", {}).get(key)
        if not isinstance(proof, _DetailIdentityProof):
            return False
        return detail_identity_proof_matches(
            proof,
            expected_card_id,
            source_card_box,
            self.port._detail_identity_transaction(key),
        )

    def _skill_card_detail_disappeared(
        self,
        key: tuple[int, int],
        opened_image: Any,
        opened_capture_time: float | None,
        transaction_token: int | None,
        source_frames: tuple[Any, ...] | None,
        recent_frames: tuple[tuple[float, Any], ...],
    ) -> bool:
        """Classify a final body failure before dismissal, using cached frames only.

        A successful close after an ordinary semantic failure proves nothing
        about premature disappearance. Require the opened, title-confirmed
        transaction to differ from its source and two later frames to already
        match that source, under the existing overlay-restoration pixel gate.
        """
        if (
            transaction_token is None
            or getattr(self.port, "_card_transaction_tokens", {}).get(key) != transaction_token
            or key not in getattr(self.port, "_card_transaction_started", {})
            or source_frames is None
            or len(source_frames) != 3
            or getattr(self.port, "_card_source_guard_frames", {}).get(key[0]) is not source_frames
            or opened_image is None
            or opened_capture_time is None
            or len(recent_frames) != 2
            or not opened_capture_time < recent_frames[0][0] < recent_frames[1][0]
            or recent_frames[0][1] is recent_frames[1][1]
            or any(image is opened_image for _, image in recent_frames)
            or self.port._card_detail_images.get(key) is not recent_frames[-1][1]
        ):
            return False
        started = self.clock.perf_counter()
        try:
            shape = getattr(opened_image, "shape", None)
            if shape is None or any(
                getattr(image, "shape", None) != shape
                for image in (
                    *source_frames,
                    *(image for _, image in recent_frames),
                )
            ):
                return False
            for _, image in recent_frames:
                evidence = self.port._cached_full_frame_ocr_evidence(image)
                if evidence is None or not evidence.hit:
                    return False
                if (
                    len(self.port._matching_ocr_items(evidence.filtered_items, r"^体力$")) != 1
                    or len(self.port._matching_ocr_items(evidence.filtered_items, r"^総合力$")) != 1
                ):
                    return False
            boxes = self.port._skill_card_detail_overlay_guard_boxes(opened_image)
            source_signatures = tuple(self.source_signatures(image, boxes) for image in source_frames)

            def source_matches(image: Any) -> int:
                signature = self.source_signatures(image, boxes)
                errors = tuple(self.source_signature_errors(source, signature) for source in source_signatures)
                return sum(bool(values) and max(values) <= self.content_error_limit for values in errors)

            return source_matches(opened_image) == 0 and all(source_matches(image) >= 2 for _, image in recent_frames)
        except (ArenaReaderError, PItemReferenceError, TypeError, ValueError):
            # Missing or uncertain proof retains the original semantic error.
            return False
        finally:
            self.port._record_duration_sample(
                "skill_card_detail_disappearance_check",
                self.clock.perf_counter() - started,
            )

    def _record_detail_semantic_confirmation(
        self,
        key: tuple[int, int],
        resolved: ClickedSkillCard,
        *,
        reason: str,
        resolution_conflicts: int,
    ) -> None:
        self.port._detail_semantic_confirmations[key] = {
            "slot": key[1],
            "card_id": resolved.card_id,
            "resolved_detail_count": sum(int(value) for value in resolved.customizations.values()),
            "customizations": dict(resolved.customizations),
            "reason": reason,
            "detail_confirmation_reads": resolved.detail_confirmation_reads,
            "resolution_conflicts": resolution_conflicts,
            "evidence_mode": resolved.detail_evidence_mode,
        }

    def close_skill_card(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
    ) -> None:
        diagnostic = getattr(self.port, "_detail_failure_frames", None)
        if diagnostic is not None:
            diagnostic["position"]["phase"] = "close"
        try:
            self.port._close_skill_card_once(target, stage_number, member_slot, group_index, card_slot)
        except Exception as error:
            self.port._finish_card_transaction((group_index, card_slot), failed=True, error=str(error))
            raise

    def _close_skill_card_once(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
    ) -> None:
        del target, stage_number, member_slot
        key = (group_index, card_slot)
        if self.port._active_inferred_clicked_card == key:
            # The candidate-detail inference already dismissed the overlay and
            # proved this exact slot against the accepted source generation.
            # Reusing the cached result performs no UI action, so a second
            # screenshot/signature guard here cannot add state evidence.
            self.port._active_inferred_clicked_card = None
            self.port._inferred_clicked_cards.pop(key, None)
            self.port._increment("skill_card_inferred_detail_reuses")
            return
        self.port._dismiss_skill_card_detail()
        try:
            self.port._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
                expected_card_id=self.port._card_predictions.get(key),
            )
        except ArenaReaderError:
            image = self.port._capture()
            if len(self.port._ocr(image, r"^ステージ\s*[123]$")) >= 3:
                raise ArenaReaderError(
                    "skill_card_close_left_member",
                    "closing a skill card unexpectedly left the member detail",
                )
            self.port._dismiss_skill_card_detail()
            self.port._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
                expected_card_id=self.port._card_predictions.get(key),
            )
        self.port._finish_card_transaction(key)

    def _dismiss_skill_card_detail(self, *, image: Any = None) -> None:
        """Close a skill-card detail by tapping the inert upper-left backdrop."""

        # A successfully opened skill-card transaction already owns a fresh,
        # accepted detail capture.  The dismissal point consumes only the
        # normalized frame dimensions, never its pixels, so taking another
        # screenshot here adds no state evidence.  Reuse is deliberately
        # limited to the sole active transaction; failed opens, P-item flows,
        # diagnostic calls, malformed frames, and inconsistent state retain
        # the original capture fallback.
        active_keys = tuple(getattr(self.port, "_card_transaction_started", {}))
        dimensions: tuple[int, int] | None = None
        # Progressive cleanup has just captured and checked this frame. Only
        # its dimensions are needed for dismissal; no pixels are reused later.
        shape = getattr(image, "shape", ())
        if len(shape) >= 2 and int(shape[0]) > 0 and int(shape[1]) > 0:
            dimensions = (int(shape[0]), int(shape[1]))
            self.port._increment("skill_card_detail_dismiss_shape_reuses")
        detail_images = getattr(self.port, "_card_detail_images", {})
        if dimensions is None and len(active_keys) == 1 and active_keys[0] in detail_images:
            detail_image = detail_images[active_keys[0]]
            shape = getattr(detail_image, "shape", ())
            if len(shape) >= 2:
                height, width = int(shape[0]), int(shape[1])
                if height > 0 and width > 0:
                    dimensions = (height, width)
                    self.port._increment("skill_card_detail_dismiss_shape_reuses")
        if dimensions is None:
            image = self.port._capture()
            height, width = image.shape[:2]
        else:
            height, width = dimensions
        self.port._increment("skill_card_detail_event_driven_dismissals")
        self.port._click(
            (
                max(1, int(width * 0.015)),
                max(1, int(height * 0.04)),
                1,
                1,
            ),
            settle_seconds=0,
        )

    def _skill_card_detail_close_retry_proven(
        self,
        group_index: int,
        expected_card_id: int,
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> bool:
        """Prove on two fresh frames that the same detail overlay stayed open."""

        exact_resolver = getattr(
            self.port.catalog,
            "confirm_clicked_skill_card_by_exact_title",
            None,
        )
        if exact_resolver is None:
            return False
        for frame_index in range(2):
            image = self.port._capture()
            title_text = self.port._authoritative_skill_card_title_text(
                group_index,
                image,
                candidate_card_ids=(expected_card_id,),
                source_card_box=source_card_box,
            )
            if title_text is None:
                return False
            try:
                confirmed_card_id = self.port._resolve_proven_skill_card_title(title_text)
            except ArenaCatalogError:
                return False
            if confirmed_card_id != expected_card_id:
                return False
            self.port._increment("skill_card_detail_close_retry_evidence_frames")
            if frame_index == 0:
                self.port._sleep(self.port._source_restore_poll_seconds)
        return True

    def _assert_inferred_card_group_visible_after_dismiss(
        self,
        group_index: int,
        *,
        card_slot: int,
        expected_card_id: int,
    ) -> None:
        """Restore one inferred-card source, retrying one proven dropped dismiss."""

        diagnostic = getattr(self.port, "_detail_failure_frames", None)
        if diagnostic is not None:
            diagnostic["position"]["phase"] = "close"
        try:
            self.port._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
                expected_card_id=expected_card_id,
            )
        except ArenaReaderError as error:
            if error.code != "skill_card_close_failed":
                raise
            if not self.port._skill_card_detail_close_retry_proven(
                group_index,
                expected_card_id,
                self.port._skill_card_source_box((group_index, card_slot)),
            ):
                raise
            self.port._increment("skill_card_detail_close_retries")
            self.port._dismiss_skill_card_detail()
            self.port._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
                expected_card_id=expected_card_id,
            )


def _same_clicked_card_resolution(left: ClickedSkillCard | None, right: ClickedSkillCard) -> bool:
    return (
        left is not None
        and left.card_id == right.card_id
        and dict(left.customizations) == dict(right.customizations)
        and getattr(left, "conservative_cost_assumption", None) == getattr(right, "conservative_cost_assumption", None)
    )


def _detail_confirmation_wait_message(reason: str, *, expected_customization_count: int | None, resolved_count: int) -> str:
    if reason == "badge_count_override":
        return (
            f"card-face count {expected_customization_count} conflicts with "
            f"unique detail count {resolved_count}; awaiting one fresh confirmation"
        )
    return (
        f"badge count {resolved_count} admits multiple legal customization groups; "
        "awaiting the same final-effect resolution from one fresh frame"
    )
