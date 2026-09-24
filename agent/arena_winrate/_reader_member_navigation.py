"""Member navigation component; no imports from the Maa facade."""

from __future__ import annotations

import re
import json
from typing import Any, Protocol
from collections import Counter
from collections.abc import Sequence

from arena_winrate import (
    TeamTarget,
    MemberSlotState,
    ArenaReaderError,
    MemberSlotMetrics,
    stage_member_cap,
    fixed_member_slot_boxes,
    infer_parameter_label_boxes,
)
from arena_winrate._reader_visual import _box, _text
from arena_winrate._reader_evidence import _DetailIdentityProof


class MemberNavigationReaderPort(Protocol):
    """Only the state and callbacks needed by this responsibility."""

    def _add_timing(self, name: str, elapsed: float) -> None: ...
    def _assert_member_detail(self, timeout_seconds: float = ...) -> None: ...
    def _assert_support_bonus_closed(self) -> None: ...
    def _back(self, *, image: Any = ...) -> None: ...
    def _capture(self) -> Any: ...

    _card_stage_plan: Any

    def _check_cancelled(self) -> None: ...
    def _click(self, box: tuple[int, int, int, int], *, settle_seconds: float = ...) -> None: ...

    _detail_failure_frames: Any

    def _detail_identity_proof_matches(
        self, key: tuple[int, int], expected_card_id: int | None, source_card_box: tuple[int, int, int, int] | None
    ) -> bool: ...

    _detail_identity_proofs: Any

    def _full_ocr_text(self, image: Any, *, spatial_reading_order: bool = ...) -> str: ...

    _grade: Any
    _has_support_bonus_heading: Any

    def _increment(self, name: str) -> None: ...

    _last_detail_failure_location: Any

    def _long_press(self, box: tuple[int, int, int, int]) -> None: ...

    _matching_ocr_items: Any
    _member_boxes: Any
    _member_failure_frames: Any
    _member_metric_baseline: Any

    def _member_recovery_page_matches(self, items: Sequence[Any], stage_number: int, detail: dict[str, Any]) -> bool: ...

    _member_slot_observations: Any

    def _note_confirmed_detail_name(self, card_id: int) -> None: ...
    def _ocr(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = ..., only_rec: bool = ...) -> list[Any]: ...

    _pending_stage_preview: Any

    def _read_enhanced_numeric_roi(self, image: Any, roi: tuple[int, int, int, int]) -> tuple[int | None, tuple[str, ...]]: ...
    def _read_member_recovery_page(self, image: Any = ...) -> tuple[Any, Sequence[Any]]: ...
    def _record_duration_sample(self, name: str, elapsed: float) -> None: ...
    def _reset_member_card_state(self) -> None: ...
    def _retry_transient_communication_items(self, items: Sequence[Any]) -> bool: ...

    _runtime_counts: Any
    _runtime_duration_samples: Any
    _runtime_timing_seconds: Any

    def _same_proven_skill_card_title(self, first: str, second: str, card_id: int) -> bool: ...
    def _sleep(self, seconds: float) -> None: ...

    _spatial_ocr_rows: Any
    _support_bonus_from_overlay: Any
    _support_bonus_retried_teams: Any

    def _team_stage_total_anchors(self, target: TeamTarget, image: Any) -> tuple[tuple[int, int, int, int], ...]: ...
    def _wait_for_stage_member_list(self, expected_total_counts: tuple[int, ...], timeout_seconds: float = ...) -> None: ...
    def close_member(
        self, target: TeamTarget, stage_number: int, *, recovery_image: Any = ..., recovery_detail: dict[str, Any] | None = ...
    ) -> None: ...

    season: Any


class MemberNavigationReader:
    """Member navigation with live, reader-owned state."""

    def __init__(self, port: MemberNavigationReaderPort, *, classify_member_slot: Any, logger: Any, clock: Any) -> None:
        self.port = port
        self.classify_member_slot = classify_member_slot
        self.logger = logger
        self.clock = clock

    def member_slots(self, target: TeamTarget, stage_number: int) -> Sequence[int]:
        preview = getattr(self.port, "_pending_stage_preview", None)
        self.port._pending_stage_preview = None
        if preview is not None and preview[:2] == (target, stage_number):
            self.port._check_cancelled()
            _, _, image, totals = preview
            self.port._increment("stage_preview_observation_reuses")
        else:
            image = self.port._capture()
            totals = self.port._team_stage_total_anchors(target, image)
        anchor = totals[stage_number - 1]
        height, width = image.shape[:2]
        if self.port._grade is None:
            raise ArenaReaderError("arena_grade_missing", "contest Grade was not read from the arena main screen")
        slot_cap = stage_member_cap(self.port._grade, stage_number)
        avatar_height = int(height * 0.0906)
        avatar_top = anchor[1] + 35
        occupied: dict[int, tuple[int, int, int, int]] = {}
        observations: dict[int, tuple[MemberSlotState, MemberSlotMetrics, tuple[int, int, int, int]]] = {}
        for slot, box in enumerate(
            fixed_member_slot_boxes(
                width,
                avatar_top,
                avatar_height,
                3,
                group_center_ratio=(0.6430555556 if target.is_own_team else 0.7361111111),
            ),
            start=1,
        ):
            x0, y0, avatar_width, slot_height = box
            crop = image[y0 : y0 + slot_height, x0 : x0 + avatar_width]
            if crop.shape[:2] != (slot_height, avatar_width):
                raise ArenaReaderError("member_slot_roi_invalid", f"member slot exceeds frame: {box}")
            state, metrics = self.classify_member_slot(crop)
            observations[slot] = (state, metrics, box)
        for slot, (state, metrics, box) in observations.items():
            if slot > slot_cap:
                if state is MemberSlotState.OCCUPIED:
                    raise ArenaReaderError(
                        "member_slot_outside_grade_cap_occupied",
                        f"{target.team_id}/stage-{stage_number}/slot-{slot} is occupied "
                        f"outside Grade {self.port._grade} cap {slot_cap}: {metrics}",
                    )
                if state is MemberSlotState.AMBIGUOUS:
                    raise ArenaReaderError(
                        "member_slot_outside_grade_cap_ambiguous",
                        f"{target.team_id}/stage-{stage_number}/slot-{slot} is not proven empty "
                        f"outside Grade {self.port._grade} cap {slot_cap}: {metrics}",
                    )
                continue
            if state is MemberSlotState.OCCUPIED:
                occupied[slot] = box
            elif state is MemberSlotState.AMBIGUOUS:
                raise ArenaReaderError(
                    "member_slot_ambiguous",
                    f"{target.team_id}/stage-{stage_number}/slot-{slot} is neither an explicit black blank nor a confirmed portrait: {metrics}",
                )
        if not occupied:
            raise ArenaReaderError(
                "member_count_mismatch",
                f"{target.team_id}/stage-{stage_number} has no occupied member within Grade {self.port._grade} cap {slot_cap}",
            )
        key = (target.team_id, stage_number)
        self.port._member_boxes[key] = occupied
        self.port._member_slot_observations[key] = observations
        return tuple(occupied)

    def open_member(self, target: TeamTarget, stage_number: int, slot: int) -> None:
        self.port._last_detail_failure_location = None
        boxes = self.port._member_boxes.get((target.team_id, stage_number), {})
        box = boxes.get(slot)
        if box is None:
            raise ArenaReaderError("member_slot_missing", f"member slot {slot} has no recognized card")
        self.port._reset_member_card_state()
        self.port._card_stage_plan = self.port.season.stages[stage_number - 1].plan
        self.port._member_metric_baseline = (
            dict(self.port._runtime_timing_seconds),
            dict(self.port._runtime_counts),
            {key: len(values) for key, values in self.port._runtime_duration_samples.items()},
        )
        self.port._long_press(box)
        try:
            self.port._assert_member_detail()
        except ArenaReaderError:
            # A long-press may occasionally be dropped.  Retry it once only
            # while the complete team-preview anchors prove that the first
            # action did not navigate anywhere; never repeat on an ambiguous
            # or partially transitioned screen.
            image = self.port._capture()
            self.port._team_stage_total_anchors(target, image)
            if len(self.port._ocr(image, r"^ステージ\s*[123]$")) < 3:
                raise
            self.port._sleep(0.5)
            self.port._long_press(box)
            self.port._assert_member_detail()

    def read_support_bonus(self, target: TeamTarget) -> float:
        image = self.port._capture()
        height, width = image.shape[:2]
        info_box = (
            int(width * 0.955),
            int(height * 0.022),
            1,
            1,
        )
        self.port._click(info_box, settle_seconds=0)
        started = self.clock.perf_counter()
        deadline = self.clock.monotonic() + 2.0
        value: float | None = None
        last_text = ""
        last_close_count = 0
        recent_member_frames: list[bool] = []

        def read_overlay(*, reuse_close_label: bool = False) -> float | None:
            nonlocal last_text, last_close_count
            overlay = self.port._capture()
            items = self.port._ocr(overlay, r".+")
            try:
                last_text = self.port._full_ocr_text(overlay)
            except ArenaReaderError:
                last_text = ""
            # Preserve the original normal two-query cadence. Recovery reuses
            # the full frame so its three-read allowance stays three OCR calls.
            last_close_count = len(
                self.port._matching_ocr_items(items, r"^閉じる$") if reuse_close_label else self.port._ocr(overlay, r"^閉じる$")
            )
            communication_dialog = self.port._retry_transient_communication_items(items)
            recent_member_frames.append(
                bool(self.port._matching_ocr_items(items, r"^体力$"))
                and bool(self.port._matching_ocr_items(items, r"^総合力$"))
                and not self.port._has_support_bonus_heading(last_text)
                and last_close_count == 0
                and not communication_dialog
            )
            del recent_member_frames[:-2]
            self.port._increment("support_bonus_overlay_reads")
            result = self.port._support_bonus_from_overlay(last_text, close_count=last_close_count)
            return None if communication_dialog else result

        while self.clock.monotonic() < deadline:
            value = read_overlay()
            if value is not None:
                break
            self.port._sleep(0.08)
        self.port._add_timing("support_bonus_open_wait", self.clock.perf_counter() - started)
        # A slow OCR call can consume the original deadline on an animation
        # frame. Observe fresh captures before deciding to click or abort.
        if value is None and (self.port._has_support_bonus_heading(last_text) or any(recent_member_frames)):
            transition_deadline = self.clock.monotonic() + 8.0
            for _ in range(3):
                if self.clock.monotonic() >= transition_deadline:
                    break
                value = read_overlay(reuse_close_label=True)
                self.port._increment("support_bonus_transition_reads")
                if value is not None or recent_member_frames == [True, True]:
                    break
                self.port._sleep(0.12)
        retries = getattr(self.port, "_support_bonus_retried_teams", None)
        if retries is None:
            retries = set()
            self.port._support_bonus_retried_teams = retries
        if value is None and recent_member_frames == [True, True] and target.team_id not in retries:
            # This allowance belongs to the reader/backend, not member state;
            # a whole-provider retry cannot rearm it for the same team.
            retries.add(target.team_id)
            recovery_started = self.clock.perf_counter()
            recovery_deadline = self.clock.monotonic() + 8.0
            self.port._increment("support_bonus_open_retries")
            self.port._click(info_box, settle_seconds=0)
            reads = 0
            while reads < 3 and self.clock.monotonic() < recovery_deadline:
                reads += 1
                value = read_overlay(reuse_close_label=True)
                if value is not None:
                    break
                if reads < 3 and self.clock.monotonic() + 0.08 < recovery_deadline:
                    self.port._sleep(0.08)
            elapsed = self.clock.perf_counter() - recovery_started
            self.port._record_duration_sample("support_bonus_recovery", elapsed)
            self.logger.info(
                json.dumps(
                    {
                        "event": "arena_reader_recovery",
                        "team_id": target.team_id,
                        "action": "support_bonus_reclick",
                        "succeeded": value is not None,
                        "extra_ocr": reads,
                        "extra_clicks": 1,
                        "wall_seconds": round(elapsed, 6),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if value is None:
            bonuses = re.findall(
                r"(?<![0-9])\+([0-9]+(?:\.[0-9]+)?)%",
                last_text,
            )
            cleanup = "not attempted: overlay identity unproven"
            if self.port._has_support_bonus_heading(last_text) and last_close_count <= 1:
                try:
                    self.port._click(info_box, settle_seconds=0)
                    self.port._assert_support_bonus_closed()
                    cleanup = "closed and member page confirmed"
                except ArenaReaderError as error:
                    cleanup = f"member return unconfirmed: {error}"
            raise ArenaReaderError(
                "support_bonus_missing",
                "support overlay did not reach one same-frame semantic decision: "
                f"bonuses={bonuses!r}, close_count={last_close_count}; cleanup={cleanup}",
            )
        # The same info icon is the stable toggle on both own and opponent
        # member pages.  OCR's visible 閉じる label belongs to overlay content
        # and is not a reliable hit target across these two layouts.
        self.port._click(info_box, settle_seconds=0)
        self.port._assert_support_bonus_closed()
        return value

    def _assert_support_bonus_closed(self) -> None:
        deadline = self.clock.monotonic() + 8.0
        for _ in range(3):
            if self.clock.monotonic() >= deadline:
                break
            image = self.port._capture()
            items = self.port._ocr(image, r".+")
            text = "\n".join(_text(item) for item in items)
            communication = self.port._retry_transient_communication_items(items)
            if (
                not communication
                and not self.port._has_support_bonus_heading(text)
                and not self.port._matching_ocr_items(items, r"^閉じる$")
                and self.port._matching_ocr_items(items, r"^体力$")
                and self.port._matching_ocr_items(items, r"^総合力$")
            ):
                return
            self.port._sleep(0.12)
        raise ArenaReaderError(
            "support_bonus_close_unconfirmed",
            "support overlay did not disappear before returning to the member page",
        )

    def read_params(self, target: TeamTarget, stage_number: int, slot: int) -> Sequence[int]:
        del target, stage_number, slot
        failure = "parameter OCR did not run"
        previous_values: tuple[int, ...] | None = None
        for attempt in range(4):
            image = self.port._capture()
            height, width = image.shape[:2]
            observations = self.port._ocr(
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
                    failure = f"parameter labels yielded {sorted(label_items)!r}: {error}"
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
                            and label_box[1] + label_box[3] - 6 <= _box(item)[1] <= label_box[1] + label_box[3] + int(height * 0.04)
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
                            self.port._ocr(image, r"^[0-9]{1,6}$", roi=value_roi),
                            key=lambda item: (_box(item)[0], _box(item)[1]),
                        )
                    enhanced_value: int | None = None
                    enhanced_observations: tuple[str, ...] = ()
                    if not candidates:
                        enhanced_value, enhanced_observations = self.port._read_enhanced_numeric_roi(
                            image,
                            value_roi,
                        )
                        if enhanced_value is not None:
                            values.append(enhanced_value)
                            continue
                    if len(candidates) != 1:
                        numeric_observations = [(_text(item), _box(item)) for item in observations if re.fullmatch(r"[0-9]{1,6}", _text(item))]
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
                self.port._sleep(0.25)
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
            results = self.port._ocr(variant, r"^[0-9]{1,6}$")
            texts = tuple(_text(item) for item in results if re.fullmatch(r"[0-9]{1,6}", _text(item)))
            observed.append(f"{name}:{','.join(texts) or '-'}")
            if len(texts) == 1:
                votes[int(texts[0])] += 1
        ranked = votes.most_common()
        if ranked and ranked[0][1] >= 2 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            return ranked[0][0], tuple(observed)
        return None, tuple(observed)

    def close_member(
        self,
        target: TeamTarget,
        stage_number: int,
        *,
        recovery_image: Any = None,
        recovery_detail: dict[str, Any] | None = None,
    ) -> None:
        self.port._reset_member_card_state()
        if recovery_image is None:
            self.port._back()
        else:
            self.port._back(image=recovery_image)
        expected_totals = (1, 2, 3) if target.is_own_team else (6,)
        try:
            self.port._wait_for_stage_member_list(
                expected_totals,
                timeout_seconds=2.0,
            )
        except ArenaReaderError:
            # A single retry is allowed only when the unchanged 体力/総合力
            # anchors prove that the first back action was dropped. Any other
            # page remains fail-closed rather than issuing another blind back.
            image = self.port._capture()
            if recovery_detail is None:
                if not self.port._ocr(image, r"^体力$") or not self.port._ocr(image, r"^総合力$"):
                    raise
            else:
                image, items = self.port._read_member_recovery_page(image)
                if not self.port._member_recovery_page_matches(items, stage_number, recovery_detail):
                    raise
            self.port._increment("close_member_retries")
            if recovery_detail is None:
                self.port._back()
            else:
                self.port._back(image=image)
            self.port._wait_for_stage_member_list(
                expected_totals,
                timeout_seconds=2.0,
            )
        self.port._member_metric_baseline = None

    def _retain_failed_skill_detail_identity(self, key: tuple[int, int]) -> None:
        """Keep the failed transaction's proven title until its member recovery."""
        member = getattr(self.port, "_member_failure_frames", None)
        proof = getattr(self.port, "_detail_identity_proofs", {}).get(key)
        if member is None or not isinstance(proof, _DetailIdentityProof):
            return
        if not self.port._detail_identity_proof_matches(key, proof.card_id, proof.source_card_box):
            return
        position = member["position"]
        member["skill_detail_recovery"] = {
            "team_id": position.get("team_id"),
            "stage_number": position.get("stage_number"),
            "member_slot": position.get("member_slot"),
            "group_index": key[0],
            "card_slot": key[1],
            "card_id": proof.card_id,
            "title": proof.title,
        }
        diagnostic = getattr(self.port, "_detail_failure_frames", None)
        if diagnostic is not None:
            diagnostic["position"].update(group_index=key[0], card_slot=key[1])
            self.port._note_confirmed_detail_name(proof.card_id)

    def _member_recovery_page_matches(
        self,
        items: Sequence[Any],
        stage_number: int,
        detail: dict[str, Any],
    ) -> bool:
        """Recognize the member or its already confirmed skill detail on this frame."""
        stages = self.port._matching_ocr_items(items, r"^ステージ\s*[123]$")
        if len(stages) != 1 or not self.port._matching_ocr_items(stages, rf"^ステージ\s*{stage_number}$"):
            return False
        if len(self.port._matching_ocr_items(items, r"^総合力$")) != 1:
            return False
        if len(self.port._matching_ocr_items(items, r"^体力$")) == 1:
            return True
        title_matches = 0
        for row_text, _row_box, components in self.port._spatial_ocr_rows(items):
            atom_matches = sum(
                self.port._same_proven_skill_card_title(text, detail["title"], detail["card_id"]) for text, _component_box in components
            )
            title_matches += atom_matches or self.port._same_proven_skill_card_title(
                row_text,
                detail["title"],
                detail["card_id"],
            )
        return title_matches == 1

    def recover_member_preview(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        *,
        error_code: str,
    ) -> None:
        """Use the existing back operation, stopping at this team's preview."""
        started = self.clock.perf_counter()
        before = dict(self.port._runtime_counts)
        succeeded = False
        try:
            # A close may already have returned to the preview. Check before
            # Back so that recovery cannot leave the current opponent.
            image, items = self.port._read_member_recovery_page()
            expected_totals = (1, 2, 3) if target.is_own_team else (6,)
            stages = self.port._matching_ocr_items(items, r"^ステージ\s*[123]$")
            totals = self.port._matching_ocr_items(items, r"^総合力$")
            member = getattr(self.port, "_member_failure_frames", None)
            detail = member.pop("skill_detail_recovery", None) if member is not None else None
            detail_matches_member = bool(
                detail is not None
                and error_code == "skill_card_close_failed"
                and (detail["team_id"], detail["stage_number"], detail["member_slot"]) == (target.team_id, stage_number, member_slot)
            )
            if len(stages) >= 3 and len(totals) in expected_totals:
                self.port._reset_member_card_state()
                self.port._member_metric_baseline = None
            elif (
                (not stages or (len(stages) == 1 and self.port._matching_ocr_items(stages, rf"^ステージ\s*{stage_number}$")))
                and len(self.port._matching_ocr_items(items, r"^体力$")) == 1
                and len(totals) == 1
            ):
                self.port.close_member(target, stage_number)
            elif detail_matches_member and self.port._member_recovery_page_matches(items, stage_number, detail):
                self.port._increment("member_preview_open_detail_recoveries")
                self.port.close_member(
                    target,
                    stage_number,
                    recovery_image=image,
                    recovery_detail=detail,
                )
            else:
                raise ArenaReaderError(
                    "member_preview_recovery_unproven",
                    "current frame proves neither this team's preview nor a member detail; no back sent",
                )
            succeeded = True
        finally:
            elapsed = self.clock.perf_counter() - started
            self.port._record_duration_sample("member_preview_recovery", elapsed)
            self.logger.info(
                json.dumps(
                    {
                        "event": "arena_reader_recovery",
                        "action": "member_preview",
                        "team_id": target.team_id,
                        "stage_number": stage_number,
                        "member_slot": member_slot,
                        "error_code": error_code,
                        "succeeded": succeeded,
                        "wall_seconds": round(elapsed, 6),
                        "extra_counts": {
                            key: value - before.get(key, 0) for key, value in self.port._runtime_counts.items() if value > before.get(key, 0)
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

    def _assert_member_detail(self, timeout_seconds: float = 2.0) -> None:
        deadline = self.clock.monotonic() + timeout_seconds
        while self.clock.monotonic() < deadline:
            image = self.port._capture()
            items = self.port._ocr(image, r".+")
            communication_dialog = self.port._retry_transient_communication_items(items)
            # Keep the original two-query member cadence so P-item sampling
            # does not start earlier during the page transition.
            if not communication_dialog and self.port._matching_ocr_items(items, r"^体力$") and self.port._ocr(image, r"^総合力$"):
                return
            self.port._sleep(0.25)
        raise ArenaReaderError(
            "member_detail_anchor_missing",
            "体力/総合力 anchors did not become visible",
        )
