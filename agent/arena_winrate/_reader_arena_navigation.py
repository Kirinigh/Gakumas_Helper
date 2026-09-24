"""Arena navigation component; no imports from the Maa facade."""

from __future__ import annotations

import json
from typing import Any, Protocol
from collections.abc import Sequence

from arena_winrate import (
    TeamTarget,
    ArenaPageState,
    ArenaReaderError,
    classify_arena_page,
    team_stage_total_anchors,
    arena_page_allows_team_entry,
)
from arena_winrate._reader_visual import _box, _text
from arena_winrate._arena_page_flow import RecoveryPageObservation


class ArenaNavigationReaderPort(Protocol):
    """Only the state and callbacks needed by this responsibility."""

    def _arena_page_state(self, image: Any) -> tuple[ArenaPageState, tuple[int, int, int, int, int]]: ...

    _badge_candidate_detail_checks: Any
    _box_center_point: Any

    def _capture(self) -> Any: ...

    _challenge_selection_committed: Any

    def _click(self, box: tuple[int, int, int, int], *, settle_seconds: float = ...) -> None: ...
    def _dismiss_grade_reward_hint(self, image: Any, observations: Sequence[Any]) -> bool: ...
    def _dismiss_skill_card_detail(self, *, image: Any = None) -> None: ...

    _grade: Any
    _grade_reward_hint_dismissed: Any

    def _increment(self, name: str) -> None: ...

    _matching_ocr_items: Any

    def _ocr(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = ..., only_rec: bool = ...) -> list[Any]: ...
    def _ocr_detail(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = ..., only_rec: bool = ...) -> Any: ...

    _pending_stage_preview: Any
    _progressive_abandoned_member: Any

    def _read_grade(self, image: Any, *, observations: Sequence[Any] | None = ...) -> int: ...
    def _recognize(self, entry: str, image: Any) -> list[Any]: ...
    def _recover_arena_main(self, *, maximum_steps: int, error_code: str, require_opponents: bool,
                           initial_observation: RecoveryPageObservation | None = None) -> None: ...
    def _sleep(self, seconds: float) -> None: ...
    def _stage_total_anchors(self, image: Any) -> tuple[tuple[int, int, int, int], ...]: ...
    def _team_stage_total_anchors(self, target: TeamTarget, image: Any) -> tuple[tuple[int, int, int, int], ...]: ...
    def _wait_for_stage_member_list(self, expected_total_counts: tuple[int, ...], timeout_seconds: float = ...) -> None: ...
    def ensure_arena_main(self, *, require_opponents: bool = ...) -> None: ...
    def recover_to_arena_main(self, *, require_opponents: bool = ...,
                             initial_observation: RecoveryPageObservation | None = None) -> None: ...


class ArenaNavigationReader:
    """Arena navigation with live, reader-owned state."""

    def __init__(self, port: ArenaNavigationReaderPort, *, grade_ocr_value: Any, normalize_latin_anchor: Any, logger: Any, clock: Any) -> None:
        self.port = port
        self.grade_ocr_value = grade_ocr_value
        self.normalize_latin_anchor = normalize_latin_anchor
        self.logger = logger
        self.clock = clock

    def ensure_arena_main(self, *, require_opponents: bool = True) -> None:
        state = ArenaPageState.AMBIGUOUS
        counts = (0, 0, 0, 0, 0)
        grade_observations: tuple[Any, ...] = ()
        grade = self.port._grade
        grade_error: ArenaReaderError | None = None
        unavailable_reads = 0
        ambiguous_reads = 0
        for attempt in range(4):
            image = self.port._capture()
            # A fresh own-team read resolves Grade once.  Cached/opponent reads
            # inject that formal decision and must not repeat Grade OCR on each
            # arena-main guard.  The page-state check still runs on every
            # capture.  When Grade is unknown, keep the broad OCR first so the
            # shared OCR entry cannot be polluted by narrower page filters.
            if grade is None:
                grade_observations = tuple(self.port._ocr(image, r".+"))
            state, counts = self.port._arena_page_state(image)
            if arena_page_allows_team_entry(
                state,
                require_opponents=require_opponents,
            ):
                if grade is None:
                    try:
                        grade = self.port._read_grade(
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
                            if error.code == "arena_grade_label_ambiguous":
                                self.port._dismiss_grade_reward_hint(image, grade_observations)
                            self.port._sleep(0.25)
                            continue
                        raise
                    self.logger.info(
                        json.dumps(
                            {
                                "event": "arena_grade_recognized",
                                "grade": grade,
                                "source": "live_screen",
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                break
            if state is ArenaPageState.AMBIGUOUS:
                ambiguous_reads += 1
                if ambiguous_reads < 2:
                    self.port._sleep(0.5)
                    continue
                break
            if state is ArenaPageState.OPPONENTS_UNAVAILABLE:
                unavailable_reads += 1
                if unavailable_reads < 3:
                    self.port._sleep(0.25)
                    continue
            break
        (
            rehearsal_count,
            opponent_count,
            stage_label_count,
            stage_total_count,
            exhausted_notice_count,
        ) = counts
        if state is ArenaPageState.OPPONENTS_UNAVAILABLE and require_opponents and unavailable_reads == 3:
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
        if self.port._grade is not None and self.port._grade != grade:
            raise ArenaReaderError(
                "arena_grade_changed",
                f"contest Grade changed during one read: {self.port._grade} -> {grade}",
            )
        self.port._grade = grade

    def _dismiss_grade_reward_hint(self, image: Any, observations: Sequence[Any]) -> bool:
        """Dismiss one proven reward hint, only after arena-main confirmation.

        The label can be physically covered while the digit remains visible.
        This action authorizes a new observation, never a Grade value.
        """
        if getattr(self.port, "_grade_reward_hint_dismissed", False):
            return False
        if any(self.normalize_latin_anchor(_text(item)) == "GRADE" for item in observations):
            return False
        if image.ndim != 3 or image.shape[2] < 3:
            return False
        height, width = image.shape[:2]
        if height <= 0 or width <= 0 or not 0.54 <= width / height <= 0.58:
            return False
        next_labels = self.port._matching_ocr_items(observations, r"^次まで$")
        if len(next_labels) != 1:
            return False
        label = _box(next_labels[0])
        lx, ly, lw, lh = label
        if not (width * 0.16 <= lx < lx + lw <= width * 0.43 and height * 0.18 <= ly < ly + lh <= height * 0.27):
            return False
        amounts = []
        for item in self.port._matching_ocr_items(observations, r"^\d[\d,]*\s*Pt$"):
            x, y, w, h = _box(item)
            if (
                width * 0.18 <= x < x + w <= width * 0.48
                and ly + lh * 0.5 <= y < y + h <= height * 0.29
                and abs((x + w / 2) - (lx + lw / 2)) <= width * 0.16
            ):
                amounts.append((x, y, w, h))
        if len(amounts) != 1:
            return False
        x, y, w, h = amounts[0]
        left, right = max(0, min(x, lx) - round(width * 0.01)), min(width, max(x + w, lx + lw) + round(width * 0.01))
        top, bottom = max(0, ly - round(height * 0.006)), min(height, y + h + round(height * 0.006))
        import numpy as np

        panel = np.asarray(image)[top:bottom, left:right, :3].astype(np.int16)
        if not panel.size or float(np.mean((panel.min(axis=2) >= 225) & (np.ptp(panel, axis=2) <= 20))) < 0.55:
            return False
        # The unoccupied centre of the top header is outside the emblem,
        # reward/navigation buttons, currency, and the lower opponent rows.
        point = (round(width * 0.5), round(height * 0.08), 1, 1)
        margin = round(width * 0.015)
        if any(
            bx - margin <= point[0] <= bx + bw + margin and by - margin <= point[1] <= by + bh + margin
            for bx, by, bw, bh in (_box(item) for item in observations)
        ):
            return False
        self.port._grade_reward_hint_dismissed = True  # Consume before input; no budget renewal on recovery.
        started = self.clock.perf_counter()
        succeeded = False
        try:
            self.port._click(point, settle_seconds=0)
            self.port._increment("arena_grade_reward_hint_dismissals")
            succeeded = True
        finally:
            self.logger.info(
                json.dumps(
                    {
                        "event": "arena_reader_recovery",
                        "action": "grade_reward_hint_dismiss",
                        "click_point": point,
                        "hint_boxes": [label, amounts[0]],
                        "succeeded": succeeded,
                        "grade_confirmed": False,
                        "extra_ocr": 0,
                        "extra_clicks": 1,
                        "wall_seconds": round(self.clock.perf_counter() - started, 6),
                    },
                    ensure_ascii=False,
                )
            )
        return True

    def enter_team(self, target: TeamTarget) -> None:
        self.port._badge_candidate_detail_checks.clear()
        image = self.port._capture()
        if target.is_own_team:
            matches = self.port._ocr(image, r"^リハーサル$")
            if len(matches) != 1:
                raise ArenaReaderError("rehearsal_anchor_ambiguous", f"found {len(matches)} rehearsal anchors")
            self.port._click(self.port._box_center_point(_box(matches[0])))
        else:
            opponents = self.port._recognize("ArenaReaderOpponentCards", image)
            if target.opponent_position is None or len(opponents) != 3:
                raise ArenaReaderError("opponent_count_mismatch", "three ordered opponent cards are required")
            ordered = sorted(opponents, key=lambda item: (_box(item)[1], _box(item)[0]))
            self.port._click(_box(ordered[target.opponent_position]))
        self.port._wait_for_stage_member_list((3,) if target.is_own_team else (6,))

    def select_opponent_for_challenge(self, position: int) -> dict[str, Any]:
        """Perform one lightweight same-session guard, then open the target.

        The configured selection rule has already chosen this opponent. This
        guard does not require its lineup or a simulated rate: it proves the
        arena main page and three ordered cards still exist, then confirms the
        selected preview exposes the existing challenge-start control.
        """

        if self.port._challenge_selection_committed:
            raise ArenaReaderError(
                "arena_challenge_already_committed",
                "this reader session already sent an opponent-selection click",
            )
        if position not in (0, 1, 2):
            raise ArenaReaderError(
                "arena_challenge_position_invalid",
                f"selected opponent position is outside 0..2: {position}",
            )
        self.port.ensure_arena_main(require_opponents=True)
        image = self.port._capture()
        state, counts = self.port._arena_page_state(image)
        if state is not ArenaPageState.READY or counts[1] != 3:
            raise ArenaReaderError(
                "arena_challenge_preclick_guard_failed",
                f"arena main is not stable immediately before selection: state={state}, counts={counts}",
            )
        opponents = self.port._recognize("ArenaReaderOpponentCards", image)
        if len(opponents) != 3:
            raise ArenaReaderError(
                "arena_challenge_preclick_guard_failed",
                f"expected three opponent cards immediately before selection, got {len(opponents)}",
            )
        ordered = sorted(opponents, key=lambda item: (_box(item)[1], _box(item)[0]))
        target_box = _box(ordered[position])
        self.port._click(target_box)
        self.port._wait_for_stage_member_list((6,))
        deadline = self.clock.monotonic() + 3.0
        while self.clock.monotonic() < deadline:
            preview = self.port._capture()
            starts = self.port._recognize("ChallengeStart", preview)
            if len(starts) == 1:
                self.port._challenge_selection_committed = True
                self.port._increment("arena_challenge_selection_clicks")
                return {
                    "position": position,
                    "opponent_count": 3,
                    "target_box": list(target_box),
                    "challenge_start_count": 1,
                    "full_lineup_reread": False,
                }
            if len(starts) > 1:
                break
            self.port._sleep(0.2)
        raise ArenaReaderError(
            "arena_challenge_start_ambiguous",
            "selected opponent preview did not expose one challenge-start control",
        )

    def select_stage(self, target: TeamTarget, stage_number: int) -> None:
        self.port._pending_stage_preview = None
        if stage_number not in (1, 2, 3):
            raise ArenaReaderError("stage_number_invalid", f"stage number is outside 1..3: {stage_number}")
        image = self.port._capture()
        totals = self.port._team_stage_total_anchors(target, image)
        # No stage-selection input exists on this three-stage preview. The
        # immediately following slot read consumes this fresh observation once.
        self.port._pending_stage_preview = (target, stage_number, image, totals)

    def leave_team(self, target: TeamTarget) -> None:
        self.port._recover_arena_main(
            maximum_steps=3,
            error_code="team_return_failed",
            require_opponents=not target.is_own_team,
        )

    def finish_progressive_read(self) -> None:
        """Dismiss an interrupted member's tooltip before the existing return loop."""
        position = getattr(self.port, "_progressive_abandoned_member", None)
        self.port._progressive_abandoned_member = None
        initial_observation = None
        if position is not None:
            image = self.port._capture()
            items = self.port._ocr(image, r".+")
            if self.port._matching_ocr_items(items, r"サポートボーナス") and self.port._matching_ocr_items(items, r"^閉じる$"):
                height, width = image.shape[:2]
                self.port._click((int(width * 0.955), int(height * 0.022), 1, 1), settle_seconds=0)
            elif (
                len(self.port._matching_ocr_items(items, r"^体力$")) == 1
                and len(self.port._matching_ocr_items(items, r"^総合力$")) == 1
                and len(self.port._matching_ocr_items(items, r"^ステージ\s*[123]$")) <= 1
            ):
                # Both detail types use the same inert backdrop. It is harmless
                # when the just-issued open/close already left the member bare.
                self.port._dismiss_skill_card_detail(image=image)
            else:
                initial_observation = RecoveryPageObservation(image, tuple(items))
        if initial_observation is None:
            self.port.recover_to_arena_main(require_opponents=True)
        else:
            self.port.recover_to_arena_main(require_opponents=True, initial_observation=initial_observation)

    def recover_to_arena_main(self, *, require_opponents: bool = True,
                             initial_observation: RecoveryPageObservation | None = None) -> None:
        self.port._recover_arena_main(
            maximum_steps=6,
            error_code="arena_recovery_failed",
            require_opponents=require_opponents,
            initial_observation=initial_observation,
        )

    def _assert_stage_navigation(self) -> None:
        results = self.port._ocr(self.port._capture(), r"^ステージ\s*[123]$")
        if len(results) < 3:
            raise ArenaReaderError("stage_navigation_missing", f"found only {len(results)} stage anchors")

    def _stage_total_anchors(self, image: Any) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            sorted(
                (_box(item) for item in self.port._ocr(image, r"^総合力$")),
                key=lambda box: (box[1], box[0]),
            )
        )

    def _team_stage_total_anchors(
        self,
        target: TeamTarget,
        image: Any,
    ) -> tuple[tuple[int, int, int, int], ...]:
        totals = self.port._stage_total_anchors(image)
        try:
            return team_stage_total_anchors(totals, own_team=target.is_own_team)
        except ValueError as error:
            raise ArenaReaderError(
                "stage_total_count_mismatch",
                f"{target.team_id} stage totals are invalid: {error}",
            ) from error

    def _arena_main_visible(self) -> bool:
        image = self.port._capture()
        state, _ = self.port._arena_page_state(image)
        return state is ArenaPageState.READY

    def _arena_page_state(
        self,
        image: Any,
        *,
        observations: Sequence[Any] | None = None,
    ) -> tuple[ArenaPageState, tuple[int, int, int, int, int]]:
        items = self.port._ocr(image, r".+") if observations is None else observations
        rehearsal_count = len(self.port._matching_ocr_items(items, r"^リハーサル$"))
        opponent_count = len(self.port._recognize("ArenaReaderOpponentCards", image))
        stage_label_count = len(self.port._matching_ocr_items(items, r"^ステージ\s*[123]$"))
        stage_total_count = len(self.port._matching_ocr_items(items, r"^総合力$"))
        exhausted_notice_count = len(self.port._matching_ocr_items(items, r"^本日の挑戦権を消費しました$"))
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
        observations = tuple(self.port._ocr(image, r".+") if observations is None else observations)
        labels = tuple(item for item in observations if self.normalize_latin_anchor(_text(item)) == "GRADE")
        if len(labels) != 1:
            raise ArenaReaderError(
                "arena_grade_label_ambiguous",
                f"found {len(labels)} GRADE labels on the arena main screen",
            )
        label_x, label_y, label_width, label_height = _box(labels[0])
        # The Grade digit and label are part of one emblem: the digit is
        # centred above the GRADE word.  Derive a tight, scale-aware region
        # from the label itself so stage numbers and the inter-stage ornament
        # below the label can never become Grade evidence.
        left = max(0, label_x)
        right = min(width, label_x + label_width)
        vertical_span = max(label_width * 2, label_height * 4)
        top = max(0, label_y - vertical_span)
        bottom = min(height, label_y)
        minimum_digit_height = max(1, round(label_width * 0.35))
        emblem_roi = (
            left,
            top,
            max(0, right - left),
            max(0, bottom - top),
        )
        matches = tuple(
            (item, grade_value)
            for item in observations
            if (grade_value := self.grade_ocr_value(_text(item))) is not None
            and _box(item)[3] >= minimum_digit_height
            and (left <= _box(item)[0] + _box(item)[2] / 2 <= right and top <= _box(item)[1] + _box(item)[3] / 2 <= bottom)
        )
        targeted_views: list[dict[str, Any]] = []
        if not matches:
            # Full-screen OCR can consistently omit the large stylised Grade
            # digit while still resolving the surrounding small menu text. A
            # crop that includes the whole emblem is not a valid fallback:
            # its ornament and digit can merge into one CJK-shaped OCR line.
            # Isolate only the fixed digit lane above the unique GRADE label,
            # then require all non-empty detector/recognizer views to agree.
            digit_left = max(0, round(label_x + label_width * 0.08))
            digit_right = min(width, round(label_x + label_width * 0.92))
            digit_top = max(0, round(label_y - label_width * 1.55))
            digit_bottom = min(height, round(label_y - label_width * 0.60))
            digit_roi = (
                digit_left,
                digit_top,
                max(0, digit_right - digit_left),
                max(0, digit_bottom - digit_top),
            )
            accepted_values: list[int] = []
            observed_values: set[int] = set()
            for only_rec in (False, True):
                detail = self.port._ocr_detail(
                    image,
                    r"^[1-7]$",
                    roi=digit_roi,
                    only_rec=only_rec,
                )
                raw_items = tuple(detail.all_results or () if detail is not None else ())
                filtered_items = tuple(detail.filtered_results or () if detail is not None and detail.hit else ())
                raw_matches = tuple(
                    (item, grade_value)
                    for item in (*raw_items, *filtered_items)
                    if (grade_value := self.grade_ocr_value(_text(item))) is not None and _box(item)[3] >= minimum_digit_height
                )
                filtered_matches = tuple(
                    (item, grade_value)
                    for item in filtered_items
                    if (grade_value := self.grade_ocr_value(_text(item))) is not None and _box(item)[3] >= minimum_digit_height
                )
                raw_values = tuple(dict.fromkeys(value for _, value in raw_matches))
                filtered_values = tuple(dict.fromkeys(value for _, value in filtered_matches))
                targeted_views.append(
                    {
                        "only_rec": only_rec,
                        "hit": bool(detail is not None and detail.hit),
                        "raw": tuple((_text(item), _box(item)) for item in raw_items),
                        "filtered": tuple((_text(item), _box(item)) for item in filtered_items),
                        "raw_values": raw_values,
                        "accepted_values": filtered_values,
                    }
                )
                observed_values.update(raw_values)
                if len(filtered_values) == 1:
                    accepted_values.append(filtered_values[0])
            if accepted_values and len(set(accepted_values)) == 1 and observed_values == {accepted_values[0]}:
                return accepted_values[0]
        if len(matches) != 1:
            nearby = tuple(
                (_text(item), _box(item))
                for item in observations
                if (left <= _box(item)[0] + _box(item)[2] / 2 <= right and top <= _box(item)[1] + _box(item)[3] / 2 <= label_y + label_height)
            )
            raise ArenaReaderError(
                "arena_grade_ambiguous",
                f"found {len(matches)} geometrically valid Grade digits above the "
                f"GRADE anchor in {emblem_roi}; targeted_ocr="
                f"{tuple(targeted_views)!r}; "
                f"nearby_ocr={nearby!r}",
            )
        return matches[0][1]
