"""Bounded page navigation through explicit capture, recognition and input capabilities.

The reader owns the one PageRecoveryState. Each flow invocation keeps only its
original local observation counters and deadlines; no card/member state enters
this component, and input continues through the reader's cancellation guards.
"""

from __future__ import annotations

import json
from typing import Any, Protocol
from dataclasses import dataclass
from collections.abc import Sequence

from .reader import ArenaPageState, ArenaReaderError, arena_page_allows_team_entry
from .recovery import error_retry_box
from ._reader_visual import _box


@dataclass
class PageRecoveryState:
    """Input opportunities shared by member cleanup and outer page recovery."""

    communication_retry_attempted: bool = False
    contest_details_close_attempts: int = 0
    last_retry_dialog_seen: bool = False


@dataclass(frozen=True)
class RecoveryPageObservation:
    """One current frame and its OCR, passed only before any intervening input."""

    image: Any
    items: Sequence[Any]


class PageFlowClock(Protocol):
    def monotonic(self) -> float: ...
    def perf_counter(self) -> float: ...


class PageFlowLogger(Protocol):
    def info(self, message: str) -> Any: ...


class ArenaPagePort(Protocol):
    @property
    def state(self) -> PageRecoveryState: ...

    def capture(self) -> Any: ...
    def ocr(self, image: Any, expected: str) -> Any: ...
    def recognize(self, entry: str, image: Any) -> Any: ...
    def run_recognition(self, *args: Any, **kwargs: Any) -> Any: ...
    def click(self, box: Any, **kwargs: Any) -> Any: ...
    def sleep(self, seconds: float) -> Any: ...
    def check_cancelled(self) -> Any: ...
    def arena_page_state(self, image: Any, *, observations: Sequence[Any] | None = None) -> Any: ...
    def matching_ocr_items(self, items: Sequence[Any], expected: str) -> Any: ...
    def box_center_point(self, box: Any) -> Any: ...
    def increment(self, name: str) -> Any: ...
    def add_timing(self, name: str, elapsed: float) -> Any: ...
    def record_duration_sample(self, name: str, elapsed: float) -> Any: ...
    def back(self, *, image: Any = None) -> Any: ...
    def dismiss_known_blocking_overlay(self, image: Any, *, items: Sequence[Any] | None = None) -> Any: ...
    def dismiss_contest_details_items(self, image: Any, items: Sequence[Any]) -> Any: ...
    def retry_transient_communication_items(self, items: Sequence[Any]) -> Any: ...


class ArenaPageFlow:
    def __init__(self, port: ArenaPagePort, clock: PageFlowClock, logger: PageFlowLogger):
        self.port = port
        self.clock = clock
        self.logger = logger

    def read_member_recovery_page(self, image: Any = None) -> tuple[Any, Sequence[Any]]:
        deadline = self.clock.monotonic() + 2.0
        for observation in range(3):
            if image is None:
                image = self.port.capture()
            items = self.port.ocr(image, r".+")
            communication = self.port.retry_transient_communication_items(items)
            contest_details = not communication and self.port.dismiss_contest_details_items(image, items)
            if not communication and not contest_details:
                return image, items
            if observation == 2 or self.clock.monotonic() >= deadline:
                break
            self.port.sleep(0.1)
            image = None
        if contest_details:
            raise ArenaReaderError(
                "arena_contest_details_close_failed",
                "contest details did not return within the member recovery observations; no back sent",
            )
        raise ArenaReaderError(
            "arena_communication_retry_exhausted",
            "member recovery still sees a communication dialog after its existing Retry opportunity",
        )

    def recover_arena_main(
        self,
        *,
        maximum_steps: int,
        error_code: str,
        require_opponents: bool,
        initial_observation: RecoveryPageObservation | None = None,
    ) -> None:
        unavailable_reads = 0
        steps = 0
        loading_reads = 0
        loading_deadline: float | None = None
        while steps < maximum_steps:
            observation, initial_observation = initial_observation, None
            if observation is None:
                image = self.port.capture()
            else:
                # Replacing capture must retain its cancellation boundary. The
                # handoff is consumed before navigation, waits, or another loop.
                self.port.check_cancelled()
                image = observation.image
            self.port.state.last_retry_dialog_seen = False
            if observation is None:
                state, _ = self.port.arena_page_state(image)
            else:
                state, _ = self.port.arena_page_state(image, observations=observation.items)
                self.port.check_cancelled()
            if arena_page_allows_team_entry(
                state,
                require_opponents=require_opponents,
            ):
                return
            items = self.port.ocr(image, r".+") if observation is None else observation.items
            if self.port.matching_ocr_items(items, r"(?i)^\s*NOW\s*LOADING[.\s…]*$"):
                loading_reads += 1
                self.port.increment("arena_recovery_loading_frames")
                if loading_deadline is None:
                    loading_deadline = self.clock.monotonic() + 5.0
                if loading_reads >= 12 or self.clock.monotonic() >= loading_deadline:
                    raise ArenaReaderError(
                        "arena_recovery_loading_timeout",
                        "arena recovery stayed in NOW LOADING within its 12-frame/5-second budget; no back sent",
                    )
                started = self.clock.perf_counter()
                try:
                    self.port.sleep(0.25)
                finally:
                    self.port.add_timing("arena_recovery_loading_wait", self.clock.perf_counter() - started)
                continue
            steps += 1
            if state is ArenaPageState.OPPONENTS_UNAVAILABLE:
                unavailable_reads += 1
                if unavailable_reads >= 3:
                    raise ArenaReaderError(
                        "arena_opponents_unavailable",
                        "opponent cards stayed absent on the arena main page; recovery will not navigate away",
                    )
                self.port.sleep(0.25)
                continue
            unavailable_reads = 0
            if self.port.dismiss_known_blocking_overlay(image, items=items):
                self.port.sleep(0.25)
                continue
            # Navigation must use the frame whose page/overlays were checked.
            try:
                self.port.back(image=image)
            except ArenaReaderError as error:
                if error.code != "maa_back_anchor_ambiguous" or "found 0 " not in error.detail:
                    raise
                if steps >= maximum_steps:
                    raise
                # A returned click/recognition can precede the next page's
                # render. Spend the existing recovery steps on new frames.
                self.port.sleep(0.1)
        if self.port.state.last_retry_dialog_seen:
            raise ArenaReaderError(
                "arena_communication_retry_exhausted",
                "the error dialog remained after its one retry and bounded recovery wait",
            )
        required_state = "with three opponents" if require_opponents else "with the rehearsal anchor"
        raise ArenaReaderError(
            error_code,
            f"back navigation did not reach an arena main page {required_state}",
        )

    def dismiss_contest_details_items(self, image: Any, items: Sequence[Any]) -> bool:
        """Close a proven contest overlay using recovery's existing OCR frame."""
        self.port.check_cancelled()
        titles = self.port.matching_ocr_items(items, r"^コンテスト\s*詳細$")
        if not titles:
            return False
        height, width = image.shape[:2]
        closes = self.port.matching_ocr_items(items, r"^閉じる$")
        if (
            len(titles) != 1 or len(closes) != 1
            or not 0 <= _box(titles[0])[1] < height * 0.25
            or not height * 0.75 <= _box(closes[0])[1] < height
            or not 0 <= _box(closes[0])[0] < width
        ):
            raise ArenaReaderError(
                "arena_contest_details_close_ambiguous",
                "contest details is open but its title/close anchors are not unique; no back sent",
            )
        attempts = self.port.state.contest_details_close_attempts
        if attempts >= 2:
            raise ArenaReaderError(
                "arena_contest_details_close_failed",
                "contest details remains after two close attempts shared by member and arena recovery; no back sent",
            )
        # Consume before input. Member cleanup and outer recovery must not renew it.
        self.port.state.contest_details_close_attempts = attempts + 1
        self.port.increment("arena_contest_details_close_attempts")
        started = self.clock.perf_counter()
        sent = False
        try:
            self.port.click(_box(closes[0]), settle_seconds=0)
            sent = True
        finally:
            elapsed = self.clock.perf_counter() - started
            self.port.record_duration_sample("arena_contest_details_close", elapsed)
            self.logger.info(json.dumps({
                "event": "arena_reader_recovery", "action": "contest_details_close",
                "attempt": attempts + 1, "click_sent": sent,
                "extra_ocr": 0, "extra_clicks": 1,
                "wall_seconds": round(elapsed, 6),
            }, ensure_ascii=False, sort_keys=True))
        return True

    def dismiss_known_blocking_overlay(self, image: Any, *, items: Sequence[Any] | None = None) -> bool:
        """Dismiss only overlays proven by independent page-specific anchors."""

        # Recovery gets one full-frame problem check; reuse the same OCR boxes.
        if items is None:
            items = self.port.ocr(image, r".+")
        if self.port.retry_transient_communication_items(items):
            return True
        if self.port.dismiss_contest_details_items(image, items):
            return True
        menu_profile = self.port.matching_ocr_items(items, r"^プロフィール$")
        menu_settings = self.port.matching_ocr_items(items, r"^設定$")
        if len(menu_profile) == 1 and len(menu_settings) == 1:
            height, width = image.shape[:2]
            # The upper-centre backdrop is outside the menu's functional grid
            # at every supported aspect ratio.  A click there dismisses the
            # proven menu without relying on a scale-sensitive close template.
            self.port.click((int(width * 0.50), int(height * 0.16), 1, 1))
            self.port.increment("arena_menu_overlays_dismissed")
            return True

        error_titles = self.port.matching_ocr_items(items, r"^通信エラ(?:ー)?$")
        failure_details = self.port.matching_ocr_items(items, r"^アセット取得に失敗$")
        if len(error_titles) != 1 or len(failure_details) != 1:
            return False
        close_buttons = self.port.recognize("CloseRoundButton", image)
        if len(close_buttons) != 1:
            raise ArenaReaderError(
                "arena_blocking_overlay_close_ambiguous",
                "a communication-error overlay is proven but its close button is not unique",
            )
        self.port.click(_box(close_buttons[0]))
        self.port.increment("arena_blocking_overlays_dismissed")
        return True

    def retry_transient_communication_items(self, items: Sequence[Any]) -> bool:
        """Return whether a proven dialog occupies this already-read frame."""
        self.port.check_cancelled()
        retry_box = error_retry_box(items)
        self.port.state.last_retry_dialog_seen = retry_box is not None
        if retry_box is None:
            return False
        if self.port.state.communication_retry_attempted:
            # The original deadline may still be observing the sent Retry's
            # transition. Do not resend and do not extend that deadline.
            return True
        self.port.state.communication_retry_attempted = True
        started = self.clock.perf_counter()
        succeeded = False
        try:
            self.port.click(retry_box, settle_seconds=0)
            self.port.increment("arena_communication_retries")
            succeeded = True
        finally:
            self.logger.info(json.dumps({
                "event": "arena_reader_recovery", "action": "communication_retry",
                "succeeded": succeeded, "extra_ocr": 0, "extra_clicks": 1,
                "wall_seconds": round(self.clock.perf_counter() - started, 6),
            }, ensure_ascii=False, sort_keys=True))
        return True

    def back(self, *, image: Any = None) -> None:
        if image is None:
            image = self.port.capture()
        height, width = image.shape[:2]
        detail = self.port.run_recognition(
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
        self.port.click(self.port.box_center_point(_box(results[0])))

    def wait_for_stage_member_list(
        self,
        expected_total_counts: tuple[int, ...],
        timeout_seconds: float = 3.0,
    ) -> None:
        deadline = self.clock.monotonic() + timeout_seconds
        last_stage_count = 0
        last_total_count = 0
        while self.clock.monotonic() < deadline:
            image = self.port.capture()
            items = self.port.ocr(image, r".+")
            last_stage_count = len(self.port.matching_ocr_items(items, r"^ステージ\s*[123]$"))
            last_total_count = len(self.port.matching_ocr_items(items, r"^総合力$"))
            communication_dialog = self.port.retry_transient_communication_items(items)
            if not communication_dialog and last_stage_count >= 3 and last_total_count in expected_total_counts:
                return
            self.port.sleep(0.25)
        raise ArenaReaderError(
            "stage_member_list_timeout",
            f"stage anchors={last_stage_count}, total anchors={last_total_count}, "
            f"expected totals={expected_total_counts!r}",
        )
