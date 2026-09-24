"""Capture, recognition and input with cancellation at the original boundaries."""

from __future__ import annotations

from typing import Any, Protocol
from collections.abc import Sequence

from arena_winrate import ArenaReaderError
from arena_winrate.cancellation import ArenaTaskCancelled, ArenaReadSuperseded, cancellation_for
from arena_winrate._reader_visual import _text
from arena_winrate._detail_capture_evidence import _FullFrameOcrEvidence


class IoReaderPort(Protocol):
    """Only the state and callbacks needed by this responsibility."""

    def _add_timing(self, name: str, elapsed: float) -> None: ...

    _cancellation: Any

    def _capture(self) -> Any: ...

    _card_swipe_duration_ms: Any
    _detail_failure_frames: dict[str, Any] | None
    _member_failure_frames: dict[str, Any] | None

    def _check_cancelled(self) -> None: ...
    def _click(self, box: tuple[int, int, int, int], *, settle_seconds: float = ...) -> None: ...
    def _full_frame_ocr_evidence_for(self, image: Any) -> _FullFrameOcrEvidence: ...
    def _increment(self, name: str) -> None: ...

    _member_long_press_seconds: Any

    def _ocr_detail(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = ..., only_rec: bool = ...) -> Any: ...

    _pending_stage_preview: Any
    _progressive_contact_active: Any
    _progressive_read_check: Any

    def _record_detail_action(self, kind: str, box: Sequence[int], *, succeeded: bool) -> None: ...
    def _run_recognition(self, *args, **kwargs): ...
    def _sleep(self, seconds: float) -> None: ...

    _spatial_ocr_text: Any
    context: Any


class IoReader:
    """Use current Maa bindings and the reader's single live state."""

    def __init__(self, port: IoReaderPort, *, action_types: Any, click_action: Any, clock: Any) -> None:
        self.port = port
        self.action_types = action_types
        self.click_action = click_action
        self.clock = clock

    def _check_cancelled(self) -> None:
        # Any intervening reader operation invalidates the one-use preview.
        # member_slots takes its local reference before this boundary check.
        self.port._pending_stage_preview = None
        cancellation = getattr(self.port, "_cancellation", None)
        if cancellation is None:
            cancellation = cancellation_for(getattr(self.port, "context", None))
            self.port._cancellation = cancellation
        cancellation.check()
        progress_check = getattr(self.port, "_progressive_read_check", None)
        if progress_check is not None and not getattr(self.port, "_progressive_contact_active", False):
            progress_check()

    def _sleep(self, seconds: float) -> None:
        self.port._check_cancelled()
        self.clock.sleep(seconds)
        self.port._check_cancelled()

    def _run_recognition(self, *args, **kwargs):
        self.port._check_cancelled()
        result = self.port.context.run_recognition(*args, **kwargs)
        self.port._check_cancelled()
        return result

    def _capture(self) -> Any:
        self.port._check_cancelled()
        started = self.clock.perf_counter()
        try:
            image = self.port.context.tasker.controller.post_screencap().wait().get()
            self.port._check_cancelled()
            for name in ("_detail_failure_frames", "_member_failure_frames"):
                diagnostic = getattr(self.port, name, None)
                if diagnostic is not None and not diagnostic.get("persisted", False):
                    diagnostic["frames"].append((self.clock.perf_counter(), image))
            return image
        finally:
            self.port._increment("screenshots")
            self.port._add_timing("screenshots", self.clock.perf_counter() - started)

    def _recognize(self, entry: str, image: Any) -> list[Any]:
        detail = self.port._run_recognition(entry, image)
        if not detail or not detail.hit:
            return []
        return list(detail.filtered_results or detail.all_results or [])

    def _ocr(
        self,
        image: Any,
        expected: str,
        *,
        roi: tuple[int, int, int, int] | None = None,
        only_rec: bool = False,
    ) -> list[Any]:
        if expected == r".+" and roi is None and not only_rec:
            evidence = self.port._full_frame_ocr_evidence_for(image)
            return list(evidence.filtered_items) if evidence.hit else []
        detail_kwargs: dict[str, Any] = {}
        if roi is not None:
            detail_kwargs["roi"] = roi
        if only_rec:
            detail_kwargs["only_rec"] = True
        detail = self.port._ocr_detail(image, expected, **detail_kwargs)
        if not detail or not detail.hit:
            return []
        return list(detail.filtered_results or detail.all_results or [])

    def _ocr_detail(
        self,
        image: Any,
        expected: str,
        *,
        roi: tuple[int, int, int, int] | None = None,
        only_rec: bool = False,
    ) -> Any:
        """Return one OCR detail while preserving filtered and raw boxes."""

        override: dict[str, Any] = {
            "ArenaReaderOCR": {
                "recognition": "OCR",
                "expected": expected,
                "order_by": "Vertical",
            }
        }
        if roi is not None:
            override["ArenaReaderOCR"]["roi"] = list(roi)
        if only_rec:
            override["ArenaReaderOCR"]["only_rec"] = True
        return self.port._run_recognition(
            "ArenaReaderOCR",
            image,
            pipeline_override=override,
        )

    def _full_ocr_text(
        self,
        image: Any,
        *,
        spatial_reading_order: bool = False,
    ) -> str:
        evidence = self.port._full_frame_ocr_evidence_for(image)
        if not evidence.hit:
            raise ArenaReaderError("ocr_empty", "full-screen OCR returned no text")
        items = list(evidence.all_items)
        if spatial_reading_order:
            return self.port._spatial_ocr_text(items)
        return "\n".join(_text(item) for item in items)

    def _click(
        self,
        box: tuple[int, int, int, int],
        *,
        settle_seconds: float = 0.25,
    ) -> None:
        self.port._check_cancelled()
        started = self.clock.perf_counter()
        dispatch = "context_box"
        if box[2:] == (1, 1):
            dispatch = "controller_direct"
            job = self.port.context.tasker.controller.post_click(box[0], box[1]).wait()
            succeeded = bool(job.succeeded)
        else:
            result = self.port.context.run_action_direct(self.action_types.Click, self.click_action(), box)
            succeeded = bool(result is not None and result.success)
        elapsed = self.clock.perf_counter() - started
        self.port._add_timing("click_actions", elapsed)
        self.port._add_timing(f"{dispatch}_click_actions", elapsed)
        self.port._increment("click_actions")
        self.port._increment(f"{dispatch}_click_actions")
        self.port._record_detail_action("click", box, succeeded=succeeded)
        self.port._check_cancelled()
        if not succeeded:
            raise ArenaReaderError("maa_click_failed", f"Maa could not click {box}")
        if settle_seconds > 0:
            self.port._sleep(settle_seconds)

    def _long_press(self, box: tuple[int, int, int, int]) -> None:
        self.port._check_cancelled()
        x, y, width, height = box
        point = (x + width // 2, y + height // 2)
        started = self.clock.perf_counter()
        controller = self.port.context.tasker.controller
        down_succeeded = False
        up_succeeded = False
        interaction_error: BaseException | None = None
        self.port._progressive_contact_active = True
        try:
            down_succeeded = bool(controller.post_touch_down(*point).wait().succeeded)
            if down_succeeded:
                self.port._sleep(self.port._member_long_press_seconds)
        except (ArenaTaskCancelled, ArenaReadSuperseded, Exception) as error:
            interaction_error = error
        finally:
            try:
                up_succeeded = bool(controller.post_touch_up().wait().succeeded)
            except Exception as error:  # pragma: no cover - native Maa boundary
                if interaction_error is None:
                    interaction_error = error
            finally:
                self.port._progressive_contact_active = False
        elapsed = self.clock.perf_counter() - started
        self.port._add_timing("long_press_actions", elapsed)
        self.port._add_timing("controller_direct_long_press_actions", elapsed)
        self.port._increment("long_press_actions")
        self.port._increment("controller_direct_long_press_actions")
        self.port._record_detail_action("long_press", box, succeeded=down_succeeded and up_succeeded)
        self.port._check_cancelled()
        if interaction_error is not None or not down_succeeded or not up_succeeded:
            detail = f"Maa could not long-press {box} at {point}; down={down_succeeded}, up={up_succeeded}"
            if interaction_error is not None:
                detail += f", error={interaction_error}"
            raise ArenaReaderError("maa_long_press_failed", detail)

    def _short_press(
        self,
        box: tuple[int, int, int, int],
        *,
        duration_ms: int,
    ) -> None:
        """Send one bounded tap-like contact when an instantaneous click drops."""

        self.port._check_cancelled()
        if not 50 <= duration_ms <= 200:
            raise ArenaReaderError(
                "maa_short_press_duration_invalid",
                f"short press duration is outside 50..200 ms: {duration_ms}",
            )
        x, y, width, height = box
        point = (x + width // 2, y + height // 2)
        started = self.clock.perf_counter()
        controller = self.port.context.tasker.controller
        down_succeeded = False
        up_succeeded = False
        interaction_error: BaseException | None = None
        self.port._progressive_contact_active = True
        try:
            down_succeeded = bool(controller.post_touch_down(*point).wait().succeeded)
            if down_succeeded:
                self.port._sleep(duration_ms / 1000.0)
        except (ArenaTaskCancelled, ArenaReadSuperseded, Exception) as error:
            interaction_error = error
        finally:
            try:
                up_succeeded = bool(controller.post_touch_up().wait().succeeded)
            except Exception as error:  # pragma: no cover - native Maa boundary
                if interaction_error is None:
                    interaction_error = error
            finally:
                self.port._progressive_contact_active = False
        elapsed = self.clock.perf_counter() - started
        self.port._add_timing("short_press_actions", elapsed)
        self.port._add_timing("controller_direct_short_press_actions", elapsed)
        self.port._increment("short_press_actions")
        self.port._increment("controller_direct_short_press_actions")
        self.port._record_detail_action("short_press", box, succeeded=down_succeeded and up_succeeded)
        self.port._check_cancelled()
        if interaction_error is not None or not down_succeeded or not up_succeeded:
            detail = f"Maa could not short-press {box} at {point}; down={down_succeeded}, up={up_succeeded}"
            if interaction_error is not None:
                detail += f", error={interaction_error}"
            raise ArenaReaderError(
                "maa_short_press_failed",
                detail,
            )

    def _swipe(
        self,
        *,
        vertical: str,
        distance_ratio: float = 0.0625,
        distance_pixels: int | None = None,
    ) -> None:
        image = self.port._capture()
        height, width = image.shape[:2]
        if not 0.0 < distance_ratio <= 0.20:
            raise ValueError("distance_ratio must be in (0, 0.20]")
        if distance_pixels is None:
            distance_pixels = int(round(height * distance_ratio))
        if not 1 <= distance_pixels <= int(height * 0.20):
            raise ValueError("distance_pixels must be in [1, 20% of frame height]")
        lower_y = int(height * 0.34)
        upper_y = lower_y - distance_pixels
        if vertical == "up":
            begin, end = (
                (int(width * 0.90), lower_y, 1, 1),
                (
                    int(width * 0.90),
                    upper_y,
                    1,
                    1,
                ),
            )
        else:
            begin, end = (
                (int(width * 0.90), upper_y, 1, 1),
                (
                    int(width * 0.90),
                    lower_y,
                    1,
                    1,
                ),
            )
        started = self.clock.perf_counter()
        self.port._check_cancelled()
        job = self.port.context.tasker.controller.post_swipe(
            begin[0],
            begin[1],
            end[0],
            end[1],
            duration=self.port._card_swipe_duration_ms,
        ).wait()
        elapsed = self.clock.perf_counter() - started
        self.port._add_timing("swipe_actions", elapsed)
        self.port._add_timing("controller_direct_swipe_actions", elapsed)
        self.port._increment("swipe_actions")
        self.port._increment("controller_direct_swipe_actions")
        self.port._record_detail_action("swipe", (*begin[:2], *end[:2]), succeeded=bool(job.succeeded))
        self.port._check_cancelled()
        if not job.succeeded:
            raise ArenaReaderError("maa_swipe_failed", f"Maa could not swipe {vertical}")

    def _close_overlay(self) -> None:
        self.port._check_cancelled()
        result = self.port.context.run_task("CloseButton")
        self.port._check_cancelled()
        if not result or not result.status.succeeded:
            image = self.port._capture()
            height, width = image.shape[:2]
            self.port._click((int(width * 0.91), int(height * 0.02), int(width * 0.07), int(height * 0.07)))
