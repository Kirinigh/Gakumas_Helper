"""P-item observations and return guards using the current reader-owned evidence.

The original capture/OCR order, frame windows and deadlines remain here.
Pixel/geometry decisions continue through the existing pure source-evidence
functions; this component neither owns an input budget nor copies frame state.
"""

from __future__ import annotations

from typing import Any, Protocol
from collections.abc import Sequence

from p_item_recognition import PItemReferenceError
from card_selection.model import frame_identifier

from ._reader_errors import ArenaReaderError
from ._reader_visual import _text
from ._reader_evidence import _PItemPanelRow, _PItemOcrObservation
from ._p_item_source_evidence import p_item_source_frame_evidence, p_item_source_generation_evidence


class PItemObservationClock(Protocol):
    def monotonic(self) -> float: ...
    def perf_counter(self) -> float: ...


class PItemObservationPort(Protocol):
    """Capture/OCR capabilities and the live P-item evidence destinations only."""

    p_item_reader: Any
    _p_item_generation_evidence: dict[str, Any]
    _detail_failure_frames: dict[str, Any] | None
    _last_p_item_restore_anchors: dict[str, Any] | None

    def _capture(self) -> Any: ...
    def _sleep(self, seconds: float) -> None: ...
    def _increment(self, name: str) -> None: ...
    def _add_timing(self, name: str, elapsed: float) -> None: ...
    def _ocr(self, image: Any, expected: str) -> Sequence[Any]: ...
    def _run_recognition(self, entry: str, image: Any, **kwargs: Any) -> Any: ...
    def _observe_recognition_probe(self, image: Any, items: Any, *, reco_id: Any = None) -> None: ...
    def _p_item_detail_panel_rows(
        self,
        image: Any,
        items: Sequence[Any],
        *,
        background_only: bool = False,
    ) -> tuple[_PItemPanelRow, ...]: ...
    def _cached_full_frame_ocr_evidence(self, image: Any) -> Any: ...
    def _p_item_source_frame_matches(
        self,
        source_images: Sequence[Any],
        image: Any,
        boxes: Sequence[tuple[int, int, int, int]],
        member_anchors_visible: bool | None = None,
    ) -> tuple[bool, tuple[float, ...]]: ...


class PItemPageObservation:
    def __init__(self, port: PItemObservationPort, *, clock: PItemObservationClock) -> None:
        self.port = port
        self.clock = clock

    def capture_stable_generation(
        self,
        boxes: Sequence[tuple[int, int, int, int]],
        *,
        initial_image: Any | None = None,
        timeout_seconds: float = 1.5,
        interval_seconds: float = 0.08,
    ) -> tuple[tuple[Any, Any, Any], tuple[float, ...], int]:
        """Wait for three consecutive frames from one P-item render generation."""

        if self.port.p_item_reader is None:
            raise ArenaReaderError(
                "p_item_reader_missing",
                "P-item content stability requires an initialized reader",
            )
        threshold = float(self.port.p_item_reader.content_generation_max_mean_abs_error)
        frames: list[Any] = []
        rejected_windows = 0
        last_errors: tuple[float, ...] = ()
        deadline = self.clock.monotonic() + timeout_seconds
        started = self.clock.perf_counter()
        pending = initial_image
        while self.clock.monotonic() < deadline:
            image = pending if pending is not None else self.port._capture()
            pending = None
            frames.append(image)
            if len(frames) > 3:
                frames.pop(0)
            if len(frames) == 3:
                try:
                    last_errors = tuple(
                        float(value)
                        for value in self.port.p_item_reader.content_generation_errors(
                            frames,
                            boxes,
                        )
                    )
                except (PItemReferenceError, TypeError, ValueError) as error:
                    raise ArenaReaderError(
                        "p_item_content_generation_measurement_failed",
                        "P-item content-generation measurement failed",
                    ) from error
                if len(last_errors) != len(boxes):
                    raise ArenaReaderError(
                        "p_item_content_generation_count_mismatch",
                        f"P-item content-generation measurement returned {len(last_errors)} slots for {len(boxes)} boxes",
                    )
                if all(value <= threshold for value in last_errors):
                    self.port._add_timing(
                        "p_item_content_stable_wait",
                        self.clock.perf_counter() - started,
                    )
                    return (
                        (frames[0], frames[1], frames[2]),
                        last_errors,
                        rejected_windows,
                    )
                rejected_windows += 1
                self.port._increment("p_item_content_generation_resets")
            if interval_seconds > 0:
                self.port._sleep(interval_seconds)
        self.port._add_timing(
            "p_item_content_stable_wait",
            self.clock.perf_counter() - started,
        )
        self.port._p_item_generation_evidence = {
            "frame_ids": [frame_identifier(image) for image in frames],
            "content_stable": False,
            "identity_resolved": False,
            "content_generation_errors": [round(value, 6) for value in last_errors],
            "content_generation_threshold": threshold,
            "content_generation_rejected_windows": rejected_windows,
        }
        raise ArenaReaderError(
            "p_item_content_generation_unstable",
            "P-item slots did not yield three consecutive stable frames within "
            f"{timeout_seconds:.2f}s; threshold={threshold:.6f}; "
            f"last_errors={[round(value, 6) for value in last_errors]!r}",
        )

    def ocr_observation(
        self,
        image: Any,
    ) -> _PItemOcrObservation:
        """OCR one frame once and retain full text, title geometry and anchors."""

        detail = self.port._run_recognition(
            "ArenaReaderOCR",
            image,
            pipeline_override={
                "ArenaReaderOCR": {
                    "recognition": "OCR",
                    "expected": r".+",
                    "order_by": "Vertical",
                }
            },
        )
        if not detail or not detail.hit:
            raise ArenaReaderError("ocr_empty", "full-screen OCR returned no text")
        items = list(detail.all_results or detail.filtered_results or [])
        self.port._observe_recognition_probe(image, items, reco_id=getattr(detail, "reco_id", None))
        full_text = "\n".join(_text(item) for item in items)
        panel_rows = self.port._p_item_detail_panel_rows(image, items)
        background_rows = self.port._p_item_detail_panel_rows(
            image,
            items,
            background_only=True,
        )
        title_text = "\n".join(text for text, _, _ in panel_rows)
        exact_ocr_rows = {_text(item).strip() for item in items}
        member_anchors_visible = "体力" in exact_ocr_rows and "総合力" in exact_ocr_rows
        return _PItemOcrObservation(
            full_text,
            title_text,
            member_anchors_visible,
            panel_rows,
            background_rows,
            tuple(items),
        )

    def assert_source_restored(
        self,
        source_images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
        timeout_seconds: float = 8.0,
    ) -> None:
        """Prove a P-item overlay closed before another slot can be clicked."""

        if not boxes:
            raise ArenaReaderError(
                "p_item_source_restore_evidence_missing",
                "P-item detail recovery requires the accepted source-row boxes",
            )
        threshold = self.port.p_item_reader.content_generation_max_mean_abs_error
        # Four fresh observations, within a fixed ceiling, allow a transition
        # followed by TWO matching frames even when one OCR takes > 2 seconds.
        # Success still exits immediately on the original second matching frame.
        deadline = self.clock.monotonic() + timeout_seconds
        consecutive = 0
        last_errors: tuple[float, ...] = ()
        restore_observations: list[dict[str, Any]] = []
        diagnostic = getattr(self.port, "_detail_failure_frames", None)
        if diagnostic is not None:
            diagnostic["p_item_restore"] = restore_observations
            diagnostic["probe_phase"] = "return"
        for _ in range(4):
            if self.clock.monotonic() >= deadline:
                break
            image = self.port._capture()
            self.port._last_p_item_restore_anchors = None
            matches_source, last_errors = self.port._p_item_source_frame_matches(
                source_images,
                image,
                boxes,
            )
            restore_observations.append(
                {
                    "matches_source": matches_source,
                    "anchors": getattr(self.port, "_last_p_item_restore_anchors", None),
                    "errors": tuple(round(value, 6) for value in last_errors),
                    "consecutive_before": consecutive,
                }
            )
            if matches_source:
                consecutive += 1
                if consecutive >= 2:
                    return
            else:
                consecutive = 0
            self.port._sleep(0.12)
        raise ArenaReaderError(
            "p_item_source_restore_unproven",
            "member anchors and two stable source-row generations did not return: "
            f"errors={tuple(round(value, 6) for value in last_errors)!r}; "
            f"threshold={threshold}; observations={restore_observations!r}",
        )

    def assert_source_generation_stable(
        self,
        source_images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
    ) -> None:
        if len(source_images) < 2:
            return
        threshold = self.port.p_item_reader.content_generation_max_mean_abs_error
        evidence = p_item_source_generation_evidence(
            source_images,
            boxes,
            threshold=threshold,
        )
        if not evidence.matches:
            raise ArenaReaderError(
                "p_item_source_generation_unstable",
                "P-item frozen source frames changed in an item or detail-panel "
                f"guard region: errors={tuple(round(value, 6) for value in evidence.errors)!r}; "
                f"threshold={threshold}",
            )

    def source_frame_matches(
        self,
        source_images: Sequence[Any],
        image: Any,
        boxes: Sequence[tuple[int, int, int, int]],
        member_anchors_visible: bool | None = None,
    ) -> tuple[bool, tuple[float, ...]]:
        """Prove one frame is the frozen member page, not a visible detail."""

        if not boxes:
            raise ArenaReaderError(
                "p_item_source_restore_evidence_missing",
                "P-item source-page evidence requires the accepted source-row boxes",
            )
        if member_anchors_visible is None:
            # Both labels belong to this same frozen frame. Reuse its full
            # OCR result instead of recognizing the whole image twice.
            exact_rows = {_text(item) for item in self.port._ocr(image, r".+")}
            anchors_visible = "体力" in exact_rows and "総合力" in exact_rows
            evidence = self.port._cached_full_frame_ocr_evidence(image)
            self.port._last_p_item_restore_anchors = {
                "visible": anchors_visible,
                "filtered_texts": sorted(exact_rows),
                "ocr_hit": evidence.hit if evidence is not None else None,
                "reco_id": evidence.reco_id if evidence is not None else None,
                "all_texts": [_text(item) for item in evidence.all_items] if evidence is not None else None,
            }
        else:
            anchors_visible = member_anchors_visible
        evidence = p_item_source_frame_evidence(
            source_images,
            image,
            boxes,
            member_anchors_visible=anchors_visible,
            threshold=self.port.p_item_reader.content_generation_max_mean_abs_error,
        )
        if evidence.border_animation_accepted:
            self.port._increment("p_item_source_border_animation_acceptances")
        return evidence.matches, evidence.errors
