"""Telemetry component; no imports from the Maa facade."""

from __future__ import annotations

import re
import json
from copy import deepcopy
from uuid import uuid4
from typing import Any, Protocol
from collections import deque
from collections.abc import Mapping, Sequence

from arena_winrate import TeamTarget, ArenaReaderError, _reader_metrics
from arena_winrate.recognition_probe import RecognitionProbe
from arena_winrate._card_detail_state import card_detail_state


class TelemetryReaderPort(Protocol):
    """Only the state and callbacks needed by this responsibility."""

    _badge_candidate_detail_checks: Any
    _badge_count_diagnostics: Any
    _badge_glyph_count_diagnostics: Any
    _badge_worker_selection: Any
    _card_candidate_groups: Any
    _card_content_generation_diagnostics: Any
    _card_detail_texts: Any
    _card_empty_diagnostics: Any
    _card_empty_flags: Any
    _card_excluded_duplicate_flags: Any
    _card_face_cost_diagnostics: Any
    _card_predictions: Any
    _card_rows: Any
    _card_source_guard_frames: Any
    _card_stage_plan: Any
    _card_swipe_duration_ms: Any
    _card_transaction_started: Any
    _cost_customization_fallbacks: Any
    _detail_count_overrides: Any
    _detail_failure_frames: Any
    _detail_semantic_confirmations: Any
    _detail_title_disambiguations: Any

    def _emit_detail_fallback_diagnostic(self, outcome: str, error: str | None = ...) -> None: ...
    def _finish_card_transaction(
        self, key: tuple[int, int], *, failed: bool = ..., error: str | None = ..., superseded: bool = ..., cancelled: bool = ...
    ) -> None: ...
    def _increment(self, name: str) -> None: ...

    _last_detail_failure_location: Any
    _member_failure_frames: Any
    _member_long_press_seconds: Any
    _member_metric_baseline: Any

    @staticmethod
    def _member_read_metrics_event(target: TeamTarget, stage_number: int, member_slot: int, evidence: dict[str, Any]) -> dict[str, Any]: ...

    _p_item_diagnostics: Any
    _p_item_generation_evidence: Any

    def _persist_detail_failure(self, error: str) -> None: ...

    _progressive_abandoned_member: Any
    _recognition_probe: Any

    def _record_duration_sample(self, name: str, elapsed: float) -> None: ...
    def _retain_failed_skill_detail_identity(self, key: tuple[int, int]) -> None: ...

    _runtime_badge_glyph_exemplar_comparisons: Any
    _runtime_badge_glyph_exemplar_diagnostics: Any
    _runtime_cost_customization_fallbacks: Any
    _runtime_counts: Any
    _runtime_detail_title_disambiguations: Any
    _runtime_duration_samples: Any
    _runtime_timing_seconds: Any

    def _save_failure_evidence(self, diagnostic: dict[str, Any], error: str, *, event: str) -> None: ...
    def _save_recognition_probe(self, detail: dict, outcome: str, error: str | None) -> None: ...

    _secondary_presence_diagnostics: Any

    def _skill_card_source_box(self, key: tuple[int, int] | None) -> tuple[int, int, int, int] | None: ...

    _zero_card_detail_candidates: Any
    _zero_card_identity_fusion_diagnostics: Any
    catalog: Any

    def runtime_sample_cursor(self) -> dict[str, int]: ...
    def runtime_samples_since(self, cursor: Mapping[str, int]) -> dict[str, Any]: ...
    def set_member_read_phase(self, phase: str, **position: Any) -> None: ...


class TelemetryReader:
    """Telemetry with live, reader-owned state."""

    def __init__(self, port: TelemetryReaderPort, *, record_root: Any, logger: Any, clock: Any) -> None:
        self.port = port
        self.record_root = record_root
        self.logger = logger
        self.clock = clock

    def runtime_sample_cursor(self) -> dict[str, int]:
        return {name: len(values) for name, values in self.port._runtime_duration_samples.items()}

    def runtime_counts_cursor(self) -> dict[str, int]:
        return dict(self.port._runtime_counts)

    def record_lineup_read_attempt(self, record: dict[str, Any]) -> None:
        self.logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True))

    def last_failure_location(self, error: object = None) -> dict[str, Any] | None:
        failure = getattr(self.port, "_last_detail_failure_location", None)
        if failure is None:
            return None
        original, location = str(failure[0]), failure[1]
        # The evaluation service adds this one fixed prefix to provider errors.
        current = str(error).removeprefix("snapshot provider failed: ")
        member = (
            str(location.get("team_id")),
            str(location.get("stage_number")),
            str(location.get("member_slot")),
        )
        contexts = re.findall(r"\b(own|self|opponent-\d+)/stage-(\d+)/member-(\d+)(?!\d)", current)
        if any(context != member for context in contexts):
            return None
        if original not in current:
            # _read_team keeps the code but inserts this member before the
            # original detail. Require the exact cause and coordinates, not
            # merely a recurring error code from an earlier transaction.
            code, separator, detail = original.partition(":")
            prefix = f"{member[0]}/stage-{member[1]}/member-{member[2]}: "
            if not separator or current != f"{code}: {prefix}{detail.lstrip()}":
                return None
        return dict(location)

    def _note_confirmed_detail_name(self, card_id: int) -> None:
        diagnostic = getattr(self.port, "_detail_failure_frames", None)
        if diagnostic is None:
            return
        diagnostic["position"]["observed_detail_card_id"] = card_id
        try:
            diagnostic["position"]["card_name"] = self.port.catalog.skill_card_title(card_id)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    def _begin_detail_diagnostics(self, *, source: Any = None, **position: Any) -> None:
        self.port._last_detail_failure_location = None
        target = position.pop("target", None)
        if target is not None:
            position.update(team_id=getattr(target, "team_id", None), opponent_position=getattr(target, "opponent_position", None))
        self.port._detail_failure_frames = {
            "position": position,
            "source": source,
            "frames": deque(maxlen=5),
            "actions": deque(maxlen=16),
            "started": self.clock.perf_counter(),
            "transaction_id": uuid4().hex,
            "member_read_id": (getattr(self.port, "_member_failure_frames", None) or {}).get("read_id"),
            "count_baseline": dict(getattr(self.port, "_runtime_counts", {})),
        }
        if position.get("kind") == "skill_card":
            probe = getattr(self.port, "_recognition_probe", None)
            if probe is None:
                probe = self.port._recognition_probe = RecognitionProbe(self.record_root)
            if probe.active():
                group = position.get("group_index")
                key = (group, position.get("card_slot"))
                self.port._detail_failure_frames.update(
                    probe_sources=tuple(getattr(self.port, "_card_source_guard_frames", {}).get(group, ())),
                    probe_source_box=self.port._skill_card_source_box(key),
                    probe_row_boxes=getattr(self.port, "_card_rows", {}).get(group, ()),
                    probe_candidates=list(getattr(self.port, "_card_candidate_groups", {}).get(key, ())),
                    probe_stage_plan=getattr(self.port, "_card_stage_plan", None),
                    probe_frames=[],
                )

    def _save_recognition_probe(self, detail: dict, outcome: str, error: str | None) -> None:
        if "probe_sources" not in detail or detail.get("probe_finished"):
            return
        detail["probe_finished"] = True
        started = self.clock.perf_counter()
        probe = self.port._recognition_probe
        try:
            position = detail["position"]
            key = (position.get("group_index"), position.get("card_slot"))
            detail["probe_body_text"] = getattr(self.port, "_card_detail_texts", {}).get(key)
            result = probe.save(detail, outcome, error)
            if result is not None:
                self.logger.info(json.dumps(result, ensure_ascii=False))
        except Exception as probe_error:
            probe.disabled = True
            self.logger.warning(json.dumps({"event": "arena_recognition_probe_disabled", "error": str(probe_error)}, ensure_ascii=False))
        finally:
            self.port._record_duration_sample("recognition_probe_save", self.clock.perf_counter() - started)

    def _observe_recognition_probe(self, image: Any, items: Any, *, reco_id=None) -> None:
        detail = getattr(self.port, "_detail_failure_frames", None)
        if detail is None or "probe_frames" not in detail:
            return
        try:
            # Derived OCR crops are not original screenshots; keep only frames
            # already delivered by _capture in this transaction.
            if any(frame[1] is image for frame in detail["frames"]):
                RecognitionProbe.observe(
                    detail,
                    image,
                    items,
                    str(detail.get("probe_phase", detail["position"].get("phase", "body"))),
                    reco_id=reco_id,
                    seconds=round(self.clock.perf_counter() - detail["started"], 6),
                )
        except Exception:
            self.port._recognition_probe.disabled = True

    def _emit_detail_fallback_diagnostic(self, outcome: str, error: str | None = None) -> None:
        """Emit existing transaction evidence, including interrupted reads."""
        detail = getattr(self.port, "_detail_failure_frames", None)
        if detail is not None:
            self.port._save_recognition_probe(detail, outcome, error)
        if detail is None or detail.get("fallback_logged"):
            return
        before = detail.get("count_baseline", {})
        counts = {
            name: value - before.get(name, 0)
            for name, value in getattr(self.port, "_runtime_counts", {}).items()
            if value > before.get(name, 0)
            and any(
                part in name
                for part in (
                    "fallback",
                    "retries",
                    "reopen",
                    "id_mismatch",
                    "one_character_recoveries",
                    "effect_roi_reads",
                    "isolated_title_roi",
                )
            )
        }
        if not counts and outcome == "completed":
            return
        detail["fallback_logged"] = True
        member = getattr(self.port, "_member_failure_frames", None) or {}
        self.logger.info(
            json.dumps(
                {
                    "event": "arena_detail_fallback_result",
                    **member.get("position", {}),
                    **detail["position"],
                    "member_read_id": detail.get("member_read_id"),
                    "transaction_id": detail.get("transaction_id"),
                    "outcome": outcome,
                    "error": error,
                    "duration_seconds": round(self.clock.perf_counter() - detail["started"], 6),
                    "fallback_counts": counts,
                },
                ensure_ascii=False,
            )
        )

    def begin_member_read_diagnostics(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
    ) -> None:
        """Keep references to existing observations until this attempt finishes."""
        self.port._last_detail_failure_location = None
        previous = getattr(self.port, "_detail_failure_frames", None)
        if previous is not None and any(
            previous["position"].get(key) not in (None, value)
            for key, value in (
                ("team_id", target.team_id),
                ("stage_number", stage_number),
                ("member_slot", member_slot),
            )
        ):
            self.port._detail_failure_frames = None
        self.port._member_failure_frames = {
            "read_id": uuid4().hex,
            "position": {
                "team_id": target.team_id,
                "opponent_position": target.opponent_position,
                "stage_number": stage_number,
                "member_slot": member_slot,
                "member_name": None,
            },
            "source": None,
            "frames": deque(maxlen=6),
            "actions": deque(maxlen=16),
            "started": self.clock.perf_counter(),
            "persisted": False,
        }
        self.port.set_member_read_phase("member_open")

    def set_member_read_phase(self, phase: str, **position: Any) -> None:
        diagnostic = getattr(self.port, "_member_failure_frames", None)
        if diagnostic is not None:
            # Coordinates from a completed card must not name a later failure.
            diagnostic["position"].update(
                phase=phase,
                kind=None,
                group_index=None,
                card_slot=None,
                screen_slot=None,
                card_name=None,
                p_item_name=None,
            )
            diagnostic["position"].update(position)

    def end_member_read_diagnostics(self) -> None:
        self.port._member_failure_frames = None

    def note_member_confirmed_card(self, card_id: int) -> None:
        diagnostic = getattr(self.port, "_member_failure_frames", None)
        if diagnostic is not None:
            diagnostic["position"]["card_name"] = self.port.catalog.skill_card_title(card_id)

    def record_skill_card_group_observation(
        self,
        group_index: int,
        card_ids: Sequence[int],
        customizations: Sequence[Mapping[str, int]],
        excluded: Sequence[bool],
        empty: Sequence[bool],
    ) -> None:
        """Bind earlier slot diagnostics to the reader's validated group result."""
        member = getattr(self.port, "_member_failure_frames", None) or {}
        cards = []
        for slot, (card_id, effects, duplicate, vacant) in enumerate(
            zip(card_ids, customizations, excluded, empty, strict=True),
            start=1,
        ):
            name = None
            if card_id:
                try:
                    name = self.port.catalog.skill_card_title(card_id)
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
            cards.append(
                {
                    "card_slot": slot,
                    "card_id": card_id,
                    "card_name": name,
                    "state": "excluded_duplicate" if duplicate else "empty" if vacant else "present",
                    "customizations": dict(effects),
                }
            )
        self.logger.info(
            json.dumps(
                {
                    **member.get("position", {}),
                    "event": "arena_skill_card_group_resolved",
                    "member_read_id": member.get("read_id"),
                    "group_index": group_index,
                    "cards": cards,
                },
                ensure_ascii=False,
            )
        )

    def persist_member_read_failure(self, error: Exception) -> None:
        """Freeze the original failure before any recovery can leave its page."""
        diagnostic = getattr(self.port, "_member_failure_frames", None)
        if diagnostic is None or diagnostic["persisted"]:
            return
        diagnostic["persisted"] = True
        detail = getattr(self.port, "_detail_failure_frames", None)
        selected = detail if detail is not None else diagnostic
        causes = []
        cause: BaseException | None = error
        seen: set[int] = set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            causes.append(
                {
                    "type": type(cause).__name__,
                    "code": getattr(cause, "code", None),
                    "error": str(cause),
                }
            )
            cause = cause.__cause__ or cause.__context__
        selected["error_chain"] = causes
        if detail is not None:
            # Freeze evidence only. Recovery and the original attempt recorder
            # retain ownership of identity state and transaction accounting.
            detail["persisted"] = True
            self.port._save_failure_evidence(detail, str(error), event="arena_detail_failure")
        else:
            self.port._save_failure_evidence(diagnostic, str(error), event="arena_reader_failure")

    def _record_detail_action(self, kind: str, box: Sequence[int], *, succeeded: bool) -> None:
        for name in ("_detail_failure_frames", "_member_failure_frames"):
            diagnostic = getattr(self.port, name, None)
            if diagnostic is not None and not diagnostic.get("persisted", False):
                diagnostic["actions"].append(
                    {
                        "kind": kind,
                        "box": list(box),
                        "succeeded": succeeded,
                        "seconds": round(self.clock.perf_counter() - diagnostic["started"], 6),
                    }
                )

    def _persist_detail_failure(self, error: str) -> None:
        """Save at most six already captured frames; never capture for logging."""
        self.port._emit_detail_fallback_diagnostic("failed", error)
        diagnostic = getattr(self.port, "_detail_failure_frames", None)
        self.port._detail_failure_frames = None
        if diagnostic is None or diagnostic.get("persisted", False):
            return
        member = getattr(self.port, "_member_failure_frames", None)
        if member is not None:
            member["persisted"] = True
        self.port._save_failure_evidence(diagnostic, error, event="arena_detail_failure")

    def _save_failure_evidence(
        self,
        diagnostic: dict[str, Any],
        error: str,
        *,
        event: str,
    ) -> None:
        """The shared writer never observes or operates the game."""
        self.port._last_detail_failure_location = (error, dict(diagnostic["position"]))
        save_started = self.clock.perf_counter()
        try:
            import cv2

            folder = self.record_root / "reader-failures" / uuid4().hex
            folder.mkdir(parents=True)
            frames = []
            if diagnostic["source"] is not None:
                frames.append((diagnostic["started"], diagnostic["source"]))
            confirmed_open = diagnostic.get("confirmed_open")
            if confirmed_open is not None:
                frames.append(confirmed_open)
                recent = [frame for frame in diagnostic["frames"] if frame[1] is not confirmed_open[1]]
                frames.extend(recent[-(6 - len(frames)) :])
            else:
                frames.extend(diagnostic["frames"])
            saved = []
            unsaved = []
            for index, (captured_at, image) in enumerate(frames[:6]):
                name = f"{index:02d}.png"
                # imencode + write_bytes supports Windows non-ASCII paths.
                try:
                    ok, encoded = cv2.imencode(".png", image)
                    if not ok:
                        raise ValueError("PNG encoding failed")
                    (folder / name).write_bytes(encoded.tobytes())
                    saved.append({"file": name, "seconds": round(captured_at - diagnostic["started"], 6)})
                    if confirmed_open is not None and image is confirmed_open[1]:
                        saved[-1]["role"] = "title_confirmed_open"
                except Exception as frame_error:
                    unsaved.append({"file": name, "error": str(frame_error)})
            evidence = {
                "event": event,
                **diagnostic["position"],
                "member_read_id": diagnostic.get("member_read_id", diagnostic.get("read_id")),
                "transaction_id": diagnostic.get("transaction_id"),
                "error": error,
                "actions": list(diagnostic["actions"]),
                "frames": saved,
                "folder": str(folder),
            }
            if unsaved:
                evidence["unsaved_frames"] = unsaved
            if event == "arena_reader_failure":
                evidence["unconfirmed_fields"] = [
                    key
                    for key in ("member_name", "group_index", "card_slot", "card_name", "screen_slot", "p_item_name")
                    if diagnostic["position"].get(key) is None
                ]
            if diagnostic.get("error_chain") is not None:
                evidence["error_chain"] = diagnostic["error_chain"]
            if diagnostic.get("p_item_text") is not None:
                evidence["p_item_text"] = diagnostic["p_item_text"]
            if diagnostic.get("p_item_restore") is not None:
                evidence["p_item_restore"] = diagnostic["p_item_restore"]
            (folder / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            self.logger.warning(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
        except Exception as diagnostic_error:
            self.logger.warning(
                json.dumps(
                    {
                        "event": "arena_failure_evidence_save_failed",
                        **diagnostic["position"],
                        "error": error,
                        "diagnostic_error": str(diagnostic_error),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        finally:
            self.port._record_duration_sample("detail_failure_evidence_save", self.clock.perf_counter() - save_started)

    def runtime_samples_since(self, cursor: Mapping[str, int]) -> dict[str, Any]:
        """Export failed and successful samples without another observation."""
        return {
            "duration_samples_seconds": {
                name: [round(value, 6) for value in values[cursor.get(name, 0) :]]
                for name, values in self.port._runtime_duration_samples.items()
                if len(values) > cursor.get(name, 0)
            },
            "unfinished_detail_transactions": [list(key) for key in self.port._card_transaction_started],
        }

    def record_member_read_attempt(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        *,
        wall_seconds: float,
        succeeded: bool,
        error_code: str | None,
        reopened: bool,
        sample_cursor: Mapping[str, int] | None = None,
        superseded: bool = False,
    ) -> None:
        self.port._record_duration_sample("member_read_attempt", wall_seconds)
        if superseded:
            diagnostic = getattr(self.port, "_member_failure_frames", None)
            self.port._progressive_abandoned_member = dict(diagnostic["position"]) if diagnostic is not None else None
            self.port._record_duration_sample("member_read_superseded", wall_seconds)
            self.port._emit_detail_fallback_diagnostic("superseded")
            for key in tuple(self.port._card_transaction_started):
                self.port._finish_card_transaction(key, superseded=True)
        elif not succeeded:
            self.port._record_duration_sample("member_read_failed_attempt", wall_seconds)
            # Finish only real accepted transactions. Open failures use a
            # separate diagnostic timer and never create an active identity.
            for key in tuple(self.port._card_transaction_started):
                self.port._finish_card_transaction(key, failed=True)
        baseline = self.port._member_metric_baseline
        cursor = sample_cursor if sample_cursor is not None else baseline[2] if baseline is not None else self.port.runtime_sample_cursor()
        self.logger.info(
            json.dumps(
                {
                    "event": "arena_member_read_attempt",
                    "member_read_id": (getattr(self.port, "_member_failure_frames", None) or {}).get("read_id"),
                    "team_id": target.team_id,
                    "opponent_position": target.opponent_position,
                    "stage_number": stage_number,
                    "member_slot": member_slot,
                    "wall_seconds": round(wall_seconds, 6),
                    "succeeded": succeeded,
                    "superseded": superseded,
                    "error_code": error_code,
                    "reopened": reopened,
                    **self.port.runtime_samples_since(cursor),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    def _finish_card_transaction(
        self,
        key: tuple[int, int],
        *,
        failed: bool = False,
        error: str | None = None,
        superseded: bool = False,
        cancelled: bool = False,
    ) -> None:
        if failed:
            self.port._retain_failed_skill_detail_identity(key)
        started, kind, ocr_started = card_detail_state(self.port).finish(key)
        if started is None:
            return
        elapsed = self.clock.perf_counter() - started
        if superseded or cancelled:
            self.port._record_duration_sample("skill_card_detail_superseded" if superseded else "skill_card_detail_cancelled", elapsed)
        else:
            self.port._record_duration_sample("skill_card_detail_transaction", elapsed)
            self.port._record_duration_sample(kind, elapsed)
        if failed:
            self.port._record_duration_sample("skill_card_detail_failed_transaction", elapsed)
            self.port._increment("skill_card_detail_failed_transactions")
            phase = getattr(self.port, "_detail_failure_frames", None)
            if phase is not None:
                self.port._record_duration_sample(f"skill_card_detail_{phase['position'].get('phase', 'body')}_failed", elapsed)
            self.port._persist_detail_failure(error or (phase.get("error") if phase is not None else None) or "skill_card_transaction_failed")
        else:
            self.port._emit_detail_fallback_diagnostic("superseded" if superseded else "cancelled" if cancelled else "completed")
            self.port._detail_failure_frames = None
        if ocr_started is not None:
            backend_started, cache_hits_started = ocr_started
            self.port._record_duration_sample(
                "skill_card_detail_capture_full_frame_ocr_backend_calls",
                float(
                    self.port._runtime_counts.get(
                        "detail_capture_broad_ocr_backend_calls",
                        0,
                    )
                    - backend_started
                ),
            )
            self.port._record_duration_sample(
                "skill_card_detail_capture_full_frame_ocr_cache_hits",
                float(
                    self.port._runtime_counts.get(
                        "detail_capture_broad_ocr_cache_hits",
                        0,
                    )
                    - cache_hits_started
                ),
            )

    def runtime_metrics(self) -> dict[str, Any]:
        selection = self.port._badge_worker_selection
        sample_summaries = _reader_metrics.duration_percentiles(self.port._runtime_duration_samples)
        return {
            "timing_seconds": {key: round(value, 6) for key, value in sorted(self.port._runtime_timing_seconds.items())},
            "counts": dict(sorted(self.port._runtime_counts.items())),
            "duration_percentiles_seconds": sample_summaries,
            "badge_workers": None if selection is None else selection.workers,
            "badge_worker_cache_hit": None if selection is None else selection.cache_hit,
            "badge_worker_benchmarks": ([] if selection is None else list(selection.benchmarks)),
            "card_swipe_duration_ms": self.port._card_swipe_duration_ms,
            "member_long_press_duration_ms": int(round(self.port._member_long_press_seconds * 1000)),
            "point_click_dispatch": "controller_point_context_box",
            "cost_customization_fallbacks": [
                dict(value)
                for value in getattr(
                    self.port,
                    "_runtime_cost_customization_fallbacks",
                    (),
                )
            ],
            "badge_glyph_exemplar_diagnostics": [
                deepcopy(value)
                for value in getattr(
                    self.port,
                    "_runtime_badge_glyph_exemplar_diagnostics",
                    (),
                )
            ],
            "badge_glyph_exemplar_comparisons": [
                deepcopy(value)
                for value in getattr(
                    self.port,
                    "_runtime_badge_glyph_exemplar_comparisons",
                    (),
                )
            ],
            "detail_title_disambiguations": [
                deepcopy(value)
                for value in getattr(
                    self.port,
                    "_runtime_detail_title_disambiguations",
                    (),
                )
            ],
        }

    def member_observation_evidence(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
    ) -> dict[str, Any]:
        """Return only low-dimensional evidence from the active member page."""

        if self.port._member_metric_baseline is None:
            raise ArenaReaderError(
                "member_observation_generation_missing",
                f"stage-{stage_number}/member-{member_slot} has no active metric generation",
            )
        metric_deltas = _reader_metrics.member_metric_deltas(
            self.port._runtime_timing_seconds,
            self.port._runtime_counts,
            self.port._runtime_duration_samples,
            self.port._member_metric_baseline,
        )
        groups = []
        for group_index in (0, 1):
            groups.append(
                {
                    "group_index": group_index,
                    "badge_count_diagnostics": [dict(value) for value in self.port._badge_count_diagnostics.get(group_index, ())],
                    "detail_count_overrides": [
                        dict(value)
                        for (detail_group, _), value in sorted(self.port._detail_count_overrides.items())
                        if detail_group == group_index
                    ],
                    "detail_semantic_confirmations": [
                        dict(value)
                        for (detail_group, _), value in sorted(self.port._detail_semantic_confirmations.items())
                        if detail_group == group_index
                    ],
                    "auxiliary_badge_glyph_counts": [
                        dict(value)
                        for (glyph_group, _), value in sorted(self.port._badge_glyph_count_diagnostics.items())
                        if glyph_group == group_index
                    ],
                    "card_face_cost_evidence": [
                        dict(value)
                        for (cost_group, _), value in sorted(self.port._card_face_cost_diagnostics.items())
                        if cost_group == group_index
                    ],
                    "zero_card_identity_fusions": [
                        dict(value)
                        for (fusion_group, _), value in sorted(self.port._zero_card_identity_fusion_diagnostics.items())
                        if fusion_group == group_index
                    ],
                    "zero_card_detail_candidates": [
                        {
                            "slot": slot,
                            "candidate_ids": list(candidate_ids),
                        }
                        for (detail_group, slot), candidate_ids in sorted(self.port._zero_card_detail_candidates.items())
                        if detail_group == group_index
                    ],
                    "excluded_duplicate_flags": list(self.port._card_excluded_duplicate_flags.get(group_index, ())),
                    "empty_flags": list(getattr(self.port, "_card_empty_flags", {}).get(group_index, ())),
                    "empty_slot_diagnostics": [
                        dict(value)
                        for value in getattr(
                            self.port,
                            "_card_empty_diagnostics",
                            {},
                        ).get(group_index, ())
                    ],
                    "resolved_card_ids": [self.port._card_predictions.get((group_index, slot), 0) for slot in range(1, 7)],
                    "candidate_business_ids": [list(self.port._card_candidate_groups.get((group_index, slot), ())) for slot in range(1, 7)],
                }
            )
        evidence = {
            "member_read_id": (getattr(self.port, "_member_failure_frames", None) or {}).get("read_id"),
            "stage_number": stage_number,
            "member_slot": member_slot,
            "p_items": [dict(value) for value in self.port._p_item_diagnostics],
            "p_item_generation": dict(self.port._p_item_generation_evidence),
            "groups": groups,
            "secondary_presence_diagnostics": [dict(value) for value in self.port._secondary_presence_diagnostics],
            "card_content_generation_diagnostics": [dict(value) for value in self.port._card_content_generation_diagnostics],
            **metric_deltas,
            "badge_candidate_detail_checks": [
                dict(value)
                for value in self.port._badge_candidate_detail_checks
                if value.get("stage_number") == stage_number and value.get("member_slot") == member_slot
            ],
            "cost_customization_fallbacks": [
                dict(value)
                for value in getattr(
                    self.port,
                    "_cost_customization_fallbacks",
                    {},
                ).values()
                if value.get("team_id") == target.team_id
                and value.get("stage_number") == stage_number
                and value.get("member_slot") == member_slot
            ],
            "detail_title_disambiguations": [
                deepcopy(value)
                for value in getattr(
                    self.port,
                    "_detail_title_disambiguations",
                    (),
                )
            ],
            "screenshots_persisted": False,
        }
        self.logger.info(
            json.dumps(
                self.port._member_read_metrics_event(
                    target,
                    stage_number,
                    member_slot,
                    evidence,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return evidence


def _member_read_metrics_event(target: TeamTarget, stage_number: int, member_slot: int, evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep production pace evidence low-dimensional and image-free."""

    return {
        "event": "arena_member_read_metrics",
        "member_read_id": evidence.get("member_read_id"),
        "team_id": target.team_id,
        "stage_number": stage_number,
        "member_slot": member_slot,
        "timing_seconds": evidence["timing_seconds"],
        "counts": evidence["counts"],
        "duration_samples_seconds": evidence["duration_samples_seconds"],
        "detail_title_disambiguations": [dict(value) for value in evidence.get("detail_title_disambiguations", ())],
        "screenshots_persisted": False,
    }
