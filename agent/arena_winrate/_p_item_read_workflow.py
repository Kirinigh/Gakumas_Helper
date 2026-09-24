"""P-item matching and bounded detail confirmation through the reader's live port.

Member observations, diagnostics and resource objects remain owned by the
reader. These components retain only explicit dependencies, never snapshots of
mutable state. Detail parsing and recovery reuse the existing detail session.
"""

from __future__ import annotations

import json
from typing import Any, Protocol
from pathlib import Path
from collections.abc import Mapping, Callable, Sequence

from p_item_recognition import PItemReferenceError, PItemReferenceDecision
from card_selection.model import frame_identifier

from .reader import TeamTarget
from .stages import ContestSeasonDefinition
from .catalog import ArenaEntityCatalog
from .geometry import p_item_screen_slot_to_engine_slot, p_item_screen_order_to_engine_order
from ._reader_errors import ArenaReaderError


class PItemWorkflowClock(Protocol):
    def monotonic(self) -> float: ...
    def perf_counter(self) -> float: ...


class PItemWorkflowLogger(Protocol):
    def info(self, message: str) -> Any: ...


class PItemReadPort(Protocol):
    """Only the reference reader, catalog scope and this member's result views."""

    p_item_reader: Any
    catalog: ArenaEntityCatalog
    season: ContestSeasonDefinition
    _p_item_source_ocr_cache: dict[Any, Any]
    _p_item_catalog_compatibility_checked: bool
    _p_item_reference_gallery_ids: tuple[int, ...]
    _p_item_reference_missing_arena_ids: tuple[int, ...]
    _p_item_reference_unknown_catalog_ids: tuple[int, ...]
    _p_item_reference_provisional_ids: tuple[int, ...]
    _p_item_diagnostics: tuple[dict[str, Any], ...]
    _p_item_generation_evidence: dict[str, Any]
    _member_failure_frames: dict[str, Any] | None

    def _capture(self) -> Any: ...
    def _increment(self, name: str) -> None: ...
    def _add_timing(self, name: str, elapsed: float) -> None: ...
    def _record_duration_sample(self, name: str, elapsed: float) -> None: ...
    def _load_static_resource(self, name: str, root: str | Path, factory: Callable[..., Any]) -> Any: ...
    def _assert_p_item_reference_catalog_compatibility(self) -> None: ...
    def _capture_stable_p_item_generation(
        self,
        boxes: Sequence[tuple[int, int, int, int]],
        *,
        initial_image: Any | None = None,
    ) -> tuple[tuple[Any, Any, Any], tuple[float, ...], int]: ...
    def _begin_detail_diagnostics(self, **position: Any) -> None: ...
    def _confirm_p_item_detail(self, box: tuple[int, int, int, int], **kwargs: Any) -> int: ...
    def _p_item_decision_evidence(
        self,
        slot: int,
        decision: PItemReferenceDecision,
        *,
        path: str,
        resolved_id: int | None = None,
        screen_slot: int | None = None,
    ) -> dict[str, Any]: ...


class PItemDetailPort(Protocol):
    """Existing live diagnostic/probe views and the detail-session entry point."""

    _runtime_duration_samples: Mapping[str, Sequence[float]]
    _detail_failure_frames: dict[str, Any] | None
    _recognition_probe: Any

    def _begin_detail_diagnostics(self, **position: Any) -> None: ...
    def _confirm_p_item_detail_once(self, box: tuple[int, int, int, int], **kwargs: Any) -> int: ...
    def _record_duration_sample(self, name: str, elapsed: float) -> None: ...
    def _persist_detail_failure(self, error: str) -> None: ...
    def _save_recognition_probe(self, detail: dict[str, Any], outcome: str, error: str | None) -> None: ...


def p_item_decision_evidence(
    slot: int,
    decision: PItemReferenceDecision,
    *,
    path: str,
    resolved_id: int | None = None,
    screen_slot: int | None = None,
) -> dict[str, Any]:
    evidence = {
        "slot": slot,
        "state": decision.status,
        "reason": decision.reason,
        "path": path,
        "p_item_id": decision.p_item_id if resolved_id is None else resolved_id,
        "candidates": list(decision.candidates),
        "similarity": (None if decision.similarity is None else round(decision.similarity, 6)),
        "margin": None if decision.margin is None else round(decision.margin, 6),
        "content_generation_max_mean_abs_error": round(
            decision.content_generation_max_mean_abs_error,
            6,
        ),
        "frames_byte_identical": decision.frames_byte_identical,
        "ranking_route": decision.ranking_route,
        "full_fallback_used": decision.full_fallback_used,
        "scale_fallback": decision.scale_fallback,
        "completeness": [
            {
                "foreground_fraction": round(item.foreground_fraction, 6),
                "minimum_opposite_half_fraction": round(
                    item.minimum_opposite_half_fraction,
                    6,
                ),
            }
            for item in decision.completeness
        ],
        "top_k": [
            {
                "p_item_id": hit.p_item_id,
                "similarity": round(hit.similarity, 6),
                "size": list(hit.size),
                "offset": list(hit.offset),
            }
            for hit in decision.top_k
        ],
    }
    if screen_slot is not None:
        evidence["screen_slot"] = screen_slot
    return evidence


class PItemReadWorkflow:
    """Match a stable four-slot generation, then confirm only authorized slots."""

    def __init__(
        self,
        port: PItemReadPort,
        *,
        clock: PItemWorkflowClock,
        logger: PItemWorkflowLogger,
        reference_root: str | Path,
        reader_factory: Callable[..., Any],
    ) -> None:
        self.port = port
        self.clock = clock
        self.logger = logger
        self.reference_root = reference_root
        self.reader_factory = reader_factory

    def assert_catalog_compatibility(self) -> None:
        """Bind fixed-gallery drift to a safe global-title confirmation route."""

        if getattr(self.port, "_p_item_catalog_compatibility_checked", False):
            return
        gallery = getattr(self.port.p_item_reader, "gallery", None)
        raw_gallery_ids = getattr(gallery, "p_item_ids", None)
        if raw_gallery_ids is None:
            # Explicit injected readers are used by isolated tests and
            # diagnostics. Production Task085PItemReader always exposes its
            # fixed gallery and therefore cannot bypass this compatibility gate.
            return
        gallery_ids = frozenset(int(value) for value in raw_gallery_ids)
        catalog_ids = frozenset(self.port.catalog.p_item_business_ids())
        required_ids = frozenset(self.port.catalog.arena_p_item_reference_required_ids())
        missing = tuple(sorted(required_ids - gallery_ids))
        unknown = tuple(sorted(gallery_ids - catalog_ids))
        self.port._p_item_reference_missing_arena_ids = missing
        self.port._p_item_reference_unknown_catalog_ids = unknown
        self.port._p_item_reference_provisional_ids = tuple(sorted(gallery_ids.intersection(getattr(gallery, "provisional_p_item_ids", ()))))
        self.port._p_item_reference_gallery_ids = tuple(sorted(gallery_ids))
        self.port._p_item_catalog_compatibility_checked = True

    def read_ids(
        self,
        target: TeamTarget,
        stage_number: int,
        slot: int,
    ) -> Sequence[int]:
        self.port._p_item_source_ocr_cache = {}
        if self.port.p_item_reader is None:
            self.port.p_item_reader = self.port._load_static_resource(
                "p_item_reference",
                self.reference_root,
                self.reader_factory,
            )
        plan = self.port.season.stages[stage_number - 1].plan
        self.port._assert_p_item_reference_catalog_compatibility()
        catalog = getattr(self.port, "catalog", None)
        scope_resolver = getattr(catalog, "arena_p_item_candidate_ids", None)
        eligible_ids = scope_resolver(plan=plan) if scope_resolver is not None else None
        slot_domains = tuple(scope_resolver(plan=plan, slot_index=index) for index in range(4)) if scope_resolver is not None else None
        read_scope = {} if slot_domains is None else {"eligible_p_item_ids_by_slot": slot_domains}
        first = self.port._capture()
        height, width = first.shape[:2]
        icon = int(width * 0.09)
        boxes = tuple((int(width * x), int(height * 0.207), icon, icon) for x in (0.05, 0.147, 0.247, 0.34))
        decisions: tuple[PItemReferenceDecision, ...] = ()
        images: tuple[Any, ...] = ()
        scale_budget = [True] * 4
        scale_history: list[dict[str, Any]] = []
        generation_attempt = 0
        matching_seconds = 0.0
        content_generation_errors: tuple[float, ...] = ()
        content_generation_rejected_windows = 0
        while generation_attempt < 2:
            generation_attempt += 1
            (
                images,
                content_generation_errors,
                rejected_windows,
            ) = self.port._capture_stable_p_item_generation(
                boxes,
                initial_image=first if generation_attempt == 1 else None,
            )
            content_generation_rejected_windows += rejected_windows
            if slot_domains is not None:
                read_scope["scale_fallback_allowed_by_slot"] = tuple(scale_budget)
            started = self.clock.perf_counter()
            try:
                decisions = tuple(self.port.p_item_reader.read(images, boxes, plan=plan, **read_scope))
            except (PItemReferenceError, OSError, KeyError, TypeError, ValueError) as error:
                raise ArenaReaderError(
                    "p_item_reference_runtime_failed",
                    "fixed P-item reference matching failed",
                ) from error
            matching_seconds += self.clock.perf_counter() - started
            if len(decisions) != 4:
                raise ArenaReaderError(
                    "p_item_count_mismatch",
                    f"P-item reference reader returned {len(decisions)} slots",
                )
            for index, decision in enumerate(decisions):
                if decision.scale_fallback is not None:
                    scale_budget[index] = False
                    diagnostics = decision.scale_fallback
                    scale_history.append({"generation": generation_attempt, "screen_slot": index + 1, **diagnostics})
                    self.port._increment("p_item_scale_fallback_attempts")
                    self.port._increment("p_item_scale_fallback_" + str(diagnostics["outcome"]))
                    elapsed = float(diagnostics["duration_seconds"])
                    self.port._add_timing("p_item_scale_fallback", elapsed)
                    self.port._record_duration_sample("p_item_scale_fallback", elapsed)
                    self.logger.info(
                        json.dumps(
                            {
                                "event": "arena_p_item_scale_fallback",
                                "team_id": getattr(target, "team_id", None),
                                "opponent_position": getattr(target, "opponent_position", None),
                                "stage_number": stage_number,
                                "member_slot": slot,
                                "member_read_id": (getattr(self.port, "_member_failure_frames", None) or {}).get("read_id"),
                                "screen_slot": index + 1,
                                "generation": generation_attempt,
                                "p_item_id": decision.p_item_id,
                                **diagnostics,
                            },
                            ensure_ascii=False,
                        )
                    )
            if all(decision.accepted for decision in decisions):
                break
            if generation_attempt == 1:
                self.port._increment("p_item_generation_resamples")
                continue
            break
        self.port._add_timing("p_item_reference_matching", matching_seconds)
        ambiguous = tuple((index, decision) for index, decision in enumerate(decisions, start=1) if not decision.accepted)
        detail_eligible = tuple(
            (index, decision)
            for index, decision in ambiguous
            if decision.reason
            in {
                "reference_margin_below_threshold",
                "reference_similarity_below_threshold",
                "reference_domain_empty",
            }
            and decision.candidates
        )
        if len(detail_eligible) != len(ambiguous):
            raise ArenaReaderError(
                "p_item_unknown",
                "P-item slots stayed ambiguous after one fresh generation: "
                + ", ".join(
                    f"slot {index}={decision.reason}"
                    "("
                    f"content_error={decision.content_generation_max_mean_abs_error:.6f},"
                    f"similarity={decision.similarity!r},"
                    f"margin={decision.margin!r},"
                    f"candidates={decision.candidates!r},"
                    "top_k="
                    f"{tuple((hit.p_item_id, round(hit.similarity, 6)) for hit in decision.top_k)!r},"
                    "minimum_opposite_half="
                    f"{tuple(round(item.minimum_opposite_half_fraction, 6) for item in decision.completeness)!r}"
                    ")"
                    for index, decision in ambiguous
                ),
            )
        if getattr(self.port, "_p_item_catalog_compatibility_checked", False):
            gallery_ids = frozenset(getattr(self.port, "_p_item_reference_gallery_ids", ()))
            missing_reference_ids = tuple(sorted(frozenset(self.port.catalog.arena_p_item_reference_required_ids(plan=plan)) - gallery_ids))
        else:
            missing_reference_ids = tuple(getattr(self.port, "_p_item_reference_missing_arena_ids", ()))
        unknown_catalog_ids = frozenset(getattr(self.port, "_p_item_reference_unknown_catalog_ids", ()))
        provisional_reference_ids = frozenset(getattr(self.port, "_p_item_reference_provisional_ids", ()))
        visual_tiebreak_blocked_ids = tuple(sorted(frozenset(missing_reference_ids).union(provisional_reference_ids)))
        force_missing_reference_detail = bool(missing_reference_ids)
        force_catalog_superset_detail = bool(unknown_catalog_ids)
        force_global_detail = force_missing_reference_detail or force_catalog_superset_detail
        provisional_plan_ids = frozenset(
            provisional_reference_ids.intersection(self.port.catalog.arena_p_item_reference_required_ids(plan=plan))
            if provisional_reference_ids
            else ()
        )
        detail_by_slot = dict(detail_eligible)
        global_detail_slots: set[int] = set()
        if force_global_detail:
            detail_by_slot.update((index, decision) for index, decision in enumerate(decisions, start=1) if decision.status != "EMPTY")
            global_detail_slots.update(detail_by_slot)
        else:
            # Plan reachability only says that a provisional ID may occur; it
            # does not authorize opening unrelated, already accepted slots.
            # Keep the provisional safety check local to the slot whose
            # accepted ID or unresolved candidate set actually observed it.
            for index, decision in enumerate(decisions, start=1):
                observed_ids = (decision.p_item_id,) if decision.accepted and decision.p_item_id is not None else decision.candidates
                if provisional_reference_ids.intersection(observed_ids):
                    detail_by_slot[index] = decision
                    global_detail_slots.add(index)
        detail_budget = self.port.p_item_reader.maximum_detail_fallbacks_per_member
        budgeted_detail_slots = set(detail_by_slot) - global_detail_slots
        if force_global_detail:
            # Catalog drift is the one case that intentionally confirms every
            # non-empty slot.  It is bounded by the four-slot page itself.port.
            detail_budget = 4
            budgeted_detail_slots = set(detail_by_slot)
        if len(budgeted_detail_slots) > detail_budget:
            raise ArenaReaderError(
                "p_item_detail_budget_exceeded",
                f"P-item bounded detail fallbacks {len(budgeted_detail_slots)} exceed the per-member budget {detail_budget}",
            )

        screen_resolved_ids: list[int] = []
        screen_diagnostics: list[dict[str, Any]] = []
        global_detail_candidates = self.port.catalog.arena_p_item_reference_required_ids(plan=plan) if global_detail_slots else ()
        for screen_slot, (box, decision) in enumerate(
            zip(boxes, decisions, strict=True),
            start=1,
        ):
            engine_slot = p_item_screen_slot_to_engine_slot(screen_slot)
            if screen_slot in detail_by_slot:
                use_global_detail = screen_slot in global_detail_slots
                detail_candidates = global_detail_candidates if use_global_detail else decision.candidates
                if slot_domains is not None:
                    detail_candidates = tuple(item_id for item_id in detail_candidates if item_id in slot_domains[screen_slot - 1])
                decision_observed_ids = (decision.p_item_id,) if decision.accepted and decision.p_item_id is not None else decision.candidates
                visual_tiebreak_ids = decision_observed_ids if decision.accepted else ()
                self.port._begin_detail_diagnostics(
                    target=target,
                    stage_number=stage_number,
                    member_slot=slot,
                    kind="p_item",
                    screen_slot=screen_slot,
                    source=images[0],
                )
                resolved_id = self.port._confirm_p_item_detail(
                    box,
                    candidate_ids=detail_candidates,
                    global_title_scope=use_global_detail,
                    visual_tiebreak_ids=visual_tiebreak_ids,
                    unrepresented_ids=visual_tiebreak_blocked_ids,
                    source_images=images,
                    source_boxes=boxes,
                    plan=plan,
                    **({} if slot_domains is None else {"slot_index": screen_slot - 1}),
                )
                screen_resolved_ids.append(resolved_id)
                screen_diagnostics.append(
                    self.port._p_item_decision_evidence(
                        engine_slot,
                        decision,
                        path=(
                            "catalog_drift_global_detail_confirmation"
                            if force_missing_reference_detail
                            else "catalog_superset_global_detail_confirmation"
                            if force_catalog_superset_detail
                            else "provisional_reference_global_detail_confirmation"
                            if use_global_detail
                            else "bounded_detail_confirmation"
                        ),
                        resolved_id=resolved_id,
                        screen_slot=screen_slot,
                    )
                )
                continue
            if decision.p_item_id is None:
                raise ArenaReaderError(
                    "p_item_unknown",
                    f"P-item screen slot {screen_slot} has no resolved business ID",
                )
            screen_resolved_ids.append(decision.p_item_id)
            path = "stable_empty_slot" if decision.status == "EMPTY" else "three_frame_rendered_reference"
            self.port._increment("p_item_empty_slots" if decision.status == "EMPTY" else "p_item_exact_reference_resolutions")
            screen_diagnostics.append(
                self.port._p_item_decision_evidence(
                    engine_slot,
                    decision,
                    path=path,
                    screen_slot=screen_slot,
                )
            )
        if eligible_ids is not None:
            for screen_slot, resolved_id in enumerate(screen_resolved_ids, start=1):
                if resolved_id != 0 and resolved_id not in slot_domains[screen_slot - 1]:
                    raise ArenaReaderError(
                        "p_item_unknown",
                        f"P-item screen slot {screen_slot}: ID {resolved_id} is outside the {plan} arena domain",
                    )
        gallery = getattr(self.port.p_item_reader, "gallery", None)
        resolved_ids = p_item_screen_order_to_engine_order(screen_resolved_ids)
        self.port._p_item_diagnostics = p_item_screen_order_to_engine_order(screen_diagnostics)
        self.port._p_item_generation_evidence = {
            "frame_ids": [frame_identifier(image) for image in images],
            "fresh_resamples": generation_attempt - 1,
            "content_stable": True,
            "identity_resolved": True,
            "content_generation_errors": [round(value, 6) for value in content_generation_errors],
            "content_generation_threshold": (self.port.p_item_reader.content_generation_max_mean_abs_error),
            "content_generation_rejected_windows": (content_generation_rejected_windows),
            "eligible_candidate_count": None if eligible_ids is None else len(eligible_ids),
            "scale_fallbacks": scale_history,
            "eligible_candidate_counts_by_screen_slot": (None if slot_domains is None else [len(domain) for domain in slot_domains]),
            "reference_gallery_sha256": getattr(gallery, "gallery_sha256", None),
            "background_workers": 2,
            "ranking_routes": [decision.ranking_route for decision in decisions],
            "full_fallbacks": sum(int(decision.full_fallback_used) for decision in decisions),
            "screen_slot_to_engine_slot": [1, 4, 3, 2],
            "matching_seconds": round(matching_seconds, 6),
            "missing_arena_reference_ids": list(missing_reference_ids),
            "unknown_catalog_gallery_ids": sorted(unknown_catalog_ids),
            "provisional_reference_ids": sorted(provisional_reference_ids),
            "provisional_plan_ids": sorted(provisional_plan_ids),
            "catalog_drift_global_detail_confirmations": (len(global_detail_slots) if force_missing_reference_detail else 0),
            "catalog_superset_global_detail_confirmations": (len(global_detail_slots) if force_catalog_superset_detail else 0),
            "provisional_reference_global_detail_confirmations": (
                sum(
                    bool(
                        provisional_reference_ids.intersection(
                            (decision.p_item_id,) if decision.accepted and decision.p_item_id is not None else decision.candidates
                        )
                    )
                    for index, decision in enumerate(decisions, start=1)
                    if index in global_detail_slots
                )
                if not force_global_detail
                else 0
            ),
        }
        return resolved_ids


class PItemDetailWorkflow:
    """Bracket the existing session with its original diagnostic/probe lifetime."""

    def __init__(
        self,
        port: PItemDetailPort,
        *,
        clock: PItemWorkflowClock,
        session_factory: Callable[..., Any],
        probe_factory: Callable[..., Any],
        probe_root: str | Path,
    ) -> None:
        self.port = port
        self.clock = clock
        self.session_factory = session_factory
        self.probe_factory = probe_factory
        self.probe_root = probe_root

    def confirm_detail(
        self,
        box: tuple[int, int, int, int],
        *,
        candidate_ids: Sequence[int],
        global_title_scope: bool = False,
        visual_tiebreak_ids: Sequence[int] = (),
        unrepresented_ids: Sequence[int] = (),
        source_images: Sequence[Any] = (),
        source_boxes: Sequence[tuple[int, int, int, int]] = (),
        plan: str | None = None,
        slot_index: int | None = None,
    ) -> int:
        started = self.clock.perf_counter()
        failed_before = len(getattr(self.port, "_runtime_duration_samples", {}).get("p_item_detail_failed_transaction", ()))
        if getattr(self.port, "_detail_failure_frames", None) is None:
            self.port._begin_detail_diagnostics(kind="p_item", box=list(box), source=source_images[0] if source_images else None)
        probe = getattr(self.port, "_recognition_probe", None)
        if probe is None:
            probe = self.port._recognition_probe = self.probe_factory(self.probe_root)
        if probe.active():
            self.port._detail_failure_frames.update(
                probe_sources=tuple(source_images),
                probe_source_box=box,
                probe_row_boxes=tuple(source_boxes),
                probe_candidates=list(candidate_ids),
                probe_stage_plan=plan,
                probe_frames=[],
            )
        try:
            result = self.port._confirm_p_item_detail_once(
                box,
                candidate_ids=candidate_ids,
                global_title_scope=global_title_scope,
                visual_tiebreak_ids=visual_tiebreak_ids,
                unrepresented_ids=unrepresented_ids,
                source_images=source_images,
                source_boxes=source_boxes,
                plan=plan,
                **({} if slot_index is None else {"slot_index": slot_index}),
            )
        except Exception as error:
            if len(getattr(self.port, "_runtime_duration_samples", {}).get("p_item_detail_failed_transaction", ())) == failed_before:
                self.port._record_duration_sample("p_item_detail_failed_transaction", self.clock.perf_counter() - started)
            self.port._persist_detail_failure(str(error))
            raise
        self.port._save_recognition_probe(self.port._detail_failure_frames, "completed", None)
        self.port._detail_failure_frames = None
        return result

    def confirm_detail_once(
        self,
        box: tuple[int, int, int, int],
        *,
        candidate_ids: Sequence[int],
        global_title_scope: bool = False,
        visual_tiebreak_ids: Sequence[int] = (),
        unrepresented_ids: Sequence[int] = (),
        source_images: Sequence[Any] = (),
        source_boxes: Sequence[tuple[int, int, int, int]] = (),
        plan: str | None = None,
        slot_index: int | None = None,
    ) -> int:
        return self.session_factory(
            self.port,
            self.clock,
            box=box,
            candidate_ids=candidate_ids,
            global_title_scope=global_title_scope,
            plan=plan,
            slot_index=slot_index,
            source_boxes=source_boxes,
            source_images=source_images,
            unrepresented_ids=unrepresented_ids,
            visual_tiebreak_ids=visual_tiebreak_ids,
        ).run()
