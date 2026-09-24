"""Card identity component; no imports from the Maa facade."""

from __future__ import annotations

import json
from typing import Any, Protocol
from pathlib import Path
from collections.abc import Callable, Sequence

from utils import logger as _base_logger
from arena_winrate import (
    TeamTarget,
    ArenaReaderError,
    ClickedSkillCard,
    ArenaCatalogError,
    BadgeReferenceError,
    BadgeReferenceGallery,
    stable_reference_business_candidates,
)
from card_selection import EmbeddingCardRecognizer
from card_selection.model import frame_identifier, frame_identifiers
from arena_winrate.task_log import arena_task_log
from arena_winrate.cancellation import ArenaTaskCancelled, ArenaReadSuperseded


class CardIdentityReaderPort(Protocol):
    """Only the state and callbacks needed by this responsibility."""

    def _add_timing(self, name: str, elapsed: float) -> None: ...
    def _assert_skill_card_reference_catalog_compatibility(self) -> None: ...

    _card_candidate_groups: Any
    _card_count_frames: Any
    _card_empty_flags: Any
    _card_identity_frames: Any
    _card_images: Any
    _card_predictions: Any

    def _card_recognizer(self, *, customized: bool) -> tuple[EmbeddingCardRecognizer, float, float]: ...

    _card_recognizer_lock: Any
    _card_recognizers: Any
    _card_reference_gallery: Any

    def _card_references(self) -> BadgeReferenceGallery: ...

    _card_rows: Any
    _card_stage_plan: Any

    def _classify_card_candidate(self, recognizer, image, candidate, *, preprocess_cache=..., **kwargs): ...
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
    def _infer_zero_card_identity_from_detail(
        self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int, candidate_ids: Sequence[int]
    ) -> ClickedSkillCard: ...

    _inferred_clicked_cards: Any
    _last_detail_failure_location: Any

    def _load_static_resource(self, name: str, root: str | Path, factory: Callable[[], Any]) -> Any: ...
    def _reader_compute(self, metric: str, operation: Callable[[], Any]) -> Any: ...
    def _record_duration_sample(self, name: str, elapsed: float) -> None: ...
    def _refresh_visible_card_groups(self, requested_group: int) -> tuple[Any, Any, Any]: ...
    def _resolve_zero_card_reference_ids(self, group_index: int, slot_indices: Sequence[int], *, plan: str) -> dict[int, int]: ...
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

    _runtime_counts: Any

    def _skill_card_catalog_candidates_for_plan(self, plan: str, *, slot_index: int | None = ...) -> tuple[int, ...]: ...

    _skill_card_catalog_compatibility_checked: Any

    def _skill_card_forward_drift_for_plan(self, plan: str) -> tuple[int, ...]: ...
    def _skill_card_forward_drift_slot_indices(self, missing_card_ids: Sequence[int], *, plan: str | None = ...) -> tuple[int, ...]: ...

    _skill_card_gallery_fallback_started: Any
    _skill_card_reference_gallery_ids: Any
    _skill_card_reference_missing_catalog_ids: Any

    def _skill_card_scope_kwargs(self, slot_index: int, *, plan: str | None = ...) -> dict[str, Any]: ...
    def _skill_card_scoped_candidates(self, candidate_ids: Sequence[int], *, plan: str, slot_index: int) -> tuple[int, ...]: ...
    def _stable_zero_card_embedding_family_candidates(self, group_index: int, slot_index: int, *, plan: str) -> tuple[int, ...]: ...
    def _start_skill_card_gallery_fallback(self) -> None: ...
    def _validated_card_group_row(
        self, image: Any, group_index: int, *, allow_fixed_secondary: bool = ...
    ) -> tuple[tuple[int, int, int, int], ...]: ...

    _zero_card_detail_candidates: Any
    _zero_card_embedding_detail_candidates: Any
    _zero_card_identity_fusion_diagnostics: Any
    catalog: Any
    context: Any
    season: Any


class CardIdentityReader:
    """Card identity with live, reader-owned state."""

    def __init__(self, port: CardIdentityReaderPort, *, arena_card_model_root: Any, card_model_root: Any, logger: Any, clock: Any) -> None:
        self.port = port
        self.arena_card_model_root = arena_card_model_root
        self.card_model_root = card_model_root
        self.logger = logger
        self.clock = clock

    def _card_recognizer(self, *, customized: bool) -> tuple[EmbeddingCardRecognizer, float, float]:
        """Load a visual-family candidate source without accepting its business ID.

        Neither embedding becomes a business ID directly.  Positive cards are
        resolved by their mandatory detail page; zero cards require independent
        agreement with the fixed clean-reference gallery.  The intentionally
        null production thresholds therefore remain a ban on accepting a
        prediction ID while a permissive score supplies bounded candidates.
        """

        with self.port._card_recognizer_lock:
            cached = self.port._card_recognizers.get(customized)
            if cached is not None:
                return cached
            root = self.arena_card_model_root if customized else self.card_model_root
            try:

                def load():
                    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
                    recognizer = EmbeddingCardRecognizer.load(root)
                    runtime = manifest["runtime"]
                    if "acceptance_threshold" not in runtime or "minimum_margin" not in runtime:
                        raise KeyError("runtime candidate-source metadata")
                    return recognizer, 0.000001, 0.0

                loaded = self.port._load_static_resource(
                    "card_model_custom" if customized else "card_model_base",
                    root,
                    load,
                )
            except (OSError, KeyError, TypeError, ValueError, ImportError) as error:
                model_label = "arena custom-card" if customized else "base card"
                raise ArenaReaderError(
                    "skill_card_runtime_not_approved",
                    f"{model_label} embedding candidate assets are unavailable",
                ) from error
            self.port._card_recognizers[customized] = loaded
            return loaded

    def _card_visual_family_candidates(
        self,
        *,
        stage_number: int,
        image: Any,
        box: tuple[int, int, int, int],
        slot_index: int,
    ) -> tuple[int, ...]:
        """Union base-card and arena custom-card Top-K families before a detail click.

        A seeded green plate is intentionally only a high-recall shortlist, so
        it may come from either a real customized overlay or clean card art.
        The two embeddings only bound title verification; neither one decides
        badge presence or supplies the final business ID.
        """

        from card_selection.types import CandidateBox

        plan = self.port.season.stages[stage_number - 1].plan
        candidate_box = CandidateBox(
            slot=slot_index,
            box=box,
            detector_label="cards",
            detector_score=1.0,
        )
        candidates: list[int] = []
        frame_ids = self.port._reader_compute(
            "card_frame_identifier",
            lambda: frame_identifiers(image, (slot_index + 100, slot_index + 200)),
        )
        preprocess_cache: dict[Any, Any] = {}
        for customized, frame_id in zip((True, False), frame_ids, strict=True):
            recognizer, acceptance_threshold, minimum_margin = self.port._card_recognizer(customized=customized)
            prediction = self.port._classify_card_candidate(
                recognizer,
                image,
                candidate_box,
                frame_id=frame_id,
                preprocess_cache=preprocess_cache,
                plan=plan,
                acceptance_threshold=acceptance_threshold,
                min_margin=minimum_margin,
                resolve_upgrade_state=False,
                **self.port._skill_card_scope_kwargs(slot_index, plan=plan),
            )
            if not prediction.accepted:
                continue
            if prediction.card_id.isdigit():
                candidates.append(int(prediction.card_id))
            candidates.extend(int(value) for visual_family in prediction.top_k_card_ids for value in visual_family)
        ordered = tuple(dict.fromkeys(candidates))
        if not ordered:
            raise ArenaReaderError(
                "skill_card_badge_detail_identity_unknown",
                "neither card embedding supplied a visual family for the seeded-plate detail candidate",
            )
        return ordered

    def _assert_skill_card_reference_catalog_compatibility(self) -> None:
        """Record forward RIS drift without treating the fixed gallery as a gate."""

        if getattr(self.port, "_skill_card_catalog_compatibility_checked", False):
            return
        if not hasattr(self.port, "_card_reference_gallery"):
            # Minimal object.__new__ test doubles do not represent a packaged
            # runtime and intentionally omit the gallery cache slot.
            return
        gallery = self.port._card_references()
        raw_gallery_ids = getattr(gallery, "business_ids", None)
        catalog_ids_resolver = getattr(
            self.port.catalog,
            "arena_skill_card_reference_required_ids",
            None,
        )
        if raw_gallery_ids is None or catalog_ids_resolver is None:
            # Isolated test doubles may not expose an inventory. Production
            # BadgeReferenceGallery always does, so this cannot bypass the
            # forward-drift route in a packaged runtime.
            return
        try:
            gallery_ids = frozenset(int(value) for value in raw_gallery_ids)
            catalog_ids = frozenset(catalog_ids_resolver())
        except (TypeError, ValueError) as error:
            raise ArenaReaderError(
                "skill_card_reference_catalog_invalid",
                "skill-card reference or active RIS catalog contains an invalid ID",
            ) from error
        self.port._skill_card_reference_gallery_ids = tuple(sorted(gallery_ids))
        self.port._skill_card_reference_missing_catalog_ids = tuple(sorted(catalog_ids - gallery_ids))
        self.port._skill_card_catalog_compatibility_checked = True

    def _skill_card_forward_drift_for_plan(self, plan: str) -> tuple[int, ...]:
        """Return active-plan IDs added after the fixed card gallery."""

        self.port._assert_skill_card_reference_catalog_compatibility()
        missing = tuple(getattr(self.port, "_skill_card_reference_missing_catalog_ids", ()))
        if not missing:
            return ()
        try:
            resolver = getattr(self.port.catalog, "arena_skill_card_candidates", None)
            if resolver is not None:
                eligible = resolver(plan=plan)
                return tuple(value for value in missing if value in eligible)
            return self.port.catalog.skill_card_candidates_for_plan(missing, plan=plan)
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_reference_catalog_invalid",
                "active RIS skill-card drift could not be restricted to the stage plan",
            ) from error

    def _skill_card_scope_kwargs(
        self,
        slot_index: int,
        *,
        plan: str | None = None,
    ) -> dict[str, Any]:
        """Share one stage/slot domain across initial, refreshed and restored frames."""
        plan = plan or getattr(self.port, "_card_stage_plan", None)
        resolver = getattr(getattr(self.port, "catalog", None), "arena_skill_card_candidates", None)
        if plan is None or resolver is None:
            # Legacy isolated test doubles have no member stage/catalog context.
            return {}
        return {"eligible_card_ids": resolver(plan=plan, slot_index=slot_index)}

    def _skill_card_scoped_candidates(
        self,
        candidate_ids: Sequence[int],
        *,
        plan: str,
        slot_index: int,
    ) -> tuple[int, ...]:
        scope = self.port._skill_card_scope_kwargs(slot_index, plan=plan).get("eligible_card_ids")
        if scope is None:
            return self.port.catalog.skill_card_candidates_for_plan(candidate_ids, plan=plan)
        return tuple(value for value in candidate_ids if value in scope)

    def _skill_card_catalog_candidates_for_plan(
        self,
        plan: str,
        *,
        slot_index: int | None = None,
    ) -> tuple[int, ...]:
        """Return the complete active-plan title scope for drift-only recovery."""

        try:
            resolver = getattr(getattr(self.port, "catalog", None), "arena_skill_card_candidates", None)
            candidate_ids = (
                tuple(sorted(resolver(plan=plan, slot_index=slot_index)))
                if resolver is not None
                else self.port.catalog.skill_card_candidates_for_plan(
                    self.port.catalog.arena_skill_card_reference_required_ids(),
                    plan=plan,
                )
            )
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_reference_catalog_invalid",
                "active RIS skill-card catalog could not provide a title scope",
            ) from error
        if not candidate_ids:
            raise ArenaReaderError(
                "skill_card_reference_catalog_invalid",
                "active RIS skill-card catalog provided an empty title scope",
            )
        return candidate_ids

    def _skill_card_forward_drift_slot_indices(
        self,
        missing_card_ids: Sequence[int],
        *,
        plan: str | None = None,
    ) -> tuple[int, ...]:
        """Map forward catalog drift to the game's fixed deck-order positions.

        Every six-card row starts with the P-idol's intrinsic card at index 0.
        A support-provided card, when present, is fixed at index 1. Ordinary
        cards can occupy indices 1 through 5. Only during catalog drift these
        positions need an authoritative detail; a closed-set gallery score
        cannot exclude a missing ordinary card at any of them.
        """

        if not missing_card_ids:
            return ()
        resolver = getattr(getattr(self.port, "catalog", None), "arena_skill_card_candidates", None)
        if resolver is not None and plan is not None:
            slots = tuple(slot for slot in range(6) if resolver(plan=plan, slot_index=slot).intersection(missing_card_ids))
            if any(slot >= 2 for slot in slots):
                self.port._start_skill_card_gallery_fallback()
            return slots
        source_type_resolver = getattr(
            self.port.catalog,
            "skill_card_source_type",
            None,
        )
        if source_type_resolver is None:
            raise ArenaReaderError(
                "skill_card_reference_catalog_invalid",
                "active RIS skill-card drift has no source-type resolver",
            )
        try:
            source_types = {str(source_type_resolver(int(card_id))) for card_id in missing_card_ids}
        except (ArenaCatalogError, TypeError, ValueError) as error:
            raise ArenaReaderError(
                "skill_card_reference_catalog_invalid",
                "active RIS skill-card drift has no reliable source-type mapping",
            ) from error
        unsupported_source_types = tuple(sorted(source_types - {"pIdol", "support", "produce"}))
        if not source_types or unsupported_source_types:
            gallery_ids = getattr(self.port, "_skill_card_reference_gallery_ids", ())
            gallery_id_range = (min(gallery_ids), max(gallery_ids)) if gallery_ids else ()
            raise ArenaReaderError(
                "skill_card_reference_gallery_update_required",
                "active RIS added unknown skill-card sources that "
                "require a matching gallery: "
                "missing_catalog_reference_ids="
                f"{getattr(self.port, '_skill_card_reference_missing_catalog_ids', ())!r}; "
                f"gallery_id_range={gallery_id_range!r}; "
                "fallback_status=not_started; "
                f"ids={tuple(missing_card_ids)!r}; "
                f"source_types={unsupported_source_types!r}",
            )
        slots = set()
        if "pIdol" in source_types:
            slots.add(0)
        if "support" in source_types:
            slots.add(1)
        if "produce" in source_types:
            slots.update(range(1, 6))
            self.port._start_skill_card_gallery_fallback()
        return tuple(sorted(slots))

    def _start_skill_card_gallery_fallback(self) -> None:
        if getattr(self.port, "_skill_card_gallery_fallback_started", False):
            return
        self.port._skill_card_gallery_fallback_started = True
        missing_ids = getattr(self.port, "_skill_card_reference_missing_catalog_ids", ())
        self.port._increment("skill_card_gallery_fallback_starts")
        self.logger.info(
            json.dumps(
                {
                    "event": "arena_skill_card_gallery_fallback",
                    "fallback_status": "started",
                    "missing_catalog_reference_ids": list(missing_ids),
                    "maximum_ordinary_slots_per_row": 5,
                    "maximum_detail_transactions_per_slot": 1,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        arena_task_log.gallery_fallback(
            getattr(self.port, "context", None),
            missing_ids,
            _base_logger,
        )

    def _run_skill_card_gallery_detail(
        self,
        operation: Callable[[], ClickedSkillCard],
        *,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        additional_zero_detail: bool = False,
    ) -> ClickedSkillCard:
        """Measure one existing detail transaction without granting another retry."""
        started = self.clock.perf_counter()
        before = dict(self.port._runtime_counts)
        missing_ids = tuple(getattr(self.port, "_skill_card_reference_missing_catalog_ids", ()))
        record = {
            "event": "arena_skill_card_gallery_fallback",
            "team_id": target.team_id,
            "stage_number": stage_number,
            "member_slot": member_slot,
            "group_index": group_index,
            "card_slot": card_slot,
            "missing_catalog_reference_ids": list(missing_ids),
            "additional_zero_detail": additional_zero_detail,
            "fallback_status": "attempted",
        }
        self.port._increment("skill_card_gallery_fallback_attempts")
        if additional_zero_detail:
            self.port._increment("skill_card_gallery_fallback_additional_zero_details")
        self.logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True))
        try:
            resolved = operation()
        except BaseException as error:
            cancelled = isinstance(error, (ArenaTaskCancelled, ArenaReadSuperseded))
            record.update(fallback_status="cancelled" if cancelled else "failed", error=str(error))
            self.port._increment("skill_card_gallery_fallback_cancellations" if cancelled else "skill_card_gallery_fallback_failures")
            if isinstance(error, ArenaReaderError):
                annotated = ArenaReaderError(
                    error.code,
                    f"{error.detail}; gallery_fallback_status=failed; missing_catalog_reference_ids={missing_ids!r}",
                    retry_whole_read=error.retry_whole_read,
                )
                failure = getattr(self.port, "_last_detail_failure_location", None)
                if failure is not None and str(failure[0]) in str(error):
                    self.port._last_detail_failure_location = (str(annotated), failure[1])
                raise annotated from error
            raise
        else:
            record.update(fallback_status="succeeded", card_id=resolved.card_id)
            self.port._increment("skill_card_gallery_fallback_successes")
            return resolved
        finally:
            elapsed = self.clock.perf_counter() - started
            self.port._add_timing("skill_card_gallery_fallback", elapsed)
            self.port._record_duration_sample("skill_card_gallery_fallback", elapsed)
            record["wall_seconds"] = round(elapsed, 6)
            record["operation_counts"] = {
                name: value - before.get(name, 0) for name, value in self.port._runtime_counts.items() if value > before.get(name, 0)
            }
            self.logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True))

    def _resolve_zero_card_reference_ids(
        self,
        group_index: int,
        slot_indices: Sequence[int],
        *,
        plan: str,
    ) -> dict[int, int]:
        identity_frames = self.port._card_identity_frames.get(group_index, ())
        if len(identity_frames) != 3 or any(len(frame) != 6 for frame in identity_frames):
            raise BadgeReferenceError(f"group {group_index} has no stable three-frame identity input")
        resolved: dict[int, int] = {}
        for slot_index in slot_indices:
            references = tuple(frame[slot_index] for frame in identity_frames)
            try:
                stable_candidates = stable_reference_business_candidates(references)
                plan_candidates = self.port._skill_card_scoped_candidates(
                    stable_candidates,
                    plan=plan,
                    slot_index=slot_index,
                )
                embedding_candidates: tuple[int, ...] = ()
                embedding_detail_candidates: tuple[int, ...] = ()
                embedding_error: str | None = None
                fused_candidates = plan_candidates
                if len(plan_candidates) != 1:
                    try:
                        embedding_candidates = self.port._stable_zero_card_embedding_family_candidates(
                            group_index,
                            slot_index,
                            plan=plan,
                        )
                        embedding_detail_candidates = getattr(
                            self.port,
                            "_zero_card_embedding_detail_candidates",
                            {},
                        ).get(
                            (group_index, slot_index + 1),
                            embedding_candidates,
                        )
                    except BadgeReferenceError as error:
                        # The base card embedding is only an optional disambiguator for a
                        # bounded clean-reference family.  If it is unavailable,
                        # the authoritative title transaction can still resolve
                        # those candidates without guessing a business ID.
                        embedding_error = str(error)
                        self.port._increment("skill_card_zero_identity_embedding_title_fallbacks")
                        self.port._zero_card_identity_fusion_diagnostics[(group_index, slot_index + 1)] = {
                            "group_index": group_index,
                            "slot": slot_index + 1,
                            "plan": plan,
                            "embedding_error": embedding_error,
                            "mode": "task075_unavailable_requires_one_title_click",
                        }
                    fused_candidates = tuple(sorted(set(plan_candidates) & set(embedding_candidates)))
                if len(fused_candidates) == 1:
                    resolved[slot_index] = fused_candidates[0]
                    continue
                detail_candidates = tuple(dict.fromkeys((*plan_candidates, *embedding_detail_candidates)))
                if not detail_candidates:
                    raise BadgeReferenceError(
                        "fixed reference, stage plan, and stable embedding family "
                        "yielded no card identity candidate: "
                        f"stable={stable_candidates!r}; plan={plan!r}; "
                        f"eligible={plan_candidates!r}; "
                        f"embedding={embedding_candidates!r}"
                    )
                # A rare independent-source conflict is resolved by one
                # authoritative title click for this slot only.  It never
                # broadens into clicking every visually uncustomized card.
                self.port._zero_card_detail_candidates[(group_index, slot_index + 1)] = detail_candidates
                diagnostic = self.port._zero_card_identity_fusion_diagnostics.get((group_index, slot_index + 1))
                if diagnostic is not None:
                    diagnostic.update(
                        {
                            "fixed_reference_candidates": list(plan_candidates),
                            "fused_candidates": list(fused_candidates),
                            "detail_candidates": list(detail_candidates),
                            "embedding_error": embedding_error,
                            "mode": "independent_identity_conflict_requires_one_title_click",
                        }
                    )
                resolved[slot_index] = 0
            except (ArenaCatalogError, BadgeReferenceError) as error:
                raise BadgeReferenceError(f"slot {slot_index + 1}: {error}") from error
        return resolved

    def _stable_zero_card_embedding_family_candidates(
        self,
        group_index: int,
        slot_index: int,
        *,
        plan: str,
    ) -> tuple[int, ...]:
        """Return base-card top-family IDs stable in the same three frames.

        This is used only when the fixed clean reference remains non-unique
        after the authoritative stage-plan filter.  The embedding never
        supplies a business ID by itself: its stable top visual family must
        intersect the independent fixed-reference candidates uniquely.
        """

        from card_selection.types import CandidateBox

        frames = self.port._card_count_frames.get(group_index, ())
        if len(frames) != 3:
            raise BadgeReferenceError(f"group {group_index} has no stable three-frame embedding input")
        recognizer, acceptance_threshold, minimum_margin = self.port._card_recognizer(customized=False)
        frame_candidates: list[set[int]] = []
        detail_candidates: set[int] = set()
        frame_diagnostics: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(frames):
            row = self.port._validated_card_group_row(frame, group_index)
            box = row[slot_index]
            prediction = self.port._classify_card_candidate(
                recognizer,
                frame,
                CandidateBox(
                    slot=slot_index,
                    box=box,
                    detector_label="cards",
                    detector_score=1.0,
                ),
                frame_id=frame_identifier(
                    frame,
                    300 + group_index * 10 + slot_index,
                ),
                plan=plan,
                acceptance_threshold=acceptance_threshold,
                min_margin=minimum_margin,
                resolve_upgrade_state=False,
                **self.port._skill_card_scope_kwargs(slot_index, plan=plan),
            )
            if not prediction.accepted or not prediction.top_k_card_ids:
                raise BadgeReferenceError(
                    "the base card embedding did not yield an accepted top visual family for "
                    f"group {group_index}/slot {slot_index + 1}/frame {frame_index}: "
                    f"{prediction.reason}"
                )
            visual_families = tuple(tuple(dict.fromkeys(int(value) for value in family)) for family in prediction.top_k_card_ids)
            eligible_families: list[tuple[int, ...]] = []
            for family in visual_families:
                try:
                    eligible_family = self.port._skill_card_scoped_candidates(
                        family,
                        plan=plan,
                        slot_index=slot_index,
                    )
                except ArenaCatalogError as error:
                    raise BadgeReferenceError(str(error)) from error
                eligible_families.append(eligible_family)
                detail_candidates.update(eligible_family)
            top_family = visual_families[0]
            eligible = eligible_families[0]
            frame_candidates.append(set(eligible))
            frame_diagnostics.append(
                {
                    "frame_index": frame_index,
                    "top_family": list(top_family),
                    "eligible": list(eligible),
                    "top_k_families": [list(family) for family in visual_families],
                    "top_k_eligible_families": [list(family) for family in eligible_families],
                    "top_k_scores": [round(float(score), 6) for score in getattr(prediction, "top_k_scores", ())],
                    "confidence": round(float(prediction.confidence), 6),
                    "margin": round(float(prediction.margin), 6),
                }
            )
        stable = tuple(sorted(set.intersection(*frame_candidates)))
        stable_detail_candidates = tuple(sorted(detail_candidates))
        detail_candidate_cache = getattr(
            self.port,
            "_zero_card_embedding_detail_candidates",
            None,
        )
        if detail_candidate_cache is None:
            detail_candidate_cache = {}
            self.port._zero_card_embedding_detail_candidates = detail_candidate_cache
        detail_candidate_cache[(group_index, slot_index + 1)] = stable_detail_candidates
        self.port._zero_card_identity_fusion_diagnostics[(group_index, slot_index + 1)] = {
            "group_index": group_index,
            "slot": slot_index + 1,
            "plan": plan,
            "stable_embedding_candidates": list(stable),
            "bounded_top_k_detail_candidates": list(stable_detail_candidates),
            "frames": frame_diagnostics,
            "mode": (
                "fixed_reference_stage_plan_task075_top_family_fusion" if stable else "task075_top_family_unavailable_requires_one_title_click"
            ),
        }
        if stable:
            self.port._increment("skill_card_zero_identity_embedding_fusions")
        else:
            self.port._increment("skill_card_zero_identity_topk_title_fallbacks")
        return stable

    def read_skill_card_id_hints(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        customization_counts: Sequence[int],
        excluded_duplicate_flags: Sequence[bool] | None = None,
    ) -> Sequence[int]:
        row = self.port._card_rows.get(group_index, ())
        image = self.port._card_images.get(group_index)
        if len(row) != 6 or image is None or len(customization_counts) != 6:
            raise ArenaReaderError(
                "skill_card_identity_input_mismatch",
                f"group {group_index} is missing its six boxes, frame, or customization counts",
            )
        if excluded_duplicate_flags is None:
            excluded_duplicate_flags = (False,) * 6
        if len(excluded_duplicate_flags) != 6:
            raise ArenaReaderError(
                "skill_card_excluded_duplicate_count_mismatch",
                f"group {group_index} yielded duplicate flags {excluded_duplicate_flags!r}",
            )
        empty_flags = getattr(self.port, "_card_empty_flags", {}).get(
            group_index,
            (False,) * 6,
        )
        if len(empty_flags) != 6:
            raise ArenaReaderError(
                "skill_card_empty_slot_count_mismatch",
                f"group {group_index} yielded empty flags {empty_flags!r}",
            )
        plan = self.port.season.stages[stage_number - 1].plan
        forward_drift_ids = self.port._skill_card_forward_drift_for_plan(plan)
        drift_slot_indices = self.port._skill_card_forward_drift_slot_indices(
            forward_drift_ids,
            plan=plan,
        )
        drift_title_candidates = {slot: self.port._skill_card_catalog_candidates_for_plan(plan, slot_index=slot) for slot in drift_slot_indices}
        ordinary_drift = any(slot >= 2 for slot in drift_slot_indices)
        if forward_drift_ids:
            self.port._increment("skill_card_catalog_drift_groups")
        predictions: list[int] = []
        from card_selection.types import CandidateBox

        zero_slot_indices = tuple(
            index
            for index, (customization_count, excluded, empty) in enumerate(
                zip(
                    customization_counts,
                    excluded_duplicate_flags,
                    empty_flags,
                    strict=True,
                )
            )
            if not excluded
            and not empty
            and customization_count == 0
            and self.port._inferred_clicked_cards.get((group_index, index + 1)) is None
        )
        forced_detail_slot_indices = tuple(slot_index for slot_index in zero_slot_indices if slot_index in drift_slot_indices)
        reference_slot_indices = tuple(slot_index for slot_index in zero_slot_indices if slot_index not in drift_slot_indices)
        zero_reference_ids: dict[int, int] = {slot_index: 0 for slot_index in forced_detail_slot_indices}
        if forced_detail_slot_indices:
            # A closed-set visual gallery cannot prove that a stable unique
            # match is not a newly added RIS card projected onto an older ID.
            # Fixed deck order narrows the affected positions; ordinary
            # drift still needs every eligible zero slot's exact detail.
            for slot_index in forced_detail_slot_indices:
                key = (group_index, slot_index + 1)
                self.port._zero_card_detail_candidates[key] = drift_title_candidates[slot_index]
                self.port._zero_card_identity_fusion_diagnostics[key] = {
                    "group_index": group_index,
                    "slot": slot_index + 1,
                    "plan": plan,
                    "missing_catalog_reference_ids": list(forward_drift_ids),
                    "detail_candidates": list(drift_title_candidates[slot_index]),
                    "fallback_slot_indices": list(drift_slot_indices),
                    "mode": "active_catalog_forward_drift_fixed_slots_exact_title",
                }
        if reference_slot_indices:
            try:
                zero_reference_ids.update(
                    self.port._resolve_zero_card_reference_ids(
                        group_index,
                        reference_slot_indices,
                        plan=plan,
                    )
                )
            except BadgeReferenceError as first_error:
                # A geometry-complete frame generation can still become stale
                # between badge and identity phases.  Re-acquire the complete
                # visible generation once; never vote or introduce an alias.
                try:
                    self.port._refresh_visible_card_groups(group_index)
                    self.port._increment("skill_card_identity_generation_refreshes")
                    zero_reference_ids.update(
                        self.port._resolve_zero_card_reference_ids(
                            group_index,
                            reference_slot_indices,
                            plan=plan,
                        )
                    )
                except (ArenaReaderError, BadgeReferenceError) as retry_error:
                    raise ArenaReaderError(
                        "skill_card_reference_identity_ambiguous",
                        f"group {group_index} initial generation failed: {first_error}; fresh generation failed: {retry_error}",
                    ) from retry_error
                row = self.port._card_rows.get(group_index, ())
                image = self.port._card_images.get(group_index)
                if len(row) != 6 or image is None:
                    raise ArenaReaderError(
                        "skill_card_identity_input_mismatch",
                        f"group {group_index} lost its frame after identity refresh",
                    )

        for index, (box, customization_count) in enumerate(
            zip(row, customization_counts, strict=True),
            start=1,
        ):
            if excluded_duplicate_flags[index - 1] or empty_flags[index - 1]:
                self.port._card_candidate_groups.pop((group_index, index), None)
                self.port._card_predictions[(group_index, index)] = 0
                predictions.append(0)
                continue
            key = (group_index, index)
            inferred = self.port._inferred_clicked_cards.get(key)
            if inferred is not None:
                if ordinary_drift and index - 1 in drift_slot_indices:
                    self.port._increment("skill_card_gallery_fallback_cached_detail_reuses")
                inferred_count = sum(int(value) for value in inferred.customizations.values())
                if inferred_count != customization_count:
                    raise ArenaReaderError(
                        "skill_card_inferred_count_changed",
                        f"group {group_index}/slot {index} inferred {inferred_count} but identity read requested {customization_count}",
                    )
                self.port._card_candidate_groups[key] = (inferred.card_id,)
                self.port._card_predictions[key] = inferred.card_id
                predictions.append(inferred.card_id)
                continue
            customized = customization_count > 0
            if customized:
                # The arena custom-card embedding supplies only bounded visual-family candidates.  A
                # positive card's mandatory detail title remains authoritative.
                recognizer, acceptance_threshold, minimum_margin = self.port._card_recognizer(customized=True)
                prediction = self.port._classify_card_candidate(
                    recognizer,
                    image,
                    CandidateBox(
                        slot=index - 1,
                        box=box,
                        detector_label="cards",
                        detector_score=1.0,
                    ),
                    frame_id=frame_identifier(image, index),
                    plan=plan,
                    acceptance_threshold=acceptance_threshold,
                    min_margin=minimum_margin,
                    resolve_upgrade_state=False,
                    **self.port._skill_card_scope_kwargs(index - 1, plan=plan),
                )
                candidate_ids = (
                    tuple(dict.fromkeys(int(value) for visual_family in prediction.top_k_card_ids for value in visual_family))
                    if prediction.accepted
                    else ()
                )
                structural_drift_slot = index - 1 in drift_slot_indices
                if structural_drift_slot:
                    # Positive cards are clicked later regardless of their
                    # closed-set prediction.  Keep that mandatory title check
                    # authoritative over the complete active-plan catalog: a
                    # high-confidence old-gallery hit cannot exclude a newly
                    # added card in any of the affected deck positions.
                    candidate_ids = drift_title_candidates[index - 1]
                    if prediction.accepted and prediction.card_id.isdigit() and not ordinary_drift:
                        resolved_card_id = int(prediction.card_id)
                    else:

                        def infer_positive():
                            return self.port._infer_badge_card_from_detail(
                                target,
                                stage_number,
                                member_slot,
                                group_index,
                                index,
                                image,
                                box,
                                customization_count,
                                "catalog_drift_positive_identity_transaction",
                                candidate_ids_override=drift_title_candidates[index - 1],
                            )

                        inferred = (
                            self.port._run_skill_card_gallery_detail(
                                infer_positive,
                                target=target,
                                stage_number=stage_number,
                                member_slot=member_slot,
                                group_index=group_index,
                                card_slot=index,
                            )
                            if ordinary_drift
                            else infer_positive()
                        )
                        resolved_card_id = inferred.card_id
                        candidate_ids = (resolved_card_id,)
                        self.port._increment("skill_card_catalog_drift_positive_detail_confirmations")
                else:
                    if not prediction.accepted or not prediction.card_id.isdigit() or not candidate_ids:
                        if prediction.accepted and prediction.card_id.isdigit():
                            raise ArenaReaderError(
                                "skill_card_icon_unknown",
                                f"group {group_index}/slot {index} has no visual-family candidates",
                            )
                        raise ArenaReaderError(
                            "skill_card_icon_unknown",
                            f"group {group_index}/slot {index} was rejected: {prediction.reason}",
                        )
                    resolved_card_id = int(prediction.card_id)
            else:
                # Zero-card identities were resolved as one page generation
                # before the loop.  This prevents one ambiguous slot from
                # mixing fresh frames with already accepted sibling slots.
                resolved_card_id = zero_reference_ids[index - 1]
                if resolved_card_id:
                    candidate_ids = (resolved_card_id,)
                    self.port._increment("skill_card_exact_reference_resolutions")
                else:
                    candidate_ids = self.port._zero_card_detail_candidates.get(key, ())
                    if not candidate_ids:
                        raise ArenaReaderError(
                            "skill_card_zero_identity_candidates_missing",
                            f"group {group_index}/slot {index} has no bounded title candidates",
                        )

                    def infer_zero():
                        return self.port._infer_zero_card_identity_from_detail(
                            target,
                            stage_number,
                            member_slot,
                            group_index,
                            index,
                            candidate_ids,
                        )

                    inferred = (
                        self.port._run_skill_card_gallery_detail(
                            infer_zero,
                            target=target,
                            stage_number=stage_number,
                            member_slot=member_slot,
                            group_index=group_index,
                            card_slot=index,
                            additional_zero_detail=True,
                        )
                        if ordinary_drift and index - 1 in drift_slot_indices
                        else infer_zero()
                    )
                    resolved_card_id = inferred.card_id
                    candidate_ids = (resolved_card_id,)
                    if forward_drift_ids:
                        self.port._increment("skill_card_catalog_drift_zero_detail_confirmations")
            self.port._card_candidate_groups[key] = candidate_ids
            self.port._card_predictions[key] = resolved_card_id
            predictions.append(resolved_card_id)
        if group_index == 1 and tuple(predictions) == tuple(self.port._card_predictions.get((0, index)) for index in range(1, 7)):
            raise ArenaReaderError(
                "skill_card_group_unchanged",
                "upper-detail swipe did not expose a distinct second six-card group",
            )
        return tuple(predictions)
