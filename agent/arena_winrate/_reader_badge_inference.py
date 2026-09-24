"""Existing exemplar and auxiliary OCR count inference, using borrowed evidence."""

from __future__ import annotations

import re
from typing import Any, Protocol
from collections import Counter
from collections.abc import Sequence

from arena_winrate import ArenaReaderError
from arena_winrate._reader_visual import _text


class BadgeInferencePort(Protocol):
    """Live reader state and callbacks used by this component; never copied."""

    def _add_timing(self, name: str, elapsed: float) -> None: ...

    _badge_glyph_count_diagnostics: Any

    def _badge_glyph_exemplar_count(
        self, key: tuple[int, int], *, maximum_count: int, admissible_counts: Sequence[int] | None = ...
    ) -> int | None: ...

    _badge_glyph_exemplars: Any

    @classmethod
    def _badge_glyph_frames_are_bounded_raster_settling(cls, observations: Sequence[dict[str, Any]]) -> bool: ...

    def _badge_glyph_member_calibrated_count(
        self,
        key: tuple[int, int],
        descriptor: tuple[int, ...],
        geometry: tuple[int, int, int, int, int],
        *,
        support_frames: int,
        admissible_counts: tuple[int, ...],
    ) -> int | None: ...

    _badge_glyph_observations: Any

    @staticmethod
    def _badge_glyph_ocr_canvases(descriptor: Any) -> tuple[tuple[int, str, Any], ...]: ...

    _badge_glyph_polluted_descriptors: Any

    _badge_glyph_task_pool: Any

    def _increment(self, name: str) -> None: ...

    @staticmethod
    def _normalized_badge_glyph_domain(maximum_count: int, admissible_counts: Sequence[int] | None) -> tuple[int, ...]: ...

    def _ocr(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = ..., only_rec: bool = ...) -> list[Any]: ...

    def _record_badge_glyph_exemplar_comparisons(
        self, key: tuple[int, int], descriptor: tuple[int, ...], geometry: tuple[int, int, int, int, int], *, maximum_count: int
    ) -> None: ...

    def _record_badge_glyph_runtime_label(
        self, key: tuple[int, int], descriptor: tuple[int, ...] | None, *, count: int, evidence: str
    ) -> None: ...

    @staticmethod
    def _stable_badge_glyph_exemplar_signature(
        observations: Sequence[dict[str, Any]],
    ) -> tuple[tuple[int, ...], tuple[int, int, int, int, int], int] | None: ...

    def _validate_badge_glyph_runtime_label(
        self, key: tuple[int, int], descriptor: tuple[int, ...], *, count: int, evidence: str, check_nearby_samples: bool = ...
    ) -> None: ...

    def _validate_badge_glyph_tail_outlier(
        self, key: tuple[int, int], *, count: int, support_frames: int, detail_confirmed: bool = ...
    ) -> tuple[int, ...] | None: ...


class BadgeInference:
    """Stateless operation component over the reader's explicit port."""

    def __init__(self, port: BadgeInferencePort, *, clock: Any, validated_acceptance_counts: frozenset[int]) -> None:
        self.port = port
        self.clock = clock
        self.validated_acceptance_counts = validated_acceptance_counts

    def _badge_glyph_exemplar_count(
        self,
        key: tuple[int, int],
        *,
        maximum_count: int,
        admissible_counts: Sequence[int] | None = None,
    ) -> int | None:
        """Resolve exact or calibrated independent glyph evidence."""

        self.port._increment("skill_card_badge_glyph_pool_lookups")
        task_pool = getattr(self.port, "_badge_glyph_task_pool", None)
        if task_pool is not None and not task_pool.active:
            return None
        signature = self.port._stable_badge_glyph_exemplar_signature(getattr(self.port, "_badge_glyph_observations", {}).get(key, ()))
        if signature is None:
            return None
        descriptor, geometry, support_frames = signature
        domain = self.port._normalized_badge_glyph_domain(
            maximum_count,
            admissible_counts,
        )
        polluted = getattr(self.port, "_badge_glyph_polluted_descriptors", set())
        if descriptor in polluted:
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_polluted",
                f"{key!r} exact glyph descriptor was authoritatively mapped to multiple counts",
            )
        self.port._record_badge_glyph_exemplar_comparisons(
            key,
            descriptor,
            geometry,
            maximum_count=maximum_count,
        )
        exemplar = getattr(self.port, "_badge_glyph_exemplars", {}).get(descriptor)
        if exemplar is None or geometry not in exemplar.get("geometries", set()):
            return self.port._badge_glyph_member_calibrated_count(
                key,
                descriptor,
                geometry,
                support_frames=support_frames,
                admissible_counts=domain,
            )
        count = exemplar.get("count")
        if type(count) is not int or not 1 <= count <= 9:
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_invalid",
                f"{key!r} exact glyph exemplar has an invalid count",
            )
        if count not in domain:
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_domain_conflict",
                f"{key!r} exact glyph exemplar count {count} is outside the detail-compatible domain {domain!r}",
            )
        outlier_descriptor = self.port._validate_badge_glyph_tail_outlier(
            key,
            count=count,
            support_frames=support_frames,
        )
        self.port._record_badge_glyph_runtime_label(
            key,
            outlier_descriptor,
            count=count,
            evidence="bounded first-frame glyph",
        )
        # Exact known exemplars are not approximate inference claims. Remember
        # their use on the exemplar so a later collision can invalidate it.
        exemplar["used_for_inference"] = True
        return count

    def _auxiliary_badge_glyph_count(
        self,
        key: tuple[int, int],
        *,
        maximum_count: int,
        admissible_counts: Sequence[int] | None = None,
        allow_ocr: bool = True,
    ) -> int:
        """Use one stable count only after detail semantics remain non-unique.

        This never classifies badge presence. All three source frames must
        already be detail candidates, expose exactly one co-located glyph from
        the same green-plate interior, and independently agree through the
        fixed recognition-only multi-view vote.
        """

        if isinstance(maximum_count, bool) or not isinstance(maximum_count, int) or not 1 <= maximum_count <= 9:
            raise ArenaReaderError(
                "skill_card_badge_glyph_domain_invalid",
                f"{key!r} has an unsupported single-glyph badge-count maximum: {maximum_count!r}",
            )
        domain = self.port._normalized_badge_glyph_domain(
            maximum_count,
            admissible_counts,
        )
        cached = self.port._badge_glyph_count_diagnostics.get(key)
        if cached is not None and (cached.get("maximum_count") != maximum_count or tuple(cached.get("admissible_counts", ())) != domain):
            # Detail identity is authoritative over a pre-click visual family.
            # Re-evaluate the frozen plate glyph when the confirmed card changes
            # the only legal OCR domain instead of keeping a stale rejection.
            if allow_ocr:
                self.port._badge_glyph_count_diagnostics.pop(key, None)
            cached = None
        if cached is not None:
            count = cached.get("resolved_count")
            if type(count) is int and (count == 0 or count in domain):
                return count
            raise ArenaReaderError(
                "skill_card_badge_glyph_ambiguous",
                f"cached badge glyph evidence is not resolvable for {key!r}; diagnostic={cached!r}",
            )

        observations = self.port._badge_glyph_observations.get(key, ())
        if len(observations) != 3:
            raise ArenaReaderError(
                "skill_card_badge_glyph_frames_missing",
                f"{key!r} has no stable three-frame badge glyph evidence",
            )
        started = self.clock.perf_counter()
        frame_digits: list[int] = []
        descriptor_digests: list[str] = []
        variant_texts: list[list[str]] = []
        frame_vote_counts: list[dict[str, int]] = []
        frame_view_diagnostics: list[list[dict[str, Any]]] = []
        frame_paired_quiet_zones: list[list[int]] = []
        frame_winners: list[int | None] = []
        frame_resolution_modes: list[str] = []
        frame_isolated_out_of_domain_conflicts: list[int | None] = []
        stable_signature = None
        bounded_raster_settling = False
        try:
            import hashlib

            positive_descriptors: list[tuple[int, ...]] = []
            for observation in observations:
                internal = observation.get("internal_features", {})
                descriptor = internal.get("glyph_descriptor_12x18_q4")
                component_count = internal.get("glyph_component_count")
                if (
                    type(component_count) is int
                    and component_count == 1
                    and isinstance(descriptor, (list, tuple))
                    and len(descriptor) == 216
                    and all(not isinstance(value, bool) and isinstance(value, int) and 0 <= value <= 15 for value in descriptor)
                ):
                    positive_descriptors.append(tuple(descriptor))
            stable_signature = self.port._stable_badge_glyph_exemplar_signature(observations)
            bounded_raster_settling = stable_signature is None and self.port._badge_glyph_frames_are_bounded_raster_settling(observations)
            if bounded_raster_settling:
                self.port._increment("skill_card_badge_glyph_bounded_raster_settling")
            exemplar_count = self.port._badge_glyph_exemplar_count(
                key,
                maximum_count=maximum_count,
                admissible_counts=domain,
            )
            if exemplar_count is not None:
                signature = self.port._stable_badge_glyph_exemplar_signature(observations)
                if signature is None:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_exemplar_invalid",
                        f"{key!r} resolved without a stable exemplar signature",
                    )
                descriptor, _geometry, support_frames = signature
                exact_exemplar = getattr(
                    self.port,
                    "_badge_glyph_exemplars",
                    {},
                ).get(descriptor)
                exact_resolution = exact_exemplar is not None and _geometry in exact_exemplar.get("geometries", set())
                descriptor_digests = [hashlib.sha256(bytes(value)).hexdigest() for value in positive_descriptors]
                frame_digits = [exemplar_count if value == descriptor else None for value in positive_descriptors]
                variant_texts = [[], [], []]
                self.port._badge_glyph_count_diagnostics[key] = {
                    "group_index": key[0],
                    "slot": key[1],
                    "maximum_count": maximum_count,
                    "admissible_counts": list(domain),
                    "resolved_count": exemplar_count,
                    "frame_digits": list(frame_digits),
                    "descriptor_sha256": descriptor_digests,
                    "ocr_texts": variant_texts,
                    "exemplar_support_frames": support_frames,
                    "mode": (
                        "detail_ambiguity_same_member_exact_glyph_exemplar"
                        if exact_resolution
                        else "detail_ambiguity_same_member_calibrated_glyph"
                    ),
                }
                task_pool = getattr(self.port, "_badge_glyph_task_pool", None)
                if task_pool is not None:
                    task_scope = task_pool.task_id is not None
                    scope = "task" if task_scope else "reader"
                    self.port._badge_glyph_count_diagnostics[key]["pool_scope"] = scope
                    self.port._badge_glyph_count_diagnostics[key]["task_id"] = task_pool.task_id
                    self.port._badge_glyph_count_diagnostics[key]["mode"] = (
                        f"detail_ambiguity_{scope}_exact_glyph_exemplar" if exact_resolution else f"detail_ambiguity_{scope}_calibrated_glyph"
                    )
                    self.port._increment(
                        "skill_card_badge_glyph_task_pool_resolutions" if task_scope else "skill_card_badge_glyph_reader_pool_resolutions"
                    )
                    if allow_ocr:
                        self.port._increment("skill_card_badge_glyph_pool_ocr_fallbacks_avoided")
                self.port._increment(
                    "skill_card_badge_glyph_exemplar_resolutions" if exact_resolution else "skill_card_badge_glyph_calibrated_resolutions"
                )
                return exemplar_count

            if positive_descriptors and (len(positive_descriptors) != 3 or (stable_signature is None and not bounded_raster_settling)):
                raise ArenaReaderError(
                    "skill_card_badge_glyph_frame_disagreement",
                    f"{key!r} badge glyph frames do not form one stable bounded signature",
                )
            strict_descriptor = positive_descriptors[0] if len(positive_descriptors) == 3 and len(set(positive_descriptors)) == 1 else None
            polluted_descriptors = getattr(
                self.port,
                "_badge_glyph_polluted_descriptors",
                set(),
            )
            if any(descriptor in polluted_descriptors for descriptor in positive_descriptors):
                getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
                self.port._increment("skill_card_badge_glyph_exemplar_transition_conflicts")
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_transition_conflict",
                    f"{key!r} multi-view OCR glyph is already polluted",
                )

            if not allow_ocr:
                raise ArenaReaderError(
                    "skill_card_badge_glyph_cached_evidence_missing",
                    "error-only text recovery has no existing badge evidence; additional glyph OCR is outside its budget",
                )

            descriptor_votes: dict[
                tuple[int, ...],
                tuple[
                    int | None,
                    list[str],
                    dict[str, int],
                    list[dict[str, Any]],
                    list[int],
                    str,
                    int | None,
                ],
            ] = {}
            for observation in observations:
                internal = observation.get("internal_features", {})
                descriptor = internal.get("glyph_descriptor_12x18_q4")
                component_count = internal.get("glyph_component_count")
                if (
                    type(component_count) is int
                    and component_count == 0
                    and descriptor is None
                    and internal.get("glyph_non_badge_art_candidate") is True
                ):
                    frame_digits.append(0)
                    descriptor_digests.append("NON_BADGE_ART")
                    variant_texts.append([])
                    frame_vote_counts.append({})
                    frame_view_diagnostics.append([])
                    frame_paired_quiet_zones.append([])
                    frame_winners.append(0)
                    frame_resolution_modes.append("non_badge_art_zero")
                    frame_isolated_out_of_domain_conflicts.append(None)
                    continue
                if (
                    type(component_count) is not int
                    or component_count != 1
                    or not isinstance(descriptor, (list, tuple))
                    or len(descriptor) != 216
                    or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15 for value in descriptor)
                ):
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_invalid",
                        f"{key!r} lacks one strictly typed bounded glyph in every source frame",
                    )
                if stable_signature is None and not bounded_raster_settling:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_frame_disagreement",
                        f"{key!r} positive glyph frames lack one stable bounded signature",
                    )
                descriptor_tuple = tuple(descriptor)
                descriptor_digests.append(hashlib.sha256(bytes(descriptor_tuple)).hexdigest())
                cached_vote = descriptor_votes.get(descriptor_tuple)
                if cached_vote is None:
                    texts: list[str] = []
                    votes: Counter[int] = Counter()
                    view_diagnostics: list[dict[str, Any]] = []
                    pair_votes: dict[int, dict[str, int | None]] = {}
                    observed_digits: set[int] = set()
                    for quiet_cells, polarity, canvas in self.port._badge_glyph_ocr_canvases(descriptor_tuple):
                        # ``canvas`` is already one tightly bounded glyph, not
                        # a scene that needs text detection.  Recognition must
                        # cover the entire static glyph domain before the
                        # clicked card's legal domain is consulted; otherwise
                        # a stronger out-of-domain digit could be hidden by the
                        # filter and a weaker in-domain vote accepted.
                        observed = tuple(
                            _text(item).strip()
                            for item in self.port._ocr(
                                canvas,
                                r"^[1-9]$",
                                only_rec=True,
                            )
                        )
                        texts.extend(observed)
                        legal = {int(value) for value in observed if re.fullmatch(r"[1-9]", value) is not None}
                        observed_digits.update(legal)
                        view_digit = next(iter(legal)) if len(legal) == 1 else None
                        if view_digit is not None:
                            votes[view_digit] += 1
                        pair_votes.setdefault(quiet_cells, {})[polarity] = view_digit
                        view_diagnostics.append(
                            {
                                "quiet_cells": quiet_cells,
                                "polarity": polarity,
                                "inverted": polarity == "inverted",
                                "texts": list(observed),
                                "digits": sorted(legal),
                                "vote": view_digit,
                            }
                        )
                    sole_digit = next(iter(observed_digits)) if len(observed_digits) == 1 else None
                    paired_quiet_zones = [
                        quiet_cells
                        for quiet_cells, polarities in sorted(pair_votes.items())
                        if sole_digit is not None and polarities.get("normal") == sole_digit and polarities.get("inverted") == sole_digit
                    ]
                    voting_quiet_zones = {
                        quiet_cells
                        for quiet_cells, polarities in pair_votes.items()
                        if sole_digit is not None and sole_digit in polarities.values()
                    }
                    digit = sole_digit if sole_digit is not None and len(paired_quiet_zones) >= 2 else None
                    resolution_mode = "conflict_free_pairs" if digit is not None else "unresolved"
                    if (
                        digit is None
                        and sole_digit == 1
                        and stable_signature is not None
                        and stable_signature[2] == 3
                        and sole_digit in self.validated_acceptance_counts
                        and votes[sole_digit] >= 3
                        and len(voting_quiet_zones) >= 2
                        and len(paired_quiet_zones) >= 1
                    ):
                        # Empty OCR views are abstentions, not contradictory
                        # digits.  Permit the narrow one-pair shape only when
                        # all three independent source frames have byte-identical
                        # descriptor and geometry, every numeric view agrees on
                        # the sole independently validated count, and another
                        # quiet-zone transform supplies a third vote.  Tail-only
                        # or bounded-raster signatures retain the stricter
                        # two-pair requirement.
                        digit = sole_digit
                        resolution_mode = "stable_exact_one_pair_consensus"
                    isolated_out_of_domain_conflict = None
                    if digit is None:
                        # The narrowest canvas can add one polarity-specific
                        # OCR artifact even when both wider canvases form
                        # complete, matching polarity pairs.  Tolerate only
                        # that exact 5:1 shape: the winner must already be in
                        # the independently validated OCR acceptance domain,
                        # while the single competing glyph is neither a
                        # validated count nor compatible with the clicked
                        # card's detail-derived domain.  The full OCR domain
                        # remains visible above; the detail domain never
                        # filters votes or chooses the winner.
                        wide_pair_digits = [
                            pair_votes.get(quiet_cells, {}).get("normal")
                            for quiet_cells in (16, 24)
                            if pair_votes.get(quiet_cells, {}).get("normal") is not None
                            and pair_votes.get(quiet_cells, {}).get("normal") == pair_votes.get(quiet_cells, {}).get("inverted")
                        ]
                        narrow_normal = pair_votes.get(8, {}).get("normal")
                        narrow_inverted = pair_votes.get(8, {}).get("inverted")
                        wide_winner = wide_pair_digits[0] if len(wide_pair_digits) == 2 and len(set(wide_pair_digits)) == 1 else None
                        narrow_digits = {value for value in (narrow_normal, narrow_inverted) if value is not None}
                        conflicts = narrow_digits - {wide_winner}
                        if (
                            wide_winner in self.validated_acceptance_counts
                            and len(narrow_digits) == 2
                            and wide_winner in narrow_digits
                            and len(conflicts) == 1
                            and observed_digits == narrow_digits
                            and votes[wide_winner] == 5
                        ):
                            conflict = next(iter(conflicts))
                            if votes[conflict] == 1 and conflict not in domain and conflict not in self.validated_acceptance_counts:
                                digit = wide_winner
                                paired_quiet_zones = [16, 24]
                                resolution_mode = "isolated_out_of_domain_narrow_view_conflict"
                                isolated_out_of_domain_conflict = conflict
                    cached_vote = (
                        digit,
                        texts,
                        {str(value): count for value, count in sorted(votes.items())},
                        view_diagnostics,
                        paired_quiet_zones,
                        resolution_mode,
                        isolated_out_of_domain_conflict,
                    )
                    descriptor_votes[descriptor_tuple] = cached_vote
                (
                    digit,
                    texts,
                    vote_counts,
                    view_diagnostics,
                    paired_quiet_zones,
                    resolution_mode,
                    isolated_out_of_domain_conflict,
                ) = cached_vote
                variant_texts.append(list(texts))
                frame_vote_counts.append(dict(vote_counts))
                frame_view_diagnostics.append([dict(value) for value in view_diagnostics])
                frame_paired_quiet_zones.append(list(paired_quiet_zones))
                frame_winners.append(digit)
                frame_resolution_modes.append(resolution_mode)
                frame_isolated_out_of_domain_conflicts.append(isolated_out_of_domain_conflict)
                if digit is None:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_ambiguous",
                        f"{key!r} multi-view OCR did not produce one conflict-free "
                        "digit in two complete polarity pairs, the strict "
                        "stable three-frame one-pair shape, or the exact "
                        "validated isolated-conflict shape; "
                        f"views={view_diagnostics!r}, paired={paired_quiet_zones!r}",
                    )
                if digit not in domain:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_domain_conflict",
                        f"{key!r} multi-view OCR digit {digit} is outside the detail-compatible domain {domain!r}",
                    )
                frame_digits.append(digit)
            if len(set(frame_digits)) != 1:
                raise ArenaReaderError(
                    "skill_card_badge_glyph_frame_disagreement",
                    f"{key!r} badge glyph OCR disagreed across source frames: {frame_digits!r}",
                )
            count = frame_digits[0]
            if count > 0 and count not in self.validated_acceptance_counts:
                # An unvalidated digit can never be accepted or recorded,
                # but it must still be checked against already consumed
                # member truth so a direct contradiction is not hidden
                # behind the broader coverage stop.  Do this only after all
                # three frame winners agree, preserving the stronger temporal
                # disagreement diagnosis.
                for descriptor in dict.fromkeys(positive_descriptors):
                    self.port._validate_badge_glyph_runtime_label(
                        key,
                        descriptor,
                        count=count,
                        evidence="multi-view OCR glyph",
                    )
                raise ArenaReaderError(
                    "skill_card_badge_glyph_unvalidated_count",
                    f"{key!r} multi-view OCR digit {count} has no independent positive holdout coverage for this extractor",
                )
            if count > 0:
                if stable_signature is not None:
                    descriptor, _geometry, support_frames = stable_signature
                    outlier_descriptor = self.port._validate_badge_glyph_tail_outlier(
                        key,
                        count=count,
                        support_frames=support_frames,
                    )
                    self.port._record_badge_glyph_runtime_label(
                        key,
                        descriptor,
                        count=count,
                        evidence="multi-view OCR glyph",
                    )
                    self.port._record_badge_glyph_runtime_label(
                        key,
                        outlier_descriptor,
                        count=count,
                        evidence="multi-view OCR bounded first-frame glyph",
                    )
                elif strict_descriptor is not None:
                    self.port._record_badge_glyph_runtime_label(
                        key,
                        strict_descriptor,
                        count=count,
                        evidence="strict three-frame multi-view OCR glyph",
                    )
            if any(mode == "isolated_out_of_domain_narrow_view_conflict" for mode in frame_resolution_modes):
                self.port._increment("skill_card_badge_glyph_isolated_out_of_domain_resolutions")
            if any(mode == "stable_exact_one_pair_consensus" for mode in frame_resolution_modes):
                self.port._increment("skill_card_badge_glyph_stable_exact_one_pair_consensus_resolutions")
            self.port._badge_glyph_count_diagnostics[key] = {
                "group_index": key[0],
                "slot": key[1],
                "maximum_count": maximum_count,
                "admissible_counts": list(domain),
                "resolved_count": count,
                "frame_digits": list(frame_digits),
                "descriptor_sha256": descriptor_digests,
                "ocr_texts": variant_texts,
                "ocr_vote_domain": list(range(1, 10)),
                "ocr_validated_acceptance_domain": sorted(self.validated_acceptance_counts),
                "ocr_preprocess": ("q4-positive-binary/pad-8-16-24/cubic/h48/polarity-2/v3"),
                "frame_vote_counts": frame_vote_counts,
                "frame_views": frame_view_diagnostics,
                "frame_paired_quiet_zones": frame_paired_quiet_zones,
                "frame_winners": frame_winners,
                "frame_resolution_modes": frame_resolution_modes,
                "frame_isolated_out_of_domain_conflicts": (frame_isolated_out_of_domain_conflicts),
                "temporal_signature_mode": (
                    "bounded_monotone_raster_settling"
                    if bounded_raster_settling
                    else "stable_exemplar_signature"
                    if stable_signature is not None
                    else "none"
                ),
                "mode": ("detail_ambiguity_same_plate_non_badge_art_zero" if count == 0 else "detail_ambiguity_auxiliary_multiview_glyph_ocr"),
            }
            self.port._increment("skill_card_badge_auxiliary_glyph_resolutions")
            if count > 0:
                self.port._increment("skill_card_badge_auxiliary_glyph_ocr")
            return count
        except ArenaReaderError as error:
            if not allow_ocr:
                # A budget-limited cache probe is not a failed recognition.
                # Leave the original retry's glyph evidence untouched.
                raise
            self.port._badge_glyph_count_diagnostics[key] = {
                "group_index": key[0],
                "slot": key[1],
                "maximum_count": maximum_count,
                "admissible_counts": list(domain),
                "resolved_count": None,
                "frame_digits": list(frame_digits),
                "descriptor_sha256": descriptor_digests,
                "ocr_texts": variant_texts,
                "ocr_vote_domain": list(range(1, 10)),
                "ocr_validated_acceptance_domain": sorted(self.validated_acceptance_counts),
                "ocr_preprocess": ("q4-positive-binary/pad-8-16-24/cubic/h48/polarity-2/v3"),
                "frame_vote_counts": frame_vote_counts,
                "frame_views": frame_view_diagnostics,
                "frame_paired_quiet_zones": frame_paired_quiet_zones,
                "frame_winners": frame_winners,
                "frame_resolution_modes": frame_resolution_modes,
                "frame_isolated_out_of_domain_conflicts": (frame_isolated_out_of_domain_conflicts),
                "temporal_signature_mode": (
                    "bounded_monotone_raster_settling"
                    if bounded_raster_settling
                    else "stable_exemplar_signature"
                    if stable_signature is not None
                    else "none"
                ),
                "mode": "fail_closed",
                "error_code": error.code,
                "error_detail": error.detail,
            }
            raise
        finally:
            self.port._add_timing(
                "skill_card_badge_auxiliary_glyph_ocr",
                self.clock.perf_counter() - started,
            )
