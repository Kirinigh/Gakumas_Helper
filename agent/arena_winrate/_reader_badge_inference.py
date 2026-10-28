"""Infer badge counts only from independent detail-labelled image exemplars."""

from __future__ import annotations

from typing import Any, Protocol
from collections.abc import Sequence

from arena_winrate import ArenaReaderError


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

    _badge_glyph_polluted_descriptors: Any

    _badge_glyph_task_pool: Any

    def _increment(self, name: str) -> None: ...

    @staticmethod
    def _normalized_badge_glyph_domain(maximum_count: int, admissible_counts: Sequence[int] | None) -> tuple[int, ...]: ...

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

    def __init__(self, port: BadgeInferencePort, *, clock: Any) -> None:
        self.port = port
        self.clock = clock

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
    ) -> int:
        """Disambiguate with independent image truth; never recognize a digit.

        Diagnostic caches are not evidence: old numeric OCR entries and prior
        lookup successes must both pass the current exemplar/conflict checks.
        The existing pure-image, three-frame non-badge zero remains admissible.
        """
        if isinstance(maximum_count, bool) or not isinstance(maximum_count, int) or not 1 <= maximum_count <= 9:
            raise ArenaReaderError(
                "skill_card_badge_glyph_domain_invalid",
                f"{key!r} has an unsupported single-glyph badge-count maximum: {maximum_count!r}",
            )
        domain = self.port._normalized_badge_glyph_domain(maximum_count, admissible_counts)
        observations = self.port._badge_glyph_observations.get(key, ())
        if len(observations) != 3:
            raise ArenaReaderError(
                "skill_card_badge_glyph_frames_missing",
                f"{key!r} has no stable three-frame badge glyph evidence",
            )
        started = self.clock.perf_counter()
        try:
            exemplar_count = self.port._badge_glyph_exemplar_count(
                key, maximum_count=maximum_count, admissible_counts=domain,
            )
            if exemplar_count is not None:
                signature = self.port._stable_badge_glyph_exemplar_signature(observations)
                if signature is None:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_exemplar_invalid",
                        f"{key!r} resolved without a stable exemplar signature",
                    )
                import hashlib

                descriptor, geometry, support_frames = signature
                exemplar = getattr(self.port, "_badge_glyph_exemplars", {}).get(descriptor)
                exact = exemplar is not None and geometry in exemplar.get("geometries", set())
                task_pool = getattr(self.port, "_badge_glyph_task_pool", None)
                scope = ("task" if task_pool.task_id is not None else "reader") if task_pool is not None else "same_member"
                self.port._badge_glyph_count_diagnostics[key] = {
                    "group_index": key[0], "slot": key[1], "maximum_count": maximum_count,
                    "admissible_counts": list(domain), "resolved_count": exemplar_count,
                    "exemplar_support_frames": support_frames,
                    "frame_digits": [exemplar_count if tuple(item.get("internal_features", {}).get("glyph_descriptor_12x18_q4", ())) == descriptor else None for item in observations],
                    "descriptor_sha256": [hashlib.sha256(bytes(item["internal_features"]["glyph_descriptor_12x18_q4"])).hexdigest() for item in observations],
                    "ocr_texts": [[], [], []],
                    "mode": f"detail_ambiguity_{scope}_{'exact_glyph_exemplar' if exact else 'calibrated_glyph'}",
                }
                if task_pool is not None:
                    self.port._badge_glyph_count_diagnostics[key].update(pool_scope=scope, task_id=task_pool.task_id)
                    self.port._increment("skill_card_badge_glyph_task_pool_resolutions" if scope == "task" else "skill_card_badge_glyph_reader_pool_resolutions")
                self.port._increment("skill_card_badge_glyph_pool_only_resolutions")
                self.port._increment("skill_card_badge_glyph_exemplar_resolutions" if exact else "skill_card_badge_glyph_calibrated_resolutions")
                return exemplar_count

            positive_descriptors = []
            zero_frames = 0
            for observation in observations:
                internal = observation.get("internal_features", {})
                descriptor = internal.get("glyph_descriptor_12x18_q4")
                count = internal.get("glyph_component_count")
                if type(count) is int and count == 0 and descriptor is None and internal.get("glyph_non_badge_art_candidate") is True:
                    zero_frames += 1
                elif (type(count) is int and count == 1 and isinstance(descriptor, (list, tuple))
                      and len(descriptor) == 216
                      and all(type(value) is int and 0 <= value <= 15 for value in descriptor)):
                    positive_descriptors.append(tuple(descriptor))
                else:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_invalid",
                        f"{key!r} lacks one strictly typed bounded glyph in every source frame",
                    )
            if zero_frames == 3:
                self.port._badge_glyph_count_diagnostics[key] = {
                    "group_index": key[0], "slot": key[1], "maximum_count": maximum_count,
                    "admissible_counts": list(domain), "resolved_count": 0,
                    "frame_digits": [0, 0, 0], "mode": "detail_ambiguity_same_plate_non_badge_art_zero",
                }
                self.port._increment("skill_card_badge_auxiliary_glyph_resolutions")
                return 0
            stable = self.port._stable_badge_glyph_exemplar_signature(observations)
            settling = stable is None and self.port._badge_glyph_frames_are_bounded_raster_settling(observations)
            if len(positive_descriptors) != 3 or (stable is None and not settling):
                raise ArenaReaderError(
                    "skill_card_badge_glyph_frame_disagreement",
                    f"{key!r} badge glyph frames do not form one stable bounded signature",
                )
            if any(value in getattr(self.port, "_badge_glyph_polluted_descriptors", set()) for value in positive_descriptors):
                self.port._badge_glyph_count_diagnostics.clear()
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_transition_conflict",
                    f"{key!r} glyph is already polluted",
                )
            raise ArenaReaderError(
                "skill_card_badge_glyph_ambiguous",
                f"{key!r} has no independently confirmed image exemplar; badge digit OCR was removed",
            )
        except ArenaReaderError as error:
            self.port._badge_glyph_count_diagnostics[key] = {
                "group_index": key[0], "slot": key[1], "maximum_count": maximum_count,
                "admissible_counts": list(domain), "resolved_count": None,
                "mode": "fail_closed", "error_code": error.code, "error_detail": error.detail,
            }
            raise
        finally:
            self.port._add_timing("skill_card_badge_auxiliary_glyph_images", self.clock.perf_counter() - started)
