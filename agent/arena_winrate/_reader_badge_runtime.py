"""Runtime badge labels, conflict vetoes and calibrated prototype matching."""

from __future__ import annotations

from typing import Any, Protocol
from collections.abc import Sequence

from arena_winrate import ArenaReaderError


class BadgeRuntimePort(Protocol):
    """Live reader state and callbacks used by this component; never copied."""

    @staticmethod
    def _badge_glyph_calibration_inner_match(
        target_geometry: tuple[int, int, int, int, int],
        sample_prototypes: set[tuple[tuple[int, int, int, int, int], int]],
        metrics: dict[str, int | float],
    ) -> bool: ...

    @staticmethod
    def _badge_glyph_calibration_outer_near(metrics: dict[str, int | float]) -> bool: ...

    @staticmethod
    def _badge_glyph_comparison_metrics(target: tuple[int, ...], sample: tuple[int, ...]) -> dict[str, int | float]: ...

    def _badge_glyph_consumed_claims(self, descriptor: tuple[int, ...], count: int): ...

    _badge_glyph_count_diagnostics: Any

    @staticmethod
    def _badge_glyph_descriptor_sha256(descriptor: Sequence[int]) -> str: ...

    _badge_glyph_domain: Any

    _badge_glyph_exemplars: Any

    _badge_glyph_observations: Any

    _badge_glyph_polluted_descriptors: Any

    _badge_glyph_runtime_labels: Any

    def _increment(self, name: str) -> None: ...

    def _record_badge_glyph_runtime_label(
        self, key: tuple[int, int], descriptor: tuple[int, ...] | None, *, count: int, evidence: str
    ) -> None: ...

    _runtime_badge_glyph_exemplar_comparisons: Any

    def _validate_badge_glyph_runtime_label(
        self, key: tuple[int, int], descriptor: tuple[int, ...], *, count: int, evidence: str, check_nearby_samples: bool = ...
    ) -> None: ...


def _badge_glyph_descriptor_sha256(
    descriptor: Sequence[int],
) -> str:
    """Return a stable identity without persisting the glyph grid."""

    import hashlib

    return hashlib.sha256(bytes(descriptor)).hexdigest()


def _badge_glyph_comparison_metrics(
    target: tuple[int, ...],
    sample: tuple[int, ...],
) -> dict[str, int | float]:
    differences = tuple(
        abs(target_value - sample_value)
        for target_value, sample_value in zip(
            target,
            sample,
            strict=True,
        )
    )
    target_foreground = tuple(value > 0 for value in target)
    sample_foreground = tuple(value > 0 for value in sample)
    foreground_intersection = sum(
        target_value and sample_value
        for target_value, sample_value in zip(
            target_foreground,
            sample_foreground,
            strict=True,
        )
    )
    foreground_union = sum(
        target_value or sample_value
        for target_value, sample_value in zip(
            target_foreground,
            sample_foreground,
            strict=True,
        )
    )
    return {
        "l1_sum": sum(differences),
        "mse": round(
            sum(value * value for value in differences) / len(differences),
            6,
        ),
        "q4_changed_cells": sum(value > 0 for value in differences),
        "binary_xor_cells": sum(
            target_value != sample_value
            for target_value, sample_value in zip(
                target_foreground,
                sample_foreground,
                strict=True,
            )
        ),
        "foreground_iou": round(
            foreground_intersection / max(1, foreground_union),
            6,
        ),
    }


def _badge_glyph_calibration_outer_near(
    metrics: dict[str, int | float],
    *,
    BADGE_GLYPH_CALIBRATED_OUTER_CHANGED_MAX: float,
    BADGE_GLYPH_CALIBRATED_OUTER_IOU_MIN: float,
    BADGE_GLYPH_CALIBRATED_OUTER_L1_MAX: float,
    BADGE_GLYPH_CALIBRATED_OUTER_MSE_MAX: float,
    BADGE_GLYPH_CALIBRATED_OUTER_XOR_MAX: float,
) -> bool:
    """Return whether another label is too close for calibrated transfer."""

    # Bitwise OR preserves the scalar predicate and applies the identical
    # thresholds to a batch of task-history comparisons.
    return (
        (metrics["l1_sum"] <= BADGE_GLYPH_CALIBRATED_OUTER_L1_MAX)
        | (metrics["mse"] <= BADGE_GLYPH_CALIBRATED_OUTER_MSE_MAX)
        | (metrics["q4_changed_cells"] <= BADGE_GLYPH_CALIBRATED_OUTER_CHANGED_MAX)
        | (metrics["binary_xor_cells"] <= BADGE_GLYPH_CALIBRATED_OUTER_XOR_MAX)
        | (metrics["foreground_iou"] >= BADGE_GLYPH_CALIBRATED_OUTER_IOU_MIN)
    )


def _badge_glyph_calibration_inner_match(
    target_geometry: tuple[int, int, int, int, int],
    sample_prototypes: set[tuple[tuple[int, int, int, int, int], int]],
    metrics: dict[str, int | float],
    *,
    BADGE_GLYPH_CALIBRATED_INNER_CHANGED_MAX: float,
    BADGE_GLYPH_CALIBRATED_INNER_IOU_MIN: float,
    BADGE_GLYPH_CALIBRATED_INNER_L1_MAX: float,
    BADGE_GLYPH_CALIBRATED_INNER_MSE_MAX: float,
    BADGE_GLYPH_CALIBRATED_INNER_XOR_MAX: float,
) -> bool:
    """Accept one prototype only inside the existing calibrated radius."""

    geometry_matches = any(
        target_geometry[:4] == sample_geometry[:4] and abs(target_geometry[4] - sample_geometry[4]) <= 2 and support_frames == 3
        for sample_geometry, support_frames in sample_prototypes
    )
    return (
        geometry_matches
        and metrics["l1_sum"] <= BADGE_GLYPH_CALIBRATED_INNER_L1_MAX
        and metrics["mse"] <= BADGE_GLYPH_CALIBRATED_INNER_MSE_MAX
        and metrics["q4_changed_cells"] <= BADGE_GLYPH_CALIBRATED_INNER_CHANGED_MAX
        and metrics["binary_xor_cells"] <= BADGE_GLYPH_CALIBRATED_INNER_XOR_MAX
        and metrics["foreground_iou"] >= BADGE_GLYPH_CALIBRATED_INNER_IOU_MIN
    )


def _normalized_badge_glyph_domain(
    maximum_count: int,
    admissible_counts: Sequence[int] | None,
) -> tuple[int, ...]:
    values = tuple(range(1, maximum_count + 1)) if admissible_counts is None else tuple(admissible_counts)
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum_count for value in values):
        raise ArenaReaderError(
            "skill_card_badge_glyph_domain_invalid",
            f"badge glyph admissible counts must be positive integers within 1..{maximum_count}: {values!r}",
        )
    return tuple(sorted(set(values)))


class BadgeRuntime:
    """Stateless operation component over the reader's explicit port."""

    def __init__(self, port: BadgeRuntimePort) -> None:
        self.port = port

    def _validate_badge_glyph_tail_outlier(
        self,
        key: tuple[int, int],
        *,
        count: int,
        support_frames: int,
        detail_confirmed: bool = False,
    ) -> tuple[int, ...] | None:
        """Check first-frame conflicts without turning detail into a glyph vote.

        An independently confirmed detail does not consume its first-frame
        outlier. Keep exact contradictions and checks against consumed runtime
        labels, but do not require this unused shape to be far from all samples.
        Runtime inference still requires the original full separation checks.
        """

        if support_frames == 3:
            return None
        observations = getattr(self.port, "_badge_glyph_observations", {}).get(
            key,
            (),
        )
        first_internal = observations[0]["internal_features"]
        first_descriptor = tuple(first_internal["glyph_descriptor_12x18_q4"])
        self.port._validate_badge_glyph_runtime_label(
            key,
            first_descriptor,
            count=count,
            evidence="bounded first-frame glyph",
            check_nearby_samples=not detail_confirmed,
        )
        return first_descriptor

    def _validate_authoritative_badge_glyph_runtime_transition(
        self,
        key: tuple[int, int],
        descriptor: tuple[int, ...],
        *,
        count: int,
    ) -> None:
        """Reject only prior runtime claims contradicted by fresh detail truth.

        Authoritative detail may legitimately teach two nearby glyphs with
        different counts.  That makes approximate reuse ineligible, but must
        not invalidate either independently confirmed exemplar.  Polluted and
        exemplar-neighbour checks therefore remain on runtime claims; this
        direction checks only whether an earlier runtime claim would be
        disproved by the new truth.
        """

        for runtime_descriptor, claimed_count in self.port._badge_glyph_consumed_claims(descriptor, count):
            if type(claimed_count) is not int or not 1 <= claimed_count <= 9:
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_invalid",
                    f"{key!r} authoritative glyph encountered an invalid runtime label",
                )
            if claimed_count == count:
                continue
            exact_conflict = runtime_descriptor == descriptor
            if not exact_conflict:
                metrics = self.port._badge_glyph_comparison_metrics(
                    descriptor,
                    runtime_descriptor,
                )
                if not self.port._badge_glyph_calibration_outer_near(metrics):
                    continue
            getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
            self.port._increment("skill_card_badge_glyph_exemplar_transition_conflicts")
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_transition_conflict",
                (
                    f"{key!r} authoritative glyph maps to count {count}, after a prior runtime label claimed count {claimed_count}"
                    if exact_conflict
                    else f"{key!r} authoritative glyph is not separated from prior runtime count {claimed_count}"
                ),
            )

    def _validate_badge_glyph_runtime_label(
        self,
        key: tuple[int, int],
        descriptor: tuple[int, ...],
        *,
        count: int,
        evidence: str,
        check_nearby_samples: bool = True,
    ) -> None:
        """Cross-check a runtime label against independent truth and prior use."""

        polluted = getattr(self.port, "_badge_glyph_polluted_descriptors", set())
        exemplar = getattr(self.port, "_badge_glyph_exemplars", {}).get(descriptor)
        if descriptor in polluted:
            getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
            self.port._increment("skill_card_badge_glyph_exemplar_transition_conflicts")
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_transition_conflict",
                f"{key!r} {evidence} is already polluted",
            )
        exemplar_count = None if exemplar is None else exemplar.get("count")
        if exemplar is not None and (type(exemplar_count) is not int or not 1 <= exemplar_count <= 9):
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_invalid",
                f"{key!r} {evidence} has an invalid exemplar count",
            )
        runtime_count = getattr(
            self.port,
            "_badge_glyph_runtime_labels",
            {},
        ).get(descriptor)
        pool_domain = getattr(self.port, "_badge_glyph_domain", None)
        if runtime_count is None and pool_domain is not None:
            runtime_count = pool_domain.consumed_labels.get(descriptor)
        conflicting_count = (
            exemplar_count
            if exemplar_count is not None and exemplar_count != count
            else (runtime_count if runtime_count is not None and runtime_count != count else None)
        )
        if conflicting_count is not None:
            getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
            self.port._increment("skill_card_badge_glyph_exemplar_transition_conflicts")
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_transition_conflict",
                f"{key!r} {evidence} maps to count {conflicting_count}, not count {count}",
            )
        for polluted_descriptor in polluted if check_nearby_samples else ():
            if polluted_descriptor == descriptor:
                continue
            metrics = self.port._badge_glyph_comparison_metrics(
                descriptor,
                polluted_descriptor,
            )
            if self.port._badge_glyph_calibration_outer_near(metrics):
                getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
                self.port._increment("skill_card_badge_glyph_exemplar_transition_conflicts")
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_transition_conflict",
                    f"{key!r} {evidence} is too close to a polluted member glyph",
                )
        if check_nearby_samples:
            for exemplar_descriptor, candidate in getattr(
                self.port,
                "_badge_glyph_exemplars",
                {},
            ).items():
                candidate_count = candidate.get("count")
                if type(candidate_count) is not int or not 1 <= candidate_count <= 9:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_exemplar_invalid",
                        f"{key!r} {evidence} encountered an invalid exemplar count",
                    )
                if candidate_count == count or exemplar_descriptor == descriptor:
                    continue
                metrics = self.port._badge_glyph_comparison_metrics(
                    descriptor,
                    exemplar_descriptor,
                )
                if self.port._badge_glyph_calibration_outer_near(metrics):
                    getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
                    self.port._increment("skill_card_badge_glyph_exemplar_transition_conflicts")
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_exemplar_transition_conflict",
                        f"{key!r} {evidence} is not separated from count {candidate_count} independent truth",
                    )
        for runtime_descriptor, candidate_count in self.port._badge_glyph_consumed_claims(descriptor, count):
            if type(candidate_count) is not int or not 1 <= candidate_count <= 9:
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_invalid",
                    f"{key!r} {evidence} encountered an invalid runtime label",
                )
            if candidate_count == count or runtime_descriptor == descriptor:
                continue
            metrics = self.port._badge_glyph_comparison_metrics(
                descriptor,
                runtime_descriptor,
            )
            if self.port._badge_glyph_calibration_outer_near(metrics):
                getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
                self.port._increment("skill_card_badge_glyph_exemplar_transition_conflicts")
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_transition_conflict",
                    f"{key!r} {evidence} is not separated from prior runtime count {candidate_count}",
                )

    def _record_badge_glyph_runtime_label(
        self,
        key: tuple[int, int],
        descriptor: tuple[int, ...] | None,
        *,
        count: int,
        evidence: str,
    ) -> None:
        if descriptor is None:
            return
        self.port._validate_badge_glyph_runtime_label(
            key,
            descriptor,
            count=count,
            evidence=evidence,
        )
        labels = getattr(self.port, "_badge_glyph_runtime_labels", None)
        if labels is None:
            labels = {}
            self.port._badge_glyph_runtime_labels = labels
        labels[descriptor] = count
        pool_domain = getattr(self.port, "_badge_glyph_domain", None)
        if pool_domain is not None:
            pool_domain.consumed_labels[descriptor] = count

    def _badge_glyph_consumed_claims(self, descriptor: tuple[int, ...], count: int):
        """Yield inference history only to contradiction guards, never lookup."""

        pool_domain = getattr(self.port, "_badge_glyph_domain", None)
        shared = {} if pool_domain is None else pool_domain.consumed_labels
        if shared:
            rows, metrics = pool_domain.consumed_comparisons(descriptor, count)
            if metrics is None:
                yield from rows
            else:
                nearby = self.port._badge_glyph_calibration_outer_near(metrics)
                for index, row in enumerate(rows):
                    if nearby[index]:
                        yield row
        for descriptor, count in getattr(self.port, "_badge_glyph_runtime_labels", {}).items():
            if shared.get(descriptor) != count:
                yield descriptor, count

    def _record_badge_glyph_exemplar_comparisons(
        self,
        key: tuple[int, int],
        descriptor: tuple[int, ...],
        geometry: tuple[int, int, int, int, int],
        *,
        maximum_count: int,
    ) -> None:
        """Persist only scalar distances from one target to active exemplars."""

        comparisons: list[dict[str, Any]] = []
        for exemplar_descriptor, exemplar in sorted(
            getattr(self.port, "_badge_glyph_exemplars", {}).items(),
            key=lambda item: (
                int(item[1].get("count", 0)),
                self.port._badge_glyph_descriptor_sha256(item[0]),
            ),
        ):
            metrics = self.port._badge_glyph_comparison_metrics(
                descriptor,
                exemplar_descriptor,
            )
            comparisons.append(
                {
                    "count": exemplar.get("count"),
                    "descriptor_sha256": self.port._badge_glyph_descriptor_sha256(exemplar_descriptor),
                    "geometries": [list(value) for value in sorted(exemplar.get("geometries", ()))],
                    **metrics,
                }
            )
        if not comparisons:
            return
        record = {
            "group_index": key[0],
            "card_slot": key[1],
            "maximum_count": maximum_count,
            "target_descriptor_sha256": self.port._badge_glyph_descriptor_sha256(descriptor),
            "target_geometry": list(geometry),
            "comparisons": comparisons,
        }
        runtime = getattr(
            self.port,
            "_runtime_badge_glyph_exemplar_comparisons",
            None,
        )
        if runtime is None:
            runtime = []
            self.port._runtime_badge_glyph_exemplar_comparisons = runtime
        if record not in runtime:
            runtime.append(record)

    def _badge_glyph_member_calibrated_count(
        self,
        key: tuple[int, int],
        descriptor: tuple[int, ...],
        geometry: tuple[int, int, int, int, int],
        *,
        support_frames: int,
        admissible_counts: tuple[int, ...],
    ) -> int | None:
        """Transfer a count only inside a fully covered capture-size domain.

        This is a bounded calibration, not a global nearest-neighbour model.
        The target must be strictly stable in all three source frames, every
        detail-compatible label must already have independent detail truth,
        one three-frame prototype must enter the inner radius, and every other
        label must remain outside the wider rejection guard.
        """

        pool_domain = getattr(self.port, "_badge_glyph_domain", None)
        if support_frames != 3 or (pool_domain is not None and pool_domain.saturated):
            return None
        exemplars = getattr(self.port, "_badge_glyph_exemplars", {})
        if not exemplars:
            return None
        polluted = getattr(self.port, "_badge_glyph_polluted_descriptors", set())
        for polluted_descriptor in polluted:
            metrics = self.port._badge_glyph_comparison_metrics(
                descriptor,
                polluted_descriptor,
            )
            if self.port._badge_glyph_calibration_outer_near(metrics):
                getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
                self.port._increment("skill_card_badge_glyph_calibration_conflicts")
                raise ArenaReaderError(
                    "skill_card_badge_glyph_calibration_ambiguous",
                    f"{key!r} calibrated glyph is too close to polluted evidence",
                )

        candidates: list[dict[str, Any]] = []
        covered_counts: set[int] = set()
        for sample_descriptor, exemplar in exemplars.items():
            count = exemplar.get("count")
            geometries = exemplar.get("geometries")
            sample_prototypes = exemplar.get("prototypes")
            if (
                type(count) is not int
                or not 1 <= count <= 9
                or not isinstance(geometries, set)
                or not geometries
                or not all(isinstance(value, tuple) and len(value) == 5 and all(type(item) is int for item in value) for value in geometries)
                or not isinstance(sample_prototypes, set)
                or not sample_prototypes
                or not all(
                    isinstance(value, tuple)
                    and len(value) == 2
                    and isinstance(value[0], tuple)
                    and len(value[0]) == 5
                    and all(type(item) is int for item in value[0])
                    and value[0] in geometries
                    and value[1] in {2, 3}
                    for value in sample_prototypes
                )
            ):
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_invalid",
                    f"{key!r} calibrated glyph encountered invalid exemplar provenance",
                )
            metrics = self.port._badge_glyph_comparison_metrics(
                descriptor,
                sample_descriptor,
            )
            if count in admissible_counts:
                covered_counts.add(count)
            candidates.append(
                {
                    "count": count,
                    "metrics": metrics,
                    "inner": (
                        count in admissible_counts
                        and self.port._badge_glyph_calibration_inner_match(
                            geometry,
                            sample_prototypes,
                            metrics,
                        )
                    ),
                }
            )
        if covered_counts != set(admissible_counts):
            return None

        inner_counts = {int(candidate["count"]) for candidate in candidates if candidate["inner"]}
        if not inner_counts:
            return None
        if len(inner_counts) != 1:
            getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
            self.port._increment("skill_card_badge_glyph_calibration_conflicts")
            raise ArenaReaderError(
                "skill_card_badge_glyph_calibration_ambiguous",
                f"{key!r} enters calibrated inner radii for multiple counts {sorted(inner_counts)!r}",
            )
        count = next(iter(inner_counts))
        competing = tuple(
            candidate
            for candidate in candidates
            if candidate["count"] != count and self.port._badge_glyph_calibration_outer_near(candidate["metrics"])
        )
        if competing:
            getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
            self.port._increment("skill_card_badge_glyph_calibration_conflicts")
            raise ArenaReaderError(
                "skill_card_badge_glyph_calibration_ambiguous",
                f"{key!r} has a competing independent label inside the outer guard",
            )
        self.port._record_badge_glyph_runtime_label(
            key,
            descriptor,
            count=count,
            evidence="independently calibrated glyph",
        )
        return count
