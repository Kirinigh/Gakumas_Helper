"""Independent badge-glyph exemplars and bounded raster-settling validation."""

from __future__ import annotations

from typing import Any, Protocol
from collections.abc import Sequence

from arena_winrate import TeamTarget, ArenaReaderError, ClickedSkillCard


class BadgeCalibrationPort(Protocol):
    """Live reader state and callbacks used by this component; never copied."""

    @staticmethod
    def _badge_glyph_comparison_metrics(target: tuple[int, ...], sample: tuple[int, ...]) -> dict[str, int | float]: ...

    _badge_glyph_count_diagnostics: Any

    @staticmethod
    def _badge_glyph_descriptor_sha256(descriptor: Sequence[int]) -> str: ...

    _badge_glyph_domain: Any

    _badge_glyph_exemplars: Any

    _badge_glyph_observations: Any

    _badge_glyph_polluted_descriptors: Any

    _badge_glyph_pool_size: Any

    _badge_glyph_runtime_labels: Any

    _badge_glyph_task_pool: Any

    def _increment(self, name: str) -> None: ...

    _runtime_badge_glyph_exemplar_diagnostics: Any

    @staticmethod
    def _stable_badge_glyph_exemplar_signature(
        observations: Sequence[dict[str, Any]],
    ) -> tuple[tuple[int, ...], tuple[int, int, int, int, int], int] | None: ...

    def _validate_authoritative_badge_glyph_runtime_transition(
        self, key: tuple[int, int], descriptor: tuple[int, ...], *, count: int
    ) -> None: ...

    def _validate_badge_glyph_tail_outlier(
        self, key: tuple[int, int], *, count: int, support_frames: int, detail_confirmed: bool = ...
    ) -> tuple[int, ...] | None: ...


def _stable_badge_glyph_exemplar_signature(
    observations: Sequence[dict[str, Any]],
) -> (
    tuple[
        tuple[int, ...],
        tuple[int, int, int, int, int],
        int,
    ]
    | None
):
    """Return one exact tail-stable glyph plus its normalized geometry.

    This is stricter than the generic OCR fallback.  An exemplar is valid
    only when every frozen source frame contains exactly one component and
    the final two chronological frames have byte-identical descriptors and
    normalized geometry.  The first frame may be a narrowly bounded
    contour-rasterization outlier with the same component box; an unordered
    two-of-three vote is never accepted.  No frames are averaged and no
    nearest-neighbour vote is used.  Position is omitted from the returned
    signature: the source-frame domain fixes capture dimensions, while a
    complete rendered digit may occupy another card slot or member. The
    same-frame plate and component checks still bind each source locally.
    """

    if len(observations) != 3:
        return None
    descriptors: list[tuple[int, ...]] = []
    geometries: list[tuple[int, int, int, int, int]] = []
    component_boxes: list[tuple[int, int, int, int]] = []
    badge_centers: list[tuple[int, int]] = []
    for observation in observations:
        internal = observation.get("internal_features", {})
        if not isinstance(internal, dict):
            return None
        descriptor = internal.get("glyph_descriptor_12x18_q4")
        normalization_size = internal.get("normalization_size")
        badge_center = internal.get("badge_center")
        component_box = internal.get("glyph_component_box")
        component_area = internal.get("glyph_component_area")
        if (
            internal.get("status") != "MEASURED"
            or internal.get("seeded_plate_candidate") is not True
            or internal.get("glyph_non_badge_art_candidate") is not False
            or type(internal.get("glyph_component_count")) is not int
            or internal.get("glyph_component_count") != 1
            or not isinstance(descriptor, (list, tuple))
            or len(descriptor) != 216
            or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15 for value in descriptor)
            or not isinstance(normalization_size, (list, tuple))
            or len(normalization_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in normalization_size)
            or not isinstance(badge_center, (list, tuple))
            or len(badge_center) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in badge_center)
            or not isinstance(component_box, (list, tuple))
            or len(component_box) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in component_box)
            or isinstance(component_area, bool)
            or not isinstance(component_area, int)
        ):
            return None
        x, y, width, height = component_box
        badge_x, badge_y = badge_center
        if (
            x < 0
            or y < 0
            or width < 1
            or height < 1
            or component_area < 1
            or component_area > width * height
            or x + width > normalization_size[0]
            or y + height > normalization_size[1]
            or badge_x < 0
            or badge_y < 0
            or badge_x >= normalization_size[0]
            or badge_y >= normalization_size[1]
        ):
            return None
        descriptors.append(tuple(descriptor))
        component_boxes.append((x, y, width, height))
        badge_centers.append((badge_x, badge_y))
        geometries.append(
            (
                int(normalization_size[0]),
                int(normalization_size[1]),
                int(width),
                int(height),
                component_area,
            )
        )
    signatures = list(zip(descriptors, geometries, strict=True))
    if len(set(component_boxes)) != 1 or len(set(badge_centers)) != 1:
        return None
    tail_signature = signatures[1]
    if signatures[2] != tail_signature:
        return None
    if signatures[0] == tail_signature:
        descriptor, geometry = tail_signature
        return descriptor, geometry, 3

    first_descriptor, first_geometry = signatures[0]
    tail_descriptor, tail_geometry = tail_signature
    if first_geometry[:4] != tail_geometry[:4] or abs(first_geometry[4] - tail_geometry[4]) > 2:
        return None
    differences = tuple(
        abs(first - tail)
        for first, tail in zip(
            first_descriptor,
            tail_descriptor,
            strict=True,
        )
    )
    first_foreground = tuple(value > 0 for value in first_descriptor)
    tail_foreground = tuple(value > 0 for value in tail_descriptor)
    foreground_intersection = sum(
        first and tail
        for first, tail in zip(
            first_foreground,
            tail_foreground,
            strict=True,
        )
    )
    foreground_union = sum(
        first or tail
        for first, tail in zip(
            first_foreground,
            tail_foreground,
            strict=True,
        )
    )
    first_subset_tail = all(
        (not first) or tail
        for first, tail in zip(
            first_foreground,
            tail_foreground,
            strict=True,
        )
    )
    tail_subset_first = all(
        (not tail) or first
        for first, tail in zip(
            first_foreground,
            tail_foreground,
            strict=True,
        )
    )
    area_delta = first_geometry[4] - tail_geometry[4]
    area_direction_matches = (
        (first_subset_tail and not tail_subset_first and area_delta < 0)
        or (tail_subset_first and not first_subset_tail and area_delta > 0)
        or (first_subset_tail and tail_subset_first and area_delta == 0)
    )
    if (
        sum(value > 0 for value in differences) > 8
        or sum(differences) > 120
        or sum(
            first != tail
            for first, tail in zip(
                first_foreground,
                tail_foreground,
                strict=True,
            )
        )
        > 8
        or foreground_union < 1
        or foreground_intersection * 100 < foreground_union * 94
        or not (first_subset_tail or tail_subset_first)
        or not area_direction_matches
    ):
        return None
    return tail_descriptor, tail_geometry, 2


def _badge_glyph_frames_are_bounded_raster_settling(
    cls: BadgeCalibrationPort,
    observations: Sequence[dict[str, Any]],
    *,
    BADGE_GLYPH_RASTER_SETTLING_AREA_DELTA_MAX: float,
    BADGE_GLYPH_RASTER_SETTLING_CHANGED_MAX: float,
    BADGE_GLYPH_RASTER_SETTLING_IOU_MIN: float,
    BADGE_GLYPH_RASTER_SETTLING_L1_MAX: float,
    BADGE_GLYPH_RASTER_SETTLING_XOR_MAX: float,
) -> bool:
    """Allow independent OCR across one narrowly settling glyph contour.

    This is deliberately weaker than an exemplar signature and therefore
    returns only a boolean.  All three frames still go through multi-view
    OCR and must independently choose the same validated digit.  An
    unordered majority, averaged descriptor, nearest-neighbour label, or
    changing component geometry remains ineligible.
    """

    if len(observations) != 3:
        return False
    descriptors: list[tuple[int, ...]] = []
    normalization_sizes: list[tuple[int, int]] = []
    badge_centers: list[tuple[int, int]] = []
    component_boxes: list[tuple[int, int, int, int]] = []
    component_areas: list[int] = []
    for observation in observations:
        internal = observation.get("internal_features", {})
        if not isinstance(internal, dict):
            return False
        descriptor = internal.get("glyph_descriptor_12x18_q4")
        normalization_size = internal.get("normalization_size")
        badge_center = internal.get("badge_center")
        component_box = internal.get("glyph_component_box")
        component_area = internal.get("glyph_component_area")
        if (
            internal.get("status") != "MEASURED"
            or internal.get("seeded_plate_candidate") is not True
            or internal.get("glyph_non_badge_art_candidate") is not False
            or type(internal.get("glyph_component_count")) is not int
            or internal.get("glyph_component_count") != 1
            or not isinstance(descriptor, (list, tuple))
            or len(descriptor) != 216
            or any(type(value) is not int or not 0 <= value <= 15 for value in descriptor)
            or not isinstance(normalization_size, (list, tuple))
            or len(normalization_size) != 2
            or any(type(value) is not int or value < 1 for value in normalization_size)
            or not isinstance(badge_center, (list, tuple))
            or len(badge_center) != 2
            or any(type(value) is not int for value in badge_center)
            or not isinstance(component_box, (list, tuple))
            or len(component_box) != 4
            or any(type(value) is not int for value in component_box)
            or type(component_area) is not int
        ):
            return False
        x, y, width, height = component_box
        badge_x, badge_y = badge_center
        if (
            x < 0
            or y < 0
            or width < 1
            or height < 1
            or component_area < 1
            or component_area > width * height
            or x + width > normalization_size[0]
            or y + height > normalization_size[1]
            or not 0 <= badge_x < normalization_size[0]
            or not 0 <= badge_y < normalization_size[1]
        ):
            return False
        descriptors.append(tuple(descriptor))
        normalization_sizes.append(tuple(normalization_size))
        badge_centers.append(tuple(badge_center))
        component_boxes.append(tuple(component_box))
        component_areas.append(component_area)

    if (
        len(set(normalization_sizes)) != 1
        or len(set(badge_centers)) != 1
        or len(set(component_boxes)) != 1
        or max(component_areas) - min(component_areas) > BADGE_GLYPH_RASTER_SETTLING_AREA_DELTA_MAX
    ):
        return False

    foregrounds = tuple(tuple(value > 0 for value in descriptor) for descriptor in descriptors)
    contour_adds = all(
        all((not before) or after for before, after in zip(left, right, strict=True))
        for left, right in zip(foregrounds[:-1], foregrounds[1:], strict=True)
    )
    contour_removes = all(
        all((not after) or before for before, after in zip(left, right, strict=True))
        for left, right in zip(foregrounds[:-1], foregrounds[1:], strict=True)
    )
    area_increases = all(
        left <= right
        for left, right in zip(
            component_areas[:-1],
            component_areas[1:],
            strict=True,
        )
    )
    area_decreases = all(
        left >= right
        for left, right in zip(
            component_areas[:-1],
            component_areas[1:],
            strict=True,
        )
    )
    if not ((contour_adds and area_increases) or (contour_removes and area_decreases)):
        return False

    for left_index, right_index in ((0, 1), (1, 2), (0, 2)):
        metrics = cls._badge_glyph_comparison_metrics(
            descriptors[left_index],
            descriptors[right_index],
        )
        if (
            metrics["l1_sum"] > BADGE_GLYPH_RASTER_SETTLING_L1_MAX
            or metrics["q4_changed_cells"] > BADGE_GLYPH_RASTER_SETTLING_CHANGED_MAX
            or metrics["binary_xor_cells"] > BADGE_GLYPH_RASTER_SETTLING_XOR_MAX
            or metrics["foreground_iou"] < BADGE_GLYPH_RASTER_SETTLING_IOU_MIN
        ):
            return False
    return True


class BadgeCalibration:
    """Stateless operation component over the reader's explicit port."""

    def __init__(self, port: BadgeCalibrationPort) -> None:
        self.port = port

    def _maybe_register_badge_glyph_exemplar(
        self,
        key: tuple[int, int],
        resolved: ClickedSkillCard,
        *,
        target: TeamTarget | None = None,
        stage_number: int | None = None,
        member_slot: int | None = None,
    ) -> bool:
        """Learn a count glyph only from fresh, detail-independent truth.

        Badge-constrained, auxiliary-glyph, card-face cost and conservative
        results are intentionally ineligible: allowing any of them to teach
        this table would make the fallback self-authenticating.  The narrow
        source/mode contract below currently admits only the unconstrained
        positive detail result after its required fresh semantic confirmation.
        """

        source = resolved.resolution_source
        reads = resolved.detail_confirmation_reads
        count = sum(int(value) for value in resolved.customizations.values())
        task_pool = getattr(self.port, "_badge_glyph_task_pool", None)
        if task_pool is not None and not task_pool.active:
            return False
        if (
            resolved.detail_evidence_mode != "positive_unique"
            or isinstance(reads, bool)
            or not isinstance(reads, int)
            or reads < 2
            or not source.startswith("detail_unconstrained")
            or "positive_confirmed" not in source
            or any(
                forbidden in source
                for forbidden in (
                    "auxiliary_badge_glyph",
                    "badge_constrained",
                    "generic_cost",
                    "card_face",
                )
            )
            or not 1 <= count <= 9
        ):
            return False
        signature = self.port._stable_badge_glyph_exemplar_signature(getattr(self.port, "_badge_glyph_observations", {}).get(key, ()))
        if signature is None:
            return False
        descriptor, geometry, support_frames = signature
        exemplars = getattr(self.port, "_badge_glyph_exemplars", None)
        if exemplars is None:
            exemplars = {}
            self.port._badge_glyph_exemplars = exemplars
        polluted = getattr(self.port, "_badge_glyph_polluted_descriptors", None)
        if polluted is None:
            polluted = set()
            self.port._badge_glyph_polluted_descriptors = polluted
        runtime_diagnostics = getattr(
            self.port,
            "_runtime_badge_glyph_exemplar_diagnostics",
            None,
        )
        if runtime_diagnostics is None:
            runtime_diagnostics = []
            self.port._runtime_badge_glyph_exemplar_diagnostics = runtime_diagnostics
        diagnostic = {
            "side": (None if target is None else ("own" if target.is_own_team else "opponent")),
            "team_id": None if target is None else target.team_id,
            "opponent_position": (None if target is None else target.opponent_position),
            "stage_number": stage_number,
            "member_slot": member_slot,
            "group_index": key[0],
            "card_slot": key[1],
            "card_id": resolved.card_id,
            "count": count,
            "normalization_size": [geometry[0], geometry[1]],
            "component_size": [geometry[2], geometry[3]],
            "component_area": geometry[4],
            "descriptor_support_frames": support_frames,
            "descriptor_sha256": self.port._badge_glyph_descriptor_sha256(descriptor),
            "resolution_source": resolved.resolution_source,
            "detail_evidence_mode": resolved.detail_evidence_mode,
            "detail_confirmation_reads": resolved.detail_confirmation_reads,
        }
        if getattr(self.port, "_badge_glyph_pool_size", None) is not None:
            diagnostic["capture_size"] = list(self.port._badge_glyph_pool_size)
            diagnostic["task_id"] = self.port._badge_glyph_task_pool.task_id
        if descriptor in polluted:
            # Quarantine applies to reuse, not to independently confirmed
            # detail truth. Still reject any contradiction of a prior inference.
            self.port._validate_authoritative_badge_glyph_runtime_transition(
                key,
                descriptor,
                count=count,
            )
            getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
            self.port._increment("skill_card_badge_glyph_quarantined_learning_skips")
            return False
        existing = exemplars.get(descriptor)
        runtime_labels = getattr(self.port, "_badge_glyph_runtime_labels", None)
        if runtime_labels is None:
            runtime_labels = {}
            self.port._badge_glyph_runtime_labels = runtime_labels
        if existing is not None and existing.get("count") != count:
            exemplars.pop(descriptor, None)
            polluted.add(descriptor)
            # Keep consumed labels: a later detail retry must not erase a
            # contradiction merely because the descriptor is quarantined.
            if existing.get("used_for_inference"):
                runtime_labels[descriptor] = existing["count"]
                pool_domain = getattr(self.port, "_badge_glyph_domain", None)
                if pool_domain is not None:
                    pool_domain.consumed_labels[descriptor] = existing["count"]
            # A prior ambiguous card may already have consumed this exemplar
            # and cached its count.  Invalidate every member-local glyph
            # decision before aborting so no bounded retry can reuse the
            # now-disproved label.
            getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
            self.port._increment("skill_card_badge_glyph_exemplar_collisions")
            runtime_diagnostics.append(
                {
                    **diagnostic,
                    "status": "quarantined_conflict",
                    "previous_count": existing.get("count"),
                    "previous_source": existing.get("source"),
                }
            )
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_collision",
                f"{key!r} authoritatively maps an already consumed exact glyph descriptor from count {existing.get('count')!r} to {count}",
            )
        self.port._validate_authoritative_badge_glyph_runtime_transition(
            key,
            descriptor,
            count=count,
        )
        outlier_descriptor = self.port._validate_badge_glyph_tail_outlier(
            key,
            count=count,
            support_frames=support_frames,
            detail_confirmed=True,
        )
        if outlier_descriptor is not None:
            # Detail taught only the exact stable tail. The unused first frame
            # is not a runtime count claim and must not veto later detail truth.
            diagnostic["excluded_first_frame_sha256"] = self.port._badge_glyph_descriptor_sha256(outlier_descriptor)
            self.port._increment("skill_card_badge_glyph_detail_outliers_excluded")
        if existing is None:
            pool_domain = getattr(self.port, "_badge_glyph_domain", None)
            if pool_domain is not None and not pool_domain.admit(descriptor):
                self.port._increment("skill_card_badge_glyph_pool_capacity_skips")
                return False
            runtime_labels.pop(descriptor, None)
            if pool_domain is not None:
                pool_domain.consumed_labels.pop(descriptor, None)
            runtime_diagnostics.append(diagnostic)
            exemplars[descriptor] = {
                "count": count,
                "geometries": {geometry},
                "prototypes": {(geometry, support_frames)},
                "source": diagnostic,
            }
            self.port._increment("skill_card_badge_glyph_exemplars_registered")
            return True
        geometries = existing.get("geometries")
        prototype_provenance = existing.get("prototypes")
        if not isinstance(geometries, set) or not isinstance(
            prototype_provenance,
            set,
        ):
            exemplars.pop(descriptor, None)
            polluted.add(descriptor)
            runtime_labels.pop(descriptor, None)
            getattr(self.port, "_badge_glyph_count_diagnostics", {}).clear()
            self.port._increment("skill_card_badge_glyph_exemplar_collisions")
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_invalid",
                f"{key!r} exact glyph exemplar has invalid geometry provenance",
            )
        runtime_labels.pop(descriptor, None)
        pool_domain = getattr(self.port, "_badge_glyph_domain", None)
        if pool_domain is not None:
            pool_domain.consumed_labels.pop(descriptor, None)
        runtime_diagnostics.append(diagnostic)
        geometries.add(geometry)
        prototype_provenance.add((geometry, support_frames))
        return True
