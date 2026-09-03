from __future__ import annotations

import json
import hashlib
import threading
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np


class PItemReferenceError(RuntimeError):
    """Raised when the fixed rendered-reference runtime is unavailable or invalid."""


@dataclass(frozen=True)
class PItemReferenceAccelerationProfile:
    minimum_frame_width: int
    minimum_frame_height: int
    canonical_slot_width: int
    canonical_slot_height: int
    coarse_delta: int
    candidate_limit: int
    acceptance_rank_limit: int


@dataclass(frozen=True)
class _PItemCoarseFineResult:
    ranking: tuple[PItemReferenceHit, ...]
    guard_passed: bool


@dataclass(frozen=True)
class PItemReferenceRuntime:
    source_color_order: str = "BGR"
    stable_frame_count: int = 3
    content_generation_max_mean_abs_error: float = 0.75
    empty_foreground_maximum: float = 0.05
    nonempty_foreground_minimum: float = 0.15
    complete_opposite_half_minimum: float = 0.05
    identity_minimum_similarity: float = 0.65
    identity_minimum_margin: float = 0.01
    maximum_detail_fallbacks_per_member: int = 2
    coarse_fine_profiles: tuple[PItemReferenceAccelerationProfile, ...] = ()

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "PItemReferenceRuntime":
        try:
            runtime = cls(
                source_color_order=str(value["source_color_order"]),
                stable_frame_count=int(value["stable_frame_count"]),
                content_generation_max_mean_abs_error=float(
                    value["content_generation_max_mean_abs_error"]
                ),
                empty_foreground_maximum=float(value["empty_foreground_maximum"]),
                nonempty_foreground_minimum=float(value["nonempty_foreground_minimum"]),
                complete_opposite_half_minimum=float(
                    value["complete_opposite_half_minimum"]
                ),
                identity_minimum_similarity=float(value["identity_minimum_similarity"]),
                identity_minimum_margin=float(value["identity_minimum_margin"]),
                maximum_detail_fallbacks_per_member=int(
                    value["maximum_detail_fallbacks_per_member"]
                ),
                coarse_fine_profiles=tuple(
                    PItemReferenceAccelerationProfile(
                        minimum_frame_width=int(item["minimum_capture_size"][0]),
                        minimum_frame_height=int(item["minimum_capture_size"][1]),
                        canonical_slot_width=int(item["canonical_slot_size"][0]),
                        canonical_slot_height=int(item["canonical_slot_size"][1]),
                        coarse_delta=int(item["coarse_delta"]),
                        candidate_limit=int(item["candidate_limit"]),
                        acceptance_rank_limit=int(item["acceptance_rank_limit"]),
                    )
                    for item in value["coarse_fine_profiles"]
                ),
            )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise PItemReferenceError("P-item reference runtime is incomplete") from error
        if runtime.source_color_order not in {"RGB", "BGR"}:
            raise PItemReferenceError("P-item source_color_order must be RGB or BGR")
        if runtime.stable_frame_count != 3:
            raise PItemReferenceError("P-item runtime requires exactly three stable frames")
        if not 0 <= runtime.content_generation_max_mean_abs_error <= 10:
            raise PItemReferenceError("P-item content-generation threshold is invalid")
        if not (
            0 <= runtime.empty_foreground_maximum
            < runtime.nonempty_foreground_minimum
            <= 1
        ):
            raise PItemReferenceError("P-item foreground thresholds are invalid")
        if not 0 <= runtime.complete_opposite_half_minimum <= 1:
            raise PItemReferenceError("P-item completeness threshold is invalid")
        if not -1 <= runtime.identity_minimum_similarity <= 1:
            raise PItemReferenceError("P-item identity similarity threshold is invalid")
        if not 0 <= runtime.identity_minimum_margin <= 2:
            raise PItemReferenceError("P-item identity margin threshold is invalid")
        if not 0 <= runtime.maximum_detail_fallbacks_per_member <= 4:
            raise PItemReferenceError("P-item detail fallback budget is invalid")
        signatures = tuple(
            (
                profile.minimum_frame_width,
                profile.minimum_frame_height,
                profile.canonical_slot_width,
                profile.canonical_slot_height,
            )
            for profile in runtime.coarse_fine_profiles
        )
        if len(signatures) != len(set(signatures)):
            raise PItemReferenceError("P-item acceleration profiles must be unique")
        if any(
            profile.minimum_frame_width < profile.canonical_slot_width
            or profile.minimum_frame_height < profile.canonical_slot_height
            or profile.canonical_slot_width < 8
            or profile.canonical_slot_height < 8
            or profile.coarse_delta not in {-6, -4, -2, 0, 2, 4, 6}
            or profile.canonical_slot_width + profile.coarse_delta < 8
            or profile.canonical_slot_height + profile.coarse_delta < 8
            or not 5 <= profile.candidate_limit <= 420
            or not 2
            <= profile.acceptance_rank_limit
            <= profile.candidate_limit
            for profile in runtime.coarse_fine_profiles
        ):
            raise PItemReferenceError("P-item acceleration profile is invalid")
        return runtime


@dataclass(frozen=True)
class PItemCompleteness:
    foreground_fraction: float
    minimum_opposite_half_fraction: float


@dataclass(frozen=True)
class PItemReferenceHit:
    p_item_id: int
    similarity: float
    size: tuple[int, int]
    offset: tuple[int, int]


@dataclass(frozen=True)
class PItemReferenceDecision:
    status: str
    p_item_id: int | None
    reason: str
    candidates: tuple[int, ...]
    similarity: float | None
    margin: float | None
    content_generation_max_mean_abs_error: float
    completeness: tuple[PItemCompleteness, ...]
    top_k: tuple[PItemReferenceHit, ...]
    frames_byte_identical: bool = False
    ranking_route: str = "full_rendered_reference"
    full_fallback_used: bool = False

    @property
    def accepted(self) -> bool:
        return self.status in {"ACCEPTED", "EMPTY"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validated_crop(
    image: Any,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    array = np.asarray(image)
    x, y, width, height = box
    if (
        array.ndim != 3
        or array.shape[2] not in {3, 4}
        or width < 8
        or height < 8
        or x < 0
        or y < 0
        or x + width > array.shape[1]
        or y + height > array.shape[0]
    ):
        raise PItemReferenceError("P-item ROI is outside the frame")
    return np.ascontiguousarray(array[y : y + height, x : x + width, :3])


def _content_signature(image: Any, box: tuple[int, int, int, int]) -> np.ndarray:
    import cv2

    crop = _validated_crop(image, box)
    return cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)


def p_item_content_generation_signatures(
    image: Any,
    boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[np.ndarray, ...]:
    """Return the exact fixed-slot signatures used by the generation metric."""

    if not boxes:
        raise PItemReferenceError("P-item content generation requires fixed slots")
    return tuple(_content_signature(image, box) for box in boxes)


def measure_p_item_content_generation_from_signatures(
    source_signatures: Sequence[np.ndarray],
    current_signatures: Sequence[np.ndarray],
) -> tuple[float, ...]:
    """Compare precomputed source/current signatures without changing the metric."""

    if not source_signatures or len(source_signatures) != len(current_signatures):
        raise PItemReferenceError(
            "P-item content signatures require equal non-empty fixed slots"
        )
    errors: list[float] = []
    for source_signature, current_signature in zip(
        source_signatures,
        current_signatures,
        strict=True,
    ):
        if source_signature.shape != current_signature.shape:
            raise PItemReferenceError(
                "P-item content signatures must use the same fixed shape"
            )
        # Keep the subtraction order, float32 arrays, NumPy mean, and Python
        # float conversion identical to ``measure_p_item_content_generation``.
        errors.append(
            float(np.mean(np.abs(current_signature - source_signature)))
        )
    return tuple(errors)


def measure_p_item_content_generation(
    images: Sequence[Any],
    boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[float, ...]:
    """Measure cross-frame rendering change for each fixed P-item slot."""

    if len(images) < 2:
        raise PItemReferenceError(
            "P-item content generation requires at least two frames"
        )
    if not boxes:
        raise PItemReferenceError("P-item content generation requires fixed slots")
    errors: list[float] = []
    for box in boxes:
        signatures = tuple(_content_signature(image, box) for image in images)
        errors.append(
            max(
                float(np.mean(np.abs(signature - signatures[0])))
                for signature in signatures[1:]
            )
        )
    return tuple(errors)


def measure_p_item_completeness(
    image: Any,
    box: tuple[int, int, int, int],
    *,
    source_color_order: str,
) -> PItemCompleteness:
    from PIL import Image

    from .preprocess import focus_p_item_art

    crop = _validated_crop(image, box)
    if source_color_order == "BGR":
        crop = crop[:, :, ::-1]
    focused = np.asarray(
        focus_p_item_art(Image.fromarray(crop, mode="RGB")),
        dtype=np.uint8,
    )
    foreground = focused.min(axis=2) < 245
    height, width = foreground.shape
    halves = (
        foreground[:, : width // 2],
        foreground[:, width // 2 :],
        foreground[: height // 2, :],
        foreground[height // 2 :, :],
    )
    return PItemCompleteness(
        foreground_fraction=float(foreground.mean()),
        minimum_opposite_half_fraction=min(float(mask.mean()) for mask in halves),
    )


class PItemRenderedReferenceGallery:
    """Exact rendered-domain P-item matcher with fail-closed completeness gates."""

    PLAN_AMBIGUITY = {
        "sense": 406,
        "logic": 407,
        "anomaly": 408,
    }
    PLAN_AMBIGUOUS_IDS = frozenset(PLAN_AMBIGUITY.values())

    def __init__(
        self,
        p_item_ids: Sequence[int],
        rendered_bgr: Any,
        *,
        runtime: PItemReferenceRuntime,
        gallery_sha256: str = "",
        provisional_p_item_ids: Sequence[int] = (),
    ) -> None:
        ids = tuple(int(value) for value in p_item_ids)
        if not ids or len(ids) != len(set(ids)) or any(value < 1 for value in ids):
            raise PItemReferenceError("P-item reference IDs must be unique positive integers")
        if isinstance(rendered_bgr, np.ndarray) and rendered_bgr.ndim == 4:
            source_images = tuple(rendered_bgr)
        else:
            source_images = tuple(rendered_bgr)
        images = tuple(
            np.ascontiguousarray(image, dtype=np.uint8) for image in source_images
        )
        if len(images) != len(ids) or any(
            image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] < 8
            or image.shape[1] < 8
            for image in images
        ):
            raise PItemReferenceError(
                "P-item rendered gallery must contain one decodable BGR image per ID"
            )
        self.p_item_ids = ids
        self.rendered_bgr = images
        self.runtime = runtime
        self.gallery_sha256 = gallery_sha256
        provisional_ids = tuple(dict.fromkeys(int(value) for value in provisional_p_item_ids))
        if any(value not in ids for value in provisional_ids):
            raise PItemReferenceError(
                "provisional P-item reference IDs must belong to the fixed gallery"
            )
        self.provisional_p_item_ids = provisional_ids
        self._id_to_index = {
            p_item_id: index for index, p_item_id in enumerate(self.p_item_ids)
        }
        self._resize_cache_lock = threading.Lock()
        self._resize_cache_signature: tuple[int, int] | None = None
        self._resize_cache: dict[int, tuple[np.ndarray, ...]] = {}

    @classmethod
    def load(cls, root: str | Path) -> "PItemRenderedReferenceGallery":
        directory = Path(root)
        try:
            manifest = json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8-sig")
            )
            if manifest["recognizer"] != "p_item_rendered_reference_v1":
                raise PItemReferenceError("unsupported P-item reference recognizer")
            gallery = manifest["reference_gallery"]
            gallery_path = directory / str(gallery["path"])
            expected_sha256 = str(gallery["sha256"]).upper()
            if _sha256_file(gallery_path) != expected_sha256:
                raise PItemReferenceError("P-item reference gallery SHA-256 mismatch")
            with np.load(gallery_path, allow_pickle=False) as arrays:
                p_item_ids = arrays["p_item_ids"]
                if {"png_bytes", "png_offsets"}.issubset(arrays.files):
                    import cv2

                    png_bytes = arrays["png_bytes"]
                    offsets = arrays["png_offsets"]
                    rendered_bgr = tuple(
                        cv2.imdecode(
                            np.ascontiguousarray(
                                png_bytes[int(offsets[index]) : int(offsets[index + 1])]
                            ),
                            cv2.IMREAD_COLOR,
                        )
                        for index in range(len(p_item_ids))
                    )
                else:
                    rendered_bgr = arrays["rendered_bgr"]
            runtime = PItemReferenceRuntime.from_manifest(manifest["runtime"])
            declared_count = int(gallery["business_id_count"])
            if declared_count != len(p_item_ids):
                raise PItemReferenceError(
                    "P-item reference manifest count does not match the gallery ID set"
                )
            source_count = manifest.get("source", {}).get("business_id_count")
            if source_count is not None and int(source_count) != declared_count:
                raise PItemReferenceError(
                    "P-item reference source count does not match the gallery count"
                )
            provisional_ids = tuple(gallery.get("provisional_business_ids", ()))
        except (OSError, KeyError, TypeError, ValueError) as error:
            if isinstance(error, PItemReferenceError):
                raise
            raise PItemReferenceError(
                f"P-item reference gallery is unavailable or invalid: {directory}"
            ) from error
        return cls(
            p_item_ids,
            rendered_bgr,
            runtime=runtime,
            gallery_sha256=expected_sha256,
            provisional_p_item_ids=provisional_ids,
        )

    def _rank(
        self,
        image: Any,
        box: tuple[int, int, int, int],
    ) -> tuple[PItemReferenceHit, ...]:
        profile = self._acceleration_profile(image, box)
        if profile is None:
            return self._rank_full(image, box)
        return self._rank_coarse_fine(image, box, profile=profile).ranking

    def _acceleration_profile(
        self,
        image: Any,
        box: tuple[int, int, int, int],
    ) -> PItemReferenceAccelerationProfile | None:
        array = np.asarray(image)
        if array.ndim < 2:
            return None
        frame_height, frame_width = array.shape[:2]
        width, height = box[2:]
        if width < 8 or height < 8 or abs(width - height) > max(2, round(width * 0.03)):
            return None

        def capture_size_is_eligible(
            profile: PItemReferenceAccelerationProfile,
        ) -> bool:
            minimum_width = profile.minimum_frame_width
            minimum_height = profile.minimum_frame_height
            if frame_width >= minimum_width and frame_height >= minimum_height:
                return True

            # MaaFW fixes the target short side and preserves the raw aspect
            # ratio.  Its integer resize can therefore turn a raw 721x1281
            # minimum-size window into a 720x1279 Agent frame.  Accept only
            # that one-pixel rounding deficit on the profile's long axis; the
            # short-axis minimum and the square-slot geometry above remain
            # strict, and a two-pixel deficit still fails closed.
            if minimum_height > minimum_width:
                return (
                    frame_width >= minimum_width
                    and frame_height == minimum_height - 1
                )
            if minimum_width > minimum_height:
                return (
                    frame_height >= minimum_height
                    and frame_width == minimum_width - 1
                )
            return False

        eligible = tuple(
            profile
            for profile in self.runtime.coarse_fine_profiles
            if capture_size_is_eligible(profile)
        )
        return max(
            eligible,
            key=lambda profile: (
                profile.minimum_frame_width * profile.minimum_frame_height,
                profile.canonical_slot_width * profile.canonical_slot_height,
            ),
            default=None,
        )

    def _resized_references(
        self,
        width: int,
        height: int,
        delta: int,
    ) -> tuple[np.ndarray, ...]:
        import cv2

        signature = (width, height)
        with self._resize_cache_lock:
            if self._resize_cache_signature != signature:
                self._resize_cache_signature = signature
                self._resize_cache.clear()
            cached = self._resize_cache.get(delta)
            if cached is None:
                candidate_width = width + delta
                candidate_height = height + delta
                if candidate_width < 8 or candidate_height < 8:
                    raise PItemReferenceError("P-item reference scale is invalid")
                cached = tuple(
                    cv2.resize(
                        rendered,
                        (candidate_width, candidate_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    for rendered in self.rendered_bgr
                )
                self._resize_cache[delta] = cached
            return cached

    def _rank_candidates(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        *,
        candidate_indexes: Sequence[int],
        deltas: Sequence[int],
        canonical_slot_size: tuple[int, int] | None = None,
    ) -> tuple[PItemReferenceHit, ...]:
        import cv2

        array = np.asarray(image)
        x, y, width, height = box
        _validated_crop(image, box)
        native_pad_x = max(4, width // 8)
        native_pad_y = max(4, height // 8)
        left = max(0, x - native_pad_x)
        top = max(0, y - native_pad_y)
        right = min(array.shape[1], x + width + native_pad_x)
        bottom = min(array.shape[0], y + height + native_pad_y)
        query = np.ascontiguousarray(array[top:bottom, left:right, :3], dtype=np.uint8)
        if self.runtime.source_color_order == "RGB":
            query = query[:, :, ::-1]
        if canonical_slot_size is None:
            reference_width, reference_height = width, height
        else:
            reference_width, reference_height = canonical_slot_size
            canonical_pad_x = max(4, reference_width // 8)
            canonical_pad_y = max(4, reference_height // 8)
            query = cv2.resize(
                query,
                (
                    reference_width + canonical_pad_x * 2,
                    reference_height + canonical_pad_y * 2,
                ),
                interpolation=cv2.INTER_AREA,
            )
        pad_x = max(4, reference_width // 8)
        pad_y = max(4, reference_height // 8)
        templates = {
            delta: self._resized_references(reference_width, reference_height, delta)
            for delta in deltas
        }
        rankings: list[PItemReferenceHit] = []
        for index in candidate_indexes:
            p_item_id = self.p_item_ids[index]
            best_similarity = -1.0
            best_size = (reference_width, reference_height)
            best_location = (pad_x, pad_y)
            for delta in deltas:
                candidate_width = reference_width + delta
                candidate_height = reference_height + delta
                candidate = templates[delta][index]
                if (
                    candidate.shape[0] > query.shape[0]
                    or candidate.shape[1] > query.shape[1]
                ):
                    continue
                response = cv2.matchTemplate(
                    query,
                    candidate,
                    cv2.TM_CCOEFF_NORMED,
                )
                _, similarity, _, location = cv2.minMaxLoc(response)
                if similarity > best_similarity:
                    best_similarity = float(similarity)
                    best_size = (candidate_width, candidate_height)
                    best_location = location
            rankings.append(
                PItemReferenceHit(
                    p_item_id=p_item_id,
                    similarity=best_similarity,
                    size=best_size,
                    offset=(
                        best_location[0] - pad_x,
                        best_location[1] - pad_y,
                    ),
                )
            )
        return tuple(
            sorted(rankings, key=lambda hit: (-hit.similarity, hit.p_item_id))
        )

    def _rank_full(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        *,
        profile: PItemReferenceAccelerationProfile | None = None,
    ) -> tuple[PItemReferenceHit, ...]:
        return self._rank_candidates(
            image,
            box,
            candidate_indexes=tuple(range(len(self.p_item_ids))),
            deltas=(-6, -4, -2, 0, 2, 4, 6),
            canonical_slot_size=(
                None
                if profile is None
                else (profile.canonical_slot_width, profile.canonical_slot_height)
            ),
        )

    def _rank_coarse_fine(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        *,
        profile: PItemReferenceAccelerationProfile,
    ) -> _PItemCoarseFineResult:
        coarse = self._rank_candidates(
            image,
            box,
            candidate_indexes=tuple(range(len(self.p_item_ids))),
            deltas=(profile.coarse_delta,),
            canonical_slot_size=(
                profile.canonical_slot_width,
                profile.canonical_slot_height,
            ),
        )
        candidate_ids = {
            hit.p_item_id for hit in coarse[: profile.candidate_limit]
        }
        if candidate_ids & self.PLAN_AMBIGUOUS_IDS:
            candidate_ids.update(self.PLAN_AMBIGUOUS_IDS)
        candidate_indexes = tuple(
            self._id_to_index[p_item_id]
            for p_item_id in candidate_ids
            if p_item_id in self._id_to_index
        )
        fine = self._rank_candidates(
            image,
            box,
            candidate_indexes=candidate_indexes,
            deltas=(-6, -4, -2, 0, 2, 4, 6),
            canonical_slot_size=(
                profile.canonical_slot_width,
                profile.canonical_slot_height,
            ),
        )
        coarse_ranks = {
            hit.p_item_id: index
            for index, hit in enumerate(coarse, start=1)
        }
        fine_top2_ranks = tuple(
            coarse_ranks.get(hit.p_item_id, profile.candidate_limit + 1)
            for hit in fine[:2]
        )
        guard_passed = (
            len(fine_top2_ranks) == 2
            and fine[0].p_item_id == coarse[0].p_item_id
            and max(fine_top2_ranks) <= profile.acceptance_rank_limit
        )
        return _PItemCoarseFineResult(
            ranking=fine,
            guard_passed=guard_passed,
        )

    def _effective_rankings(
        self,
        rankings: Sequence[PItemReferenceHit],
        *,
        plan: str,
    ) -> tuple[PItemReferenceHit, ...]:
        plan_id = self.PLAN_AMBIGUITY.get(plan)
        selected: dict[int, PItemReferenceHit] = {}
        for hit in rankings:
            if hit.p_item_id in self.PLAN_AMBIGUOUS_IDS:
                if plan_id is None:
                    continue
                effective_id = plan_id
            else:
                effective_id = hit.p_item_id
            effective = PItemReferenceHit(
                p_item_id=effective_id,
                similarity=hit.similarity,
                size=hit.size,
                offset=hit.offset,
            )
            existing = selected.get(effective_id)
            if existing is None or effective.similarity > existing.similarity:
                selected[effective_id] = effective
        return tuple(
            sorted(selected.values(), key=lambda hit: (-hit.similarity, hit.p_item_id))
        )

    def classify(
        self,
        images: Sequence[Any],
        box: tuple[int, int, int, int],
        *,
        plan: str,
    ) -> PItemReferenceDecision:
        if len(images) != self.runtime.stable_frame_count:
            raise PItemReferenceError(
                f"P-item identity requires {self.runtime.stable_frame_count} frames"
            )
        crops = tuple(_validated_crop(image, box) for image in images)
        byte_identical = all(np.array_equal(crops[0], crop) for crop in crops[1:])
        content_error = measure_p_item_content_generation(images, (box,))[0]
        completeness = tuple(
            measure_p_item_completeness(
                image,
                box,
                source_color_order=self.runtime.source_color_order,
            )
            for image in images
        )
        if content_error > self.runtime.content_generation_max_mean_abs_error:
            return PItemReferenceDecision(
                status="AMBIGUOUS",
                p_item_id=None,
                reason="content_generation_unstable",
                candidates=(),
                similarity=None,
                margin=None,
                content_generation_max_mean_abs_error=content_error,
                completeness=completeness,
                top_k=(),
                frames_byte_identical=byte_identical,
                ranking_route="not_ranked",
            )
        empty = tuple(
            item.foreground_fraction <= self.runtime.empty_foreground_maximum
            for item in completeness
        )
        if all(empty):
            return PItemReferenceDecision(
                status="EMPTY",
                p_item_id=0,
                reason="stable_empty_slot",
                candidates=(),
                similarity=None,
                margin=None,
                content_generation_max_mean_abs_error=content_error,
                completeness=completeness,
                top_k=(),
                frames_byte_identical=byte_identical,
                ranking_route="not_ranked",
            )
        if any(empty):
            reason = "presence_generation_unstable"
        elif any(
            item.foreground_fraction < self.runtime.nonempty_foreground_minimum
            or item.minimum_opposite_half_fraction
            < self.runtime.complete_opposite_half_minimum
            for item in completeness
        ):
            reason = "incomplete_or_non_item_roi"
        else:
            reason = ""
        if reason:
            return PItemReferenceDecision(
                status="AMBIGUOUS",
                p_item_id=None,
                reason=reason,
                candidates=(),
                similarity=None,
                margin=None,
                content_generation_max_mean_abs_error=content_error,
                completeness=completeness,
                top_k=(),
                frames_byte_identical=byte_identical,
                ranking_route="not_ranked",
            )
        # The low-dimensional content-generation gate above is the temporal
        # identity authority: a different rendered item must fail there before
        # business-ID matching. Rank the first stable frame once and reuse that
        # result for the same proven page generation. A calibrated slot size may
        # use fixed-gallery coarse/fine acceleration, but any threshold ambiguity
        # reruns the complete gallery before detail confirmation or failure.
        def decide(
            first_ranking: tuple[PItemReferenceHit, ...],
            *,
            ranking_route: str,
            full_fallback_used: bool,
        ) -> PItemReferenceDecision:
            raw_rankings = (first_ranking,) * len(images)
            if self.PLAN_AMBIGUITY.get(plan) is None and any(
                ranking and ranking[0].p_item_id in self.PLAN_AMBIGUOUS_IDS
                for ranking in raw_rankings
            ):
                return PItemReferenceDecision(
                    status="AMBIGUOUS",
                    p_item_id=None,
                    reason="plan_required_for_visual_alias",
                    candidates=(),
                    similarity=None,
                    margin=None,
                    content_generation_max_mean_abs_error=content_error,
                    completeness=completeness,
                    top_k=(),
                    frames_byte_identical=byte_identical,
                    ranking_route=ranking_route,
                    full_fallback_used=full_fallback_used,
                )
            rankings_by_frame = tuple(
                self._effective_rankings(ranking, plan=plan)
                for ranking in raw_rankings
            )
            if any(not ranking for ranking in rankings_by_frame):
                raise PItemReferenceError("P-item reference gallery yielded no candidates")
            top_ids = tuple(ranking[0].p_item_id for ranking in rankings_by_frame)
            first_top_k = rankings_by_frame[0][:5]
            if len(set(top_ids)) != 1:
                candidates = tuple(dict.fromkeys(top_ids))
                return PItemReferenceDecision(
                    status="AMBIGUOUS",
                    p_item_id=None,
                    reason="reference_identity_generation_unstable",
                    candidates=candidates,
                    similarity=min(
                        ranking[0].similarity for ranking in rankings_by_frame
                    ),
                    margin=None,
                    content_generation_max_mean_abs_error=content_error,
                    completeness=completeness,
                    top_k=first_top_k,
                    frames_byte_identical=byte_identical,
                    ranking_route=ranking_route,
                    full_fallback_used=full_fallback_used,
                )
            similarity = min(ranking[0].similarity for ranking in rankings_by_frame)
            frame_margins = tuple(
                (
                    None
                    if len(ranking) < 2
                    else ranking[0].similarity - ranking[1].similarity
                )
                for ranking in rankings_by_frame
            )
            margin = (
                None
                if any(value is None for value in frame_margins)
                else min(float(value) for value in frame_margins)
            )
            candidates = tuple(hit.p_item_id for hit in first_top_k)
            if similarity < self.runtime.identity_minimum_similarity:
                reason = "reference_similarity_below_threshold"
            elif margin is None or margin < self.runtime.identity_minimum_margin:
                reason = "reference_margin_below_threshold"
            else:
                return PItemReferenceDecision(
                    status="ACCEPTED",
                    p_item_id=first_top_k[0].p_item_id,
                    reason="rendered_reference_accepted",
                    candidates=candidates,
                    similarity=similarity,
                    margin=margin,
                    content_generation_max_mean_abs_error=content_error,
                    completeness=completeness,
                    top_k=first_top_k,
                    frames_byte_identical=byte_identical,
                    ranking_route=ranking_route,
                    full_fallback_used=full_fallback_used,
                )
            return PItemReferenceDecision(
                status="AMBIGUOUS",
                p_item_id=None,
                reason=reason,
                candidates=candidates,
                similarity=similarity,
                margin=margin,
                content_generation_max_mean_abs_error=content_error,
                completeness=completeness,
                top_k=first_top_k,
                frames_byte_identical=byte_identical,
                ranking_route=ranking_route,
                full_fallback_used=full_fallback_used,
            )

        profile = self._acceleration_profile(images[0], box)
        if self.runtime.coarse_fine_profiles and profile is None:
            return PItemReferenceDecision(
                status="AMBIGUOUS",
                p_item_id=None,
                reason="unsupported_capture_resolution",
                candidates=(),
                similarity=None,
                margin=None,
                content_generation_max_mean_abs_error=content_error,
                completeness=completeness,
                top_k=(),
                frames_byte_identical=byte_identical,
                ranking_route="not_ranked",
            )
        if profile is None:
            return decide(
                self._rank_full(images[0], box),
                ranking_route="full_rendered_reference",
                full_fallback_used=False,
            )
        accelerated = self._rank_coarse_fine(images[0], box, profile=profile)
        if not accelerated.guard_passed:
            return decide(
                self._rank_full(images[0], box, profile=profile),
                ranking_route="fixed_coarse_fine_guard_then_full_fallback",
                full_fallback_used=True,
            )
        decision = decide(
            accelerated.ranking,
            ranking_route="fixed_coarse_fine",
            full_fallback_used=False,
        )
        if decision.reason not in {
            "reference_margin_below_threshold",
            "reference_similarity_below_threshold",
        }:
            return decision
        return decide(
            self._rank_full(images[0], box, profile=profile),
            ranking_route="fixed_coarse_fine_then_full_fallback",
            full_fallback_used=True,
        )
