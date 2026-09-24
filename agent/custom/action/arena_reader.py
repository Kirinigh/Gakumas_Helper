"""Administrator-Maa backend for the read-only arena lineup reader.

Normal screenshots stay in memory; failed details retain at most six cached frames.
The backend performs only ordinary navigation,
long-press, detail-card clicks, scrolling and back/close actions.  It deliberately
has no operation capable of starting a contest match.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any, Protocol, NamedTuple
from pathlib import Path
from threading import Lock
from collections.abc import Mapping, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from utils import logger as _base_logger
from maa.context import Context
from maa.pipeline import JClick, JActionType
from arena_winrate import (
    TeamTarget,
    ArenaPageState,
    MemberSlotState,
    ArenaReaderError,
    ClickedSkillCard,
    MemberSlotMetrics,
    ArenaEntityCatalog,
    BadgeWorkerSelection,
    BadgeReferenceGallery,
    BadgeWorkerCalibrator,
    ContestSeasonDefinition,
    CustomizationBadgeDecision,
    GenericCostReferenceGallery,
    _reader_visual,
    _reader_metrics,
    classify_member_slot,
    duplicate_marker_visual_features,
    validate_conservative_cost_fallback,
    is_duplicate_marker_visual_candidate,
)
from card_selection import EmbeddingCardRecognizer
from p_item_recognition import (
    PItemReferenceError,
    PItemReferenceDecision,
    PItemRenderedReferenceGallery,
    measure_p_item_content_generation,
    p_item_content_generation_signatures,
    measure_p_item_content_generation_from_signatures,
)
from arena_winrate.task_log import diagnostic_logger
from arena_winrate._reader_io import IoReader
from card_selection.embedding import OnnxCardEmbedder
from arena_winrate.cancellation import cancellation_for
from arena_winrate.challenge_flow import DEFAULT_CHALLENGE_RECORD_ROOT
from arena_winrate._arena_page_flow import ArenaPageFlow, PageRecoveryState, RecoveryPageObservation
from arena_winrate._detail_identity import (
    DetailIdentityTransaction,
    detail_identity_proof_diagnostic,
)
from arena_winrate._reader_evidence import (
    _PItemOcrObservation,
    _DetailIdentityObservation,
)
from arena_winrate.badge_glyph_pool import BadgeGlyphDomain, badge_glyph_task_pools
from arena_winrate._reader_resources import manifest_identity, reader_resource_pools
from arena_winrate._reader_telemetry import TelemetryReader
from arena_winrate._reader_telemetry import _member_read_metrics_event as _component_function_member_read_metrics_event
from arena_winrate.recognition_probe import RecognitionProbe
from arena_winrate._card_detail_state import CardDetailState, ClosedCardDetail, card_detail_field, card_detail_state
from arena_winrate._detail_effect_ocr import DetailEffectOcr
from arena_winrate._detail_text_layout import (
    recover_distant_number_text,
    recover_wrapped_signed_text,
)
from arena_winrate._reader_card_layout import CardLayoutReader
from arena_winrate._reader_card_layout import _stable_reference_visual_group as _component_function_stable_reference_visual_group
from arena_winrate._reader_card_layout import _card_content_generation_shifted as _component_function_card_content_generation_shifted
from arena_winrate._reader_card_layout import _fixed_slot_aligned_content_proof as _component_function_fixed_slot_aligned_content_proof
from arena_winrate._reader_card_layout import _stable_reference_identity_candidates as _component_function_stable_reference_identity_candidates
from arena_winrate._reader_card_layout import (
    _reference_identity_projection_is_strict as _component_function_reference_identity_projection_is_strict,
)
from arena_winrate._skill_detail_retry import SkillDetailRetrySession
from arena_winrate._member_read_session import (
    MemberDetailRecoveryState,
    end_member_detail_recovery,
    begin_member_detail_recovery,
    member_detail_recovery_state,
)
from arena_winrate._p_item_read_workflow import PItemReadWorkflow, PItemDetailWorkflow
from arena_winrate._p_item_read_workflow import p_item_decision_evidence as _component_function_p_item_decision_evidence
from arena_winrate._reader_badge_capture import BadgeCapture
from arena_winrate._reader_badge_capture import _parallel_badge_jobs as _component_function_parallel_badge_jobs
from arena_winrate._reader_badge_capture import _badge_detail_inferable as _component_function_badge_detail_inferable
from arena_winrate._reader_badge_capture import _badge_glyph_ocr_canvases as _component_function_badge_glyph_ocr_canvases
from arena_winrate._reader_badge_capture import (
    _badge_glyph_observations_need_fresh_group as _component_function_badge_glyph_observations_need_fresh_group,
)
from arena_winrate._reader_badge_capture import (
    _badge_glyph_observations_are_temporally_stable as _component_function_badge_glyph_observations_are_temporally_stable,
)
from arena_winrate._reader_badge_runtime import BadgeRuntime
from arena_winrate._reader_badge_runtime import _badge_glyph_descriptor_sha256 as _component_function_badge_glyph_descriptor_sha256
from arena_winrate._reader_badge_runtime import _normalized_badge_glyph_domain as _component_function_normalized_badge_glyph_domain
from arena_winrate._reader_badge_runtime import _badge_glyph_comparison_metrics as _component_function_badge_glyph_comparison_metrics
from arena_winrate._reader_badge_runtime import _badge_glyph_calibration_outer_near as _component_function_badge_glyph_calibration_outer_near
from arena_winrate._reader_badge_runtime import _badge_glyph_calibration_inner_match as _component_function_badge_glyph_calibration_inner_match
from arena_winrate._reader_card_identity import CardIdentityReader
from arena_winrate._skill_detail_session import SkillDetailSession
from arena_winrate._detail_title_identity import DetailTitleIdentity
from arena_winrate._detail_title_identity import skill_card_title_ocr_segments as _component_function_skill_card_title_ocr_segments
from arena_winrate._p_item_detail_session import PItemDetailSession
from arena_winrate._reader_badge_inference import BadgeInference
from arena_winrate._source_restore_session import SourceRestoreSession
from arena_winrate._detail_capture_evidence import FrameEvidenceState, DetailCaptureEvidence, frame_evidence_field, frame_evidence_state
from arena_winrate._detail_capture_evidence import _FullFrameOcrEvidence as _FullFrameOcrEvidence
from arena_winrate._detail_capture_evidence import _TitleAnchorOcrEvidence as _TitleAnchorOcrEvidence
from arena_winrate._detail_capture_evidence import _TrustedSkillCardTitleRowsEvidence as _TrustedSkillCardTitleRowsEvidence
from arena_winrate._p_item_page_observation import PItemPageObservation
from arena_winrate._reader_arena_navigation import ArenaNavigationReader
from arena_winrate._reader_detail_lifecycle import DetailLifecycleReader
from arena_winrate._reader_detail_lifecycle import _same_clicked_card_resolution as _component_function_same_clicked_card_resolution
from arena_winrate._reader_detail_lifecycle import _detail_confirmation_wait_message as _component_function_detail_confirmation_wait_message
from arena_winrate._reader_badge_calibration import BadgeCalibration
from arena_winrate._reader_badge_calibration import (
    _stable_badge_glyph_exemplar_signature as _component_function_stable_badge_glyph_exemplar_signature,
)
from arena_winrate._reader_badge_calibration import (
    _badge_glyph_frames_are_bounded_raster_settling as _component_function_badge_glyph_frames_are_bounded_raster_settling,
)
from arena_winrate._reader_member_navigation import MemberNavigationReader
from arena_winrate._reader_card_customizations import CardCustomizationsReader
from arena_winrate._skill_card_effect_recovery import recover_skill_effect_body

logger = diagnostic_logger(_base_logger)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _normalize_latin_anchor(value: str) -> str:
    """Normalize OCR-only Latin diacritics without weakening anchor identity."""

    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if not unicodedata.combining(character)).strip().upper()


def _grade_ocr_value(value: str) -> int | None:
    """Map only the supported single-digit Grade domain to its value."""

    normalized = _normalize_latin_anchor(value)
    if re.fullmatch(r"[1-7]", normalized) is not None:
        return int(normalized)
    return None


def _resource_model_root(*parts: str) -> Path:
    """Resolve both source-tree and installed Maa resource layouts."""

    candidates = (
        PROJECT_ROOT / "assets" / "resource" / "base" / "model" / Path(*parts),
        PROJECT_ROOT / "resource" / "base" / "model" / Path(*parts),
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


CARD_MODEL_ROOT = _resource_model_root("classify", "card_embedding")
ARENA_CARD_MODEL_ROOT = _resource_model_root("embedding", "arena_card")
CARD_REFERENCE_ROOT = _resource_model_root("embedding", "arena_badge_reference")
CARD_COST_REFERENCE_ROOT = _resource_model_root(
    "embedding",
    "arena_card_cost_reference",
)
P_ITEM_REFERENCE_ROOT = _resource_model_root("embedding", "p_item_reference")
CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR = _reader_visual.CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR
# The inner radius is at most twice the already accepted first-frame raster
# jitter while retaining exact normalized width/height.  The outer radius is
# rejection-only: a competing label inside it disables calibrated transfer.
BADGE_GLYPH_CALIBRATED_INNER_L1_MAX = 240
BADGE_GLYPH_CALIBRATED_INNER_MSE_MAX = 16.0
BADGE_GLYPH_CALIBRATED_INNER_CHANGED_MAX = 16
BADGE_GLYPH_CALIBRATED_INNER_XOR_MAX = 16
BADGE_GLYPH_CALIBRATED_INNER_IOU_MIN = 0.90
BADGE_GLYPH_CALIBRATED_OUTER_L1_MAX = 480
BADGE_GLYPH_CALIBRATED_OUTER_MSE_MAX = 32.0
BADGE_GLYPH_CALIBRATED_OUTER_CHANGED_MAX = 32
BADGE_GLYPH_CALIBRATED_OUTER_XOR_MAX = 32
BADGE_GLYPH_CALIBRATED_OUTER_IOU_MIN = 0.80
# A three-frame descriptor need not be byte-identical before each frame is sent
# through the independent OCR vote.  The live 96x96 plate raster can settle by
# one foreground pixel per capture while the component box and semantic glyph
# stay fixed.  This gate admits only one chronological, nested contour change;
# it never creates or transfers an exemplar label.
BADGE_GLYPH_RASTER_SETTLING_AREA_DELTA_MAX = 2
BADGE_GLYPH_RASTER_SETTLING_L1_MAX = 120
BADGE_GLYPH_RASTER_SETTLING_CHANGED_MAX = 12
BADGE_GLYPH_RASTER_SETTLING_XOR_MAX = 3
BADGE_GLYPH_RASTER_SETTLING_IOU_MIN = 0.98
# A temporal disagreement may be a short-lived source-page raster transition.
# Before the first detail click, permit one bounded fresh-capture attempt to
# find a stable rolling three-frame window.  This is intentionally one fixed
# wall-clock budget rather than an extendable settle loop.
BADGE_GLYPH_FRESH_GROUP_TIMEOUT_SECONDS = 0.60
BADGE_GLYPH_FRESH_GROUP_INTERVAL_SECONDS = 0.08
# Capture starts can occur at 0.00 through 0.56 seconds; a ninth start at
# 0.64 seconds would exceed the fixed deadline.
BADGE_GLYPH_FRESH_GROUP_MAX_FRAMES = 8
# The packaged OCR and current extractor have independent positive holdout
# coverage for count 1 only.  The wider 1-9 recognition domain remains active
# as a veto except for one audited preprocessing shape: both wider polarity
# pairs agree on the validated winner while the narrowest pair contains that
# winner plus exactly one digit outside both the detail and validated domains.
# A readable competing legal or validated count still stops and can never be
# hidden by the clicked card's legal-state filter.
BADGE_GLYPH_OCR_VALIDATED_ACCEPTANCE_COUNTS = frozenset((1,))


class _BadgeShortlistResult(NamedTuple):
    """Complete output of the sole card-face badge shortlist route."""

    decision: CustomizationBadgeDecision
    observations: tuple[dict[str, Any], ...]


class PItemReader(Protocol):
    @property
    def content_generation_max_mean_abs_error(self) -> float: ...

    @property
    def maximum_detail_fallbacks_per_member(self) -> int: ...

    def content_generation_errors(
        self,
        images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
    ) -> Sequence[float]: ...

    def read(
        self,
        images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
        *,
        plan: str,
        eligible_p_item_ids: frozenset[int] | None = None,
        eligible_p_item_ids_by_slot: Sequence[frozenset[int]] | None = None,
        scale_fallback_allowed_by_slot: Sequence[bool] | None = None,
    ) -> Sequence[PItemReferenceDecision]: ...


class Task085PItemReader:
    """Use the fixed rendered gallery with calibrated coarse/fine acceleration."""

    def __init__(self, gallery: PItemRenderedReferenceGallery) -> None:
        self.gallery = gallery

    @property
    def maximum_detail_fallbacks_per_member(self) -> int:
        return self.gallery.runtime.maximum_detail_fallbacks_per_member

    @property
    def content_generation_max_mean_abs_error(self) -> float:
        return self.gallery.runtime.content_generation_max_mean_abs_error

    def content_generation_errors(
        self,
        images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
    ) -> Sequence[float]:
        return measure_p_item_content_generation(images, boxes)

    @classmethod
    def from_model_root(
        cls,
        model_root: str | Path = P_ITEM_REFERENCE_ROOT,
    ) -> "Task085PItemReader":
        root = Path(model_root)
        try:
            gallery = PItemRenderedReferenceGallery.load(root)
        except (OSError, KeyError, TypeError, ValueError, PItemReferenceError) as error:
            raise ArenaReaderError(
                "p_item_runtime_not_approved",
                "the fixed P-item rendered-reference runtime is unavailable or invalid",
            ) from error
        return cls(gallery)

    def read(
        self,
        images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
        *,
        plan: str,
        eligible_p_item_ids: frozenset[int] | None = None,
        eligible_p_item_ids_by_slot: Sequence[frozenset[int]] | None = None,
        scale_fallback_allowed_by_slot: Sequence[bool] | None = None,
    ) -> Sequence[PItemReferenceDecision]:
        if eligible_p_item_ids_by_slot is not None and len(eligible_p_item_ids_by_slot) != len(boxes):
            raise PItemReferenceError("P-item candidate domains must match the screen slots")

        if scale_fallback_allowed_by_slot is not None and len(scale_fallback_allowed_by_slot) != len(boxes):
            raise PItemReferenceError("P-item scale budgets must match the screen slots")

        def classify(entry: tuple[int, tuple[int, int, int, int]]) -> PItemReferenceDecision:
            slot_index, box = entry
            return self.gallery.classify(
                images,
                box,
                plan=plan,
                allow_scale_fallback=(True if scale_fallback_allowed_by_slot is None else scale_fallback_allowed_by_slot[slot_index]),
                eligible_p_item_ids=(eligible_p_item_ids if eligible_p_item_ids_by_slot is None else eligible_p_item_ids_by_slot[slot_index]),
            )

        with ThreadPoolExecutor(max_workers=min(2, len(boxes))) as executor:
            return tuple(executor.map(classify, enumerate(boxes)))


class ArenaReaderDiagnosticPort:
    """Explicit development-only access to low-level reader diagnostics.

    Production observations never use this port.  Keeping calibration access
    behind one named public object prevents probes from silently depending on
    backend-private cache layout.
    """

    def __init__(self, backend: "MaaArenaReaderBackend") -> None:
        self._backend = backend

    @property
    def grade(self) -> int | None:
        return self._backend._grade

    @property
    def member_boxes(self) -> Any:
        return self._backend._member_boxes

    @property
    def member_slot_observations(self) -> Any:
        return self._backend._member_slot_observations

    @property
    def card_rows(self) -> Any:
        return self._backend._card_rows

    @property
    def card_predictions(self) -> Any:
        return self._backend._card_predictions

    @property
    def card_candidate_groups(self) -> Any:
        return self._backend._card_candidate_groups

    @property
    def card_images(self) -> Any:
        return self._backend._card_images

    @property
    def card_count_frames(self) -> Any:
        return self._backend._card_count_frames

    @property
    def card_excluded_duplicate_flags(self) -> Any:
        return self._backend._card_excluded_duplicate_flags

    @property
    def card_empty_flags(self) -> Any:
        return self._backend._card_empty_flags

    @property
    def card_empty_diagnostics(self) -> Any:
        return self._backend._card_empty_diagnostics

    @property
    def card_duplicate_marker_diagnostics(self) -> Any:
        return self._backend._card_duplicate_marker_diagnostics

    @property
    def badge_count_diagnostics(self) -> Any:
        return self._backend._badge_count_diagnostics

    @property
    def badge_candidate_detail_checks(self) -> Any:
        return self._backend._badge_candidate_detail_checks

    @property
    def card_content_generation_diagnostics(self) -> Any:
        return self._backend._card_content_generation_diagnostics

    @property
    def inferred_clicked_cards(self) -> Any:
        return self._backend._inferred_clicked_cards

    def capture(self) -> Any:
        return self._backend._capture()

    def ocr(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._ocr(*args, **kwargs)

    def click(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._click(*args, **kwargs)

    def back(self) -> Any:
        return self._backend._back()

    def swipe(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._swipe(*args, **kwargs)

    def long_press(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._long_press(*args, **kwargs)

    def full_ocr_text(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._full_ocr_text(*args, **kwargs)

    def stage_total_anchors(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._stage_total_anchors(*args, **kwargs)

    def team_stage_total_anchors(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._team_stage_total_anchors(*args, **kwargs)

    def assert_member_detail(self) -> Any:
        return self._backend._assert_member_detail()

    def assert_card_group_visible(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._assert_card_group_visible(*args, **kwargs)

    def dismiss_skill_card_detail(self) -> Any:
        return self._backend._dismiss_skill_card_detail()

    def dismiss_known_blocking_overlay(self, image: Any) -> bool:
        return self._backend._dismiss_known_blocking_overlay(image)

    def detect_card_rows(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._detect_card_rows(*args, **kwargs)

    def raw_card_candidate_boxes(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._raw_card_candidate_boxes(*args, **kwargs)

    def targeted_duplicate_marker_observations(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._targeted_duplicate_marker_observations(*args, **kwargs)

    def card_recognizer(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._card_recognizer(*args, **kwargs)


class _ArenaPageFlowPort:
    """Expose page capabilities without copying reader state or binding overrides."""

    def __init__(self, backend: "MaaArenaReaderBackend") -> None:
        self._backend = backend

    @property
    def state(self) -> PageRecoveryState:
        return self._backend._page_recovery_state

    def capture(self) -> Any:
        return self._backend._capture()

    def ocr(self, image: Any, expected: str) -> Any:
        return self._backend._ocr(image, expected)

    def recognize(self, entry: str, image: Any) -> Any:
        return self._backend._recognize(entry, image)

    def run_recognition(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._run_recognition(*args, **kwargs)

    def click(self, box: Any, **kwargs: Any) -> Any:
        return self._backend._click(box, **kwargs)

    def sleep(self, seconds: float) -> Any:
        return self._backend._sleep(seconds)

    def check_cancelled(self) -> Any:
        return self._backend._check_cancelled()

    def arena_page_state(self, image: Any, *, observations: Sequence[Any] | None = None) -> Any:
        if observations is None:
            return self._backend._arena_page_state(image)
        return self._backend._arena_page_state(image, observations=observations)

    def matching_ocr_items(self, items: Sequence[Any], expected: str) -> Any:
        return self._backend._matching_ocr_items(items, expected)

    def box_center_point(self, box: Any) -> Any:
        return self._backend._box_center_point(box)

    def increment(self, name: str) -> Any:
        return self._backend._increment(name)

    def add_timing(self, name: str, elapsed: float) -> Any:
        return self._backend._add_timing(name, elapsed)

    def record_duration_sample(self, name: str, elapsed: float) -> Any:
        return self._backend._record_duration_sample(name, elapsed)

    def back(self, *, image: Any = None) -> Any:
        return self._backend._back(image=image)

    def dismiss_known_blocking_overlay(self, image: Any, *, items: Sequence[Any] | None = None) -> Any:
        return self._backend._dismiss_known_blocking_overlay(image, items=items)

    def dismiss_contest_details_items(self, image: Any, items: Sequence[Any]) -> Any:
        return self._backend._dismiss_contest_details_items(image, items)

    def retry_transient_communication_items(self, items: Sequence[Any]) -> Any:
        return self._backend._retry_transient_communication_items(items)


class MaaArenaReaderBackend:
    """Compose arena reading responsibilities inside the elevated Maa agent.

    This facade owns member/task state and preserves the established entry
    points. Responsibility components receive its live state through explicit
    ports; they never keep a second cache or a second retry budget. Factories
    resolve current dependencies on each call so overrides and cancellation
    continue to reach the same backend during a transaction.

    Recognition, navigation and recovery algorithms belong in the component
    modules below, not in these forwarding methods.
    """

    _member_long_press_seconds = 0.7
    _source_restore_poll_seconds = 0.08

    def __init__(
        self,
        context: Context,
        season: ContestSeasonDefinition,
        bundle_dir: str | Path,
        *,
        p_item_reader: PItemReader | None = None,
        card_swipe_duration_ms: int = 100,
        known_grade: int | None = None,
    ) -> None:
        if card_swipe_duration_ms not in (50, 100, 150, 200):
            raise ValueError("card_swipe_duration_ms must be 50, 100, 150, or 200")
        if known_grade is not None and (type(known_grade) is not int or not 1 <= known_grade <= 7):
            raise ValueError("known_grade must be an integer from 1 to 7")
        self.context = context
        self._cancellation = cancellation_for(context)
        self.season = season
        self._runtime_timing_seconds: dict[str, float] = {}
        self._runtime_counts: dict[str, int] = {}
        self._runtime_duration_samples: dict[str, list[float]] = {}
        self._reader_resources = reader_resource_pools.for_context(context)
        self._reader_bundle_identity = manifest_identity(bundle_dir)
        self.catalog = self._load_static_resource(
            "catalog",
            bundle_dir,
            lambda: ArenaEntityCatalog.from_bundle(bundle_dir),
        )
        self.p_item_reader = p_item_reader
        self._card_swipe_duration_ms = card_swipe_duration_ms
        self._grade = known_grade
        self._member_boxes: dict[tuple[str, int], dict[int, tuple[int, int, int, int]]] = {}
        self._member_slot_observations: dict[
            tuple[str, int],
            dict[int, tuple[MemberSlotState, MemberSlotMetrics, tuple[int, int, int, int]]],
        ] = {}
        self._card_rows: dict[int, tuple[tuple[int, int, int, int], ...]] = {}
        self._card_predictions: dict[tuple[int, int], int] = {}
        self._card_candidate_groups: dict[tuple[int, int], tuple[int, ...]] = {}
        self._card_images: dict[int, Any] = {}
        self._card_count_frames: dict[int, tuple[Any, Any, Any]] = {}
        self._card_identity_frames: dict[int, tuple[tuple[dict[str, Any], ...], ...]] = {}
        self._card_source_ocr_counts: dict[int, dict[str, int]] = {}
        # A capture is immutable evidence.  Full-frame OCR, authoritative title
        # extraction and title-bound ROI geometry must therefore share the raw
        # boxes from that exact capture rather than asking the OCR backend to
        # interpret the same pixels repeatedly.  Identity, not image bytes, is
        # the boundary: a fresh capture must always receive fresh recognition.
        self._frame_evidence_state = FrameEvidenceState()
        self._card_source_guard_frames: dict[int, tuple[Any, ...]] = {}
        self._card_source_guard_signature_cache: dict[
            tuple[tuple[int, ...], tuple[tuple[int, int, int, int], ...]],
            tuple[tuple[Any, ...], ...],
        ] = {}
        self._card_restoration_signatures: dict[
            int,
            tuple[tuple[tuple[str, Any], ...], ...],
        ] = {}
        self._card_excluded_duplicate_flags: dict[int, tuple[bool, ...]] = {}
        self._card_empty_flags: dict[int, tuple[bool, ...]] = {}
        self._card_empty_diagnostics: dict[int, tuple[dict[str, Any], ...]] = {}
        self._card_duplicate_marker_diagnostics: dict[int, tuple[dict[str, Any], ...]] = {}
        self._badge_count_diagnostics: dict[int, tuple[dict[str, Any], ...]] = {}
        self._detail_count_overrides: dict[tuple[int, int], dict[str, Any]] = {}
        self._detail_semantic_confirmations: dict[tuple[int, int], dict[str, Any]] = {}
        self._card_detail_state = CardDetailState()
        self._member_detail_recovery = MemberDetailRecoveryState()
        self._detail_title_disambiguations: list[dict[str, Any]] = []
        self._badge_local_results: dict[int, tuple[Any, ...]] = {}
        self._badge_glyph_observations: dict[tuple[int, int], tuple[dict[str, Any], ...]] = {}
        self._badge_glyph_count_diagnostics: dict[tuple[int, int], dict[str, Any]] = {}
        # Only independent detail truth crosses member/reader boundaries.
        # Source-frame dimensions select the task domain before observation.
        self._badge_glyph_task_pool = badge_glyph_task_pools.for_context(context)
        self._badge_glyph_pool_size: tuple[int, int] | None = None
        self._badge_glyph_domain = BadgeGlyphDomain()
        self._badge_glyph_exemplars = self._badge_glyph_domain.exemplars
        self._badge_glyph_polluted_descriptors = self._badge_glyph_domain.polluted
        self._badge_glyph_runtime_labels: dict[tuple[int, ...], int] = {}
        # Diagnostics retain scalar provenance, never card images or masks.
        self._runtime_badge_glyph_exemplar_diagnostics: list[dict[str, Any]] = []
        self._runtime_badge_glyph_exemplar_comparisons: list[dict[str, Any]] = []
        self._inferred_clicked_cards: dict[tuple[int, int], ClickedSkillCard] = {}
        self._active_inferred_clicked_card: tuple[int, int] | None = None
        self._p_item_diagnostics: tuple[dict[str, Any], ...] = ()
        self._p_item_generation_evidence: dict[str, Any] = {}
        self._card_recognizers: dict[bool, tuple[EmbeddingCardRecognizer, float, float]] = {}
        self._card_recognizer_lock = Lock()
        self._card_reference_gallery: BadgeReferenceGallery | None = None
        self._card_cost_reference_gallery: GenericCostReferenceGallery | None = None
        self._zero_card_identity_fusion_diagnostics: dict[tuple[int, int], dict[str, Any]] = {}
        self._zero_card_embedding_detail_candidates: dict[tuple[int, int], tuple[int, ...]] = {}
        self._zero_card_detail_candidates: dict[tuple[int, int], tuple[int, ...]] = {}
        self._badge_worker_calibrator = BadgeWorkerCalibrator()
        self._badge_worker_selection: BadgeWorkerSelection | None = None
        self._badge_candidate_detail_checks: list[dict[str, Any]] = []
        self._secondary_fixed_slot_fallback_enabled = False
        self._secondary_presence_diagnostics: tuple[dict[str, Any], ...] = ()
        self._card_content_generation_diagnostics: tuple[dict[str, Any], ...] = ()
        self._card_face_cost_diagnostics: dict[tuple[int, int], dict[str, Any]] = {}
        self._card_face_cost_optional_errors: dict[tuple[int, int], str] = {}
        self._cost_customization_fallbacks: dict[tuple[int, int], dict[str, Any]] = {}
        self._card_face_cost_failed_sources: dict[tuple[int, int], dict[str, Any]] = {}
        self._runtime_cost_customization_fallbacks: list[dict[str, Any]] = []
        self._runtime_detail_title_disambiguations: list[dict[str, Any]] = []
        self._secondary_presence_cache: dict[
            int,
            tuple[Any, tuple[dict[str, Any], ...]],
        ] = {}
        self._member_metric_baseline: (
            tuple[
                dict[str, float],
                dict[str, int],
                dict[str, int],
            ]
            | None
        ) = None
        self._challenge_selection_committed = False
        self._p_item_catalog_compatibility_checked = False
        self._p_item_reference_gallery_ids: tuple[int, ...] = ()
        self._p_item_reference_missing_arena_ids: tuple[int, ...] = ()
        self._p_item_reference_unknown_catalog_ids: tuple[int, ...] = ()
        self._p_item_reference_provisional_ids: tuple[int, ...] = ()
        self._skill_card_catalog_compatibility_checked = False
        self._skill_card_reference_gallery_ids: tuple[int, ...] = ()
        self._skill_card_reference_missing_catalog_ids: tuple[int, ...] = ()

    # Legacy session/diagnostic entry points all address the same state owner.
    _full_frame_ocr_evidence = frame_evidence_field("full_frame_ocr_evidence")
    _title_anchor_ocr_evidence = frame_evidence_field("title_anchor_ocr_evidence")
    _trusted_skill_card_title_rows_cache = frame_evidence_field("trusted_skill_card_title_rows_cache")
    _skill_card_recovered_title_frames = frame_evidence_field("skill_card_recovered_title_frames")
    _upgrade_title_roi_evidence = frame_evidence_field("upgrade_title_roi_evidence")
    _isolated_title_evidence = frame_evidence_field("isolated_title_evidence")

    _card_transaction_serial = card_detail_field("serial")
    _card_transaction_started = card_detail_field("started")
    _card_transaction_tokens = card_detail_field("tokens")
    _card_transaction_contact_counts = card_detail_field("contact_counts")
    _card_transaction_source_boxes = card_detail_field("source_boxes")
    _card_transaction_interaction_boxes = card_detail_field("interaction_boxes")
    _card_transaction_kinds = card_detail_field("kinds")
    _card_transaction_ocr_started = card_detail_field("ocr_started")
    _card_detail_open_ids = card_detail_field("open_ids")
    _card_detail_rebind_ids = card_detail_field("rebind_ids")
    _card_detail_texts = card_detail_field("texts")
    _card_detail_images = card_detail_field("images")
    _card_detail_last_contact_released_at = card_detail_field("contact_released_at")
    _card_detail_capture_started_at = card_detail_field("capture_started_at")
    _detail_identity_proofs = card_detail_field("identity_proofs")

    @property
    def grade(self) -> int | None:
        """Return the recognized or explicitly injected Grade for this session."""

        return self._grade

    @property
    def _page_recovery_state(self) -> PageRecoveryState:
        # Lazy creation preserves minimal diagnostic/test readers. Cleanup of a
        # member never replaces this reader-lifetime input budget.
        state = self.__dict__.get("_page_recovery")
        if state is None:
            state = PageRecoveryState()
            self.__dict__["_page_recovery"] = state
        return state

    @property
    def _arena_communication_retry_attempted(self) -> bool:
        return self._page_recovery_state.communication_retry_attempted

    @_arena_communication_retry_attempted.setter
    def _arena_communication_retry_attempted(self, value: bool) -> None:
        self._page_recovery_state.communication_retry_attempted = value

    @property
    def _contest_details_close_attempts(self) -> int:
        return self._page_recovery_state.contest_details_close_attempts

    @_contest_details_close_attempts.setter
    def _contest_details_close_attempts(self, value: int) -> None:
        self._page_recovery_state.contest_details_close_attempts = value

    @property
    def _last_retry_dialog_seen(self) -> bool:
        return self._page_recovery_state.last_retry_dialog_seen

    @_last_retry_dialog_seen.setter
    def _last_retry_dialog_seen(self, value: bool) -> None:
        self._page_recovery_state.last_retry_dialog_seen = value

    def _add_timing(self, name: str, elapsed: float) -> None:
        self._runtime_timing_seconds[name] = self._runtime_timing_seconds.get(name, 0.0) + elapsed

    def _increment(self, name: str) -> None:
        self._runtime_counts[name] = self._runtime_counts.get(name, 0) + 1

    def _reader_compute(self, metric: str, operation: Callable[[], Any]) -> Any:
        """Time one existing call; cancellation forbids starting its successor."""
        self._check_cancelled()
        started = time.perf_counter()
        try:
            result = operation()
        finally:
            if hasattr(self, "_runtime_timing_seconds"):
                self._add_timing(metric, time.perf_counter() - started)
            if hasattr(self, "_runtime_counts"):
                self._increment(metric + "_calls")
        self._check_cancelled()
        return result

    def _load_static_resource(self, name: str, root: str | Path, factory: Callable[[], Any]) -> Any:
        self._check_cancelled()
        pool = getattr(self, "_reader_resources", None)
        bundle = getattr(self, "_reader_bundle_identity", None)
        asset = bundle if name == "catalog" else manifest_identity(root)
        identity = (bundle, asset) if bundle is not None and asset is not None else None

        def load():
            return self._reader_compute("reader_resource_load_" + name, factory)

        if pool is None:
            return load()
        value, cache_hit = pool.load(name, identity, load)
        self._check_cancelled()
        if cache_hit:
            self._increment("reader_resource_cache_hits_" + name)
        return value

    def _classify_card_candidate(self, recognizer, image, candidate, *, preprocess_cache=None, **kwargs):
        """Keep the existing single-candidate contract, exposing its CPU costs."""
        if type(recognizer) is not EmbeddingCardRecognizer:
            return self._reader_compute(
                "card_candidate_inference",
                lambda: recognizer.classify(image, candidate, **kwargs),
            )
        self._check_cancelled()
        embedder = recognizer.embedder
        contract = None
        if (
            preprocess_cache is not None
            and type(embedder) is OnnxCardEmbedder
            and getattr(embedder.preprocess, "__func__", None) is OnnxCardEmbedder.preprocess
        ):
            contract = (embedder.source_color_order, embedder.INPUT_SIZE, OnnxCardEmbedder.preprocess)
        cached = preprocess_cache.get(contract) if contract is not None else None
        if cached is not None and cached[0] is image and cached[1] == candidate.box:
            tensor = cached[2]
            self._increment("card_candidate_preprocess_reuse_hits")
        else:
            tensor = self._reader_compute(
                "card_candidate_preprocess",
                lambda: embedder.preprocess(image, candidate.box),
            )
            if contract is not None:
                preprocess_cache[contract] = (image, candidate.box, tensor)
        if recognizer.embedder._session is None:
            self._reader_compute("card_model_session_load", recognizer.embedder._load_session)
        embedding = self._reader_compute(
            "card_candidate_encode",
            lambda: recognizer.embedder.embed_tensor(tensor),
        )
        kwargs.setdefault("resolve_upgrade_state", True)
        return self._reader_compute(
            "card_candidate_retrieval",
            lambda: recognizer.classify_embedding(embedding, candidate, upgrade_image=image, **kwargs),
        )

    def _record_duration_sample(self, name: str, elapsed: float) -> None:
        samples = getattr(self, "_runtime_duration_samples", None)
        if samples is None:
            samples = {}
            self._runtime_duration_samples = samples
        samples.setdefault(name, []).append(elapsed)

    def runtime_sample_cursor(self) -> dict[str, int]:
        return self._component_telemetry_reader().runtime_sample_cursor()

    def runtime_counts_cursor(self) -> dict[str, int]:
        return self._component_telemetry_reader().runtime_counts_cursor()

    def record_lineup_read_attempt(self, record: dict[str, Any]) -> None:
        return self._component_telemetry_reader().record_lineup_read_attempt(record)

    def last_failure_location(self, error: object = None) -> dict[str, Any] | None:
        return self._component_telemetry_reader().last_failure_location(error)

    def _note_confirmed_detail_name(self, card_id: int) -> None:
        return self._component_telemetry_reader()._note_confirmed_detail_name(card_id)

    def _begin_detail_diagnostics(self, *, source: Any = None, **position: Any) -> None:
        return self._component_telemetry_reader()._begin_detail_diagnostics(source=source, **position)

    def _save_recognition_probe(self, detail: dict, outcome: str, error: str | None) -> None:
        return self._component_telemetry_reader()._save_recognition_probe(detail, outcome, error)

    def _observe_recognition_probe(self, image: Any, items: Any, *, reco_id=None) -> None:
        return self._component_telemetry_reader()._observe_recognition_probe(image, items, reco_id=reco_id)

    def _emit_detail_fallback_diagnostic(self, outcome: str, error: str | None = None) -> None:
        return self._component_telemetry_reader()._emit_detail_fallback_diagnostic(outcome, error)

    def begin_member_read_diagnostics(self, target: TeamTarget, stage_number: int, member_slot: int) -> None:
        # Preserve direct diagnostic callers; the read session owns this hook
        # independently so disabling diagnostics cannot disable recovery.
        if not member_detail_recovery_state(self).managed_by_session:
            begin_member_detail_recovery(self, target, stage_number, member_slot)
        return self._component_telemetry_reader().begin_member_read_diagnostics(target, stage_number, member_slot)

    def _begin_member_read_attempt(self, target: TeamTarget, stage_number: int, member_slot: int) -> None:
        begin_member_detail_recovery(self, target, stage_number, member_slot, managed_by_session=True)

    def _end_member_read_attempt(self) -> None:
        end_member_detail_recovery(self)

    def _finalize_member_read_attempt(
        self, target: TeamTarget, stage_number: int, member_slot: int, *, succeeded: bool, superseded: bool
    ) -> None:
        if superseded:
            diagnostic = getattr(self, "_member_failure_frames", None)
            position = diagnostic.get("position") if isinstance(diagnostic, dict) else None
            self._progressive_abandoned_member = dict(position) if isinstance(position, dict) else {}
            self._progressive_abandoned_member.update(
                team_id=target.team_id, opponent_position=target.opponent_position,
                stage_number=stage_number, member_slot=member_slot,
            )
        self._component_detail_lifecycle_reader().finish_member_transactions(succeeded=succeeded, superseded=superseded)

    def set_member_read_phase(self, phase: str, **position: Any) -> None:
        return self._component_telemetry_reader().set_member_read_phase(phase, **position)

    def end_member_read_diagnostics(self) -> None:
        try:
            return self._component_telemetry_reader().end_member_read_diagnostics()
        finally:
            if not member_detail_recovery_state(self).managed_by_session:
                end_member_detail_recovery(self)

    def note_member_confirmed_card(self, card_id: int) -> None:
        return self._component_telemetry_reader().note_member_confirmed_card(card_id)

    def record_skill_card_group_observation(
        self,
        group_index: int,
        card_ids: Sequence[int],
        customizations: Sequence[Mapping[str, int]],
        excluded: Sequence[bool],
        empty: Sequence[bool],
    ) -> None:
        return self._component_telemetry_reader().record_skill_card_group_observation(group_index, card_ids, customizations, excluded, empty)

    def persist_member_read_failure(self, error: Exception) -> None:
        return self._component_telemetry_reader().persist_member_read_failure(error)

    def _record_detail_action(self, kind: str, box: Sequence[int], *, succeeded: bool) -> None:
        return self._component_telemetry_reader()._record_detail_action(kind, box, succeeded=succeeded)

    def _persist_detail_failure(self, error: str) -> None:
        return self._component_telemetry_reader()._persist_detail_failure(error)

    def _save_failure_evidence(self, diagnostic: dict[str, Any], error: str, *, event: str) -> None:
        return self._component_telemetry_reader()._save_failure_evidence(diagnostic, error, event=event)

    def runtime_samples_since(self, cursor: Mapping[str, int]) -> dict[str, Any]:
        return self._component_telemetry_reader().runtime_samples_since(cursor)

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
        if not member_detail_recovery_state(self).managed_by_session:
            self._finalize_member_read_attempt(target, stage_number, member_slot, succeeded=succeeded, superseded=superseded)
        return self._component_telemetry_reader().record_member_read_attempt(
            target,
            stage_number,
            member_slot,
            wall_seconds=wall_seconds,
            succeeded=succeeded,
            error_code=error_code,
            reopened=reopened,
            sample_cursor=sample_cursor,
            superseded=superseded,
        )

    @staticmethod
    def _percentile(values: Sequence[float], quantile: float) -> float:
        return _reader_metrics.percentile(values, quantile)

    def _finish_card_transaction(
        self, key: tuple[int, int], *, failed: bool = False, error: str | None = None, superseded: bool = False, cancelled: bool = False
    ) -> None:
        return self._component_detail_lifecycle_reader()._finish_card_transaction(
            key, failed=failed, error=error, superseded=superseded, cancelled=cancelled
        )

    def _report_card_transaction_end(
        self, closed: ClosedCardDetail, *, failed: bool = False, error: str | None = None, superseded: bool = False, cancelled: bool = False
    ) -> None:
        return self._component_telemetry_reader().record_card_transaction_end(
            closed, failed=failed, error=error, superseded=superseded, cancelled=cancelled
        )

    def runtime_metrics(self) -> dict[str, Any]:
        return self._component_telemetry_reader().runtime_metrics()

    @staticmethod
    def _member_read_metrics_event(target: TeamTarget, stage_number: int, member_slot: int, evidence: dict[str, Any]) -> dict[str, Any]:
        return _component_function_member_read_metrics_event(target, stage_number, member_slot, evidence)

    def diagnostic_port(self) -> ArenaReaderDiagnosticPort:
        """Return the explicit non-production calibration interface."""

        return ArenaReaderDiagnosticPort(self)

    def member_observation_evidence(self, target: TeamTarget, stage_number: int, member_slot: int) -> dict[str, Any]:
        return self._component_telemetry_reader().member_observation_evidence(target, stage_number, member_slot)

    def _card_references(self) -> BadgeReferenceGallery:
        return self._component_badge_capture()._card_references()

    def _card_cost_references(self) -> GenericCostReferenceGallery:
        return self._component_badge_capture()._card_cost_references()

    @staticmethod
    def _parallel_badge_jobs(workers: int, jobs: Sequence[tuple[int, int]], operation: Any) -> tuple[Any, ...]:
        return _component_function_parallel_badge_jobs(workers, jobs, operation)

    def _local_badge_slot_decision(
        self, frames: Sequence[Any], frame_rows: Sequence[Sequence[tuple[int, int, int, int]]], group_index: int, slot_index: int
    ) -> _BadgeShortlistResult:
        return self._component_badge_capture()._local_badge_slot_decision(frames, frame_rows, group_index, slot_index)

    def _badge_workers(self, frames: Sequence[Any], jobs: Sequence[tuple[int, int]], operation: Any) -> int:
        return self._component_badge_capture()._badge_workers(frames, jobs, operation)

    @staticmethod
    def _badge_detail_inferable(decision: CustomizationBadgeDecision) -> bool:
        return _component_function_badge_detail_inferable(decision)

    def _record_badge_candidate_detail(
        self, *, stage_number: int, member_slot: int, group_index: int, card_slot: int, phase: str, reason: str
    ) -> dict[str, Any]:
        return self._component_badge_capture()._record_badge_candidate_detail(
            stage_number=stage_number, member_slot=member_slot, group_index=group_index, card_slot=card_slot, phase=phase, reason=reason
        )

    def _finish_badge_candidate_detail(self, entry: dict[str, Any], result: ClickedSkillCard) -> None:
        return self._component_badge_capture()._finish_badge_candidate_detail(entry, result)

    @classmethod
    def _badge_glyph_observations_are_temporally_stable(cls, observations: Sequence[dict[str, Any]]) -> bool:
        return _component_function_badge_glyph_observations_are_temporally_stable(cls, observations)

    @classmethod
    def _badge_glyph_observations_need_fresh_group(cls, observations: Sequence[dict[str, Any]]) -> bool:
        return _component_function_badge_glyph_observations_need_fresh_group(cls, observations)

    def _capture_one_fresh_badge_glyph_group(
        self, visible_groups: Sequence[int], related_jobs: Sequence[tuple[int, int]], *, workers: int
    ) -> tuple[Any, ...] | None:
        return self._component_badge_capture()._capture_one_fresh_badge_glyph_group(visible_groups, related_jobs, workers=workers)

    def _batched_badge_local_results(self, requested_group: int) -> tuple[Any, ...]:
        return self._component_badge_capture()._batched_badge_local_results(requested_group)

    def _bind_badge_glyph_task_domain(self, frames: Sequence[Any]) -> None:
        return self._component_badge_capture()._bind_badge_glyph_task_domain(frames)

    @staticmethod
    def _badge_glyph_ocr_canvases(descriptor: Any) -> tuple[tuple[int, str, Any], ...]:
        return _component_function_badge_glyph_ocr_canvases(descriptor)

    @staticmethod
    def _stable_badge_glyph_exemplar_signature(
        observations: Sequence[dict[str, Any]],
    ) -> tuple[tuple[int, ...], tuple[int, int, int, int, int], int] | None:
        return _component_function_stable_badge_glyph_exemplar_signature(observations)

    @classmethod
    def _badge_glyph_frames_are_bounded_raster_settling(cls, observations: Sequence[dict[str, Any]]) -> bool:
        return _component_function_badge_glyph_frames_are_bounded_raster_settling(
            cls,
            observations,
            BADGE_GLYPH_RASTER_SETTLING_AREA_DELTA_MAX=BADGE_GLYPH_RASTER_SETTLING_AREA_DELTA_MAX,
            BADGE_GLYPH_RASTER_SETTLING_CHANGED_MAX=BADGE_GLYPH_RASTER_SETTLING_CHANGED_MAX,
            BADGE_GLYPH_RASTER_SETTLING_IOU_MIN=BADGE_GLYPH_RASTER_SETTLING_IOU_MIN,
            BADGE_GLYPH_RASTER_SETTLING_L1_MAX=BADGE_GLYPH_RASTER_SETTLING_L1_MAX,
            BADGE_GLYPH_RASTER_SETTLING_XOR_MAX=BADGE_GLYPH_RASTER_SETTLING_XOR_MAX,
        )

    def _maybe_register_badge_glyph_exemplar(
        self,
        key: tuple[int, int],
        resolved: ClickedSkillCard,
        *,
        target: TeamTarget | None = None,
        stage_number: int | None = None,
        member_slot: int | None = None,
    ) -> bool:
        return self._component_badge_calibration()._maybe_register_badge_glyph_exemplar(
            key, resolved, target=target, stage_number=stage_number, member_slot=member_slot
        )

    @staticmethod
    def _badge_glyph_descriptor_sha256(descriptor: Sequence[int]) -> str:
        return _component_function_badge_glyph_descriptor_sha256(descriptor)

    def _validate_badge_glyph_tail_outlier(
        self, key: tuple[int, int], *, count: int, support_frames: int, detail_confirmed: bool = False
    ) -> tuple[int, ...] | None:
        return self._component_badge_runtime()._validate_badge_glyph_tail_outlier(
            key, count=count, support_frames=support_frames, detail_confirmed=detail_confirmed
        )

    def _validate_authoritative_badge_glyph_runtime_transition(self, key: tuple[int, int], descriptor: tuple[int, ...], *, count: int) -> None:
        return self._component_badge_runtime()._validate_authoritative_badge_glyph_runtime_transition(key, descriptor, count=count)

    def _validate_badge_glyph_runtime_label(
        self, key: tuple[int, int], descriptor: tuple[int, ...], *, count: int, evidence: str, check_nearby_samples: bool = True
    ) -> None:
        return self._component_badge_runtime()._validate_badge_glyph_runtime_label(
            key, descriptor, count=count, evidence=evidence, check_nearby_samples=check_nearby_samples
        )

    def _record_badge_glyph_runtime_label(self, key: tuple[int, int], descriptor: tuple[int, ...] | None, *, count: int, evidence: str) -> None:
        return self._component_badge_runtime()._record_badge_glyph_runtime_label(key, descriptor, count=count, evidence=evidence)

    def _badge_glyph_consumed_claims(self, descriptor: tuple[int, ...], count: int):
        return self._component_badge_runtime()._badge_glyph_consumed_claims(descriptor, count)

    @staticmethod
    def _badge_glyph_comparison_metrics(target: tuple[int, ...], sample: tuple[int, ...]) -> dict[str, int | float]:
        return _component_function_badge_glyph_comparison_metrics(target, sample)

    @staticmethod
    def _badge_glyph_calibration_outer_near(metrics: dict[str, int | float]) -> bool:
        return _component_function_badge_glyph_calibration_outer_near(
            metrics,
            BADGE_GLYPH_CALIBRATED_OUTER_CHANGED_MAX=BADGE_GLYPH_CALIBRATED_OUTER_CHANGED_MAX,
            BADGE_GLYPH_CALIBRATED_OUTER_IOU_MIN=BADGE_GLYPH_CALIBRATED_OUTER_IOU_MIN,
            BADGE_GLYPH_CALIBRATED_OUTER_L1_MAX=BADGE_GLYPH_CALIBRATED_OUTER_L1_MAX,
            BADGE_GLYPH_CALIBRATED_OUTER_MSE_MAX=BADGE_GLYPH_CALIBRATED_OUTER_MSE_MAX,
            BADGE_GLYPH_CALIBRATED_OUTER_XOR_MAX=BADGE_GLYPH_CALIBRATED_OUTER_XOR_MAX,
        )

    @staticmethod
    def _badge_glyph_calibration_inner_match(
        target_geometry: tuple[int, int, int, int, int],
        sample_prototypes: set[tuple[tuple[int, int, int, int, int], int]],
        metrics: dict[str, int | float],
    ) -> bool:
        return _component_function_badge_glyph_calibration_inner_match(
            target_geometry,
            sample_prototypes,
            metrics,
            BADGE_GLYPH_CALIBRATED_INNER_CHANGED_MAX=BADGE_GLYPH_CALIBRATED_INNER_CHANGED_MAX,
            BADGE_GLYPH_CALIBRATED_INNER_IOU_MIN=BADGE_GLYPH_CALIBRATED_INNER_IOU_MIN,
            BADGE_GLYPH_CALIBRATED_INNER_L1_MAX=BADGE_GLYPH_CALIBRATED_INNER_L1_MAX,
            BADGE_GLYPH_CALIBRATED_INNER_MSE_MAX=BADGE_GLYPH_CALIBRATED_INNER_MSE_MAX,
            BADGE_GLYPH_CALIBRATED_INNER_XOR_MAX=BADGE_GLYPH_CALIBRATED_INNER_XOR_MAX,
        )

    def _record_badge_glyph_exemplar_comparisons(
        self, key: tuple[int, int], descriptor: tuple[int, ...], geometry: tuple[int, int, int, int, int], *, maximum_count: int
    ) -> None:
        return self._component_badge_runtime()._record_badge_glyph_exemplar_comparisons(key, descriptor, geometry, maximum_count=maximum_count)

    @staticmethod
    def _normalized_badge_glyph_domain(maximum_count: int, admissible_counts: Sequence[int] | None) -> tuple[int, ...]:
        return _component_function_normalized_badge_glyph_domain(maximum_count, admissible_counts)

    def _badge_glyph_member_calibrated_count(
        self,
        key: tuple[int, int],
        descriptor: tuple[int, ...],
        geometry: tuple[int, int, int, int, int],
        *,
        support_frames: int,
        admissible_counts: tuple[int, ...],
    ) -> int | None:
        return self._component_badge_runtime()._badge_glyph_member_calibrated_count(
            key, descriptor, geometry, support_frames=support_frames, admissible_counts=admissible_counts
        )

    def _badge_glyph_exemplar_count(
        self, key: tuple[int, int], *, maximum_count: int, admissible_counts: Sequence[int] | None = None
    ) -> int | None:
        return self._component_badge_inference()._badge_glyph_exemplar_count(
            key, maximum_count=maximum_count, admissible_counts=admissible_counts
        )

    def _auxiliary_badge_glyph_count(
        self, key: tuple[int, int], *, maximum_count: int, admissible_counts: Sequence[int] | None = None, allow_ocr: bool = True
    ) -> int:
        return self._component_badge_inference()._auxiliary_badge_glyph_count(
            key, maximum_count=maximum_count, admissible_counts=admissible_counts, allow_ocr=allow_ocr
        )

    def _card_recognizer(self, *, customized: bool) -> tuple[EmbeddingCardRecognizer, float, float]:
        return self._component_card_identity_reader()._card_recognizer(customized=customized)

    def _card_visual_family_candidates(
        self, *, stage_number: int, image: Any, box: tuple[int, int, int, int], slot_index: int
    ) -> tuple[int, ...]:
        return self._component_card_identity_reader()._card_visual_family_candidates(
            stage_number=stage_number, image=image, box=box, slot_index=slot_index
        )

    def ensure_arena_main(self, *, require_opponents: bool = True) -> None:
        return self._component_arena_navigation_reader().ensure_arena_main(require_opponents=require_opponents)

    def _dismiss_grade_reward_hint(self, image: Any, observations: Sequence[Any]) -> bool:
        return self._component_arena_navigation_reader()._dismiss_grade_reward_hint(image, observations)

    def enter_team(self, target: TeamTarget) -> None:
        return self._component_arena_navigation_reader().enter_team(target)

    def select_opponent_for_challenge(self, position: int) -> dict[str, Any]:
        return self._component_arena_navigation_reader().select_opponent_for_challenge(position)

    def select_stage(self, target: TeamTarget, stage_number: int) -> None:
        return self._component_arena_navigation_reader().select_stage(target, stage_number)

    def member_slots(self, target: TeamTarget, stage_number: int) -> Sequence[int]:
        return self._component_member_navigation_reader().member_slots(target, stage_number)

    def open_member(self, target: TeamTarget, stage_number: int, slot: int) -> None:
        return self._component_member_navigation_reader().open_member(target, stage_number, slot)

    def read_support_bonus(self, target: TeamTarget) -> float:
        return self._component_member_navigation_reader().read_support_bonus(target)

    _has_support_bonus_heading = staticmethod(_reader_visual._has_support_bonus_heading)

    def _assert_support_bonus_closed(self) -> None:
        return self._component_member_navigation_reader()._assert_support_bonus_closed()

    _support_bonus_from_overlay = staticmethod(_reader_visual._support_bonus_from_overlay)

    def read_params(self, target: TeamTarget, stage_number: int, slot: int) -> Sequence[int]:
        return self._component_member_navigation_reader().read_params(target, stage_number, slot)

    def _read_enhanced_numeric_roi(self, image: Any, roi: tuple[int, int, int, int]) -> tuple[int | None, tuple[str, ...]]:
        return self._component_member_navigation_reader()._read_enhanced_numeric_roi(image, roi)

    @staticmethod
    def _p_item_decision_evidence(
        slot: int, decision: PItemReferenceDecision, *, path: str, resolved_id: int | None = None, screen_slot: int | None = None
    ) -> dict[str, Any]:
        return _component_function_p_item_decision_evidence(slot, decision, path=path, resolved_id=resolved_id, screen_slot=screen_slot)

    def _confirm_p_item_detail(
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
        return self._component_p_item_detail_workflow().confirm_detail(
            box,
            candidate_ids=candidate_ids,
            global_title_scope=global_title_scope,
            visual_tiebreak_ids=visual_tiebreak_ids,
            unrepresented_ids=unrepresented_ids,
            source_images=source_images,
            source_boxes=source_boxes,
            plan=plan,
            slot_index=slot_index,
        )

    def _confirm_p_item_detail_once(
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
        return self._component_p_item_detail_workflow().confirm_detail_once(
            box,
            candidate_ids=candidate_ids,
            global_title_scope=global_title_scope,
            visual_tiebreak_ids=visual_tiebreak_ids,
            unrepresented_ids=unrepresented_ids,
            source_images=source_images,
            source_boxes=source_boxes,
            plan=plan,
            slot_index=slot_index,
        )

    def _assert_p_item_reference_catalog_compatibility(self) -> None:
        return self._component_p_item_read_workflow().assert_catalog_compatibility()

    def _assert_skill_card_reference_catalog_compatibility(self) -> None:
        return self._component_card_identity_reader()._assert_skill_card_reference_catalog_compatibility()

    def _skill_card_forward_drift_for_plan(self, plan: str) -> tuple[int, ...]:
        return self._component_card_identity_reader()._skill_card_forward_drift_for_plan(plan)

    def _skill_card_scope_kwargs(self, slot_index: int, *, plan: str | None = None) -> dict[str, Any]:
        return self._component_card_identity_reader()._skill_card_scope_kwargs(slot_index, plan=plan)

    def _skill_card_scoped_candidates(self, candidate_ids: Sequence[int], *, plan: str, slot_index: int) -> tuple[int, ...]:
        return self._component_card_identity_reader()._skill_card_scoped_candidates(candidate_ids, plan=plan, slot_index=slot_index)

    def _skill_card_catalog_candidates_for_plan(self, plan: str, *, slot_index: int | None = None) -> tuple[int, ...]:
        return self._component_card_identity_reader()._skill_card_catalog_candidates_for_plan(plan, slot_index=slot_index)

    def _skill_card_forward_drift_slot_indices(self, missing_card_ids: Sequence[int], *, plan: str | None = None) -> tuple[int, ...]:
        return self._component_card_identity_reader()._skill_card_forward_drift_slot_indices(missing_card_ids, plan=plan)

    def _start_skill_card_gallery_fallback(self) -> None:
        return self._component_card_identity_reader()._start_skill_card_gallery_fallback()

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
        return self._component_card_identity_reader()._run_skill_card_gallery_detail(
            operation,
            target=target,
            stage_number=stage_number,
            member_slot=member_slot,
            group_index=group_index,
            card_slot=card_slot,
            additional_zero_detail=additional_zero_detail,
        )

    _p_item_interaction_box = staticmethod(_reader_visual._p_item_interaction_box)

    def _capture_stable_p_item_generation(
        self,
        boxes: Sequence[tuple[int, int, int, int]],
        *,
        initial_image: Any | None = None,
        timeout_seconds: float = 1.5,
        interval_seconds: float = 0.08,
    ) -> tuple[tuple[Any, Any, Any], tuple[float, ...], int]:
        return self._component_p_item_page_observation().capture_stable_generation(
            boxes, initial_image=initial_image, timeout_seconds=timeout_seconds, interval_seconds=interval_seconds
        )

    def read_p_item_ids(self, target: TeamTarget, stage_number: int, slot: int) -> Sequence[int]:
        return self._component_p_item_read_workflow().read_ids(target, stage_number, slot)

    def prepare_skill_card_group(self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int) -> None:
        return self._component_card_layout_reader().prepare_skill_card_group(target, stage_number, member_slot, group_index)

    _card_rows_shifted = staticmethod(_reader_visual._card_rows_shifted)

    _card_content_generation_deltas = staticmethod(_reader_visual._card_content_generation_deltas)

    @classmethod
    def _card_content_generation_shifted(
        cls,
        actual: Sequence[tuple[str, Any]],
        expected: Sequence[tuple[str, Any]],
        *,
        maximum_mean_absolute_error: float = CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR,
    ) -> bool:
        return _component_function_card_content_generation_shifted(
            cls, actual, expected, maximum_mean_absolute_error=maximum_mean_absolute_error
        )

    _aligned_card_content_query = staticmethod(_reader_visual._aligned_card_content_query)

    @classmethod
    def _fixed_slot_aligned_content_proof(
        cls, source_frames: Sequence[Any], current_image: Any, box: tuple[int, int, int, int], visual_group: str
    ) -> dict[str, Any]:
        return _component_function_fixed_slot_aligned_content_proof(cls, source_frames, current_image, box, visual_group)

    _reference_business_ids = staticmethod(_reader_visual._reference_business_ids)

    _reference_visual_groups = staticmethod(_reader_visual._reference_visual_groups)

    _reference_identity_candidates = staticmethod(_reader_visual._reference_identity_candidates)

    @classmethod
    def _reference_identity_projection_is_strict(cls, identity: dict[str, Any]) -> bool:
        return _component_function_reference_identity_projection_is_strict(cls, identity)

    @classmethod
    def _stable_reference_identity_candidates(cls, identities: Sequence[dict[str, Any]]) -> tuple[tuple[int, str], ...]:
        return _component_function_stable_reference_identity_candidates(cls, identities)

    @classmethod
    def _stable_reference_visual_group(cls, identities: Sequence[dict[str, Any]]) -> str | None:
        return _component_function_stable_reference_visual_group(cls, identities)

    def _source_visual_group_matches_expected_card(
        self, source_visual_group: str, expected_card_id: int | None, source_business_ids: Sequence[int]
    ) -> bool:
        return self._component_card_layout_reader()._source_visual_group_matches_expected_card(
            source_visual_group, expected_card_id, source_business_ids
        )

    def _fixed_gallery_visual_groups_for_business_id(self, business_id: int | None) -> tuple[str, ...] | None:
        return self._component_card_layout_reader()._fixed_gallery_visual_groups_for_business_id(business_id)

    _unique_reference_business_id = staticmethod(_reader_visual._unique_reference_business_id)

    def _card_content_generation_signatures(
        self, image: Any, rows: dict[int, tuple[tuple[int, int, int, int], ...]]
    ) -> tuple[dict[int, tuple[tuple[str, Any], ...]], dict[int, tuple[dict[str, Any], ...]]]:
        return self._component_card_layout_reader()._card_content_generation_signatures(image, rows)

    def _capture_stable_card_groups(
        self,
        group_indices: Sequence[int],
        *,
        timeout_seconds: float = 1.5,
        maximum_timeout_seconds: float = 2.5,
        confirmation_grace_seconds: float = 0.6,
        interval_seconds: float = 0.08,
        allow_fixed_secondary: bool = False,
    ) -> None:
        return self._component_card_layout_reader()._capture_stable_card_groups(
            group_indices,
            timeout_seconds=timeout_seconds,
            maximum_timeout_seconds=maximum_timeout_seconds,
            confirmation_grace_seconds=confirmation_grace_seconds,
            interval_seconds=interval_seconds,
            allow_fixed_secondary=allow_fixed_secondary,
        )

    def _resolve_zero_card_reference_ids(self, group_index: int, slot_indices: Sequence[int], *, plan: str) -> dict[int, int]:
        return self._component_card_identity_reader()._resolve_zero_card_reference_ids(group_index, slot_indices, plan=plan)

    def _stable_zero_card_embedding_family_candidates(self, group_index: int, slot_index: int, *, plan: str) -> tuple[int, ...]:
        return self._component_card_identity_reader()._stable_zero_card_embedding_family_candidates(group_index, slot_index, plan=plan)

    def _refresh_visible_card_groups(self, requested_group: int) -> tuple[Any, Any, Any]:
        return self._component_card_layout_reader()._refresh_visible_card_groups(requested_group)

    def _confirm_clicked_skill_card_id(
        self,
        detail_text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = None,
        detail_image: Any | None = None,
        source_card_box: tuple[int, int, int, int] | None = None,
        source_card_slot: int | None = None,
    ) -> int:
        return self._component_detail_title_identity()._confirm_clicked_skill_card_id(
            detail_text,
            candidate_ids,
            expected_customization_count=expected_customization_count,
            source_group_index=source_group_index,
            detail_image=detail_image,
            source_card_box=source_card_box,
            source_card_slot=source_card_slot,
        )

    def _recover_missing_upgrade_title(self, image: Any, key: tuple[int, int], source_box: Any, base_id: int, expected_id: int) -> int | None:
        return self._component_detail_title_identity()._recover_missing_upgrade_title(image, key, source_box, base_id, expected_id)

    def _confirm_clicked_skill_card_id_from_title(
        self,
        detail_text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = None,
        detail_image: Any | None = None,
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> int:
        return self._component_detail_title_identity()._confirm_clicked_skill_card_id_from_title(
            detail_text,
            candidate_ids,
            expected_customization_count=expected_customization_count,
            source_group_index=source_group_index,
            detail_image=detail_image,
            source_card_box=source_card_box,
        )

    def _resolve_proven_skill_card_title(self, title_text: str, *, allow_one_character: bool = True) -> int:
        return self._component_detail_title_identity()._resolve_proven_skill_card_title(title_text, allow_one_character=allow_one_character)

    def _same_proven_skill_card_title(self, first: str, second: str, card_id: int) -> bool:
        return self._component_detail_title_identity()._same_proven_skill_card_title(first, second, card_id)

    def _normalized_skill_card_ocr_lines(self, text: str) -> tuple[str, ...]:
        return self._component_detail_title_identity()._normalized_skill_card_ocr_lines(text)

    def _stable_skill_card_source_ocr_counts(self, group_index: int) -> dict[str, int]:
        return self._component_detail_capture_evidence()._stable_skill_card_source_ocr_counts(group_index)

    def _cached_full_frame_ocr_evidence(self, image: Any) -> _FullFrameOcrEvidence | None:
        return self._component_detail_capture_evidence()._cached_full_frame_ocr_evidence(image)

    def _record_full_frame_ocr_cache_hit(self, evidence: _FullFrameOcrEvidence) -> None:
        return self._component_detail_capture_evidence()._record_full_frame_ocr_cache_hit(evidence)

    def _with_full_frame_ocr_kind(self, kind: str, operation: Any, *args: Any, **kwargs: Any) -> Any:
        return self._component_detail_capture_evidence()._with_full_frame_ocr_kind(kind, operation, *args, **kwargs)

    def _full_frame_ocr_evidence_for(self, image: Any) -> _FullFrameOcrEvidence:
        return self._component_detail_capture_evidence()._full_frame_ocr_evidence_for(image)

    _skill_card_title_row_has_neutral_ink = staticmethod(_reader_visual._skill_card_title_row_has_neutral_ink)

    def _trusted_skill_card_title_rows(
        self, image: Any, *, candidate_card_ids: Sequence[int] = (), source_card_box: tuple[int, int, int, int] | None = None
    ) -> tuple[str, ...]:
        return self._component_detail_title_identity()._trusted_skill_card_title_rows(
            image, candidate_card_ids=candidate_card_ids, source_card_box=source_card_box
        )

    @classmethod
    def _skill_card_title_ocr_segments(cls, items: Sequence[Any]) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
        return _component_function_skill_card_title_ocr_segments(items, spatial_ocr_rows=cls._spatial_ocr_rows)

    _skill_card_detail_overlay_guard_boxes = staticmethod(_reader_visual._skill_card_detail_overlay_guard_boxes)

    def _authoritative_skill_card_title_text(
        self,
        group_index: int | None,
        detail_image: Any | None,
        *,
        candidate_card_ids: Sequence[int] = (),
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> str | None:
        return self._component_detail_title_identity()._authoritative_skill_card_title_text(
            group_index, detail_image, candidate_card_ids=candidate_card_ids, source_card_box=source_card_box
        )

    def _recover_unmatched_skill_card_title(
        self, group_index: int, image: Any, candidate_ids: Sequence[int], source_card_box: tuple[int, int, int, int]
    ) -> str | None:
        return self._component_detail_title_identity()._recover_unmatched_skill_card_title(group_index, image, candidate_ids, source_card_box)

    def _isolated_skill_card_title_key(self, image: Any, group: int, box: Any) -> tuple:
        return self._component_detail_title_identity()._isolated_skill_card_title_key(image, group, box)

    def _isolated_skill_card_title_entry(self, image: Any, group: int, box: Any) -> Any:
        return self._component_detail_title_identity()._isolated_skill_card_title_entry(image, group, box)

    def _recover_isolated_skill_card_title(self, group: int, image: Any, box: Any) -> str | None:
        return self._component_detail_title_identity()._recover_isolated_skill_card_title(group, image, box)

    def _skill_card_frame_title_pattern(self, image: Any, card_id: int) -> str:
        return self._component_detail_title_identity()._skill_card_frame_title_pattern(image, card_id)

    def _skill_detail_visual_ids(self, key: tuple[int, int]) -> tuple[int, ...]:
        return self._component_detail_lifecycle_reader()._skill_detail_visual_ids(key)

    def _skill_detail_other_source_slots(self, key: tuple[int, int], card_id: int) -> list[tuple[int, int]]:
        return self._component_detail_lifecycle_reader()._skill_detail_other_source_slots(key, card_id)

    def _accept_skill_detail_open_id(self, key: tuple[int, int], card_id: int, attempt: int) -> bool:
        return self._component_detail_lifecycle_reader()._accept_skill_detail_open_id(key, card_id, attempt)

    def _assert_skill_detail_open_id(self, key: tuple[int, int] | None, card_id: int) -> None:
        return self._component_detail_lifecycle_reader()._assert_skill_detail_open_id(key, card_id)

    def _skill_card_source_box(self, key: tuple[int, int] | None) -> tuple[int, int, int, int] | None:
        return self._component_detail_lifecycle_reader()._skill_card_source_box(key)

    def open_skill_card(
        self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int, expected_customization_count: int
    ) -> None:
        return self._component_detail_lifecycle_reader().open_skill_card(
            target, stage_number, member_slot, group_index, card_slot, expected_customization_count
        )

    def _open_skill_card_once(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        expected_customization_count: int,
        *,
        contact_start_index: int = 0,
    ) -> None:
        return self._component_detail_lifecycle_reader()._open_skill_card_once(
            target, stage_number, member_slot, group_index, card_slot, expected_customization_count, contact_start_index=contact_start_index
        )

    _skill_card_interaction_box = staticmethod(_reader_visual._skill_card_interaction_box)

    _skill_card_retry_interaction_box = staticmethod(_reader_visual._skill_card_retry_interaction_box)

    def _open_detail_with_kind(self, kind: str, *args: Any) -> None:
        return self._component_detail_lifecycle_reader()._open_detail_with_kind(kind, *args)

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
        allow_zero_without_badge_count: bool = False,
        candidate_ids_override: Sequence[int] | None = None,
    ) -> ClickedSkillCard:
        return self._component_detail_lifecycle_reader()._infer_badge_card_from_detail(
            target,
            stage_number,
            member_slot,
            group_index,
            card_slot,
            image,
            box,
            expected_customization_count,
            detail_kind,
            allow_zero_without_badge_count=allow_zero_without_badge_count,
            candidate_ids_override=candidate_ids_override,
        )

    def _infer_zero_card_identity_from_detail(
        self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int, candidate_ids: Sequence[int]
    ) -> ClickedSkillCard:
        return self._component_detail_lifecycle_reader()._infer_zero_card_identity_from_detail(
            target, stage_number, member_slot, group_index, card_slot, candidate_ids
        )

    def _infer_zero_card_identity_from_detail_once(
        self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int, candidate_ids: Sequence[int]
    ) -> ClickedSkillCard:
        return self._component_detail_lifecycle_reader()._infer_zero_card_identity_from_detail_once(
            target, stage_number, member_slot, group_index, card_slot, candidate_ids
        )

    def read_skill_card_id_hints(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        customization_counts: Sequence[int],
        excluded_duplicate_flags: Sequence[bool] | None = None,
    ) -> Sequence[int]:
        return self._component_card_identity_reader().read_skill_card_id_hints(
            target, stage_number, member_slot, group_index, customization_counts, excluded_duplicate_flags
        )

    def _targeted_duplicate_marker_observations(
        self,
        frames: Sequence[Any],
        row: Sequence[tuple[int, int, int, int]],
        target_slots: Sequence[int],
        *,
        phase: str,
        visual_features_by_slot: Sequence[Sequence[dict[str, Any]]] | None = None,
    ) -> tuple[tuple[tuple[bool, ...], ...], list[dict[str, Any]]]:
        return self._component_card_layout_reader()._targeted_duplicate_marker_observations(
            frames, row, target_slots, phase=phase, visual_features_by_slot=visual_features_by_slot
        )

    def read_skill_card_excluded_duplicate_flags(
        self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int
    ) -> Sequence[bool]:
        return self._component_card_layout_reader().read_skill_card_excluded_duplicate_flags(target, stage_number, member_slot, group_index)

    def current_skill_card_excluded_duplicate_flags(self, group_index: int) -> Sequence[bool]:
        return self._component_card_layout_reader().current_skill_card_excluded_duplicate_flags(group_index)

    def read_skill_card_customization_counts(self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int) -> Sequence[int]:
        return self._component_card_customizations_reader().read_skill_card_customization_counts(target, stage_number, member_slot, group_index)

    def read_clicked_skill_card(
        self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int, expected_customization_count: int
    ) -> ClickedSkillCard:
        return self._component_card_customizations_reader().read_clicked_skill_card(
            target, stage_number, member_slot, group_index, card_slot, expected_customization_count
        )

    def _resolve_clicked_card_text(
        self,
        candidate_ids: Sequence[int],
        text: str,
        *,
        expected_customization_count: int | None,
        allow_zero_without_badge_count: bool = False,
        allow_auxiliary_badge_glyph: bool = True,
        generic_cost_fallback_policy: str | None = None,
        allow_generic_cost_fallback: bool = False,
        generic_cost_coverage_context: tuple[str, str, int] | None = None,
        key: tuple[int, int] | None = None,
    ) -> ClickedSkillCard:
        return self._component_card_customizations_reader()._resolve_clicked_card_text(
            candidate_ids,
            text,
            expected_customization_count=expected_customization_count,
            allow_zero_without_badge_count=allow_zero_without_badge_count,
            allow_auxiliary_badge_glyph=allow_auxiliary_badge_glyph,
            generic_cost_fallback_policy=generic_cost_fallback_policy,
            allow_generic_cost_fallback=allow_generic_cost_fallback,
            generic_cost_coverage_context=generic_cost_coverage_context,
            key=key,
        )

    def read_clicked_skill_card_unconstrained(
        self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int
    ) -> ClickedSkillCard:
        return self._component_card_customizations_reader().read_clicked_skill_card_unconstrained(
            target, stage_number, member_slot, group_index, card_slot
        )

    def _measure_card_face_generic_cost(self, key: tuple[int, int], card_id: int) -> tuple[int, int, int] | None:
        return self._component_card_customizations_reader()._measure_card_face_generic_cost(key, card_id)

    def _measure_optional_card_face_generic_cost(self, key: tuple[int, int], card_id: int) -> tuple[int, int, int] | None:
        return self._component_card_customizations_reader()._measure_optional_card_face_generic_cost(key, card_id)

    def _certify_card_face_generic_cost(self, key: tuple[int, int], card_id: int, customizations: Any) -> tuple[int, int, int] | None:
        return self._component_card_customizations_reader()._certify_card_face_generic_cost(key, card_id, customizations)

    def _detail_identity_observation(
        self, key: tuple[int, int], candidate_ids: Sequence[int], resolved: ClickedSkillCard
    ) -> _DetailIdentityObservation | None:
        return self._component_detail_lifecycle_reader()._detail_identity_observation(key, candidate_ids, resolved)

    def _detail_identity_transaction(self, key: tuple[int, int]) -> DetailIdentityTransaction:
        return self._component_detail_lifecycle_reader()._detail_identity_transaction(key)

    def _record_detail_identity_proof(
        self, key: tuple[int, int], resolved: ClickedSkillCard, observations: Sequence[_DetailIdentityObservation]
    ) -> None:
        return self._component_detail_lifecycle_reader()._record_detail_identity_proof(key, resolved, observations)

    def _detail_identity_proof_matches(
        self, key: tuple[int, int], expected_card_id: int | None, source_card_box: tuple[int, int, int, int] | None
    ) -> bool:
        return self._component_detail_lifecycle_reader()._detail_identity_proof_matches(key, expected_card_id, source_card_box)

    _detail_identity_proof_diagnostic = staticmethod(detail_identity_proof_diagnostic)

    def _recover_failed_skill_card_effect_text(
        self,
        key: tuple[int, int],
        card_id: int,
        error: ArenaReaderError,
        *,
        expected_customization_count: int | None,
        allow_zero_without_badge_count: bool,
    ) -> tuple[str, ClickedSkillCard] | None:
        return self._component_detail_effect_ocr()._recover_failed_skill_card_effect_text(
            key,
            card_id,
            error,
            expected_customization_count=expected_customization_count,
            allow_zero_without_badge_count=allow_zero_without_badge_count,
        )

    def _recover_failed_skill_card_body_text(self, key: tuple[int, int]) -> str | None:
        return self._component_detail_effect_ocr()._recover_failed_skill_card_body_text(key)

    def _read_resolved_card_detail(
        self,
        key: tuple[int, int],
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        timeout_seconds: float = 1.50,
        allow_zero_without_badge_count: bool = False,
        target: TeamTarget | None = None,
        stage_number: int | None = None,
        member_slot: int | None = None,
    ) -> ClickedSkillCard:
        """Reopen a prematurely vanished detail using only its unused contact."""
        return SkillDetailRetrySession(self, clock=time, logger=logger).run(
            key,
            candidate_ids,
            expected_customization_count=expected_customization_count,
            timeout_seconds=timeout_seconds,
            allow_zero_without_badge_count=allow_zero_without_badge_count,
            target=target,
            stage_number=stage_number,
            member_slot=member_slot,
        )

    def _read_resolved_card_detail_once(
        self,
        key: tuple[int, int],
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        timeout_seconds: float = 1.50,
        allow_zero_without_badge_count: bool = False,
        target: TeamTarget | None = None,
        stage_number: int | None = None,
        member_slot: int | None = None,
    ) -> ClickedSkillCard:
        return SkillDetailSession(
            self,
            key,
            candidate_ids,
            expected_customization_count=expected_customization_count,
            timeout_seconds=timeout_seconds,
            allow_zero_without_badge_count=allow_zero_without_badge_count,
            target=target,
            stage_number=stage_number,
            member_slot=member_slot,
            clock=time,
        ).run()

    def _skill_card_detail_disappeared(
        self,
        key: tuple[int, int],
        opened_image: Any,
        opened_capture_time: float | None,
        transaction_token: int | None,
        source_frames: tuple[Any, ...] | None,
        recent_frames: tuple[tuple[float, Any], ...],
    ) -> bool:
        return self._component_detail_lifecycle_reader()._skill_card_detail_disappeared(
            key, opened_image, opened_capture_time, transaction_token, source_frames, recent_frames
        )

    @staticmethod
    def _same_clicked_card_resolution(left: ClickedSkillCard | None, right: ClickedSkillCard) -> bool:
        return _component_function_same_clicked_card_resolution(left, right)

    @staticmethod
    def _detail_confirmation_wait_message(reason: str, *, expected_customization_count: int | None, resolved_count: int) -> str:
        return _component_function_detail_confirmation_wait_message(
            reason, expected_customization_count=expected_customization_count, resolved_count=resolved_count
        )

    def _record_detail_semantic_confirmation(
        self, key: tuple[int, int], resolved: ClickedSkillCard, *, reason: str, resolution_conflicts: int
    ) -> None:
        return self._component_detail_lifecycle_reader()._record_detail_semantic_confirmation(
            key, resolved, reason=reason, resolution_conflicts=resolution_conflicts
        )

    def _skill_card_effect_roi_text(self, image: Any) -> str:
        return self._component_detail_effect_ocr()._skill_card_effect_roi_text(image)

    def _confirm_skill_card_detail_identity(
        self,
        text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = None,
        detail_image: Any | None = None,
        source_card_box: tuple[int, int, int, int] | None = None,
        source_card_slot: int | None = None,
    ) -> int:
        return self._component_detail_title_identity()._confirm_skill_card_detail_identity(
            text,
            candidate_ids,
            expected_customization_count=expected_customization_count,
            source_group_index=source_group_index,
            detail_image=detail_image,
            source_card_box=source_card_box,
            source_card_slot=source_card_slot,
        )

    def _skill_card_detail_ocr_text(self, image: Any, *, phase: str) -> str:
        return self._component_detail_effect_ocr()._skill_card_detail_ocr_text(image, phase=phase)

    _title_anchored_effect_roi = staticmethod(_reader_visual._title_anchored_effect_roi)

    _background_parameter_column_left = staticmethod(_reader_visual._background_parameter_column_left)

    def _title_anchored_effect_roi_without_background_parameters(
        self, frame_width: int, frame_height: int, title_box: tuple[int, int, int, int], items: Sequence[Any]
    ) -> tuple[int, int, int, int]:
        return self._component_detail_effect_ocr()._title_anchored_effect_roi_without_background_parameters(
            frame_width, frame_height, title_box, items
        )

    def _isolated_skill_card_title_atoms(self, image: Any, title_pattern: str) -> Any:
        return self._component_detail_title_identity()._isolated_skill_card_title_atoms(image, title_pattern)

    def _skill_card_title_anchor_evidence(self, image: Any, title_pattern: str) -> tuple[list[Any], list[Any]]:
        return self._component_detail_capture_evidence()._skill_card_title_anchor_evidence(image, title_pattern)

    def _skill_card_title_anchored_effect_roi_text(self, image: Any, card_id: int) -> str:
        return self._component_detail_effect_ocr()._skill_card_title_anchored_effect_roi_text(image, card_id)

    def _skill_card_title_anchored_enhanced_effect_roi_text(self, image: Any, card_id: int) -> str:
        return self._component_detail_effect_ocr()._skill_card_title_anchored_enhanced_effect_roi_text(image, card_id)

    def _merge_skill_card_effect_detail_views(self, card_id: int, detail_texts: Sequence[str]) -> str:
        return self._component_detail_effect_ocr()._merge_skill_card_effect_detail_views(card_id, detail_texts)

    def _record_detail_count_override(self, key: tuple[int, int], observed_badge_count: int, resolved: ClickedSkillCard) -> None:
        return self._component_card_customizations_reader()._record_detail_count_override(key, observed_badge_count, resolved)

    def _record_cost_customization_fallback(
        self, key: tuple[int, int], resolved: ClickedSkillCard, *, target: TeamTarget | None, stage_number: int | None, member_slot: int | None
    ) -> None:
        return self._component_card_customizations_reader()._record_cost_customization_fallback(
            key, resolved, target=target, stage_number=stage_number, member_slot=member_slot
        )

    def close_skill_card(self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int) -> None:
        return self._component_detail_lifecycle_reader().close_skill_card(target, stage_number, member_slot, group_index, card_slot)

    def _close_skill_card_once(self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int, card_slot: int) -> None:
        return self._component_detail_lifecycle_reader()._close_skill_card_once(target, stage_number, member_slot, group_index, card_slot)

    def close_member(
        self, target: TeamTarget, stage_number: int, *, recovery_image: Any = None, recovery_detail: dict[str, Any] | None = None
    ) -> None:
        return self._component_member_navigation_reader().close_member(
            target, stage_number, recovery_image=recovery_image, recovery_detail=recovery_detail
        )

    def _retain_failed_skill_detail_identity(self, key: tuple[int, int]) -> None:
        return self._component_member_navigation_reader()._retain_failed_skill_detail_identity(key)

    def _member_recovery_page_matches(self, items: Sequence[Any], stage_number: int, detail: dict[str, Any]) -> bool:
        return self._component_member_navigation_reader()._member_recovery_page_matches(items, stage_number, detail)

    def _read_member_recovery_page(self, image: Any = None) -> tuple[Any, Sequence[Any]]:
        return ArenaPageFlow(_ArenaPageFlowPort(self), time, logger).read_member_recovery_page(
            image,
        )

    def recover_member_preview(self, target: TeamTarget, stage_number: int, member_slot: int, *, error_code: str) -> None:
        return self._component_member_navigation_reader().recover_member_preview(target, stage_number, member_slot, error_code=error_code)

    def _reset_member_card_state(self) -> None:
        """Discard card geometry, hints and frames that belong to the prior member."""

        self._card_stage_plan = None
        self._card_rows.clear()
        detail_state = card_detail_state(self)
        detail_state.clear_open_identity()
        self._card_predictions.clear()
        self._card_candidate_groups.clear()
        self._card_images.clear()
        self._card_count_frames.clear()
        self._card_identity_frames.clear()
        self._card_source_ocr_counts.clear()
        frame_evidence_state(self).clear_member_observations()
        self._card_source_guard_frames.clear()
        getattr(self, "_card_source_guard_signature_cache", {}).clear()
        self._card_restoration_signatures.clear()
        self._card_excluded_duplicate_flags.clear()
        self._card_empty_flags.clear()
        self._card_empty_diagnostics.clear()
        self._card_duplicate_marker_diagnostics.clear()
        self._badge_count_diagnostics.clear()
        self._detail_count_overrides.clear()
        self._detail_semantic_confirmations.clear()
        self._detail_identity_proofs.clear()
        self._detail_title_disambiguations.clear()
        self._badge_local_results.clear()
        self._badge_glyph_observations.clear()
        self._badge_glyph_count_diagnostics.clear()
        getattr(self, "_badge_glyph_runtime_labels", {}).clear()
        self._inferred_clicked_cards.clear()
        self._active_inferred_clicked_card = None
        detail_state.clear_member_observations()
        self._p_item_diagnostics = ()
        self._p_item_generation_evidence.clear()
        for key in tuple(self._card_transaction_started):
            self._finish_card_transaction(key, failed=True)
        detail_state.clear_transaction_metadata()
        self._secondary_fixed_slot_fallback_enabled = False
        self._secondary_presence_diagnostics = ()
        self._secondary_presence_cache.clear()
        self._card_content_generation_diagnostics = ()
        self._card_face_cost_diagnostics.clear()
        self._card_face_cost_optional_errors.clear()
        getattr(self, "_card_face_cost_failed_sources", {}).clear()
        self._cost_customization_fallbacks.clear()
        self._zero_card_identity_fusion_diagnostics.clear()
        self._zero_card_embedding_detail_candidates.clear()
        self._zero_card_detail_candidates.clear()

    def leave_team(self, target: TeamTarget) -> None:
        return self._component_arena_navigation_reader().leave_team(target)

    def finish_progressive_read(self) -> None:
        return self._component_arena_navigation_reader().finish_progressive_read()

    def recover_to_arena_main(self, *, require_opponents: bool = True,
                             initial_observation: RecoveryPageObservation | None = None) -> None:
        return self._component_arena_navigation_reader().recover_to_arena_main(
            require_opponents=require_opponents, initial_observation=initial_observation)

    def _recover_arena_main(
        self,
        *,
        maximum_steps: int,
        error_code: str,
        require_opponents: bool,
        initial_observation: RecoveryPageObservation | None = None,
    ) -> None:
        return ArenaPageFlow(_ArenaPageFlowPort(self), time, logger).recover_arena_main(
            maximum_steps=maximum_steps,
            error_code=error_code,
            require_opponents=require_opponents,
            initial_observation=initial_observation,
        )

    def _dismiss_contest_details_items(self, image: Any, items: Sequence[Any]) -> bool:
        return ArenaPageFlow(_ArenaPageFlowPort(self), time, logger).dismiss_contest_details_items(
            image,
            items,
        )

    def _dismiss_known_blocking_overlay(self, image: Any, *, items: Sequence[Any] | None = None) -> bool:
        return ArenaPageFlow(_ArenaPageFlowPort(self), time, logger).dismiss_known_blocking_overlay(
            image,
            items=items,
        )

    def _retry_transient_communication_items(self, items: Sequence[Any]) -> bool:
        return ArenaPageFlow(_ArenaPageFlowPort(self), time, logger).retry_transient_communication_items(
            items,
        )

    def _check_cancelled(self) -> None:
        return self._component_io_reader()._check_cancelled()

    def _sleep(self, seconds: float) -> None:
        return self._component_io_reader()._sleep(seconds)

    def _run_recognition(self, *args, **kwargs):
        return self._component_io_reader()._run_recognition(*args, **kwargs)

    def _capture(self) -> Any:
        return self._component_io_reader()._capture()

    def _recognize(self, entry: str, image: Any) -> list[Any]:
        return self._component_io_reader()._recognize(entry, image)

    def _ocr(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = None, only_rec: bool = False) -> list[Any]:
        return self._component_io_reader()._ocr(image, expected, roi=roi, only_rec=only_rec)

    _matching_ocr_items = staticmethod(_reader_visual._matching_ocr_items)

    def _ocr_detail(self, image: Any, expected: str, *, roi: tuple[int, int, int, int] | None = None, only_rec: bool = False) -> Any:
        return self._component_io_reader()._ocr_detail(image, expected, roi=roi, only_rec=only_rec)

    _spatial_ocr_rows = staticmethod(_reader_visual._spatial_ocr_rows)

    _spatial_ocr_text = staticmethod(_reader_visual._spatial_ocr_text)

    def _full_ocr_text(self, image: Any, *, spatial_reading_order: bool = False) -> str:
        return self._component_io_reader()._full_ocr_text(image, spatial_reading_order=spatial_reading_order)

    _p_item_detail_panel_rows = staticmethod(_reader_visual._p_item_detail_panel_rows)

    _isolated_p_item_title_row = staticmethod(_reader_visual._isolated_p_item_title_row)

    _p_item_detail_title_text = staticmethod(_reader_visual._p_item_detail_title_text)

    def _p_item_ocr_observation(self, image: Any) -> _PItemOcrObservation:
        return self._component_p_item_page_observation().ocr_observation(image)

    def _click(self, box: tuple[int, int, int, int], *, settle_seconds: float = 0.25) -> None:
        return self._component_io_reader()._click(box, settle_seconds=settle_seconds)

    _box_center_point = staticmethod(_reader_visual._box_center_point)

    def _long_press(self, box: tuple[int, int, int, int]) -> None:
        return self._component_io_reader()._long_press(box)

    def _short_press(self, box: tuple[int, int, int, int], *, duration_ms: int) -> None:
        return self._component_io_reader()._short_press(box, duration_ms=duration_ms)

    def _swipe(self, *, vertical: str, distance_ratio: float = 0.0625, distance_pixels: int | None = None) -> None:
        return self._component_io_reader()._swipe(vertical=vertical, distance_ratio=distance_ratio, distance_pixels=distance_pixels)

    def _back(self, *, image: Any = None) -> None:
        return ArenaPageFlow(_ArenaPageFlowPort(self), time, logger).back(
            image=image,
        )

    def _close_overlay(self) -> None:
        return self._component_io_reader()._close_overlay()

    def _dismiss_skill_card_detail(self, *, image: Any = None) -> None:
        return self._component_detail_lifecycle_reader()._dismiss_skill_card_detail(image=image)

    def _skill_card_detail_close_retry_proven(
        self, group_index: int, expected_card_id: int, source_card_box: tuple[int, int, int, int] | None = None
    ) -> bool:
        return self._component_detail_lifecycle_reader()._skill_card_detail_close_retry_proven(group_index, expected_card_id, source_card_box)

    def _assert_inferred_card_group_visible_after_dismiss(self, group_index: int, *, card_slot: int, expected_card_id: int) -> None:
        return self._component_detail_lifecycle_reader()._assert_inferred_card_group_visible_after_dismiss(
            group_index, card_slot=card_slot, expected_card_id=expected_card_id
        )

    def _assert_stage_navigation(self) -> None:
        return self._component_arena_navigation_reader()._assert_stage_navigation()

    def _wait_for_stage_member_list(
        self,
        expected_total_counts: tuple[int, ...],
        timeout_seconds: float = 3.0,
    ) -> None:
        return ArenaPageFlow(_ArenaPageFlowPort(self), time, logger).wait_for_stage_member_list(
            expected_total_counts,
            timeout_seconds=timeout_seconds,
        )

    def _stage_total_anchors(self, image: Any) -> tuple[tuple[int, int, int, int], ...]:
        return self._component_arena_navigation_reader()._stage_total_anchors(image)

    def _team_stage_total_anchors(self, target: TeamTarget, image: Any) -> tuple[tuple[int, int, int, int], ...]:
        return self._component_arena_navigation_reader()._team_stage_total_anchors(target, image)

    def _arena_main_visible(self) -> bool:
        return self._component_arena_navigation_reader()._arena_main_visible()

    def _arena_page_state(self, image: Any, *, observations: Sequence[Any] | None = None) -> tuple[ArenaPageState, tuple[int, int, int, int, int]]:
        return self._component_arena_navigation_reader()._arena_page_state(image, observations=observations)

    def _read_grade(self, image: Any, *, observations: Sequence[Any] | None = None) -> int:
        return self._component_arena_navigation_reader()._read_grade(image, observations=observations)

    def _assert_member_detail(self, timeout_seconds: float = 2.0) -> None:
        return self._component_member_navigation_reader()._assert_member_detail(timeout_seconds)

    def _assert_p_item_source_restored(
        self, source_images: Sequence[Any], boxes: Sequence[tuple[int, int, int, int]], timeout_seconds: float = 8.0
    ) -> None:
        return self._component_p_item_page_observation().assert_source_restored(source_images, boxes, timeout_seconds)

    _p_item_source_proof_boxes = staticmethod(_reader_visual._p_item_source_proof_boxes)

    def _assert_p_item_source_generation_stable(self, source_images: Sequence[Any], boxes: Sequence[tuple[int, int, int, int]]) -> None:
        return self._component_p_item_page_observation().assert_source_generation_stable(source_images, boxes)

    def _p_item_source_frame_matches(
        self, source_images: Sequence[Any], image: Any, boxes: Sequence[tuple[int, int, int, int]], member_anchors_visible: bool | None = None
    ) -> tuple[bool, tuple[float, ...]]:
        return self._component_p_item_page_observation().source_frame_matches(source_images, image, boxes, member_anchors_visible)

    _p_item_border_only_change = staticmethod(_reader_visual._p_item_border_only_change)

    def _assert_card_group_visible(
        self,
        group_index: int,
        timeout_seconds: float = 2.0,
        *,
        card_slot: int | None = None,
        expected_card_id: int | None = None,
    ) -> None:
        return SourceRestoreSession(
            self,
            group_index,
            timeout_seconds,
            card_slot,
            expected_card_id,
            time,
            signatures=p_item_content_generation_signatures,
            signature_errors=measure_p_item_content_generation_from_signatures,
        ).run()

    def _detect_card_rows(self, image: Any) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
        return self._component_card_layout_reader()._detect_card_rows(image)

    def _validated_card_group_rows(self, image: Any, groups: Sequence[int], **kwargs):
        return self._component_card_layout_reader()._validated_card_group_rows(image, groups, **kwargs)

    def _validated_card_group_row(
        self, image: Any, group_index: int, *, allow_fixed_secondary: bool = False
    ) -> tuple[tuple[int, int, int, int], ...]:
        return self._component_card_layout_reader()._validated_card_group_row(image, group_index, allow_fixed_secondary=allow_fixed_secondary)

    def _secondary_fixed_slot_presence(self, image: Any, box: tuple[int, int, int, int], slot: int) -> dict[str, Any]:
        return self._component_card_layout_reader()._secondary_fixed_slot_presence(image, box, slot)

    _skill_card_empty_observation = staticmethod(_reader_visual._skill_card_empty_observation)

    def read_skill_card_empty_flags(self, target: TeamTarget, stage_number: int, member_slot: int, group_index: int) -> Sequence[bool]:
        return self._component_card_layout_reader().read_skill_card_empty_flags(target, stage_number, member_slot, group_index)

    def _raw_card_candidate_boxes(self, image: Any) -> tuple[tuple[int, int, int, int], ...]:
        return self._component_card_layout_reader()._raw_card_candidate_boxes(image)

    def _detect_raw_card_candidate_boxes(self, image: Any) -> tuple[tuple[int, int, int, int], ...]:
        return self._component_card_layout_reader()._detect_raw_card_candidate_boxes(image)

    def _component_arena_navigation_reader(self) -> ArenaNavigationReader:
        return ArenaNavigationReader(
            self, grade_ocr_value=_grade_ocr_value, normalize_latin_anchor=_normalize_latin_anchor, logger=logger, clock=time
        )

    def _component_badge_calibration(self) -> BadgeCalibration:
        return BadgeCalibration(self)

    def _component_badge_capture(self) -> BadgeCapture:
        return BadgeCapture(
            self,
            clock=time,
            logger=logger,
            reference_root=CARD_REFERENCE_ROOT,
            cost_reference_root=CARD_COST_REFERENCE_ROOT,
            shortlist_result_type=_BadgeShortlistResult,
            fresh_group_timeout_seconds=BADGE_GLYPH_FRESH_GROUP_TIMEOUT_SECONDS,
            fresh_group_interval_seconds=BADGE_GLYPH_FRESH_GROUP_INTERVAL_SECONDS,
            fresh_group_max_frames=BADGE_GLYPH_FRESH_GROUP_MAX_FRAMES,
        )

    def _component_badge_inference(self) -> BadgeInference:
        return BadgeInference(self, clock=time, validated_acceptance_counts=BADGE_GLYPH_OCR_VALIDATED_ACCEPTANCE_COUNTS)

    def _component_badge_runtime(self) -> BadgeRuntime:
        return BadgeRuntime(self)

    def _component_card_customizations_reader(self) -> CardCustomizationsReader:
        return CardCustomizationsReader(self, logger=logger, clock=time, validate_cost_fallback=validate_conservative_cost_fallback)

    def _component_card_identity_reader(self) -> CardIdentityReader:
        return CardIdentityReader(self, arena_card_model_root=ARENA_CARD_MODEL_ROOT, card_model_root=CARD_MODEL_ROOT, logger=logger, clock=time)

    def _component_card_layout_reader(self) -> CardLayoutReader:
        return CardLayoutReader(
            self,
            duplicate_marker_visual_features=duplicate_marker_visual_features,
            is_duplicate_marker_visual_candidate=is_duplicate_marker_visual_candidate,
            clock=time,
        )

    def _component_detail_capture_evidence(self) -> DetailCaptureEvidence:
        return DetailCaptureEvidence(self, clock=time)

    def _component_detail_effect_ocr(self) -> DetailEffectOcr:
        return DetailEffectOcr(
            self,
            clock=time,
            logger=logger,
            recover_distant_number_text=recover_distant_number_text,
            recover_wrapped_signed_text=recover_wrapped_signed_text,
            recover_skill_effect_body=recover_skill_effect_body,
        )

    def _component_detail_lifecycle_reader(self) -> DetailLifecycleReader:
        return DetailLifecycleReader(
            self,
            io=self,
            content_error_limit=CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR,
            logger=logger,
            source_signature_errors=measure_p_item_content_generation_from_signatures,
            source_signatures=p_item_content_generation_signatures,
            clock=time,
        )

    def _component_detail_title_identity(self) -> DetailTitleIdentity:
        return DetailTitleIdentity(self, clock=time, logger=logger)

    def _component_io_reader(self) -> IoReader:
        return IoReader(self, action_types=JActionType, click_action=JClick, clock=time)

    def _component_member_navigation_reader(self) -> MemberNavigationReader:
        return MemberNavigationReader(self, classify_member_slot=classify_member_slot, logger=logger, clock=time)

    def _component_p_item_detail_workflow(self) -> PItemDetailWorkflow:
        return PItemDetailWorkflow(
            self, clock=time, session_factory=PItemDetailSession, probe_factory=RecognitionProbe, probe_root=DEFAULT_CHALLENGE_RECORD_ROOT
        )

    def _component_p_item_page_observation(self) -> PItemPageObservation:
        return PItemPageObservation(self, clock=time)

    def _component_p_item_read_workflow(self) -> PItemReadWorkflow:
        return PItemReadWorkflow(
            self, clock=time, logger=logger, reference_root=P_ITEM_REFERENCE_ROOT, reader_factory=Task085PItemReader.from_model_root
        )

    def _component_telemetry_reader(self) -> TelemetryReader:
        return TelemetryReader(self, record_root=DEFAULT_CHALLENGE_RECORD_ROOT, logger=logger, clock=time)
