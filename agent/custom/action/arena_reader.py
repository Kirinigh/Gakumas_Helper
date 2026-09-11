"""Administrator-Maa backend for the read-only arena lineup reader.

Normal screenshots stay in memory; failed details retain at most six cached frames.
The backend performs only ordinary navigation,
long-press, detail-card clicks, scrolling and back/close actions.  It deliberately
has no operation capable of starting a contest match.
"""

from __future__ import annotations

import re
import json
import time
import statistics
import unicodedata
from copy import deepcopy
from uuid import uuid4
from typing import Any, Protocol, NamedTuple
from pathlib import Path
from threading import Lock
from collections import Counter, deque
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
    ArenaCatalogError,
    MemberSlotMetrics,
    ArenaEntityCatalog,
    BadgeReferenceError,
    BadgeWorkerSelection,
    BadgeReferenceGallery,
    BadgeWorkerCalibrator,
    ContestSeasonDefinition,
    CustomizationBadgeState,
    GenericCostReferenceError,
    CustomizationBadgeDecision,
    GenericCostReferenceGallery,
    stage_member_cap,
    classify_arena_page,
    classify_member_slot,
    fixed_member_slot_boxes,
    team_stage_total_anchors,
    canonical_skill_card_rows,
    stable_generic_cost_value,
    infer_parameter_label_boxes,
    measure_customization_badge,
    arena_page_allows_team_entry,
    duplicate_marker_visual_features,
    p_item_screen_slot_to_engine_slot,
    p_item_screen_order_to_engine_order,
    stable_customization_badge_presence,
    validate_conservative_cost_fallback,
    is_duplicate_marker_visual_candidate,
    stable_excluded_duplicate_card_flags,
    stable_reference_business_candidates,
    expected_scrolled_secondary_skill_card_row,
    canonical_scrolled_secondary_skill_card_row,
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
from card_selection.model import frame_identifier, isolate_card_candidates
from arena_winrate.recovery import error_retry_box
from arena_winrate.task_log import arena_task_log, diagnostic_logger
from arena_winrate.cancellation import ArenaTaskCancelled, cancellation_for
from arena_winrate.challenge_flow import DEFAULT_CHALLENGE_RECORD_ROOT
from arena_winrate._detail_text_layout import (
    recover_distant_number_text,
    recover_wrapped_signed_text,
)

logger = diagnostic_logger(_base_logger)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _normalize_latin_anchor(value: str) -> str:
    """Normalize OCR-only Latin diacritics without weakening anchor identity."""

    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    ).strip().upper()


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
CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR = 0.75
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


class _FullFrameOcrEvidence(NamedTuple):
    """One immutable OCR observation bound to one frozen capture object."""

    image: Any
    kind: str
    hit: bool
    filtered_items: tuple[Any, ...]
    all_items: tuple[Any, ...]


_PItemPanelRow = tuple[
    str,
    tuple[float, float, float, float],
    tuple[tuple[str, tuple[float, float, float, float]], ...],
]
_NormalizedPItemRow = tuple[
    str,
    str,
    tuple[float, float, float, float],
    tuple[tuple[str, str, tuple[float, float, float, float]], ...],
]


class _PItemOcrObservation(NamedTuple):
    """One OCR call, with separate title and size-excluded source evidence."""

    full_text: str
    panel_text: str
    anchors_visible: bool
    panel_rows: tuple[_PItemPanelRow, ...]
    background_rows: tuple[_PItemPanelRow, ...] = ()
    all_items: tuple[Any, ...] = ()


class _TitleAnchorOcrEvidence(NamedTuple):
    """One title-specific fallback bound to the same frozen capture object."""

    image: Any
    matches: tuple[Any, ...]
    native_items: tuple[Any, ...]


class _TrustedSkillCardTitleRowsEvidence(NamedTuple):
    """One semantic title result bound to one frozen capture object."""

    image: Any
    rows: tuple[str, ...]


class _DetailIdentityObservation(NamedTuple):
    """One exact-title semantic observation from the active click transaction."""

    transaction_token: int
    transaction_started: float
    source_card_box: tuple[int, int, int, int]
    interaction_box: tuple[int, int, int, int]
    contact_released_at: float
    capture_started_at: float
    detail_image: Any
    source_guard_frames: Any
    restoration_signatures: Any
    identity_frames: Any
    title: str
    card_id: int
    customizations: tuple[tuple[str, int], ...]
    evidence_mode: str


class _DetailIdentityProof(NamedTuple):
    """Exact-title observations bound to one physical click transaction."""

    transaction_token: int
    transaction_started: float
    source_card_box: tuple[int, int, int, int]
    interaction_box: tuple[int, int, int, int]
    contact_released_at: float
    capture_started_at: tuple[float, ...]
    detail_images: tuple[Any, ...]
    source_guard_frames: Any
    restoration_signatures: Any
    identity_frames: Any
    title: str
    card_id: int
    customizations: tuple[tuple[str, int], ...]
    resolution_source: str
    evidence_mode: str
    detail_confirmation_reads: int


class _TitleBoundEffectRoiText(str):
    """Merged ROI text carrying how many non-empty OCR passes produced it."""

    observation_count: int

    def __new__(cls, value: str, observation_count: int) -> "_TitleBoundEffectRoiText":
        instance = str.__new__(cls, value)
        instance.observation_count = observation_count
        return instance


def _effect_roi_observation_count(value: str) -> int:
    count = getattr(value, "observation_count", 1)
    return count if type(count) is int and count > 0 else 1


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
    ) -> Sequence[PItemReferenceDecision]: ...


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _box(item: Any) -> tuple[int, int, int, int]:
    value = _value(item, "box")
    if value is None or len(value) != 4:
        raise ArenaReaderError("recognition_box_missing", "recognition result has no four-value box")
    return tuple(int(round(number)) for number in value)


def _text(item: Any) -> str:
    return str(_value(item, "text", ""))


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
    ) -> Sequence[PItemReferenceDecision]:
        def classify(box: tuple[int, int, int, int]) -> PItemReferenceDecision:
            return self.gallery.classify(
                images,
                box,
                plan=plan,
            )

        with ThreadPoolExecutor(max_workers=min(2, len(boxes))) as executor:
            return tuple(executor.map(classify, boxes))


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

    def targeted_duplicate_marker_observations(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        return self._backend._targeted_duplicate_marker_observations(
            *args, **kwargs
        )

    def card_recognizer(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend._card_recognizer(*args, **kwargs)


class MaaArenaReaderBackend:
    """Arena screen backend intended to run inside the elevated Maa agent."""

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
        if known_grade is not None and (
            type(known_grade) is not int or not 1 <= known_grade <= 7
        ):
            raise ValueError("known_grade must be an integer from 1 to 7")
        self.context = context
        self._cancellation = cancellation_for(context)
        self.season = season
        self.catalog = ArenaEntityCatalog.from_bundle(bundle_dir)
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
        self._card_identity_frames: dict[
            int, tuple[tuple[dict[str, Any], ...], ...]
        ] = {}
        self._card_source_ocr_counts: dict[int, dict[str, int]] = {}
        # A capture is immutable evidence.  Full-frame OCR, authoritative title
        # extraction and title-bound ROI geometry must therefore share the raw
        # boxes from that exact capture rather than asking the OCR backend to
        # interpret the same pixels repeatedly.  Identity, not image bytes, is
        # the boundary: a fresh capture must always receive fresh recognition.
        self._full_frame_ocr_evidence: dict[int, _FullFrameOcrEvidence] = {}
        self._title_anchor_ocr_evidence: dict[
            tuple[int, str], _TitleAnchorOcrEvidence
        ] = {}
        self._trusted_skill_card_title_rows_cache: dict[
            tuple[
                int,
                tuple[int, ...],
                tuple[int, int, int, int] | None,
            ],
            _TrustedSkillCardTitleRowsEvidence,
        ] = {}
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
        self._detail_semantic_confirmations: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        self._detail_identity_proofs: dict[
            tuple[int, int], _DetailIdentityProof
        ] = {}
        self._detail_title_disambiguations: list[dict[str, Any]] = []
        self._badge_local_results: dict[int, tuple[Any, ...]] = {}
        self._badge_glyph_observations: dict[
            tuple[int, int], tuple[dict[str, Any], ...]
        ] = {}
        self._badge_glyph_count_diagnostics: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        # Member-local glyph exemplars are learned only from independently
        # confirmed detail semantics.  They deliberately do not outlive the
        # frozen twelve-card layout: a window/layout change or a new member
        # starts with an empty calibration set.
        self._badge_glyph_exemplars: dict[
            tuple[int, ...], dict[str, Any]
        ] = {}
        self._badge_glyph_polluted_descriptors: set[tuple[int, ...]] = set()
        self._badge_glyph_runtime_labels: dict[tuple[int, ...], int] = {}
        # Persist only the low-dimensional, image-free provenance needed to
        # audit whether a member-local exemplar can transfer across cards.
        # The active exemplar table itself is still cleared at member close.
        self._runtime_badge_glyph_exemplar_diagnostics: list[
            dict[str, Any]
        ] = []
        self._runtime_badge_glyph_exemplar_comparisons: list[
            dict[str, Any]
        ] = []
        self._inferred_clicked_cards: dict[tuple[int, int], ClickedSkillCard] = {}
        self._active_inferred_clicked_card: tuple[int, int] | None = None
        self._card_detail_texts: dict[tuple[int, int], str] = {}
        self._card_detail_images: dict[tuple[int, int], Any] = {}
        self._card_detail_last_contact_released_at: dict[
            tuple[int, int], float
        ] = {}
        self._card_detail_capture_started_at: dict[tuple[int, int], float] = {}
        self._p_item_diagnostics: tuple[dict[str, Any], ...] = ()
        self._p_item_generation_evidence: dict[str, Any] = {}
        self._card_recognizers: dict[bool, tuple[EmbeddingCardRecognizer, float, float]] = {}
        self._card_recognizer_lock = Lock()
        self._card_reference_gallery: BadgeReferenceGallery | None = None
        self._card_cost_reference_gallery: GenericCostReferenceGallery | None = None
        self._zero_card_identity_fusion_diagnostics: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        self._zero_card_embedding_detail_candidates: dict[
            tuple[int, int], tuple[int, ...]
        ] = {}
        self._zero_card_detail_candidates: dict[tuple[int, int], tuple[int, ...]] = {}
        self._badge_worker_calibrator = BadgeWorkerCalibrator()
        self._badge_worker_selection: BadgeWorkerSelection | None = None
        self._runtime_timing_seconds: dict[str, float] = {}
        self._runtime_counts: dict[str, int] = {}
        self._runtime_duration_samples: dict[str, list[float]] = {}
        self._card_transaction_started: dict[tuple[int, int], float] = {}
        self._card_transaction_serial = 0
        self._card_transaction_tokens: dict[tuple[int, int], int] = {}
        self._card_transaction_source_boxes: dict[
            tuple[int, int], tuple[int, int, int, int]
        ] = {}
        self._card_transaction_interaction_boxes: dict[
            tuple[int, int], tuple[int, int, int, int]
        ] = {}
        self._card_transaction_kinds: dict[tuple[int, int], str] = {}
        self._card_transaction_ocr_started: dict[
            tuple[int, int], tuple[int, int]
        ] = {}
        self._badge_candidate_detail_checks: list[dict[str, Any]] = []
        self._secondary_fixed_slot_fallback_enabled = False
        self._secondary_presence_diagnostics: tuple[dict[str, Any], ...] = ()
        self._card_content_generation_diagnostics: tuple[dict[str, Any], ...] = ()
        self._card_face_cost_diagnostics: dict[tuple[int, int], dict[str, Any]] = {}
        self._card_face_cost_optional_errors: dict[tuple[int, int], str] = {}
        self._cost_customization_fallbacks: dict[tuple[int, int], dict[str, Any]] = {}
        self._runtime_cost_customization_fallbacks: list[dict[str, Any]] = []
        self._runtime_detail_title_disambiguations: list[dict[str, Any]] = []
        self._secondary_presence_cache: dict[
            int,
            tuple[Any, tuple[dict[str, Any], ...]],
        ] = {}
        self._member_metric_baseline: tuple[
            dict[str, float],
            dict[str, int],
            dict[str, int],
        ] | None = None
        self._challenge_selection_committed = False
        self._p_item_catalog_compatibility_checked = False
        self._p_item_reference_gallery_ids: tuple[int, ...] = ()
        self._p_item_reference_missing_arena_ids: tuple[int, ...] = ()
        self._p_item_reference_unknown_catalog_ids: tuple[int, ...] = ()
        self._p_item_reference_provisional_ids: tuple[int, ...] = ()
        self._skill_card_catalog_compatibility_checked = False
        self._skill_card_reference_gallery_ids: tuple[int, ...] = ()
        self._skill_card_reference_missing_catalog_ids: tuple[int, ...] = ()

    @property
    def grade(self) -> int | None:
        """Return the recognized or explicitly injected Grade for this session."""

        return self._grade

    def _add_timing(self, name: str, elapsed: float) -> None:
        self._runtime_timing_seconds[name] = (
            self._runtime_timing_seconds.get(name, 0.0) + elapsed
        )

    def _increment(self, name: str) -> None:
        self._runtime_counts[name] = self._runtime_counts.get(name, 0) + 1

    def _record_duration_sample(self, name: str, elapsed: float) -> None:
        samples = getattr(self, "_runtime_duration_samples", None)
        if samples is None:
            samples = {}
            self._runtime_duration_samples = samples
        samples.setdefault(name, []).append(elapsed)

    def runtime_sample_cursor(self) -> dict[str, int]:
        return {
            name: len(values)
            for name, values in self._runtime_duration_samples.items()
        }

    def runtime_counts_cursor(self) -> dict[str, int]:
        return dict(self._runtime_counts)

    def record_lineup_read_attempt(self, record: dict[str, Any]) -> None:
        logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True))

    def last_failure_location(self, error: object = None) -> dict[str, Any] | None:
        failure = getattr(self, "_last_detail_failure_location", None)
        if failure is None:
            return None
        original, location = str(failure[0]), failure[1]
        # The evaluation service adds this one fixed prefix to provider errors.
        current = str(error).removeprefix("snapshot provider failed: ")
        member = (
            str(location.get("team_id")), str(location.get("stage_number")),
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
        diagnostic = getattr(self, "_detail_failure_frames", None)
        if diagnostic is None:
            return
        try:
            diagnostic["position"]["card_name"] = self.catalog.skill_card_title(card_id)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    def _begin_detail_diagnostics(self, *, source: Any = None, **position: Any) -> None:
        self._last_detail_failure_location = None
        target = position.pop("target", None)
        if target is not None:
            position.update(team_id=getattr(target, "team_id", None), opponent_position=getattr(target, "opponent_position", None))
        self._detail_failure_frames = {
            "position": position, "source": source,
            "frames": deque(maxlen=5), "actions": deque(maxlen=16),
            "started": time.perf_counter(),
        }

    def begin_member_read_diagnostics(
        self, target: TeamTarget, stage_number: int, member_slot: int,
    ) -> None:
        """Keep references to existing observations until this attempt finishes."""
        self._last_detail_failure_location = None
        previous = getattr(self, "_detail_failure_frames", None)
        if previous is not None and any(
            previous["position"].get(key) not in (None, value)
            for key, value in (
                ("team_id", target.team_id), ("stage_number", stage_number),
                ("member_slot", member_slot),
            )
        ):
            self._detail_failure_frames = None
        self._member_failure_frames = {
            "position": {
                "team_id": target.team_id,
                "opponent_position": target.opponent_position,
                "stage_number": stage_number,
                "member_slot": member_slot,
                "member_name": None,
            },
            "source": None, "frames": deque(maxlen=6),
            "actions": deque(maxlen=16), "started": time.perf_counter(),
            "persisted": False,
        }
        self.set_member_read_phase("member_open")

    def set_member_read_phase(self, phase: str, **position: Any) -> None:
        diagnostic = getattr(self, "_member_failure_frames", None)
        if diagnostic is not None:
            # Coordinates from a completed card must not name a later failure.
            diagnostic["position"].update(
                phase=phase, kind=None, group_index=None, card_slot=None,
                screen_slot=None, card_name=None, p_item_name=None,
            )
            diagnostic["position"].update(position)

    def end_member_read_diagnostics(self) -> None:
        self._member_failure_frames = None

    def note_member_confirmed_card(self, card_id: int) -> None:
        diagnostic = getattr(self, "_member_failure_frames", None)
        if diagnostic is not None:
            diagnostic["position"]["card_name"] = self.catalog.skill_card_title(card_id)

    def persist_member_read_failure(self, error: Exception) -> None:
        """Freeze the original failure before any recovery can leave its page."""
        diagnostic = getattr(self, "_member_failure_frames", None)
        if diagnostic is None or diagnostic["persisted"]:
            return
        diagnostic["persisted"] = True
        detail = getattr(self, "_detail_failure_frames", None)
        selected = detail if detail is not None else diagnostic
        causes = []
        cause: BaseException | None = error
        seen: set[int] = set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            causes.append({
                "type": type(cause).__name__, "code": getattr(cause, "code", None),
                "error": str(cause),
            })
            cause = cause.__cause__ or cause.__context__
        selected["error_chain"] = causes
        if detail is not None:
            # Freeze evidence only. Recovery and the original attempt recorder
            # retain ownership of identity state and transaction accounting.
            detail["persisted"] = True
            self._save_failure_evidence(detail, str(error), event="arena_detail_failure")
        else:
            self._save_failure_evidence(diagnostic, str(error), event="arena_reader_failure")

    def _record_detail_action(self, kind: str, box: Sequence[int], *, succeeded: bool) -> None:
        for name in ("_detail_failure_frames", "_member_failure_frames"):
            diagnostic = getattr(self, name, None)
            if diagnostic is not None and not diagnostic.get("persisted", False):
                diagnostic["actions"].append({
                    "kind": kind, "box": list(box), "succeeded": succeeded,
                    "seconds": round(time.perf_counter() - diagnostic["started"], 6),
                })

    def _persist_detail_failure(self, error: str) -> None:
        """Save at most six already captured frames; never capture for logging."""
        diagnostic = getattr(self, "_detail_failure_frames", None)
        self._detail_failure_frames = None
        if diagnostic is None or diagnostic.get("persisted", False):
            return
        member = getattr(self, "_member_failure_frames", None)
        if member is not None:
            member["persisted"] = True
        self._save_failure_evidence(diagnostic, error, event="arena_detail_failure")

    def _save_failure_evidence(
        self, diagnostic: dict[str, Any], error: str, *, event: str,
    ) -> None:
        """The shared writer never observes or operates the game."""
        self._last_detail_failure_location = (error, dict(diagnostic["position"]))
        save_started = time.perf_counter()
        try:
            import cv2

            folder = DEFAULT_CHALLENGE_RECORD_ROOT / "reader-failures" / uuid4().hex
            folder.mkdir(parents=True)
            frames = []
            if diagnostic["source"] is not None:
                frames.append((diagnostic["started"], diagnostic["source"]))
            confirmed_open = diagnostic.get("confirmed_open")
            if confirmed_open is not None:
                frames.append(confirmed_open)
                recent = [
                    frame for frame in diagnostic["frames"]
                    if frame[1] is not confirmed_open[1]
                ]
                frames.extend(recent[-(6 - len(frames)):])
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
                "event": event, **diagnostic["position"],
                "error": error, "actions": list(diagnostic["actions"]),
                "frames": saved, "folder": str(folder),
            }
            if unsaved:
                evidence["unsaved_frames"] = unsaved
            if event == "arena_reader_failure":
                evidence["unconfirmed_fields"] = [
                    key for key in ("member_name", "group_index", "card_slot", "card_name", "screen_slot", "p_item_name")
                    if diagnostic["position"].get(key) is None
                ]
            if diagnostic.get("error_chain") is not None:
                evidence["error_chain"] = diagnostic["error_chain"]
            if diagnostic.get("p_item_text") is not None:
                evidence["p_item_text"] = diagnostic["p_item_text"]
            (folder / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.warning(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
        except Exception as diagnostic_error:
            logger.warning(json.dumps({
                "event": "arena_failure_evidence_save_failed", **diagnostic["position"],
                "error": error, "diagnostic_error": str(diagnostic_error),
            }, ensure_ascii=False, sort_keys=True))
        finally:
            self._record_duration_sample("detail_failure_evidence_save", time.perf_counter() - save_started)

    def runtime_samples_since(self, cursor: Mapping[str, int]) -> dict[str, Any]:
        """Export failed and successful samples without another observation."""
        return {
            "duration_samples_seconds": {
                name: [round(value, 6) for value in values[cursor.get(name, 0):]]
                for name, values in self._runtime_duration_samples.items()
                if len(values) > cursor.get(name, 0)
            },
            "unfinished_detail_transactions": [list(key) for key in self._card_transaction_started],
        }

    def record_member_read_attempt(
        self, target: TeamTarget, stage_number: int, member_slot: int, *,
        wall_seconds: float, succeeded: bool, error_code: str | None,
        reopened: bool,
        sample_cursor: Mapping[str, int] | None = None,
    ) -> None:
        self._record_duration_sample("member_read_attempt", wall_seconds)
        if not succeeded:
            self._record_duration_sample("member_read_failed_attempt", wall_seconds)
            # Finish only real accepted transactions. Open failures use a
            # separate diagnostic timer and never create an active identity.
            for key in tuple(self._card_transaction_started):
                self._finish_card_transaction(key, failed=True)
        baseline = self._member_metric_baseline
        cursor = sample_cursor if sample_cursor is not None else baseline[2] if baseline is not None else self.runtime_sample_cursor()
        logger.info(json.dumps({
            "event": "arena_member_read_attempt",
            "team_id": target.team_id,
            "opponent_position": target.opponent_position,
            "stage_number": stage_number,
            "member_slot": member_slot,
            "wall_seconds": round(wall_seconds, 6),
            "succeeded": succeeded,
            "error_code": error_code,
            "reopened": reopened,
            **self.runtime_samples_since(cursor),
        }, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _percentile(values: Sequence[float], quantile: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    def _finish_card_transaction(
        self, key: tuple[int, int], *, failed: bool = False, error: str | None = None,
    ) -> None:
        if failed:
            self._retain_failed_skill_detail_identity(key)
        started = self._card_transaction_started.pop(key, None)
        getattr(self, "_detail_identity_proofs", {}).pop(key, None)
        getattr(self, "_card_transaction_tokens", {}).pop(key, None)
        getattr(self, "_card_transaction_contact_counts", {}).pop(key, None)
        getattr(self, "_card_transaction_source_boxes", {}).pop(key, None)
        getattr(self, "_card_transaction_interaction_boxes", {}).pop(key, None)
        getattr(self, "_card_detail_last_contact_released_at", {}).pop(key, None)
        getattr(self, "_card_detail_capture_started_at", {}).pop(key, None)
        ocr_started = getattr(
            self,
            "_card_transaction_ocr_started",
            {},
        ).pop(key, None)
        kind = self._card_transaction_kinds.pop(
            key,
            "necessary_skill_card_detail_transaction",
        )
        if started is None:
            return
        elapsed = time.perf_counter() - started
        self._record_duration_sample("skill_card_detail_transaction", elapsed)
        self._record_duration_sample(kind, elapsed)
        if failed:
            self._record_duration_sample("skill_card_detail_failed_transaction", elapsed)
            self._increment("skill_card_detail_failed_transactions")
            phase = getattr(self, "_detail_failure_frames", None)
            if phase is not None:
                self._record_duration_sample(f"skill_card_detail_{phase['position'].get('phase', 'body')}_failed", elapsed)
            self._persist_detail_failure(
                error or (phase.get("error") if phase is not None else None)
                or "skill_card_transaction_failed"
            )
        else:
            self._detail_failure_frames = None
        if ocr_started is not None:
            backend_started, cache_hits_started = ocr_started
            self._record_duration_sample(
                "skill_card_detail_capture_full_frame_ocr_backend_calls",
                float(
                    self._runtime_counts.get(
                        "detail_capture_broad_ocr_backend_calls",
                        0,
                    )
                    - backend_started
                ),
            )
            self._record_duration_sample(
                "skill_card_detail_capture_full_frame_ocr_cache_hits",
                float(
                    self._runtime_counts.get(
                        "detail_capture_broad_ocr_cache_hits",
                        0,
                    )
                    - cache_hits_started
                ),
            )

    def runtime_metrics(self) -> dict[str, Any]:
        selection = self._badge_worker_selection
        sample_summaries = {
            name: {
                "count": len(values),
                "p50": round(self._percentile(values, 0.50), 6),
                "p95": round(self._percentile(values, 0.95), 6),
                "max": round(max(values), 6),
            }
            for name, values in sorted(self._runtime_duration_samples.items())
            if values
        }
        return {
            "timing_seconds": {
                key: round(value, 6)
                for key, value in sorted(self._runtime_timing_seconds.items())
            },
            "counts": dict(sorted(self._runtime_counts.items())),
            "duration_percentiles_seconds": sample_summaries,
            "badge_workers": None if selection is None else selection.workers,
            "badge_worker_cache_hit": None
            if selection is None
            else selection.cache_hit,
            "badge_worker_benchmarks": (
                [] if selection is None else list(selection.benchmarks)
            ),
            "card_swipe_duration_ms": self._card_swipe_duration_ms,
            "member_long_press_duration_ms": int(
                round(self._member_long_press_seconds * 1000)
            ),
            "point_click_dispatch": "controller_point_context_box",
            "cost_customization_fallbacks": [
                dict(value)
                for value in getattr(
                    self,
                    "_runtime_cost_customization_fallbacks",
                    (),
                )
            ],
            "badge_glyph_exemplar_diagnostics": [
                deepcopy(value)
                for value in getattr(
                    self,
                    "_runtime_badge_glyph_exemplar_diagnostics",
                    (),
                )
            ],
            "badge_glyph_exemplar_comparisons": [
                deepcopy(value)
                for value in getattr(
                    self,
                    "_runtime_badge_glyph_exemplar_comparisons",
                    (),
                )
            ],
            "detail_title_disambiguations": [
                deepcopy(value)
                for value in getattr(
                    self,
                    "_runtime_detail_title_disambiguations",
                    (),
                )
            ],
        }

    @staticmethod
    def _member_read_metrics_event(
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep production pace evidence low-dimensional and image-free."""

        return {
            "event": "arena_member_read_metrics",
            "team_id": target.team_id,
            "stage_number": stage_number,
            "member_slot": member_slot,
            "timing_seconds": evidence["timing_seconds"],
            "counts": evidence["counts"],
            "duration_samples_seconds": evidence["duration_samples_seconds"],
            "detail_title_disambiguations": [
                dict(value)
                for value in evidence.get("detail_title_disambiguations", ())
            ],
            "screenshots_persisted": False,
        }

    def diagnostic_port(self) -> ArenaReaderDiagnosticPort:
        """Return the explicit non-production calibration interface."""

        return ArenaReaderDiagnosticPort(self)

    def member_observation_evidence(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
    ) -> dict[str, Any]:
        """Return only low-dimensional evidence from the active member page."""

        if self._member_metric_baseline is None:
            raise ArenaReaderError(
                "member_observation_generation_missing",
                f"stage-{stage_number}/member-{member_slot} has no active metric generation",
            )
        timing_before, counts_before, sample_lengths = self._member_metric_baseline
        timing_delta = {
            key: round(value - timing_before.get(key, 0.0), 6)
            for key, value in self._runtime_timing_seconds.items()
            if value - timing_before.get(key, 0.0) > 0
        }
        counts_delta = {
            key: value - counts_before.get(key, 0)
            for key, value in self._runtime_counts.items()
            if value - counts_before.get(key, 0) > 0
        }
        duration_samples = {
            key: [round(value, 6) for value in values[sample_lengths.get(key, 0) :]]
            for key, values in self._runtime_duration_samples.items()
            if values[sample_lengths.get(key, 0) :]
        }
        groups = []
        for group_index in (0, 1):
            groups.append(
                {
                    "group_index": group_index,
                    "badge_count_diagnostics": [
                        dict(value)
                        for value in self._badge_count_diagnostics.get(group_index, ())
                    ],
                    "detail_count_overrides": [
                        dict(value)
                        for (detail_group, _), value in sorted(
                            self._detail_count_overrides.items()
                        )
                        if detail_group == group_index
                    ],
                    "detail_semantic_confirmations": [
                        dict(value)
                        for (detail_group, _), value in sorted(
                            self._detail_semantic_confirmations.items()
                        )
                        if detail_group == group_index
                    ],
                    "auxiliary_badge_glyph_counts": [
                        dict(value)
                        for (glyph_group, _), value in sorted(
                            self._badge_glyph_count_diagnostics.items()
                        )
                        if glyph_group == group_index
                    ],
                    "card_face_cost_evidence": [
                        dict(value)
                        for (cost_group, _), value in sorted(
                            self._card_face_cost_diagnostics.items()
                        )
                        if cost_group == group_index
                    ],
                    "zero_card_identity_fusions": [
                        dict(value)
                        for (fusion_group, _), value in sorted(
                            self._zero_card_identity_fusion_diagnostics.items()
                        )
                        if fusion_group == group_index
                    ],
                    "zero_card_detail_candidates": [
                        {
                            "slot": slot,
                            "candidate_ids": list(candidate_ids),
                        }
                        for (detail_group, slot), candidate_ids in sorted(
                            self._zero_card_detail_candidates.items()
                        )
                        if detail_group == group_index
                    ],
                    "excluded_duplicate_flags": list(
                        self._card_excluded_duplicate_flags.get(group_index, ())
                    ),
                    "empty_flags": list(
                        getattr(self, "_card_empty_flags", {}).get(group_index, ())
                    ),
                    "empty_slot_diagnostics": [
                        dict(value)
                        for value in getattr(
                            self,
                            "_card_empty_diagnostics",
                            {},
                        ).get(group_index, ())
                    ],
                    "resolved_card_ids": [
                        self._card_predictions.get((group_index, slot), 0)
                        for slot in range(1, 7)
                    ],
                    "candidate_business_ids": [
                        list(self._card_candidate_groups.get((group_index, slot), ()))
                        for slot in range(1, 7)
                    ],
                }
            )
        evidence = {
            "stage_number": stage_number,
            "member_slot": member_slot,
            "p_items": [dict(value) for value in self._p_item_diagnostics],
            "p_item_generation": dict(self._p_item_generation_evidence),
            "groups": groups,
            "secondary_presence_diagnostics": [
                dict(value) for value in self._secondary_presence_diagnostics
            ],
            "card_content_generation_diagnostics": [
                dict(value) for value in self._card_content_generation_diagnostics
            ],
            "timing_seconds": timing_delta,
            "counts": counts_delta,
            "duration_samples_seconds": duration_samples,
            "badge_candidate_detail_checks": [
                dict(value)
                for value in self._badge_candidate_detail_checks
                if value.get("stage_number") == stage_number
                and value.get("member_slot") == member_slot
            ],
            "cost_customization_fallbacks": [
                dict(value)
                for value in getattr(
                    self,
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
                    self,
                    "_detail_title_disambiguations",
                    (),
                )
            ],
            "screenshots_persisted": False,
        }
        logger.info(
            json.dumps(
                self._member_read_metrics_event(
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

    def _card_references(self) -> BadgeReferenceGallery:
        if self._card_reference_gallery is not None:
            return self._card_reference_gallery
        try:
            self._card_reference_gallery = BadgeReferenceGallery.load(CARD_REFERENCE_ROOT)
        except (OSError, KeyError, TypeError, ValueError, BadgeReferenceError) as error:
            raise ArenaReaderError(
                "skill_card_reference_unavailable",
                "compact clean-card reference gallery is unavailable or invalid",
            ) from error
        return self._card_reference_gallery

    def _card_cost_references(self) -> GenericCostReferenceGallery:
        if self._card_cost_reference_gallery is not None:
            return self._card_cost_reference_gallery
        try:
            self._card_cost_reference_gallery = GenericCostReferenceGallery.load(
                CARD_COST_REFERENCE_ROOT
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            GenericCostReferenceError,
        ) as error:
            raise ArenaReaderError(
                "skill_card_cost_reference_unavailable",
                "compact card-face cost gallery is unavailable or invalid",
            ) from error
        return self._card_cost_reference_gallery

    @staticmethod
    def _parallel_badge_jobs(
        workers: int,
        jobs: Sequence[tuple[int, int]],
        operation: Any,
    ) -> tuple[Any, ...]:
        if workers <= 1:
            return tuple(operation(job) for job in jobs)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return tuple(executor.map(operation, jobs))

    def _local_badge_slot_decision(
        self,
        frames: Sequence[Any],
        frame_rows: Sequence[Sequence[tuple[int, int, int, int]]],
        group_index: int,
        slot_index: int,
    ) -> _BadgeShortlistResult:
        repeated = tuple(
            measure_customization_badge(image, frame_row[slot_index])
            for image, frame_row in zip(frames, frame_rows, strict=True)
        )
        presence = stable_customization_badge_presence(repeated)
        return _BadgeShortlistResult(
            decision=presence,
            observations=repeated,
        )

    def _badge_workers(
        self,
        frames: Sequence[Any],
        jobs: Sequence[tuple[int, int]],
        operation: Any,
    ) -> int:
        if self._badge_worker_selection is not None:
            return self._badge_worker_selection.workers
        signature = self._badge_worker_calibrator.signature(
            frame_shape=frames[0].shape,
            batch_slots=len(jobs),
            reference_version="seeded-plate-shortlist-v1",
            recognition_asset_signature="seeded-plate-detail-authority-v1",
        )

        def run_batch(workers: int) -> None:
            self._parallel_badge_jobs(workers, jobs, operation)

        started = time.perf_counter()
        self._badge_worker_selection = self._badge_worker_calibrator.select(
            signature=signature,
            run_batch=run_batch,
        )
        self._add_timing("badge_worker_selection", time.perf_counter() - started)
        return self._badge_worker_selection.workers

    @staticmethod
    def _badge_detail_inferable(decision: CustomizationBadgeDecision) -> bool:
        """Only the typed shortlist state may authorize a detail."""

        return decision.state is CustomizationBadgeState.DETAIL_CANDIDATE

    def _record_badge_candidate_detail(
        self,
        *,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        phase: str,
        reason: str,
    ) -> dict[str, Any]:
        """Record every authorized shortlist click before touching the UI."""

        entry = {
            "stage_number": int(stage_number),
            "member_slot": int(member_slot),
            "group_index": int(group_index),
            "card_slot": int(card_slot),
            "phase": phase,
            "reason": reason,
            "detail_click_allowed": True,
            "resolved_state": None,
        }
        self._badge_candidate_detail_checks.append(entry)
        return entry

    def _finish_badge_candidate_detail(
        self,
        entry: dict[str, Any],
        result: ClickedSkillCard,
    ) -> None:
        count = sum(int(value) for value in result.customizations.values())
        entry.update(
            {
                "resolved_state": "POSITIVE" if count else "CONFIDENT_ZERO",
                "resolved_card_id": result.card_id,
                "resolved_customization_count": count,
            }
        )
        self._increment(
            "badge_candidate_positive_details"
            if count
            else "badge_candidate_zero_details"
        )

    @classmethod
    def _badge_glyph_observations_are_temporally_stable(
        cls,
        observations: Sequence[dict[str, Any]],
    ) -> bool:
        return bool(
            cls._stable_badge_glyph_exemplar_signature(observations) is not None
            or cls._badge_glyph_frames_are_bounded_raster_settling(observations)
        )

    @classmethod
    def _badge_glyph_observations_need_fresh_group(
        cls,
        observations: Sequence[dict[str, Any]],
    ) -> bool:
        """Retry only three individually valid positive glyphs that disagree.

        Repeating each observation through the strict signature gate reuses its
        complete typed/geometry validation without turning a missing or
        fragmented glyph into a new prerequisite for opening card detail.
        """

        if len(observations) != 3:
            return False
        if not all(
            cls._stable_badge_glyph_exemplar_signature(
                (observation, observation, observation)
            )
            is not None
            for observation in observations
        ):
            return False
        return not cls._badge_glyph_observations_are_temporally_stable(observations)

    def _capture_one_fresh_badge_glyph_group(
        self,
        visible_groups: Sequence[int],
        related_jobs: Sequence[tuple[int, int]],
        *,
        workers: int,
    ) -> tuple[Any, ...] | None:
        """Return one atomic stable fresh window, or preserve the original one.

        The first eligible window is entirely separate from the initially
        accepted card frames.  Every additional frame must still match the
        frozen six-slot geometry and one complete source-generation signature
        for every visible row.  Unstable-but-individually-valid windows slide
        by one frame until the single deadline expires; the deadline is never
        extended and slots are never combined across different windows.
        """

        groups = tuple(visible_groups)
        jobs = tuple(related_jobs)
        if not groups or not jobs:
            return None
        accepted_rows = {
            group_index: tuple(self._card_rows.get(group_index, ()))
            for group_index in groups
        }
        frozen_signatures = {
            group_index: tuple(
                getattr(self, "_card_restoration_signatures", {}).get(
                    group_index,
                    (),
                )
            )
            for group_index in groups
        }
        initial_frame_ids = {
            id(image)
            for group_index in groups
            for image in self._card_count_frames.get(group_index, ())
        }
        if any(
            len(accepted_rows[group_index]) != 6
            or len(frozen_signatures[group_index]) != 3
            or any(len(frame) != 6 for frame in frozen_signatures[group_index])
            for group_index in groups
        ):
            return None

        self._increment("skill_card_badge_glyph_fresh_group_attempts")
        monotonic_started = time.monotonic()
        deadline = monotonic_started + BADGE_GLYPH_FRESH_GROUP_TIMEOUT_SECONDS
        next_capture_not_before = monotonic_started
        capture_times: list[float] = []
        frames: list[Any] = []
        frame_ids: set[int] = set()
        rows_by_group: dict[
            int,
            list[tuple[tuple[int, int, int, int], ...]],
        ] = {group_index: [] for group_index in groups}
        timing_started = time.perf_counter()

        def reject(reason: str) -> None:
            self._increment("skill_card_badge_glyph_fresh_group_rejections")
            self._increment(
                f"skill_card_badge_glyph_fresh_group_{reason}_rejections"
            )
            return None

        try:
            for _ in range(BADGE_GLYPH_FRESH_GROUP_MAX_FRAMES):
                remaining = next_capture_not_before - time.monotonic()
                if remaining > 0:
                    if time.monotonic() + remaining >= deadline:
                        return reject("deadline")
                    self._sleep(remaining)
                if time.monotonic() >= deadline:
                    return reject("deadline")
                capture_started = time.monotonic()
                next_capture_not_before = (
                    capture_started + BADGE_GLYPH_FRESH_GROUP_INTERVAL_SECONDS
                )
                try:
                    image = self._capture()
                except ArenaReaderError:
                    return reject("capture")
                self._increment("skill_card_badge_glyph_fresh_group_reads")
                if time.monotonic() >= deadline:
                    return reject("deadline")
                image_id = id(image)
                if image_id in initial_frame_ids or image_id in frame_ids:
                    return reject("duplicate")
                frame_ids.add(image_id)
                frames.append(image)
                capture_times.append(capture_started)
                try:
                    observed = {
                        group_index: self._validated_card_group_row(
                            image,
                            group_index,
                        )
                        for group_index in groups
                    }
                except ArenaReaderError:
                    return reject("source")
                if any(
                    self._card_rows_shifted(
                        observed[group_index],
                        accepted_rows[group_index],
                    )
                    for group_index in groups
                ):
                    return reject("source")
                if time.monotonic() >= deadline:
                    return reject("deadline")
                try:
                    current_signatures, _ = self._card_content_generation_signatures(
                        image,
                        observed,
                    )
                except (ArenaReaderError, BadgeReferenceError):
                    return reject("source")
                if any(
                    not any(
                        not self._card_content_generation_shifted(
                            current_signatures[group_index],
                            frozen_frame,
                        )
                        for frozen_frame in frozen_signatures[group_index]
                    )
                    for group_index in groups
                ):
                    return reject("source")
                if time.monotonic() >= deadline:
                    return reject("deadline")
                for group_index in groups:
                    rows_by_group[group_index].append(observed[group_index])

                if len(frames) < 3:
                    continue
                required_span = BADGE_GLYPH_FRESH_GROUP_INTERVAL_SECONDS * 2
                if capture_times[-1] - capture_times[-3] + 1e-6 < required_span:
                    return reject("deadline")
                stable_frames = tuple(frames[-3:])
                stable_rows_by_group = {
                    group_index: tuple(rows_by_group[group_index][-3:])
                    for group_index in groups
                }

                def operation(job: tuple[int, int]) -> Any:
                    group_index, slot_index = job
                    return self._local_badge_slot_decision(
                        stable_frames,
                        stable_rows_by_group[group_index],
                        group_index,
                        slot_index,
                    )

                try:
                    fresh_results = self._parallel_badge_jobs(
                        workers,
                        jobs,
                        operation,
                    )
                except (ArenaReaderError, ValueError):
                    return reject("measurement")
                if time.monotonic() >= deadline:
                    return reject("deadline")
                self._increment("skill_card_badge_glyph_fresh_group_windows")
                if any(
                    result.decision.state
                    is not CustomizationBadgeState.DETAIL_CANDIDATE
                    for result in fresh_results
                ):
                    return reject("temporal")
                stable_results = tuple(
                    self._badge_glyph_observations_are_temporally_stable(
                        result.observations
                    )
                    for result in fresh_results
                )
                if all(stable_results):
                    self._increment("skill_card_badge_glyph_fresh_group_acceptances")
                    return fresh_results
                if any(
                    not stable
                    and not self._badge_glyph_observations_need_fresh_group(
                        result.observations
                    )
                    for stable, result in zip(
                        stable_results,
                        fresh_results,
                        strict=True,
                    )
                ):
                    return reject("temporal")
                self._increment(
                    "skill_card_badge_glyph_fresh_group_unstable_windows"
                )
            return reject("deadline")
        finally:
            self._add_timing(
                "badge_glyph_fresh_group_wait",
                time.perf_counter() - timing_started,
            )

    def _batched_badge_local_results(
        self,
        requested_group: int,
    ) -> tuple[Any, ...]:
        cached = self._badge_local_results.get(requested_group)
        if cached is not None:
            return cached
        visible_groups = tuple(
            index
            for index in (0, 1)
            if len(self._card_rows.get(index, ())) == 6
            and len(self._card_count_frames.get(index, ())) == 3
        )
        if requested_group not in visible_groups:
            raise ArenaReaderError(
                "skill_card_badge_batch_missing",
                f"group {requested_group} has no stable three-frame card observation",
            )
        frame_rows = {
            group_index: tuple(
                self._validated_card_group_row(image, group_index)
                for image in self._card_count_frames[group_index]
            )
            for group_index in visible_groups
        }
        duplicate_flags = {
            group_index: self._card_excluded_duplicate_flags.get(
                group_index,
                (False,) * 6,
            )
            for group_index in visible_groups
        }
        empty_flags = {
            group_index: getattr(self, "_card_empty_flags", {}).get(
                group_index,
                (False,) * 6,
            )
            for group_index in visible_groups
        }
        jobs = tuple(
            (group_index, slot_index)
            for group_index in visible_groups
            for slot_index in range(6)
        )

        def operation(job: tuple[int, int]) -> Any:
            group_index, slot_index = job
            if (
                duplicate_flags[group_index][slot_index]
                or empty_flags[group_index][slot_index]
            ):
                return None
            return self._local_badge_slot_decision(
                self._card_count_frames[group_index],
                frame_rows[group_index],
                group_index,
                slot_index,
            )

        workers = self._badge_workers(
            self._card_count_frames[requested_group],
            jobs,
            operation,
        )
        started = time.perf_counter()
        results = self._parallel_badge_jobs(workers, jobs, operation)
        self._add_timing("badge_local_batch", time.perf_counter() - started)
        grouped: dict[int, list[Any]] = {
            group_index: [None] * 6 for group_index in visible_groups
        }
        for (group_index, slot_index), result in zip(jobs, results, strict=True):
            grouped[group_index][slot_index] = result
        related_jobs = tuple(
            (group_index, slot_index)
            for group_index, slot_index in jobs
            if (
                (result := grouped[group_index][slot_index]) is not None
                and result.decision.state
                is CustomizationBadgeState.DETAIL_CANDIDATE
                and self._badge_glyph_observations_need_fresh_group(
                    result.observations
                )
            )
        )
        if related_jobs:
            fresh_results = self._capture_one_fresh_badge_glyph_group(
                visible_groups,
                related_jobs,
                workers=workers,
            )
            if fresh_results is not None:
                for (group_index, slot_index), result in zip(
                    related_jobs,
                    fresh_results,
                    strict=True,
                ):
                    grouped[group_index][slot_index] = result
        for group_index, values in grouped.items():
            self._badge_local_results[group_index] = tuple(values)
            for slot_index, result in enumerate(values, start=1):
                if result is not None:
                    self._badge_glyph_observations[(group_index, slot_index)] = (
                        result.observations
                    )
        return self._badge_local_results[requested_group]

    @staticmethod
    def _badge_glyph_ocr_canvases(
        descriptor: Any,
    ) -> tuple[tuple[int, str, Any], ...]:
        """Render one centre-plate glyph mask for recognition-only OCR.

        The input is the quantized mask cut from the sole centre-seeded green
        component. No card pixels, frame, green artwork, template asset, or
        card identity participates in these canvases.  The recognizer needs a
        much wider quiet zone than ordinary scene OCR: the former fixed
        16-pixel border was only two cells at the 8x glyph scale and caused a
        real ``1`` to be decoded as CJK strokes.  Three fixed quiet zones and
        both polarities form six deterministic recognition views of the same
        bounded glyph; they are robustness transforms, not independent source
        observations.
        """

        import cv2
        import numpy as np

        values = np.asarray(descriptor)
        if (
            values.shape != (216,)
            or values.dtype.kind not in "iu"
            or bool(np.any(values < 0))
            or bool(np.any(values > 15))
        ):
            raise ArenaReaderError(
                "skill_card_badge_glyph_invalid",
                "badge glyph descriptor must contain 216 integers in [0, 15]",
            )
        mask = (values.reshape(18, 12) > 0).astype(np.uint8) * 255
        active = mask > 0
        ys, xs = np.nonzero(active)
        if not len(xs):
            raise ArenaReaderError(
                "skill_card_badge_glyph_invalid",
                "badge glyph descriptor has no foreground",
            )
        crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        canvases: list[tuple[int, str, Any]] = []
        for quiet_cells in (8, 16, 24):
            padded = np.zeros(
                (
                    crop.shape[0] + 2 * quiet_cells,
                    crop.shape[1] + 2 * quiet_cells,
                ),
                dtype=np.uint8,
            )
            padded[
                quiet_cells : quiet_cells + crop.shape[0],
                quiet_cells : quiet_cells + crop.shape[1],
            ] = crop
            target_width = max(
                1,
                int(round(padded.shape[1] * 48 / padded.shape[0])),
            )
            normal = cv2.resize(
                padded,
                (target_width, 48),
                interpolation=cv2.INTER_CUBIC,
            )
            for polarity, canvas in (
                ("normal", normal),
                ("inverted", 255 - normal),
            ):
                canvases.append(
                    (
                        quiet_cells,
                        polarity,
                        cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR),
                    )
                )
        return tuple(canvases)

    @staticmethod
    def _stable_badge_glyph_exemplar_signature(
        observations: Sequence[dict[str, Any]],
    ) -> tuple[
        tuple[int, ...],
        tuple[int, int, int, int, int],
        int,
    ] | None:
        """Return one exact tail-stable glyph plus its normalized geometry.

        This is stricter than the generic OCR fallback.  An exemplar is valid
        only when every frozen source frame contains exactly one component and
        the final two chronological frames have byte-identical descriptors and
        normalized geometry.  The first frame may be a narrowly bounded
        contour-rasterization outlier with the same component box; an unordered
        two-of-three vote is never accepted.  No frames are averaged and no
        nearest-neighbour vote is used.  Position is omitted from the returned
        signature: the member-local lifetime already fixes the window/layout
        generation, while the same rendered digit may occupy another one of
        the twelve card slots.
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
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= 15
                    for value in descriptor
                )
                or not isinstance(normalization_size, (list, tuple))
                or len(normalization_size) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    for value in normalization_size
                )
                or not isinstance(badge_center, (list, tuple))
                or len(badge_center) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in badge_center
                )
                or not isinstance(component_box, (list, tuple))
                or len(component_box) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in component_box
                )
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
        if (
            first_geometry[:4] != tail_geometry[:4]
            or abs(first_geometry[4] - tail_geometry[4]) > 2
        ):
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

    @classmethod
    def _badge_glyph_frames_are_bounded_raster_settling(
        cls,
        observations: Sequence[dict[str, Any]],
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
            or max(component_areas) - min(component_areas)
            > BADGE_GLYPH_RASTER_SETTLING_AREA_DELTA_MAX
        ):
            return False

        foregrounds = tuple(
            tuple(value > 0 for value in descriptor)
            for descriptor in descriptors
        )
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
        if not (
            (contour_adds and area_increases)
            or (contour_removes and area_decreases)
        ):
            return False

        for left_index, right_index in ((0, 1), (1, 2), (0, 2)):
            metrics = cls._badge_glyph_comparison_metrics(
                descriptors[left_index],
                descriptors[right_index],
            )
            if (
                metrics["l1_sum"] > BADGE_GLYPH_RASTER_SETTLING_L1_MAX
                or metrics["q4_changed_cells"]
                > BADGE_GLYPH_RASTER_SETTLING_CHANGED_MAX
                or metrics["binary_xor_cells"]
                > BADGE_GLYPH_RASTER_SETTLING_XOR_MAX
                or metrics["foreground_iou"]
                < BADGE_GLYPH_RASTER_SETTLING_IOU_MIN
            ):
                return False
        return True

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
        signature = self._stable_badge_glyph_exemplar_signature(
            getattr(self, "_badge_glyph_observations", {}).get(key, ())
        )
        if signature is None:
            return False
        descriptor, geometry, support_frames = signature
        exemplars = getattr(self, "_badge_glyph_exemplars", None)
        if exemplars is None:
            exemplars = {}
            self._badge_glyph_exemplars = exemplars
        polluted = getattr(self, "_badge_glyph_polluted_descriptors", None)
        if polluted is None:
            polluted = set()
            self._badge_glyph_polluted_descriptors = polluted
        runtime_diagnostics = getattr(
            self,
            "_runtime_badge_glyph_exemplar_diagnostics",
            None,
        )
        if runtime_diagnostics is None:
            runtime_diagnostics = []
            self._runtime_badge_glyph_exemplar_diagnostics = runtime_diagnostics
        diagnostic = {
            "side": (
                None
                if target is None
                else ("own" if target.is_own_team else "opponent")
            ),
            "team_id": None if target is None else target.team_id,
            "opponent_position": (
                None if target is None else target.opponent_position
            ),
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
            "descriptor_sha256": self._badge_glyph_descriptor_sha256(
                descriptor
            ),
            "resolution_source": resolved.resolution_source,
            "detail_evidence_mode": resolved.detail_evidence_mode,
            "detail_confirmation_reads": resolved.detail_confirmation_reads,
        }
        if descriptor in polluted:
            getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_polluted",
                f"{key!r} exact glyph descriptor was already mapped to multiple counts",
            )
        existing = exemplars.get(descriptor)
        runtime_labels = getattr(self, "_badge_glyph_runtime_labels", None)
        if runtime_labels is None:
            runtime_labels = {}
            self._badge_glyph_runtime_labels = runtime_labels
        if existing is not None and existing.get("count") != count:
            exemplars.pop(descriptor, None)
            polluted.add(descriptor)
            runtime_labels.pop(descriptor, None)
            # A prior ambiguous card may already have consumed this exemplar
            # and cached its count.  Invalidate every member-local glyph
            # decision before aborting so no bounded retry can reuse the
            # now-disproved label.
            getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
            self._increment("skill_card_badge_glyph_exemplar_collisions")
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_collision",
                f"{key!r} authoritatively maps an already consumed exact glyph "
                f"descriptor from count {existing.get('count')!r} to {count}",
            )
        self._validate_authoritative_badge_glyph_runtime_transition(
            key,
            descriptor,
            count=count,
        )
        outlier_descriptor = self._validate_badge_glyph_tail_outlier(
            key,
            count=count,
            support_frames=support_frames,
        )
        if existing is None:
            runtime_labels.pop(descriptor, None)
            runtime_diagnostics.append(diagnostic)
            exemplars[descriptor] = {
                "count": count,
                "geometries": {geometry},
                "prototypes": {(geometry, support_frames)},
            }
            self._increment("skill_card_badge_glyph_exemplars_registered")
            self._record_badge_glyph_runtime_label(
                key,
                outlier_descriptor,
                count=count,
                evidence="bounded first-frame glyph",
            )
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
            getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
            self._increment("skill_card_badge_glyph_exemplar_collisions")
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_invalid",
                f"{key!r} exact glyph exemplar has invalid geometry provenance",
            )
        runtime_labels.pop(descriptor, None)
        runtime_diagnostics.append(diagnostic)
        geometries.add(geometry)
        prototype_provenance.add((geometry, support_frames))
        self._record_badge_glyph_runtime_label(
            key,
            outlier_descriptor,
            count=count,
            evidence="bounded first-frame glyph",
        )
        return True

    @staticmethod
    def _badge_glyph_descriptor_sha256(
        descriptor: Sequence[int],
    ) -> str:
        """Return a stable identity without persisting the glyph grid."""

        import hashlib

        return hashlib.sha256(bytes(descriptor)).hexdigest()

    def _validate_badge_glyph_tail_outlier(
        self,
        key: tuple[int, int],
        *,
        count: int,
        support_frames: int,
    ) -> tuple[int, ...] | None:
        """Reject a bounded first-frame outlier with contrary member truth."""

        if support_frames == 3:
            return None
        observations = getattr(self, "_badge_glyph_observations", {}).get(
            key,
            (),
        )
        first_internal = observations[0]["internal_features"]
        first_descriptor = tuple(
            first_internal["glyph_descriptor_12x18_q4"]
        )
        self._validate_badge_glyph_runtime_label(
            key,
            first_descriptor,
            count=count,
            evidence="bounded first-frame glyph",
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

        for runtime_descriptor, claimed_count in getattr(
            self,
            "_badge_glyph_runtime_labels",
            {},
        ).items():
            if type(claimed_count) is not int or not 1 <= claimed_count <= 9:
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_invalid",
                    f"{key!r} authoritative glyph encountered an invalid "
                    "runtime label",
            )
            if claimed_count == count:
                continue
            exact_conflict = runtime_descriptor == descriptor
            if not exact_conflict:
                metrics = self._badge_glyph_comparison_metrics(
                    descriptor,
                    runtime_descriptor,
                )
                if not self._badge_glyph_calibration_outer_near(metrics):
                    continue
            getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
            self._increment(
                "skill_card_badge_glyph_exemplar_transition_conflicts"
            )
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_transition_conflict",
                (
                    f"{key!r} authoritative glyph maps to count {count}, "
                    f"after a prior runtime label claimed count {claimed_count}"
                    if exact_conflict
                    else f"{key!r} authoritative glyph is not separated from "
                    f"prior runtime count {claimed_count}"
                ),
            )

    def _validate_badge_glyph_runtime_label(
        self,
        key: tuple[int, int],
        descriptor: tuple[int, ...],
        *,
        count: int,
        evidence: str,
        check_exemplars: bool = True,
    ) -> None:
        """Cross-check any runtime glyph label against all member-local truth."""

        polluted = getattr(self, "_badge_glyph_polluted_descriptors", set())
        exemplar = getattr(self, "_badge_glyph_exemplars", {}).get(
            descriptor
        )
        if descriptor in polluted:
            getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
            self._increment(
                "skill_card_badge_glyph_exemplar_transition_conflicts"
            )
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_transition_conflict",
                f"{key!r} {evidence} is already polluted",
            )
        exemplar_count = None if exemplar is None else exemplar.get("count")
        if exemplar is not None and (
            type(exemplar_count) is not int or not 1 <= exemplar_count <= 9
        ):
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_invalid",
                f"{key!r} {evidence} has an invalid exemplar count",
            )
        runtime_count = getattr(
            self,
            "_badge_glyph_runtime_labels",
            {},
        ).get(descriptor)
        conflicting_count = (
            exemplar_count
            if exemplar_count is not None and exemplar_count != count
            else (
                runtime_count
                if runtime_count is not None and runtime_count != count
                else None
            )
        )
        if conflicting_count is not None:
            getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
            self._increment(
                "skill_card_badge_glyph_exemplar_transition_conflicts"
            )
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_transition_conflict",
                f"{key!r} {evidence} maps to count {conflicting_count}, "
                f"not count {count}",
            )
        for polluted_descriptor in polluted:
            if polluted_descriptor == descriptor:
                continue
            metrics = self._badge_glyph_comparison_metrics(
                descriptor,
                polluted_descriptor,
            )
            if self._badge_glyph_calibration_outer_near(metrics):
                getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
                self._increment(
                    "skill_card_badge_glyph_exemplar_transition_conflicts"
                )
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_transition_conflict",
                    f"{key!r} {evidence} is too close to a polluted member glyph",
                )
        if check_exemplars:
            for exemplar_descriptor, candidate in getattr(
                self,
                "_badge_glyph_exemplars",
                {},
            ).items():
                candidate_count = candidate.get("count")
                if (
                    type(candidate_count) is not int
                    or not 1 <= candidate_count <= 9
                ):
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_exemplar_invalid",
                        f"{key!r} {evidence} encountered an invalid exemplar count",
                    )
                if candidate_count == count or exemplar_descriptor == descriptor:
                    continue
                metrics = self._badge_glyph_comparison_metrics(
                    descriptor,
                    exemplar_descriptor,
                )
                if self._badge_glyph_calibration_outer_near(metrics):
                    getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
                    self._increment(
                        "skill_card_badge_glyph_exemplar_transition_conflicts"
                    )
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_exemplar_transition_conflict",
                        f"{key!r} {evidence} is not separated from count "
                        f"{candidate_count} member truth",
                    )
        for runtime_descriptor, candidate_count in getattr(
            self,
            "_badge_glyph_runtime_labels",
            {},
        ).items():
            if (
                type(candidate_count) is not int
                or not 1 <= candidate_count <= 9
            ):
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_invalid",
                    f"{key!r} {evidence} encountered an invalid runtime label",
                )
            if candidate_count == count or runtime_descriptor == descriptor:
                continue
            metrics = self._badge_glyph_comparison_metrics(
                descriptor,
                runtime_descriptor,
            )
            if self._badge_glyph_calibration_outer_near(metrics):
                getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
                self._increment(
                    "skill_card_badge_glyph_exemplar_transition_conflicts"
                )
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_transition_conflict",
                    f"{key!r} {evidence} is not separated from prior runtime "
                    f"count {candidate_count}",
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
        self._validate_badge_glyph_runtime_label(
            key,
            descriptor,
            count=count,
            evidence=evidence,
        )
        labels = getattr(self, "_badge_glyph_runtime_labels", None)
        if labels is None:
            labels = {}
            self._badge_glyph_runtime_labels = labels
        labels[descriptor] = count

    @staticmethod
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
                sum(value * value for value in differences)
                / len(differences),
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

    @staticmethod
    def _badge_glyph_calibration_outer_near(
        metrics: dict[str, int | float],
    ) -> bool:
        """Return whether another label is too close for calibrated transfer."""

        return (
            metrics["l1_sum"] <= BADGE_GLYPH_CALIBRATED_OUTER_L1_MAX
            or metrics["mse"] <= BADGE_GLYPH_CALIBRATED_OUTER_MSE_MAX
            or metrics["q4_changed_cells"]
            <= BADGE_GLYPH_CALIBRATED_OUTER_CHANGED_MAX
            or metrics["binary_xor_cells"]
            <= BADGE_GLYPH_CALIBRATED_OUTER_XOR_MAX
            or metrics["foreground_iou"]
            >= BADGE_GLYPH_CALIBRATED_OUTER_IOU_MIN
        )

    @staticmethod
    def _badge_glyph_calibration_inner_match(
        target_geometry: tuple[int, int, int, int, int],
        sample_prototypes: set[
            tuple[tuple[int, int, int, int, int], int]
        ],
        metrics: dict[str, int | float],
    ) -> bool:
        """Accept one prototype only inside the member-local inner radius."""

        geometry_matches = any(
            target_geometry[:4] == sample_geometry[:4]
            and abs(target_geometry[4] - sample_geometry[4]) <= 2
            and support_frames == 3
            for sample_geometry, support_frames in sample_prototypes
        )
        return (
            geometry_matches
            and metrics["l1_sum"] <= BADGE_GLYPH_CALIBRATED_INNER_L1_MAX
            and metrics["mse"] <= BADGE_GLYPH_CALIBRATED_INNER_MSE_MAX
            and metrics["q4_changed_cells"]
            <= BADGE_GLYPH_CALIBRATED_INNER_CHANGED_MAX
            and metrics["binary_xor_cells"]
            <= BADGE_GLYPH_CALIBRATED_INNER_XOR_MAX
            and metrics["foreground_iou"]
            >= BADGE_GLYPH_CALIBRATED_INNER_IOU_MIN
        )

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
            getattr(self, "_badge_glyph_exemplars", {}).items(),
            key=lambda item: (
                int(item[1].get("count", 0)),
                self._badge_glyph_descriptor_sha256(item[0]),
            ),
        ):
            metrics = self._badge_glyph_comparison_metrics(
                descriptor,
                exemplar_descriptor,
            )
            comparisons.append(
                {
                    "count": exemplar.get("count"),
                    "descriptor_sha256": self._badge_glyph_descriptor_sha256(
                        exemplar_descriptor
                    ),
                    "geometries": [
                        list(value)
                        for value in sorted(exemplar.get("geometries", ()))
                    ],
                    **metrics,
                }
            )
        if not comparisons:
            return
        record = {
            "group_index": key[0],
            "card_slot": key[1],
            "maximum_count": maximum_count,
            "target_descriptor_sha256": self._badge_glyph_descriptor_sha256(
                descriptor
            ),
            "target_geometry": list(geometry),
            "comparisons": comparisons,
        }
        runtime = getattr(
            self,
            "_runtime_badge_glyph_exemplar_comparisons",
            None,
        )
        if runtime is None:
            runtime = []
            self._runtime_badge_glyph_exemplar_comparisons = runtime
        if record not in runtime:
            runtime.append(record)

    @staticmethod
    def _normalized_badge_glyph_domain(
        maximum_count: int,
        admissible_counts: Sequence[int] | None,
    ) -> tuple[int, ...]:
        values = (
            tuple(range(1, maximum_count + 1))
            if admissible_counts is None
            else tuple(admissible_counts)
        )
        if (
            not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum_count
                for value in values
            )
        ):
            raise ArenaReaderError(
                "skill_card_badge_glyph_domain_invalid",
                "badge glyph admissible counts must be positive integers "
                f"within 1..{maximum_count}: {values!r}",
            )
        return tuple(sorted(set(values)))

    def _badge_glyph_member_calibrated_count(
        self,
        key: tuple[int, int],
        descriptor: tuple[int, ...],
        geometry: tuple[int, int, int, int, int],
        *,
        support_frames: int,
        admissible_counts: tuple[int, ...],
    ) -> int | None:
        """Transfer a count only inside a fully covered member-local domain.

        This is a bounded calibration, not a global nearest-neighbour model.
        The target must be strictly stable in all three source frames, every
        detail-compatible label must already have independent member truth,
        one three-frame prototype must enter the inner radius, and every other
        label must remain outside the wider rejection guard.
        """

        if support_frames != 3:
            return None
        exemplars = getattr(self, "_badge_glyph_exemplars", {})
        if not exemplars:
            return None
        polluted = getattr(self, "_badge_glyph_polluted_descriptors", set())
        for polluted_descriptor in polluted:
            metrics = self._badge_glyph_comparison_metrics(
                descriptor,
                polluted_descriptor,
            )
            if self._badge_glyph_calibration_outer_near(metrics):
                getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
                self._increment(
                    "skill_card_badge_glyph_calibration_conflicts"
                )
                raise ArenaReaderError(
                    "skill_card_badge_glyph_calibration_ambiguous",
                    f"{key!r} calibrated glyph is too close to polluted member evidence",
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
                or not all(
                    isinstance(value, tuple)
                    and len(value) == 5
                    and all(type(item) is int for item in value)
                    for value in geometries
                )
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
            metrics = self._badge_glyph_comparison_metrics(
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
                        and self._badge_glyph_calibration_inner_match(
                            geometry,
                            sample_prototypes,
                            metrics,
                        )
                    ),
                }
            )
        if covered_counts != set(admissible_counts):
            return None

        inner_counts = {
            int(candidate["count"])
            for candidate in candidates
            if candidate["inner"]
        }
        if not inner_counts:
            return None
        if len(inner_counts) != 1:
            getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
            self._increment("skill_card_badge_glyph_calibration_conflicts")
            raise ArenaReaderError(
                "skill_card_badge_glyph_calibration_ambiguous",
                f"{key!r} enters member-local inner radii for multiple counts "
                f"{sorted(inner_counts)!r}",
            )
        count = next(iter(inner_counts))
        competing = tuple(
            candidate
            for candidate in candidates
            if candidate["count"] != count
            and self._badge_glyph_calibration_outer_near(
                candidate["metrics"]
            )
        )
        if competing:
            getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
            self._increment("skill_card_badge_glyph_calibration_conflicts")
            raise ArenaReaderError(
                "skill_card_badge_glyph_calibration_ambiguous",
                f"{key!r} has a competing member label inside the outer guard",
            )
        self._record_badge_glyph_runtime_label(
            key,
            descriptor,
            count=count,
            evidence="member-local calibrated glyph",
        )
        return count

    def _badge_glyph_exemplar_count(
        self,
        key: tuple[int, int],
        *,
        maximum_count: int,
        admissible_counts: Sequence[int] | None = None,
    ) -> int | None:
        """Resolve exact or safely calibrated member-local glyph evidence."""

        signature = self._stable_badge_glyph_exemplar_signature(
            getattr(self, "_badge_glyph_observations", {}).get(key, ())
        )
        if signature is None:
            return None
        descriptor, geometry, support_frames = signature
        domain = self._normalized_badge_glyph_domain(
            maximum_count,
            admissible_counts,
        )
        polluted = getattr(self, "_badge_glyph_polluted_descriptors", set())
        if descriptor in polluted:
            raise ArenaReaderError(
                "skill_card_badge_glyph_exemplar_polluted",
                f"{key!r} exact glyph descriptor was authoritatively mapped to multiple counts",
            )
        self._record_badge_glyph_exemplar_comparisons(
            key,
            descriptor,
            geometry,
            maximum_count=maximum_count,
        )
        exemplar = getattr(self, "_badge_glyph_exemplars", {}).get(descriptor)
        if exemplar is None or geometry not in exemplar.get("geometries", set()):
            return self._badge_glyph_member_calibrated_count(
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
                f"{key!r} exact glyph exemplar count {count} is outside "
                f"the detail-compatible domain {domain!r}",
            )
        outlier_descriptor = self._validate_badge_glyph_tail_outlier(
            key,
            count=count,
            support_frames=support_frames,
        )
        self._record_badge_glyph_runtime_label(
            key,
            outlier_descriptor,
            count=count,
            evidence="bounded first-frame glyph",
        )
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

        if (
            isinstance(maximum_count, bool)
            or not isinstance(maximum_count, int)
            or not 1 <= maximum_count <= 9
        ):
            raise ArenaReaderError(
                "skill_card_badge_glyph_domain_invalid",
                f"{key!r} has an unsupported single-glyph badge-count maximum: "
                f"{maximum_count!r}",
            )
        domain = self._normalized_badge_glyph_domain(
            maximum_count,
            admissible_counts,
        )
        cached = self._badge_glyph_count_diagnostics.get(key)
        if cached is not None and (
            cached.get("maximum_count") != maximum_count
            or tuple(cached.get("admissible_counts", ())) != domain
        ):
            # Detail identity is authoritative over a pre-click visual family.
            # Re-evaluate the frozen plate glyph when the confirmed card changes
            # the only legal OCR domain instead of keeping a stale rejection.
            if allow_ocr:
                self._badge_glyph_count_diagnostics.pop(key, None)
            cached = None
        if cached is not None:
            count = cached.get("resolved_count")
            if type(count) is int and (count == 0 or count in domain):
                return count
            raise ArenaReaderError(
                "skill_card_badge_glyph_ambiguous",
                "cached badge glyph evidence is not resolvable for "
                f"{key!r}; diagnostic={cached!r}",
            )

        observations = self._badge_glyph_observations.get(key, ())
        if len(observations) != 3:
            raise ArenaReaderError(
                "skill_card_badge_glyph_frames_missing",
                f"{key!r} has no stable three-frame badge glyph evidence",
            )
        started = time.perf_counter()
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
                    and all(
                        not isinstance(value, bool)
                        and isinstance(value, int)
                        and 0 <= value <= 15
                        for value in descriptor
                    )
                ):
                    positive_descriptors.append(tuple(descriptor))
            stable_signature = self._stable_badge_glyph_exemplar_signature(
                observations
            )
            bounded_raster_settling = (
                stable_signature is None
                and self._badge_glyph_frames_are_bounded_raster_settling(
                    observations
                )
            )
            if bounded_raster_settling:
                self._increment("skill_card_badge_glyph_bounded_raster_settling")
            exemplar_count = self._badge_glyph_exemplar_count(
                key,
                maximum_count=maximum_count,
                admissible_counts=domain,
            )
            if exemplar_count is not None:
                signature = self._stable_badge_glyph_exemplar_signature(
                    observations
                )
                if signature is None:
                    raise ArenaReaderError(
                        "skill_card_badge_glyph_exemplar_invalid",
                        f"{key!r} resolved without a stable exemplar signature",
                    )
                descriptor, _geometry, support_frames = signature
                exact_exemplar = getattr(
                    self,
                    "_badge_glyph_exemplars",
                    {},
                ).get(descriptor)
                exact_resolution = (
                    exact_exemplar is not None
                    and _geometry
                    in exact_exemplar.get("geometries", set())
                )
                descriptor_digests = [
                    hashlib.sha256(bytes(value)).hexdigest()
                    for value in positive_descriptors
                ]
                frame_digits = [
                    exemplar_count if value == descriptor else None
                    for value in positive_descriptors
                ]
                variant_texts = [[], [], []]
                self._badge_glyph_count_diagnostics[key] = {
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
                self._increment(
                    "skill_card_badge_glyph_exemplar_resolutions"
                    if exact_resolution
                    else "skill_card_badge_glyph_calibrated_resolutions"
                )
                return exemplar_count

            if positive_descriptors and (
                len(positive_descriptors) != 3
                or (
                    stable_signature is None
                    and not bounded_raster_settling
                )
            ):
                raise ArenaReaderError(
                    "skill_card_badge_glyph_frame_disagreement",
                    f"{key!r} badge glyph frames do not form one stable bounded signature",
                )
            strict_descriptor = (
                positive_descriptors[0]
                if len(positive_descriptors) == 3
                and len(set(positive_descriptors)) == 1
                else None
            )
            polluted_descriptors = getattr(
                self,
                "_badge_glyph_polluted_descriptors",
                set(),
            )
            if any(
                descriptor in polluted_descriptors
                for descriptor in positive_descriptors
            ):
                getattr(self, "_badge_glyph_count_diagnostics", {}).clear()
                self._increment(
                    "skill_card_badge_glyph_exemplar_transition_conflicts"
                )
                raise ArenaReaderError(
                    "skill_card_badge_glyph_exemplar_transition_conflict",
                    f"{key!r} multi-view OCR glyph is already polluted",
                )

            if not allow_ocr:
                raise ArenaReaderError(
                    "skill_card_badge_glyph_cached_evidence_missing",
                    "error-only text recovery has no existing badge evidence; "
                    "additional glyph OCR is outside its budget",
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
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or not 0 <= value <= 15
                        for value in descriptor
                    )
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
                descriptor_digests.append(
                    hashlib.sha256(bytes(descriptor_tuple)).hexdigest()
                )
                cached_vote = descriptor_votes.get(descriptor_tuple)
                if cached_vote is None:
                    texts: list[str] = []
                    votes: Counter[int] = Counter()
                    view_diagnostics: list[dict[str, Any]] = []
                    pair_votes: dict[int, dict[str, int | None]] = {}
                    observed_digits: set[int] = set()
                    for quiet_cells, polarity, canvas in self._badge_glyph_ocr_canvases(
                        descriptor_tuple
                    ):
                        # ``canvas`` is already one tightly bounded glyph, not
                        # a scene that needs text detection.  Recognition must
                        # cover the entire static glyph domain before the
                        # clicked card's legal domain is consulted; otherwise
                        # a stronger out-of-domain digit could be hidden by the
                        # filter and a weaker in-domain vote accepted.
                        observed = tuple(
                            _text(item).strip()
                            for item in self._ocr(
                                canvas,
                                r"^[1-9]$",
                                only_rec=True,
                            )
                        )
                        texts.extend(observed)
                        legal = {
                            int(value)
                            for value in observed
                            if re.fullmatch(r"[1-9]", value) is not None
                        }
                        observed_digits.update(legal)
                        view_digit = next(iter(legal)) if len(legal) == 1 else None
                        if view_digit is not None:
                            votes[view_digit] += 1
                        pair_votes.setdefault(quiet_cells, {})[polarity] = (
                            view_digit
                        )
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
                    sole_digit = (
                        next(iter(observed_digits))
                        if len(observed_digits) == 1
                        else None
                    )
                    paired_quiet_zones = [
                        quiet_cells
                        for quiet_cells, polarities in sorted(pair_votes.items())
                        if sole_digit is not None
                        and polarities.get("normal") == sole_digit
                        and polarities.get("inverted") == sole_digit
                    ]
                    voting_quiet_zones = {
                        quiet_cells
                        for quiet_cells, polarities in pair_votes.items()
                        if sole_digit is not None
                        and sole_digit in polarities.values()
                    }
                    digit = (
                        sole_digit
                        if sole_digit is not None
                        and len(paired_quiet_zones) >= 2
                        else None
                    )
                    resolution_mode = (
                        "conflict_free_pairs"
                        if digit is not None
                        else "unresolved"
                    )
                    if (
                        digit is None
                        and sole_digit == 1
                        and stable_signature is not None
                        and stable_signature[2] == 3
                        and sole_digit
                        in BADGE_GLYPH_OCR_VALIDATED_ACCEPTANCE_COUNTS
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
                            if pair_votes.get(quiet_cells, {}).get("normal")
                            is not None
                            and pair_votes.get(quiet_cells, {}).get("normal")
                            == pair_votes.get(quiet_cells, {}).get("inverted")
                        ]
                        narrow_normal = pair_votes.get(8, {}).get("normal")
                        narrow_inverted = pair_votes.get(8, {}).get("inverted")
                        wide_winner = (
                            wide_pair_digits[0]
                            if len(wide_pair_digits) == 2
                            and len(set(wide_pair_digits)) == 1
                            else None
                        )
                        narrow_digits = {
                            value
                            for value in (narrow_normal, narrow_inverted)
                            if value is not None
                        }
                        conflicts = narrow_digits - {wide_winner}
                        if (
                            wide_winner
                            in BADGE_GLYPH_OCR_VALIDATED_ACCEPTANCE_COUNTS
                            and len(narrow_digits) == 2
                            and wide_winner in narrow_digits
                            and len(conflicts) == 1
                            and observed_digits == narrow_digits
                            and votes[wide_winner] == 5
                        ):
                            conflict = next(iter(conflicts))
                            if (
                                votes[conflict] == 1
                                and conflict not in domain
                                and conflict
                                not in BADGE_GLYPH_OCR_VALIDATED_ACCEPTANCE_COUNTS
                            ):
                                digit = wide_winner
                                paired_quiet_zones = [16, 24]
                                resolution_mode = (
                                    "isolated_out_of_domain_narrow_view_conflict"
                                )
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
                frame_view_diagnostics.append(
                    [dict(value) for value in view_diagnostics]
                )
                frame_paired_quiet_zones.append(list(paired_quiet_zones))
                frame_winners.append(digit)
                frame_resolution_modes.append(resolution_mode)
                frame_isolated_out_of_domain_conflicts.append(
                    isolated_out_of_domain_conflict
                )
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
                        f"{key!r} multi-view OCR digit {digit} is outside "
                        f"the detail-compatible domain {domain!r}",
                    )
                frame_digits.append(digit)
            if len(set(frame_digits)) != 1:
                raise ArenaReaderError(
                    "skill_card_badge_glyph_frame_disagreement",
                    f"{key!r} badge glyph OCR disagreed across source frames: {frame_digits!r}",
                )
            count = frame_digits[0]
            if (
                count > 0
                and count not in BADGE_GLYPH_OCR_VALIDATED_ACCEPTANCE_COUNTS
            ):
                # An unvalidated digit can never be accepted or recorded,
                # but it must still be checked against already consumed
                # member truth so a direct contradiction is not hidden
                # behind the broader coverage stop.  Do this only after all
                # three frame winners agree, preserving the stronger temporal
                # disagreement diagnosis.
                for descriptor in dict.fromkeys(positive_descriptors):
                    self._validate_badge_glyph_runtime_label(
                        key,
                        descriptor,
                        count=count,
                        evidence="multi-view OCR glyph",
                    )
                raise ArenaReaderError(
                    "skill_card_badge_glyph_unvalidated_count",
                    f"{key!r} multi-view OCR digit {count} has no independent "
                    "positive holdout coverage for this extractor",
                )
            if count > 0:
                if stable_signature is not None:
                    descriptor, _geometry, support_frames = stable_signature
                    outlier_descriptor = self._validate_badge_glyph_tail_outlier(
                        key,
                        count=count,
                        support_frames=support_frames,
                    )
                    self._record_badge_glyph_runtime_label(
                        key,
                        descriptor,
                        count=count,
                        evidence="multi-view OCR glyph",
                    )
                    self._record_badge_glyph_runtime_label(
                        key,
                        outlier_descriptor,
                        count=count,
                        evidence="multi-view OCR bounded first-frame glyph",
                    )
                elif strict_descriptor is not None:
                    self._record_badge_glyph_runtime_label(
                        key,
                        strict_descriptor,
                        count=count,
                        evidence="strict three-frame multi-view OCR glyph",
                    )
            if any(
                mode == "isolated_out_of_domain_narrow_view_conflict"
                for mode in frame_resolution_modes
            ):
                self._increment(
                    "skill_card_badge_glyph_isolated_out_of_domain_resolutions"
                )
            if any(
                mode == "stable_exact_one_pair_consensus"
                for mode in frame_resolution_modes
            ):
                self._increment(
                    "skill_card_badge_glyph_stable_exact_one_pair_consensus_resolutions"
                )
            self._badge_glyph_count_diagnostics[key] = {
                "group_index": key[0],
                "slot": key[1],
                "maximum_count": maximum_count,
                "admissible_counts": list(domain),
                "resolved_count": count,
                "frame_digits": list(frame_digits),
                "descriptor_sha256": descriptor_digests,
                "ocr_texts": variant_texts,
                "ocr_vote_domain": list(range(1, 10)),
                "ocr_validated_acceptance_domain": sorted(
                    BADGE_GLYPH_OCR_VALIDATED_ACCEPTANCE_COUNTS
                ),
                "ocr_preprocess": (
                    "q4-positive-binary/pad-8-16-24/cubic/h48/"
                    "polarity-2/v3"
                ),
                "frame_vote_counts": frame_vote_counts,
                "frame_views": frame_view_diagnostics,
                "frame_paired_quiet_zones": frame_paired_quiet_zones,
                "frame_winners": frame_winners,
                "frame_resolution_modes": frame_resolution_modes,
                "frame_isolated_out_of_domain_conflicts": (
                    frame_isolated_out_of_domain_conflicts
                ),
                "temporal_signature_mode": (
                    "bounded_monotone_raster_settling"
                    if bounded_raster_settling
                    else "stable_exemplar_signature"
                    if stable_signature is not None
                    else "none"
                ),
                "mode": (
                    "detail_ambiguity_same_plate_non_badge_art_zero"
                    if count == 0
                    else "detail_ambiguity_auxiliary_multiview_glyph_ocr"
                ),
            }
            self._increment("skill_card_badge_auxiliary_glyph_resolutions")
            if count > 0:
                self._increment("skill_card_badge_auxiliary_glyph_ocr")
            return count
        except ArenaReaderError as error:
            if not allow_ocr:
                # A budget-limited cache probe is not a failed recognition.
                # Leave the original retry's glyph evidence untouched.
                raise
            self._badge_glyph_count_diagnostics[key] = {
                "group_index": key[0],
                "slot": key[1],
                "maximum_count": maximum_count,
                "admissible_counts": list(domain),
                "resolved_count": None,
                "frame_digits": list(frame_digits),
                "descriptor_sha256": descriptor_digests,
                "ocr_texts": variant_texts,
                "ocr_vote_domain": list(range(1, 10)),
                "ocr_validated_acceptance_domain": sorted(
                    BADGE_GLYPH_OCR_VALIDATED_ACCEPTANCE_COUNTS
                ),
                "ocr_preprocess": (
                    "q4-positive-binary/pad-8-16-24/cubic/h48/"
                    "polarity-2/v3"
                ),
                "frame_vote_counts": frame_vote_counts,
                "frame_views": frame_view_diagnostics,
                "frame_paired_quiet_zones": frame_paired_quiet_zones,
                "frame_winners": frame_winners,
                "frame_resolution_modes": frame_resolution_modes,
                "frame_isolated_out_of_domain_conflicts": (
                    frame_isolated_out_of_domain_conflicts
                ),
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
            self._add_timing(
                "skill_card_badge_auxiliary_glyph_ocr",
                time.perf_counter() - started,
            )

    def _card_recognizer(self, *, customized: bool) -> tuple[EmbeddingCardRecognizer, float, float]:
        """Load a visual-family candidate source without accepting its business ID.

        Neither embedding becomes a business ID directly.  Positive cards are
        resolved by their mandatory detail page; zero cards require independent
        agreement with the fixed clean-reference gallery.  The intentionally
        null production thresholds therefore remain a ban on accepting a
        prediction ID while a permissive score supplies bounded candidates.
        """

        with self._card_recognizer_lock:
            cached = self._card_recognizers.get(customized)
            if cached is not None:
                return cached
            root = ARENA_CARD_MODEL_ROOT if customized else CARD_MODEL_ROOT
            try:
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
                recognizer = EmbeddingCardRecognizer.load(root)
                runtime = manifest["runtime"]
                if "acceptance_threshold" not in runtime or "minimum_margin" not in runtime:
                    raise KeyError("runtime candidate-source metadata")
                acceptance_threshold = 0.000001
                minimum_margin = 0.0
            except (OSError, KeyError, TypeError, ValueError, ImportError) as error:
                model_label = "arena custom-card" if customized else "base card"
                raise ArenaReaderError(
                    "skill_card_runtime_not_approved",
                    f"{model_label} embedding candidate assets are unavailable",
                ) from error
            loaded = (recognizer, acceptance_threshold, minimum_margin)
            self._card_recognizers[customized] = loaded
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

        plan = self.season.stages[stage_number - 1].plan
        candidate_box = CandidateBox(
            slot=slot_index,
            box=box,
            detector_label="cards",
            detector_score=1.0,
        )
        candidates: list[int] = []
        for customized in (True, False):
            recognizer, acceptance_threshold, minimum_margin = self._card_recognizer(
                customized=customized
            )
            prediction = recognizer.classify(
                image,
                candidate_box,
                frame_id=frame_identifier(image, slot_index + (100 if customized else 200)),
                plan=plan,
                acceptance_threshold=acceptance_threshold,
                min_margin=minimum_margin,
                resolve_upgrade_state=False,
            )
            if not prediction.accepted:
                continue
            if prediction.card_id.isdigit():
                candidates.append(int(prediction.card_id))
            candidates.extend(
                int(value)
                for visual_family in prediction.top_k_card_ids
                for value in visual_family
            )
        ordered = tuple(dict.fromkeys(candidates))
        if not ordered:
            raise ArenaReaderError(
                "skill_card_badge_detail_identity_unknown",
                "neither card embedding supplied a visual family for the "
                "seeded-plate detail candidate",
            )
        return ordered

    def ensure_arena_main(self, *, require_opponents: bool = True) -> None:
        state = ArenaPageState.AMBIGUOUS
        counts = (0, 0, 0, 0, 0)
        grade_observations: tuple[Any, ...] = ()
        grade = self._grade
        grade_error: ArenaReaderError | None = None
        unavailable_reads = 0
        ambiguous_reads = 0
        for attempt in range(4):
            image = self._capture()
            # A fresh own-team read resolves Grade once.  Cached/opponent reads
            # inject that formal decision and must not repeat Grade OCR on each
            # arena-main guard.  The page-state check still runs on every
            # capture.  When Grade is unknown, keep the broad OCR first so the
            # shared OCR entry cannot be polluted by narrower page filters.
            if grade is None:
                grade_observations = tuple(self._ocr(image, r".+"))
            state, counts = self._arena_page_state(image)
            if arena_page_allows_team_entry(
                state,
                require_opponents=require_opponents,
            ):
                if grade is None:
                    try:
                        grade = self._read_grade(
                            image,
                            observations=grade_observations,
                        )
                    except ArenaReaderError as error:
                        if error.code not in {
                            "arena_grade_label_ambiguous",
                            "arena_grade_ambiguous",
                        }:
                            raise
                        grade_error = error
                        if attempt < 3:
                            self._sleep(0.25)
                            continue
                        raise
                    logger.info(
                        json.dumps(
                            {
                                "event": "arena_grade_recognized",
                                "grade": grade,
                                "source": "live_screen",
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                break
            if state is ArenaPageState.AMBIGUOUS:
                ambiguous_reads += 1
                if ambiguous_reads < 2:
                    self._sleep(0.5)
                    continue
                break
            if state is ArenaPageState.OPPONENTS_UNAVAILABLE:
                unavailable_reads += 1
                if unavailable_reads < 3:
                    self._sleep(0.25)
                    continue
            break
        (
            rehearsal_count,
            opponent_count,
            stage_label_count,
            stage_total_count,
            exhausted_notice_count,
        ) = counts
        if (
            state is ArenaPageState.OPPONENTS_UNAVAILABLE
            and require_opponents
            and unavailable_reads == 3
        ):
            raise ArenaReaderError(
                "arena_opponents_unavailable",
                "opponent cards are absent on three stable arena-main reads; today's five contest attempts are exhausted",
            )
        if state is ArenaPageState.TEAM_PREVIEW:
            raise ArenaReaderError(
                "arena_team_preview_visible",
                "rehearsal/team preview is still open and must be left before reading opponents",
            )
        if not arena_page_allows_team_entry(
            state,
            require_opponents=require_opponents,
        ):
            raise ArenaReaderError(
                "arena_main_ambiguous",
                "arena page anchors are inconsistent: "
                f"rehearsal={rehearsal_count}, opponents={opponent_count}, "
                f"stage_labels={stage_label_count}, stage_totals={stage_total_count}, "
                f"exhausted_notices={exhausted_notice_count}",
            )
        if grade is None:
            if grade_error is not None:
                raise grade_error
            raise ArenaReaderError(
                "arena_grade_ambiguous",
                "arena main became available without a same-frame Grade decision",
            )
        if self._grade is not None and self._grade != grade:
            raise ArenaReaderError(
                "arena_grade_changed",
                f"contest Grade changed during one read: {self._grade} -> {grade}",
            )
        self._grade = grade

    def enter_team(self, target: TeamTarget) -> None:
        self._badge_candidate_detail_checks.clear()
        image = self._capture()
        if target.is_own_team:
            matches = self._ocr(image, r"^リハーサル$")
            if len(matches) != 1:
                raise ArenaReaderError("rehearsal_anchor_ambiguous", f"found {len(matches)} rehearsal anchors")
            self._click(self._box_center_point(_box(matches[0])))
        else:
            opponents = self._recognize("ArenaReaderOpponentCards", image)
            if target.opponent_position is None or len(opponents) != 3:
                raise ArenaReaderError("opponent_count_mismatch", "three ordered opponent cards are required")
            ordered = sorted(opponents, key=lambda item: (_box(item)[1], _box(item)[0]))
            self._click(_box(ordered[target.opponent_position]))
        self._wait_for_stage_member_list((3,) if target.is_own_team else (6,))

    def select_opponent_for_challenge(self, position: int) -> dict[str, Any]:
        """Perform one lightweight same-session guard, then open the target.

        The complete lineup has already been read and simulated.  This guard
        intentionally does not repeat that expensive work: it only proves the
        arena main page and three ordered cards still exist, then confirms the
        selected preview exposes the existing challenge-start control.
        """

        if self._challenge_selection_committed:
            raise ArenaReaderError(
                "arena_challenge_already_committed",
                "this reader session already sent an opponent-selection click",
            )
        if position not in (0, 1, 2):
            raise ArenaReaderError(
                "arena_challenge_position_invalid",
                f"selected opponent position is outside 0..2: {position}",
            )
        self.ensure_arena_main(require_opponents=True)
        image = self._capture()
        state, counts = self._arena_page_state(image)
        if state is not ArenaPageState.READY or counts[1] != 3:
            raise ArenaReaderError(
                "arena_challenge_preclick_guard_failed",
                f"arena main is not stable immediately before selection: state={state}, counts={counts}",
            )
        opponents = self._recognize("ArenaReaderOpponentCards", image)
        if len(opponents) != 3:
            raise ArenaReaderError(
                "arena_challenge_preclick_guard_failed",
                f"expected three opponent cards immediately before selection, got {len(opponents)}",
            )
        ordered = sorted(opponents, key=lambda item: (_box(item)[1], _box(item)[0]))
        target_box = _box(ordered[position])
        self._click(target_box)
        self._wait_for_stage_member_list((6,))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            preview = self._capture()
            starts = self._recognize("ChallengeStart", preview)
            if len(starts) == 1:
                self._challenge_selection_committed = True
                self._increment("arena_challenge_selection_clicks")
                return {
                    "position": position,
                    "opponent_count": 3,
                    "target_box": list(target_box),
                    "challenge_start_count": 1,
                    "full_lineup_reread": False,
                }
            if len(starts) > 1:
                break
            self._sleep(0.2)
        raise ArenaReaderError(
            "arena_challenge_start_ambiguous",
            "selected opponent preview did not expose one challenge-start control",
        )

    def select_stage(self, target: TeamTarget, stage_number: int) -> None:
        if stage_number not in (1, 2, 3):
            raise ArenaReaderError("stage_number_invalid", f"stage number is outside 1..3: {stage_number}")
        self._team_stage_total_anchors(target, self._capture())

    def member_slots(self, target: TeamTarget, stage_number: int) -> Sequence[int]:
        image = self._capture()
        totals = self._team_stage_total_anchors(target, image)
        anchor = totals[stage_number - 1]
        height, width = image.shape[:2]
        if self._grade is None:
            raise ArenaReaderError("arena_grade_missing", "contest Grade was not read from the arena main screen")
        slot_cap = stage_member_cap(self._grade, stage_number)
        avatar_height = int(height * 0.0906)
        avatar_top = anchor[1] + 35
        occupied: dict[int, tuple[int, int, int, int]] = {}
        observations: dict[int, tuple[MemberSlotState, MemberSlotMetrics, tuple[int, int, int, int]]] = {}
        for slot, box in enumerate(
            fixed_member_slot_boxes(
                width,
                avatar_top,
                avatar_height,
                3,
                group_center_ratio=(0.6430555556 if target.is_own_team else 0.7361111111),
            ),
            start=1,
        ):
            x0, y0, avatar_width, slot_height = box
            crop = image[y0 : y0 + slot_height, x0 : x0 + avatar_width]
            if crop.shape[:2] != (slot_height, avatar_width):
                raise ArenaReaderError("member_slot_roi_invalid", f"member slot exceeds frame: {box}")
            state, metrics = classify_member_slot(crop)
            observations[slot] = (state, metrics, box)
        for slot, (state, metrics, box) in observations.items():
            if slot > slot_cap:
                if state is MemberSlotState.OCCUPIED:
                    raise ArenaReaderError(
                        "member_slot_outside_grade_cap_occupied",
                        f"{target.team_id}/stage-{stage_number}/slot-{slot} is occupied "
                        f"outside Grade {self._grade} cap {slot_cap}: {metrics}",
                    )
                if state is MemberSlotState.AMBIGUOUS:
                    raise ArenaReaderError(
                        "member_slot_outside_grade_cap_ambiguous",
                        f"{target.team_id}/stage-{stage_number}/slot-{slot} is not proven empty "
                        f"outside Grade {self._grade} cap {slot_cap}: {metrics}",
                    )
                continue
            if state is MemberSlotState.OCCUPIED:
                occupied[slot] = box
            elif state is MemberSlotState.AMBIGUOUS:
                raise ArenaReaderError(
                    "member_slot_ambiguous",
                    f"{target.team_id}/stage-{stage_number}/slot-{slot} is neither an explicit black blank "
                    f"nor a confirmed portrait: {metrics}",
                )
        if not occupied:
            raise ArenaReaderError(
                "member_count_mismatch",
                f"{target.team_id}/stage-{stage_number} has no occupied member within Grade {self._grade} cap {slot_cap}",
            )
        key = (target.team_id, stage_number)
        self._member_boxes[key] = occupied
        self._member_slot_observations[key] = observations
        return tuple(occupied)

    def open_member(self, target: TeamTarget, stage_number: int, slot: int) -> None:
        self._last_detail_failure_location = None
        boxes = self._member_boxes.get((target.team_id, stage_number), {})
        box = boxes.get(slot)
        if box is None:
            raise ArenaReaderError("member_slot_missing", f"member slot {slot} has no recognized card")
        self._reset_member_card_state()
        self._member_metric_baseline = (
            dict(self._runtime_timing_seconds),
            dict(self._runtime_counts),
            {
                key: len(values)
                for key, values in self._runtime_duration_samples.items()
            },
        )
        self._long_press(box)
        try:
            self._assert_member_detail()
        except ArenaReaderError:
            # A long-press may occasionally be dropped.  Retry it once only
            # while the complete team-preview anchors prove that the first
            # action did not navigate anywhere; never repeat on an ambiguous
            # or partially transitioned screen.
            image = self._capture()
            self._team_stage_total_anchors(target, image)
            if len(self._ocr(image, r"^ステージ\s*[123]$")) < 3:
                raise
            self._sleep(0.5)
            self._long_press(box)
            self._assert_member_detail()

    def read_support_bonus(self, target: TeamTarget) -> float:
        image = self._capture()
        height, width = image.shape[:2]
        info_box = (
            int(width * 0.955),
            int(height * 0.022),
            1,
            1,
        )
        self._click(info_box, settle_seconds=0)
        started = time.perf_counter()
        deadline = time.monotonic() + 2.0
        value: float | None = None
        last_text = ""
        last_close_count = 0
        recent_member_frames: list[bool] = []

        def read_overlay(*, reuse_close_label: bool = False) -> float | None:
            nonlocal last_text, last_close_count
            overlay = self._capture()
            items = self._ocr(overlay, r".+")
            try:
                last_text = self._full_ocr_text(overlay)
            except ArenaReaderError:
                last_text = ""
            # Preserve the original normal two-query cadence. Recovery reuses
            # the full frame so its three-read allowance stays three OCR calls.
            last_close_count = len(
                self._matching_ocr_items(items, r"^閉じる$")
                if reuse_close_label else self._ocr(overlay, r"^閉じる$")
            )
            communication_dialog = self._retry_transient_communication_items(items)
            recent_member_frames.append(
                bool(self._matching_ocr_items(items, r"^体力$"))
                and bool(self._matching_ocr_items(items, r"^総合力$"))
                and "サポートボーナス" not in last_text
                and last_close_count == 0
                and not communication_dialog
            )
            del recent_member_frames[:-2]
            self._increment("support_bonus_overlay_reads")
            result = self._support_bonus_from_overlay(last_text, close_count=last_close_count)
            return None if communication_dialog else result

        while time.monotonic() < deadline:
            value = read_overlay()
            if value is not None:
                break
            self._sleep(0.08)
        self._add_timing("support_bonus_open_wait", time.perf_counter() - started)
        retries = getattr(self, "_support_bonus_retried_teams", None)
        if retries is None:
            retries = set()
            self._support_bonus_retried_teams = retries
        if value is None and recent_member_frames == [True, True] and target.team_id not in retries:
            # This allowance belongs to the reader/backend, not member state;
            # a whole-provider retry cannot rearm it for the same team.
            retries.add(target.team_id)
            recovery_started = time.perf_counter()
            recovery_deadline = time.monotonic() + 2.0
            self._increment("support_bonus_open_retries")
            self._click(info_box, settle_seconds=0)
            reads = 0
            while reads < 3 and time.monotonic() < recovery_deadline:
                reads += 1
                value = read_overlay(reuse_close_label=True)
                if value is not None:
                    break
                if reads < 3 and time.monotonic() + 0.08 < recovery_deadline:
                    self._sleep(0.08)
            elapsed = time.perf_counter() - recovery_started
            self._record_duration_sample("support_bonus_recovery", elapsed)
            logger.info(json.dumps({
                "event": "arena_reader_recovery", "team_id": target.team_id,
                "action": "support_bonus_reclick", "succeeded": value is not None,
                "extra_ocr": reads, "extra_clicks": 1, "wall_seconds": round(elapsed, 6),
            }, ensure_ascii=False, sort_keys=True))
        if value is None:
            bonuses = re.findall(
                r"(?<![0-9])\+([0-9]+(?:\.[0-9]+)?)%",
                last_text,
            )
            raise ArenaReaderError(
                "support_bonus_missing",
                "support overlay did not reach one same-frame semantic decision: "
                f"bonuses={bonuses!r}, close_count={last_close_count}",
            )
        # The same info icon is the stable toggle on both own and opponent
        # member pages.  OCR's visible 閉じる label belongs to overlay content
        # and is not a reliable hit target across these two layouts.
        self._click(info_box, settle_seconds=0)
        self._assert_member_detail()
        return value

    @staticmethod
    def _support_bonus_from_overlay(
        text: str,
        *,
        close_count: int,
    ) -> float | None:
        """Resolve only a complete same-frame support-bonus overlay."""

        bonuses = re.findall(
            r"(?<![0-9])\+([0-9]+(?:\.[0-9]+)?)%",
            text,
        )
        if "サポートボーナス" not in text or len(bonuses) != 1 or close_count != 1:
            return None
        value = float(bonuses[0]) / 100
        if not 0 <= value <= 1:
            raise ArenaReaderError(
                "support_bonus_invalid",
                f"support bonus is outside [0,1]: {value}",
            )
        return value

    def read_params(self, target: TeamTarget, stage_number: int, slot: int) -> Sequence[int]:
        del target, stage_number, slot
        failure = "parameter OCR did not run"
        previous_values: tuple[int, ...] | None = None
        for attempt in range(4):
            image = self._capture()
            height, width = image.shape[:2]
            observations = self._ocr(
                image,
                r".+",
                roi=(0, 0, width, int(height * 0.25)),
            )
            label_items: dict[str, list[Any]] = {}
            for item in observations:
                label = _text(item)
                if label in {"ボーカル", "ダンス", "ビジュアル", "体力"}:
                    label_items.setdefault(label, []).append(item)
            label_boxes: tuple[tuple[int, int, int, int], ...] = ()
            if any(len(items) != 1 for items in label_items.values()):
                failure = "parameter labels were duplicated"
            else:
                try:
                    label_boxes = infer_parameter_label_boxes(
                        {label: _box(items[0]) for label, items in label_items.items()},
                        frame_height=height,
                    )
                except ValueError as error:
                    failure = (
                        f"parameter labels yielded {sorted(label_items)!r}: {error}"
                    )
                    label_boxes = ()
            if label_boxes:
                values: list[int] = []
                failure = ""
                for label, label_box in zip(
                    ("ボーカル", "ダンス", "ビジュアル", "体力"),
                    label_boxes,
                    strict=True,
                ):
                    candidates = sorted(
                        (
                            item
                            for item in observations
                            if re.fullmatch(r"[0-9]{1,6}", _text(item))
                            and int(width * 0.55) <= _box(item)[0] < int(width * 0.68)
                            and label_box[1] + label_box[3] - 6
                            <= _box(item)[1]
                            <= label_box[1] + label_box[3] + int(height * 0.04)
                        ),
                        key=lambda item: (_box(item)[0], _box(item)[1]),
                    )
                    if not candidates:
                        value_roi = (
                            int(width * 0.55),
                            max(0, label_box[1] + label_box[3] - 8),
                            int(width * 0.16),
                            int(height * 0.055),
                        )
                        candidates = sorted(
                            self._ocr(image, r"^[0-9]{1,6}$", roi=value_roi),
                            key=lambda item: (_box(item)[0], _box(item)[1]),
                        )
                    enhanced_value: int | None = None
                    enhanced_observations: tuple[str, ...] = ()
                    if not candidates:
                        enhanced_value, enhanced_observations = self._read_enhanced_numeric_roi(
                            image,
                            value_roi,
                        )
                        if enhanced_value is not None:
                            values.append(enhanced_value)
                            continue
                    if len(candidates) != 1:
                        numeric_observations = [
                            (_text(item), _box(item))
                            for item in observations
                            if re.fullmatch(r"[0-9]{1,6}", _text(item))
                        ]
                        failure = (
                            f"{label} yielded "
                            f"{[(_text(item), _box(item)) for item in candidates]!r}; "
                            f"numeric observations={numeric_observations!r}; "
                            f"enhanced observations={enhanced_observations!r}"
                        )
                        break
                    values.append(int(_text(candidates[0])))
                if not failure and len(values) == 4:
                    current_values = tuple(values)
                    if current_values == previous_values:
                        return current_values
                    previous_values = current_values
                    failure = f"parameter values have not stabilized: {current_values!r}"
            if attempt < 3:
                self._sleep(0.25)
        raise ArenaReaderError("param_value_ambiguous", f"stable in-memory reads failed: {failure}")

    def _read_enhanced_numeric_roi(
        self,
        image: Any,
        roi: tuple[int, int, int, int],
    ) -> tuple[int | None, tuple[str, ...]]:
        from collections import Counter

        import cv2

        x, y, width, height = roi
        crop = image[y : y + height, x : x + width]
        if crop.shape[:2] != (height, width):
            return None, ()
        enlarged_2x = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        enlarged_3x = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        enlarged_4x = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        contrast = cv2.convertScaleAbs(enlarged_3x, alpha=1.5, beta=-20)
        gray = cv2.cvtColor(enlarged_3x, cv2.COLOR_BGR2GRAY)
        _, binary_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2BGR)
        observed: list[str] = []
        votes: Counter[int] = Counter()
        for name, variant in (
            ("2x", enlarged_2x),
            ("3x", enlarged_3x),
            ("4x", enlarged_4x),
            ("contrast", contrast),
            ("binary", binary),
        ):
            results = self._ocr(variant, r"^[0-9]{1,6}$")
            texts = tuple(_text(item) for item in results if re.fullmatch(r"[0-9]{1,6}", _text(item)))
            observed.append(f"{name}:{','.join(texts) or '-'}")
            if len(texts) == 1:
                votes[int(texts[0])] += 1
        ranked = votes.most_common()
        if ranked and ranked[0][1] >= 2 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            return ranked[0][0], tuple(observed)
        return None, tuple(observed)

    @staticmethod
    def _p_item_decision_evidence(
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
            "similarity": (
                None if decision.similarity is None else round(decision.similarity, 6)
            ),
            "margin": None if decision.margin is None else round(decision.margin, 6),
            "content_generation_max_mean_abs_error": round(
                decision.content_generation_max_mean_abs_error,
                6,
            ),
            "frames_byte_identical": decision.frames_byte_identical,
            "ranking_route": decision.ranking_route,
            "full_fallback_used": decision.full_fallback_used,
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

    def _confirm_p_item_detail(
        self, box: tuple[int, int, int, int], *, candidate_ids: Sequence[int],
        global_title_scope: bool = False, visual_tiebreak_ids: Sequence[int] = (),
        unrepresented_ids: Sequence[int] = (), source_images: Sequence[Any] = (),
        source_boxes: Sequence[tuple[int, int, int, int]] = (),
        plan: str | None = None,
    ) -> int:
        started = time.perf_counter()
        failed_before = len(getattr(self, "_runtime_duration_samples", {}).get("p_item_detail_failed_transaction", ()))
        if getattr(self, "_detail_failure_frames", None) is None:
            self._begin_detail_diagnostics(kind="p_item", box=list(box), source=source_images[0] if source_images else None)
        try:
            result = self._confirm_p_item_detail_once(
                box, candidate_ids=candidate_ids, global_title_scope=global_title_scope,
                visual_tiebreak_ids=visual_tiebreak_ids, unrepresented_ids=unrepresented_ids,
                source_images=source_images, source_boxes=source_boxes,
                plan=plan,
            )
        except Exception as error:
            if len(getattr(self, "_runtime_duration_samples", {}).get("p_item_detail_failed_transaction", ())) == failed_before:
                self._record_duration_sample("p_item_detail_failed_transaction", time.perf_counter() - started)
            self._persist_detail_failure(str(error))
            raise
        self._detail_failure_frames = None
        return result

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
    ) -> int:
        if not candidate_ids:
            raise ArenaReaderError(
                "p_item_detail_candidates_missing",
                "P-item detail fallback requires fixed-reference candidates",
            )
        started = time.perf_counter()

        def match_title_rows(text: str) -> tuple[int, ...]:
            matches: list[int] = []
            for title_row in str(text or "").splitlines():
                if not title_row.strip():
                    continue
                family_matcher = getattr(self.catalog, "p_item_detail_title_family_ids", None)
                if callable(family_matcher):
                    row_matches = family_matcher(title_row, plan=plan)
                    if not row_matches:
                        recovered = family_matcher(
                            title_row, plan=plan, allow_one_substitution=True,
                        )
                        # Preserve the existing visual boundary only for a
                        # one-character OCR repair. An exact complete title
                        # always searches its full catalog family.
                        old_matches = self.catalog.clicked_p_item_candidate_matches(
                            title_row, candidate_p_item_ids=candidate_ids,
                        ) if recovered else ()
                        if set(recovered).intersection(old_matches):
                            row_matches = recovered
                elif global_title_scope:
                    row_matches = self.catalog.clicked_p_item_global_detail_matches(
                        title_row,
                        candidate_p_item_ids=candidate_ids,
                        visual_tiebreak_p_item_ids=visual_tiebreak_ids,
                        unrepresented_p_item_ids=unrepresented_ids,
                        allow_one_substitution=True,
                    )
                else:
                    row_matches = self.catalog.clicked_p_item_candidate_matches(
                        title_row,
                        candidate_p_item_ids=candidate_ids,
                    )
                matches.extend(row_matches)
            return tuple(dict.fromkeys(matches))

        def match_title_rows_exact(text: str) -> tuple[int, ...]:
            matches: list[int] = []
            for title_row in str(text or "").splitlines():
                if not title_row.strip():
                    continue
                family_matcher = getattr(self.catalog, "p_item_detail_title_family_ids", None)
                if callable(family_matcher):
                    matches.extend(family_matcher(title_row, plan=plan))
                else:
                    matches.extend(self.catalog.clicked_p_item_global_detail_matches(
                        title_row,
                        candidate_p_item_ids=candidate_ids,
                        visual_tiebreak_p_item_ids=visual_tiebreak_ids,
                        unrepresented_p_item_ids=unrepresented_ids,
                        allow_one_substitution=False,
                    ))
            return tuple(dict.fromkeys(matches))

        def normalize_title_row(text: str) -> str:
            return re.sub(
                r"\s+",
                "",
                unicodedata.normalize("NFKC", str(text or "")),
            )

        def title_rows(text: str) -> tuple[tuple[str, str], ...]:
            rows: list[tuple[str, str]] = []
            for raw_row in str(text or "").splitlines():
                normalized = normalize_title_row(raw_row)
                if normalized:
                    rows.append((raw_row.strip(), normalized))
            return tuple(rows)

        def observe_p_item(image: Any) -> tuple[
            str,
            str,
            bool,
            tuple[_NormalizedPItemRow, ...],
            tuple[_NormalizedPItemRow, ...],
            tuple[dict[str, Any], ...],
        ]:
            observation = self._p_item_ocr_observation(image)

            def normalize_rows(geometric_rows: Sequence[Any]) -> tuple[
                _NormalizedPItemRow, ...
            ]:
                normalized_rows_list = []
                for geometric_row in geometric_rows:
                    if len(geometric_row) == 3:
                        raw, row_box, components = geometric_row
                    elif len(geometric_row) == 2:
                        raw, row_box = geometric_row
                        components = ((raw, row_box),)
                    else:
                        raise ArenaReaderError(
                            "p_item_ocr_panel_row_invalid",
                            "P-item OCR panel row has an unsupported shape",
                        )
                    normalized = normalize_title_row(raw)
                    if not normalized:
                        continue
                    normalized_components = tuple(
                        (
                            str(component_raw).strip(),
                            normalize_title_row(component_raw),
                            tuple(float(value) for value in component_box),
                        )
                        for component_raw, component_box in components
                        if normalize_title_row(component_raw)
                    )
                    normalized_rows_list.append(
                        (
                            str(raw).strip(),
                            normalized,
                            tuple(float(value) for value in row_box),
                            normalized_components,
                        )
                    )
                return tuple(normalized_rows_list)

            background_rows: tuple[_NormalizedPItemRow, ...] = ()
            atoms: tuple[dict[str, Any], ...] = ()
            if isinstance(observation, _PItemOcrObservation):
                full_text = observation.full_text
                panel_text = observation.panel_text
                anchors_visible = observation.anchors_visible
                normalized_rows = normalize_rows(observation.panel_rows)
                background_rows = normalize_rows(observation.background_rows)
                if observation.all_items:
                    height, width = image.shape[:2]
                    atoms = tuple(
                        {
                            "text": _text(item),
                            "box": tuple(
                                value / scale
                                for value, scale in zip(
                                    _box(item),
                                    (width / 720.0, height / 1280.0) * 2,
                                    strict=True,
                                )
                            ),
                        }
                        for item in observation.all_items
                    )
            elif len(observation) == 4:
                full_text, panel_text, anchors_visible, geometric_rows = observation
                normalized_rows = normalize_rows(geometric_rows)
            elif len(observation) == 3:
                # Isolated tests and injected diagnostic backends predate row
                # geometry. Give their ordered rows stable synthetic positions;
                # production always supplies normalized OCR boxes.
                full_text, panel_text, anchors_visible = observation
                normalized_rows = tuple(
                    (
                        raw,
                        normalized,
                        (18.0, 16.0 + index * 48.0, 320.0, 24.0),
                        (
                            (
                                raw,
                                normalized,
                                (18.0, 16.0 + index * 48.0, 320.0, 24.0),
                            ),
                        ),
                    )
                    for index, (raw, normalized) in enumerate(
                        title_rows(panel_text)
                    )
                )
            else:
                raise ArenaReaderError(
                    "p_item_ocr_observation_invalid",
                    "P-item OCR observation has an unsupported shape",
                )
            return (
                str(full_text),
                str(panel_text),
                bool(anchors_visible),
                normalized_rows,
                background_rows,
                atoms,
            )

        def one_substitution(left: str, right: str) -> bool:
            return (
                len(left) >= 3
                and len(left) == len(right)
                and sum(a != b for a, b in zip(left, right)) == 1
            )

        def one_insertion_or_deletion(left: str, right: str) -> bool:
            if abs(len(left) - len(right)) != 1:
                return False
            shorter, longer = sorted((left, right), key=len)
            first_difference = next(
                (index for index, value in enumerate(shorter) if value != longer[index]),
                len(shorter),
            )
            return shorter[first_difference:] == longer[first_difference + 1 :]

        def numeric_source_row_extension(
            current_row: str,
            source_row: str,
        ) -> bool:
            if current_row == source_row or source_row not in current_row:
                return False
            residue = current_row.replace(source_row, "", 1)
            return bool(
                residue
                and any(character.isdigit() for character in residue)
                and re.fullmatch(r"[0-9A-Za-z.,%+\-]+", residue)
            )

        def verified_numeric_source_row_extension(
            components: tuple[
                tuple[str, str, tuple[float, float, float, float]],
                ...,
            ],
            source_row: str,
            source_box: tuple[float, float, float, float],
        ) -> bool:
            source_components = tuple(
                index
                for index, (_, normalized, component_box) in enumerate(components)
                if normalized == source_row
                and same_source_position(component_box, source_box)
            )
            for source_component in source_components:
                remaining = tuple(
                    normalized
                    for index, (_, normalized, _) in enumerate(components)
                    if index != source_component
                )
                if remaining and all(
                    any(character.isdigit() for character in value)
                    and re.fullmatch(r"[0-9A-Za-z.,%+\-]+", value)
                    for value in remaining
                ):
                    return True
            return False

        def same_source_position(
            current_box: tuple[float, float, float, float],
            source_box: tuple[float, float, float, float],
        ) -> bool:
            current_x, current_y, current_width, current_height = current_box
            source_x, source_y, source_width, source_height = source_box
            intersection_width = max(
                0.0,
                min(current_x + current_width, source_x + source_width)
                - max(current_x, source_x),
            )
            intersection_height = max(
                0.0,
                min(current_y + current_height, source_y + source_height)
                - max(current_y, source_y),
            )
            minimum_area = min(
                current_width * current_height,
                source_width * source_height,
            )
            if minimum_area <= 0:
                return False
            current_centre_y = current_y + current_height / 2.0
            source_centre_y = source_y + source_height / 2.0
            return (
                intersection_width * intersection_height / minimum_area >= 0.50
                and abs(current_centre_y - source_centre_y) <= 8.0
            )

        stable_source_title_signatures: frozenset[tuple[str, ...]] = frozenset()
        stable_source_title_rows: tuple[
            tuple[str, tuple[float, float, float, float]],
            ...,
        ] = ()
        stable_source_row_frames: tuple[
            tuple[Any, tuple[tuple[float, float, float, float], ...]],
            ...,
        ] = ()
        stable_source_background_frames: tuple[
            tuple[int, tuple[_NormalizedPItemRow, ...]], ...
        ] = ()
        background_majority_required = 2

        def source_row_pixels_unchanged(
            current_image: Any,
            current_box: tuple[float, float, float, float],
            source_index: int,
        ) -> bool:
            import math

            import numpy as np

            comparison_started = time.perf_counter()
            try:
                if (
                    not isinstance(current_image, np.ndarray)
                    or current_image.ndim != 3
                    or current_image.shape[2] != 3
                ):
                    return False
                height, width = current_image.shape[:2]
                for source_image, source_row_boxes in stable_source_row_frames:
                    if (
                        not isinstance(source_image, np.ndarray)
                        or source_image.shape != current_image.shape
                        or source_image.dtype != current_image.dtype
                    ):
                        continue
                    source_box = source_row_boxes[source_index]
                    if not same_source_position(current_box, source_box):
                        continue
                    current_x, current_y, current_width, current_height = current_box
                    source_x, source_y, source_width, source_height = source_box
                    left = math.floor(min(current_x, source_x) * width / 720.0)
                    top = math.floor(min(current_y, source_y) * height / 1280.0)
                    right = math.ceil(
                        max(current_x + current_width, source_x + source_width)
                        * width / 720.0
                    )
                    bottom = math.ceil(
                        max(current_y + current_height, source_y + source_height)
                        * height / 1280.0
                    )
                    # Compare the complete union at its original coordinates;
                    # clipping, resizing or mixing pixels from several source
                    # frames would no longer prove this source row unchanged.
                    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                        continue
                    if np.array_equal(
                        current_image[top:bottom, left:right],
                        source_image[top:bottom, left:right],
                    ):
                        return True
                return False
            finally:
                self._add_timing(
                    "p_item_source_row_pixel_identity",
                    time.perf_counter() - comparison_started,
                )

        if source_images:
            cache = getattr(self, "_p_item_source_ocr_cache", None)
            if cache is None:
                cache = {}
                self._p_item_source_ocr_cache = cache
            # ``read_p_item_ids`` clears this bounded cache once per member.
            # The frozen frame objects stay alive for all four slot
            # transactions, so their in-process identities are sufficient and
            # avoid hashing three full captures again for every slot.
            cache_key = tuple(id(image) for image in source_images)
            source_observations = cache.get(cache_key)
            if source_observations is None:
                ocr_started = time.perf_counter()
                try:
                    source_observations = tuple(
                        observe_p_item(source_image)
                        for source_image in source_images
                    )
                    self._assert_p_item_source_generation_stable(
                        source_images,
                        source_boxes,
                    )
                except Exception as error:
                    raise ArenaReaderError(
                        "p_item_source_title_evidence_invalid",
                        "P-item detail recovery could not freeze source-page "
                        "catalog-title evidence before clicking",
                    ) from error
                self._add_timing(
                    "p_item_source_title_ocr",
                    time.perf_counter() - ocr_started,
                )
                cache[cache_key] = source_observations
            signature_counts: dict[tuple[str, ...], int] = {}
            for _, _, _, source_rows, _, _ in source_observations:
                signature = tuple(row[1] for row in source_rows)
                if signature:
                    signature_counts[signature] = signature_counts.get(signature, 0) + 1
            strict_majority = len(source_observations) // 2 + 1
            background_majority_required = max(2, strict_majority)
            if len(source_observations) >= 2:
                stable_source_title_signatures = frozenset(
                    signature
                    for signature, count in signature_counts.items()
                    if count >= strict_majority
                )
                if stable_source_title_signatures:
                    stable_signature = next(iter(stable_source_title_signatures))
                    agreeing_rows = tuple(
                        source_rows
                        for _, _, _, source_rows, _, _ in source_observations
                        if tuple(row[1] for row in source_rows)
                        == stable_signature
                    )
                    stable_source_row_frames = tuple(
                        (source_image, tuple(row[2] for row in source_rows))
                        for source_image, (_, _, _, source_rows, _, _) in zip(
                            source_images, source_observations, strict=True,
                        )
                        if tuple(row[1] for row in source_rows) == stable_signature
                    )
                    stable_source_title_rows = tuple(
                        (
                            normalized,
                            tuple(
                                statistics.median(
                                    rows[index][2][coordinate]
                                    for rows in agreeing_rows
                                )
                                for coordinate in range(4)
                            ),
                        )
                        for index, normalized in enumerate(stable_signature)
                    )
                    # Extra rows never alter the old source signature or its
                    # pixel-row indexes. Only its agreeing source frames may
                    # contribute exact background evidence.
                    stable_source_background_frames = tuple(
                        (id(source_image), background_rows)
                        for source_image, (_, _, _, source_rows, background_rows, _)
                        in zip(source_images, source_observations, strict=True)
                        if tuple(row[1] for row in source_rows) == stable_signature
                    )
                else:
                    raise ArenaReaderError(
                        "p_item_source_title_evidence_unstable",
                        "P-item detail recovery could not establish a strict-"
                        "majority spatial source-row signature before clicking",
                    )
        safe_region = self._p_item_interaction_box(box)
        interaction_point = self._box_center_point(safe_region)
        self._click(interaction_point, settle_seconds=0)
        self._increment("p_item_detail_clicks")
        self._increment("p_item_safe_region_clicks")
        transaction_states = ["SOURCE_STABLE", "CLICK_SENT"]
        deadline = time.monotonic() + 3.0
        last_text = ""
        last_panel_text = ""
        last_title_text = ""
        last_error = "P-item detail OCR did not run"
        last_text_resolution = ""
        resolved: int | None = None
        candidate_title_seen = False
        detail_confirmed = False
        consecutive_source_reads = 0
        consecutive_non_source_reads = 0
        last_unique_match: int | None = None
        consecutive_unique_matches = 0
        unique_match_used_effect = False
        retried = False
        terminal_error: Exception | None = None
        last_source_visual_errors: tuple[float, ...] = ()
        last_source_match_route = "none"
        last_source_row_uncertain = False
        consecutive_source_transition_reads = 0
        last_frame_may_have_overlay = False
        open_wait_started = time.perf_counter()
        while time.monotonic() < deadline:
            last_text_resolution = ""
            diagnostic = getattr(self, "_detail_failure_frames", None)
            if diagnostic is not None:
                diagnostic.pop("p_item_text", None)
            try:
                image = self._capture()
                ocr_started = time.perf_counter()
                (
                    last_text,
                    last_panel_text,
                    member_anchors_visible,
                    current_rows,
                    _,
                    current_atoms,
                ) = observe_p_item(image)
                self._add_timing(
                    "p_item_detail_ocr",
                    time.perf_counter() - ocr_started,
                )
                current_title_signature = tuple(
                    normalized for _, normalized, _, _ in current_rows
                )
                last_source_row_uncertain = False
                used_source_rows: set[int] = set()
                used_background_rows: set[tuple[int, int]] = set()
                last_title_text = ""
                for raw, normalized, row_box, components in current_rows:
                    exact_sources = tuple(
                        (index,)
                        for index, (source_row, source_box) in enumerate(
                            stable_source_title_rows
                        )
                        if index not in used_source_rows
                        and normalized == source_row
                        and same_source_position(row_box, source_box)
                    )
                    if exact_sources:
                        (source_index,) = min(exact_sources)
                        used_source_rows.add(source_index)
                        continue
                    extended_sources = tuple(
                        (index,)
                        for index, (source_row, source_box) in enumerate(
                            stable_source_title_rows
                        )
                        if index not in used_source_rows
                        and same_source_position(row_box, source_box)
                        and numeric_source_row_extension(normalized, source_row)
                    )
                    verified_extended_sources = tuple(
                        (index,)
                        for (index,) in extended_sources
                        if verified_numeric_source_row_extension(
                            components,
                            stable_source_title_rows[index][0],
                            stable_source_title_rows[index][1],
                        )
                    )
                    if verified_extended_sources:
                        if match_title_rows_exact(raw):
                            last_title_text = raw
                            break
                        # A numeric value can be misread as one Latin letter and
                        # joined to an unchanged source label (for example,
                        # ``総合力`` + ``15Z``). The independently positioned
                        # source label still proves this is a source-row
                        # extension, not the detail title.
                        (source_index,) = min(verified_extended_sources)
                        used_source_rows.add(source_index)
                        continue
                    if extended_sources:
                        last_source_row_uncertain = True
                        break
                    edited_sources = tuple(
                        index
                        for index, (source_row, source_box) in enumerate(
                            stable_source_title_rows
                        )
                        if index not in used_source_rows
                        and same_source_position(row_box, source_box)
                        and one_insertion_or_deletion(normalized, source_row)
                    )
                    if edited_sources:
                        # A genuine catalog title (including a new '+') wins
                        # over source-row reuse, even with identical pixels.
                        if match_title_rows_exact(raw):
                            last_title_text = raw
                            break
                        unchanged_source = next(
                            (
                                index for index in edited_sources
                                if source_row_pixels_unchanged(image, row_box, index)
                            ),
                            None,
                        )
                        if unchanged_source is not None:
                            used_source_rows.add(unchanged_source)
                            self._increment("p_item_source_row_pixel_identity_reuses")
                            continue
                    approximate_source = any(
                        index not in used_source_rows
                        and same_source_position(row_box, source_box)
                        and one_substitution(normalized, source_row)
                        for index, (source_row, source_box) in enumerate(
                            stable_source_title_rows
                        )
                    )
                    if approximate_source:
                        # A one-character difference at a frozen source
                        # position is not removable evidence: doing so could
                        # promote effect prose into the title slot.
                        last_source_row_uncertain = True
                        break
                    extra_sources: set[tuple[int, int]] = set()
                    for frame_id, background_rows in stable_source_background_frames:
                        candidates = tuple(
                            index
                            for index, (_, source_row, source_box, _) in enumerate(
                                background_rows
                            )
                            if normalized == source_row
                            and same_source_position(row_box, source_box)
                        )
                        if len(candidates) != 1:
                            continue
                        source_index = candidates[0]
                        source_key = (frame_id, source_index)
                        source_box = background_rows[source_index][2]
                        if (
                            source_key in used_background_rows
                            or sum(
                                current_normalized == normalized
                                and same_source_position(current_box, source_box)
                                for _, current_normalized, current_box, _ in current_rows
                            ) != 1
                            or any(
                                index in used_source_rows
                                and source_row == normalized
                                and same_source_position(source_box, old_box)
                                for index, (source_row, old_box) in enumerate(
                                    stable_source_title_rows
                                )
                            )
                        ):
                            continue
                        extra_sources.add(source_key)
                    if (
                        len({frame_id for frame_id, _ in extra_sources})
                        >= background_majority_required
                        and not match_title_rows_exact(raw)
                    ):
                        used_background_rows.update(extra_sources)
                        self._increment("p_item_source_background_row_reuses")
                        continue
                    last_title_text = raw
                    break
                # The first row newly introduced by the detail panel is the
                # identity row. Later rows are effect prose and can contain
                # other catalog titles, so they are never matching evidence.
                raw_matches = match_title_rows(last_title_text)
                matches = raw_matches
                effect_disambiguated = False
                text_resolver = getattr(self.catalog, "resolve_clicked_p_item_text", None)
                if callable(text_resolver) and last_title_text and raw_matches:
                    # The first new row was fixed above without consulting
                    # effect prose. Merge only its original title components;
                    # every other same-frame OCR atom retains its own boundary.
                    title_components = {(value, tuple(box)) for value, _, box in components}
                    text_atoms = [
                        atom for atom in current_atoms
                        if (str(atom["text"]).strip(), tuple(atom["box"])) not in title_components
                    ]
                    title_index = len(text_atoms)
                    if current_atoms:
                        text_atoms.append({"text": last_title_text, "box": row_box})
                    text_started = time.perf_counter()
                    text_result = text_resolver(
                        text_atoms, title_index=title_index,
                        source_frames=tuple(value[5] for value in source_observations)
                        if source_images else (),
                        plan=plan, candidate_p_item_ids=candidate_ids,
                        visual_tiebreak_p_item_ids=visual_tiebreak_ids,
                        title_text=last_title_text, detail_text=last_text,
                        allow_one_substitution=not bool(match_title_rows_exact(last_title_text)),
                    )
                    self._add_timing("p_item_detail_text", time.perf_counter() - text_started)
                    self._increment(f"p_item_detail_text_{text_result.status}")
                    last_text_resolution = f"{text_result.status}: {text_result.reason}"
                    if diagnostic is not None:
                        diagnostic["p_item_text"] = {
                            "status": text_result.status, "reason": text_result.reason,
                            "ids": text_result.ids, **text_result.diagnostics,
                        }
                    matches = text_result.ids if text_result.status == "unique" else raw_matches
                    if text_result.status != "unique" and len(matches) == 1:
                        matches = ()
                    effect_disambiguated = text_result.status == "unique" and (
                        len(raw_matches) != 1 or text_result.ids != raw_matches
                    )
                elif len(raw_matches) > 1:
                    effect_matcher = getattr(
                        self.catalog,
                        "clicked_p_item_effect_detail_matches",
                        None,
                    )
                    if callable(effect_matcher):
                        effect_matches = effect_matcher(
                            last_text,
                            candidate_p_item_ids=raw_matches,
                        )
                        if (
                            len(effect_matches) == 1
                            and effect_matches[0] in raw_matches
                        ):
                            matches = effect_matches
                            effect_disambiguated = True
            except Exception as error:
                terminal_error = error
                last_error = str(error)
                raw_matches = ()
                matches = ()
                effect_disambiguated = False
                image = None
                current_title_signature = ()
                last_source_row_uncertain = False
            source_visible = False
            source_transition = False
            if terminal_error is not None:
                break
            if image is not None:
                last_frame_may_have_overlay = True
                try:
                    if source_images and source_boxes:
                        source_visible, last_source_visual_errors = (
                            self._p_item_source_frame_matches(
                            source_images,
                            image,
                            source_boxes,
                            member_anchors_visible,
                        )
                        )
                        if source_visible:
                            last_source_match_route = "visual_frozen_generation"
                        elif (
                            member_anchors_visible
                            and current_title_signature
                            and current_title_signature
                            in stable_source_title_signatures
                        ):
                            source_transition = True
                            last_source_match_route = "semantic_source_transition"
                        else:
                            last_source_match_route = "none"
                    else:
                        source_visible = (
                            "体力" in last_text and "総合力" in last_text
                        ) or bool(
                            self._ocr(image, r"^体力$")
                            and self._ocr(image, r"^総合力$")
                        )
                except Exception as error:
                    terminal_error = error
                    last_error = str(error)
                    break
            if source_visible:
                last_frame_may_have_overlay = False
                last_unique_match = None
                consecutive_unique_matches = 0
                unique_match_used_effect = False
                consecutive_source_reads += 1
                consecutive_non_source_reads = 0
                consecutive_source_transition_reads = 0
                last_error = "the original member page remained stable after the click"
            elif source_transition:
                last_unique_match = None
                consecutive_unique_matches = 0
                unique_match_used_effect = False
                consecutive_source_reads = 0
                consecutive_non_source_reads = 0
                consecutive_source_transition_reads += 1
                self._increment("p_item_source_transition_frames")
                last_error = (
                    "the frozen source title generation remained visible while "
                    "source pixels were transitioning"
                )
                if consecutive_source_transition_reads >= 8:
                    detail_confirmed = True
                    transaction_states.extend(
                        ("SOURCE_TRANSITION_EXHAUSTED", "AMBIGUOUS")
                    )
                    break
            elif len(matches) == 1:
                consecutive_source_transition_reads = 0
                current_match = matches[0]
                if last_unique_match == current_match:
                    consecutive_unique_matches += 1
                    unique_match_used_effect = (
                        unique_match_used_effect or effect_disambiguated
                    )
                else:
                    consecutive_unique_matches = 1
                    unique_match_used_effect = effect_disambiguated
                last_unique_match = current_match
                if consecutive_unique_matches < 2:
                    candidate_title_seen = True
                    detail_confirmed = True
                    consecutive_source_reads = 0
                    last_error = (
                        "P-item title needs a second consecutive targeted OCR "
                        f"confirmation for {current_match}"
                    )
                    self._sleep(0.12)
                    continue
                resolved = matches[0]
                if diagnostic is not None:
                    diagnostic["position"]["p_item_name"] = last_title_text
                if unique_match_used_effect:
                    self._increment("p_item_detail_effect_disambiguations")
                detail_confirmed = True
                transaction_states.extend(("DETAIL_CONFIRMED", "RESOLVED"))
                break
            elif matches:
                consecutive_source_transition_reads = 0
                last_unique_match = None
                consecutive_unique_matches = 0
                unique_match_used_effect = False
                candidate_title_seen = True
                detail_confirmed = True
                consecutive_source_reads = 0
                transaction_states.append("DETAIL_CONFIRMED")
                last_error = (
                    "P-item detail matched multiple candidate titles "
                    f"{matches!r}"
                )
            else:
                consecutive_source_transition_reads = 0
                last_unique_match = None
                consecutive_unique_matches = 0
                unique_match_used_effect = False
                consecutive_source_reads = 0
                consecutive_non_source_reads += 1
                if last_text:
                    if last_source_row_uncertain:
                        last_error = (
                            "P-item detail first changed row remained one OCR "
                            "substitution from a frozen source row"
                        )
                    else:
                        last_error = (
                            "P-item detail exposed no fixed candidate title "
                            f"within {tuple(candidate_ids)!r}"
                        )
                if consecutive_non_source_reads >= 2 and last_text:
                    detail_confirmed = True
                    transaction_states.extend(("DETAIL_CONFIRMED", "AMBIGUOUS"))
                    break
            if (
                not retried
                and not detail_confirmed
                and consecutive_source_reads >= 3
            ):
                self._click(interaction_point, settle_seconds=0)
                self._increment("p_item_detail_click_retries")
                self._increment("p_item_safe_region_clicks")
                retried = True
                consecutive_source_reads = 0
                transaction_states.append("CLICK_SENT_RETRY")
            elif retried and consecutive_source_reads >= 3:
                transaction_states.append("OPEN_FAILED")
                break
            self._sleep(0.12)
        self._add_timing(
            "p_item_detail_open_wait",
            time.perf_counter() - open_wait_started,
        )
        recovery_started = time.perf_counter()
        try:
            if (
                detail_confirmed
                or terminal_error is not None
                or last_frame_may_have_overlay
            ):
                self._dismiss_skill_card_detail()
                transaction_states.append("DISMISS_SENT")
            if source_images and source_boxes:
                self._assert_p_item_source_restored(
                    source_images,
                    source_boxes,
                )
            else:
                self._assert_member_detail()
            transaction_states.append("SOURCE_RESTORED")
        except ArenaReaderError as recovery_error:
            elapsed = time.perf_counter() - started
            self._add_timing("p_item_detail_transactions", elapsed)
            self._add_timing(
                "p_item_detail_close_restore",
                time.perf_counter() - recovery_started,
            )
            self._record_duration_sample("p_item_detail_failed_transaction", elapsed)
            raise ArenaReaderError(
                "p_item_detail_recovery_failed",
                "P-item detail transaction did not restore the original member page; "
                f"states={transaction_states!r}; cause={recovery_error}",
            ) from recovery_error
        self._add_timing(
            "p_item_detail_close_restore",
            time.perf_counter() - recovery_started,
        )
        elapsed = time.perf_counter() - started
        self._add_timing("p_item_detail_transactions", elapsed)
        if terminal_error is not None:
            self._record_duration_sample("p_item_detail_failed_transaction", elapsed)
            raise ArenaReaderError(
                "p_item_detail_parser_failed",
                "P-item detail parsing failed after the source page was restored; "
                f"states={transaction_states!r}; error={last_error}",
            ) from terminal_error
        if resolved is None:
            self._record_duration_sample("p_item_detail_failed_transaction", elapsed)
            error_code = (
                "p_item_detail_ambiguous"
                if detail_confirmed or candidate_title_seen
                else "p_item_detail_open_or_title_failed"
            )
            raise ArenaReaderError(
                error_code,
                f"P-item detail did not uniquely confirm {tuple(candidate_ids)!r}: "
                f"{last_error}; states={transaction_states!r}; "
                f"text_resolution={last_text_resolution!r}; "
                f"safe_box={safe_region!r}; click_point={interaction_point!r}; "
                f"retried={retried}; "
                f"source_match_route={last_source_match_route!r}; "
                f"source_row_uncertain={last_source_row_uncertain}; "
                f"source_visual_errors="
                f"{tuple(round(value, 6) for value in last_source_visual_errors)!r}; "
                f"title_OCR={last_title_text[:160]!r}; "
                f"panel_OCR={last_panel_text[:240]!r}; OCR={last_text[:400]!r}",
            )
        self._record_duration_sample("p_item_detail_transaction", elapsed)
        return resolved

    def _assert_p_item_reference_catalog_compatibility(self) -> None:
        """Bind fixed-gallery drift to a safe global-title confirmation route."""

        if getattr(self, "_p_item_catalog_compatibility_checked", False):
            return
        gallery = getattr(self.p_item_reader, "gallery", None)
        raw_gallery_ids = getattr(gallery, "p_item_ids", None)
        if raw_gallery_ids is None:
            # Explicit injected readers are used by isolated tests and
            # diagnostics. Production Task085PItemReader always exposes its
            # fixed gallery and therefore cannot bypass this compatibility gate.
            return
        gallery_ids = frozenset(int(value) for value in raw_gallery_ids)
        catalog_ids = frozenset(self.catalog.p_item_business_ids())
        required_ids = frozenset(
            self.catalog.arena_p_item_reference_required_ids()
        )
        missing = tuple(sorted(required_ids - gallery_ids))
        unknown = tuple(sorted(gallery_ids - catalog_ids))
        self._p_item_reference_missing_arena_ids = missing
        self._p_item_reference_unknown_catalog_ids = unknown
        self._p_item_reference_provisional_ids = tuple(
            sorted(
                gallery_ids.intersection(
                    getattr(gallery, "provisional_p_item_ids", ())
                )
            )
        )
        self._p_item_reference_gallery_ids = tuple(sorted(gallery_ids))
        self._p_item_catalog_compatibility_checked = True

    def _assert_skill_card_reference_catalog_compatibility(self) -> None:
        """Record forward RIS drift without treating the fixed gallery as a gate."""

        if getattr(self, "_skill_card_catalog_compatibility_checked", False):
            return
        if not hasattr(self, "_card_reference_gallery"):
            # Minimal object.__new__ test doubles do not represent a packaged
            # runtime and intentionally omit the gallery cache slot.
            return
        gallery = self._card_references()
        raw_gallery_ids = getattr(gallery, "business_ids", None)
        catalog_ids_resolver = getattr(
            self.catalog,
            "skill_card_business_ids",
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
        self._skill_card_reference_gallery_ids = tuple(sorted(gallery_ids))
        self._skill_card_reference_missing_catalog_ids = tuple(
            sorted(catalog_ids - gallery_ids)
        )
        self._skill_card_catalog_compatibility_checked = True

    def _skill_card_forward_drift_for_plan(self, plan: str) -> tuple[int, ...]:
        """Return active-plan IDs added after the fixed card gallery."""

        self._assert_skill_card_reference_catalog_compatibility()
        missing = tuple(
            getattr(self, "_skill_card_reference_missing_catalog_ids", ())
        )
        if not missing:
            return ()
        try:
            return self.catalog.skill_card_candidates_for_plan(
                missing,
                plan=plan,
            )
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_reference_catalog_invalid",
                "active RIS skill-card drift could not be restricted to the stage plan",
            ) from error

    def _skill_card_catalog_candidates_for_plan(self, plan: str) -> tuple[int, ...]:
        """Return the complete active-plan title scope for drift-only recovery."""

        try:
            candidate_ids = self.catalog.skill_card_candidates_for_plan(
                self.catalog.skill_card_business_ids(),
                plan=plan,
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
        source_type_resolver = getattr(
            self.catalog,
            "skill_card_source_type",
            None,
        )
        if source_type_resolver is None:
            raise ArenaReaderError(
                "skill_card_reference_catalog_invalid",
                "active RIS skill-card drift has no source-type resolver",
            )
        try:
            source_types = {
                str(source_type_resolver(int(card_id)))
                for card_id in missing_card_ids
            }
        except (ArenaCatalogError, TypeError, ValueError) as error:
            raise ArenaReaderError(
                "skill_card_reference_catalog_invalid",
                "active RIS skill-card drift has no reliable source-type mapping",
            ) from error
        unsupported_source_types = tuple(
            sorted(source_types - {"pIdol", "support", "produce"})
        )
        if not source_types or unsupported_source_types:
            gallery_ids = getattr(self, "_skill_card_reference_gallery_ids", ())
            gallery_id_range = (
                (min(gallery_ids), max(gallery_ids)) if gallery_ids else ()
            )
            raise ArenaReaderError(
                "skill_card_reference_gallery_update_required",
                "active RIS added unknown skill-card sources that "
                "require a matching gallery: "
                "missing_catalog_reference_ids="
                f"{getattr(self, '_skill_card_reference_missing_catalog_ids', ())!r}; "
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
            self._start_skill_card_gallery_fallback()
        return tuple(sorted(slots))

    def _start_skill_card_gallery_fallback(self) -> None:
        if getattr(self, "_skill_card_gallery_fallback_started", False):
            return
        self._skill_card_gallery_fallback_started = True
        missing_ids = getattr(self, "_skill_card_reference_missing_catalog_ids", ())
        self._increment("skill_card_gallery_fallback_starts")
        logger.info(json.dumps({
            "event": "arena_skill_card_gallery_fallback",
            "fallback_status": "started",
            "missing_catalog_reference_ids": list(missing_ids),
            "maximum_ordinary_slots_per_row": 5,
            "maximum_detail_transactions_per_slot": 1,
        }, ensure_ascii=False, sort_keys=True))
        arena_task_log.gallery_fallback(
            getattr(self, "context", None), missing_ids, _base_logger,
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
        started = time.perf_counter()
        before = dict(self._runtime_counts)
        missing_ids = tuple(getattr(self, "_skill_card_reference_missing_catalog_ids", ()))
        record = {
            "event": "arena_skill_card_gallery_fallback",
            "team_id": target.team_id,
            "stage_number": stage_number, "member_slot": member_slot,
            "group_index": group_index, "card_slot": card_slot,
            "missing_catalog_reference_ids": list(missing_ids),
            "additional_zero_detail": additional_zero_detail,
            "fallback_status": "attempted",
        }
        self._increment("skill_card_gallery_fallback_attempts")
        if additional_zero_detail:
            self._increment("skill_card_gallery_fallback_additional_zero_details")
        logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True))
        try:
            resolved = operation()
        except BaseException as error:
            cancelled = isinstance(error, ArenaTaskCancelled)
            record.update(fallback_status="cancelled" if cancelled else "failed", error=str(error))
            self._increment(
                "skill_card_gallery_fallback_cancellations" if cancelled
                else "skill_card_gallery_fallback_failures"
            )
            if isinstance(error, ArenaReaderError):
                annotated = ArenaReaderError(
                    error.code,
                    f"{error.detail}; gallery_fallback_status=failed; "
                    f"missing_catalog_reference_ids={missing_ids!r}",
                    retry_whole_read=error.retry_whole_read,
                )
                failure = getattr(self, "_last_detail_failure_location", None)
                if failure is not None and str(failure[0]) in str(error):
                    self._last_detail_failure_location = (str(annotated), failure[1])
                raise annotated from error
            raise
        else:
            record.update(fallback_status="succeeded", card_id=resolved.card_id)
            self._increment("skill_card_gallery_fallback_successes")
            return resolved
        finally:
            elapsed = time.perf_counter() - started
            self._add_timing("skill_card_gallery_fallback", elapsed)
            self._record_duration_sample("skill_card_gallery_fallback", elapsed)
            record["wall_seconds"] = round(elapsed, 6)
            record["operation_counts"] = {
                name: value - before.get(name, 0)
                for name, value in self._runtime_counts.items()
                if value > before.get(name, 0)
            }
            logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _p_item_interaction_box(
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Keep Maa's click target inside the central, unobscured P-item artwork."""

        x, y, width, height = box
        if width < 16 or height < 16:
            raise ArenaReaderError(
                "p_item_interaction_box_invalid",
                f"P-item box is too small for a safe click: {box!r}",
            )
        inset_x = max(2, width // 4)
        inset_y = max(2, height // 4)
        return (
            x + inset_x,
            y + inset_y,
            width - inset_x * 2,
            height - inset_y * 2,
        )

    def _capture_stable_p_item_generation(
        self,
        boxes: Sequence[tuple[int, int, int, int]],
        *,
        initial_image: Any | None = None,
        timeout_seconds: float = 1.5,
        interval_seconds: float = 0.08,
    ) -> tuple[tuple[Any, Any, Any], tuple[float, ...], int]:
        """Wait for three consecutive frames from one P-item render generation."""

        if self.p_item_reader is None:
            raise ArenaReaderError(
                "p_item_reader_missing",
                "P-item content stability requires an initialized reader",
            )
        threshold = float(
            self.p_item_reader.content_generation_max_mean_abs_error
        )
        frames: list[Any] = []
        rejected_windows = 0
        last_errors: tuple[float, ...] = ()
        deadline = time.monotonic() + timeout_seconds
        started = time.perf_counter()
        pending = initial_image
        while time.monotonic() < deadline:
            image = pending if pending is not None else self._capture()
            pending = None
            frames.append(image)
            if len(frames) > 3:
                frames.pop(0)
            if len(frames) == 3:
                try:
                    last_errors = tuple(
                        float(value)
                        for value in self.p_item_reader.content_generation_errors(
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
                        "P-item content-generation measurement returned "
                        f"{len(last_errors)} slots for {len(boxes)} boxes",
                    )
                if all(value <= threshold for value in last_errors):
                    self._add_timing(
                        "p_item_content_stable_wait",
                        time.perf_counter() - started,
                    )
                    return (
                        (frames[0], frames[1], frames[2]),
                        last_errors,
                        rejected_windows,
                    )
                rejected_windows += 1
                self._increment("p_item_content_generation_resets")
            if interval_seconds > 0:
                self._sleep(interval_seconds)
        self._add_timing(
            "p_item_content_stable_wait",
            time.perf_counter() - started,
        )
        self._p_item_generation_evidence = {
            "frame_ids": [frame_identifier(image) for image in frames],
            "content_stable": False,
            "identity_resolved": False,
            "content_generation_errors": [
                round(value, 6) for value in last_errors
            ],
            "content_generation_threshold": threshold,
            "content_generation_rejected_windows": rejected_windows,
        }
        raise ArenaReaderError(
            "p_item_content_generation_unstable",
            "P-item slots did not yield three consecutive stable frames within "
            f"{timeout_seconds:.2f}s; threshold={threshold:.6f}; "
            f"last_errors={[round(value, 6) for value in last_errors]!r}",
        )

    def read_p_item_ids(
        self,
        target: TeamTarget,
        stage_number: int,
        slot: int,
    ) -> Sequence[int]:
        self._p_item_source_ocr_cache = {}
        if self.p_item_reader is None:
            self.p_item_reader = Task085PItemReader.from_model_root()
        plan = self.season.stages[stage_number - 1].plan
        self._assert_p_item_reference_catalog_compatibility()
        first = self._capture()
        height, width = first.shape[:2]
        icon = int(width * 0.09)
        boxes = tuple(
            (int(width * x), int(height * 0.207), icon, icon)
            for x in (0.05, 0.147, 0.247, 0.34)
        )
        decisions: tuple[PItemReferenceDecision, ...] = ()
        images: tuple[Any, ...] = ()
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
            ) = self._capture_stable_p_item_generation(
                boxes,
                initial_image=first if generation_attempt == 1 else None,
            )
            content_generation_rejected_windows += rejected_windows
            started = time.perf_counter()
            try:
                decisions = tuple(
                    self.p_item_reader.read(images, boxes, plan=plan)
                )
            except (PItemReferenceError, OSError, KeyError, TypeError, ValueError) as error:
                raise ArenaReaderError(
                    "p_item_reference_runtime_failed",
                    "fixed P-item reference matching failed",
                ) from error
            matching_seconds += time.perf_counter() - started
            if len(decisions) != 4:
                raise ArenaReaderError(
                    "p_item_count_mismatch",
                    f"P-item reference reader returned {len(decisions)} slots",
                )
            if all(decision.accepted for decision in decisions):
                break
            if generation_attempt == 1:
                self._increment("p_item_generation_resamples")
                continue
            break
        self._add_timing("p_item_reference_matching", matching_seconds)
        ambiguous = tuple(
            (index, decision)
            for index, decision in enumerate(decisions, start=1)
            if not decision.accepted
        )
        detail_eligible = tuple(
            (index, decision)
            for index, decision in ambiguous
            if decision.reason
            in {
                "reference_margin_below_threshold",
                "reference_similarity_below_threshold",
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
        if getattr(self, "_p_item_catalog_compatibility_checked", False):
            gallery_ids = frozenset(
                getattr(self, "_p_item_reference_gallery_ids", ())
            )
            missing_reference_ids = tuple(
                sorted(
                    frozenset(
                        self.catalog.arena_p_item_reference_required_ids(plan=plan)
                    )
                    - gallery_ids
                )
            )
        else:
            missing_reference_ids = tuple(
                getattr(self, "_p_item_reference_missing_arena_ids", ())
            )
        unknown_catalog_ids = frozenset(
            getattr(self, "_p_item_reference_unknown_catalog_ids", ())
        )
        provisional_reference_ids = frozenset(
            getattr(self, "_p_item_reference_provisional_ids", ())
        )
        visual_tiebreak_blocked_ids = tuple(
            sorted(
                frozenset(missing_reference_ids).union(
                    provisional_reference_ids
                )
            )
        )
        force_missing_reference_detail = bool(missing_reference_ids)
        force_catalog_superset_detail = bool(unknown_catalog_ids)
        force_global_detail = (
            force_missing_reference_detail or force_catalog_superset_detail
        )
        provisional_plan_ids = frozenset(
            provisional_reference_ids.intersection(
                self.catalog.arena_p_item_reference_required_ids(plan=plan)
            )
            if provisional_reference_ids
            else ()
        )
        detail_by_slot = dict(detail_eligible)
        global_detail_slots: set[int] = set()
        if force_global_detail:
            detail_by_slot.update(
                (index, decision)
                for index, decision in enumerate(decisions, start=1)
                if decision.status != "EMPTY"
            )
            global_detail_slots.update(detail_by_slot)
        else:
            # Plan reachability only says that a provisional ID may occur; it
            # does not authorize opening unrelated, already accepted slots.
            # Keep the provisional safety check local to the slot whose
            # accepted ID or unresolved candidate set actually observed it.
            for index, decision in enumerate(decisions, start=1):
                observed_ids = (
                    (decision.p_item_id,)
                    if decision.accepted and decision.p_item_id is not None
                    else decision.candidates
                )
                if provisional_reference_ids.intersection(observed_ids):
                    detail_by_slot[index] = decision
                    global_detail_slots.add(index)
        detail_budget = self.p_item_reader.maximum_detail_fallbacks_per_member
        budgeted_detail_slots = set(detail_by_slot) - global_detail_slots
        if force_global_detail:
            # Catalog drift is the one case that intentionally confirms every
            # non-empty slot.  It is bounded by the four-slot page itself.
            detail_budget = 4
            budgeted_detail_slots = set(detail_by_slot)
        if len(budgeted_detail_slots) > detail_budget:
            raise ArenaReaderError(
                "p_item_detail_budget_exceeded",
                f"P-item bounded detail fallbacks {len(budgeted_detail_slots)} "
                "exceed the per-member budget "
                f"{detail_budget}",
            )

        screen_resolved_ids: list[int] = []
        screen_diagnostics: list[dict[str, Any]] = []
        global_detail_candidates = (
            self.catalog.arena_p_item_reference_required_ids(plan=plan)
            if global_detail_slots
            else ()
        )
        for screen_slot, (box, decision) in enumerate(
            zip(boxes, decisions, strict=True),
            start=1,
        ):
            engine_slot = p_item_screen_slot_to_engine_slot(screen_slot)
            if screen_slot in detail_by_slot:
                use_global_detail = screen_slot in global_detail_slots
                detail_candidates = (
                    global_detail_candidates
                    if use_global_detail
                    else decision.candidates
                )
                decision_observed_ids = (
                    (decision.p_item_id,)
                    if decision.accepted and decision.p_item_id is not None
                    else decision.candidates
                )
                visual_tiebreak_ids = (
                    decision_observed_ids if decision.accepted else ()
                )
                self._begin_detail_diagnostics(
                    target=target, stage_number=stage_number, member_slot=slot,
                    kind="p_item", screen_slot=screen_slot, source=images[0],
                )
                resolved_id = self._confirm_p_item_detail(
                    box,
                    candidate_ids=detail_candidates,
                    global_title_scope=use_global_detail,
                    visual_tiebreak_ids=visual_tiebreak_ids,
                    unrepresented_ids=visual_tiebreak_blocked_ids,
                    source_images=images,
                    source_boxes=boxes,
                    plan=plan,
                )
                screen_resolved_ids.append(resolved_id)
                screen_diagnostics.append(
                    self._p_item_decision_evidence(
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
            path = (
                "stable_empty_slot"
                if decision.status == "EMPTY"
                else "three_frame_rendered_reference"
            )
            self._increment(
                "p_item_empty_slots"
                if decision.status == "EMPTY"
                else "p_item_exact_reference_resolutions"
            )
            screen_diagnostics.append(
                self._p_item_decision_evidence(
                    engine_slot,
                    decision,
                    path=path,
                    screen_slot=screen_slot,
                )
            )
        gallery = getattr(self.p_item_reader, "gallery", None)
        resolved_ids = p_item_screen_order_to_engine_order(screen_resolved_ids)
        self._p_item_diagnostics = p_item_screen_order_to_engine_order(screen_diagnostics)
        self._p_item_generation_evidence = {
            "frame_ids": [frame_identifier(image) for image in images],
            "fresh_resamples": generation_attempt - 1,
            "content_stable": True,
            "identity_resolved": True,
            "content_generation_errors": [
                round(value, 6) for value in content_generation_errors
            ],
            "content_generation_threshold": (
                self.p_item_reader.content_generation_max_mean_abs_error
            ),
            "content_generation_rejected_windows": (
                content_generation_rejected_windows
            ),
            "reference_gallery_sha256": getattr(gallery, "gallery_sha256", None),
            "background_workers": 2,
            "ranking_routes": [decision.ranking_route for decision in decisions],
            "full_fallbacks": sum(
                int(decision.full_fallback_used) for decision in decisions
            ),
            "screen_slot_to_engine_slot": [1, 4, 3, 2],
            "matching_seconds": round(matching_seconds, 6),
            "missing_arena_reference_ids": list(missing_reference_ids),
            "unknown_catalog_gallery_ids": sorted(unknown_catalog_ids),
            "provisional_reference_ids": sorted(provisional_reference_ids),
            "provisional_plan_ids": sorted(provisional_plan_ids),
            "catalog_drift_global_detail_confirmations": (
                len(global_detail_slots) if force_missing_reference_detail else 0
            ),
            "catalog_superset_global_detail_confirmations": (
                len(global_detail_slots) if force_catalog_superset_detail else 0
            ),
            "provisional_reference_global_detail_confirmations": (
                sum(
                    bool(
                        provisional_reference_ids.intersection(
                            (decision.p_item_id,)
                            if decision.accepted and decision.p_item_id is not None
                            else decision.candidates
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

    def prepare_skill_card_group(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> None:
        required_groups = (0, 1) if group_index == 1 else (0,)
        if all(
            index in self._card_rows
            and index in self._card_images
            and len(self._card_count_frames.get(index, ())) == 3
            for index in required_groups
        ):
            return
        if group_index == 1:
            self._swipe(vertical="up")
        first_error: ArenaReaderError | None = None
        try:
            self._capture_stable_card_groups(
                required_groups,
                allow_fixed_secondary=group_index == 1,
            )
            if group_index == 1:
                self._secondary_fixed_slot_fallback_enabled = True
            return
        except ArenaReaderError as error:
            if group_index != 1:
                raise
            first_error = error

        # A second gesture is allowed only when screen geometry proves that
        # the first gesture did not reach the calibrated position. If the
        # primary row did move, repeating the gesture cannot expose more
        # content and only adds latency; the fixed-slot presence gate has
        # already been attempted on every stable frame.
        image = self._capture()
        rows = canonical_skill_card_rows(
            image.shape,
            self._raw_card_candidate_boxes(image),
        )
        assert first_error is not None
        if rows and rows[0][0][1] <= int(image.shape[0] * 0.205):
            if first_error.code == "skill_card_layout_incomplete":
                raise ArenaReaderError(
                    "skill_card_secondary_row_missing",
                    "primary row reached the calibrated position but group 1 lacked "
                    f"six stable fixed-slot card observations: {first_error}",
                ) from first_error
            raise first_error
        self._swipe(vertical="up")
        self._increment("corrective_swipes")
        self._capture_stable_card_groups(
            required_groups,
            allow_fixed_secondary=True,
        )
        self._secondary_fixed_slot_fallback_enabled = True

    @staticmethod
    def _card_rows_shifted(
        actual: Sequence[tuple[int, int, int, int]],
        expected: Sequence[tuple[int, int, int, int]],
        *,
        tolerance: int = 3,
    ) -> bool:
        return any(
            abs(actual_value - expected_value) > tolerance
            for actual_box, expected_box in zip(actual, expected, strict=True)
            for actual_value, expected_value in zip(actual_box, expected_box, strict=True)
        )

    @staticmethod
    def _card_content_generation_deltas(
        actual: Sequence[tuple[str, Any]],
        expected: Sequence[tuple[str, Any]],
        *,
        maximum_mean_absolute_error: float = (
            CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR
        ),
    ) -> tuple[dict[str, Any], ...]:
        """Measure why a rendered-card generation differs after geometry settles."""

        import numpy as np

        if len(actual) != len(expected) or not actual:
            return (
                {
                    "slot": 0,
                    "visual_group_changed": False,
                    "shape_changed": True,
                    "mean_absolute_error": None,
                    "shifted": True,
                },
            )
        deltas: list[dict[str, Any]] = []
        for slot, ((actual_group, actual_query), (expected_group, expected_query)) in enumerate(
            zip(actual, expected, strict=True),
            start=1,
        ):
            actual_array = np.asarray(actual_query)
            expected_array = np.asarray(expected_query)
            visual_group_changed = actual_group != expected_group
            shape_changed = actual_array.shape != expected_array.shape
            motion = None
            zero_mean_motion = None
            edge_motion = None
            structural_correlation = None
            mean_color_shift = None
            if not shape_changed:
                actual_float = actual_array.astype(np.float32)
                expected_float = expected_array.astype(np.float32)
                difference = actual_float - expected_float
                motion = float(
                    np.mean(np.abs(difference))
                )
                channel_shift = np.mean(difference, axis=(0, 1), keepdims=True)
                mean_color_shift = float(np.mean(np.abs(channel_shift)))
                zero_mean_motion = float(
                    np.mean(np.abs(difference - channel_shift))
                )
                actual_gray = np.mean(actual_float, axis=2)
                expected_gray = np.mean(expected_float, axis=2)
                actual_edges = np.concatenate(
                    (
                        np.diff(actual_gray, axis=0).ravel(),
                        np.diff(actual_gray, axis=1).ravel(),
                    )
                )
                expected_edges = np.concatenate(
                    (
                        np.diff(expected_gray, axis=0).ravel(),
                        np.diff(expected_gray, axis=1).ravel(),
                    )
                )
                edge_motion = float(np.mean(np.abs(actual_edges - expected_edges)))
                actual_centered = actual_float.ravel() - float(np.mean(actual_float))
                expected_centered = expected_float.ravel() - float(
                    np.mean(expected_float)
                )
                denominator = float(
                    np.linalg.norm(actual_centered) * np.linalg.norm(expected_centered)
                )
                structural_correlation = (
                    1.0
                    if denominator == 0.0 and motion == 0.0
                    else (
                        0.0
                        if denominator == 0.0
                        else float(
                            np.dot(actual_centered, expected_centered) / denominator
                        )
                    )
                )
            deltas.append(
                {
                    "slot": slot,
                    "visual_group_changed": visual_group_changed,
                    "shape_changed": shape_changed,
                    "mean_absolute_error": motion,
                    "mean_color_shift": mean_color_shift,
                    "zero_mean_absolute_error": zero_mean_motion,
                    "edge_mean_absolute_error": edge_motion,
                    "structural_correlation": structural_correlation,
                    "shifted": (
                        visual_group_changed
                        or shape_changed
                        or motion is None
                        or motion > maximum_mean_absolute_error
                    ),
                }
            )
        return tuple(deltas)

    @classmethod
    def _card_content_generation_shifted(
        cls,
        actual: Sequence[tuple[str, Any]],
        expected: Sequence[tuple[str, Any]],
        *,
        maximum_mean_absolute_error: float = (
            CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR
        ),
    ) -> bool:
        """Detect a new rendered-card generation after geometry has settled."""

        return any(
            bool(delta["shifted"])
            for delta in cls._card_content_generation_deltas(
                actual,
                expected,
                maximum_mean_absolute_error=maximum_mean_absolute_error,
            )
        )

    @staticmethod
    def _aligned_card_content_query(
        image: Any,
        box: tuple[int, int, int, int],
        shift_x: int,
        shift_y: int,
        *,
        current_side: bool,
    ) -> Any:
        """Return the existing 16x16 content query on one common overlap.

        ``shift_x``/``shift_y`` describe where the current render moved relative
        to the frozen source.  Both sides discard the non-overlapping outer
        pixel before running the exact 96x96 -> 16x16 production resize.  This
        is deliberately narrower than changing the global MAE threshold.
        """

        import cv2
        import numpy as np

        if shift_x not in {-1, 0, 1} or shift_y not in {-1, 0, 1}:
            raise ArenaReaderError(
                "skill_card_source_alignment_invalid",
                f"source-card alignment is outside the fixed +/-1 domain: "
                f"({shift_x}, {shift_y})",
            )
        array = np.asarray(image)
        x, y, width, height = box
        overlap_width = width - abs(shift_x)
        overlap_height = height - abs(shift_y)
        if (
            array.ndim != 3
            or array.shape[2] not in {3, 4}
            or overlap_width < 8
            or overlap_height < 8
        ):
            raise ArenaReaderError(
                "skill_card_source_alignment_invalid",
                f"source-card alignment ROI is invalid: box={box!r}",
            )
        offset_x = max(0, shift_x if current_side else -shift_x)
        offset_y = max(0, shift_y if current_side else -shift_y)
        crop_x = x + offset_x
        crop_y = y + offset_y
        crop = array[
            crop_y : crop_y + overlap_height,
            crop_x : crop_x + overlap_width,
            :3,
        ]
        if crop.shape[:2] != (overlap_height, overlap_width):
            raise ArenaReaderError(
                "skill_card_source_alignment_invalid",
                f"source-card alignment ROI is outside the frame: box={box!r}; "
                f"shift=({shift_x}, {shift_y}); current_side={current_side}",
            )
        live = cv2.resize(
            np.ascontiguousarray(crop),
            (96, 96),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.resize(
            live,
            (16, 16),
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint8)

    @classmethod
    def _fixed_slot_aligned_content_proof(
        cls,
        source_frames: Sequence[Any],
        current_image: Any,
        box: tuple[int, int, int, int],
        visual_group: str,
    ) -> dict[str, Any]:
        """Prove one current raw slot matches 2/3 frozen frames at one shift.

        A clean-reference family is only a routing hint here.  Acceptance uses
        the existing production content query and unchanged 0.75 MAE gate.
        Exactly one native-pixel alignment must reach quorum, which rejects
        flat or repetitive content that cannot identify one physical render.
        """

        if len(source_frames) != 3 or not visual_group:
            return {
                "matched": False,
                "reason": "source_generation_invalid",
                "passing_shifts": [],
            }
        shift_diagnostics: list[dict[str, Any]] = []
        passing_shifts: list[dict[str, Any]] = []
        try:
            for shift_y in (-1, 0, 1):
                for shift_x in (-1, 0, 1):
                    current_query = cls._aligned_card_content_query(
                        current_image,
                        box,
                        shift_x,
                        shift_y,
                        current_side=True,
                    )
                    errors: list[float | None] = []
                    support = 0
                    for source_image in source_frames:
                        source_query = cls._aligned_card_content_query(
                            source_image,
                            box,
                            shift_x,
                            shift_y,
                            current_side=False,
                        )
                        delta = cls._card_content_generation_deltas(
                            ((visual_group, current_query),),
                            ((visual_group, source_query),),
                        )[0]
                        error = delta["mean_absolute_error"]
                        errors.append(
                            None if error is None else round(float(error), 6)
                        )
                        if not bool(delta["shifted"]):
                            support += 1
                    diagnostic = {
                        "shift": [shift_x, shift_y],
                        "source_support": support,
                        "source_errors": errors,
                    }
                    shift_diagnostics.append(diagnostic)
                    if support >= 2:
                        passing_shifts.append(diagnostic)
        except ArenaReaderError as error:
            return {
                "matched": False,
                "reason": error.code,
                "passing_shifts": [],
            }
        matched = len(passing_shifts) == 1
        return {
            "matched": matched,
            "reason": (
                "unique_alignment"
                if matched
                else (
                    "no_alignment"
                    if not passing_shifts
                    else "alignment_not_unique"
                )
            ),
            "shift": passing_shifts[0]["shift"] if matched else None,
            "source_support": (
                passing_shifts[0]["source_support"] if matched else 0
            ),
            "source_errors": (
                passing_shifts[0]["source_errors"] if matched else []
            ),
            "passing_shifts": [
                {
                    "shift": list(value["shift"]),
                    "source_support": value["source_support"],
                }
                for value in passing_shifts
            ],
            "best_source_error": min(
                (
                    error
                    for value in shift_diagnostics
                    for error in value["source_errors"]
                    if error is not None
                ),
                default=None,
            ),
        }

    @staticmethod
    def _reference_business_ids(identity: dict[str, Any]) -> tuple[int, ...]:
        """Return every high-confidence clean-reference ID in one observation."""

        raw_candidates: Sequence[dict[str, Any]]
        if identity.get("status") == "MEASURED":
            raw_candidates = (identity,)
        elif identity.get("status") == "MEASURED_CANDIDATES":
            raw_candidates = tuple(
                candidate
                for candidate in identity.get("candidates", ())
                if isinstance(candidate, dict)
            )
        else:
            return ()
        candidate_ids = {
            int(candidate["reference_business_id"])
            for candidate in raw_candidates
            if candidate.get("identity_low_confidence") is not True
            and isinstance(candidate.get("reference_business_id"), int)
            and not isinstance(candidate.get("reference_business_id"), bool)
            and int(candidate["reference_business_id"]) > 0
        }
        return tuple(sorted(candidate_ids))

    @staticmethod
    def _reference_visual_groups(identity: dict[str, Any]) -> tuple[str, ...]:
        """Return every high-confidence visual family in one observation."""

        raw_candidates: Sequence[dict[str, Any]]
        if identity.get("status") == "MEASURED":
            raw_candidates = (identity,)
        elif identity.get("status") == "MEASURED_CANDIDATES":
            raw_candidates = tuple(
                candidate
                for candidate in identity.get("candidates", ())
                if isinstance(candidate, dict)
            )
        else:
            return ()
        groups = {
            str(candidate["reference_visual_group"])
            for candidate in raw_candidates
            if candidate.get("identity_low_confidence") is not True
            and isinstance(candidate.get("reference_visual_group"), str)
            and str(candidate["reference_visual_group"]).strip()
        }
        return tuple(sorted(groups))

    @staticmethod
    def _reference_identity_candidates(
        identity: dict[str, Any],
    ) -> tuple[tuple[int, str], ...]:
        """Return every high-confidence business-ID/visual-family pair."""

        if not isinstance(identity, dict):
            return ()
        raw_candidates: Sequence[dict[str, Any]]
        if identity.get("status") == "MEASURED":
            raw_candidates = (identity,)
        elif identity.get("status") == "MEASURED_CANDIDATES":
            raw_value = identity.get("candidates")
            if (
                not isinstance(raw_value, Sequence)
                or isinstance(raw_value, (str, bytes))
                or not raw_value
                or any(not isinstance(candidate, dict) for candidate in raw_value)
            ):
                return ()
            raw_candidates = tuple(raw_value)
        else:
            return ()
        candidates: set[tuple[int, str]] = set()
        for candidate in raw_candidates:
            business_id = candidate.get("reference_business_id")
            visual_group = candidate.get("reference_visual_group")
            if (
                candidate.get("status") != "MEASURED"
                or candidate.get("identity_low_confidence") is not False
                or not isinstance(business_id, int)
                or isinstance(business_id, bool)
                or business_id < 1
                or not isinstance(visual_group, str)
                or not visual_group.strip()
            ):
                return ()
            candidates.add((business_id, visual_group))
        if len(candidates) != len(raw_candidates):
            return ()
        return tuple(sorted(candidates))

    @classmethod
    def _reference_identity_projection_is_strict(
        cls,
        identity: dict[str, Any],
    ) -> bool:
        """Require the legacy ID/group projections to describe every exact pair."""

        candidates = cls._reference_identity_candidates(identity)
        if not candidates:
            return False
        candidate_business_ids = tuple(
            sorted({business_id for business_id, _visual_group in candidates})
        )
        candidate_visual_groups = tuple(
            sorted({visual_group for _business_id, visual_group in candidates})
        )
        return bool(
            candidate_business_ids == cls._reference_business_ids(identity)
            and candidate_visual_groups == cls._reference_visual_groups(identity)
        )

    @classmethod
    def _stable_reference_identity_candidates(
        cls,
        identities: Sequence[dict[str, Any]],
    ) -> tuple[tuple[int, str], ...]:
        """Return one non-empty candidate set reproduced exactly in every frame."""

        if len(identities) != 3:
            return ()
        observed = tuple(
            cls._reference_identity_candidates(identity) for identity in identities
        )
        if not observed or any(not candidates for candidates in observed):
            return ()
        first = observed[0]
        if any(candidates != first for candidates in observed[1:]):
            return ()
        return first

    @classmethod
    def _stable_reference_visual_group(
        cls,
        identities: Sequence[dict[str, Any]],
    ) -> str | None:
        """Return one visual family present without ambiguity in every frame."""

        observed = tuple(cls._reference_visual_groups(value) for value in identities)
        if not observed or any(len(groups) != 1 for groups in observed):
            return None
        stable = {groups[0] for groups in observed}
        if len(stable) != 1:
            return None
        return stable.pop()

    def _source_visual_group_matches_expected_card(
        self,
        source_visual_group: str,
        expected_card_id: int | None,
        source_business_ids: Sequence[int],
    ) -> bool:
        """Gate visual-only restore to the expected gallery family or a new ID."""

        if expected_card_id is None:
            return True
        expected_groups = self._fixed_gallery_visual_groups_for_business_id(
            expected_card_id
        )
        if expected_groups is None:
            return expected_card_id in source_business_ids
        if not expected_groups:
            # The active RIS catalog may legitimately be newer than the fixed
            # visual gallery. Exact title evidence is then the business-ID
            # authority and the frozen source family remains the close guard.
            return True
        return expected_groups == (source_visual_group,)

    def _fixed_gallery_visual_groups_for_business_id(
        self,
        business_id: int | None,
    ) -> tuple[str, ...] | None:
        """Return the fixed-gallery families for one ID, or ``None`` if unknown."""

        if business_id is None:
            return ()
        gallery = self._card_references()
        raw_business_ids = getattr(gallery, "business_ids", None)
        raw_visual_groups = getattr(gallery, "visual_group_ids", None)
        if raw_business_ids is None or raw_visual_groups is None:
            return None
        try:
            business_ids = tuple(int(value) for value in raw_business_ids)
            visual_groups = tuple(str(value) for value in raw_visual_groups)
        except (TypeError, ValueError):
            return None
        if len(business_ids) != len(visual_groups):
            return None
        return tuple(
            sorted(
                {
                    visual_group
                    for candidate_id, visual_group in zip(
                        business_ids,
                        visual_groups,
                        strict=True,
                    )
                    if candidate_id == business_id and visual_group.strip()
                }
            )
        )

    @staticmethod
    def _unique_reference_business_id(identity: dict[str, Any]) -> int | None:
        """Return one high-confidence clean-reference ID, never a shortlist."""

        candidate_ids = MaaArenaReaderBackend._reference_business_ids(identity)
        if len(candidate_ids) != 1:
            return None
        return candidate_ids[0]

    def _card_content_generation_signatures(
        self,
        image: Any,
        rows: dict[int, tuple[tuple[int, int, int, int], ...]],
    ) -> tuple[
        dict[int, tuple[tuple[str, Any], ...]],
        dict[int, tuple[dict[str, Any], ...]],
    ]:
        gallery = self._card_references()
        signatures: dict[int, tuple[tuple[str, Any], ...]] = {}
        identities: dict[int, tuple[dict[str, Any], ...]] = {}
        for group_index, row in rows.items():
            measured = tuple(
                gallery.content_signature_with_identity(image, box)
                for box in row
            )
            signatures[group_index] = tuple(
                (visual_group, query)
                for visual_group, query, _ in measured
            )
            identities[group_index] = tuple(
                identity for _, _, identity in measured
            )
        return signatures, identities

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
        """Capture one three-frame observation shared by every visible card row."""

        groups = tuple(dict.fromkeys(int(index) for index in group_indices))
        if not groups or any(index not in (0, 1) for index in groups):
            raise ValueError("group_indices must contain card groups 0 and/or 1")
        if timeout_seconds <= 0 or maximum_timeout_seconds < timeout_seconds:
            raise ValueError(
                "maximum_timeout_seconds must be at least the positive base timeout"
            )
        if confirmation_grace_seconds < 0:
            raise ValueError("confirmation_grace_seconds cannot be negative")
        frames: list[Any] = []
        rows_by_group: dict[int, list[tuple[tuple[int, int, int, int], ...]]] = {
            index: [] for index in groups
        }
        content_by_group: dict[int, list[tuple[tuple[str, Any], ...]]] = {
            index: [] for index in groups
        }
        identity_by_group: dict[int, list[tuple[dict[str, Any], ...]]] = {
            index: [] for index in groups
        }
        monotonic_started = time.monotonic()
        deadline = monotonic_started + timeout_seconds
        absolute_deadline = monotonic_started + maximum_timeout_seconds
        next_capture_not_before = monotonic_started
        capture_times: list[float] = []
        started = time.perf_counter()
        last_error = "no complete card frame"
        last_failure_code = "skill_card_layout_unstable"
        instability: dict[tuple[int, int], dict[str, Any]] = {}

        def instability_summary() -> tuple[dict[str, Any], ...]:
            return tuple(
                {
                    **value,
                    **{
                        key: round(float(value[key]), 6)
                        for key in (
                            "maximum_mean_absolute_error",
                            "maximum_mean_color_shift",
                            "maximum_zero_mean_absolute_error",
                            "maximum_edge_mean_absolute_error",
                            "minimum_structural_correlation",
                        )
                    },
                }
                for _, value in sorted(instability.items())
            )

        while time.monotonic() < deadline:
            remaining = next_capture_not_before - time.monotonic()
            while remaining > 0:
                self._sleep(remaining)
                remaining = next_capture_not_before - time.monotonic()
            if time.monotonic() >= deadline:
                break
            capture_started = time.monotonic()
            next_capture_not_before = capture_started + interval_seconds
            image = self._capture()
            try:
                observed = {
                    index: self._validated_card_group_row(
                        image,
                        index,
                        allow_fixed_secondary=allow_fixed_secondary,
                    )
                    for index in groups
                }
            except ArenaReaderError as error:
                last_error = str(error)
                last_failure_code = "skill_card_layout_incomplete"
                frames.clear()
                capture_times.clear()
                for values in rows_by_group.values():
                    values.clear()
                for values in content_by_group.values():
                    values.clear()
                for values in identity_by_group.values():
                    values.clear()
            else:
                content_started = time.perf_counter()
                content, identities = self._card_content_generation_signatures(
                    image,
                    observed,
                )
                self._add_timing(
                    "card_content_stability_check",
                    time.perf_counter() - content_started,
                )
                geometry_shifted = bool(frames) and any(
                    self._card_rows_shifted(observed[index], rows_by_group[index][-1])
                    for index in groups
                )
                content_deltas = (
                    {
                        index: self._card_content_generation_deltas(
                            content[index],
                            content_by_group[index][-1],
                        )
                        for index in groups
                    }
                    if frames
                    else {}
                )
                content_shifted = any(
                    bool(delta["shifted"])
                    for deltas in content_deltas.values()
                    for delta in deltas
                )
                if geometry_shifted or content_shifted:
                    frames.clear()
                    capture_times.clear()
                    for values in rows_by_group.values():
                        values.clear()
                    for values in content_by_group.values():
                        values.clear()
                    for values in identity_by_group.values():
                        values.clear()
                    if content_shifted:
                        last_failure_code = "skill_card_content_generation_unstable"
                        for group_index, deltas in content_deltas.items():
                            for delta in deltas:
                                if not delta["shifted"]:
                                    continue
                                key = (group_index, int(delta["slot"]))
                                aggregate = instability.setdefault(
                                    key,
                                    {
                                        "group": group_index,
                                        "slot": int(delta["slot"]),
                                        "shifts": 0,
                                        "visual_group_changes": 0,
                                        "shape_changes": 0,
                                        "maximum_mean_absolute_error": 0.0,
                                        "maximum_mean_color_shift": 0.0,
                                        "maximum_zero_mean_absolute_error": 0.0,
                                        "maximum_edge_mean_absolute_error": 0.0,
                                        "minimum_structural_correlation": 1.0,
                                    },
                                )
                                aggregate["shifts"] += 1
                                aggregate["visual_group_changes"] += int(
                                    bool(delta["visual_group_changed"])
                                )
                                aggregate["shape_changes"] += int(
                                    bool(delta["shape_changed"])
                                )
                                motion = delta["mean_absolute_error"]
                                if motion is not None:
                                    aggregate["maximum_mean_absolute_error"] = max(
                                        float(aggregate["maximum_mean_absolute_error"]),
                                        float(motion),
                                    )
                                for source, target in (
                                    ("mean_color_shift", "maximum_mean_color_shift"),
                                    (
                                        "zero_mean_absolute_error",
                                        "maximum_zero_mean_absolute_error",
                                    ),
                                    (
                                        "edge_mean_absolute_error",
                                        "maximum_edge_mean_absolute_error",
                                    ),
                                ):
                                    value = delta[source]
                                    if value is not None:
                                        aggregate[target] = max(
                                            float(aggregate[target]),
                                            float(value),
                                        )
                                correlation = delta["structural_correlation"]
                                if correlation is not None:
                                    aggregate["minimum_structural_correlation"] = min(
                                        float(
                                            aggregate[
                                                "minimum_structural_correlation"
                                            ]
                                        ),
                                        float(correlation),
                                    )
                        last_error = (
                            "card content generation changed before three-frame stability; "
                            "instability="
                            + json.dumps(
                                instability_summary(),
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                        )
                        self._increment("card_content_generation_resets")
                    elif geometry_shifted:
                        last_failure_code = "skill_card_geometry_unstable"
                        last_error = (
                            "card geometry changed before three-frame stability"
                        )
                    extension_deadline = min(
                        absolute_deadline,
                        time.monotonic() + confirmation_grace_seconds,
                    )
                    if extension_deadline > deadline:
                        deadline = extension_deadline
                        self._increment("card_stability_deadline_extensions")
                frames.append(image)
                capture_times.append(capture_started)
                for index in groups:
                    rows_by_group[index].append(observed[index])
                    content_by_group[index].append(content[index])
                    identity_by_group[index].append(identities[index])
                required_span = interval_seconds * 2
                stable_span = capture_times[-1] - capture_times[0]
                if len(frames) == 3 and stable_span + 1e-6 >= required_span:
                    stable_frames = tuple(frames)
                    self._card_content_generation_diagnostics = instability_summary()
                    self._badge_local_results.clear()
                    for index in groups:
                        self._card_rows[index] = rows_by_group[index][0]
                        self._card_images[index] = stable_frames[0]
                        self._card_count_frames[index] = stable_frames
                        guard_store = getattr(
                            self,
                            "_card_source_guard_frames",
                            None,
                        )
                        if guard_store is None:
                            guard_store = {}
                            self._card_source_guard_frames = guard_store
                        guard_store[index] = stable_frames
                        # A new accepted source generation invalidates every
                        # cached source-side overlay signature.  Clearing here
                        # also prevents Python object-ID reuse from ever
                        # binding a later generation to stale signatures.
                        getattr(
                            self,
                            "_card_source_guard_signature_cache",
                            {},
                        ).clear()
                        self._card_identity_frames[index] = tuple(
                            identity_by_group[index]
                        )
                        getattr(
                            self,
                            "_card_source_ocr_counts",
                            {},
                        ).pop(index, None)
                        # Preserve the accepted three-frame generation itself.
                        # Detail dismissal may then prove that the original
                        # source card has returned without a fixed sleep.  All
                        # three accepted signatures remain valid baselines so
                        # harmless render jitter does not create a false stop.
                        self._card_restoration_signatures[index] = tuple(
                            content_by_group[index]
                        )
                    self._add_timing(
                        "card_layout_stable_wait",
                        time.perf_counter() - started,
                    )
                    self._add_timing(
                        "card_layout_accepted_stable_span",
                        stable_span,
                    )
                    return
                if len(frames) == 3:
                    frames.pop(0)
                    capture_times.pop(0)
                    for values in rows_by_group.values():
                        values.pop(0)
                    for values in content_by_group.values():
                        values.pop(0)
                    for values in identity_by_group.values():
                        values.pop(0)
            remaining = max(0.0, next_capture_not_before - time.monotonic())
            self._add_timing(
                "card_stability_processing_overlap",
                max(0.0, interval_seconds - remaining),
            )
        self._add_timing("card_layout_stable_wait", time.perf_counter() - started)
        self._card_content_generation_diagnostics = instability_summary()
        raise ArenaReaderError(
            last_failure_code,
            f"card groups {groups!r} did not yield three consecutive stable frames: {last_error}",
        )

    def _resolve_zero_card_reference_ids(
        self,
        group_index: int,
        slot_indices: Sequence[int],
        *,
        plan: str,
    ) -> dict[int, int]:
        identity_frames = self._card_identity_frames.get(group_index, ())
        if len(identity_frames) != 3 or any(
            len(frame) != 6 for frame in identity_frames
        ):
            raise BadgeReferenceError(
                f"group {group_index} has no stable three-frame identity input"
            )
        resolved: dict[int, int] = {}
        for slot_index in slot_indices:
            references = tuple(
                frame[slot_index] for frame in identity_frames
            )
            try:
                stable_candidates = stable_reference_business_candidates(references)
                plan_candidates = self.catalog.skill_card_candidates_for_plan(
                    stable_candidates,
                    plan=plan,
                )
                embedding_candidates: tuple[int, ...] = ()
                embedding_detail_candidates: tuple[int, ...] = ()
                embedding_error: str | None = None
                fused_candidates = plan_candidates
                if len(plan_candidates) != 1:
                    try:
                        embedding_candidates = (
                            self._stable_zero_card_embedding_family_candidates(
                                group_index,
                                slot_index,
                                plan=plan,
                            )
                        )
                        embedding_detail_candidates = getattr(
                            self,
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
                        self._increment(
                            "skill_card_zero_identity_embedding_title_fallbacks"
                        )
                        self._zero_card_identity_fusion_diagnostics[
                            (group_index, slot_index + 1)
                        ] = {
                            "group_index": group_index,
                            "slot": slot_index + 1,
                            "plan": plan,
                            "embedding_error": embedding_error,
                            "mode": "task075_unavailable_requires_one_title_click",
                        }
                    fused_candidates = tuple(
                        sorted(set(plan_candidates) & set(embedding_candidates))
                    )
                if len(fused_candidates) == 1:
                    resolved[slot_index] = fused_candidates[0]
                    continue
                detail_candidates = tuple(
                    dict.fromkeys(
                        (*plan_candidates, *embedding_detail_candidates)
                    )
                )
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
                self._zero_card_detail_candidates[
                    (group_index, slot_index + 1)
                ] = detail_candidates
                diagnostic = self._zero_card_identity_fusion_diagnostics.get(
                    (group_index, slot_index + 1)
                )
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
                raise BadgeReferenceError(
                    f"slot {slot_index + 1}: {error}"
                ) from error
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

        frames = self._card_count_frames.get(group_index, ())
        if len(frames) != 3:
            raise BadgeReferenceError(
                f"group {group_index} has no stable three-frame embedding input"
            )
        recognizer, acceptance_threshold, minimum_margin = self._card_recognizer(
            customized=False
        )
        frame_candidates: list[set[int]] = []
        detail_candidates: set[int] = set()
        frame_diagnostics: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(frames):
            row = self._validated_card_group_row(frame, group_index)
            box = row[slot_index]
            prediction = recognizer.classify(
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
            )
            if not prediction.accepted or not prediction.top_k_card_ids:
                raise BadgeReferenceError(
                    "the base card embedding did not yield an accepted top visual family for "
                    f"group {group_index}/slot {slot_index + 1}/frame {frame_index}: "
                    f"{prediction.reason}"
                )
            visual_families = tuple(
                tuple(dict.fromkeys(int(value) for value in family))
                for family in prediction.top_k_card_ids
            )
            eligible_families: list[tuple[int, ...]] = []
            for family in visual_families:
                try:
                    eligible_family = self.catalog.skill_card_candidates_for_plan(
                        family,
                        plan=plan,
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
                    "top_k_eligible_families": [
                        list(family) for family in eligible_families
                    ],
                    "top_k_scores": [
                        round(float(score), 6)
                        for score in getattr(prediction, "top_k_scores", ())
                    ],
                    "confidence": round(float(prediction.confidence), 6),
                    "margin": round(float(prediction.margin), 6),
                }
            )
        stable = tuple(sorted(set.intersection(*frame_candidates)))
        stable_detail_candidates = tuple(sorted(detail_candidates))
        detail_candidate_cache = getattr(
            self,
            "_zero_card_embedding_detail_candidates",
            None,
        )
        if detail_candidate_cache is None:
            detail_candidate_cache = {}
            self._zero_card_embedding_detail_candidates = detail_candidate_cache
        detail_candidate_cache[
            (group_index, slot_index + 1)
        ] = stable_detail_candidates
        self._zero_card_identity_fusion_diagnostics[
            (group_index, slot_index + 1)
        ] = {
            "group_index": group_index,
            "slot": slot_index + 1,
            "plan": plan,
            "stable_embedding_candidates": list(stable),
            "bounded_top_k_detail_candidates": list(stable_detail_candidates),
            "frames": frame_diagnostics,
            "mode": (
                "fixed_reference_stage_plan_task075_top_family_fusion"
                if stable
                else "task075_top_family_unavailable_requires_one_title_click"
            ),
        }
        if stable:
            self._increment("skill_card_zero_identity_embedding_fusions")
        else:
            self._increment("skill_card_zero_identity_topk_title_fallbacks")
        return stable

    def _refresh_visible_card_groups(self, requested_group: int) -> tuple[Any, Any, Any]:
        self._increment("corrective_card_frame_reads")
        groups = tuple(
            index for index in (0, 1) if len(self._card_rows.get(index, ())) == 6
        )
        if requested_group not in groups:
            groups = (requested_group,)
        self._capture_stable_card_groups(groups)
        frames = self._card_count_frames.get(requested_group, ())
        if len(frames) != 3:
            raise ArenaReaderError(
                "skill_card_layout_unstable",
                f"group {requested_group} was not cached after a stable refresh",
            )
        return frames

    def _confirm_clicked_skill_card_id(
        self,
        detail_text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = None,
        detail_image: Any | None = None,
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> int:
        """Bind a clicked detail to one business ID under bounded fallbacks.

        A complete exact title line from a full member-detail frame may rebind
        outside the visual family when the active RIS catalog is newer than
        the fixed gallery.  Existing bounded title recovery remains available
        for historical OCR cases; the positive-count fallback is never called
        for an explicit zero.
        """

        candidate_ids = tuple(dict.fromkeys(candidate_ids))
        if not candidate_ids or any(
            isinstance(card_id, bool)
            or not isinstance(card_id, int)
            or card_id < 1
            for card_id in candidate_ids
        ):
            raise ArenaCatalogError(
                "visual identity contains no valid skill-card candidate IDs"
            )

        known_ids_resolver = getattr(
            self.catalog,
            "skill_card_business_ids",
            None,
        )
        if known_ids_resolver is not None:
            known_ids = frozenset(known_ids_resolver())
            unknown_ids = tuple(
                card_id for card_id in candidate_ids if card_id not in known_ids
            )
            if unknown_ids:
                raise ArenaCatalogError(
                    "visual identity references unknown skill-card IDs "
                    f"{unknown_ids!r}"
                )

        title_text = self._authoritative_skill_card_title_text(
            source_group_index,
            detail_image,
            candidate_card_ids=candidate_ids,
            source_card_box=source_card_box,
        )
        # Production detail transactions always carry their frame. When that
        # frame exists, every identity resolver is restricted to one dynamically
        # proven title row. A missing or ambiguous title therefore fails closed
        # instead of letting an effect row such as ``眠気`` masquerade as another
        # card name.
        identity_text = (
            detail_text
            if detail_image is None
            else ""
            if title_text is None
            else title_text
        )
        exact_title_error: ArenaCatalogError | None = None
        exact_resolver = getattr(
            self.catalog,
            "confirm_clicked_skill_card_by_exact_title",
            None,
        )
        if exact_resolver is not None and title_text is not None:
            try:
                return exact_resolver(title_text)
            except ArenaCatalogError as error:
                exact_title_error = error

        candidate_error: ArenaCatalogError | None = None
        try:
            return self.catalog.confirm_clicked_skill_card_candidates(
                identity_text,
                candidate_card_ids=candidate_ids,
            )
        except ArenaCatalogError as error:
            candidate_error = error
        if title_text is not None:
            note_resolver = getattr(
                self.catalog, "confirm_clicked_skill_card_by_terminal_note_title", None,
            )
            if note_resolver is not None:
                try:
                    recovered_id = note_resolver(title_text)
                except ArenaCatalogError:
                    pass
                else:
                    self._increment("skill_card_terminal_note_title_recoveries")
                    return recovered_id
        try:
            return self.catalog.confirm_clicked_customizable_skill_card_without_badge_count(
                identity_text
            )
        except ArenaCatalogError:
            if (
                expected_customization_count is not None
                and expected_customization_count > 0
            ):
                return self.catalog.confirm_clicked_customizable_skill_card(
                    identity_text,
                    expected_count=expected_customization_count,
                )
            if exact_title_error is not None:
                raise exact_title_error
            assert candidate_error is not None
            raise candidate_error

    def _resolve_proven_skill_card_title(self, title_text: str) -> int:
        """Resolve a geometrically proven title without widening other aliases."""
        exact_resolver = getattr(self.catalog, "confirm_clicked_skill_card_by_exact_title", None)
        if exact_resolver is None:
            raise ArenaCatalogError("catalog has no exact title resolver")
        try:
            return exact_resolver(title_text)
        except ArenaCatalogError:
            note_resolver = getattr(
                self.catalog, "confirm_clicked_skill_card_by_terminal_note_title", None,
            )
            if note_resolver is None:
                raise
            return note_resolver(title_text)

    def _same_proven_skill_card_title(self, first: str, second: str, card_id: int) -> bool:
        if first == second:
            return True
        if not any(
            (note := re.fullmatch(r"([^♪+]{3,})♪(\+?)", complete)) is not None
            and "".join(note.groups()) == omitted
            for complete, omitted in ((first, second), (second, first))
        ):
            return False
        try:
            return (
                self._resolve_proven_skill_card_title(first)
                == self._resolve_proven_skill_card_title(second) == card_id
            )
        except ArenaCatalogError:
            return False

    def _normalized_skill_card_ocr_lines(self, text: str) -> tuple[str, ...]:
        normalizer = getattr(
            self.catalog,
            "normalize_skill_card_title_text",
            None,
        )
        return tuple(
            normalized
            for line in str(text or "").splitlines()
            if (
                normalized := (
                    normalizer(line)
                    if normalizer is not None
                    else re.sub(r"\s+", "", line)
                )
            )
        )

    def _stable_skill_card_source_ocr_counts(
        self,
        group_index: int,
    ) -> dict[str, int]:
        cache = getattr(self, "_card_source_ocr_counts", None)
        if cache is None:
            cache = {}
            self._card_source_ocr_counts = cache
        cached = cache.get(group_index)
        if cached is not None:
            return dict(cached)
        frames = getattr(self, "_card_count_frames", {}).get(group_index, ())
        if len(frames) != 3:
            raise ArenaCatalogError(
                f"group {group_index} has no frozen three-frame OCR source"
            )
        started = time.perf_counter()
        try:
            frame_counts = tuple(
                Counter(
                    self._with_full_frame_ocr_kind(
                        "skill_card_source",
                        self._trusted_skill_card_title_rows,
                        frame,
                    )
                )
                for frame in frames
            )
        except Exception as error:
            raise ArenaCatalogError(
                f"group {group_index} source OCR could not be frozen"
            ) from error
        stable = {
            line: max(counts[line] for counts in frame_counts)
            for line in set().union(*(counts.keys() for counts in frame_counts))
            if max(counts[line] for counts in frame_counts) > 0
        }
        cache[group_index] = stable
        self._increment("skill_card_source_title_ocr_batches")
        self._add_timing(
            "skill_card_source_title_ocr",
            time.perf_counter() - started,
        )
        return dict(stable)

    def _cached_full_frame_ocr_evidence(
        self,
        image: Any,
    ) -> _FullFrameOcrEvidence | None:
        cache = getattr(self, "_full_frame_ocr_evidence", None)
        if cache is None:
            return None
        evidence = cache.get(id(image))
        if evidence is None or evidence.image is not image:
            return None
        return evidence

    def _record_full_frame_ocr_cache_hit(
        self,
        evidence: _FullFrameOcrEvidence,
    ) -> None:
        if not hasattr(self, "_runtime_counts"):
            return
        self._increment("broad_ocr_cache_hits")
        self._increment(f"{evidence.kind}_broad_ocr_cache_hits")

    def _with_full_frame_ocr_kind(
        self,
        kind: str,
        operation: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Tag one OCR call without changing the callable's public signature."""

        had_previous = hasattr(self, "_full_frame_ocr_kind_hint")
        previous = getattr(self, "_full_frame_ocr_kind_hint", None)
        self._full_frame_ocr_kind_hint = kind
        try:
            return operation(*args, **kwargs)
        finally:
            if had_previous:
                self._full_frame_ocr_kind_hint = previous
            else:
                del self._full_frame_ocr_kind_hint

    def _full_frame_ocr_evidence_for(self, image: Any) -> _FullFrameOcrEvidence:
        """Recognize a frozen frame once while keeping fresh frames isolated."""

        cached = self._cached_full_frame_ocr_evidence(image)
        if cached is not None:
            self._record_full_frame_ocr_cache_hit(cached)
            return cached

        cache = getattr(self, "_full_frame_ocr_evidence", None)
        if cache is None:
            cache = {}
            self._full_frame_ocr_evidence = cache
        started = time.perf_counter()
        detail = self._run_recognition(
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
        hit = bool(detail and detail.hit)
        filtered_items = tuple(
            detail.filtered_results or detail.all_results or ()
        ) if hit else ()
        all_items = tuple(
            detail.all_results or detail.filtered_results or ()
        ) if hit else ()
        evidence = _FullFrameOcrEvidence(
            image,
            str(getattr(self, "_full_frame_ocr_kind_hint", "generic")),
            hit,
            filtered_items,
            all_items,
        )
        cache[id(image)] = evidence
        # Retain enough member-local entries for both accepted source groups,
        # their detail confirmations and transformed ROI views without holding
        # captures beyond the member transaction.
        while len(cache) > 64:
            oldest_key = next(iter(cache))
            if oldest_key == id(image):
                break
            cache.pop(oldest_key, None)
        if hasattr(self, "_runtime_counts"):
            self._increment("broad_ocr_backend_calls")
            self._increment(f"{evidence.kind}_broad_ocr_backend_calls")
        if hasattr(self, "_runtime_timing_seconds"):
            self._add_timing(
                "broad_ocr_backend",
                time.perf_counter() - started,
            )
        return evidence

    @staticmethod
    def _skill_card_title_row_has_neutral_ink(
        image: Any,
        box: tuple[int, int, int, int],
    ) -> bool:
        """Reject coloured effect links while retaining the neutral title ink."""

        import numpy as np

        values = np.asarray(image)
        if values.ndim != 3 or values.shape[2] < 3:
            raise ArenaReaderError(
                "skill_card_detail_title_image_invalid",
                "skill-card title evidence is not a colour image",
            )
        image_height, image_width = values.shape[:2]
        left, top, width, height = box
        right = left + width
        bottom = top + height
        if (
            width < 1
            or height < 1
            or left < 0
            or top < 0
            or right > image_width
            or bottom > image_height
        ):
            raise ArenaReaderError(
                "skill_card_detail_title_box_invalid",
                f"title OCR box is outside the capture: {box!r}",
            )
        crop = values[top:bottom, left:right, :3].astype(np.int16)
        channel_min = np.min(crop, axis=2)
        channel_max = np.max(crop, axis=2)
        ink = channel_min < 235
        ink_count = int(np.count_nonzero(ink))
        minimum_ink = max(8, int(round(width * height * 0.004)))
        if ink_count < minimum_ink:
            return False
        neutral = ink & ((channel_max - channel_min) <= 45)
        dark_neutral = neutral & (channel_max <= 205)
        return bool(
            int(np.count_nonzero(neutral)) / ink_count >= 0.72
            and int(np.count_nonzero(dark_neutral)) >= minimum_ink
        )

    def _trusted_skill_card_title_rows(
        self,
        image: Any,
        *,
        candidate_card_ids: Sequence[int] = (),
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> tuple[str, ...]:
        """Return authoritative catalog titles at the popover's dynamic title anchor."""

        candidate_ids = tuple(sorted(set(candidate_card_ids)))
        normalized_source_card_box = (
            None
            if source_card_box is None
            else tuple(int(value) for value in source_card_box)
        )
        if hasattr(self, "_runtime_counts"):
            self._increment("skill_card_trusted_title_row_requests")
        cache = getattr(self, "_trusted_skill_card_title_rows_cache", None)
        if cache is None:
            cache = {}
            self._trusted_skill_card_title_rows_cache = cache
        cache_key = (id(image), candidate_ids, normalized_source_card_box)
        cached = cache.get(cache_key)
        if cached is not None and cached.image is image:
            if hasattr(self, "_runtime_counts"):
                self._increment(
                    "skill_card_trusted_title_row_object_cache_hits"
                )
            return cached.rows

        height, width = image.shape[:2]
        if width < 320 or height < 568:
            raise ArenaReaderError(
                "skill_card_detail_title_frame_invalid",
                f"capture {width}x{height} is too small for title evidence",
            )
        exact_resolver = getattr(
            self.catalog,
            "confirm_clicked_skill_card_by_exact_title",
            None,
        )
        if exact_resolver is None:
            return ()
        anchor_resolver = getattr(
            self.catalog,
            "skill_card_title_anchor_pattern",
            None,
        )

        def title_row_is_authoritative(line: str) -> bool:
            if hasattr(self, "_runtime_counts"):
                self._increment("skill_card_exact_title_index_queries")
            try:
                self._resolve_proven_skill_card_title(line)
                return True
            except ArenaCatalogError:
                pass
            if not candidate_ids or anchor_resolver is None:
                return False
            matches = []
            for card_id in candidate_ids:
                try:
                    pattern = anchor_resolver(card_id)
                except ArenaCatalogError:
                    continue
                if re.fullmatch(pattern, line) is not None:
                    matches.append(card_id)
            return len(matches) == 1

        if hasattr(self, "_runtime_counts"):
            self._increment("skill_card_trusted_title_row_parse_calls")
        started = time.perf_counter()
        try:
            rows: list[tuple[int, int, str, tuple[int, int, int, int]]] = []
            items = self._ocr(image, r".+")
            candidates = tuple(
                (_text(item).strip(), _box(item))
                for item in items
                if _text(item).strip()
            ) + self._skill_card_title_ocr_segments(items)
            seen_candidates: set[
                tuple[str, tuple[int, int, int, int]]
            ] = set()
            for raw_text, box in candidates:
                candidate_key = (raw_text, box)
                if candidate_key in seen_candidates:
                    continue
                seen_candidates.add(candidate_key)
                normalized = self._normalized_skill_card_ocr_lines(raw_text)
                if len(normalized) != 1:
                    continue
                line = normalized[0]
                left, top, row_width, row_height = box
                in_legacy_title_band = top <= int(round(height * 0.36))
                in_lower_popover_title_band = False
                if normalized_source_card_box is not None:
                    _, card_top, _, card_height = normalized_source_card_box
                    card_bottom = card_top + card_height
                    # The adaptive panel can flip below a bottom-row card.  Do
                    # not widen the global title band: an effect row may itself
                    # be an exact catalog card name.  Bind this extra band to
                    # the clicked card and keep it inside the observed overlay
                    # envelope so the first effect row remains out of domain.
                    in_lower_popover_title_band = bool(
                        card_height > 0
                        and 0 <= card_top < card_bottom <= height
                        and card_bottom + int(round(card_height * 0.25)) <= top
                        <= card_bottom + int(round(card_height * 0.75))
                        and top + row_height <= int(round(height * 0.48))
                    )
                if not (
                    0 <= left < width
                    and 0 <= top
                    and (in_legacy_title_band or in_lower_popover_title_band)
                    and row_width >= int(round(width * 0.04))
                    and int(round(height * 0.012))
                    <= row_height
                    <= int(round(height * 0.05))
                ):
                    continue
                if not title_row_is_authoritative(line):
                    continue
                if not self._skill_card_title_row_has_neutral_ink(image, box):
                    continue
                rows.append((top, left, line, box))
            maximal_rows = tuple(
                row
                for row in rows
                if not any(
                    other is not row
                    and len(other[2]) > len(row[2])
                    and other[2].startswith(row[2])
                    and other[3][0] <= row[3][0]
                    and other[3][1] <= row[3][1]
                    and other[3][0] + other[3][2]
                    >= row[3][0] + row[3][2]
                    and other[3][1] + other[3][3]
                    >= row[3][1] + row[3][3]
                    for other in rows
                )
            )
            trusted_rows = tuple(
                line for _, _, line, _ in sorted(maximal_rows)
            )
        finally:
            if hasattr(self, "_runtime_timing_seconds"):
                self._add_timing(
                    "skill_card_trusted_title_row_resolution",
                    time.perf_counter() - started,
                )
        cache[cache_key] = _TrustedSkillCardTitleRowsEvidence(
            image,
            trusted_rows,
        )
        return trusted_rows

    @classmethod
    def _skill_card_title_ocr_segments(
        cls,
        items: Sequence[Any],
    ) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
        """Split full-screen OCR rows at cross-column horizontal gaps."""

        segments: list[tuple[str, tuple[int, int, int, int]]] = []
        for _, _, components in cls._spatial_ocr_rows(items):
            current: list[tuple[str, tuple[int, int, int, int]]] = []

            def flush() -> None:
                if not current:
                    return
                left = min(box[0] for _, box in current)
                top = min(box[1] for _, box in current)
                right = max(box[0] + box[2] for _, box in current)
                bottom = max(box[1] + box[3] for _, box in current)
                segments.append(
                    (
                        "".join(text for text, _ in current),
                        (left, top, right - left, bottom - top),
                    )
                )
                current.clear()

            for text, box in components:
                if current:
                    previous_right = max(
                        previous_box[0] + previous_box[2]
                        for _, previous_box in current
                    )
                    previous_height = max(
                        previous_box[3] for _, previous_box in current
                    )
                    gap = box[0] - previous_right
                    maximum_inline_gap = max(
                        12,
                        int(round(max(previous_height, box[3]) * 1.5)),
                    )
                    if gap > maximum_inline_gap:
                        flush()
                current.append((text, box))
            flush()
        return tuple(segments)

    @staticmethod
    def _skill_card_detail_overlay_guard_boxes(
        image: Any,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Cover every observed vertical placement of the adaptive detail panel."""

        height, width = image.shape[:2]
        if width < 320 or height < 568:
            raise ArenaReaderError(
                "skill_card_detail_overlay_guard_invalid",
                f"capture {width}x{height} is too small for overlay evidence",
            )
        horizontal_bands = ((0.00, 0.62), (0.38, 1.00))
        bands = ((0.00, 0.18), (0.14, 0.32), (0.28, 0.48))
        return tuple(
            (
                int(round(width * left_ratio)),
                int(round(height * top_ratio)),
                int(round(width * (right_ratio - left_ratio))),
                int(round(height * (bottom_ratio - top_ratio))),
            )
            for left_ratio, right_ratio in horizontal_bands
            for top_ratio, bottom_ratio in bands
        )

    def _authoritative_skill_card_title_text(
        self,
        group_index: int | None,
        detail_image: Any | None,
        *,
        candidate_card_ids: Sequence[int] = (),
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> str | None:
        """Return exactly one new, structurally proven detail-title row."""

        if group_index is None or detail_image is None:
            return None
        try:
            remaining_source = self._stable_skill_card_source_ocr_counts(
                group_index
            )
        except ArenaCatalogError:
            if hasattr(self, "_runtime_counts"):
                self._increment("skill_card_source_title_ocr_unavailable")
            return None
        try:
            detail_title_rows = self._trusted_skill_card_title_rows(
                detail_image,
                candidate_card_ids=candidate_card_ids,
                source_card_box=source_card_box,
            )
        except ArenaReaderError:
            self._increment("skill_card_detail_title_ocr_unavailable")
            return None
        new_rows: list[str] = []
        for line in detail_title_rows:
            remaining = remaining_source.get(line, 0)
            if remaining > 0:
                remaining_source[line] = remaining - 1
                continue
            new_rows.append(line)
        if len(new_rows) == 1:
            return new_rows[0]
        if new_rows:
            self._increment("skill_card_detail_title_ambiguous")
        else:
            self._increment("skill_card_detail_title_missing")
        return None

    def _skill_card_source_box(
        self,
        key: tuple[int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        """Return the frozen source-card geometry for one detail transaction."""

        if key is None:
            return None
        group_index, card_slot = key
        row = getattr(self, "_card_rows", {}).get(group_index, ())
        if not 1 <= card_slot <= len(row):
            return None
        return row[card_slot - 1]

    def open_skill_card(
        self, target: TeamTarget, stage_number: int, member_slot: int,
        group_index: int, card_slot: int, expected_customization_count: int,
    ) -> None:
        started = time.perf_counter()
        clicks_before = self._runtime_counts.get("skill_card_detail_clicks", 0)
        self._begin_detail_diagnostics(
            target=target, stage_number=stage_number, member_slot=member_slot,
            kind="skill_card", group_index=group_index, card_slot=card_slot,
            phase="open",
            source=getattr(self, "_card_images", {}).get(group_index),
        )
        try:
            self._open_skill_card_once(
                target, stage_number, member_slot, group_index, card_slot,
                expected_customization_count,
            )
        except Exception as error:
            if self._runtime_counts.get("skill_card_detail_clicks", 0) > clicks_before:
                elapsed = time.perf_counter() - started
                self._record_duration_sample("skill_card_detail_transaction", elapsed)
                self._record_duration_sample(getattr(self, "_detail_kind_hint", "necessary_skill_card_detail_transaction"), elapsed)
                self._record_duration_sample("skill_card_detail_failed_transaction", elapsed)
                self._record_duration_sample("skill_card_detail_open_failed", elapsed)
                self._increment("skill_card_detail_failed_transactions")
                self._persist_detail_failure(str(error))
            else:
                self._detail_failure_frames = None
            raise

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
        del target, stage_number, member_slot
        row = self._card_rows.get(group_index, ())
        if not 1 <= card_slot <= len(row):
            raise ArenaReaderError("skill_card_slot_missing", f"group {group_index}/slot {card_slot} is absent")
        duplicate_flags = self._card_excluded_duplicate_flags.get(group_index, ())
        if len(duplicate_flags) == 6 and duplicate_flags[card_slot - 1]:
            raise ArenaReaderError(
                "skill_card_excluded_duplicate",
                f"group {group_index}/slot {card_slot} is a non-interactive excluded duplicate",
            )
        key = (group_index, card_slot)
        inferred = self._inferred_clicked_cards.get(key)
        if inferred is not None:
            inferred_count = sum(int(value) for value in inferred.customizations.values())
            if inferred_count != expected_customization_count:
                raise ArenaReaderError(
                    "skill_card_inferred_count_changed",
                    f"group {group_index}/slot {card_slot} inferred {inferred_count} "
                    f"but the reader requested {expected_customization_count}",
                )
            self._active_inferred_clicked_card = key
            return
        candidate_ids = self._card_candidate_groups.get((group_index, card_slot))
        if candidate_ids is None:
            raise ArenaReaderError("skill_card_prediction_missing", "card icon was not classified before its click")
        card_box = row[card_slot - 1]
        primary_click_box = self._skill_card_interaction_box(card_box)
        retry_click_box = self._skill_card_retry_interaction_box(card_box)
        last_text = ""
        last_error = ""
        open_started = time.perf_counter()
        transaction_started = self._card_transaction_started.get(key, open_started)
        transaction_token = int(getattr(self, "_card_transaction_serial", 0)) + 1
        self._card_transaction_serial = transaction_token
        getattr(self, "_detail_identity_proofs", {}).pop(key, None)
        transaction_ocr_started = getattr(self, "_card_transaction_ocr_started", {}).get(key, (
            self._runtime_counts.get(
                "detail_capture_broad_ocr_backend_calls",
                0,
            ),
            self._runtime_counts.get(
                "detail_capture_broad_ocr_cache_hits",
                0,
            ),
        ))
        contact_counts = getattr(self, "_card_transaction_contact_counts", None)
        if contact_counts is None:
            contact_counts = {}
            self._card_transaction_contact_counts = contact_counts
        for attempt in range(contact_start_index, 2):
            contact_counts[key] = attempt + 1
            click_box = primary_click_box if attempt == 0 else retry_click_box
            self._increment("skill_card_detail_clicks")
            self._increment("skill_card_safe_region_clicks")
            if attempt:
                self._increment("skill_card_detail_click_retries")
                self._increment("skill_card_detail_contact_retries")
            if attempt == 0:
                self._short_press(click_box, duration_ms=80)
            else:
                self._short_press(click_box, duration_ms=120)
            last_contact_released_at = time.monotonic()
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                try:
                    detail_capture_started_at = time.monotonic()
                    detail_image = self._capture()
                    last_text = self._skill_card_detail_ocr_text(
                        detail_image,
                        phase="open",
                    )
                except ArenaReaderError:
                    last_text = ""
                try:
                    confirmed_card_id = self._confirm_clicked_skill_card_id(
                        last_text,
                        candidate_ids,
                        expected_customization_count=expected_customization_count,
                        source_group_index=group_index,
                        detail_image=detail_image,
                        source_card_box=card_box,
                    )
                    self._card_detail_texts[key] = last_text
                    self._note_confirmed_detail_name(confirmed_card_id)
                    self._card_detail_images[key] = detail_image
                    contact_times = getattr(
                        self,
                        "_card_detail_last_contact_released_at",
                        None,
                    )
                    if contact_times is None:
                        contact_times = {}
                        self._card_detail_last_contact_released_at = contact_times
                    contact_times[key] = last_contact_released_at
                    capture_times = getattr(
                        self,
                        "_card_detail_capture_started_at",
                        None,
                    )
                    if capture_times is None:
                        capture_times = {}
                        self._card_detail_capture_started_at = capture_times
                    capture_times[key] = detail_capture_started_at
                    self._card_transaction_started[key] = transaction_started
                    transaction_tokens = getattr(
                        self,
                        "_card_transaction_tokens",
                        None,
                    )
                    if transaction_tokens is None:
                        transaction_tokens = {}
                        self._card_transaction_tokens = transaction_tokens
                    transaction_tokens[key] = transaction_token
                    source_boxes = getattr(
                        self,
                        "_card_transaction_source_boxes",
                        None,
                    )
                    if source_boxes is None:
                        source_boxes = {}
                        self._card_transaction_source_boxes = source_boxes
                    source_boxes[key] = tuple(card_box)
                    interaction_boxes = getattr(
                        self,
                        "_card_transaction_interaction_boxes",
                        None,
                    )
                    if interaction_boxes is None:
                        interaction_boxes = {}
                        self._card_transaction_interaction_boxes = interaction_boxes
                    interaction_boxes[key] = tuple(click_box)
                    ocr_started = getattr(
                        self,
                        "_card_transaction_ocr_started",
                        None,
                    )
                    if ocr_started is None:
                        ocr_started = {}
                        self._card_transaction_ocr_started = ocr_started
                    ocr_started[key] = transaction_ocr_started
                    self._card_transaction_kinds.setdefault(
                        key, "necessary_skill_card_detail_transaction",
                    )
                    diagnostic = getattr(self, "_detail_failure_frames", None)
                    if diagnostic is not None and "confirmed_open" not in diagnostic:
                        for captured in reversed(diagnostic["frames"]):
                            if captured[1] is detail_image:
                                diagnostic["confirmed_open"] = captured
                                break
                    self._record_duration_sample(
                        "skill_card_detail_open_phase",
                        time.perf_counter() - open_started,
                    )
                    return
                except ArenaCatalogError as error:
                    last_error = str(error)
                    evidence = self._cached_full_frame_ocr_evidence(detail_image)
                    if evidence is not None and self._retry_transient_communication_items(evidence.filtered_items):
                        self._sleep(0.25)
                        continue
                self._sleep(0.25)
            # A failed OCR/identity read does not prove that the overlay stayed
            # closed. The same card point is a toggle while a detail is open,
            # so every retry first performs an inert backdrop dismissal and
            # proves the original source generation has returned. This reset
            # is safe when the first click was non-interactive as well.
            try:
                self._dismiss_skill_card_detail()
                self._assert_card_group_visible(
                    group_index,
                    card_slot=card_slot,
                )
            except ArenaReaderError as restore_error:
                raise ArenaReaderError(
                    "skill_card_detail_missing",
                    "clicked card detail could not be confirmed and the source "
                    "page could not be restored before another contact: "
                    f"identity_error={last_error}; "
                    f"restore_error={restore_error}",
                ) from restore_error
            self._increment("skill_card_detail_source_resets")
        # A restored source page proves only that this transaction was safely
        # reset. It cannot distinguish a genuinely non-interactive duplicate
        # from an ordinary card whose opened detail never yielded a title.
        # Preserve that uncertainty so callers cannot silently mark the slot as
        # an excluded duplicate.
        terminal_code = "skill_card_detail_ambiguous"
        raise ArenaReaderError(
            terminal_code,
            f"clicked card did not confirm visual-family IDs {candidate_ids!r} "
            "after one bounded retry: "
            f"{last_error}; card_box={card_box!r}; "
            f"primary_click_box={primary_click_box!r}; "
            f"retry_click_box={retry_click_box!r}; "
            f"raw_boxes={self._raw_card_candidate_boxes(self._card_images[group_index])!r}; "
            f"OCR={last_text[:400]!r}",
        )

    @staticmethod
    def _skill_card_interaction_box(
        card_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Return the deterministic, unobscured card-body click anchor.

        Maa treats the supplied rectangle as an action target rather than a
        promise to use its centre.  A broad rectangle can therefore land on
        the cost, type, or ``+`` overlays.  Live evidence showed that the old
        upper-art point could leave the source grid unchanged while this
        central body point opened the same cards on the bounded retry.  Make
        that proven point the first contact without widening the hit region.
        """

        x, y, width, height = card_box
        if width < 8 or height < 8:
            raise ArenaReaderError(
                "skill_card_interaction_box_invalid",
                f"card box is too small for a stable interaction region: {card_box!r}",
            )
        return (
            x + int(round(width * 0.48)),
            y + int(round(height * 0.48)),
            1,
            1,
        )

    @staticmethod
    def _skill_card_retry_interaction_box(
        card_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Return the same safe body point for one longer-contact retry."""

        x, y, width, height = card_box
        if width < 8 or height < 8:
            raise ArenaReaderError(
                "skill_card_interaction_box_invalid",
                f"card box is too small for a stable interaction region: {card_box!r}",
            )
        return (
            x + int(round(width * 0.48)),
            y + int(round(height * 0.48)),
            1,
            1,
        )

    def _open_detail_with_kind(self, kind: str, *args: Any) -> None:
        previous = getattr(self, "_detail_kind_hint", "necessary_skill_card_detail_transaction")
        self._detail_kind_hint = kind
        try:
            self.open_skill_card(*args)
        finally:
            self._detail_kind_hint = previous

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
        """Resolve one seeded-plate shortlist candidate from clicked detail."""

        candidate_ids = tuple(
            dict.fromkeys(
                candidate_ids_override
                if candidate_ids_override is not None
                else self._card_visual_family_candidates(
                    stage_number=stage_number,
                    image=image,
                    box=box,
                    slot_index=card_slot - 1,
                )
            )
        )
        if not candidate_ids:
            raise ArenaReaderError(
                "skill_card_badge_detail_identity_unknown",
                "detail recovery has no active-catalog title candidate",
            )
        key = (group_index, card_slot)
        self._card_candidate_groups[key] = candidate_ids
        overlay_opened = False
        inferred = None
        detail_failure: ArenaReaderError | None = None
        try:
            self._open_detail_with_kind(
                detail_kind,
                target,
                stage_number,
                member_slot,
                group_index,
                card_slot,
                (
                    1
                    if expected_customization_count is None
                    else expected_customization_count
                ),
            )
            overlay_opened = True
            self._card_transaction_kinds[key] = detail_kind
            inferred = self._read_resolved_card_detail(
                key,
                candidate_ids,
                expected_customization_count=expected_customization_count,
                allow_zero_without_badge_count=allow_zero_without_badge_count,
                target=target,
                stage_number=stage_number,
                member_slot=member_slot,
            )
        except ArenaReaderError as error:
            detail_failure = error
            if error.code == "skill_card_detail_noninteractive":
                raise
            raise ArenaReaderError(
                "skill_card_badge_detail_inference_ambiguous",
                f"group {group_index}/slot {card_slot}: {error}",
            ) from error
        finally:
            try:
                if overlay_opened:
                    self._dismiss_skill_card_detail()
            except Exception as error:
                if key in getattr(self, "_card_transaction_started", {}):
                    self._finish_card_transaction(key, failed=True, error=str(error))
                raise
            finally:
                if inferred is None and key in getattr(self, "_card_transaction_started", {}):
                    self._finish_card_transaction(
                        key, failed=True,
                        error=None if detail_failure is None else str(detail_failure),
                    )
        try:
            self._assert_inferred_card_group_visible_after_dismiss(
                group_index,
                card_slot=card_slot,
                expected_card_id=inferred.card_id,
            )
        except Exception as error:
            if key in getattr(self, "_card_transaction_started", {}):
                self._finish_card_transaction(key, failed=True, error=str(error))
            raise
        self._finish_card_transaction(key)
        self._inferred_clicked_cards[key] = inferred
        self._card_predictions[key] = inferred.card_id
        return inferred

    def _infer_zero_card_identity_from_detail(
        self, target: TeamTarget, stage_number: int, member_slot: int,
        group_index: int, card_slot: int, candidate_ids: Sequence[int],
    ) -> ClickedSkillCard:
        try:
            return self._infer_zero_card_identity_from_detail_once(
                target, stage_number, member_slot, group_index, card_slot, candidate_ids,
            )
        except Exception as error:
            self._finish_card_transaction((group_index, card_slot), failed=True, error=str(error))
            raise

    def _infer_zero_card_identity_from_detail_once(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        candidate_ids: Sequence[int],
    ) -> ClickedSkillCard:
        """Resolve one zero-customization card by its exact title and full effects."""

        key = (group_index, card_slot)
        self._card_candidate_groups[key] = tuple(candidate_ids)
        overlay_opened = False
        try:
            self._open_detail_with_kind(
                "zero_identity_detail_transaction",
                target,
                stage_number,
                member_slot,
                group_index,
                card_slot,
                0,
            )
            overlay_opened = True
            self._card_transaction_kinds[key] = (
                "zero_identity_detail_transaction"
            )
            inferred = self._read_resolved_card_detail(
                key,
                candidate_ids,
                expected_customization_count=0,
                allow_zero_without_badge_count=True,
                target=target,
                stage_number=stage_number,
                member_slot=member_slot,
            )
            if inferred.customizations:
                raise ArenaReaderError(
                    "skill_card_zero_identity_detail_positive",
                    f"group {group_index}/slot {card_slot} was visually zero but "
                    f"detail resolved customizations {dict(inferred.customizations)!r}",
                )
        finally:
            if overlay_opened:
                self._dismiss_skill_card_detail()
        self._assert_inferred_card_group_visible_after_dismiss(
            group_index,
            card_slot=card_slot,
            expected_card_id=inferred.card_id,
        )
        self._finish_card_transaction(key)
        self._inferred_clicked_cards[key] = inferred
        self._card_predictions[key] = inferred.card_id
        self._increment("skill_card_zero_identity_detail_resolutions")
        return inferred

    def read_skill_card_id_hints(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        customization_counts: Sequence[int],
        excluded_duplicate_flags: Sequence[bool] | None = None,
    ) -> Sequence[int]:
        row = self._card_rows.get(group_index, ())
        image = self._card_images.get(group_index)
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
        empty_flags = getattr(self, "_card_empty_flags", {}).get(
            group_index,
            (False,) * 6,
        )
        if len(empty_flags) != 6:
            raise ArenaReaderError(
                "skill_card_empty_slot_count_mismatch",
                f"group {group_index} yielded empty flags {empty_flags!r}",
            )
        plan = self.season.stages[stage_number - 1].plan
        forward_drift_ids = self._skill_card_forward_drift_for_plan(plan)
        drift_title_candidates = (
            self._skill_card_catalog_candidates_for_plan(plan)
            if forward_drift_ids
            else ()
        )
        drift_slot_indices = self._skill_card_forward_drift_slot_indices(
            forward_drift_ids
        )
        ordinary_drift = any(
            self.catalog.skill_card_source_type(value) == "produce"
            for value in forward_drift_ids
        )
        if forward_drift_ids:
            self._increment("skill_card_catalog_drift_groups")
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
            and self._inferred_clicked_cards.get((group_index, index + 1)) is None
        )
        forced_detail_slot_indices = tuple(
            slot_index
            for slot_index in zero_slot_indices
            if slot_index in drift_slot_indices
        )
        reference_slot_indices = tuple(
            slot_index
            for slot_index in zero_slot_indices
            if slot_index not in drift_slot_indices
        )
        zero_reference_ids: dict[int, int] = {
            slot_index: 0 for slot_index in forced_detail_slot_indices
        }
        if forced_detail_slot_indices:
            # A closed-set visual gallery cannot prove that a stable unique
            # match is not a newly added RIS card projected onto an older ID.
            # Fixed deck order narrows the affected positions; ordinary
            # drift still needs every eligible zero slot's exact detail.
            for slot_index in forced_detail_slot_indices:
                key = (group_index, slot_index + 1)
                self._zero_card_detail_candidates[key] = drift_title_candidates
                self._zero_card_identity_fusion_diagnostics[key] = {
                    "group_index": group_index,
                    "slot": slot_index + 1,
                    "plan": plan,
                    "missing_catalog_reference_ids": list(forward_drift_ids),
                    "detail_candidates": list(drift_title_candidates),
                    "fallback_slot_indices": list(drift_slot_indices),
                    "mode": "active_catalog_forward_drift_fixed_slots_exact_title",
                }
        if reference_slot_indices:
            try:
                zero_reference_ids.update(
                    self._resolve_zero_card_reference_ids(
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
                    self._refresh_visible_card_groups(group_index)
                    self._increment("skill_card_identity_generation_refreshes")
                    zero_reference_ids.update(
                        self._resolve_zero_card_reference_ids(
                            group_index,
                            reference_slot_indices,
                            plan=plan,
                        )
                    )
                except (ArenaReaderError, BadgeReferenceError) as retry_error:
                    raise ArenaReaderError(
                        "skill_card_reference_identity_ambiguous",
                        f"group {group_index} initial generation failed: {first_error}; "
                        f"fresh generation failed: {retry_error}",
                    ) from retry_error
                row = self._card_rows.get(group_index, ())
                image = self._card_images.get(group_index)
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
                self._card_candidate_groups.pop((group_index, index), None)
                self._card_predictions[(group_index, index)] = 0
                predictions.append(0)
                continue
            key = (group_index, index)
            inferred = self._inferred_clicked_cards.get(key)
            if inferred is not None:
                if ordinary_drift and index - 1 in drift_slot_indices:
                    self._increment("skill_card_gallery_fallback_cached_detail_reuses")
                inferred_count = sum(
                    int(value) for value in inferred.customizations.values()
                )
                if inferred_count != customization_count:
                    raise ArenaReaderError(
                        "skill_card_inferred_count_changed",
                        f"group {group_index}/slot {index} inferred {inferred_count} "
                        f"but identity read requested {customization_count}",
                    )
                self._card_candidate_groups[key] = (inferred.card_id,)
                self._card_predictions[key] = inferred.card_id
                predictions.append(inferred.card_id)
                continue
            customized = customization_count > 0
            if customized:
                # The arena custom-card embedding supplies only bounded visual-family candidates.  A
                # positive card's mandatory detail title remains authoritative.
                recognizer, acceptance_threshold, minimum_margin = self._card_recognizer(
                    customized=True
                )
                prediction = recognizer.classify(
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
                )
                candidate_ids = (
                    tuple(
                        dict.fromkeys(
                            int(value)
                            for visual_family in prediction.top_k_card_ids
                            for value in visual_family
                        )
                    )
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
                    candidate_ids = drift_title_candidates
                    if (
                        prediction.accepted and prediction.card_id.isdigit()
                        and not ordinary_drift
                    ):
                        resolved_card_id = int(prediction.card_id)
                    else:
                        def infer_positive():
                            return self._infer_badge_card_from_detail(
                                target,
                                stage_number,
                                member_slot,
                                group_index,
                                index,
                                image,
                                box,
                                customization_count,
                                "catalog_drift_positive_identity_transaction",
                                candidate_ids_override=drift_title_candidates,
                            )

                        inferred = self._run_skill_card_gallery_detail(
                            infer_positive,
                            target=target, stage_number=stage_number,
                            member_slot=member_slot, group_index=group_index,
                            card_slot=index,
                        ) if ordinary_drift else infer_positive()
                        resolved_card_id = inferred.card_id
                        candidate_ids = (resolved_card_id,)
                        self._increment(
                            "skill_card_catalog_drift_positive_detail_confirmations"
                        )
                else:
                    if (
                        not prediction.accepted
                        or not prediction.card_id.isdigit()
                        or not candidate_ids
                    ):
                        if prediction.accepted and prediction.card_id.isdigit():
                            raise ArenaReaderError(
                                "skill_card_icon_unknown",
                                f"group {group_index}/slot {index} has no "
                                "visual-family candidates",
                            )
                        raise ArenaReaderError(
                            "skill_card_icon_unknown",
                            f"group {group_index}/slot {index} was rejected: "
                            f"{prediction.reason}",
                        )
                    resolved_card_id = int(prediction.card_id)
            else:
                # Zero-card identities were resolved as one page generation
                # before the loop.  This prevents one ambiguous slot from
                # mixing fresh frames with already accepted sibling slots.
                resolved_card_id = zero_reference_ids[index - 1]
                if resolved_card_id:
                    candidate_ids = (resolved_card_id,)
                    self._increment("skill_card_exact_reference_resolutions")
                else:
                    candidate_ids = self._zero_card_detail_candidates.get(key, ())
                    if not candidate_ids:
                        raise ArenaReaderError(
                            "skill_card_zero_identity_candidates_missing",
                            f"group {group_index}/slot {index} has no bounded title candidates",
                        )
                    def infer_zero():
                        return self._infer_zero_card_identity_from_detail(
                            target,
                            stage_number,
                            member_slot,
                            group_index,
                            index,
                            candidate_ids,
                        )

                    inferred = self._run_skill_card_gallery_detail(
                        infer_zero,
                        target=target, stage_number=stage_number,
                        member_slot=member_slot, group_index=group_index,
                        card_slot=index, additional_zero_detail=True,
                    ) if ordinary_drift and index - 1 in drift_slot_indices else infer_zero()
                    resolved_card_id = inferred.card_id
                    candidate_ids = (resolved_card_id,)
                    if forward_drift_ids:
                        self._increment(
                            "skill_card_catalog_drift_zero_detail_confirmations"
                        )
            self._card_candidate_groups[key] = candidate_ids
            self._card_predictions[key] = resolved_card_id
            predictions.append(resolved_card_id)
        if group_index == 1 and tuple(predictions) == tuple(
            self._card_predictions.get((0, index)) for index in range(1, 7)
        ):
            raise ArenaReaderError(
                "skill_card_group_unchanged",
                "upper-detail swipe did not expose a distinct second six-card group",
            )
        return tuple(predictions)

    def _targeted_duplicate_marker_observations(
        self,
        frames: Sequence[Any],
        row: Sequence[tuple[int, int, int, int]],
        target_slots: Sequence[int],
        *,
        phase: str,
    ) -> tuple[tuple[tuple[bool, ...], ...], list[dict[str, Any]]]:
        import cv2

        observations = []
        diagnostics = []
        targets = set(int(index) for index in target_slots)
        for image in frames:
            height, width = image.shape[:2]
            flags = [False] * len(row)
            frame_diagnostics = []
            for index in sorted(targets):
                card_x, card_y, card_width, card_height = row[index]
                left = max(0, int(round(card_x - card_width * 0.10)))
                top = max(0, int(round(card_y - card_height * 0.10)))
                right = min(width, int(round(card_x + card_width * 1.10)))
                # ``重複`` can follow a lower card-face value rather than sit
                # in the historical upper-half position.  Keep the horizontal
                # ROI slot-local, but include the complete card plus a bounded
                # lower margin so that marker text cannot escape the OCR crop.
                bottom = min(height, int(round(card_y + card_height * 1.10)))
                crop = image[top:bottom, left:right]
                card_crop = image[
                    card_y : card_y + card_height,
                    card_x : card_x + card_width,
                    :3,
                ]
                visual_features = duplicate_marker_visual_features(card_crop)
                texts_by_variant: dict[str, list[str]] = {}
                if crop.size:
                    enlarged = cv2.resize(
                        crop,
                        None,
                        fx=3.0,
                        fy=3.0,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    if hasattr(self, "_runtime_counts"):
                        self._increment("skill_card_duplicate_marker_ocr_requests")
                        self._increment(
                            "skill_card_duplicate_marker_ocr_backend_calls"
                        )
                    started = time.perf_counter()
                    try:
                        texts = [
                            _text(item) for item in self._ocr(enlarged, r".+")
                        ]
                    finally:
                        if hasattr(self, "_runtime_timing_seconds"):
                            self._add_timing(
                                "skill_card_duplicate_marker_ocr",
                                time.perf_counter() - started,
                            )
                    texts_by_variant["3x"] = texts
                    if any(
                        marker in text
                        for text in texts
                        for marker in ("重複", "制限")
                    ):
                        flags[index] = True
                frame_diagnostic = {
                    "slot": index + 1,
                    "crop_box": [left, top, right - left, bottom - top],
                    "texts": texts_by_variant,
                    "flag": flags[index],
                    "visual_features": visual_features,
                }
                frame_diagnostics.append(frame_diagnostic)
            observations.append(tuple(flags))
            diagnostics.append(
                {
                    "phase": phase,
                    "targeted": frame_diagnostics,
                    "flags": list(flags),
                }
            )
        return tuple(observations), diagnostics

    def read_skill_card_excluded_duplicate_flags(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[bool]:
        del target, stage_number, member_slot
        row = self._card_rows.get(group_index, ())
        frames = self._card_count_frames.get(group_index, ())
        if len(row) != 6:
            raise ArenaReaderError(
                "skill_card_duplicate_marker_input_mismatch",
                f"group {group_index} has no stable six-card row",
            )
        if len(frames) != 3:
            frames = self._refresh_visible_card_groups(group_index)
            frame_rows = tuple(
                self._validated_card_group_row(image, group_index) for image in frames
            )
            if any(
                abs(actual - expected) > 3
                for observed_row in frame_rows
                for actual_box, expected_box in zip(observed_row, row, strict=True)
                for actual, expected in zip(actual_box, expected_box, strict=True)
            ):
                raise ArenaReaderError(
                    "skill_card_duplicate_marker_frame_shifted",
                    f"group {group_index} card geometry changed during duplicate-marker read",
                )
            self._card_count_frames[group_index] = frames

        repeated_visual_features = tuple(
            tuple(
                duplicate_marker_visual_features(
                    image[y : y + height, x : x + width, :3]
                )
                for image in frames
            )
            for x, y, width, height in row
        )
        empty_flags = getattr(self, "_card_empty_flags", {}).get(
            group_index,
            (False,) * 6,
        )
        candidate_slots = tuple(
            index
            for index, features in enumerate(repeated_visual_features)
            if not empty_flags[index]
            and is_duplicate_marker_visual_candidate(features)
        )
        visual_diagnostic = {
            "phase": "local_visual_candidates",
            "candidate_slots": [index + 1 for index in candidate_slots],
            "slots": [
                {
                    "slot": index + 1,
                    "candidate": index in candidate_slots,
                    "features": list(features),
                }
                for index, features in enumerate(repeated_visual_features)
            ],
        }
        targeted, diagnostics = self._targeted_duplicate_marker_observations(
            frames,
            row,
            candidate_slots,
            phase="targeted",
        )
        diagnostics.insert(0, visual_diagnostic)
        stable = stable_excluded_duplicate_card_flags(targeted)
        unresolved = tuple(index for index, flag in enumerate(stable) if flag is None)
        if unresolved:
            corrective_frames_tuple = self._refresh_visible_card_groups(group_index)
            corrective_rows = tuple(
                self._validated_card_group_row(image, group_index)
                for image in corrective_frames_tuple
            )
            if any(
                abs(actual - expected) > 3
                for observed_row in corrective_rows
                for actual_box, expected_box in zip(observed_row, row, strict=True)
                for actual, expected in zip(actual_box, expected_box, strict=True)
            ):
                raise ArenaReaderError(
                    "skill_card_duplicate_marker_frame_shifted",
                    f"group {group_index} card geometry changed during corrective duplicate-marker read",
                )
            corrective_targeted, corrective_targeted_diagnostics = (
                self._targeted_duplicate_marker_observations(
                    corrective_frames_tuple,
                    row,
                    unresolved,
                    phase="corrective_targeted",
                )
            )
            diagnostics.extend(corrective_targeted_diagnostics)
            stable = stable_excluded_duplicate_card_flags(corrective_targeted)
            self._card_count_frames[group_index] = corrective_frames_tuple
        unknown_slots = tuple(index for index, flag in enumerate(stable, start=1) if flag is None)
        self._card_duplicate_marker_diagnostics[group_index] = tuple(diagnostics)
        if unknown_slots:
            raise ArenaReaderError(
                "skill_card_duplicate_marker_unstable",
                f"group {group_index}/slots {unknown_slots} remained ambiguous after one "
                "targeted and one corrective 重複 read",
            )
        flags = tuple(bool(flag) for flag in stable)
        self._card_excluded_duplicate_flags[group_index] = flags
        return flags

    def current_skill_card_excluded_duplicate_flags(
        self,
        group_index: int,
    ) -> Sequence[bool]:
        """Return OCR flags plus any functionally proven non-interactive slot."""

        flags = self._card_excluded_duplicate_flags.get(group_index)
        if flags is None or len(flags) != 6:
            raise ArenaReaderError(
                "skill_card_duplicate_marker_input_mismatch",
                f"group {group_index} has no current six-slot duplicate state",
            )
        return flags

    def read_skill_card_customization_counts(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[int]:
        row = self._card_rows.get(group_index, ())
        prepared = self._card_images.get(group_index)
        if len(row) != 6 or prepared is None:
            raise ArenaReaderError(
                "skill_card_customization_count_input_mismatch",
                f"group {group_index} is missing its six boxes or prepared frame",
            )
        plan = self.season.stages[stage_number - 1].plan
        forward_drift_ids = self._skill_card_forward_drift_for_plan(plan)
        drift_slot_indices = self._skill_card_forward_drift_slot_indices(
            forward_drift_ids
        )
        ordinary_drift = any(
            self.catalog.skill_card_source_type(value) == "produce"
            for value in forward_drift_ids
        )
        drift_title_candidates = (
            self._skill_card_catalog_candidates_for_plan(plan)
            if forward_drift_ids
            else ()
        )
        cached_frames = self._card_count_frames.get(group_index, ())
        if len(cached_frames) == 3:
            frame_samples = list(cached_frames)
        else:
            frame_samples = [prepared]
            for _ in range(2):
                self._sleep(0.12)
                frame_samples.append(self._capture())
        frames = tuple(frame_samples)
        for attempt in range(2):
            try:
                frame_rows = tuple(
                    self._validated_card_group_row(image, group_index) for image in frames
                )
            except ArenaReaderError as error:
                if error.code != "skill_card_customization_frame_incomplete" or attempt == 1:
                    raise
                frames = self._refresh_visible_card_groups(group_index)
                continue
            baseline = row if attempt == 0 else frame_rows[0]
            shifted = any(
                abs(actual - expected) > 3
                for observed_row in frame_rows
                for actual_box, expected_box in zip(observed_row, baseline, strict=True)
                for actual, expected in zip(actual_box, expected_box, strict=True)
            )
            if shifted:
                if attempt == 1:
                    raise ArenaReaderError(
                        "skill_card_customization_frame_shifted",
                        f"group {group_index} card geometry changed after one corrective reread",
                    )
                frames = self._refresh_visible_card_groups(group_index)
                continue
            duplicate_flags = self._card_excluded_duplicate_flags.get(
                group_index,
                (False,) * 6,
            )
            empty_flags = getattr(self, "_card_empty_flags", {}).get(
                group_index,
                (False,) * 6,
            )
            local_results = self._batched_badge_local_results(group_index)
            decisions = []
            count_diagnostics = []
            for slot_index, (local_result, excluded, empty) in enumerate(
                zip(local_results, duplicate_flags, empty_flags, strict=True)
            ):
                if excluded or empty:
                    reason = "excluded_duplicate" if excluded else "empty_slot"
                    decisions.append(
                        CustomizationBadgeDecision(
                            state=CustomizationBadgeState.CONFIDENT_ZERO,
                            count=0,
                            reason=reason,
                        )
                    )
                    count_diagnostics.append(
                        {
                            "slot": slot_index + 1,
                            "local_state": CustomizationBadgeState.CONFIDENT_ZERO.value,
                            "local_reason": reason,
                            "ocr_fallback": False,
                            "resolution_path": reason,
                        }
                    )
                    continue
                assert local_result is not None
                local_decision = local_result.decision
                if (
                    local_decision.reason
                    == "all_frames_oversized_center_foreground_contradiction"
                ):
                    self._increment("skill_card_badge_oversized_art_strong_zeros")
                badge_observations = local_result.observations
                plate_diagnostics = []
                for observation in badge_observations:
                    internal = observation.get("internal_features")
                    if not isinstance(internal, dict):
                        internal = {}
                    plate_diagnostics.append(
                        {
                            key: internal.get(key)
                            for key in (
                                "peak_center",
                                "badge_center",
                                "badge_hull_points",
                                "badge_interior_area",
                                "center_seed_component_count",
                                "center_seed_ring_support",
                                "discarded_outer_component_pixels",
                                "center_component_green_fill",
                                "center_core_green_fill",
                                "center_component_extent_ratio",
                                "seeded_plate_candidate",
                                "glyph_component_count",
                                "glyph_component_box",
                                "glyph_component_area",
                                "glyph_foreground_pixel_count",
                                "glyph_raw_components",
                                "glyph_non_badge_art_candidate",
                                "glyph_descriptor_12x18_q4",
                            )
                            if key in internal
                        }
                    )
                decisions.append(local_decision)
                count_diagnostics.append(
                    {
                        "slot": slot_index + 1,
                        "local_state": local_decision.state.value,
                        "local_reason": local_decision.reason,
                        "resolution_path": (
                            "fail_closed"
                            if not self._badge_detail_inferable(local_decision)
                            else "badge_candidate_detail"
                        ),
                        "plate_diagnostics": plate_diagnostics,
                    }
                )
            decisions = tuple(decisions)
            counts = [decision.count for decision in decisions]
            for index, count in enumerate(counts, start=1):
                if count is not None:
                    continue
                inferred = self._inferred_clicked_cards.get((group_index, index))
                if inferred is None:
                    continue
                inferred_count = sum(
                    int(value) for value in inferred.customizations.values()
                )
                if inferred_count < 1:
                    fallback_claimed = (
                        inferred.conservative_cost_assumption is not None
                        or inferred.resolution_source.startswith(
                            "generic_cost_conservative_fallback"
                        )
                        or inferred.detail_evidence_mode
                        == "conservative_generic_cost_bound"
                    )
                    if fallback_claimed:
                        validate_conservative_cost_fallback(
                            inferred,
                            catalog=self.catalog,
                            expected_policy=(
                                "assume_unenhanced"
                                if target.is_own_team
                                else "assume_enhanced"
                            ),
                            expected_count=None,
                            expected_count_kind="observed",
                        )
                    elif not (
                        not inferred.customizations
                        and (
                            (
                                inferred.resolution_source == "detail_known_zero"
                                and inferred.detail_evidence_mode == "known_zero"
                            )
                            or (
                                inferred.resolution_source.endswith("_zero_confirmed")
                                and not isinstance(
                                    inferred.detail_confirmation_reads,
                                    bool,
                                )
                                and isinstance(
                                    inferred.detail_confirmation_reads,
                                    int,
                                )
                                and inferred.detail_confirmation_reads >= 2
                            )
                        )
                    ):
                        raise ArenaReaderError(
                            "skill_card_zero_cache_contract_invalid",
                            f"cached zero detail for group {group_index}/slot {index} is neither known-zero nor freshly confirmed",
                        )
                counts[index - 1] = inferred_count
                count_diagnostics[index - 1].update(
                    {
                        "fallback_state": (
                            "POSITIVE"
                            if inferred_count > 0
                            else CustomizationBadgeState.CONFIDENT_ZERO.value
                        ),
                        "fallback_reason": (
                            "cached_detail_unique_positive_count"
                            if inferred_count > 0
                            else (
                                "cached_generic_cost_conservative_zero"
                                if inferred.conservative_cost_assumption is not None
                                else "cached_detail_confirmed_zero"
                            )
                        ),
                        "fallback_mode": "clicked_detail_cache",
                        "inferred_card_id": inferred.card_id,
                        "inferred_customizations": dict(inferred.customizations),
                    }
                )
            counts = tuple(counts)
            unknown_slots = tuple(
                index for index, count in enumerate(counts, start=1) if count is None
            )
            self._badge_count_diagnostics[group_index] = tuple(count_diagnostics)
            if unknown_slots:
                inferable = tuple(
                    index
                    for index in unknown_slots
                    if self._badge_detail_inferable(decisions[index - 1])
                )
                if inferable == unknown_slots:
                    inferred_counts = list(counts)
                    for index in inferable:
                        reason = decisions[index - 1].reason
                        detail_check = self._record_badge_candidate_detail(
                            stage_number=stage_number,
                            member_slot=member_slot,
                            group_index=group_index,
                            card_slot=index,
                            phase="post_swipe_card_group",
                            reason=reason,
                        )
                        try:
                            def infer_badge():
                                return self._infer_badge_card_from_detail(
                                    target,
                                    stage_number,
                                    member_slot,
                                    group_index,
                                    index,
                                    frames[0],
                                    frame_rows[0][index - 1],
                                    None,
                                    "badge_candidate_detail_transaction",
                                    allow_zero_without_badge_count=True,
                                    candidate_ids_override=(
                                        drift_title_candidates
                                        if index - 1 in drift_slot_indices
                                        else None
                                    ),
                                )

                            inferred = self._run_skill_card_gallery_detail(
                                infer_badge,
                                target=target, stage_number=stage_number,
                                member_slot=member_slot, group_index=group_index,
                                card_slot=index,
                            ) if ordinary_drift and index - 1 in drift_slot_indices else infer_badge()
                        except ArenaReaderError as error:
                            if error.code == "skill_card_detail_noninteractive":
                                current_flags = list(
                                    self._card_excluded_duplicate_flags.get(
                                        group_index,
                                        (False,) * 6,
                                    )
                                )
                                current_flags[index - 1] = True
                                self._card_excluded_duplicate_flags[group_index] = tuple(
                                    current_flags
                                )
                                inferred_counts[index - 1] = 0
                                detail_check.update(
                                    {
                                        "resolved_state": "EXCLUDED_DUPLICATE",
                                        "functional_evidence": (
                                            "click_and_bounded_press_noninteractive"
                                        ),
                                    }
                                )
                                count_diagnostics[index - 1].update(
                                    {
                                        "fallback_state": "EXCLUDED_DUPLICATE",
                                        "fallback_reason": (
                                            "stable_card_body_noninteractive"
                                        ),
                                        "fallback_mode": (
                                            "functional_duplicate_evidence"
                                        ),
                                    }
                                )
                                self._increment(
                                    "skill_card_noninteractive_duplicate_fallbacks"
                                )
                                continue
                            raise ArenaReaderError(
                                error.code,
                                f"{error.detail}; badge_diagnostic="
                                f"{count_diagnostics[index - 1]!r}",
                            ) from error
                        self._finish_badge_candidate_detail(detail_check, inferred)
                        inferred_count = sum(
                            int(value) for value in inferred.customizations.values()
                        )
                        inferred_counts[index - 1] = inferred_count
                        count_diagnostics[index - 1].update(
                            {
                                "fallback_state": (
                                    "POSITIVE"
                                    if inferred_count > 0
                                    else CustomizationBadgeState.CONFIDENT_ZERO.value
                                ),
                                "fallback_reason": (
                                    "clicked_detail_unique_positive_count"
                                    if inferred_count > 0
                                    else "clicked_detail_unique_zero"
                                ),
                                "fallback_mode": "clicked_detail_unique_state",
                                "inferred_card_id": inferred.card_id,
                                "inferred_customizations": dict(inferred.customizations),
                            }
                        )
                    self._badge_count_diagnostics[group_index] = tuple(count_diagnostics)
                    self._card_rows[group_index] = baseline
                    self._card_images[group_index] = frames[0]
                    self._card_count_frames[group_index] = frames
                    return tuple(int(count) for count in inferred_counts)
                if attempt == 0:
                    frames = self._refresh_visible_card_groups(group_index)
                    continue
                raise ArenaReaderError(
                    "skill_card_customization_count_unknown",
                    f"group {group_index}/slots {unknown_slots} did not produce three consistent "
                    "badge reads after one corrective reread; "
                    f"reasons={tuple(decisions[index - 1].reason for index in unknown_slots)!r}; "
                    f"diagnostics={tuple(count_diagnostics[index - 1] for index in unknown_slots)!r}",
                )
            row = baseline
            self._card_rows[group_index] = row
            self._card_images[group_index] = frames[0]
            self._card_count_frames[group_index] = frames
            return tuple(int(count) for count in counts)
        raise AssertionError("customization count retry loop exhausted without a terminal result")

    def read_clicked_skill_card(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
        expected_customization_count: int,
    ) -> ClickedSkillCard:
        key = (group_index, card_slot)
        inferred = self._inferred_clicked_cards.get(key)
        if self._active_inferred_clicked_card == key and inferred is not None:
            inferred_count = sum(int(value) for value in inferred.customizations.values())
            if inferred_count != expected_customization_count:
                raise ArenaReaderError(
                    "skill_card_inferred_count_changed",
                    f"group {group_index}/slot {card_slot} inferred {inferred_count} "
                    f"but detail read requested {expected_customization_count}",
                )
            return inferred
        candidate_ids = self._card_candidate_groups.get((group_index, card_slot))
        if candidate_ids is None:
            raise ArenaReaderError(
                "skill_card_prediction_missing",
                "card icon was not classified before its click",
            )
        resolved = self._read_resolved_card_detail(
            key,
            candidate_ids,
            expected_customization_count=expected_customization_count,
            target=target,
            stage_number=stage_number,
            member_slot=member_slot,
        )
        # The embedding result is only a pre-click visual-family hint.  Once
        # the detail title/effects resolve the business ID, bind that
        # authoritative identity before the close transaction verifies the
        # restored source slot.  Otherwise a valid family mismatch (for
        # example hint 752 resolving to actual card 747) is misreported as a
        # failed close even though the original grid returned unchanged.
        self._card_predictions[key] = resolved.card_id
        return resolved

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
        conservative_cost_assumption: dict[str, Any] | None = None
        try:
            card_id = self._confirm_clicked_skill_card_id(
                text,
                candidate_ids,
                expected_customization_count=expected_customization_count,
                source_group_index=(None if key is None else key[0]),
                detail_image=(
                    None
                    if key is None
                    else getattr(self, "_card_detail_images", {}).get(key)
                ),
                source_card_box=self._skill_card_source_box(key),
            )
            generic_cost_frame_values = (
                self._measure_optional_card_face_generic_cost(key, card_id)
                if key is not None
                else None
            )
            cost_error = (
                getattr(self, "_card_face_cost_optional_errors", {}).get(key)
                if key is not None
                else None
            )
            if expected_customization_count == 0:
                customizations = self.catalog.resolve_effective_customizations(
                    card_id,
                    text,
                    expected_count=0,
                    allow_positive_completion=False,
                )
                if customizations:
                    raise ArenaCatalogError(
                        "a known-zero card resolved a positive customization state"
                    )
                resolution_source = "detail_known_zero"
                detail_evidence_mode = "known_zero"
            elif generic_cost_fallback_policy is not None and cost_error is not None:
                if generic_cost_fallback_policy not in {
                    "assume_unenhanced",
                    "assume_enhanced",
                }:
                    raise ArenaReaderError(
                        "skill_card_cost_fallback_policy_invalid",
                        f"unsupported generic-cost fallback policy {generic_cost_fallback_policy!r}",
                    )
                if not allow_generic_cost_fallback:
                    self._increment("skill_card_cost_fallback_settle_waits")
                    raise ArenaReaderError(
                        "skill_card_cost_fallback_before_settle",
                        "generic-cost fallback is deferred until settled detail evidence",
                    )
                coverage_kwargs: dict[str, Any] = {}
                if generic_cost_coverage_context is not None:
                    full_detail_text, effect_roi_text, observation_count = (
                        generic_cost_coverage_context
                    )
                    coverage_kwargs = {
                        "full_detail_text": full_detail_text,
                        "effect_roi_text": effect_roi_text,
                        "effect_roi_title_bound": True,
                        "effect_roi_observation_count": observation_count,
                    }
                pair = self.catalog.resolve_generic_cost_ambiguity(
                    card_id,
                    text,
                    observed_badge_count=expected_customization_count,
                    **coverage_kwargs,
                )
                unenhanced = dict(pair.unenhanced_customizations)
                enhanced = dict(pair.enhanced_customizations)
                customizations = (
                    enhanced
                    if generic_cost_fallback_policy == "assume_enhanced"
                    else unenhanced
                )
                hypotheses = self.catalog.generic_cost_hypotheses(card_id)
                if hypotheses is None:
                    raise ArenaCatalogError(
                        f"card {card_id} lost its generic-cost hypotheses"
                    )
                conservative_cost_assumption = {
                    "reason_code": "skill_card_cost_evidence_inconclusive",
                    "policy": generic_cost_fallback_policy,
                    "card_id": card_id,
                    "generic_customization_id": pair.generic_customization_id,
                    "observed_badge_count": expected_customization_count,
                    "cost_hypotheses": list(hypotheses),
                    "non_cost_customizations": {
                        key: value
                        for key, value in unenhanced.items()
                        if key != str(pair.generic_customization_id)
                    },
                    "candidate_customizations": {
                        "unenhanced": unenhanced,
                        "enhanced": enhanced,
                    },
                    "applied_customizations": dict(customizations),
                }
                resolution_source = "generic_cost_conservative_fallback"
                detail_evidence_mode = "conservative_generic_cost_bound"
            elif expected_customization_count is None:
                detail_evidence_mode = "unconstrained"
                customizations = None
                if generic_cost_frame_values is not None:
                    try:
                        customizations = self.catalog.resolve_effective_customizations_with_generic_cost_evidence(
                            card_id,
                            text,
                            generic_cost_frame_values=generic_cost_frame_values,
                        )
                    except ArenaCatalogError:
                        # A presence shortlist may still be an uncustomized
                        # card whose green art overlaps the semantic locus.
                        # Positive-only detail+cost evidence therefore gets
                        # first refusal, not authority to reject the later
                        # zero-inclusive catalog path.
                        customizations = None
                    else:
                        resolution_source = "detail_card_face_unique"
                        detail_evidence_mode = (
                            self.catalog.effective_customization_evidence_mode(
                                card_id,
                                text,
                                expected_count=sum(
                                    int(value) for value in customizations.values()
                                ),
                                resolved=customizations,
                            )
                        )
                if customizations is None:
                    resolver = (
                        self.catalog.resolve_effective_customizations_unconstrained
                        if allow_zero_without_badge_count
                        else self.catalog.resolve_effective_customizations_without_badge_count
                    )
                    try:
                        customizations = resolver(card_id, text)
                    except ArenaCatalogError:
                        if not allow_zero_without_badge_count or key is None:
                            raise
                        if not allow_auxiliary_badge_glyph:
                            self._increment(
                                "skill_card_detail_auxiliary_glyph_deferred"
                            )
                            raise
                        try:
                            admissible_counts = (
                                self.catalog.admissible_clicked_customization_counts(
                                    card_id,
                                    text,
                                    generic_cost_frame_values=(
                                        generic_cost_frame_values
                                    ),
                                )
                            )
                            auxiliary_count = self._auxiliary_badge_glyph_count(
                                key,
                                maximum_count=(
                                    self.catalog.maximum_customization_count(card_id)
                                ),
                                admissible_counts=admissible_counts,
                            )
                        except ArenaReaderError as error:
                            cost_error = getattr(
                                self,
                                "_card_face_cost_optional_errors",
                                {},
                            ).get(key)
                            if cost_error is None:
                                raise
                            raise ArenaReaderError(
                                error.code,
                                f"{error.detail}; card_face_cost={cost_error}",
                            ) from error
                        if auxiliary_count == 0:
                            customizations = (
                                self.catalog.resolve_effective_customizations(
                                    card_id,
                                    text,
                                    expected_count=0,
                                    allow_positive_completion=False,
                                )
                            )
                            if customizations:
                                raise ArenaCatalogError(
                                    "stable non-badge card art did not resolve a zero state"
                                )
                            resolution_source = (
                                "detail_zero_auxiliary_same_plate_non_badge_art"
                            )
                            detail_evidence_mode = (
                                "stable_same_plate_non_badge_art_zero"
                            )
                            resolution = None
                        else:
                            resolution_kwargs = {
                                "observed_badge_count": auxiliary_count,
                            }
                            if generic_cost_frame_values is not None:
                                resolution_kwargs["generic_cost_frame_values"] = (
                                    generic_cost_frame_values
                                )
                            resolution = self.catalog.resolve_clicked_customizations(
                                card_id,
                                text,
                                **resolution_kwargs,
                            )
                        if resolution is not None:
                            customizations = resolution.customizations
                            resolution_source = (
                                f"{resolution.source}_auxiliary_badge_glyph"
                            )
                            detail_evidence_mode = getattr(
                                resolution,
                                "evidence_mode",
                                "negative_dependent",
                            )
                    else:
                        resolution_source = "detail_unconstrained"
                        resolved_count = sum(
                            int(value) for value in customizations.values()
                        )
                        if resolved_count:
                            detail_evidence_mode = (
                                self.catalog.effective_customization_evidence_mode(
                                    card_id,
                                    text,
                                    expected_count=resolved_count,
                                    resolved=customizations,
                                )
                            )
            else:
                resolution_kwargs: dict[str, Any] = {
                    "observed_badge_count": expected_customization_count,
                }
                if generic_cost_frame_values is not None:
                    resolution_kwargs["generic_cost_frame_values"] = (
                        generic_cost_frame_values
                    )
                resolution = self.catalog.resolve_clicked_customizations(
                    card_id,
                    text,
                    **resolution_kwargs,
                )
                customizations = resolution.customizations
                resolution_source = resolution.source
                detail_evidence_mode = getattr(
                    resolution,
                    "evidence_mode",
                    "negative_dependent",
                )
        except ArenaCatalogError as error:
            raise ArenaReaderError("skill_card_detail_ambiguous", str(error)) from error
        return ClickedSkillCard(
            card_id,
            customizations,
            resolution_source=resolution_source,
            detail_evidence_mode=detail_evidence_mode,
            conservative_cost_assumption=conservative_cost_assumption,
        )

    def read_clicked_skill_card_unconstrained(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
    ) -> ClickedSkillCard:
        """Resolve a development truth click without trusting the badge count."""

        del target, stage_number, member_slot
        key = (group_index, card_slot)
        candidate_ids = self._card_candidate_groups.get(key)
        if candidate_ids is None:
            raise ArenaReaderError(
                "skill_card_prediction_missing",
                "card identity candidates were not established before its truth click",
            )
        return self._read_resolved_card_detail(
            key,
            candidate_ids,
            expected_customization_count=None,
            allow_zero_without_badge_count=True,
        )

    def _measure_card_face_generic_cost(
        self,
        key: tuple[int, int],
        card_id: int,
    ) -> tuple[int, int, int] | None:
        diagnostics = getattr(self, "_card_face_cost_diagnostics", None)
        if diagnostics is None:
            diagnostics = {}
            self._card_face_cost_diagnostics = diagnostics
        cached = diagnostics.get(key)
        hypotheses = self.catalog.generic_cost_hypotheses(card_id)
        if hypotheses is None:
            # The visual-family hint may resolve to a different authoritative
            # detail ID.  Do not leave evidence from the hinted generic-cost
            # card attached to a slot whose confirmed card has no such field.
            diagnostics.pop(key, None)
            return None
        hypotheses = tuple(int(value) for value in hypotheses)
        if cached is not None:
            cached_hypotheses = cached.get("hypotheses")
            cache_matches = (
                cached.get("card_id") == card_id
                and isinstance(cached_hypotheses, list)
                and tuple(cached_hypotheses) == hypotheses
            )
            if cache_matches:
                values = cached.get("frame_values")
                if (
                    isinstance(values, list)
                    and len(values) == 3
                    and all(type(value) is int for value in values)
                ):
                    return int(values[0]), int(values[1]), int(values[2])
                raise ArenaReaderError(
                    "skill_card_cost_evidence_invalid",
                    f"cached card-face cost evidence is invalid for {key!r}",
                )
            # Detail title resolution is authoritative over the pre-click
            # visual-family hint.  Re-evaluate the already frozen three source
            # frames against the confirmed identity/hypotheses instead of
            # reusing a semantically stale integer or taking another screenshot.
            self._increment("skill_card_face_cost_cache_rebinds")
        group_index, card_slot = key
        frames = self._card_count_frames.get(group_index, ())
        if len(frames) != 3:
            raise ArenaReaderError(
                "skill_card_cost_frames_missing",
                f"group {group_index} has no stable three-frame cost evidence",
            )
        started = time.perf_counter()
        try:
            frame_rows = tuple(
                self._validated_card_group_row(frame, group_index)
                for frame in frames
            )
            predictions = tuple(
                self._card_cost_references().measure(
                    frame,
                    row[card_slot - 1],
                    card_id=card_id,
                    allowed_values=hypotheses,
                )
                for frame, row in zip(frames, frame_rows, strict=True)
            )
            value = stable_generic_cost_value(predictions)
            values = (value, value, value)
        except (
            GenericCostReferenceError,
        ) as error:
            raise ArenaReaderError(
                "skill_card_cost_evidence_inconclusive",
                f"group {group_index}/slot {card_slot}/card {card_id}: {error}",
            ) from error
        finally:
            self._add_timing(
                "skill_card_face_cost_evidence",
                time.perf_counter() - started,
            )
        self._increment("skill_card_face_cost_measurements")
        diagnostics[key] = {
            "group_index": group_index,
            "slot": card_slot,
            "card_id": card_id,
            "hypotheses": list(hypotheses),
            "frame_values": list(values),
            "mode": "card_face_generic_cost_measured",
            "predictions": [
                {
                    "status": prediction.status,
                    "value": prediction.value,
                    "candidate_value": prediction.candidate_value,
                    "best_error": prediction.best_error,
                    "class_margin": prediction.class_margin,
                    "pixel_count": prediction.pixel_count,
                    "evidence_mode": prediction.evidence_mode,
                    "zero_hole_box": (
                        list(prediction.zero_hole_box)
                        if prediction.zero_hole_box is not None
                        else None
                    ),
                    "zero_alternative_error": prediction.zero_alternative_error,
                    "component_boxes": [
                        list(box) for box in prediction.component_boxes
                    ],
                }
                for prediction in predictions
            ],
        }
        return values

    def _measure_optional_card_face_generic_cost(
        self,
        key: tuple[int, int],
        card_id: int,
    ) -> tuple[int, int, int] | None:
        """Return optional cost evidence without blocking independent detail truth."""

        errors = getattr(self, "_card_face_cost_optional_errors", None)
        if errors is None:
            errors = {}
            self._card_face_cost_optional_errors = errors
        try:
            values = self._measure_card_face_generic_cost(key, card_id)
        except ArenaReaderError as error:
            if error.code != "skill_card_cost_evidence_inconclusive":
                raise
            self._increment("skill_card_face_cost_optional_inconclusive")
            errors[key] = str(error)
            return None
        errors.pop(key, None)
        return values

    def _certify_card_face_generic_cost(
        self,
        key: tuple[int, int],
        card_id: int,
        customizations: Any,
    ) -> tuple[int, int, int] | None:
        values = self._measure_card_face_generic_cost(key, card_id)
        if values is None:
            return None
        group_index, card_slot = key
        try:
            mode = self.catalog.certify_card_face_generic_cost(
                card_id,
                resolved=customizations,
                frame_values=values,
            )
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_cost_evidence_conflict",
                f"group {group_index}/slot {card_slot}/card {card_id}: {error}",
            ) from error
        diagnostic = self._card_face_cost_diagnostics[key]
        if diagnostic.get("mode") != mode:
            self._increment("skill_card_face_cost_certifications")
            diagnostic["mode"] = mode
        return values

    def _detail_identity_observation(
        self,
        key: tuple[int, int],
        candidate_ids: Sequence[int],
        resolved: ClickedSkillCard,
    ) -> _DetailIdentityObservation | None:
        """Return exact-title evidence bound to the current post-contact frame."""

        transaction_started = getattr(self, "_card_transaction_started", {}).get(key)
        transaction_token = getattr(self, "_card_transaction_tokens", {}).get(key)
        source_card_box = getattr(
            self,
            "_card_transaction_source_boxes",
            {},
        ).get(key)
        interaction_box = getattr(
            self,
            "_card_transaction_interaction_boxes",
            {},
        ).get(key)
        contact_released_at = getattr(
            self,
            "_card_detail_last_contact_released_at",
            {},
        ).get(key)
        capture_started_at = getattr(
            self,
            "_card_detail_capture_started_at",
            {},
        ).get(key)
        detail_image = getattr(self, "_card_detail_images", {}).get(key)
        if (
            not isinstance(transaction_token, int)
            or isinstance(transaction_token, bool)
            or not isinstance(transaction_started, (int, float))
            or isinstance(transaction_started, bool)
            or source_card_box is None
            or interaction_box is None
            or not isinstance(contact_released_at, (int, float))
            or isinstance(contact_released_at, bool)
            or not isinstance(capture_started_at, (int, float))
            or isinstance(capture_started_at, bool)
            or capture_started_at <= contact_released_at
            or detail_image is None
            or tuple(source_card_box) != self._skill_card_source_box(key)
        ):
            return None
        source_card_box = tuple(source_card_box)
        interaction_box = tuple(interaction_box)
        if len(source_card_box) != 4 or len(interaction_box) != 4:
            return None
        source_x, source_y, source_width, source_height = source_card_box
        click_x, click_y, click_width, click_height = interaction_box
        if not (
            source_width > 0
            and source_height > 0
            and click_width > 0
            and click_height > 0
            and source_x <= click_x
            and source_y <= click_y
            and click_x + click_width <= source_x + source_width
            and click_y + click_height <= source_y + source_height
        ):
            return None
        source_guard_frames = getattr(self, "_card_source_guard_frames", {}).get(
            key[0]
        )
        restoration_signatures = getattr(
            self,
            "_card_restoration_signatures",
            {},
        ).get(key[0])
        identity_frames = getattr(self, "_card_identity_frames", {}).get(key[0])
        if (
            source_guard_frames is None
            or restoration_signatures is None
            or identity_frames is None
        ):
            return None
        exact_resolver = getattr(
            getattr(self, "catalog", None),
            "confirm_clicked_skill_card_by_exact_title",
            None,
        )
        if exact_resolver is None:
            return None
        title_text = self._authoritative_skill_card_title_text(
            key[0],
            detail_image,
            candidate_card_ids=candidate_ids,
            source_card_box=source_card_box,
        )
        if title_text is None:
            return None
        try:
            exact_card_id = self._resolve_proven_skill_card_title(title_text)
        except ArenaCatalogError:
            return None
        if exact_card_id != resolved.card_id:
            return None
        return _DetailIdentityObservation(
            transaction_token=transaction_token,
            transaction_started=float(transaction_started),
            source_card_box=source_card_box,
            interaction_box=interaction_box,
            contact_released_at=float(contact_released_at),
            capture_started_at=float(capture_started_at),
            detail_image=detail_image,
            source_guard_frames=source_guard_frames,
            restoration_signatures=restoration_signatures,
            identity_frames=identity_frames,
            title=title_text,
            card_id=resolved.card_id,
            customizations=tuple(
                sorted(
                    (str(customization_id), int(count))
                    for customization_id, count in resolved.customizations.items()
                )
            ),
            evidence_mode=resolved.detail_evidence_mode,
        )

    def _record_detail_identity_proof(
        self,
        key: tuple[int, int],
        resolved: ClickedSkillCard,
        observations: Sequence[_DetailIdentityObservation],
    ) -> None:
        """Certify exact-title identity independently from customization evidence."""

        proofs = getattr(self, "_detail_identity_proofs", None)
        if proofs is None:
            proofs = {}
            self._detail_identity_proofs = proofs
        proofs.pop(key, None)
        selected = tuple(observations[-2:])
        if not selected:
            return
        first = selected[0]
        last = selected[-1]
        expected_customizations = tuple(
            sorted(
                (str(customization_id), int(count))
                for customization_id, count in resolved.customizations.items()
            )
        )
        shared_fields_match = all(
            observation.transaction_token == first.transaction_token
            and observation.transaction_started == first.transaction_started
            and observation.source_card_box == first.source_card_box
            and observation.interaction_box == first.interaction_box
            and observation.contact_released_at == first.contact_released_at
            and observation.source_guard_frames is first.source_guard_frames
            and observation.restoration_signatures is first.restoration_signatures
            and observation.identity_frames is first.identity_frames
            and self._same_proven_skill_card_title(
                observation.title, first.title, resolved.card_id,
            )
            and observation.card_id == first.card_id == resolved.card_id
            for observation in selected
        )
        capture_started_at = tuple(
            observation.capture_started_at for observation in selected
        )
        detail_images = tuple(observation.detail_image for observation in selected)
        captures_are_fresh = bool(
            first.contact_released_at < capture_started_at[0]
            and all(
                earlier < later
                for earlier, later in zip(
                    capture_started_at,
                    capture_started_at[1:],
                    strict=False,
                )
            )
            and len({id(image) for image in detail_images}) == len(detail_images)
        )
        if not (
            shared_fields_match
            and captures_are_fresh
            and getattr(self, "_card_transaction_started", {}).get(key)
            == last.transaction_started
            and getattr(self, "_card_transaction_tokens", {}).get(key)
            == last.transaction_token
            and getattr(self, "_card_transaction_source_boxes", {}).get(key)
            == last.source_card_box
            and getattr(self, "_card_transaction_interaction_boxes", {}).get(key)
            == last.interaction_box
            and getattr(self, "_card_detail_last_contact_released_at", {}).get(key)
            == last.contact_released_at
            and getattr(self, "_card_detail_capture_started_at", {}).get(key)
            == last.capture_started_at
            and getattr(self, "_card_detail_images", {}).get(key)
            is last.detail_image
            and getattr(self, "_card_source_guard_frames", {}).get(key[0])
            is last.source_guard_frames
            and getattr(self, "_card_restoration_signatures", {}).get(key[0])
            is last.restoration_signatures
            and getattr(self, "_card_identity_frames", {}).get(key[0])
            is last.identity_frames
        ):
            return
        proofs[key] = _DetailIdentityProof(
            transaction_token=last.transaction_token,
            transaction_started=last.transaction_started,
            source_card_box=last.source_card_box,
            interaction_box=last.interaction_box,
            contact_released_at=last.contact_released_at,
            capture_started_at=capture_started_at,
            detail_images=detail_images,
            source_guard_frames=last.source_guard_frames,
            restoration_signatures=last.restoration_signatures,
            identity_frames=last.identity_frames,
            title=last.title,
            card_id=last.card_id,
            customizations=expected_customizations,
            resolution_source=resolved.resolution_source,
            evidence_mode=resolved.detail_evidence_mode,
            detail_confirmation_reads=resolved.detail_confirmation_reads,
        )

    def _detail_identity_proof_matches(
        self,
        key: tuple[int, int],
        expected_card_id: int | None,
        source_card_box: tuple[int, int, int, int] | None,
    ) -> bool:
        """Return whether an active transaction carries exact semantic identity."""

        if (
            expected_card_id is None
            or source_card_box is None
            or key not in getattr(self, "_card_transaction_started", {})
        ):
            return False
        proof = getattr(self, "_detail_identity_proofs", {}).get(key)
        if not isinstance(proof, _DetailIdentityProof):
            return False
        captures = proof.capture_started_at
        images = proof.detail_images
        observations_are_fresh = bool(
            captures
            and len(captures) == len(images)
            and proof.contact_released_at < captures[0]
            and all(
                earlier < later
                for earlier, later in zip(captures, captures[1:], strict=False)
            )
            and len({id(image) for image in images}) == len(images)
        )
        return bool(
            proof.card_id == expected_card_id
            and proof.source_card_box == tuple(source_card_box)
            and proof.title.strip()
            and observations_are_fresh
            and getattr(self, "_card_transaction_started", {}).get(key)
            == proof.transaction_started
            and getattr(self, "_card_transaction_tokens", {}).get(key)
            == proof.transaction_token
            and getattr(self, "_card_transaction_source_boxes", {}).get(key)
            == proof.source_card_box
            and getattr(self, "_card_transaction_interaction_boxes", {}).get(key)
            == proof.interaction_box
            and getattr(self, "_card_detail_last_contact_released_at", {}).get(key)
            == proof.contact_released_at
            and getattr(self, "_card_detail_capture_started_at", {}).get(key)
            == captures[-1]
            and getattr(self, "_card_detail_images", {}).get(key)
            is images[-1]
            and getattr(self, "_card_source_guard_frames", {}).get(key[0])
            is proof.source_guard_frames
            and getattr(self, "_card_restoration_signatures", {}).get(key[0])
            is proof.restoration_signatures
            and getattr(self, "_card_identity_frames", {}).get(key[0])
            is proof.identity_frames
        )

    @staticmethod
    def _detail_identity_proof_diagnostic(
        proof: _DetailIdentityProof,
    ) -> dict[str, Any]:
        """Return image-free provenance for one accepted conflict report."""

        return {
            "transaction_token": proof.transaction_token,
            "transaction_started": proof.transaction_started,
            "source_card_box": list(proof.source_card_box),
            "interaction_box": list(proof.interaction_box),
            "contact_released_at": proof.contact_released_at,
            "capture_started_at": list(proof.capture_started_at),
            "title": proof.title,
            "card_id": proof.card_id,
            "customizations": dict(proof.customizations),
            "resolution_source": proof.resolution_source,
            "evidence_mode": proof.evidence_mode,
            "detail_confirmation_reads": proof.detail_confirmation_reads,
        }

    def _recover_failed_skill_card_effect_text(
        self,
        key: tuple[int, int],
        card_id: int,
        error: ArenaReaderError,
        *,
        expected_customization_count: int | None,
        allow_zero_without_badge_count: bool,
    ) -> tuple[str, ClickedSkillCard] | None:
        """Repair cached native atoms only after the existing views failed.

        Geometry cannot select a catalog answer. The repaired text must still
        have explicit positive evidence. Resolve its count from the detail
        before considering existing auxiliary evidence; never start extra OCR.
        The caller retains all cost, identity and fresh-frame confirmation.
        """
        if error.code not in {
            "skill_card_detail_ambiguous",
            "skill_card_badge_glyph_domain_invalid",
            "skill_card_badge_glyph_ambiguous",
        }:
            return None
        started = time.perf_counter()
        self._increment("skill_card_error_numeric_attempts")
        try:
            image = self._card_detail_images.get(key)
            evidence = self._cached_full_frame_ocr_evidence(image)
            if evidence is None:
                return None
            atoms = tuple((_text(item), _box(item)) for item in evidence.all_items)
            pattern = self.catalog.skill_card_title_anchor_pattern(card_id)
            titles = tuple(box for value, box in atoms if re.search(pattern, value))
            if len(titles) != 1:
                return None
            height, width = image.shape[:2]
            roi = self._title_anchored_effect_roi(width, height, titles[0])
            wrapped = recover_wrapped_signed_text(atoms, titles[0], roi)
            if wrapped.status == "recovered":
                self._increment("skill_card_error_wrapped_row_repairs")
                logger.debug(
                    f"skill-card error wrapped row recovery: key={key} card={card_id} "
                    f"order={wrapped.reordered_indices} reason={wrapped.reason}",
                )
                clean_text = wrapped.protected_text
            else:
                if wrapped.status == "unrecoverable":
                    return None
                layout = recover_distant_number_text(atoms, titles[0], roi)
                if layout.status != "recovered":
                    self._increment("skill_card_error_numeric_no_repair")
                    return None
                self._increment("skill_card_error_numeric_layout_repairs")
                logger.debug(
                    f"skill-card error numeric isolation: key={key} card={card_id} "
                    f"far={layout.far_indices} ambiguous={layout.ambiguous_indices}",
                )
                clean_text = layout.protected_text
            if getattr(self, "_card_face_cost_optional_errors", {}).get(key):
                return None
            generic_values = self._measure_optional_card_face_generic_cost(key, card_id)
            if wrapped.status == "recovered":
                if expected_customization_count is None and not allow_zero_without_badge_count:
                    return None
                customizations = self.catalog._resolve_failed_detail_unique_customizations(
                    card_id, clean_text,
                )
                count = sum(customizations.values())
                if expected_customization_count is not None and count != expected_customization_count:
                    return None
                if generic_values is not None:
                    self.catalog.certify_card_face_generic_cost(
                        card_id, resolved=customizations, frame_values=generic_values,
                    )
                source = "detail_card_face_unique" if generic_values is not None else "detail_unconstrained"
                mode = self.catalog.effective_customization_evidence_mode(
                    card_id, clean_text, expected_count=count, resolved=customizations,
                )
            elif expected_customization_count is None:
                if not allow_zero_without_badge_count:
                    return None
                try:
                    if generic_values is None:
                        customizations = self.catalog.resolve_effective_customizations_without_badge_count(
                            card_id, clean_text,
                        )
                        source = "detail_unconstrained"
                    else:
                        customizations = self.catalog.resolve_effective_customizations_with_generic_cost_evidence(
                            card_id, clean_text, generic_cost_frame_values=generic_values,
                        )
                        source = "detail_card_face_unique"
                    mode = self.catalog.effective_customization_evidence_mode(
                        card_id, clean_text,
                        expected_count=sum(customizations.values()),
                        resolved=customizations,
                    )
                except ArenaCatalogError:
                    maximum = self.catalog.maximum_customization_count(card_id)
                    # Reuse each resolution if auxiliary evidence is still
                    # needed for a genuinely non-unique detail combination.
                    by_count = {}
                    for observed_count in range(1, maximum + 1):
                        try:
                            candidate = self.catalog.resolve_clicked_customizations(
                                card_id, clean_text, observed_badge_count=observed_count,
                                generic_cost_frame_values=generic_values,
                            )
                        except ArenaCatalogError:
                            continue
                        if (
                            candidate.badge_count_match
                            and candidate.resolved_count == observed_count
                        ):
                            by_count[observed_count] = candidate
                    count = self._auxiliary_badge_glyph_count(
                        key, maximum_count=maximum,
                        admissible_counts=tuple(by_count), allow_ocr=False,
                    )
                    if count == 0:
                        return None
                    resolution = by_count[count]
                    customizations = resolution.customizations
                    source = f"{resolution.source}_auxiliary_badge_glyph"
                    mode = resolution.evidence_mode
            else:
                if expected_customization_count == 0:
                    return None
                resolution = self.catalog.resolve_clicked_customizations(
                    card_id, clean_text,
                    observed_badge_count=expected_customization_count,
                    generic_cost_frame_values=generic_values,
                )
                customizations = resolution.customizations
                source, mode = resolution.source, resolution.evidence_mode
            if not customizations or mode != "positive_unique":
                return None
            self._increment("skill_card_error_numeric_reparse_successes")
            return clean_text, ClickedSkillCard(
                card_id, customizations,
                resolution_source=source, detail_evidence_mode=mode,
            )
        except (ArenaCatalogError, ArenaReaderError) as recovery_error:
            self._increment("skill_card_error_numeric_reparse_failures")
            logger.debug(f"skill-card error numeric recovery unresolved: {recovery_error}")
            return None
        finally:
            self._record_duration_sample(
                "skill_card_error_numeric_recovery", time.perf_counter() - started,
            )

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
        arguments = {
            "expected_customization_count": expected_customization_count,
            "timeout_seconds": timeout_seconds,
            "allow_zero_without_badge_count": allow_zero_without_badge_count,
            "target": target, "stage_number": stage_number, "member_slot": member_slot,
        }
        try:
            return self._read_resolved_card_detail_once(key, candidate_ids, **arguments)
        except ArenaReaderError as error:
            diagnostic = getattr(self, "_detail_failure_frames", None)
            if diagnostic is not None:
                diagnostic["error"] = str(error)
            used_contacts = getattr(self, "_card_transaction_contact_counts", {}).get(key)
            if error.code != "skill_card_detail_disappeared" or used_contacts != 1:
                raise
            original_error = str(error)
        except Exception as error:
            diagnostic = getattr(self, "_detail_failure_frames", None)
            if diagnostic is not None:
                diagnostic["error"] = str(error)
            raise
        started = time.perf_counter()
        before = dict(self._runtime_counts)
        succeeded = False
        self._increment("skill_card_detail_reopen_attempts")
        try:
            # The final failure already proved two source frames before any
            # dismissal. Reuse the original retry contact, never a new budget.
            self._open_skill_card_once(
                target, stage_number, member_slot, key[0], key[1],
                1 if expected_customization_count is None else expected_customization_count,
                contact_start_index=used_contacts,
            )
            resolved = self._read_resolved_card_detail_once(key, candidate_ids, **arguments)
            succeeded = True
            self._increment("skill_card_detail_reopen_successes")
            return resolved
        except Exception as error:
            diagnostic = getattr(self, "_detail_failure_frames", None)
            if diagnostic is not None:
                diagnostic["error"] = str(error)
            raise
        finally:
            elapsed = time.perf_counter() - started
            self._record_duration_sample("skill_card_detail_reopen", elapsed)
            logger.info(json.dumps({
                "event": "arena_reader_recovery", "action": "skill_card_detail_reopen",
                "team_id": None if target is None else target.team_id,
                "stage_number": stage_number, "member_slot": member_slot,
                "group_index": key[0], "card_slot": key[1],
                "reason": original_error, "succeeded": succeeded,
                "wall_seconds": round(elapsed, 6),
                "extra_counts": {
                    name: value - before.get(name, 0)
                    for name, value in self._runtime_counts.items()
                    if value > before.get(name, 0)
                },
            }, ensure_ascii=False, sort_keys=True))

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
        """Resolve the accepted overlay text, rereading only while effects settle."""

        diagnostic = getattr(self, "_detail_failure_frames", None)
        if diagnostic is not None:
            diagnostic["position"]["phase"] = "body"

        resolution_started = time.perf_counter()
        detail_identity_observations: list[_DetailIdentityObservation] = []
        opened_image = self._card_detail_images.get(key)
        opened_capture_time = getattr(self, "_card_detail_capture_started_at", {}).get(key)
        transaction_token = getattr(self, "_card_transaction_tokens", {}).get(key)
        source_guard_frames = getattr(self, "_card_source_guard_frames", {}).get(key[0])
        recent_detail_frames: deque[tuple[float, Any]] = deque(maxlen=2)

        def finish_resolution(value: ClickedSkillCard) -> ClickedSkillCard:
            # Exact title proves only the business ID.  Reuse the already
            # accepted detail frame for every customization evidence mode;
            # never add a click, capture, or reread solely to create this proof.
            identity_observation = self._detail_identity_observation(
                key,
                candidate_ids,
                value,
            )
            if identity_observation is not None:
                detail_identity_observations.append(identity_observation)
                del detail_identity_observations[:-2]
            self._record_detail_identity_proof(
                key,
                value,
                detail_identity_observations,
            )
            self._record_duration_sample(
                "skill_card_detail_resolution_phase",
                time.perf_counter() - resolution_started,
            )
            return value

        deadline = time.monotonic() + timeout_seconds
        last_contact_released_at = getattr(
            self,
            "_card_detail_last_contact_released_at",
            {},
        ).get(key)
        frame_capture_started_at = getattr(
            self,
            "_card_detail_capture_started_at",
            {},
        ).get(key)
        frame_is_settled = (
            last_contact_released_at is not None
            and frame_capture_started_at is not None
            and frame_capture_started_at >= last_contact_released_at + 0.60
        )
        text = self._card_detail_texts.get(key, "")
        last_error = "accepted overlay text was not cached"
        confirmation_candidate: ClickedSkillCard | None = None
        confirmation_reason: str | None = None
        confirmation_started = False
        resolution_reads = 0
        resolution_conflicts = 0
        effect_roi_failures = 0
        zero_candidate: ClickedSkillCard | None = None
        zero_confirmation_reads = 0
        zero_settle_wait_recorded = False
        zero_confirmation_extension_applied = False
        positive_candidate: ClickedSkillCard | None = None
        positive_confirmation_reads = 0
        positive_confirmation_extension_applied = False
        positive_confirmation_started_at: float | None = None
        cost_fallback_candidate: ClickedSkillCard | None = None
        cost_fallback_confirmation_reads = 0
        cost_fallback_confirmation_extension_applied = False
        terminal_error: ArenaReaderError | None = None
        numeric_recovery_started = False
        same_frame_effect_roi_attempted = False
        same_frame_enhanced_effect_roi_attempted = False
        generic_cost_fallback_policy = (
            None
            if target is None
            else "assume_unenhanced"
            if target.is_own_team
            else "assume_enhanced"
        )
        while True:
            # Provenance is frame-local.  A title-anchored ROI may resolve an
            # otherwise ambiguous detail on the accepted frame itself; keep
            # that exact full/ROI pair so the structural-omission certifier can
            # consume it without a redundant OCR pass or a cross-frame join.
            same_frame_effect_roi_context: tuple[str, str, int] | None = None
            if text:
                auxiliary_badge_glyph_allowed = frame_is_settled
                try:
                    try:
                        resolved = self._resolve_clicked_card_text(
                            candidate_ids,
                            text,
                            expected_customization_count=expected_customization_count,
                            allow_zero_without_badge_count=(
                                allow_zero_without_badge_count
                            ),
                            # Full-detail semantics and the title-bound effect
                            # panel both get first refusal. The fallible card-
                            # face glyph is only a final disambiguator after the
                            # same frame has exhausted those two text views.
                            allow_auxiliary_badge_glyph=False,
                            generic_cost_fallback_policy=(generic_cost_fallback_policy),
                            allow_generic_cost_fallback=(auxiliary_badge_glyph_allowed),
                            key=key,
                        )
                    except ArenaReaderError as initial_error:
                        if initial_error.code != "skill_card_detail_ambiguous":
                            raise
                        if not frame_is_settled:
                            self._increment(
                                "skill_card_detail_render_incomplete_rereads"
                            )
                            raise ArenaReaderError(
                                "skill_card_detail_render_incomplete",
                                "the captured detail frame predates the semantic "
                                "settle boundary and has no unique full-detail result",
                            ) from initial_error
                        if same_frame_effect_roi_attempted:
                            raise
                        same_frame_effect_roi_attempted = True
                        detail_image = self._card_detail_images.get(key)
                        if detail_image is None:
                            raise
                        card_id = self._confirm_skill_card_detail_identity(
                            text,
                            candidate_ids,
                            expected_customization_count=(
                                expected_customization_count
                            ),
                            source_group_index=key[0],
                            detail_image=detail_image,
                            source_card_box=self._skill_card_source_box(key),
                        )
                        roi_text = (
                            self._skill_card_title_anchored_effect_roi_text(
                                detail_image,
                                card_id,
                            )
                        )
                        same_frame_effect_roi_context = (
                            text,
                            roi_text,
                            _effect_roi_observation_count(roi_text),
                        )
                        combined_text = self._merge_skill_card_effect_detail_views(
                            card_id,
                            (text, roi_text),
                        )
                        try:
                            resolved = self._resolve_clicked_card_text(
                                candidate_ids,
                                combined_text,
                                expected_customization_count=(
                                    expected_customization_count
                                ),
                                allow_zero_without_badge_count=(
                                    allow_zero_without_badge_count
                                ),
                                allow_auxiliary_badge_glyph=False,
                                generic_cost_fallback_policy=(
                                    generic_cost_fallback_policy
                                ),
                                allow_generic_cost_fallback=(
                                    auxiliary_badge_glyph_allowed
                                ),
                                generic_cost_coverage_context=(
                                    same_frame_effect_roi_context
                                ),
                                key=key,
                            )
                        except ArenaReaderError as combined_error:
                            if combined_error.code != "skill_card_detail_ambiguous":
                                raise
                            enhanced_resolved: ClickedSkillCard | None = None
                            unresolved_error = combined_error
                            if not same_frame_enhanced_effect_roi_attempted:
                                same_frame_enhanced_effect_roi_attempted = True
                                try:
                                    enhanced_roi_text = (
                                        self._skill_card_title_anchored_enhanced_effect_roi_text(
                                            detail_image,
                                            card_id,
                                        )
                                    )
                                except ArenaReaderError as enhancement_error:
                                    if enhancement_error.code not in {
                                        "ocr_empty",
                                        "skill_card_detail_effect_roi_empty",
                                    }:
                                        raise
                                    self._increment(
                                        "skill_card_detail_enhanced_effect_roi_empty"
                                    )
                                else:
                                    roi_observation_count = (
                                        _effect_roi_observation_count(roi_text)
                                        + int(bool(enhanced_roi_text.strip()))
                                    )
                                    roi_text = _TitleBoundEffectRoiText(
                                        self._merge_skill_card_effect_detail_views(
                                            card_id,
                                            (roi_text, enhanced_roi_text),
                                        ),
                                        roi_observation_count,
                                    )
                                    combined_text = (
                                        self._merge_skill_card_effect_detail_views(
                                            card_id,
                                            (text, roi_text),
                                        )
                                    )
                                    same_frame_effect_roi_context = (
                                        text,
                                        roi_text,
                                        _effect_roi_observation_count(roi_text),
                                    )
                                    try:
                                        enhanced_resolved = (
                                            self._resolve_clicked_card_text(
                                                candidate_ids,
                                                combined_text,
                                                expected_customization_count=(
                                                    expected_customization_count
                                                ),
                                                allow_zero_without_badge_count=(
                                                    allow_zero_without_badge_count
                                                ),
                                                allow_auxiliary_badge_glyph=False,
                                                generic_cost_fallback_policy=(
                                                    generic_cost_fallback_policy
                                                ),
                                                allow_generic_cost_fallback=(
                                                    auxiliary_badge_glyph_allowed
                                                ),
                                                generic_cost_coverage_context=(
                                                    same_frame_effect_roi_context
                                                ),
                                                key=key,
                                            )
                                        )
                                    except ArenaReaderError as enhancement_error:
                                        if (
                                            enhancement_error.code
                                            != "skill_card_detail_ambiguous"
                                        ):
                                            raise
                                        unresolved_error = enhancement_error
                                    else:
                                        self._increment(
                                            "skill_card_detail_enhanced_effect_roi_recoveries"
                                        )
                            if enhanced_resolved is not None:
                                resolved = enhanced_resolved
                            else:
                                if not auxiliary_badge_glyph_allowed:
                                    raise unresolved_error
                                try:
                                    resolved = self._resolve_clicked_card_text(
                                        candidate_ids,
                                        combined_text,
                                        expected_customization_count=(
                                            expected_customization_count
                                        ),
                                        allow_zero_without_badge_count=(
                                            allow_zero_without_badge_count
                                        ),
                                        allow_auxiliary_badge_glyph=True,
                                        generic_cost_fallback_policy=(
                                            generic_cost_fallback_policy
                                        ),
                                        allow_generic_cost_fallback=True,
                                        generic_cost_coverage_context=(
                                            same_frame_effect_roi_context
                                        ),
                                        key=key,
                                    )
                                except ArenaReaderError as exhausted_error:
                                    if (
                                        not numeric_recovery_started
                                        and exhausted_error.code in {
                                            "skill_card_detail_ambiguous",
                                            "skill_card_badge_glyph_domain_invalid",
                                            "skill_card_badge_glyph_ambiguous",
                                        }
                                    ):
                                        numeric_recovery_started = True
                                        self._increment("skill_card_error_numeric_transactions")
                                    recovery = self._recover_failed_skill_card_effect_text(
                                        key, card_id, exhausted_error,
                                        expected_customization_count=(
                                            expected_customization_count
                                        ),
                                        allow_zero_without_badge_count=(
                                            allow_zero_without_badge_count
                                        ),
                                    )
                                    if recovery is None:
                                        raise
                                    combined_text, resolved = recovery
                                    # Recovery admits positive signatures only.
                                    # Discard the failed views instead of feeding
                                    # them into later coverage/cost certificates.
                                    same_frame_effect_roi_context = None
                        text = combined_text
                        self._card_detail_texts[key] = combined_text
                        self._increment(
                            "skill_card_detail_same_frame_effect_roi_recoveries"
                        )
                    conservative_cost_assumption = getattr(
                        resolved,
                        "conservative_cost_assumption",
                        None,
                    )
                    if (
                        conservative_cost_assumption is not None
                        and same_frame_effect_roi_context is None
                    ):
                        detail_image = self._card_detail_images.get(key)
                        if detail_image is None:
                            raise ArenaReaderError(
                                "skill_card_cost_fallback_detail_image_missing",
                                "generic-cost fallback has no accepted detail image",
                            )
                        roi_text = self._skill_card_title_anchored_effect_roi_text(
                            detail_image,
                            resolved.card_id,
                        )
                        same_frame_effect_roi_context = (
                            text,
                            roi_text,
                            _effect_roi_observation_count(roi_text),
                        )
                        combined_text = self._merge_skill_card_effect_detail_views(
                            resolved.card_id,
                            (text, roi_text),
                        )
                        combined_resolved = self._resolve_clicked_card_text(
                            candidate_ids,
                            combined_text,
                            expected_customization_count=(expected_customization_count),
                            allow_zero_without_badge_count=(
                                allow_zero_without_badge_count
                            ),
                            allow_auxiliary_badge_glyph=(auxiliary_badge_glyph_allowed),
                            generic_cost_fallback_policy=(generic_cost_fallback_policy),
                            allow_generic_cost_fallback=True,
                            generic_cost_coverage_context=(
                                same_frame_effect_roi_context
                            ),
                            key=key,
                        )
                        if not self._same_clicked_card_resolution(
                            resolved,
                            combined_resolved,
                        ):
                            raise ArenaReaderError(
                                "skill_card_cost_fallback_roi_conflict",
                                "full detail and title-bound effect ROI do not leave the same generic-cost pair",
                            )
                        resolved = combined_resolved
                        text = combined_text
                        self._card_detail_texts[key] = combined_text
                        self._increment(
                            "skill_card_cost_fallback_title_bound_roi_confirmations"
                        )
                    resolved_count = sum(
                        int(value) for value in resolved.customizations.values()
                    )
                    conservative_cost_assumption = getattr(
                        resolved,
                        "conservative_cost_assumption",
                        None,
                    )
                    if (
                        cost_fallback_candidate is not None
                        and conservative_cost_assumption is None
                    ):
                        raise ArenaReaderError(
                            "skill_card_cost_fallback_frame_conflict",
                            "a settled fallback frame was followed by a different non-fallback resolution",
                        )
                    if conservative_cost_assumption is not None:
                        if any(
                            candidate is not None
                            for candidate in (
                                confirmation_candidate,
                                zero_candidate,
                                positive_candidate,
                            )
                        ):
                            raise ArenaReaderError(
                                "skill_card_cost_fallback_frame_conflict",
                                "a non-fallback confirmation candidate was followed by a generic-cost fallback resolution",
                            )
                        cost_fallback_confirmation_reads += 1
                        self._increment("skill_card_cost_fallback_confirmation_reads")
                        if self._same_clicked_card_resolution(
                            cost_fallback_candidate,
                            resolved,
                        ):
                            resolved = ClickedSkillCard(
                                resolved.card_id,
                                dict(resolved.customizations),
                                resolution_source=(
                                    "generic_cost_conservative_fallback_confirmed"
                                ),
                                detail_confirmation_reads=(
                                    cost_fallback_confirmation_reads
                                ),
                                detail_evidence_mode=(resolved.detail_evidence_mode),
                                conservative_cost_assumption=dict(
                                    conservative_cost_assumption
                                ),
                            )
                            self._increment("skill_card_cost_fallback_confirmations")
                        else:
                            if cost_fallback_candidate is not None:
                                self._increment(
                                    "skill_card_cost_fallback_confirmation_conflicts"
                                )
                                raise ArenaReaderError(
                                    "skill_card_cost_fallback_frame_conflict",
                                    "settled detail frames leave different generic-cost bounds",
                                )
                            cost_fallback_candidate = resolved
                            if not cost_fallback_confirmation_extension_applied:
                                deadline = max(
                                    deadline,
                                    time.monotonic() + 0.45,
                                )
                                cost_fallback_confirmation_extension_applied = True
                                self._increment(
                                    "skill_card_cost_fallback_confirmation_extensions"
                                )
                            raise ArenaReaderError(
                                "skill_card_cost_fallback_unconfirmed",
                                "generic-cost fallback requires one fresh settled detail frame with the same bounded pair",
                            )
                    if (
                        allow_zero_without_badge_count
                        and resolved_count == 0
                        and conservative_cost_assumption is None
                    ):
                        # A detail title becomes readable before its effect rows are
                        # guaranteed to finish animating.  The prior context-action
                        # overhead happened to hide this race; direct point dispatch
                        # exposed it as a false zero.  Zero therefore crosses both a
                        # minimum render boundary and one fresh, semantically
                        # identical detail frame.
                        if not frame_is_settled:
                            if not zero_settle_wait_recorded:
                                zero_settle_wait_recorded = True
                                self._increment("skill_card_detail_zero_settle_waits")
                            raise ArenaReaderError(
                                "skill_card_detail_zero_before_settle",
                                "an unconstrained zero detail arrived before the "
                                "effect-render settle boundary",
                            )
                        same_zero_resolution = self._same_clicked_card_resolution(
                            zero_candidate,
                            resolved,
                        )
                        zero_confirmation_reads += 1
                        self._increment("skill_card_detail_zero_confirmation_reads")
                        if same_zero_resolution:
                            resolved = ClickedSkillCard(
                                resolved.card_id,
                                dict(resolved.customizations),
                                resolution_source=(
                                    f"{resolved.resolution_source}_zero_confirmed"
                                ),
                                detail_confirmation_reads=zero_confirmation_reads,
                                detail_evidence_mode=resolved.detail_evidence_mode,
                                conservative_cost_assumption=getattr(
                                    resolved,
                                    "conservative_cost_assumption",
                                    None,
                                ),
                            )
                            self._increment("skill_card_detail_zero_confirmations")
                        else:
                            if zero_candidate is not None:
                                self._increment(
                                    "skill_card_detail_zero_confirmation_conflicts"
                                )
                            zero_candidate = resolved
                            if not zero_confirmation_extension_applied:
                                # The base deadline covers positive-effect settling.
                                # Once a post-settle zero actually exists, reserve
                                # exactly one additional OCR frame for its required
                                # confirmation instead of widening every detail read.
                                deadline = max(deadline, time.monotonic() + 0.45)
                                zero_confirmation_extension_applied = True
                                self._increment(
                                    "skill_card_detail_zero_confirmation_extensions"
                                )
                            raise ArenaReaderError(
                                "skill_card_detail_zero_unconfirmed",
                                "an unconstrained zero detail requires one fresh "
                                "semantically identical post-settle frame",
                            )
                    evidence_customization_count = expected_customization_count
                    if (
                        evidence_customization_count is None
                        and resolved.resolution_source.endswith(
                            "_auxiliary_badge_glyph"
                        )
                    ):
                        auxiliary = self._badge_glyph_count_diagnostics.get(key, {})
                        auxiliary_count = auxiliary.get("resolved_count")
                        if type(auxiliary_count) is not int:
                            raise ArenaReaderError(
                                "skill_card_badge_glyph_evidence_missing",
                                "an auxiliary badge-glyph resolution has no cached count",
                            )
                        evidence_customization_count = auxiliary_count
                    count_mismatch = (
                        expected_customization_count is not None
                        and resolved_count != expected_customization_count
                    )
                    cost_fallback_confirmed = (
                        getattr(
                            resolved,
                            "conservative_cost_assumption",
                            None,
                        )
                        is not None
                        and resolved.resolution_source
                        == "generic_cost_conservative_fallback_confirmed"
                    )
                    if (
                        count_mismatch
                        and not cost_fallback_confirmed
                        and (
                            resolved.resolution_source
                            not in {
                                "detail_unique",
                                "detail_card_face_unique",
                            }
                            or resolved_count < 1
                        )
                    ):
                        raise ArenaReaderError(
                            "skill_card_detail_count_conflict",
                            "a badge-count mismatch was not resolved by one unique positive detail",
                        )

                    try:
                        generic_cost_frame_values = (
                            self._certify_card_face_generic_cost(
                                key,
                                resolved.card_id,
                                resolved.customizations,
                            )
                        )
                    except ArenaReaderError as error:
                        if error.code != "skill_card_cost_evidence_inconclusive":
                            raise
                        # Detail semantics and the bounded auxiliary badge count
                        # may already prove a unique group.  An inconclusive
                        # optional cost carrier must not preempt those independent
                        # paths; any later branch that truly needs cost evidence
                        # still fails inside the external-evidence certifier.
                        self._increment("skill_card_face_cost_optional_inconclusive")
                        generic_cost_frame_values = None
                        generic_cost_evidence_error = str(error)
                    else:
                        generic_cost_evidence_error = None

                    if resolved.detail_evidence_mode == "negative_dependent":
                        if same_frame_effect_roi_context is not None:
                            full_detail_text, roi_text, _ = (
                                same_frame_effect_roi_context
                            )
                            effect_roi_title_bound = True
                            combined_text = self._merge_skill_card_effect_detail_views(
                                resolved.card_id,
                                (full_detail_text, roi_text),
                            )
                        else:
                            detail_image = self._card_detail_images.get(key)
                            if detail_image is None:
                                raise ArenaReaderError(
                                    "skill_card_detail_positive_evidence_missing",
                                    "the accepted detail page has no cached image for an "
                                    "independent effect-ROI OCR; "
                                    f"key={key!r}; card_id={resolved.card_id}; "
                                    f"customizations={dict(resolved.customizations)!r}; "
                                    f"expected_badge_count={evidence_customization_count!r}",
                                )
                            full_detail_text = text
                            # A fixed screen column is not a valid coverage
                            # boundary: live detail panels shift horizontally,
                            # and a fresh frame cannot repair a cropped panel.
                            # Every structural-omission pass is therefore bound
                            # to the uniquely resolved title, including the
                            # first pass. A bounded fresh-frame retry remains
                            # only for genuinely incomplete effect animation.
                            effect_roi_title_bound = True
                            roi_text = self._skill_card_title_anchored_effect_roi_text(
                                detail_image,
                                resolved.card_id,
                            )
                            combined_text = self._merge_skill_card_effect_detail_views(
                                resolved.card_id,
                                (text, roi_text),
                            )
                            try:
                                resolved = self._resolve_clicked_card_text(
                                    candidate_ids,
                                    combined_text,
                                    expected_customization_count=(
                                        expected_customization_count
                                    ),
                                    allow_zero_without_badge_count=(
                                        allow_zero_without_badge_count
                                    ),
                                    allow_auxiliary_badge_glyph=False,
                                    generic_cost_fallback_policy=(
                                        generic_cost_fallback_policy
                                    ),
                                    allow_generic_cost_fallback=(
                                        auxiliary_badge_glyph_allowed
                                    ),
                                    generic_cost_coverage_context=(
                                        full_detail_text,
                                        roi_text,
                                        _effect_roi_observation_count(roi_text),
                                    ),
                                    key=key,
                                )
                            except ArenaReaderError as combined_error:
                                if (
                                    combined_error.code
                                    != "skill_card_detail_ambiguous"
                                    or not auxiliary_badge_glyph_allowed
                                ):
                                    raise
                                resolved = self._resolve_clicked_card_text(
                                    candidate_ids,
                                    combined_text,
                                    expected_customization_count=(
                                        expected_customization_count
                                    ),
                                    allow_zero_without_badge_count=(
                                        allow_zero_without_badge_count
                                    ),
                                    allow_auxiliary_badge_glyph=True,
                                    generic_cost_fallback_policy=(
                                        generic_cost_fallback_policy
                                    ),
                                    allow_generic_cost_fallback=True,
                                    generic_cost_coverage_context=(
                                        full_detail_text,
                                        roi_text,
                                        _effect_roi_observation_count(roi_text),
                                    ),
                                    key=key,
                                )
                        resolved_count = sum(
                            int(value) for value in resolved.customizations.values()
                        )
                        evidence_customization_count = expected_customization_count
                        if (
                            evidence_customization_count is None
                            and resolved.resolution_source.endswith(
                                "_auxiliary_badge_glyph"
                            )
                        ):
                            auxiliary = self._badge_glyph_count_diagnostics.get(key, {})
                            auxiliary_count = auxiliary.get("resolved_count")
                            if type(auxiliary_count) is not int:
                                raise ArenaReaderError(
                                    "skill_card_badge_glyph_evidence_missing",
                                    "an auxiliary badge-glyph reread has no cached count",
                                )
                            evidence_customization_count = auxiliary_count
                        count_mismatch = (
                            expected_customization_count is not None
                            and resolved_count != expected_customization_count
                        )
                        if resolved.detail_evidence_mode == "negative_dependent":
                            certifier = getattr(
                                self.catalog,
                                "certify_structural_omission_evidence",
                                None,
                            )
                            certified_mode = None
                            certification_error = "catalog has no structural omission certifier"
                            certification_count = evidence_customization_count
                            if certification_count is None:
                                # The coverage certifier is itself the independent
                                # evidence boundary: it accepts every selected level
                                # only when two covered detail views prove an explicit
                                # positive signature, a UI-defined typed-cost/limit
                                # omission, or a stable three-frame card-face cost.
                                # Let it certify the resolved group's total without
                                # making the fallible face-badge digit a prerequisite.
                                certification_count = resolved_count
                            if certifier is not None and certification_count is not None:
                                try:
                                    certified_mode = certifier(
                                        resolved.card_id,
                                        full_detail_text=full_detail_text,
                                        effect_roi_text=roi_text,
                                        expected_count=certification_count,
                                        resolved=resolved.customizations,
                                        generic_cost_frame_values=(
                                            generic_cost_frame_values
                                        ),
                                        effect_roi_title_bound=(
                                            effect_roi_title_bound
                                        ),
                                    )
                                except ArenaCatalogError as error:
                                    certification_error = str(error)
                            if generic_cost_evidence_error is not None:
                                certification_error = (
                                    f"{certification_error}; card_face_cost="
                                    f"{generic_cost_evidence_error}"
                                )
                            if certified_mode not in {
                                "structural_omission_unique",
                                "card_face_generic_cost_unique",
                                "external_evidence_unique",
                            }:
                                raise ArenaReaderError(
                                    "skill_card_detail_positive_evidence_missing",
                                    "full-screen and enlarged effect-ROI OCR both lacked "
                                    "positive evidence and did not certify a UI-defined "
                                    "structural omission or independent card-face cost; "
                                    f"key={key!r}; card_id={resolved.card_id}; "
                                    f"customizations={dict(resolved.customizations)!r}; "
                                    f"expected_badge_count={evidence_customization_count!r}; "
                                    f"certification={certification_error}; "
                                    f"effect_roi={roi_text[:240]!r}",
                                )
                            resolved = ClickedSkillCard(
                                resolved.card_id,
                                dict(resolved.customizations),
                                resolution_source=resolved.resolution_source,
                                detail_confirmation_reads=resolved.detail_confirmation_reads,
                                detail_evidence_mode=certified_mode,
                                conservative_cost_assumption=getattr(
                                    resolved,
                                    "conservative_cost_assumption",
                                    None,
                                ),
                            )
                            self._increment(
                                "skill_card_detail_external_evidence_certifications"
                            )
                            if certified_mode in {
                                "structural_omission_unique",
                                "external_evidence_unique",
                            }:
                                self._increment(
                                    "skill_card_detail_structural_omission_certifications"
                                )
                        text = combined_text
                        self._card_detail_texts[key] = combined_text
                    if (
                        allow_zero_without_badge_count
                        and expected_customization_count is None
                        and resolved_count > 0
                        and resolved.detail_evidence_mode == "positive_unique"
                    ):
                        # This branch supplies the badge count as well as the
                        # concrete customization IDs.  A single transitional OCR
                        # frame must not become self-authenticating positive
                        # evidence, so require the same business resolution once
                        # more from a fresh frame.  Structural-omission and
                        # card-face-cost modes already carry independent channels.
                        # This adds no navigation or broad retry.
                        same_positive_resolution = self._same_clicked_card_resolution(
                            positive_candidate,
                            resolved,
                        )
                        positive_confirmation_reads += 1
                        self._increment(
                            "skill_card_detail_positive_confirmation_reads"
                        )
                        if same_positive_resolution:
                            resolved = ClickedSkillCard(
                                resolved.card_id,
                                dict(resolved.customizations),
                                resolution_source=(
                                    f"{resolved.resolution_source}_positive_confirmed"
                                ),
                                detail_confirmation_reads=(positive_confirmation_reads),
                                detail_evidence_mode=resolved.detail_evidence_mode,
                                conservative_cost_assumption=getattr(
                                    resolved,
                                    "conservative_cost_assumption",
                                    None,
                                ),
                            )
                            self._increment(
                                "skill_card_detail_positive_confirmations"
                            )
                            if positive_confirmation_started_at is not None:
                                self._record_duration_sample(
                                    "skill_card_detail_positive_confirmation_phase",
                                    time.perf_counter()
                                    - positive_confirmation_started_at,
                                )
                        else:
                            if positive_candidate is not None:
                                self._increment(
                                    "skill_card_detail_positive_confirmation_conflicts"
                                )
                            positive_candidate = resolved
                            if positive_confirmation_started_at is None:
                                positive_confirmation_started_at = (
                                    time.perf_counter()
                                )
                            if not positive_confirmation_extension_applied:
                                deadline = max(
                                    deadline,
                                    time.monotonic() + 0.25,
                                )
                                positive_confirmation_extension_applied = True
                                self._increment(
                                    "skill_card_detail_positive_confirmation_extensions"
                                )
                            raise ArenaReaderError(
                                "skill_card_detail_positive_unconfirmed",
                                "an unconstrained positive detail requires one fresh "
                                "semantically identical frame",
                            )
                    if count_mismatch and cost_fallback_confirmed:
                        assert expected_customization_count is not None
                        self._record_detail_count_override(
                            key,
                            expected_customization_count,
                            resolved,
                        )
                    reason = (
                        "badge_count_override"
                        if count_mismatch and not cost_fallback_confirmed
                        else None
                    )
                    if reason is None:
                        self._record_cost_customization_fallback(
                            key,
                            resolved,
                            target=target,
                            stage_number=stage_number,
                            member_slot=member_slot,
                        )
                        self._maybe_register_badge_glyph_exemplar(
                            key,
                            resolved,
                            target=target,
                            stage_number=stage_number,
                            member_slot=member_slot,
                        )
                        return finish_resolution(resolved)

                    resolution_reads += 1
                    if not confirmation_started:
                        confirmation_started = True
                        self._increment("skill_card_detail_semantic_confirmations")
                    if self._same_clicked_card_resolution(
                        confirmation_candidate,
                        resolved,
                    ):
                        confirmed = ClickedSkillCard(
                            resolved.card_id,
                            dict(resolved.customizations),
                            resolution_source=f"{resolved.resolution_source}_confirmed",
                            detail_confirmation_reads=resolution_reads,
                            detail_evidence_mode=resolved.detail_evidence_mode,
                            conservative_cost_assumption=getattr(
                                resolved,
                                "conservative_cost_assumption",
                                None,
                            ),
                        )
                        self._record_detail_semantic_confirmation(
                            key,
                            confirmed,
                            reason=reason,
                            resolution_conflicts=resolution_conflicts,
                        )
                        if count_mismatch:
                            assert expected_customization_count is not None
                            self._record_detail_count_override(
                                key,
                                expected_customization_count,
                                confirmed,
                            )
                        self._record_cost_customization_fallback(
                            key,
                            confirmed,
                            target=target,
                            stage_number=stage_number,
                            member_slot=member_slot,
                        )
                        self._maybe_register_badge_glyph_exemplar(
                            key,
                            confirmed,
                            target=target,
                            stage_number=stage_number,
                            member_slot=member_slot,
                        )
                        return finish_resolution(confirmed)
                    if confirmation_candidate is not None:
                        resolution_conflicts += 1
                        self._increment("skill_card_detail_semantic_conflicts")
                    confirmation_candidate = resolved
                    confirmation_reason = reason
                    last_error = self._detail_confirmation_wait_message(
                        reason,
                        expected_customization_count=expected_customization_count,
                        resolved_count=resolved_count,
                    )
                except ArenaReaderError as error:
                    confirmation_candidate = None
                    confirmation_reason = None
                    last_error = str(error)
                    image_evidence = self._cached_full_frame_ocr_evidence(self._card_detail_images.get(key))
                    if image_evidence is not None and self._retry_transient_communication_items(image_evidence.filtered_items):
                        text = ""
                        zero_candidate = positive_candidate = cost_fallback_candidate = None
                        continue
                    if error.code in {
                        "skill_card_cost_fallback_changed",
                        "skill_card_cost_fallback_contract_invalid",
                        "skill_card_cost_fallback_frame_conflict",
                        "skill_card_cost_fallback_location_missing",
                        "skill_card_cost_fallback_roi_conflict",
                        "skill_card_detail_effect_view_conflict",
                    }:
                        terminal_error = error
                        break
                    if error.code == "skill_card_detail_positive_evidence_missing":
                        if effect_roi_failures < 1:
                            # The title can become readable before the effect rows
                            # finish animating. Retry the complete evidence pair on
                            # one fresh frame; never join full/ROI text across frames.
                            effect_roi_failures += 1
                            self._increment("skill_card_detail_effect_roi_retries")
                        else:
                            terminal_error = error
                            break
            if terminal_error is not None:
                break
            if time.monotonic() >= deadline:
                break
            # The title is intentionally not a readiness signal: live details
            # can hold a title-only plateau for longer than the former 0.75 s
            # transaction budget.  Poll semantic evidence at a short cadence
            # and return immediately on the first complete result; the 1.50 s
            # deadline is a failure bound, not a per-card fixed wait.
            self._sleep(0.08)
            self._increment("skill_card_detail_ocr_rereads")
            try:
                previous_capture_started_at = frame_capture_started_at
                detail_capture_started_at = time.monotonic()
                if previous_capture_started_at is not None:
                    self._record_duration_sample(
                        "skill_card_detail_capture_start_span",
                        detail_capture_started_at - previous_capture_started_at,
                    )
                detail_image = self._capture()
                text = self._skill_card_detail_ocr_text(
                    detail_image,
                    phase="reread",
                )
            except ArenaReaderError as error:
                text = ""
                last_error = str(error)
            else:
                self._card_detail_texts[key] = text
                self._card_detail_images[key] = detail_image
                recent_detail_frames.append((detail_capture_started_at, detail_image))
                capture_times = getattr(
                    self,
                    "_card_detail_capture_started_at",
                    None,
                )
                if capture_times is None:
                    capture_times = {}
                    self._card_detail_capture_started_at = capture_times
                capture_times[key] = detail_capture_started_at
                frame_capture_started_at = detail_capture_started_at
                frame_is_settled = (
                    last_contact_released_at is not None
                    and frame_capture_started_at
                    >= last_contact_released_at + 0.60
                )
                same_frame_effect_roi_attempted = False
                same_frame_enhanced_effect_roi_attempted = False
        self._record_duration_sample(
            "skill_card_detail_failed_resolution_phase",
            time.perf_counter() - resolution_started,
        )
        self._increment("skill_card_detail_failed_resolutions")
        if terminal_error is not None:
            # The detail transaction owns this terminal decision. Propagate it
            # through both direct and badge paths without renewing the budget.
            terminal_error.retry_whole_read = False
            raise terminal_error
        if self._skill_card_detail_disappeared(
            key, opened_image, opened_capture_time, transaction_token,
            source_guard_frames, tuple(recent_detail_frames),
        ):
            self._increment("skill_card_detail_disappeared_before_confirmation")
            raise ArenaReaderError(
                "skill_card_detail_disappeared",
                f"group {key[0]}/slot {key[1]}: the accepted detail disappeared "
                "before confirmation; the last two captured frames match the "
                "frozen member page; no extra observation was taken; "
                f"original_error={last_error}",
            )
        raise ArenaReaderError(
            "skill_card_detail_ambiguous",
            "card detail did not become uniquely resolvable within the bounded settle window: "
            f"{last_error}; pending_confirmation={confirmation_reason!r}; "
            f"OCR={text[:400]!r}",
        )

    def _skill_card_detail_disappeared(
        self,
        key: tuple[int, int],
        opened_image: Any,
        opened_capture_time: float | None,
        transaction_token: int | None,
        source_frames: tuple[Any, ...] | None,
        recent_frames: tuple[tuple[float, Any], ...],
    ) -> bool:
        """Classify a final body failure before dismissal, using cached frames only.

        A successful close after an ordinary semantic failure proves nothing
        about premature disappearance. Require the opened, title-confirmed
        transaction to differ from its source and two later frames to already
        match that source, under the existing overlay-restoration pixel gate.
        """
        if (
            transaction_token is None
            or getattr(self, "_card_transaction_tokens", {}).get(key) != transaction_token
            or key not in getattr(self, "_card_transaction_started", {})
            or source_frames is None or len(source_frames) != 3
            or getattr(self, "_card_source_guard_frames", {}).get(key[0]) is not source_frames
            or opened_image is None or opened_capture_time is None
            or len(recent_frames) != 2
            or not opened_capture_time < recent_frames[0][0] < recent_frames[1][0]
            or recent_frames[0][1] is recent_frames[1][1]
            or any(image is opened_image for _, image in recent_frames)
            or self._card_detail_images.get(key) is not recent_frames[-1][1]
        ):
            return False
        started = time.perf_counter()
        try:
            shape = getattr(opened_image, "shape", None)
            if shape is None or any(getattr(image, "shape", None) != shape for image in (
                *source_frames, *(image for _, image in recent_frames),
            )):
                return False
            for _, image in recent_frames:
                evidence = self._cached_full_frame_ocr_evidence(image)
                if evidence is None or not evidence.hit:
                    return False
                if (
                    len(self._matching_ocr_items(evidence.filtered_items, r"^体力$")) != 1
                    or len(self._matching_ocr_items(evidence.filtered_items, r"^総合力$")) != 1
                ):
                    return False
            boxes = self._skill_card_detail_overlay_guard_boxes(opened_image)
            source_signatures = tuple(
                p_item_content_generation_signatures(image, boxes)
                for image in source_frames
            )

            def source_matches(image: Any) -> int:
                signature = p_item_content_generation_signatures(image, boxes)
                errors = tuple(
                    measure_p_item_content_generation_from_signatures(source, signature)
                    for source in source_signatures
                )
                return sum(
                    bool(values) and max(values) <= CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR
                    for values in errors
                )

            return source_matches(opened_image) == 0 and all(
                source_matches(image) >= 2 for _, image in recent_frames
            )
        except (ArenaReaderError, PItemReferenceError, TypeError, ValueError):
            # Missing or uncertain proof retains the original semantic error.
            return False
        finally:
            self._record_duration_sample(
                "skill_card_detail_disappearance_check", time.perf_counter() - started,
            )

    @staticmethod
    def _same_clicked_card_resolution(
        left: ClickedSkillCard | None,
        right: ClickedSkillCard,
    ) -> bool:
        return (
            left is not None
            and left.card_id == right.card_id
            and dict(left.customizations) == dict(right.customizations)
            and getattr(left, "conservative_cost_assumption", None)
            == getattr(right, "conservative_cost_assumption", None)
        )

    @staticmethod
    def _detail_confirmation_wait_message(
        reason: str,
        *,
        expected_customization_count: int | None,
        resolved_count: int,
    ) -> str:
        if reason == "badge_count_override":
            return (
                f"card-face count {expected_customization_count} conflicts with "
                f"unique detail count {resolved_count}; awaiting one fresh confirmation"
            )
        return (
            f"badge count {resolved_count} admits multiple legal customization groups; "
            "awaiting the same final-effect resolution from one fresh frame"
        )

    def _record_detail_semantic_confirmation(
        self,
        key: tuple[int, int],
        resolved: ClickedSkillCard,
        *,
        reason: str,
        resolution_conflicts: int,
    ) -> None:
        self._detail_semantic_confirmations[key] = {
            "slot": key[1],
            "card_id": resolved.card_id,
            "resolved_detail_count": sum(
                int(value) for value in resolved.customizations.values()
            ),
            "customizations": dict(resolved.customizations),
            "reason": reason,
            "detail_confirmation_reads": resolved.detail_confirmation_reads,
            "resolution_conflicts": resolution_conflicts,
            "evidence_mode": resolved.detail_evidence_mode,
        }

    def _skill_card_effect_roi_text(self, image: Any) -> str:
        """OCR an enlarged detail-effect panel without taking another screenshot."""

        import cv2

        height, width = image.shape[:2]
        left = int(round(width * 0.055))
        top = int(round(height * 0.025))
        right = int(round(width * 0.61))
        bottom = int(round(height * 0.32))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            raise ArenaReaderError(
                "skill_card_detail_effect_roi_invalid",
                f"detail effect ROI is empty for frame {width}x{height}",
            )
        enlarged = cv2.resize(
            crop,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        started = time.perf_counter()
        try:
            return self._with_full_frame_ocr_kind(
                "derived_roi",
                self._full_ocr_text,
                enlarged,
            )
        finally:
            self._increment("skill_card_detail_effect_roi_reads")
            self._add_timing(
                "skill_card_detail_effect_roi_ocr",
                time.perf_counter() - started,
            )

    def _confirm_skill_card_detail_identity(
        self,
        text: str,
        candidate_ids: Sequence[int],
        *,
        expected_customization_count: int | None,
        source_group_index: int | None = None,
        detail_image: Any | None = None,
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> int:
        """Resolve the already-open detail title under the existing bounded gates."""

        try:
            return self._confirm_clicked_skill_card_id(
                text,
                candidate_ids,
                expected_customization_count=expected_customization_count,
                source_group_index=source_group_index,
                detail_image=detail_image,
                source_card_box=source_card_box,
            )
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_detail_ambiguous",
                str(error),
            ) from error

    def _skill_card_detail_ocr_text(self, image: Any, *, phase: str) -> str:
        """Time full detail OCR by transaction phase without changing its result."""

        if phase not in {"open", "reread"}:
            raise ValueError("skill-card detail OCR phase must be open or reread")
        if not hasattr(self, "_runtime_timing_seconds"):
            self._runtime_timing_seconds = {}
        started = time.perf_counter()
        try:
            return self._with_full_frame_ocr_kind(
                "detail_capture",
                self._full_ocr_text,
                image,
            )
        finally:
            self._increment(f"skill_card_detail_{phase}_ocr_reads")
            self._add_timing(
                f"skill_card_detail_{phase}_ocr",
                time.perf_counter() - started,
            )

    @staticmethod
    def _title_anchored_effect_roi(
        frame_width: int,
        frame_height: int,
        title_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Cover one effect panel from its title instead of a fixed screen column."""

        title_x, title_y, title_width, title_height = title_box
        # OCR boxes are intentionally tight and may cover only the first title
        # glyphs.  The effect rows can extend far to either side of that box.
        # Keep the retry bounded to the title's panel, but guarantee at least
        # two thirds of the normalized screen width so a shifted panel cannot
        # truncate its invariant action rows.
        minimum_width = int(round(frame_width * 0.66))
        left = max(0, title_x - int(round(frame_width * 0.12)))
        top = max(0, title_y - int(round(frame_height * 0.015)))
        right = min(
            frame_width,
            title_x + title_width + int(round(frame_width * 0.55)),
        )
        if right - left < minimum_width:
            left = max(0, min(left, frame_width - minimum_width))
            right = min(frame_width, max(right, left + minimum_width))
            if right - left < minimum_width:
                left = max(0, right - minimum_width)
        bottom = min(
            frame_height,
            max(
                title_y + title_height + int(round(frame_height * 0.25)),
                top + int(round(frame_height * 0.295)),
            ),
        )
        if right - left < 32 or bottom - top < 32:
            raise ArenaReaderError(
                "skill_card_detail_effect_roi_invalid",
                "title-anchored detail effect ROI is too small: "
                f"frame={frame_width}x{frame_height}, title={title_box!r}",
            )
        return left, top, right - left, bottom - top

    @staticmethod
    def _background_parameter_column_left(
        frame_width: int,
        frame_height: int,
        title_box: tuple[int, int, int, int],
        items: Sequence[Any],
    ) -> int | None:
        """Locate a proven member-stat column exposed beside a left popover.

        The member detail's parameter labels live in one fixed right-hand
        column.  A left-clamped skill-card popover can leave the lower labels
        visible beside its effect text, so a deliberately wide title-bound ROI
        may otherwise join a parameter value to the last effect token.  Treat
        the column as background only when two distinct, exact parameter
        labels each have their own aligned numeric value below them.  One
        label or an unpaired number is intentionally insufficient because the
        same words and digits may legitimately occur inside card effects.
        """

        title_x, title_y, title_width, _ = title_box
        title_right = title_x + title_width
        _, roi_top, _, roi_height = MaaArenaReaderBackend._title_anchored_effect_roi(
            frame_width,
            frame_height,
            title_box,
        )
        roi_bottom = roi_top + roi_height
        column_min_x = int(round(frame_width * 0.55))
        column_max_x = int(round(frame_width * 0.72))
        labels: list[tuple[str, tuple[int, int, int, int]]] = []
        numbers: list[tuple[int, int, int, int]] = []
        for item in items:
            text = re.sub(
                r"\s+",
                "",
                unicodedata.normalize("NFKC", _text(item)),
            )
            box = _box(item)
            x, _, width, height = box
            if width < 1 or height < 1:
                continue
            if (
                text in {"ボーカル", "ダンス", "ビジュアル", "体力"}
                and column_min_x <= x <= column_max_x
            ):
                labels.append((text, box))
            elif (
                re.fullmatch(r"[0-9]{1,6}", text) is not None
                and column_min_x <= x <= column_max_x
            ):
                numbers.append(box)

        pair_candidates: list[
            tuple[
                float,
                float,
                str,
                tuple[int, int, int, int],
                tuple[int, int, int, int],
            ]
        ] = []
        for label, label_box in labels:
            label_x, label_y, label_width, label_height = label_box
            label_centre_x = label_x + label_width / 2.0
            label_centre_y = label_y + label_height / 2.0
            for number_box in numbers:
                number_x, number_y, number_width, number_height = number_box
                number_centre_x = number_x + number_width / 2.0
                number_centre_y = number_y + number_height / 2.0
                if (
                    number_centre_y
                    >= label_centre_y + max(4.0, label_height * 0.35)
                    and number_y
                    <= label_y + label_height + frame_height * 0.045
                    and abs(number_centre_x - label_centre_x)
                    <= frame_width * 0.04
                    and abs(number_x - label_x) <= frame_width * 0.04
                ):
                    pair_candidates.append(
                        (
                            abs(number_centre_x - label_centre_x),
                            number_centre_y - label_centre_y,
                            label,
                            label_box,
                            number_box,
                        )
                    )

        # Two different parameter rows in one narrow x lane prove the fixed
        # background column.  Values are matched one-to-one; sharing one OCR
        # number between two labels cannot manufacture the required proof.
        unique_pairs: list[
            tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]]
        ] = []
        seen_labels: set[str] = set()
        seen_numbers: set[tuple[int, int, int, int]] = set()
        for _, _, label, label_box, number_box in sorted(pair_candidates):
            if label in seen_labels or number_box in seen_numbers:
                continue
            seen_labels.add(label)
            seen_numbers.add(number_box)
            unique_pairs.append((label, label_box, number_box))
        if len(unique_pairs) < 2:
            return None
        pair_top = min(
            min(pair[1][1], pair[2][1]) for pair in unique_pairs
        )
        pair_bottom = max(
            max(
                pair[1][1] + pair[1][3],
                pair[2][1] + pair[2][3],
            )
            for pair in unique_pairs
        )
        if pair_top >= roi_bottom or pair_bottom <= roi_top:
            return None
        # A title-bound ROI deliberately starts slightly above the title.  A
        # background stat row can therefore touch only that top margin while
        # the actual popover begins below it.  Such a row cannot justify
        # clipping the entire right side: doing so would discard legitimate
        # effect tokens rendered to the right of the condition text (for
        # example ``好調状態の場合、集中``).  Require the proven background
        # column to reach into the title/effect panel itself.
        if pair_bottom <= title_y:
            return None
        label_order = {
            "ボーカル": 0,
            "ダンス": 1,
            "ビジュアル": 2,
            "体力": 3,
        }
        ordered_pairs = sorted(unique_pairs, key=lambda value: label_order[value[0]])
        for earlier, later in zip(ordered_pairs, ordered_pairs[1:], strict=False):
            earlier_index = label_order[earlier[0]]
            later_index = label_order[later[0]]
            earlier_y = earlier[1][1] + earlier[1][3] / 2.0
            later_y = later[1][1] + later[1][3] / 2.0
            index_gap = later_index - earlier_index
            row_step = (later_y - earlier_y) / index_gap
            if not frame_height * 0.02 <= row_step <= frame_height * 0.08:
                return None
        label_lefts = [pair[1][0] for pair in unique_pairs]
        number_lefts = [pair[2][0] for pair in unique_pairs]
        lane_tolerance = max(4, int(round(frame_width * 0.035)))
        if (
            max(label_lefts) - min(label_lefts) > lane_tolerance
            or max(number_lefts) - min(number_lefts) > lane_tolerance
        ):
            return None
        column_left = min((*label_lefts, *number_lefts))
        boundary = column_left - max(2, int(round(frame_width * 0.008)))
        if boundary <= title_right + max(4, int(round(frame_width * 0.02))):
            return None
        return boundary

    def _title_anchored_effect_roi_without_background_parameters(
        self,
        frame_width: int,
        frame_height: int,
        title_box: tuple[int, int, int, int],
        items: Sequence[Any],
    ) -> tuple[int, int, int, int]:
        """Clip only a geometrically proven right-side member-stat column."""

        left, top, width, height = self._title_anchored_effect_roi(
            frame_width,
            frame_height,
            title_box,
        )
        boundary = self._background_parameter_column_left(
            frame_width,
            frame_height,
            title_box,
            items,
        )
        if boundary is None or boundary >= left + width:
            return left, top, width, height
        clipped_width = boundary - left
        if clipped_width < 32 or boundary < title_box[0] + title_box[2]:
            return left, top, width, height
        if hasattr(self, "_runtime_counts"):
            self._increment(
                "skill_card_detail_title_anchored_background_parameter_clips"
            )
        return left, top, clipped_width, height

    def _skill_card_title_anchor_evidence(
        self,
        image: Any,
        title_pattern: str,
    ) -> tuple[list[Any], list[Any]]:
        """Reuse raw boxes from the exact detail frame for title geometry.

        The broad full-frame OCR is the authoritative observation for the
        accepted capture.  A title-specific backend call remains a fail-safe
        fallback only when those raw boxes do not produce one unique anchor;
        this preserves the prior failure semantics for unusual OCR layouts.
        """

        evidence = self._cached_full_frame_ocr_evidence(image)
        if evidence is not None and evidence.hit:
            native_items = list(evidence.all_items)
            matches = [
                item
                for item in native_items
                if re.fullmatch(title_pattern, _text(item).strip()) is not None
            ]
            if len(matches) == 1:
                if hasattr(self, "_runtime_counts"):
                    self._record_full_frame_ocr_cache_hit(evidence)
                    self._increment("skill_card_detail_title_anchor_cache_hits")
                return matches, native_items
        cache = getattr(self, "_title_anchor_ocr_evidence", None)
        cache_key = (id(image), title_pattern)
        cached = None if cache is None else cache.get(cache_key)
        if cached is not None and cached.image is image:
            if hasattr(self, "_runtime_counts"):
                self._increment("skill_card_detail_title_anchor_cache_hits")
            return list(cached.matches), list(cached.native_items)

        if hasattr(self, "_runtime_counts"):
            self._increment("skill_card_detail_title_anchor_backend_fallbacks")
        detail = self._ocr_detail(image, title_pattern)
        matches = (
            []
            if not detail or not detail.hit
            else list(detail.filtered_results or detail.all_results or [])
        )
        native_items = (
            []
            if not detail or not detail.hit
            else list(detail.all_results or detail.filtered_results or [])
        )
        if cache is None:
            cache = {}
            self._title_anchor_ocr_evidence = cache
        cache[cache_key] = _TitleAnchorOcrEvidence(
            image,
            tuple(matches),
            tuple(native_items),
        )
        while len(cache) > 16:
            oldest_key = next(iter(cache))
            if oldest_key == cache_key:
                break
            cache.pop(oldest_key, None)
        return matches, native_items

    def _skill_card_title_anchored_effect_roi_text(
        self,
        image: Any,
        card_id: int,
    ) -> str:
        """Retry effect OCR in the panel geometrically bound to its fixed title."""

        import cv2

        try:
            title_pattern = self.catalog.skill_card_title_anchor_pattern(card_id)
        except ArenaCatalogError as error:
            raise ArenaReaderError(
                "skill_card_detail_title_catalog_missing",
                str(error),
            ) from error
        matches, native_items = self._skill_card_title_anchor_evidence(
            image,
            title_pattern,
        )
        if len(matches) != 1:
            self._increment("skill_card_detail_title_anchor_transition_retries")
            raise ArenaReaderError(
                "skill_card_detail_title_anchor_ambiguous",
                f"card {card_id} has {len(matches)} title anchors on the retry frame",
            )
        height, width = image.shape[:2]
        title_box = _box(matches[0])
        left, top, roi_width, roi_height = (
            self._title_anchored_effect_roi_without_background_parameters(
                width,
                height,
                title_box,
                native_items,
            )
        )
        crop = image[top : top + roi_height, left : left + roi_width]
        if crop.size == 0:
            raise ArenaReaderError(
                "skill_card_detail_effect_roi_invalid",
                f"title-anchored detail effect ROI is empty for frame {width}x{height}",
            )
        # Preserve the native full-frame OCR boxes from the same recognition
        # that proved the title. The enlarged crop is valuable for small
        # glyphs, but interpolation can also erase a small standalone value.
        # Spatially rebuild both same-frame views and combine their evidence;
        # catalog resolution remains fail-closed when the views conflict.
        native_roi_items = []
        for item in native_items:
            item_x, item_y, item_width, item_height = _box(item)
            centre_x = item_x + item_width / 2.0
            centre_y = item_y + item_height / 2.0
            if (
                left <= centre_x <= left + roi_width
                and top <= centre_y <= top + roi_height
            ):
                native_roi_items.append(item)
        native_text = self._spatial_ocr_text(native_roi_items)
        enlarged = cv2.resize(
            crop,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        started = time.perf_counter()
        try:
            enlarged_text = self._with_full_frame_ocr_kind(
                "derived_roi",
                self._full_ocr_text,
                enlarged,
                spatial_reading_order=True,
            )
            observations = tuple(
                value
                for value in (native_text, enlarged_text)
                if value.strip()
            )
            texts = tuple(
                dict.fromkeys(
                    observations
                )
            )
            if not texts:
                raise ArenaReaderError(
                    "skill_card_detail_effect_roi_empty",
                    "title-bound native and enlarged OCR views were both empty",
                )
            return _TitleBoundEffectRoiText(
                self._merge_skill_card_effect_detail_views(card_id, texts),
                len(observations),
            )
        finally:
            self._increment("skill_card_detail_title_anchored_effect_roi_reads")
            self._increment("skill_card_detail_title_anchored_native_spatial_reads")
            self._increment("skill_card_detail_title_anchored_enlarged_spatial_reads")
            self._increment("skill_card_detail_effect_roi_reads")
            self._add_timing(
                "skill_card_detail_effect_roi_ocr",
                time.perf_counter() - started,
            )

    def _skill_card_title_anchored_enhanced_effect_roi_text(
        self,
        image: Any,
        card_id: int,
    ) -> str:
        """Contrast one title-bound effect panel on the already-captured frame."""

        import cv2

        self._increment("skill_card_detail_enhanced_effect_roi_attempts")
        started = time.perf_counter()
        try:
            try:
                title_pattern = self.catalog.skill_card_title_anchor_pattern(card_id)
            except ArenaCatalogError as error:
                raise ArenaReaderError(
                    "skill_card_detail_title_catalog_missing",
                    str(error),
                ) from error
            matches, native_items = self._skill_card_title_anchor_evidence(
                image,
                title_pattern,
            )
            if len(matches) != 1:
                self._increment(
                    "skill_card_detail_enhanced_effect_roi_title_ambiguous"
                )
                raise ArenaReaderError(
                    "skill_card_detail_title_anchor_ambiguous",
                    f"card {card_id} has {len(matches)} title anchors on the enhanced frame",
                )
            height, width = image.shape[:2]
            left, top, roi_width, roi_height = (
                self._title_anchored_effect_roi_without_background_parameters(
                    width,
                    height,
                    _box(matches[0]),
                    native_items,
                )
            )
            crop = image[top : top + roi_height, left : left + roi_width]
            if crop.size == 0:
                raise ArenaReaderError(
                    "skill_card_detail_effect_roi_invalid",
                    f"enhanced title-bound effect ROI is empty for frame {width}x{height}",
                )

            # This is a catalog-agnostic recognition view: local contrast makes
            # light effect labels legible without interpreting isolated digits.
            # The active catalog's existing effect matchers merge and certify
            # the result, so conflicting levels still fail closed.
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            contrasted = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            ).apply(gray)
            enlarged = cv2.resize(
                contrasted,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_LANCZOS4,
            )
            enhanced = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)
            return self._with_full_frame_ocr_kind(
                "derived_roi",
                self._full_ocr_text,
                enhanced,
                spatial_reading_order=True,
            )
        finally:
            self._add_timing(
                "skill_card_detail_enhanced_effect_roi_ocr",
                time.perf_counter() - started,
            )

    def _merge_skill_card_effect_detail_views(
        self,
        card_id: int,
        detail_texts: Sequence[str],
    ) -> str:
        """Apply the catalog-wide cross-view evidence gate before OCR fusion."""

        try:
            return self.catalog.merge_compatible_effect_detail_views(
                card_id,
                detail_texts,
            )
        except ArenaCatalogError as error:
            self._increment("skill_card_detail_effect_view_conflicts")
            raise ArenaReaderError(
                "skill_card_detail_effect_view_conflict",
                str(error),
            ) from error

    def _record_detail_count_override(
        self,
        key: tuple[int, int],
        observed_badge_count: int,
        resolved: ClickedSkillCard,
    ) -> None:
        resolved_count = sum(int(value) for value in resolved.customizations.values())
        self._detail_count_overrides[key] = {
            "slot": key[1],
            "observed_badge_count": observed_badge_count,
            "resolved_detail_count": resolved_count,
            "card_id": resolved.card_id,
            "customizations": dict(resolved.customizations),
            "detail_confirmation_reads": resolved.detail_confirmation_reads,
        }
        self._increment("skill_card_detail_count_overrides")

    def _record_cost_customization_fallback(
        self,
        key: tuple[int, int],
        resolved: ClickedSkillCard,
        *,
        target: TeamTarget | None,
        stage_number: int | None,
        member_slot: int | None,
    ) -> None:
        assumption = getattr(
            resolved,
            "conservative_cost_assumption",
            None,
        )
        if assumption is None:
            return
        if (
            target is None
            or stage_number not in (1, 2, 3)
            or member_slot not in (1, 2, 3)
        ):
            raise ArenaReaderError(
                "skill_card_cost_fallback_location_missing",
                "a conservative generic-cost fallback lacks its team/member location",
            )
        validate_conservative_cost_fallback(
            resolved,
            catalog=self.catalog,
            expected_policy=(
                "assume_unenhanced" if target.is_own_team else "assume_enhanced"
            ),
            expected_count=sum(
                int(value) for value in resolved.customizations.values()
            ),
            expected_count_kind="effective",
        )
        record = {
            **dict(assumption),
            "side": "own" if target.is_own_team else "opponent",
            "team_id": target.team_id,
            "opponent_position": target.opponent_position,
            "stage_number": stage_number,
            "member_slot": member_slot,
            "group_index": key[0],
            "card_slot": key[1],
            "detail_confirmation_reads": resolved.detail_confirmation_reads,
            "resolution_source": resolved.resolution_source,
            "detail_evidence_mode": resolved.detail_evidence_mode,
        }
        per_member = getattr(self, "_cost_customization_fallbacks", None)
        if per_member is None:
            per_member = {}
            self._cost_customization_fallbacks = per_member
        prior = per_member.get(key)
        if prior is not None:
            if prior != record:
                raise ArenaReaderError(
                    "skill_card_cost_fallback_changed",
                    f"generic-cost fallback for {key!r} changed within one member",
                )
            return
        per_member[key] = record
        runtime = getattr(self, "_runtime_cost_customization_fallbacks", None)
        if runtime is None:
            runtime = []
            self._runtime_cost_customization_fallbacks = runtime
        runtime.append(dict(record))
        self._increment("skill_card_cost_customization_fallbacks")
        self._increment(
            "skill_card_cost_customization_fallbacks_own"
            if target.is_own_team
            else "skill_card_cost_customization_fallbacks_opponent"
        )

    def close_skill_card(
        self, target: TeamTarget, stage_number: int, member_slot: int,
        group_index: int, card_slot: int,
    ) -> None:
        diagnostic = getattr(self, "_detail_failure_frames", None)
        if diagnostic is not None:
            diagnostic["position"]["phase"] = "close"
        try:
            self._close_skill_card_once(target, stage_number, member_slot, group_index, card_slot)
        except Exception as error:
            self._finish_card_transaction((group_index, card_slot), failed=True, error=str(error))
            raise

    def _close_skill_card_once(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
        card_slot: int,
    ) -> None:
        del target, stage_number, member_slot
        key = (group_index, card_slot)
        if self._active_inferred_clicked_card == key:
            # The candidate-detail inference already dismissed the overlay and
            # proved this exact slot against the accepted source generation.
            # Reusing the cached result performs no UI action, so a second
            # screenshot/signature guard here cannot add state evidence.
            self._active_inferred_clicked_card = None
            self._inferred_clicked_cards.pop(key, None)
            self._increment("skill_card_inferred_detail_reuses")
            return
        self._dismiss_skill_card_detail()
        try:
            self._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
                expected_card_id=self._card_predictions.get(key),
            )
        except ArenaReaderError:
            image = self._capture()
            if len(self._ocr(image, r"^ステージ\s*[123]$")) >= 3:
                raise ArenaReaderError(
                    "skill_card_close_left_member",
                    "closing a skill card unexpectedly left the member detail",
                )
            self._dismiss_skill_card_detail()
            self._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
                expected_card_id=self._card_predictions.get(key),
            )
        self._finish_card_transaction(key)

    def close_member(
        self, target: TeamTarget, stage_number: int, *,
        recovery_image: Any = None, recovery_detail: dict[str, Any] | None = None,
    ) -> None:
        self._reset_member_card_state()
        if recovery_image is None:
            self._back()
        else:
            self._back(image=recovery_image)
        expected_totals = (1, 2, 3) if target.is_own_team else (6,)
        try:
            self._wait_for_stage_member_list(
                expected_totals,
                timeout_seconds=2.0,
            )
        except ArenaReaderError:
            # A single retry is allowed only when the unchanged 体力/総合力
            # anchors prove that the first back action was dropped. Any other
            # page remains fail-closed rather than issuing another blind back.
            image = self._capture()
            if recovery_detail is None:
                if not self._ocr(image, r"^体力$") or not self._ocr(image, r"^総合力$"):
                    raise
            else:
                image, items = self._read_member_recovery_page(image)
                if not self._member_recovery_page_matches(items, stage_number, recovery_detail):
                    raise
            self._increment("close_member_retries")
            if recovery_detail is None:
                self._back()
            else:
                self._back(image=image)
            self._wait_for_stage_member_list(
                expected_totals,
                timeout_seconds=2.0,
            )
        self._member_metric_baseline = None

    def _retain_failed_skill_detail_identity(self, key: tuple[int, int]) -> None:
        """Keep the failed transaction's proven title until its member recovery."""
        member = getattr(self, "_member_failure_frames", None)
        proof = getattr(self, "_detail_identity_proofs", {}).get(key)
        if member is None or not isinstance(proof, _DetailIdentityProof):
            return
        if not self._detail_identity_proof_matches(key, proof.card_id, proof.source_card_box):
            return
        position = member["position"]
        member["skill_detail_recovery"] = {
            "team_id": position.get("team_id"),
            "stage_number": position.get("stage_number"),
            "member_slot": position.get("member_slot"),
            "group_index": key[0], "card_slot": key[1],
            "card_id": proof.card_id, "title": proof.title,
        }
        diagnostic = getattr(self, "_detail_failure_frames", None)
        if diagnostic is not None:
            diagnostic["position"].update(group_index=key[0], card_slot=key[1])
            self._note_confirmed_detail_name(proof.card_id)

    def _member_recovery_page_matches(
        self, items: Sequence[Any], stage_number: int, detail: dict[str, Any],
    ) -> bool:
        """Recognize the member or its already confirmed skill detail on this frame."""
        stages = self._matching_ocr_items(items, r"^ステージ\s*[123]$")
        if len(stages) != 1 or not self._matching_ocr_items(stages, rf"^ステージ\s*{stage_number}$"):
            return False
        if len(self._matching_ocr_items(items, r"^総合力$")) != 1:
            return False
        if len(self._matching_ocr_items(items, r"^体力$")) == 1:
            return True
        title_matches = 0
        for row_text, _box, components in self._spatial_ocr_rows(items):
            atom_matches = sum(
                self._same_proven_skill_card_title(text, detail["title"], detail["card_id"])
                for text, _component_box in components
            )
            title_matches += atom_matches or self._same_proven_skill_card_title(
                row_text, detail["title"], detail["card_id"],
            )
        return title_matches == 1

    def _read_member_recovery_page(self, image: Any = None) -> tuple[Any, Sequence[Any]]:
        deadline = time.monotonic() + 2.0
        for observation in range(3):
            if image is None:
                image = self._capture()
            items = self._ocr(image, r".+")
            if not self._retry_transient_communication_items(items):
                return image, items
            if observation == 2 or time.monotonic() >= deadline:
                break
            self._sleep(0.1)
            image = None
        raise ArenaReaderError(
            "arena_communication_retry_exhausted",
            "member recovery still sees a communication dialog after its existing Retry opportunity",
        )

    def recover_member_preview(
        self, target: TeamTarget, stage_number: int, member_slot: int, *, error_code: str,
    ) -> None:
        """Use the existing back operation, stopping at this team's preview."""
        started = time.perf_counter()
        before = dict(self._runtime_counts)
        succeeded = False
        try:
            # A close may already have returned to the preview. Check before
            # Back so that recovery cannot leave the current opponent.
            image, items = self._read_member_recovery_page()
            expected_totals = (1, 2, 3) if target.is_own_team else (6,)
            stages = self._matching_ocr_items(items, r"^ステージ\s*[123]$")
            totals = self._matching_ocr_items(items, r"^総合力$")
            member = getattr(self, "_member_failure_frames", None)
            detail = member.pop("skill_detail_recovery", None) if member is not None else None
            detail_matches_member = bool(
                detail is not None and error_code == "skill_card_close_failed"
                and (detail["team_id"], detail["stage_number"], detail["member_slot"])
                == (target.team_id, stage_number, member_slot)
            )
            if len(stages) >= 3 and len(totals) in expected_totals:
                self._reset_member_card_state()
                self._member_metric_baseline = None
            elif (
                (not stages or (
                    len(stages) == 1
                    and self._matching_ocr_items(stages, rf"^ステージ\s*{stage_number}$")
                ))
                and len(self._matching_ocr_items(items, r"^体力$")) == 1
                and len(totals) == 1
            ):
                self.close_member(target, stage_number)
            elif detail_matches_member and self._member_recovery_page_matches(items, stage_number, detail):
                self._increment("member_preview_open_detail_recoveries")
                self.close_member(
                    target, stage_number, recovery_image=image, recovery_detail=detail,
                )
            else:
                raise ArenaReaderError(
                    "member_preview_recovery_unproven",
                    "current frame proves neither this team's preview nor a member detail; no back sent",
                )
            succeeded = True
        finally:
            elapsed = time.perf_counter() - started
            self._record_duration_sample("member_preview_recovery", elapsed)
            logger.info(json.dumps({
                "event": "arena_reader_recovery", "action": "member_preview",
                "team_id": target.team_id, "stage_number": stage_number,
                "member_slot": member_slot, "error_code": error_code,
                "succeeded": succeeded, "wall_seconds": round(elapsed, 6),
                "extra_counts": {
                    key: value - before.get(key, 0)
                    for key, value in self._runtime_counts.items()
                    if value > before.get(key, 0)
                },
            }, ensure_ascii=False, sort_keys=True))

    def _reset_member_card_state(self) -> None:
        """Discard card geometry, hints and frames that belong to the prior member."""

        self._card_rows.clear()
        self._card_predictions.clear()
        self._card_candidate_groups.clear()
        self._card_images.clear()
        self._card_count_frames.clear()
        self._card_identity_frames.clear()
        self._card_source_ocr_counts.clear()
        getattr(self, "_full_frame_ocr_evidence", {}).clear()
        getattr(self, "_title_anchor_ocr_evidence", {}).clear()
        getattr(self, "_trusted_skill_card_title_rows_cache", {}).clear()
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
        getattr(self, "_badge_glyph_exemplars", {}).clear()
        getattr(self, "_badge_glyph_polluted_descriptors", set()).clear()
        getattr(self, "_badge_glyph_runtime_labels", {}).clear()
        self._inferred_clicked_cards.clear()
        self._active_inferred_clicked_card = None
        self._card_detail_texts.clear()
        self._card_detail_images.clear()
        getattr(self, "_card_detail_last_contact_released_at", {}).clear()
        getattr(self, "_card_detail_capture_started_at", {}).clear()
        self._p_item_diagnostics = ()
        self._p_item_generation_evidence.clear()
        for key in tuple(self._card_transaction_started):
            self._finish_card_transaction(key, failed=True)
        getattr(self, "_card_transaction_tokens", {}).clear()
        getattr(self, "_card_transaction_contact_counts", {}).clear()
        getattr(self, "_card_transaction_source_boxes", {}).clear()
        getattr(self, "_card_transaction_interaction_boxes", {}).clear()
        self._card_transaction_kinds.clear()
        getattr(self, "_card_transaction_ocr_started", {}).clear()
        self._secondary_fixed_slot_fallback_enabled = False
        self._secondary_presence_diagnostics = ()
        self._secondary_presence_cache.clear()
        self._card_content_generation_diagnostics = ()
        self._card_face_cost_diagnostics.clear()
        self._card_face_cost_optional_errors.clear()
        self._cost_customization_fallbacks.clear()
        self._zero_card_identity_fusion_diagnostics.clear()
        self._zero_card_embedding_detail_candidates.clear()
        self._zero_card_detail_candidates.clear()

    def leave_team(self, target: TeamTarget) -> None:
        self._recover_arena_main(
            maximum_steps=3,
            error_code="team_return_failed",
            require_opponents=not target.is_own_team,
        )

    def recover_to_arena_main(self, *, require_opponents: bool = True) -> None:
        self._recover_arena_main(
            maximum_steps=6,
            error_code="arena_recovery_failed",
            require_opponents=require_opponents,
        )

    def _recover_arena_main(
        self,
        *,
        maximum_steps: int,
        error_code: str,
        require_opponents: bool,
    ) -> None:
        unavailable_reads = 0
        for _ in range(maximum_steps):
            image = self._capture()
            self._last_retry_dialog_seen = False
            state, _ = self._arena_page_state(image)
            if arena_page_allows_team_entry(
                state,
                require_opponents=require_opponents,
            ):
                return
            if state is ArenaPageState.OPPONENTS_UNAVAILABLE:
                unavailable_reads += 1
                if unavailable_reads >= 3:
                    raise ArenaReaderError(
                        "arena_opponents_unavailable",
                        "opponent cards stayed absent on the arena main page; recovery will not navigate away",
                    )
                self._sleep(0.25)
                continue
            unavailable_reads = 0
            if self._dismiss_known_blocking_overlay(image):
                self._sleep(0.25)
                continue
            self._back()
        if getattr(self, "_last_retry_dialog_seen", False):
            raise ArenaReaderError(
                "arena_communication_retry_exhausted",
                "the error dialog remained after its one retry and bounded recovery wait",
            )
        required_state = "with three opponents" if require_opponents else "with the rehearsal anchor"
        raise ArenaReaderError(
            error_code,
            f"back navigation did not reach an arena main page {required_state}",
        )

    def _dismiss_known_blocking_overlay(self, image: Any) -> bool:
        """Dismiss only overlays proven by independent page-specific anchors."""

        # Recovery gets one full-frame problem check; reuse the same OCR boxes.
        if self._retry_transient_communication_items(self._ocr(image, r".+")):
            return True
        menu_profile = self._ocr(image, r"^プロフィール$")
        menu_settings = self._ocr(image, r"^設定$")
        if len(menu_profile) == 1 and len(menu_settings) == 1:
            height, width = image.shape[:2]
            # The upper-centre backdrop is outside the menu's functional grid
            # at every supported aspect ratio.  A click there dismisses the
            # proven menu without relying on a scale-sensitive close template.
            self._click((int(width * 0.50), int(height * 0.16), 1, 1))
            self._increment("arena_menu_overlays_dismissed")
            return True

        error_titles = self._ocr(image, r"^通信エラ(?:ー)?$")
        failure_details = self._ocr(image, r"^アセット取得に失敗$")
        if len(error_titles) != 1 or len(failure_details) != 1:
            return False
        close_buttons = self._recognize("CloseRoundButton", image)
        if len(close_buttons) != 1:
            raise ArenaReaderError(
                "arena_blocking_overlay_close_ambiguous",
                "a communication-error overlay is proven but its close button is not unique",
            )
        self._click(_box(close_buttons[0]))
        self._increment("arena_blocking_overlays_dismissed")
        return True

    def _retry_transient_communication_items(self, items: Sequence[Any]) -> bool:
        """Return whether a proven dialog occupies this already-read frame."""
        self._check_cancelled()
        retry_box = error_retry_box(items)
        self._last_retry_dialog_seen = retry_box is not None
        if retry_box is None:
            return False
        if getattr(self, "_arena_communication_retry_attempted", False):
            # The original deadline may still be observing the sent Retry's
            # transition. Do not resend and do not extend that deadline.
            return True
        self._arena_communication_retry_attempted = True
        started = time.perf_counter()
        succeeded = False
        try:
            self._click(retry_box, settle_seconds=0)
            self._increment("arena_communication_retries")
            succeeded = True
        finally:
            logger.info(json.dumps({
                "event": "arena_reader_recovery", "action": "communication_retry",
                "succeeded": succeeded, "extra_ocr": 0, "extra_clicks": 1,
                "wall_seconds": round(time.perf_counter() - started, 6),
            }, ensure_ascii=False, sort_keys=True))
        return True

    def _check_cancelled(self) -> None:
        cancellation = getattr(self, "_cancellation", None)
        if cancellation is None:
            cancellation = cancellation_for(getattr(self, "context", None))
            self._cancellation = cancellation
        cancellation.check()

    def _sleep(self, seconds: float) -> None:
        self._check_cancelled()
        time.sleep(seconds)
        self._check_cancelled()

    def _run_recognition(self, *args, **kwargs):
        self._check_cancelled()
        result = self.context.run_recognition(*args, **kwargs)
        self._check_cancelled()
        return result

    def _capture(self) -> Any:
        self._check_cancelled()
        started = time.perf_counter()
        try:
            image = self.context.tasker.controller.post_screencap().wait().get()
            self._check_cancelled()
            for name in ("_detail_failure_frames", "_member_failure_frames"):
                diagnostic = getattr(self, name, None)
                if diagnostic is not None and not diagnostic.get("persisted", False):
                    diagnostic["frames"].append((time.perf_counter(), image))
            return image
        finally:
            self._increment("screenshots")
            self._add_timing("screenshots", time.perf_counter() - started)

    def _recognize(self, entry: str, image: Any) -> list[Any]:
        detail = self._run_recognition(entry, image)
        if not detail or not detail.hit:
            return []
        return list(detail.filtered_results or detail.all_results or [])

    def _ocr(
        self,
        image: Any,
        expected: str,
        *,
        roi: tuple[int, int, int, int] | None = None,
        only_rec: bool = False,
    ) -> list[Any]:
        if expected == r".+" and roi is None and not only_rec:
            evidence = self._full_frame_ocr_evidence_for(image)
            return list(evidence.filtered_items) if evidence.hit else []
        detail_kwargs: dict[str, Any] = {}
        if roi is not None:
            detail_kwargs["roi"] = roi
        if only_rec:
            detail_kwargs["only_rec"] = True
        detail = self._ocr_detail(image, expected, **detail_kwargs)
        if not detail or not detail.hit:
            return []
        return list(detail.filtered_results or detail.all_results or [])

    @staticmethod
    def _matching_ocr_items(items: Sequence[Any], expected: str) -> tuple[Any, ...]:
        """Apply Maa's expected search to already filtered, ordered boxes."""
        pattern = re.compile(expected)
        return tuple(item for item in items if pattern.search(_text(item)))

    def _ocr_detail(
        self,
        image: Any,
        expected: str,
        *,
        roi: tuple[int, int, int, int] | None = None,
        only_rec: bool = False,
    ) -> Any:
        """Return one OCR detail while preserving filtered and raw boxes."""

        override: dict[str, Any] = {
            "ArenaReaderOCR": {
                "recognition": "OCR",
                "expected": expected,
                "order_by": "Vertical",
            }
        }
        if roi is not None:
            override["ArenaReaderOCR"]["roi"] = list(roi)
        if only_rec:
            override["ArenaReaderOCR"]["only_rec"] = True
        return self._run_recognition(
            "ArenaReaderOCR",
            image,
            pipeline_override=override,
        )

    @staticmethod
    def _spatial_ocr_rows(
        items: Sequence[Any],
    ) -> tuple[
        tuple[
            str,
            tuple[int, int, int, int],
            tuple[tuple[str, tuple[int, int, int, int]], ...],
        ],
        ...,
    ]:
        """Rebuild visual reading rows with their union boxes.

        Maa's vertical ordering compares top coordinates. Two tokens on one
        visual line can therefore be reversed when the right token has a taller
        box that begins a few pixels earlier. Cluster only strongly overlapping
        vertical spans, then order members left-to-right and rows top-to-bottom.
        """

        entries: list[tuple[int, int, int, int, str]] = []
        for item in items:
            value = _text(item).strip()
            if not value:
                continue
            x, y, width, height = _box(item)
            if width < 1 or height < 1:
                raise ArenaReaderError(
                    "recognition_box_invalid",
                    f"OCR result has an invalid box: {(x, y, width, height)!r}",
                )
            entries.append((x, y, width, height, value))

        def median(values: Sequence[float]) -> float:
            ordered = sorted(values)
            middle = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[middle]
            return (ordered[middle - 1] + ordered[middle]) / 2.0

        rows: list[list[tuple[int, int, int, int, str]]] = []
        for entry in sorted(
            entries,
            key=lambda value: (value[1] + value[3] / 2.0, value[0]),
        ):
            _, y, _, height, _ = entry
            centre = y + height / 2.0
            candidates: list[tuple[float, int]] = []
            for row_index, row in enumerate(rows):
                row_centres = [item[1] + item[3] / 2.0 for item in row]
                row_heights = [float(item[3]) for item in row]
                row_centre = median(row_centres)
                row_height = median(row_heights)
                height_ratio = max(height, row_height) / min(height, row_height)
                overlap = min(
                    y + height,
                    row_centre + row_height / 2.0,
                ) - max(
                    y,
                    row_centre - row_height / 2.0,
                )
                overlap_ratio = max(0.0, overlap) / min(height, row_height)
                centre_gap = abs(centre - row_centre)
                if (
                    height_ratio <= 1.50
                    and overlap_ratio >= 0.50
                    and centre_gap <= 0.65 * max(height, row_height)
                ):
                    candidates.append((centre_gap, row_index))
            if candidates:
                _, row_index = min(candidates)
                rows[row_index].append(entry)
            else:
                rows.append([entry])

        rows.sort(
            key=lambda row: (
                median([item[1] + item[3] / 2.0 for item in row]),
                min(item[0] for item in row),
            )
        )
        spatial_rows: list[
            tuple[
                str,
                tuple[int, int, int, int],
                tuple[tuple[str, tuple[int, int, int, int]], ...],
            ]
        ] = []
        for row in rows:
            ordered_row = sorted(row, key=lambda value: (value[0], value[1]))
            left = min(item[0] for item in ordered_row)
            top = min(item[1] for item in ordered_row)
            right = max(item[0] + item[2] for item in ordered_row)
            bottom = max(item[1] + item[3] for item in ordered_row)
            spatial_rows.append(
                (
                    "".join(item[4] for item in ordered_row),
                    (left, top, right - left, bottom - top),
                    tuple(
                        (item[4], (item[0], item[1], item[2], item[3]))
                        for item in ordered_row
                    ),
                )
            )
        return tuple(spatial_rows)

    @staticmethod
    def _spatial_ocr_text(items: Sequence[Any]) -> str:
        """Flatten spatial OCR rows into newline-separated text."""

        return "\n".join(
            text
            for text, _, _ in MaaArenaReaderBackend._spatial_ocr_rows(items)
        )

    def _full_ocr_text(
        self,
        image: Any,
        *,
        spatial_reading_order: bool = False,
    ) -> str:
        evidence = self._full_frame_ocr_evidence_for(image)
        if not evidence.hit:
            raise ArenaReaderError("ocr_empty", "full-screen OCR returned no text")
        items = list(evidence.all_items)
        if spatial_reading_order:
            return self._spatial_ocr_text(items)
        return "\n".join(_text(item) for item in items)

    @staticmethod
    def _p_item_detail_panel_rows(
        image: Any,
        items: Sequence[Any],
        *,
        background_only: bool = False,
    ) -> tuple[
        tuple[
            str,
            tuple[float, float, float, float],
            tuple[tuple[str, tuple[float, float, float, float]], ...],
        ],
        ...,
    ]:
        """Read spatial OCR rows and normalized boxes from the left panel.

        P-item titles move vertically with source-page content, so geometry
        identifies the panel rather than guessing a title band. The transaction
        layer removes frozen source rows and treats only the first new row as
        title evidence; later effect prose remains visible for diagnostics but
        cannot participate in catalog matching.

        The background-only view contains just the same-panel atoms excluded
        by the title height gate. It never supplies title or page signatures.
        """

        height, width = image.shape[:2]
        if width < 320 or height < 568:
            raise ArenaReaderError(
                "p_item_detail_title_roi_invalid",
                f"capture {width}x{height} is too small for P-item title geometry",
            )
        scale_x = width / 720.0
        scale_y = height / 1280.0
        panel_items: list[Any] = []
        for item in items:
            x, y, item_width, item_height = _box(item)
            normalized_x = x / scale_x
            normalized_y = y / scale_y
            normalized_width = item_width / scale_x
            normalized_height = item_height / scale_y
            normalized_text = re.sub(
                r"\s+",
                "",
                unicodedata.normalize("NFKC", _text(item)),
            )
            if (
                normalized_width < 16
                or normalized_height <= 0
                or (18 <= normalized_height <= 45) == background_only
                or not normalized_text
                or not any(character.isalpha() for character in normalized_text)
            ):
                continue
            in_left_panel = (
                18 <= normalized_x
                and normalized_x <= 360
                and normalized_x + normalized_width <= 650
                # The live top-panel title box can begin at y=15.  Reserve a
                # +/-3 px detector-box tolerance without admitting rows above
                # y=12; all horizontal, size, frozen-source and first-new-row
                # identity gates remain intact.
                and 12 <= normalized_y <= 270
            )
            if in_left_panel:
                panel_items.append(item)
        if not panel_items:
            return ()

        spatial_rows = MaaArenaReaderBackend._spatial_ocr_rows(panel_items)

        def isolated_ascii_artwork_noise(
            text: str,
            row_box: tuple[int, int, int, int],
            components: tuple[tuple[str, tuple[int, int, int, int]], ...],
        ) -> bool:
            normalized = re.sub(
                r"\s+",
                "",
                unicodedata.normalize("NFKC", text),
            )
            if (
                len(components) != 1
                or re.fullmatch(r"[A-Za-z]", normalized) is None
            ):
                return False
            row_x, row_y, row_width, row_height = row_box
            normalized_row_box = (
                row_x / scale_x,
                row_y / scale_y,
                row_width / scale_x,
                row_height / scale_y,
            )
            if not (
                normalized_row_box[0] <= 30
                and normalized_row_box[1] <= 45
                and normalized_row_box[2] <= 24
                and normalized_row_box[3] <= 22
            ):
                # A low-confidence letter at the normal title position remains
                # the first new row.  Removing it could promote effect prose to
                # identity evidence, so geometry outside the observed artwork
                # corner must continue to fail closed.
                return False
            component_text, component_box = components[0]
            scores = tuple(
                float(_value(item, "score", 1.0))
                for item in panel_items
                if _text(item).strip() == component_text
                and _box(item) == component_box
            )
            # Full-screen OCR can expose one low-confidence Latin glyph from
            # the P-item artwork above the real title.  Filter only a complete
            # isolated row after spatial joining so split titles such as
            # ``P`` + ``っち+`` retain their Latin component.
            return bool(scores) and max(scores) < 0.50

        return tuple(
            (
                text,
                (
                    x / scale_x,
                    y / scale_y,
                    row_width / scale_x,
                    row_height / scale_y,
                ),
                tuple(
                    (
                        component_text,
                        (
                            component_x / scale_x,
                            component_y / scale_y,
                            component_width / scale_x,
                            component_height / scale_y,
                        ),
                    )
                    for component_text, (
                        component_x,
                        component_y,
                        component_width,
                        component_height,
                    ) in components
                ),
            )
            for text, (x, y, row_width, row_height), components in spatial_rows
            if not isolated_ascii_artwork_noise(
                text,
                (x, y, row_width, row_height),
                components,
            )
        )

    @staticmethod
    def _p_item_detail_title_text(
        image: Any,
        items: Sequence[Any],
    ) -> str:
        """Flatten normalized left-panel rows for diagnostics and tests."""

        return "\n".join(
            text
            for text, _, _ in MaaArenaReaderBackend._p_item_detail_panel_rows(
                image,
                items,
            )
        )

    def _p_item_ocr_observation(
        self,
        image: Any,
    ) -> _PItemOcrObservation:
        """OCR one frame once and retain full text, title geometry and anchors."""

        detail = self._run_recognition(
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
        full_text = "\n".join(_text(item) for item in items)
        panel_rows = self._p_item_detail_panel_rows(image, items)
        background_rows = self._p_item_detail_panel_rows(
            image, items, background_only=True,
        )
        title_text = "\n".join(text for text, _, _ in panel_rows)
        exact_ocr_rows = {_text(item).strip() for item in items}
        member_anchors_visible = "体力" in exact_ocr_rows and "総合力" in exact_ocr_rows
        return _PItemOcrObservation(
            full_text, title_text, member_anchors_visible, panel_rows, background_rows,
            tuple(items),
        )

    def _click(
        self,
        box: tuple[int, int, int, int],
        *,
        settle_seconds: float = 0.25,
    ) -> None:
        self._check_cancelled()
        started = time.perf_counter()
        dispatch = "context_box"
        if box[2:] == (1, 1):
            dispatch = "controller_direct"
            job = self.context.tasker.controller.post_click(box[0], box[1]).wait()
            succeeded = bool(job.succeeded)
        else:
            result = self.context.run_action_direct(JActionType.Click, JClick(), box)
            succeeded = bool(result is not None and result.success)
        elapsed = time.perf_counter() - started
        self._add_timing("click_actions", elapsed)
        self._add_timing(f"{dispatch}_click_actions", elapsed)
        self._increment("click_actions")
        self._increment(f"{dispatch}_click_actions")
        self._record_detail_action("click", box, succeeded=succeeded)
        self._check_cancelled()
        if not succeeded:
            raise ArenaReaderError("maa_click_failed", f"Maa could not click {box}")
        if settle_seconds > 0:
            self._sleep(settle_seconds)

    @staticmethod
    def _box_center_point(
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Collapse one uniquely recognized control to its deterministic centre."""

        x, y, width, height = box
        if width < 1 or height < 1:
            raise ArenaReaderError(
                "maa_click_box_invalid",
                f"recognized control has an invalid box: {box!r}",
            )
        return (x + width // 2, y + height // 2, 1, 1)

    def _long_press(self, box: tuple[int, int, int, int]) -> None:
        self._check_cancelled()
        x, y, width, height = box
        point = (x + width // 2, y + height // 2)
        started = time.perf_counter()
        controller = self.context.tasker.controller
        down_succeeded = False
        up_succeeded = False
        interaction_error: BaseException | None = None
        try:
            down_succeeded = bool(
                controller.post_touch_down(*point).wait().succeeded
            )
            if down_succeeded:
                self._sleep(self._member_long_press_seconds)
        except (ArenaTaskCancelled, Exception) as error:
            interaction_error = error
        finally:
            try:
                up_succeeded = bool(controller.post_touch_up().wait().succeeded)
            except Exception as error:  # pragma: no cover - native Maa boundary
                if interaction_error is None:
                    interaction_error = error
        elapsed = time.perf_counter() - started
        self._add_timing("long_press_actions", elapsed)
        self._add_timing("controller_direct_long_press_actions", elapsed)
        self._increment("long_press_actions")
        self._increment("controller_direct_long_press_actions")
        self._record_detail_action("long_press", box, succeeded=down_succeeded and up_succeeded)
        self._check_cancelled()
        if interaction_error is not None or not down_succeeded or not up_succeeded:
            detail = (
                f"Maa could not long-press {box} at {point}; "
                f"down={down_succeeded}, up={up_succeeded}"
            )
            if interaction_error is not None:
                detail += f", error={interaction_error}"
            raise ArenaReaderError("maa_long_press_failed", detail)

    def _short_press(
        self,
        box: tuple[int, int, int, int],
        *,
        duration_ms: int,
    ) -> None:
        """Send one bounded tap-like contact when an instantaneous click drops."""

        self._check_cancelled()
        if not 50 <= duration_ms <= 200:
            raise ArenaReaderError(
                "maa_short_press_duration_invalid",
                f"short press duration is outside 50..200 ms: {duration_ms}",
            )
        x, y, width, height = box
        point = (x + width // 2, y + height // 2)
        started = time.perf_counter()
        controller = self.context.tasker.controller
        down_succeeded = False
        up_succeeded = False
        interaction_error: BaseException | None = None
        try:
            down_succeeded = bool(
                controller.post_touch_down(*point).wait().succeeded
            )
            if down_succeeded:
                self._sleep(duration_ms / 1000.0)
        except (ArenaTaskCancelled, Exception) as error:
            interaction_error = error
        finally:
            try:
                up_succeeded = bool(controller.post_touch_up().wait().succeeded)
            except Exception as error:  # pragma: no cover - native Maa boundary
                if interaction_error is None:
                    interaction_error = error
        elapsed = time.perf_counter() - started
        self._add_timing("short_press_actions", elapsed)
        self._add_timing("controller_direct_short_press_actions", elapsed)
        self._increment("short_press_actions")
        self._increment("controller_direct_short_press_actions")
        self._record_detail_action("short_press", box, succeeded=down_succeeded and up_succeeded)
        self._check_cancelled()
        if interaction_error is not None or not down_succeeded or not up_succeeded:
            detail = (
                f"Maa could not short-press {box} at {point}; "
                f"down={down_succeeded}, up={up_succeeded}"
            )
            if interaction_error is not None:
                detail += f", error={interaction_error}"
            raise ArenaReaderError(
                "maa_short_press_failed",
                detail,
            )

    def _swipe(
        self,
        *,
        vertical: str,
        distance_ratio: float = 0.0625,
        distance_pixels: int | None = None,
    ) -> None:
        image = self._capture()
        height, width = image.shape[:2]
        if not 0.0 < distance_ratio <= 0.20:
            raise ValueError("distance_ratio must be in (0, 0.20]")
        if distance_pixels is None:
            distance_pixels = int(round(height * distance_ratio))
        if not 1 <= distance_pixels <= int(height * 0.20):
            raise ValueError("distance_pixels must be in [1, 20% of frame height]")
        lower_y = int(height * 0.34)
        upper_y = lower_y - distance_pixels
        if vertical == "up":
            begin, end = (int(width * 0.90), lower_y, 1, 1), (
                int(width * 0.90),
                upper_y,
                1,
                1,
            )
        else:
            begin, end = (int(width * 0.90), upper_y, 1, 1), (
                int(width * 0.90),
                lower_y,
                1,
                1,
            )
        started = time.perf_counter()
        self._check_cancelled()
        job = self.context.tasker.controller.post_swipe(
            begin[0],
            begin[1],
            end[0],
            end[1],
            duration=self._card_swipe_duration_ms,
        ).wait()
        elapsed = time.perf_counter() - started
        self._add_timing("swipe_actions", elapsed)
        self._add_timing("controller_direct_swipe_actions", elapsed)
        self._increment("swipe_actions")
        self._increment("controller_direct_swipe_actions")
        self._record_detail_action("swipe", (*begin[:2], *end[:2]), succeeded=bool(job.succeeded))
        self._check_cancelled()
        if not job.succeeded:
            raise ArenaReaderError("maa_swipe_failed", f"Maa could not swipe {vertical}")

    def _back(self, *, image: Any = None) -> None:
        if image is None:
            image = self._capture()
        height, width = image.shape[:2]
        detail = self._run_recognition(
            "ChallengeBack",
            image,
            pipeline_override={
                "ChallengeBack": {
                    "recognition": "TemplateMatch",
                    "template": "back.png",
                    "roi": [0, int(height * 0.8), int(width * 0.5), int(height * 0.2)],
                    "threshold": 0.7,
                    "order_by": "Score",
                }
            },
        )
        results = list((detail.filtered_results or detail.all_results or [])) if detail and detail.hit else []
        if len(results) != 1:
            raise ArenaReaderError("maa_back_anchor_ambiguous", f"found {len(results)} back-button anchors")
        self._click(self._box_center_point(_box(results[0])))

    def _close_overlay(self) -> None:
        self._check_cancelled()
        result = self.context.run_task("CloseButton")
        self._check_cancelled()
        if not result or not result.status.succeeded:
            image = self._capture()
            height, width = image.shape[:2]
            self._click((int(width * 0.91), int(height * 0.02), int(width * 0.07), int(height * 0.07)))

    def _dismiss_skill_card_detail(self) -> None:
        """Close a skill-card detail by tapping the inert upper-left backdrop."""

        # A successfully opened skill-card transaction already owns a fresh,
        # accepted detail capture.  The dismissal point consumes only the
        # normalized frame dimensions, never its pixels, so taking another
        # screenshot here adds no state evidence.  Reuse is deliberately
        # limited to the sole active transaction; failed opens, P-item flows,
        # diagnostic calls, malformed frames, and inconsistent state retain
        # the original capture fallback.
        active_keys = tuple(
            getattr(self, "_card_transaction_started", {})
        )
        dimensions: tuple[int, int] | None = None
        detail_images = getattr(self, "_card_detail_images", {})
        if len(active_keys) == 1 and active_keys[0] in detail_images:
            detail_image = detail_images[active_keys[0]]
            shape = getattr(detail_image, "shape", ())
            if len(shape) >= 2:
                height, width = int(shape[0]), int(shape[1])
                if height > 0 and width > 0:
                    dimensions = (height, width)
                    self._increment(
                        "skill_card_detail_dismiss_shape_reuses"
                    )
        if dimensions is None:
            image = self._capture()
            height, width = image.shape[:2]
        else:
            height, width = dimensions
        self._increment("skill_card_detail_event_driven_dismissals")
        self._click(
            (
                max(1, int(width * 0.015)),
                max(1, int(height * 0.04)),
                1,
                1,
            ),
            settle_seconds=0,
        )

    def _skill_card_detail_close_retry_proven(
        self,
        group_index: int,
        expected_card_id: int,
        source_card_box: tuple[int, int, int, int] | None = None,
    ) -> bool:
        """Prove on two fresh frames that the same detail overlay stayed open."""

        exact_resolver = getattr(
            self.catalog,
            "confirm_clicked_skill_card_by_exact_title",
            None,
        )
        if exact_resolver is None:
            return False
        for frame_index in range(2):
            image = self._capture()
            title_text = self._authoritative_skill_card_title_text(
                group_index,
                image,
                candidate_card_ids=(expected_card_id,),
                source_card_box=source_card_box,
            )
            if title_text is None:
                return False
            try:
                confirmed_card_id = self._resolve_proven_skill_card_title(title_text)
            except ArenaCatalogError:
                return False
            if confirmed_card_id != expected_card_id:
                return False
            self._increment("skill_card_detail_close_retry_evidence_frames")
            if frame_index == 0:
                self._sleep(self._source_restore_poll_seconds)
        return True

    def _assert_inferred_card_group_visible_after_dismiss(
        self,
        group_index: int,
        *,
        card_slot: int,
        expected_card_id: int,
    ) -> None:
        """Restore one inferred-card source, retrying one proven dropped dismiss."""

        diagnostic = getattr(self, "_detail_failure_frames", None)
        if diagnostic is not None:
            diagnostic["position"]["phase"] = "close"
        try:
            self._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
                expected_card_id=expected_card_id,
            )
        except ArenaReaderError as error:
            if error.code != "skill_card_close_failed":
                raise
            if not self._skill_card_detail_close_retry_proven(
                group_index,
                expected_card_id,
                self._skill_card_source_box((group_index, card_slot)),
            ):
                raise
            self._increment("skill_card_detail_close_retries")
            self._dismiss_skill_card_detail()
            self._assert_card_group_visible(
                group_index,
                card_slot=card_slot,
                expected_card_id=expected_card_id,
            )

    def _assert_stage_navigation(self) -> None:
        results = self._ocr(self._capture(), r"^ステージ\s*[123]$")
        if len(results) < 3:
            raise ArenaReaderError("stage_navigation_missing", f"found only {len(results)} stage anchors")

    def _wait_for_stage_member_list(
        self,
        expected_total_counts: tuple[int, ...],
        timeout_seconds: float = 3.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_stage_count = 0
        last_total_count = 0
        while time.monotonic() < deadline:
            image = self._capture()
            items = self._ocr(image, r".+")
            last_stage_count = len(self._matching_ocr_items(items, r"^ステージ\s*[123]$"))
            last_total_count = len(self._matching_ocr_items(items, r"^総合力$"))
            communication_dialog = self._retry_transient_communication_items(items)
            if not communication_dialog and last_stage_count >= 3 and last_total_count in expected_total_counts:
                return
            self._sleep(0.25)
        raise ArenaReaderError(
            "stage_member_list_timeout",
            f"stage anchors={last_stage_count}, total anchors={last_total_count}, "
            f"expected totals={expected_total_counts!r}",
        )

    def _stage_total_anchors(self, image: Any) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            sorted(
                (_box(item) for item in self._ocr(image, r"^総合力$")),
                key=lambda box: (box[1], box[0]),
            )
        )

    def _team_stage_total_anchors(
        self,
        target: TeamTarget,
        image: Any,
    ) -> tuple[tuple[int, int, int, int], ...]:
        totals = self._stage_total_anchors(image)
        try:
            return team_stage_total_anchors(totals, own_team=target.is_own_team)
        except ValueError as error:
            raise ArenaReaderError(
                "stage_total_count_mismatch",
                f"{target.team_id} stage totals are invalid: {error}",
            ) from error

    def _arena_main_visible(self) -> bool:
        image = self._capture()
        state, _ = self._arena_page_state(image)
        return state is ArenaPageState.READY

    def _arena_page_state(
        self,
        image: Any,
    ) -> tuple[ArenaPageState, tuple[int, int, int, int, int]]:
        items = self._ocr(image, r".+")
        rehearsal_count = len(self._matching_ocr_items(items, r"^リハーサル$"))
        opponent_count = len(self._recognize("ArenaReaderOpponentCards", image))
        stage_label_count = len(self._matching_ocr_items(items, r"^ステージ\s*[123]$"))
        stage_total_count = len(self._matching_ocr_items(items, r"^総合力$"))
        exhausted_notice_count = len(
            self._matching_ocr_items(items, r"^本日の挑戦権を消費しました$")
        )
        counts = (
            rehearsal_count,
            opponent_count,
            stage_label_count,
            stage_total_count,
            exhausted_notice_count,
        )
        return (
            classify_arena_page(
                rehearsal_count=rehearsal_count,
                opponent_count=opponent_count,
                stage_label_count=stage_label_count,
                stage_total_count=stage_total_count,
                exhausted_notice_count=exhausted_notice_count,
            ),
            counts,
        )

    def _read_grade(
        self,
        image: Any,
        *,
        observations: Sequence[Any] | None = None,
    ) -> int:
        height, width = image.shape[:2]
        observations = tuple(
            self._ocr(image, r".+")
            if observations is None
            else observations
        )
        labels = tuple(
            item
            for item in observations
            if _normalize_latin_anchor(_text(item)) == "GRADE"
        )
        if len(labels) != 1:
            raise ArenaReaderError(
                "arena_grade_label_ambiguous",
                f"found {len(labels)} GRADE labels on the arena main screen",
            )
        label_x, label_y, label_width, label_height = _box(labels[0])
        # The Grade digit and label are part of one emblem: the digit is
        # centred above the GRADE word.  Derive a tight, scale-aware region
        # from the label itself so stage numbers and the inter-stage ornament
        # below the label can never become Grade evidence.
        left = max(0, label_x)
        right = min(width, label_x + label_width)
        vertical_span = max(label_width * 2, label_height * 4)
        top = max(0, label_y - vertical_span)
        bottom = min(height, label_y)
        minimum_digit_height = max(1, round(label_width * 0.35))
        emblem_roi = (
            left,
            top,
            max(0, right - left),
            max(0, bottom - top),
        )
        matches = tuple(
            (item, grade_value)
            for item in observations
            if (grade_value := _grade_ocr_value(_text(item))) is not None
            and _box(item)[3] >= minimum_digit_height
            and (
                left
                <= _box(item)[0] + _box(item)[2] / 2
                <= right
                and top
                <= _box(item)[1] + _box(item)[3] / 2
                <= bottom
            )
        )
        targeted_views: list[dict[str, Any]] = []
        if not matches:
            # Full-screen OCR can consistently omit the large stylised Grade
            # digit while still resolving the surrounding small menu text. A
            # crop that includes the whole emblem is not a valid fallback:
            # its ornament and digit can merge into one CJK-shaped OCR line.
            # Isolate only the fixed digit lane above the unique GRADE label,
            # then require all non-empty detector/recognizer views to agree.
            digit_left = max(0, round(label_x + label_width * 0.08))
            digit_right = min(width, round(label_x + label_width * 0.92))
            digit_top = max(0, round(label_y - label_width * 1.55))
            digit_bottom = min(height, round(label_y - label_width * 0.60))
            digit_roi = (
                digit_left,
                digit_top,
                max(0, digit_right - digit_left),
                max(0, digit_bottom - digit_top),
            )
            accepted_values: list[int] = []
            observed_values: set[int] = set()
            for only_rec in (False, True):
                detail = self._ocr_detail(
                    image,
                    r"^[1-7]$",
                    roi=digit_roi,
                    only_rec=only_rec,
                )
                raw_items = tuple(
                    detail.all_results or () if detail is not None else ()
                )
                filtered_items = tuple(
                    detail.filtered_results or ()
                    if detail is not None and detail.hit
                    else ()
                )
                raw_matches = tuple(
                    (item, grade_value)
                    for item in (*raw_items, *filtered_items)
                    if (grade_value := _grade_ocr_value(_text(item))) is not None
                    and _box(item)[3] >= minimum_digit_height
                )
                filtered_matches = tuple(
                    (item, grade_value)
                    for item in filtered_items
                    if (grade_value := _grade_ocr_value(_text(item))) is not None
                    and _box(item)[3] >= minimum_digit_height
                )
                raw_values = tuple(
                    dict.fromkeys(value for _, value in raw_matches)
                )
                filtered_values = tuple(
                    dict.fromkeys(value for _, value in filtered_matches)
                )
                targeted_views.append(
                    {
                        "only_rec": only_rec,
                        "hit": bool(detail is not None and detail.hit),
                        "raw": tuple((_text(item), _box(item)) for item in raw_items),
                        "filtered": tuple(
                            (_text(item), _box(item)) for item in filtered_items
                        ),
                        "raw_values": raw_values,
                        "accepted_values": filtered_values,
                    }
                )
                observed_values.update(raw_values)
                if len(filtered_values) == 1:
                    accepted_values.append(filtered_values[0])
            if (
                accepted_values
                and len(set(accepted_values)) == 1
                and observed_values == {accepted_values[0]}
            ):
                return accepted_values[0]
        if len(matches) != 1:
            nearby = tuple(
                (_text(item), _box(item))
                for item in observations
                if (
                    left
                    <= _box(item)[0] + _box(item)[2] / 2
                    <= right
                    and top
                    <= _box(item)[1] + _box(item)[3] / 2
                    <= label_y + label_height
                )
            )
            raise ArenaReaderError(
                "arena_grade_ambiguous",
                f"found {len(matches)} geometrically valid Grade digits above the "
                f"GRADE anchor in {emblem_roi}; targeted_ocr="
                f"{tuple(targeted_views)!r}; "
                f"nearby_ocr={nearby!r}",
            )
        return matches[0][1]

    def _assert_member_detail(self, timeout_seconds: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            image = self._capture()
            items = self._ocr(image, r".+")
            communication_dialog = self._retry_transient_communication_items(items)
            # Keep the original two-query member cadence so P-item sampling
            # does not start earlier during the page transition.
            if not communication_dialog and self._matching_ocr_items(items, r"^体力$") and self._ocr(image, r"^総合力$"):
                return
            self._sleep(0.25)
        raise ArenaReaderError(
            "member_detail_anchor_missing",
            "体力/総合力 anchors did not become visible",
        )

    def _assert_p_item_source_restored(
        self,
        source_images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
        timeout_seconds: float = 2.0,
    ) -> None:
        """Prove a P-item overlay closed before another slot can be clicked."""

        if not boxes:
            raise ArenaReaderError(
                "p_item_source_restore_evidence_missing",
                "P-item detail recovery requires the accepted source-row boxes",
            )
        threshold = self.p_item_reader.content_generation_max_mean_abs_error
        deadline = time.monotonic() + timeout_seconds
        consecutive = 0
        last_errors: tuple[float, ...] = ()
        while time.monotonic() < deadline:
            image = self._capture()
            matches_source, last_errors = self._p_item_source_frame_matches(
                source_images,
                image,
                boxes,
            )
            if matches_source:
                consecutive += 1
                if consecutive >= 2:
                    return
            else:
                consecutive = 0
            self._sleep(0.12)
        raise ArenaReaderError(
            "p_item_source_restore_unproven",
            "member anchors and two stable source-row generations did not return: "
            f"errors={tuple(round(value, 6) for value in last_errors)!r}; "
            f"threshold={threshold}",
        )

    @staticmethod
    def _p_item_source_proof_boxes(
        source_image: Any,
        boxes: Sequence[tuple[int, int, int, int]],
    ) -> tuple[tuple[int, int, int, int], ...]:
        source_height, source_width = source_image.shape[:2]
        if source_width < 320 or source_height < 568:
            raise ArenaReaderError(
                "p_item_source_restore_evidence_invalid",
                "P-item source capture is too small for detail-panel restore evidence",
            )
        scale_x = source_width / 720.0
        scale_y = source_height / 1280.0

        def scaled_box(
            box: tuple[int, int, int, int],
        ) -> tuple[int, int, int, int]:
            x, y, width, height = box
            return (
                int(round(x * scale_x)),
                int(round(y * scale_y)),
                max(8, int(round(width * scale_x))),
                max(8, int(round(height * scale_y))),
            )

        panel_boxes = (
            scaled_box((18, 16, 331, 45)),
            scaled_box((100, 132, 320, 58)),
        )
        return (*boxes, *panel_boxes)

    def _assert_p_item_source_generation_stable(
        self,
        source_images: Sequence[Any],
        boxes: Sequence[tuple[int, int, int, int]],
    ) -> None:
        if len(source_images) < 2:
            return
        try:
            proof_boxes = self._p_item_source_proof_boxes(source_images[0], boxes)
            errors = measure_p_item_content_generation(source_images, proof_boxes)
        except (PItemReferenceError, TypeError, ValueError) as error:
            raise ArenaReaderError(
                "p_item_source_restore_evidence_invalid",
                "P-item frozen source-page evidence could not be measured",
            ) from error
        threshold = self.p_item_reader.content_generation_max_mean_abs_error
        if any(value > threshold for value in errors):
            raise ArenaReaderError(
                "p_item_source_generation_unstable",
                "P-item frozen source frames changed in an item or detail-panel "
                f"guard region: errors={tuple(round(value, 6) for value in errors)!r}; "
                f"threshold={threshold}",
            )

    def _p_item_source_frame_matches(
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
            exact_rows = {_text(item) for item in self._ocr(image, r".+")}
            anchors_visible = "体力" in exact_rows and "総合力" in exact_rows
        else:
            anchors_visible = member_anchors_visible
        if not source_images:
            raise ArenaReaderError(
                "p_item_source_restore_evidence_missing",
                "P-item source-page evidence requires frozen source frames",
            )
        source_height, source_width = source_images[0].shape[:2]
        image_height, image_width = image.shape[:2]
        if (image_height, image_width) != (source_height, source_width):
            raise ArenaReaderError(
                "p_item_source_restore_evidence_invalid",
                "P-item source and current captures have different dimensions",
            )
        proof_boxes = self._p_item_source_proof_boxes(source_images[0], boxes)
        threshold = self.p_item_reader.content_generation_max_mean_abs_error
        best_errors: tuple[float, ...] = ()
        try:
            for source_image in source_images:
                if source_image.shape[:2] != (source_height, source_width):
                    raise ArenaReaderError(
                        "p_item_source_restore_evidence_invalid",
                        "P-item frozen source captures have different dimensions",
                    )
                errors = tuple(
                    float(value)
                    for value in measure_p_item_content_generation(
                        (source_image, image),
                        proof_boxes,
                    )
                )
                if not best_errors or max(errors) < max(best_errors):
                    best_errors = errors
                if anchors_visible and all(value <= threshold for value in errors):
                    return True, errors
        except (PItemReferenceError, TypeError, ValueError) as error:
            raise ArenaReaderError(
                "p_item_source_restore_evidence_invalid",
                "P-item source-row restore evidence could not be measured",
            ) from error
        return False, best_errors

    def _assert_card_group_visible(
        self,
        group_index: int,
        timeout_seconds: float = 2.0,
        *,
        card_slot: int | None = None,
        expected_card_id: int | None = None,
    ) -> None:
        """Prove the original card generation is restored after a detail action.

        Geometry alone is insufficient because a fading detail overlay can
        expose the source row before it is ready for the next interaction.
        When the caller identifies the clicked slot, require that slot to
        match one of the three signatures from the already accepted source
        generation. A semantically identical card can nevertheless acquire a
        different 16x16 signature after the overlay closes (for example from
        sub-pixel row-box jitter or a top-family flip inside one stable
        shortlist). In that case an already resolved detail ID may authorize
        one stricter fallback: either the same single clean-reference family or
        the complete business-ID/visual-family pair set must remain exact in
        three consecutive frames, and each frame's raw clicked-slot content
        must match 2/3 frozen source frames at one unique +/-1 native-pixel
        alignment under the unchanged 0.75 MAE gate.
        If the fixed gallery maps that family to another business ID, the
        fallback remains available only to the active transaction whose exact
        title and complete semantic detail already resolved the expected ID,
        while the frozen page guard and six-slot geometry also match. The
        conflict is then reported explicitly instead of letting the weaker
        gallery hint override the clicked detail. Pixel thresholds and all
        failure bounds remain unchanged.
        """

        if card_slot is not None and not 1 <= card_slot <= 6:
            raise ArenaReaderError(
                "skill_card_source_slot_invalid",
                f"source restoration slot is outside 1..6: {card_slot}",
            )
        restoration_signatures = None
        source_guard_frames: tuple[Any, ...] | None = None
        source_stable_business_ids: tuple[int, ...] = ()
        source_stable_visual_group: str | None = None
        source_stable_identity_candidates: tuple[tuple[int, str], ...] = ()
        source_identity_projection_is_strict = False
        if card_slot is not None:
            restoration_signatures = getattr(
                self,
                "_card_restoration_signatures",
                {},
            ).get(group_index)
            if (
                restoration_signatures is None
                or len(restoration_signatures) != 3
                or any(len(frame) != 6 for frame in restoration_signatures)
            ):
                raise ArenaReaderError(
                    "skill_card_source_generation_missing",
                    f"group {group_index} has no accepted three-frame restoration generation",
                )
            guard_store = getattr(self, "_card_source_guard_frames", None)
            if guard_store is not None:
                source_guard_frames = tuple(guard_store.get(group_index, ()))
                if len(source_guard_frames) != 3:
                    raise ArenaReaderError(
                        "skill_card_source_guard_missing",
                        f"group {group_index} has no accepted three-frame overlay guard",
                    )
            identity_frames = getattr(
                self,
                "_card_identity_frames",
                {},
            ).get(group_index, ())
            if len(identity_frames) == 3 and all(
                len(frame) == 6 for frame in identity_frames
            ):
                source_identities = tuple(
                    frame[card_slot - 1] for frame in identity_frames
                )
                try:
                    source_stable_business_ids = (
                        stable_reference_business_candidates(
                            source_identities
                        )
                    )
                except BadgeReferenceError:
                    source_stable_business_ids = ()
                source_stable_visual_group = (
                    self._stable_reference_visual_group(source_identities)
                )
                source_stable_identity_candidates = (
                    self._stable_reference_identity_candidates(source_identities)
                )
                source_identity_projection_is_strict = bool(
                    source_stable_identity_candidates
                    and all(
                        self._reference_identity_projection_is_strict(identity)
                        for identity in source_identities
                    )
                )
        if expected_card_id is not None and (
            isinstance(expected_card_id, bool)
            or not isinstance(expected_card_id, int)
            or expected_card_id < 1
        ):
            raise ArenaReaderError(
                "skill_card_source_identity_invalid",
                f"expected source card ID is invalid: {expected_card_id!r}",
            )
        accepted_row = tuple(
            getattr(self, "_card_rows", {}).get(group_index, ())
        )
        expected_fixed_visual_groups = (
            self._fixed_gallery_visual_groups_for_business_id(expected_card_id)
            if expected_card_id is not None
            else ()
        )
        expected_fixed_visual_group = (
            expected_fixed_visual_groups[0]
            if expected_fixed_visual_groups is not None
            and len(expected_fixed_visual_groups) == 1
            else None
        )
        source_visual_group_allowed = bool(
            source_identity_projection_is_strict
            and source_stable_visual_group is not None
            and self._source_visual_group_matches_expected_card(
                source_stable_visual_group,
                expected_card_id,
                source_stable_business_ids,
            )
        )
        detail_identity_proof_matches = bool(
            card_slot is not None
            and len(accepted_row) == 6
            and self._detail_identity_proof_matches(
                (group_index, card_slot),
                expected_card_id,
                accepted_row[card_slot - 1],
            )
        )
        single_family_detail_title_override_candidate = bool(
            card_slot is not None
            and len(accepted_row) == 6
            and source_guard_frames is not None
            and source_identity_projection_is_strict
            and source_stable_visual_group is not None
            and expected_fixed_visual_group is not None
            and source_stable_visual_group != expected_fixed_visual_group
            and not source_visual_group_allowed
            and detail_identity_proof_matches
        )
        source_stable_identity_business_ids = tuple(
            sorted(
                {
                    business_id
                    for business_id, _visual_group in source_stable_identity_candidates
                }
            )
        )
        source_stable_identity_visual_groups = tuple(
            sorted(
                {
                    visual_group
                    for _business_id, visual_group in source_stable_identity_candidates
                }
            )
        )
        expected_source_identity_present = bool(
            expected_card_id is not None
            and expected_fixed_visual_group is not None
            and (expected_card_id, expected_fixed_visual_group)
            in source_stable_identity_candidates
        )
        multi_family_detail_title_override_candidate = bool(
            card_slot is not None
            and len(accepted_row) == 6
            and source_guard_frames is not None
            and source_identity_projection_is_strict
            and source_stable_visual_group is None
            and len(source_stable_identity_business_ids) > 1
            and len(source_stable_identity_visual_groups) > 1
            and source_stable_identity_business_ids == source_stable_business_ids
            and expected_source_identity_present
            and detail_identity_proof_matches
        )
        detail_title_override_candidate = bool(
            single_family_detail_title_override_candidate
            or multi_family_detail_title_override_candidate
        )
        source_visual_group_authorized = bool(
            source_visual_group_allowed
            or single_family_detail_title_override_candidate
        )
        deadline = time.monotonic() + timeout_seconds
        started = time.perf_counter()
        last_error = ""
        reads = 0
        previous_capture_started_at: float | None = None
        semantic_stable_frames = 0
        semantic_identity_frames: list[dict[str, Any]] = []
        aligned_content_frames: list[dict[str, Any]] = []
        source_guard_consecutive = 0

        def reset_consecutive_evidence() -> None:
            nonlocal semantic_stable_frames, source_guard_consecutive
            semantic_stable_frames = 0
            semantic_identity_frames.clear()
            aligned_content_frames.clear()
            source_guard_consecutive = 0

        while time.monotonic() < deadline:
            try:
                capture_started_at = time.monotonic()
                if previous_capture_started_at is not None:
                    self._record_duration_sample(
                        "skill_card_source_restore_capture_start_span",
                        capture_started_at - previous_capture_started_at,
                    )
                previous_capture_started_at = capture_started_at
                image = self._capture()
                detected_row = self._validated_card_group_row(image, group_index)
                row = detected_row
                if len(accepted_row) == 6:
                    if self._card_rows_shifted(detected_row, accepted_row):
                        raise ArenaReaderError(
                            "skill_card_source_geometry_changed",
                            f"group {group_index} returned with geometry outside the "
                            f"accepted source row; accepted={accepted_row!r}; "
                            f"detected={detected_row!r}",
                        )
                    # Detector NMS may retain a dense set before the click and
                    # only two edge anchors after the overlay closes.  Those
                    # equivalent observations can differ by a pixel or two.
                    # Once the current frame proves the same fixed row within
                    # the existing geometry tolerance, crop the clicked slot
                    # with the pre-action row itself.  This keeps both the
                    # 16x16 generation signature and clean-reference identity
                    # bound to one physical ROI instead of letting detector
                    # sparsity redefine the transaction mid-close.
                    row = accepted_row
                    if detected_row != accepted_row:
                        self._increment(
                            "skill_card_source_restore_frozen_geometry"
                        )
                reads += 1
                self._increment("skill_card_source_restore_reads")
                if source_guard_frames is not None:
                    guard_started = time.perf_counter()
                    try:
                        guard_boxes = self._skill_card_detail_overlay_guard_boxes(
                            image
                        )
                        signature_cache = getattr(
                            self,
                            "_card_source_guard_signature_cache",
                            None,
                        )
                        if signature_cache is None:
                            signature_cache = {}
                            self._card_source_guard_signature_cache = signature_cache
                        signature_cache_key = (
                            tuple(id(frame) for frame in source_guard_frames),
                            guard_boxes,
                        )
                        source_guard_signatures = signature_cache.get(
                            signature_cache_key
                        )
                        if source_guard_signatures is None:
                            source_guard_signatures = tuple(
                                p_item_content_generation_signatures(
                                    source_image,
                                    guard_boxes,
                                )
                                for source_image in source_guard_frames
                            )
                            signature_cache[signature_cache_key] = (
                                source_guard_signatures
                            )
                            self._increment(
                                "skill_card_source_guard_signature_cache_builds"
                            )
                        else:
                            self._increment(
                                "skill_card_source_guard_signature_cache_hits"
                            )
                        current_guard_signatures = (
                            p_item_content_generation_signatures(
                                image,
                                guard_boxes,
                            )
                        )
                        guard_error_sets = tuple(
                            measure_p_item_content_generation_from_signatures(
                                source_signatures,
                                current_guard_signatures,
                            )
                            for source_signatures in source_guard_signatures
                        )
                    except (PItemReferenceError, ArenaReaderError) as error:
                        raise ArenaReaderError(
                            "skill_card_source_guard_invalid",
                            "source/detail overlay guard could not be measured",
                        ) from error
                    finally:
                        self._add_timing(
                            "skill_card_source_restore_overlay_guard",
                            time.perf_counter() - guard_started,
                        )
                    guard_matches_source = any(
                        errors
                        and max(errors)
                        <= CARD_CONTENT_STABILITY_MAX_MEAN_ABS_ERROR
                        for errors in guard_error_sets
                    )
                    if not guard_matches_source:
                        reset_consecutive_evidence()
                        last_error = (
                            "detail-overlay guard still differs from the frozen "
                            "source page; errors="
                            f"{tuple(tuple(round(value, 6) for value in errors) for errors in guard_error_sets)!r}"
                        )
                        self._increment(
                            "skill_card_source_restore_overlay_guard_mismatches"
                        )
                        self._sleep(self._source_restore_poll_seconds)
                        continue
                    source_guard_consecutive += 1
                    if source_guard_consecutive < 2:
                        self._sleep(self._source_restore_poll_seconds)
                        continue
                if restoration_signatures is not None:
                    signature_started = time.perf_counter()
                    try:
                        visual_group, query, identity = (
                            self._card_references().content_signature_with_identity(
                                image,
                                row[card_slot - 1],
                            )
                        )
                    except BadgeReferenceError as error:
                        reset_consecutive_evidence()
                        last_error = (
                            "source card signature was not measurable after detail close: "
                            f"{error}"
                        )
                        self._increment(
                            "skill_card_source_restore_signature_failures"
                        )
                        self._sleep(self._source_restore_poll_seconds)
                        continue
                    finally:
                        self._add_timing(
                            "skill_card_source_restore_signature",
                            time.perf_counter() - signature_started,
                        )
                    current = ((visual_group, query),)
                    source_matches = any(
                        not self._card_content_generation_shifted(
                            current,
                            (frame[card_slot - 1],),
                        )
                        for frame in restoration_signatures
                    )
                    if not source_matches:
                        source_generation_deltas = tuple(
                            self._card_content_generation_deltas(
                                current,
                                (frame[card_slot - 1],),
                            )[0]
                            for frame in restoration_signatures
                        )
                        best_source_delta = min(
                            source_generation_deltas,
                            key=lambda delta: (
                                float("inf")
                                if delta["mean_absolute_error"] is None
                                else float(delta["mean_absolute_error"])
                            ),
                        )
                    else:
                        best_source_delta = None
                    if not source_matches:
                        measured_business_ids = self._reference_business_ids(identity)
                        measured_visual_groups = self._reference_visual_groups(identity)
                        measured_identity_candidates = (
                            self._reference_identity_candidates(identity)
                        )
                        measured_identity_projection_is_strict = (
                            self._reference_identity_projection_is_strict(identity)
                        )
                        expected_identity_matches = bool(
                            measured_identity_projection_is_strict
                            and expected_card_id is not None
                            and expected_card_id in measured_business_ids
                            and expected_fixed_visual_group is not None
                            and measured_visual_groups
                            == (expected_fixed_visual_group,)
                        )
                        source_visual_group_matches = bool(
                            measured_identity_projection_is_strict
                            and source_visual_group_authorized
                            and measured_visual_groups
                            == (source_stable_visual_group,)
                        )
                        multi_family_identity_matches = bool(
                            measured_identity_projection_is_strict
                            and multi_family_detail_title_override_candidate
                            and measured_identity_candidates
                            == source_stable_identity_candidates
                        )
                        aligned_content_proof: dict[str, Any] | None = None
                        if (
                            (
                                source_visual_group_matches
                                or multi_family_identity_matches
                            )
                            and detail_title_override_candidate
                            and source_guard_frames is not None
                        ):
                            aligned_started = time.perf_counter()
                            try:
                                aligned_content_proof = (
                                    self._fixed_slot_aligned_content_proof(
                                        source_guard_frames,
                                        image,
                                        accepted_row[card_slot - 1],
                                        (
                                            source_stable_visual_group
                                            or "detail-title-multi-family"
                                        ),
                                    )
                                )
                            finally:
                                self._add_timing(
                                    "skill_card_source_restore_aligned_content",
                                    time.perf_counter() - aligned_started,
                                )
                            self._increment(
                                "skill_card_source_restore_aligned_content_checks"
                            )
                            if aligned_content_proof["matched"]:
                                self._increment(
                                    "skill_card_source_restore_aligned_content_matches"
                                )
                            else:
                                self._increment(
                                    "skill_card_source_restore_aligned_content_mismatches"
                                )
                                source_visual_group_matches = False
                                multi_family_identity_matches = False
                        semantic_identity_matches = bool(
                            expected_identity_matches
                            or source_visual_group_matches
                            or multi_family_identity_matches
                        )
                        if semantic_identity_matches:
                            semantic_stable_frames += 1
                            semantic_identity_frames.append(identity)
                            del semantic_identity_frames[:-3]
                            if aligned_content_proof is not None:
                                aligned_content_frames.append(aligned_content_proof)
                                del aligned_content_frames[:-3]
                        else:
                            reset_consecutive_evidence()
                        stable_business_ids: tuple[int, ...] = ()
                        if len(semantic_identity_frames) == 3:
                            try:
                                stable_business_ids = (
                                    stable_reference_business_candidates(
                                        tuple(semantic_identity_frames)
                                    )
                                )
                            except BadgeReferenceError:
                                stable_business_ids = ()
                        stable_visual_group = (
                            self._stable_reference_visual_group(
                                semantic_identity_frames
                            )
                            if len(semantic_identity_frames) == 3
                            else None
                        )
                        stable_identity_candidates = (
                            self._stable_reference_identity_candidates(
                                semantic_identity_frames
                            )
                            if len(semantic_identity_frames) == 3
                            else ()
                        )
                        source_visual_group_settled = bool(
                            source_visual_group_authorized
                            and source_stable_visual_group is not None
                            and stable_visual_group == source_stable_visual_group
                        )
                        multi_family_identity_settled = bool(
                            multi_family_detail_title_override_candidate
                            and stable_identity_candidates
                            == source_stable_identity_candidates
                            and semantic_stable_frames >= 3
                            and len(aligned_content_frames) == 3
                            and all(
                                bool(proof.get("matched"))
                                for proof in aligned_content_frames
                            )
                        )
                        semantic_source_settled = bool(
                            (
                                expected_card_id is not None
                                and expected_card_id in stable_business_ids
                                and expected_fixed_visual_group is not None
                                and stable_visual_group
                                == expected_fixed_visual_group
                            )
                            or source_visual_group_settled
                            or multi_family_identity_settled
                        )
                        if semantic_source_settled:
                            self._increment(
                                "skill_card_source_restore_semantic_settled_fallbacks"
                            )
                            if source_visual_group_settled:
                                self._increment(
                                    "skill_card_source_restore_source_family_fallbacks"
                                )
                            if detail_title_override_candidate and (
                                source_visual_group_settled
                                or multi_family_identity_settled
                            ):
                                disambiguation = {
                                    "reason_code": (
                                        "exact_detail_title_overrode_ambiguous_card_face"
                                        if multi_family_identity_settled
                                        else "exact_detail_title_overrode_card_face"
                                    ),
                                    "status": "resolved",
                                    "severity": "info",
                                    "group_index": group_index,
                                    "card_slot": card_slot,
                                    "expected_card_id": expected_card_id,
                                    "expected_fixed_visual_group": (
                                        expected_fixed_visual_group
                                    ),
                                    "source_stable_card_ids": list(
                                        source_stable_business_ids
                                    ),
                                    "source_stable_visual_group": (
                                        source_stable_visual_group
                                    ),
                                    "source_stable_identity_candidates": [
                                        [business_id, visual_group]
                                        for business_id, visual_group in source_stable_identity_candidates
                                    ],
                                    "measured_card_ids": list(
                                        measured_business_ids
                                    ),
                                    "measured_visual_groups": list(
                                        measured_visual_groups
                                    ),
                                    "measured_identity_candidates": [
                                        [business_id, visual_group]
                                        for business_id, visual_group in measured_identity_candidates
                                    ],
                                    "semantic_consecutive_frames": (
                                        semantic_stable_frames
                                    ),
                                    "aligned_content_frames": deepcopy(
                                        aligned_content_frames
                                    ),
                                    "best_source_delta": dict(
                                        best_source_delta
                                    ),
                                    "detail_identity_proof": (
                                        self._detail_identity_proof_diagnostic(
                                            self._detail_identity_proofs[
                                                (group_index, card_slot)
                                            ]
                                        )
                                    ),
                                }
                                runtime_disambiguations = getattr(
                                    self,
                                    "_runtime_detail_title_disambiguations",
                                    None,
                                )
                                if runtime_disambiguations is None:
                                    runtime_disambiguations = []
                                    self._runtime_detail_title_disambiguations = (
                                        runtime_disambiguations
                                    )
                                runtime_disambiguations.append(disambiguation)
                                member_disambiguations = getattr(
                                    self,
                                    "_detail_title_disambiguations",
                                    None,
                                )
                                if member_disambiguations is None:
                                    member_disambiguations = []
                                    self._detail_title_disambiguations = (
                                        member_disambiguations
                                    )
                                member_disambiguations.append(
                                    deepcopy(disambiguation)
                                )
                                self._increment(
                                    "skill_card_source_restore_detail_title_disambiguations"
                                )
                        else:
                            last_error = (
                                f"group {group_index}/slot {card_slot} is visible but "
                                "does not match the accepted source generation; "
                                f"expected_card_id={expected_card_id!r}; "
                                "source_stable_card_ids="
                                f"{source_stable_business_ids!r}; "
                                "source_stable_visual_group="
                                f"{source_stable_visual_group!r}; "
                                "source_stable_identity_candidates="
                                f"{source_stable_identity_candidates!r}; "
                                f"measured_card_ids={measured_business_ids!r}; "
                                "measured_visual_groups="
                                f"{measured_visual_groups!r}; "
                                "measured_identity_candidates="
                                f"{measured_identity_candidates!r}; "
                                f"semantic_consecutive_frames={semantic_stable_frames}; "
                                f"stable_card_ids={stable_business_ids!r}; "
                                f"stable_visual_group={stable_visual_group!r}; "
                                f"identity_status={identity.get('status')!r}; "
                                "aligned_content_proof="
                                f"{aligned_content_proof!r}; "
                                f"best_source_delta={best_source_delta!r}"
                            )
                            self._increment(
                                "skill_card_source_restore_generation_mismatches"
                            )
                            if len(semantic_identity_frames) == 3:
                                reset_consecutive_evidence()
                            self._sleep(self._source_restore_poll_seconds)
                            continue
                self._card_rows[group_index] = row
                self._card_images[group_index] = image
                other_group = 1 - group_index
                if other_group in self._card_rows:
                    try:
                        other_row = self._validated_card_group_row(image, other_group)
                    except ArenaReaderError:
                        pass
                    else:
                        self._card_rows[other_group] = other_row
                        self._card_images[other_group] = image
                restore_elapsed = time.perf_counter() - started
                self._add_timing(
                    "skill_card_source_restore_wait",
                    restore_elapsed,
                )
                self._record_duration_sample(
                    "skill_card_source_restore_phase",
                    restore_elapsed,
                )
                if card_slot is not None and reads == 1:
                    self._increment("skill_card_source_restore_first_read")
                return
            except ArenaReaderError as error:
                reset_consecutive_evidence()
                last_error = str(error)
            self._sleep(self._source_restore_poll_seconds)
        restore_elapsed = time.perf_counter() - started
        self._add_timing(
            "skill_card_source_restore_wait",
            restore_elapsed,
        )
        self._record_duration_sample(
            "skill_card_source_restore_phase",
            restore_elapsed,
        )
        raise ArenaReaderError(
            "skill_card_close_failed",
            f"group {group_index} did not return after closing the card detail: {last_error}",
        )

    def _detect_card_rows(self, image: Any) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
        return canonical_skill_card_rows(image.shape, self._raw_card_candidate_boxes(image))

    def _validated_card_group_row(
        self,
        image: Any,
        group_index: int,
        *,
        allow_fixed_secondary: bool = False,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Require complete fixed-row anchors before treating badge absence as zero."""

        allow_fixed_secondary = bool(
            allow_fixed_secondary
            or (
                group_index == 1
                and getattr(
                    self,
                    "_secondary_fixed_slot_fallback_enabled",
                    False,
                )
            )
        )

        raw_boxes = self._raw_card_candidate_boxes(image)
        rows = canonical_skill_card_rows(image.shape, raw_boxes)
        selected = rows[0] if rows and group_index == 0 else ()
        if rows and group_index == 1:
            selected = canonical_scrolled_secondary_skill_card_row(
                image.shape,
                raw_boxes,
                rows[0],
            )
        if group_index == 1 and len(selected) != 6 and allow_fixed_secondary:
            expected_row = expected_scrolled_secondary_skill_card_row(
                image.shape,
                rows[0] if rows else (),
            )
            cache = getattr(self, "_secondary_presence_cache", None)
            if cache is None:
                cache = {}
                self._secondary_presence_cache = cache
            cached = cache.get(id(image))
            diagnostics = (
                cached[1]
                if cached is not None and cached[0]() is image
                else ()
            )
            if diagnostics:
                self._increment("secondary_presence_cache_hits")
            else:
                started = time.perf_counter()
                diagnostics = tuple(
                    self._secondary_fixed_slot_presence(image, box, slot)
                    for slot, box in enumerate(expected_row, start=1)
                )
                self._add_timing(
                    "secondary_fixed_slot_presence",
                    time.perf_counter() - started,
                )
                try:
                    from weakref import ref

                    cache[id(image)] = (ref(image), diagnostics)
                except TypeError:
                    pass
            self._secondary_presence_diagnostics = diagnostics
            if len(expected_row) == 6 and all(
                item["present"] is True or item["empty"] is True
                for item in diagnostics
            ):
                selected = expected_row
                self._increment("secondary_reference_presence_frames")
        if len(selected) != 6:
            suffix = (
                ""
                if group_index != 1 or not self._secondary_presence_diagnostics
                else f"; fixed-slot presence={self._secondary_presence_diagnostics!r}"
            )
            raise ArenaReaderError(
                "skill_card_customization_frame_incomplete",
                f"group {group_index} lacks complete card anchors on a repeated frame"
                f"{suffix}",
            )
        return selected

    def _secondary_fixed_slot_presence(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        slot: int,
    ) -> dict[str, Any]:
        """Prove one fixed second-row slot contains a card, not its identity.

        The ordinary detector remains the first route. This fallback runs only
        after the shifted row is geometrically implied by a fully scrolled
        primary row. The fixed clean-reference gallery must find a card-like
        image; this gate never accepts an ID, badge, or customization.
        """

        empty_observation = self._skill_card_empty_observation(image, box)
        if empty_observation["empty"]:
            return {
                "slot": slot,
                "present": False,
                "presence_path": "stable_blank_candidate",
                "reference_status": None,
                "top_error": None,
                "top_group": None,
                **empty_observation,
            }
        measurement = self._card_references().measure_identity(image, box)
        status = measurement.get("status")
        reference_like = status in {"MEASURED", "MEASURED_CANDIDATES"}
        return {
            "slot": slot,
            "present": reference_like,
            "empty": False,
            "presence_path": "clean_reference" if reference_like else "unresolved",
            "reference_status": status,
            "top_error": measurement.get(
                "identity_top_error",
                measurement.get("top_error"),
            ),
            "top_group": measurement.get(
                "reference_visual_group",
                measurement.get("top_group"),
            ),
        }

    @staticmethod
    def _skill_card_empty_observation(
        image: Any,
        box: tuple[int, int, int, int],
    ) -> dict[str, Any]:
        """Detect only a nearly uniform neutral/dark blank, never card art."""

        import cv2
        import numpy as np

        x, y, width, height = box
        crop = image[y : y + height, x : x + width, :3]
        if crop.size == 0 or width < 16 or height < 16:
            raise ArenaReaderError(
                "skill_card_empty_slot_input_invalid",
                f"skill-card slot crop is empty or too small: {box!r}",
            )
        inset = max(2, int(round(min(width, height) * 0.06)))
        inner = crop[inset:-inset, inset:-inset]
        if inner.size == 0:
            inner = crop
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 32, 96)
        gray_mean = float(np.mean(gray))
        gray_stddev = float(np.std(gray))
        gray_p95 = float(np.percentile(gray, 95))
        dark_pixel_ratio = float(np.mean(gray <= 24))
        edge_density = float(np.mean(edges > 0))
        channel_range = np.max(inner, axis=2) - np.min(inner, axis=2)
        mean_channel_range = float(np.mean(channel_range))
        uniform_neutral_placeholder = bool(
            mean_channel_range <= 8.0
            and gray_stddev <= 8.0
            and edge_density <= 0.01
        )
        uniform_dark_placeholder = bool(
            gray_mean <= 12.0
            and gray_stddev <= 6.0
            and gray_p95 <= 24.0
            and dark_pixel_ratio >= 0.97
            and edge_density <= 0.015
        )
        empty = bool(
            uniform_neutral_placeholder or uniform_dark_placeholder
        )
        return {
            "empty": empty,
            "uniform_neutral_placeholder": uniform_neutral_placeholder,
            "uniform_dark_placeholder": uniform_dark_placeholder,
            "gray_mean": round(gray_mean, 6),
            "gray_stddev": round(gray_stddev, 6),
            "gray_p95": round(gray_p95, 6),
            "dark_pixel_ratio": round(dark_pixel_ratio, 6),
            "edge_density": round(edge_density, 6),
            "mean_channel_range": round(mean_channel_range, 6),
        }

    def read_skill_card_empty_flags(
        self,
        target: TeamTarget,
        stage_number: int,
        member_slot: int,
        group_index: int,
    ) -> Sequence[bool]:
        """Classify stable enabled blank slots before duplicate or badge work."""

        del target, stage_number, member_slot
        row = self._card_rows.get(group_index, ())
        frames = self._card_count_frames.get(group_index, ())
        if len(row) != 6 or len(frames) != 3:
            raise ArenaReaderError(
                "skill_card_empty_slot_input_mismatch",
                f"group {group_index} has no stable six-slot three-frame input",
            )
        per_frame = tuple(
            tuple(
                self._skill_card_empty_observation(image, box)
                for box in row
            )
            for image in frames
        )
        flags: list[bool] = []
        diagnostics: list[dict[str, Any]] = []
        for slot_index in range(6):
            observations = tuple(frame[slot_index] for frame in per_frame)
            values = tuple(bool(item["empty"]) for item in observations)
            if len(set(values)) != 1:
                raise ArenaReaderError(
                    "skill_card_empty_slot_unstable",
                    f"group {group_index}/slot {slot_index + 1} blank state changed "
                    "inside one stable card generation",
                )
            flags.append(values[0])
            diagnostics.append(
                {
                    "slot": slot_index + 1,
                    "empty": values[0],
                    "frames": [dict(item) for item in observations],
                }
            )
        result = tuple(flags)
        empty_cache = getattr(self, "_card_empty_flags", None)
        if empty_cache is None:
            empty_cache = {}
            self._card_empty_flags = empty_cache
        diagnostic_cache = getattr(self, "_card_empty_diagnostics", None)
        if diagnostic_cache is None:
            diagnostic_cache = {}
            self._card_empty_diagnostics = diagnostic_cache
        empty_cache[group_index] = result
        diagnostic_cache[group_index] = tuple(diagnostics)
        for _ in range(sum(result)):
            self._increment("skill_card_empty_slots")
        return result

    def _raw_card_candidate_boxes(
        self,
        image: Any,
    ) -> tuple[tuple[int, int, int, int], ...]:
        import cv2

        # cards.onnx was validated with a stretched 640-square input. Maa 5.13
        # now letterboxes non-square inputs, so preserve that model contract
        # here without changing the shared detector or taking another frame.
        height, width = image.shape[:2]
        detector_image = cv2.resize(
            image,
            (640, 640),
            interpolation=cv2.INTER_AREA,
        )
        detail = self._run_recognition(
            "ProduceRecognitionCards",
            detector_image,
        )
        results = (detail.all_results or []) if detail and detail.hit else []
        # Match Maa's integer rectangle conversion: truncate each coordinate
        # and extent independently. Keep the source frame's coordinate system
        # for the existing row geometry and subsequent card crops/clicks.
        scale_x, scale_y = width / 640, height / 640
        return tuple(
            (
                int(candidate.box[0] * scale_x),
                int(candidate.box[1] * scale_y),
                int(candidate.box[2] * scale_x),
                int(candidate.box[3] * scale_y),
            )
            for candidate in isolate_card_candidates(results)
        )
